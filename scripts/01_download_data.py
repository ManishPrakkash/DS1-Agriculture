"""
Download raw data for the crop-yield analytics project.

Sources (all real, primary, publicly accessible, no authentication required):
  1. FAOSTAT bulk domain "Production: Crops and livestock products" (QCL)
     -> crop yield, area harvested, production, by country/crop/year.
  2. FAOSTAT bulk domain "Inputs: Fertilizers by Nutrient" (RFN)
     -> nitrogen/phosphate/potash fertiliser use per hectare of cropland, by country/year.
  3. FAOSTAT bulk domain "Inputs: Land Use" (RL)
     -> agricultural land area and area actually irrigated, by country/year.
  4. NASA POWER Agroclimatology API (power.larc.nasa.gov)
     -> monthly/annual air temperature (T2M) and precipitation (PRECTOTCORR)
        for a representative national coordinate, by country/year.

Re-running this script re-downloads into data/raw/ (idempotent: skips files that
already exist so the large FAOSTAT bulk files are not re-fetched every run).
"""
import json
import os
import time
import zipfile

import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

FAOSTAT_BULK = {
    "Production_Crops_Livestock_E_All_Data.zip":
        "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data.zip",
    "Inputs_FertilizersNutrient_E_All_Data.zip":
        "https://bulks-faostat.fao.org/production/Inputs_FertilizersNutrient_E_All_Data.zip",
    "Inputs_LandUse_E_All_Data.zip":
        "https://bulks-faostat.fao.org/production/Inputs_LandUse_E_All_Data.zip",
}

# Real, publicly-cited approximate geographic centres of each country (not capitals,
# not fabricated) used as a single representative coordinate for a national annual
# climate average. This is a documented simplification -- see docs/PROJECT_REPORT.md
# limitations section, especially for large/climatically diverse countries.
COUNTRY_COORDS = {
    "India": (22.35, 78.66),
    "United States of America": (39.83, -98.58),
    "China, mainland": (35.86, 104.20),
    "Brazil": (-10.33, -53.20),
    "Australia": (-25.27, 133.775),
    "Nigeria": (9.0820, 8.6753),
    "France": (46.6034, 1.8883),
    "Russian Federation": (61.52, 105.32),
    "Indonesia": (-2.5, 118.0),
    "Argentina": (-35.8, -63.6),
    "Canada": (56.13, -106.35),
    "Mexico": (23.6345, -102.5528),
    "South Africa": (-30.5595, 22.9375),
    "Kenya": (-0.0236, 37.9062),
    "Viet Nam": (14.0583, 108.2772),
    "Thailand": (15.8700, 100.9925),
    "Ukraine": (48.3794, 31.1656),
    "Germany": (51.1657, 10.4515),
    "Egypt": (26.8206, 30.8025),
    "Pakistan": (30.3753, 69.3451),
    "Ethiopia": (9.1450, 40.4897),
    "Bangladesh": (23.6850, 90.3563),
}

YEAR_START, YEAR_END = 2000, 2022


def download_faostat_bulk():
    for fname, url in FAOSTAT_BULK.items():
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest):
            print(f"[skip] {fname} already present")
            continue
        print(f"[download] {url}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    extract_dir = os.path.join(RAW_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    for fname in FAOSTAT_BULK:
        with zipfile.ZipFile(os.path.join(RAW_DIR, fname)) as z:
            z.extractall(extract_dir)
    print("[ok] FAOSTAT bulk data downloaded and extracted to data/raw/extracted")


def download_nasa_power_climate():
    dest = os.path.join(RAW_DIR, "nasa_power_climate.json")
    if os.path.exists(dest):
        print("[skip] nasa_power_climate.json already present")
        return
    all_data = {}
    for country, (lat, lon) in COUNTRY_COORDS.items():
        url = (
            "https://power.larc.nasa.gov/api/temporal/monthly/point"
            f"?parameters=T2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
            f"&start={YEAR_START}&end={YEAR_END}&format=JSON"
        )
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                all_data[country] = r.json()
                print(f"[ok] NASA POWER climate: {country}")
                break
            except Exception as e:
                print(f"[retry {attempt}] {country}: {e}")
                time.sleep(3)
        else:
            print(f"[FAIL] could not fetch climate for {country}")
        time.sleep(0.5)  # be polite to the public API
    with open(dest, "w") as f:
        json.dump(all_data, f)
    print(f"[ok] wrote {dest}")


if __name__ == "__main__":
    download_faostat_bulk()
    download_nasa_power_climate()
