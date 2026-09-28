"""
Build the raw analytical table by merging the four real primary sources
downloaded by 01_download_data.py.

Granularity of the final table: ONE ROW = one (Country, Crop, Year) observation,
i.e. a national annual crop-yield record -- NOT an individual farm. There is no
public, legally-usable, farm-level dataset that genuinely covers crop type,
region, rainfall, soil, irrigation, fertiliser and yield together (see
docs/PROJECT_REPORT.md, "Dataset selection" and "Limitations" sections for the
research trail). Region = FAOSTAT country ("Area"). Fertiliser use, irrigation
share and climate are national-annual figures (not crop-specific) and are
broadcast to every crop grown in that country-year; this is a real, documented
limitation, not a fabrication -- these are the true reported national values.

Output: data/processed/raw_merged.csv  (untouched "raw merge", before cleaning)
"""
import json
import os

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(BASE, "data", "raw")
EXTRACTED = os.path.join(RAW, "extracted")
PROCESSED = os.path.join(BASE, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

COUNTRIES = {
    100: "India", 231: "United States of America", 41: "China, mainland",
    21: "Brazil", 10: "Australia", 159: "Nigeria", 68: "France",
    185: "Russian Federation", 101: "Indonesia", 9: "Argentina", 33: "Canada",
    138: "Mexico", 202: "South Africa", 114: "Kenya", 237: "Viet Nam",
    216: "Thailand", 230: "Ukraine", 79: "Germany", 59: "Egypt",
    165: "Pakistan", 238: "Ethiopia", 16: "Bangladesh",
}
CROPS = {
    15: "Wheat", 56: "Maize (corn)", 27: "Rice", 116: "Potatoes",
    236: "Soya beans", 156: "Sugar cane", 328: "Seed cotton", 44: "Barley",
    83: "Sorghum", 125: "Cassava, fresh",
}
YEAR_START, YEAR_END = 2000, 2022
YEAR_COLS = [f"Y{y}" for y in range(YEAR_START, YEAR_END + 1)]


def melt_years(df, value_name):
    id_cols = [c for c in df.columns if not c.startswith("Y")]
    m = df.melt(id_vars=id_cols, value_vars=[c for c in YEAR_COLS if c in df.columns],
                var_name="YearCol", value_name=value_name)
    m["Year"] = m["YearCol"].str.replace("Y", "", regex=False).astype(int)
    return m.drop(columns="YearCol")


def load_yield():
    df = pd.read_csv(os.path.join(EXTRACTED, "Production_Crops_Livestock_E_All_Data.csv"),
                      encoding="latin1")
    df = df[df["Area Code"].isin(COUNTRIES) & df["Item Code"].isin(CROPS)
            & (df["Element Code"] == 5412)]  # Yield, kg/ha
    m = melt_years(df, "Crop_Yield_kg_ha")
    m["Country"] = m["Area Code"].map(COUNTRIES)
    m["Crop_Type"] = m["Item Code"].map(CROPS)
    m = m[["Country", "Crop_Type", "Year", "Crop_Yield_kg_ha"]]
    return m.dropna(subset=["Crop_Yield_kg_ha"])


def load_fertiliser():
    df = pd.read_csv(os.path.join(EXTRACTED, "Inputs_FertilizersNutrient_E_All_Data.csv"),
                      encoding="latin1")
    df = df[df["Area Code"].isin(COUNTRIES) & df["Item Code"].isin([3102, 3103, 3104])
            & (df["Element Code"] == 5159)]  # Use per area of cropland, kg/ha
    m = melt_years(df, "kg_ha")
    m["Country"] = m["Area Code"].map(COUNTRIES)
    grp = m.groupby(["Country", "Year"], as_index=False)["kg_ha"].sum(min_count=1)
    grp = grp.rename(columns={"kg_ha": "Fertiliser_Use_kg_ha"})
    return grp


def load_irrigation():
    df = pd.read_csv(os.path.join(EXTRACTED, "Inputs_LandUse_E_All_Data.csv"),
                      encoding="latin1")
    df = df[df["Area Code"].isin(COUNTRIES) & df["Item Code"].isin([6611, 6610])
            & (df["Element Code"] == 5110)]  # Area, 1000 ha
    m = melt_years(df, "area_1000ha")
    m["Country"] = m["Area Code"].map(COUNTRIES)
    piv = m.pivot_table(index=["Country", "Year"], columns="Item Code",
                         values="area_1000ha", aggfunc="sum")
    piv = piv.rename(columns={6611: "irrigated_1000ha", 6610: "agri_land_1000ha"})
    piv["Irrigation_pct"] = (piv["irrigated_1000ha"] / piv["agri_land_1000ha"]) * 100
    return piv.reset_index()[["Country", "Year", "Irrigation_pct"]]


def load_climate():
    with open(os.path.join(RAW, "nasa_power_climate.json")) as f:
        raw = json.load(f)
    rows = []
    for country, payload in raw.items():
        params = payload["properties"]["parameter"]
        t2m = params["T2M"]
        precip = params["PRECTOTCORR"]
        years = sorted({k[:4] for k in t2m if not k.endswith("13")})
        for y in years:
            year = int(y)
            if year < YEAR_START or year > YEAR_END:
                continue
            temp_key = f"{y}13"
            precip_key = f"{y}13"
            if temp_key not in t2m or precip_key not in precip:
                continue
            rows.append({
                "Country": country,
                "Year": year,
                "Temperature_C": t2m[temp_key],
                "Rainfall_mm": precip[precip_key] * 365.25,  # mm/day avg -> mm/year
            })
    return pd.DataFrame(rows)


def main():
    yield_df = load_yield()
    fert_df = load_fertiliser()
    irr_df = load_irrigation()
    clim_df = load_climate()

    merged = yield_df.merge(fert_df, on=["Country", "Year"], how="left")
    merged = merged.merge(irr_df, on=["Country", "Year"], how="left")
    merged = merged.merge(clim_df, on=["Country", "Year"], how="left")

    merged = merged.sort_values(["Country", "Crop_Type", "Year"]).reset_index(drop=True)
    out_path = os.path.join(PROCESSED, "raw_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}  shape={merged.shape}")
    print(merged.isna().mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
