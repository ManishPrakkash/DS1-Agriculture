"""
Section 5 (initial audit) + Section 6 (data dictionary) of the master execution
prompt, run against data/processed/raw_merged.csv (the untouched merge output).

Writes docs/data_quality_report.md and docs/data_dictionary.md.
"""
import os

import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
DF_PATH = os.path.join(BASE, "data", "processed", "raw_merged.csv")
DOCS = os.path.join(BASE, "docs")
os.makedirs(DOCS, exist_ok=True)

df = pd.read_csv(DF_PATH)

lines = []
lines.append("# Data Quality Report\n")
lines.append(f"Source file: `data/processed/raw_merged.csv`\n")
lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

lines.append("\n## Columns and dtypes\n")
lines.append(df.dtypes.to_frame("dtype").to_markdown())

lines.append("\n\n## Missing values (%)\n")
miss = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
lines.append(miss.to_frame("% missing").to_markdown())

lines.append("\n\n## Duplicate (Country, Crop_Type, Year) records\n")
dupe_count = df.duplicated(subset=["Country", "Crop_Type", "Year"]).sum()
lines.append(f"Duplicates found: {dupe_count}\n")

lines.append("\n## Categorical coverage\n")
lines.append(f"Countries ({df['Country'].nunique()}): {sorted(df['Country'].unique())}\n")
lines.append(f"\nCrop types ({df['Crop_Type'].nunique()}): {sorted(df['Crop_Type'].unique())}\n")
lines.append(f"\nYear range: {df['Year'].min()}-{df['Year'].max()} "
             f"({df['Year'].nunique()} distinct years)\n")

lines.append("\n## Numerical ranges (describe)\n")
num_cols = ["Crop_Yield_kg_ha", "Fertiliser_Use_kg_ha", "Irrigation_pct",
            "Temperature_C", "Rainfall_mm"]
lines.append(df[num_cols].describe().T.to_markdown())

lines.append("\n\n## Yield distribution by crop (agronomic units differ per crop --"
             " sugar cane/potatoes/cassava report tonnes-of-fresh-matter per ha and"
             " are naturally an order of magnitude above cereals; this is real"
             " agronomy, not an error)\n")
lines.append(df.groupby("Crop_Type")["Crop_Yield_kg_ha"]
             .agg(["count", "min", "median", "max"]).to_markdown())

lines.append("\n\n## Missingness detail: Irrigation_pct\n")
irr_missing_by_country = (df.groupby("Country")["Irrigation_pct"]
                           .apply(lambda s: s.isna().mean() * 100)
                           .sort_values(ascending=False))
lines.append(irr_missing_by_country.to_frame("% missing").to_markdown())
lines.append(
    "\n\nRoot cause (verified against raw FAOSTAT Inputs_LandUse file): "
    "FAOSTAT item 6611 'Agriculture area actually irrigated' has zero reported "
    "rows for China (mainland), Indonesia, Viet Nam and Thailand across the whole "
    "time series -- this is genuine source-level non-reporting for that specific "
    "indicator (a sparser alternative item, 6690 'Land area equipped for "
    "irrigation', exists for these countries but only for 2 census years each, "
    "too sparse for an annual series). This is documented, not imputed with "
    "invented values -- see cleaning script for the transparent handling method.\n"
)

lines.append("\n## Outlier check: extreme values are inspected, not blindly removed\n")
top5 = df.nlargest(5, "Crop_Yield_kg_ha")[["Country", "Crop_Type", "Year", "Crop_Yield_kg_ha"]]
lines.append("Top 5 yield values (checked against known agronomy -- Egypt's "
              "intensively irrigated sugar cane yields of ~120 t/ha are a "
              "real, well-documented world-leading figure, not a data error):\n")
lines.append(top5.to_markdown(index=False))

with open(os.path.join(DOCS, "data_quality_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(str(x) for x in lines))

print("[ok] wrote docs/data_quality_report.md")

# ---------------------------------------------------------------------------
# Data dictionary (Section 6)
# ---------------------------------------------------------------------------
dd = """# Data Dictionary

One row = one (Country, Crop_Type, Year) national annual observation.

| Variable | Meaning | Type | Unit | Source | Example | Role | Transformation |
|---|---|---|---|---|---|---|---|
| Country | Reporting country (proxy for "Region" in the challenge brief) | categorical | n/a | FAOSTAT Area | "India" | feature (categorical) | FAOSTAT area code mapped to name |
| Crop_Type | Crop commodity | categorical | n/a | FAOSTAT Item | "Wheat" | feature (categorical) | FAOSTAT item code mapped to name |
| Year | Calendar year | integer | year | FAOSTAT / NASA POWER | 2020 | feature (numeric) | none |
| Crop_Yield_kg_ha | Yield of the crop | float | kg per hectare | FAOSTAT QCL, element 5412 "Yield" | 3439.8 | **target** | none (already kg/ha in source) |
| Fertiliser_Use_kg_ha | Total N+P2O5+K2O nutrient applied per hectare of cropland, national annual average, all crops combined | float | kg nutrient / ha cropland | FAOSTAT RFN, element 5159 "Use per area of cropland", summed over items 3102+3103+3104 | 154.2 | feature (numeric) | summed across 3 nutrients |
| Irrigation_pct | Share of agricultural land actually irrigated | float | % | FAOSTAT RL, item 6611 / item 6610 * 100 | 34.7 | feature (numeric) | ratio of two FAOSTAT Land Use items |
| Temperature_C | Mean annual air temperature at a representative national coordinate | float | degrees C | NASA POWER (T2M, annual climatology) | 24.1 | feature (numeric) | annual mean from monthly API |
| Rainfall_mm | Mean annual precipitation at a representative national coordinate | float | mm/year | NASA POWER (PRECTOTCORR, annual climatology, mm/day x 365.25) | 1083.4 | feature (numeric) | daily mean converted to annual total |

## Identifier vs categorical vs numeric vs target
- **Identifier**: none genuinely exists at this granularity (no Farm_ID -- see limitations). `(Country, Crop_Type, Year)` is the natural composite key and is *not* used as a predictive feature by itself beyond the two categorical columns and the Year numeric trend.
- **Categorical**: `Country`, `Crop_Type`.
- **Numerical predictors**: `Year`, `Fertiliser_Use_kg_ha`, `Irrigation_pct`, `Temperature_C`, `Rainfall_mm`.
- **Target**: `Crop_Yield_kg_ha`.

## Soil properties
No real, legally-usable, country/year-level soil property dataset with adequate
coverage for these 22 countries and 23 years was found (global gridded soil products
such as SoilGrids report static, not annual, values, and are not naturally joinable
to an annual panel without fabricating a time trend). Rather than inventing a
`Soil_Properties` column, it is **excluded** and documented as a limitation.

## Note on granularity of Fertiliser_Use_kg_ha, Irrigation_pct, Temperature_C, Rainfall_mm
These four variables are national-annual figures, not crop-specific. Every crop
grown in the same country-year shares the same fertiliser/irrigation/climate
values. This is a real, honestly-documented limitation of country-level FAO/NASA
statistics, not an error -- crop-specific input data at this scale does not
exist publicly.
"""
with open(os.path.join(DOCS, "data_dictionary.md"), "w", encoding="utf-8") as f:
    f.write(dd)
print("[ok] wrote docs/data_dictionary.md")
