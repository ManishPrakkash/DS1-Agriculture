"""
Section 7 (define analytical dataset) + Section 8 (cleaning) of the master
execution prompt.

Cleaning log (Problem -> Detection -> Action -> Reason), applied in order:

1. Problem: Irrigation_pct missing entirely for 4/22 countries (China mainland,
   Indonesia, Viet Nam, Thailand) -- confirmed in docs/data_quality_report.md as
   genuine source non-reporting of FAOSTAT item 6611 for those countries, not a
   random gap.
   Detection: df['Irrigation_pct'].isna() grouped by Country.
   Action: (a) add a binary flag column Irrigation_Data_Missing so the gap stays
   visible to every downstream step instead of being silently smoothed away, then
   (b) impute the missing values with each crop's global median Irrigation_pct
   (median is robust to the right-skew of irrigation share and avoids inventing a
   spuriously precise number).
   Reason: dropping ~780 rows (18% of the data, all of 4 whole countries) would
   remove entire countries from every regional comparison in Section 10B; a flagged
   median imputation keeps those countries analysable while making clear -- via the
   flag column and this log -- that their irrigation figure is an imputed
   placeholder, not a reported one. The model and EDA both retain the flag so
   nothing downstream can mistake an imputed value for an observed one.

2. Problem: none found for Crop_Yield_kg_ha, Fertiliser_Use_kg_ha, Temperature_C,
   Rainfall_mm (0% missing, see audit report), and no duplicate
   (Country, Crop_Type, Year) keys.
   Detection: full-column isna() and duplicated() checks in 03_data_audit.py.
   Action: none required.
   Reason: n/a.

3. Problem: extreme yield values (e.g. Egyptian irrigated sugar cane at
   ~120 t/ha) could look like entry errors.
   Detection: manual cross-check of the top-5 highest yields per crop against
   known agronomic literature (docs/data_quality_report.md).
   Action: retained, no removal or capping.
   Reason: these are real, well-documented world-leading yields for intensively
   irrigated sugar cane in Egypt's Nile system, not data-entry errors; removing
   genuine extreme-but-real observations would bias the "does irrigation help"
   question the challenge explicitly asks (Section 10E).

4. Problem: raw yield is not comparable across crop types (sugar cane/potatoes/
   cassava report tonnes of fresh biomass per ha, roughly 10-40x cereal yields).
   Detection: docs/data_quality_report.md, "yield distribution by crop".
   Action: add a Yield_Zscore_Within_Crop column (yield standardised within each
   Crop_Type) for any cross-crop comparison; raw Crop_Yield_kg_ha is kept as the
   modelling target and used only for within-crop comparisons.
   Reason: master-prompt Section 10C explicitly warns not to compare crops
   ignoring scale differences.

Output:
  data/processed/cleaned_dataset.csv   (post-cleaning, still all 22 countries)
  data/final/analytical_dataset.csv    (= cleaned dataset; this IS the modelling
                                         table, kept in data/final per the required
                                         raw/cleaned/final separation)
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW_MERGED = os.path.join(BASE, "data", "processed", "raw_merged.csv")
PROCESSED = os.path.join(BASE, "data", "processed")
FINAL = os.path.join(BASE, "data", "final")
os.makedirs(FINAL, exist_ok=True)

df = pd.read_csv(RAW_MERGED)

# --- Step 1: flag + median-impute Irrigation_pct -----------------------------
df["Irrigation_Data_Missing"] = df["Irrigation_pct"].isna().astype(int)
crop_median_irrigation = df.groupby("Crop_Type")["Irrigation_pct"].transform("median")
global_median_irrigation = df["Irrigation_pct"].median()
df["Irrigation_pct"] = df["Irrigation_pct"].fillna(crop_median_irrigation)
df["Irrigation_pct"] = df["Irrigation_pct"].fillna(global_median_irrigation)

assert df.isna().sum().sum() == 0, "Unexpected remaining missing values after cleaning"

# --- Step 4: within-crop standardised yield for cross-crop comparisons -------
df["Yield_Zscore_Within_Crop"] = df.groupby("Crop_Type")["Crop_Yield_kg_ha"].transform(
    lambda s: (s - s.mean()) / s.std(ddof=0)
)

# --- sanity checks -----------------------------------------------------------
assert (df["Irrigation_pct"] >= 0).all() and (df["Irrigation_pct"] <= 100).all()
assert (df["Crop_Yield_kg_ha"] > 0).all()
assert (df["Fertiliser_Use_kg_ha"] >= 0).all()

df = df.sort_values(["Country", "Crop_Type", "Year"]).reset_index(drop=True)

cleaned_path = os.path.join(PROCESSED, "cleaned_dataset.csv")
final_path = os.path.join(FINAL, "analytical_dataset.csv")
df.to_csv(cleaned_path, index=False)
df.to_csv(final_path, index=False)

print(f"[ok] wrote {cleaned_path} shape={df.shape}")
print(f"[ok] wrote {final_path} shape={df.shape}")
print(f"Rows with imputed irrigation: {df['Irrigation_Data_Missing'].sum()} "
      f"({df['Irrigation_Data_Missing'].mean()*100:.1f}%)")
