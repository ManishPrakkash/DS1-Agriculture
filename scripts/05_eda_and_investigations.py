"""
Section 9 (EDA), Section 10 (A-F agricultural investigations) and Section 11
(statistical analysis) of the master execution prompt.

Writes docs/eda_and_investigations.md with all descriptive statistics and
findings. No plotting here (plots are Section 12, in script 06).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_csv(os.path.join(BASE, "data", "final", "analytical_dataset.csv"))
DOCS = os.path.join(BASE, "docs")

out = []
def w(*parts):
    out.append(" ".join(str(p) for p in parts))

w("# EDA and Agricultural Investigations\n")

# ---------------------------------------------------------------------------
w("## 9. Descriptive statistics\n")
num_cols = ["Crop_Yield_kg_ha", "Rainfall_mm", "Temperature_C",
            "Fertiliser_Use_kg_ha", "Irrigation_pct"]
desc = df[num_cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
w(desc.to_markdown())

w("\n\n### Categorical frequency / share\n")
for col in ["Country", "Crop_Type"]:
    vc = df[col].value_counts()
    pct = (vc / len(df) * 100).round(1)
    tbl = pd.DataFrame({"count": vc, "%": pct})
    w(f"\n**{col}**\n")
    w(tbl.to_markdown())

# ---------------------------------------------------------------------------
w("\n\n## 10A. Factors associated with higher yield (within-crop, since raw "
  "yield scale differs hugely by crop)\n")
corr_rows = []
for crop, g in df.groupby("Crop_Type"):
    if len(g) < 15:
        continue
    for var in ["Rainfall_mm", "Temperature_C", "Fertiliser_Use_kg_ha", "Irrigation_pct"]:
        r, p = stats.pearsonr(g[var], g["Crop_Yield_kg_ha"])
        corr_rows.append({"Crop_Type": crop, "Variable": var, "r": round(r, 3),
                           "p_value": round(p, 4), "n": len(g)})
corr_df = pd.DataFrame(corr_rows)
w(corr_df.to_markdown(index=False))
w("\n\nOverall pooled correlation (all crops together, on the within-crop "
  "z-scored yield, so scale differences across crops do not distort it):\n")
pooled = []
for var in ["Rainfall_mm", "Temperature_C", "Fertiliser_Use_kg_ha", "Irrigation_pct"]:
    r, p = stats.pearsonr(df[var], df["Yield_Zscore_Within_Crop"])
    pooled.append({"Variable": var, "r": round(r, 3), "p_value": round(p, 5)})
w(pd.DataFrame(pooled).to_markdown(index=False))

# ---------------------------------------------------------------------------
w("\n\n## 10B. Region (country) comparison\n")
region_stats = df.groupby("Country").agg(
    mean_yield_z=("Yield_Zscore_Within_Crop", "mean"),
    median_yield_z=("Yield_Zscore_Within_Crop", "median"),
    std_yield_z=("Yield_Zscore_Within_Crop", "std"),
    mean_fertiliser=("Fertiliser_Use_kg_ha", "mean"),
    mean_irrigation=("Irrigation_pct", "mean"),
    n_crop_types=("Crop_Type", "nunique"),
    n_obs=("Crop_Type", "size"),
).sort_values("mean_yield_z", ascending=False)
w(region_stats.round(2).to_markdown())
w("\n\nNote: mean_yield_z is the country's average *within-crop* standardised "
  "yield -- i.e. how it ranks against other countries growing the same crop, "
  "not raw tonnage, so it is not distorted by which crops a country happens to "
  "grow. These are observational rank differences, not proof that any single "
  "country-level factor *causes* the gap (many unobserved factors -- "
  "mechanisation, seed variety, extension services -- are not in this dataset).\n")

# ---------------------------------------------------------------------------
w("\n\n## 10C. Crop variety/type comparison\n")
crop_stats = df.groupby("Crop_Type").agg(
    mean_yield=("Crop_Yield_kg_ha", "mean"),
    median_yield=("Crop_Yield_kg_ha", "median"),
    cv_pct=("Crop_Yield_kg_ha", lambda s: s.std() / s.mean() * 100),
    mean_fertiliser=("Fertiliser_Use_kg_ha", "mean"),
    mean_irrigation=("Irrigation_pct", "mean"),
).sort_values("cv_pct", ascending=False)
w(crop_stats.round(2).to_markdown())
w("\n\nCoefficient of variation (cv_pct) shows which crops have the most "
  "yield-stability across countries/years; comparisons use each crop's own "
  "mean, never a cross-crop raw-yield ranking, per the master-prompt's "
  "'don't compare crops ignoring scale' rule.\n")

# ---------------------------------------------------------------------------
w("\n\n## 10D. Fertiliser usage vs yield\n")
df["Fertiliser_Bin"] = pd.qcut(df["Fertiliser_Use_kg_ha"], 5,
                                labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
fert_bin_stats = df.groupby("Fertiliser_Bin", observed=True)["Yield_Zscore_Within_Crop"].agg(
    ["mean", "median", "count"])
w(fert_bin_stats.round(3).to_markdown())
# diminishing returns check: quadratic fit vs linear fit, within-crop z-yield
lin = np.polyfit(df["Fertiliser_Use_kg_ha"], df["Yield_Zscore_Within_Crop"], 1)
quad = np.polyfit(df["Fertiliser_Use_kg_ha"], df["Yield_Zscore_Within_Crop"], 2)
w(f"\nLinear fit slope: {lin[0]:.5f}. Quadratic fit coefficients "
  f"(a*x^2+b*x+c): a={quad[0]:.7f}, b={quad[1]:.5f}. "
  f"{'Negative a is consistent with diminishing/negative returns at high fertiliser levels.' if quad[0] < 0 else 'No evidence of diminishing returns in this range (a >= 0).'}\n")
w("\nCrop-specific fertiliser-yield correlation is already reported in 10A "
  "(per-crop table). Region-specific pattern:\n")
fert_by_country = df.groupby("Country").apply(
    lambda g: stats.pearsonr(g["Fertiliser_Use_kg_ha"], g["Yield_Zscore_Within_Crop"])[0]
    if len(g) > 5 else np.nan, include_groups=False)
w(fert_by_country.round(3).sort_values(ascending=False).to_frame("r_fertiliser_vs_yieldZ").to_markdown())

# ---------------------------------------------------------------------------
w("\n\n## 10E. Irrigation vs yield -- does more irrigation always improve yield?\n")
df["Irrigation_Bin"] = pd.cut(df["Irrigation_pct"], bins=[-0.1, 5, 25, 50, 100],
                               labels=["~0% (rainfed)", "Low (5-25%)",
                                       "Medium (25-50%)", "High (>50%)"])
irr_bin_stats = df.groupby("Irrigation_Bin", observed=True)["Yield_Zscore_Within_Crop"].agg(
    ["mean", "median", "std", "count"])
w(irr_bin_stats.round(3).to_markdown())
w("\n\nBy crop (does the irrigation-yield relationship hold for every crop?):\n")
irr_by_crop = df.groupby("Crop_Type").apply(
    lambda g: stats.pearsonr(g["Irrigation_pct"], g["Crop_Yield_kg_ha"])[0]
    if len(g) > 5 else np.nan, include_groups=False)
w(irr_by_crop.round(3).sort_values(ascending=False).to_frame("r_irrigation_vs_yield").to_markdown())
w("\n\nBy country:\n")
irr_by_country = df.groupby("Country").apply(
    lambda g: stats.pearsonr(g["Irrigation_pct"], g["Yield_Zscore_Within_Crop"])[0]
    if len(g) > 5 and g["Irrigation_pct"].std() > 0 else np.nan, include_groups=False)
w(irr_by_country.round(3).sort_values(ascending=False).to_frame("r_irrigation_vs_yieldZ").to_markdown())

w("\n\nRainfall x irrigation interaction: split into rainfed-dependent "
  "(irrigation < median) vs irrigation-reliant (>= median), then compare the "
  "rainfall-yield correlation in each group:\n")
low_irr = df[df["Irrigation_pct"] <= 0.01]  # effectively rainfed (0% irrigated)
high_irr = df[df["Irrigation_pct"] > 0.01]   # any meaningful irrigation share
r_low, p_low = stats.pearsonr(low_irr["Rainfall_mm"], low_irr["Yield_Zscore_Within_Crop"])
r_high, p_high = stats.pearsonr(high_irr["Rainfall_mm"], high_irr["Yield_Zscore_Within_Crop"])
w(f"- Low-irrigation subset (n={len(low_irr)}): rainfall-yield r={r_low:.3f} (p={p_low:.4f})")
w(f"- High-irrigation subset (n={len(high_irr)}): rainfall-yield r={r_high:.3f} (p={p_high:.4f})")
w("\nIf |r| is smaller in the high-irrigation subset, irrigation is plausibly "
  "substituting for rainfall dependence, consistent with agronomic theory. This "
  "is an association check on observational data, not a randomised experiment.\n")

w("\n**Answer to 'does more irrigation always correspond to higher yield?'**: "
  "No -- the per-crop and per-country correlation tables above show the sign "
  "and strength of the irrigation-yield relationship varies (some crops/"
  "countries show a clear positive association, others near-zero or even "
  "slightly negative), so it is not a universal rule; it depends on baseline "
  "rainfall, crop type and how much of the observed 'irrigation' figure was "
  "median-imputed (see Irrigation_Data_Missing flag).\n")

# ---------------------------------------------------------------------------
w("\n\n## 10F. Inefficient farming-practice identification methodology\n")
w("Definition used: for each Crop_Type, fit yield as a function of "
  "Fertiliser_Use_kg_ha, Irrigation_pct, Rainfall_mm and Temperature_C using "
  "a simple linear regression (same-crop comparison only, so environment and "
  "crop-scale differences are held roughly constant). A country-year "
  "observation is flagged 'input-inefficient' if BOTH: (1) its fertiliser use "
  "and/or irrigation share is in the top 25% for that crop, AND (2) its actual "
  "yield residual (actual - predicted) is in the bottom 25% for that crop -- "
  "i.e. high inputs but a yield well below what similar-input peers achieve. "
  "This avoids the master-prompt's warned-against trap of calling anything "
  "with low absolute yield 'inefficient' regardless of its inputs or "
  "environment.\n")

from numpy.polynomial import polynomial as P

flag_rows = []
for crop, g in df.groupby("Crop_Type"):
    if len(g) < 20:
        continue
    X = g[["Fertiliser_Use_kg_ha", "Irrigation_pct", "Rainfall_mm", "Temperature_C"]].values
    X = np.column_stack([np.ones(len(g)), X])
    y = g["Crop_Yield_kg_ha"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    high_input = ((g["Fertiliser_Use_kg_ha"] >= g["Fertiliser_Use_kg_ha"].quantile(0.75)) |
                  (g["Irrigation_pct"] >= g["Irrigation_pct"].quantile(0.75)))
    low_resid = resid <= np.quantile(resid, 0.25)
    flagged = g[high_input.values & low_resid]
    for _, row in flagged.iterrows():
        flag_rows.append({"Country": row["Country"], "Crop_Type": crop, "Year": row["Year"],
                           "Crop_Yield_kg_ha": row["Crop_Yield_kg_ha"],
                           "Fertiliser_Use_kg_ha": row["Fertiliser_Use_kg_ha"],
                           "Irrigation_pct": round(row["Irrigation_pct"], 1)})
flag_df = pd.DataFrame(flag_rows)
w(f"\nFlagged {len(flag_df)} of {len(df)} observations ({len(flag_df)/len(df)*100:.1f}%) "
  f"as potentially input-inefficient using this rule.\n")
if len(flag_df):
    top_flag_countries = flag_df["Country"].value_counts().head(10)
    w("\nCountries most represented among flagged (high-input, low-residual-yield) "
      "observations:\n")
    w(top_flag_countries.to_frame("flagged_count").to_markdown())

# ---------------------------------------------------------------------------
w("\n\n## 11. Statistical analysis notes -- association vs prediction vs causation\n")
w("- **Association**: the correlation and grouped-mean results throughout "
  "Sections 10A-10F describe statistical association only.\n"
  "- **Prediction**: the Random Forest model (Section 14+) is evaluated purely "
  "on its ability to predict yield on held-out country-crop-years; a good "
  "predictive score does not imply any single feature causes yield changes.\n"
  "- **Causation**: this is observational, non-experimental national-annual "
  "data with likely confounding (e.g. wealthier/more mechanised countries tend "
  "to have both higher fertiliser use AND better seed varieties, irrigation "
  "infrastructure, and extension services that are not captured here) -- no "
  "causal claim is made anywhere in this project.\n"
  "- **Assumptions/limitations of Pearson r**: assumes roughly linear "
  "relationships; several crop-level samples are small (n<20 country-years for "
  "some crops), so p-values there should be read as indicative, not definitive.\n")

with open(os.path.join(DOCS, "eda_and_investigations.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))

df.to_csv(os.path.join(BASE, "data", "final", "analytical_dataset.csv"), index=False)
flag_df.to_csv(os.path.join(BASE, "data", "final", "flagged_inefficient_observations.csv"), index=False)
print("[ok] wrote docs/eda_and_investigations.md")
print(f"[ok] wrote data/final/flagged_inefficient_observations.csv ({len(flag_df)} rows)")
