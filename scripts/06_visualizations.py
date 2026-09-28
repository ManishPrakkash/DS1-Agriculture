"""
Section 12: exactly 4 major, decision-useful visualisations.
Each has a title, axis labels with units, legend where needed, and is saved as
a PNG in visualizations/. Written interpretations live in
docs/visualization_interpretations.md.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")
VIZ = os.path.join(BASE, "visualizations")
os.makedirs(VIZ, exist_ok=True)
df = pd.read_csv(os.path.join(BASE, "data", "final", "analytical_dataset.csv"))

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

interp = []

# ---------------------------------------------------------------------------
# 1. Yield distribution by crop type (log scale, since scales differ hugely)
# ---------------------------------------------------------------------------
order = df.groupby("Crop_Type")["Crop_Yield_kg_ha"].median().sort_values().index
fig, ax = plt.subplots(figsize=(9, 6))
data = [df.loc[df["Crop_Type"] == c, "Crop_Yield_kg_ha"].values for c in order]
ax.boxplot(data, vert=False, tick_labels=list(order), showfliers=True)
ax.set_xscale("log")
ax.set_xlabel("Crop yield (kg per hectare, log scale)")
ax.set_ylabel("Crop type")
ax.set_title("Fig. 1 — Crop yield distribution by crop type (2000-2022, 22 countries)")
ax.grid(axis="x", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "fig1_yield_distribution_by_crop.png"))
plt.close(fig)
interp.append(
    "**Fig. 1 — Yield distribution by crop type.** Median national yields span "
    "roughly two orders of magnitude, from seed cotton/sorghum (~2,000-2,400 kg/ha) "
    "up to sugar cane (~69,000 kg/ha), because these units are all in kg of raw "
    "harvested crop per hectare and biomass-heavy crops (sugar cane, potatoes, "
    "cassava) are simply much heavier per hectare than grains. The log scale is "
    "necessary to show all ten crops on one readable axis; this chart is why "
    "every cross-crop comparison elsewhere in this project uses the within-crop "
    "z-score, never raw kg/ha."
)

# ---------------------------------------------------------------------------
# 2. Region comparison: mean within-crop yield z-score by country
# ---------------------------------------------------------------------------
region = (df.groupby("Country")["Yield_Zscore_Within_Crop"].mean()
          .sort_values())
fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#c0392b" if v < 0 else "#1f7a4c" for v in region.values]
ax.barh(region.index, region.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Mean within-crop yield z-score (0 = average performer for the crops it grows)")
ax.set_ylabel("Country")
ax.set_title("Fig. 2 — Regional yield performance, standardised within each crop")
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "fig2_region_comparison.png"))
plt.close(fig)
interp.append(
    "**Fig. 2 — Region comparison.** Standardising within crop (so a country is "
    "compared only against other countries growing the *same* crop) removes the "
    "crop-mix distortion of Fig. 1. Germany, France and Egypt rank as the "
    "strongest relative performers across their crop portfolios; Nigeria, "
    "Ethiopia, Kenya and Pakistan rank lowest. This is an association ranking, "
    "not a causal claim — see docs/eda_and_investigations.md Section 10B for "
    "the caveats (mechanisation, seed variety and extension services are not "
    "observed in this dataset)."
)

# ---------------------------------------------------------------------------
# 3. Fertiliser usage vs yield (binned means with error bars, showing
#    diminishing returns)
# ---------------------------------------------------------------------------
df["Fertiliser_Bin_Viz"] = pd.qcut(df["Fertiliser_Use_kg_ha"], 8, duplicates="drop")
bin_stats = df.groupby("Fertiliser_Bin_Viz", observed=True)["Yield_Zscore_Within_Crop"].agg(
    ["mean", "sem"])
mid_x = [iv.mid for iv in bin_stats.index]
fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(mid_x, bin_stats["mean"], yerr=1.96 * bin_stats["sem"], marker="o",
            capsize=4, color="#2c6e91", ecolor="#88b7d5", linewidth=2)
ax.set_xlabel("Fertiliser use (kg nutrient N+P2O5+K2O per hectare of cropland, national annual average)")
ax.set_ylabel("Mean within-crop yield z-score (95% CI)")
ax.set_title("Fig. 3 — Fertiliser use vs. standardised yield (8 equal-sized bins)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "fig3_fertiliser_vs_yield.png"))
plt.close(fig)
interp.append(
    "**Fig. 3 — Fertiliser vs. yield.** Standardised yield rises with fertiliser "
    "use through the low-to-middle bins, then the curve flattens (and the "
    "quadratic fit in docs/eda_and_investigations.md Section 10D has a negative "
    "curvature coefficient), consistent with diminishing returns at the highest "
    "application rates rather than a straight-line 'more fertiliser = more "
    "yield' relationship. Error bars are 95% confidence intervals on the bin "
    "mean, not prediction intervals for an individual country-year."
)

# ---------------------------------------------------------------------------
# 4. Irrigation vs yield, split rainfed vs irrigated, testing "always improves?"
# ---------------------------------------------------------------------------
bins = [-0.1, 0.01, 5, 25, 50, 100]
labels = ["0% (rainfed)", "0-5%", "5-25%", "25-50%", ">50%"]
df["Irrigation_Bin_Viz"] = pd.cut(df["Irrigation_pct"], bins=bins, labels=labels)
irr_stats = df.groupby("Irrigation_Bin_Viz", observed=True)["Yield_Zscore_Within_Crop"].agg(
    ["mean", "sem", "count"])
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(irr_stats.index.astype(str), irr_stats["mean"],
              yerr=1.96 * irr_stats["sem"], capsize=5,
              color="#3b7dd8", edgecolor="black")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Share of agricultural land irrigated (%)")
ax.set_ylabel("Mean within-crop yield z-score (95% CI)")
ax.set_title("Fig. 4 — Does more irrigation always mean higher yield? (by irrigation-share band)")
for i, (m, c) in enumerate(zip(irr_stats["mean"], irr_stats["count"])):
    ax.annotate(f"n={c}", (i, m), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "fig4_irrigation_vs_yield.png"))
plt.close(fig)
interp.append(
    "**Fig. 4 — Irrigation vs. yield.** Standardised yield does NOT rise "
    "monotonically with irrigation share: the 0-5% and 5-25% irrigation bands "
    "actually out-perform both the fully rainfed band and the >25% irrigated "
    "bands. Sample sizes (annotated) are uneven and the >25% bands are "
    "dominated by lower-baseline-yield South Asian countries (India, Pakistan, "
    "Bangladesh) that irrigate heavily out of necessity in drier/less-mechanised "
    "systems, which is a plausible confounder — this chart is direct evidence "
    "against the assumption that 'more irrigation always improves yield', "
    "matching the correlation-table finding in Section 10E of "
    "docs/eda_and_investigations.md."
)

with open(os.path.join(BASE, "docs", "visualization_interpretations.md"), "w", encoding="utf-8") as f:
    f.write("# Visualisation Interpretations\n\n" + "\n\n".join(interp) + "\n")

print("[ok] wrote 4 PNGs to visualizations/ and docs/visualization_interpretations.md")
