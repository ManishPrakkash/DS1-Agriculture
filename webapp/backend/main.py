"""
FastAPI backend for the "Why Is Crop Yield Different?" analytics project.

Serves:
- The static dashboard frontend (webapp/frontend/)
- The 4 required charts + supplementary chart (visualizations/)
- /api/meta        -> dataset ranges, categories, model metrics, insights, action plan
- /api/predict      -> live Random Forest yield prediction from user inputs

Wires up the actual artifacts produced by scripts/07_random_forest_model.py:
model/random_forest_yield_model.joblib + model/feature_columns.joblib.
No retraining happens here -- this only loads and serves the already-trained,
already-evaluated model (see docs/PROJECT_REPORT.md for how it was built).
"""
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE, "model")
VIZ_DIR = os.path.join(BASE, "visualizations")
DATA_FINAL = os.path.join(BASE, "data", "final")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = FastAPI(title="Crop Yield Analytics API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- load trained artifacts once at startup ---------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "random_forest_yield_model.joblib"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
with open(os.path.join(MODEL_DIR, "model_report.json")) as f:
    model_report = json.load(f)

df = pd.read_csv(os.path.join(DATA_FINAL, "analytical_dataset.csv"))

COUNTRIES = sorted(df["Country"].unique().tolist())
CROPS = sorted(df["Crop_Type"].unique().tolist())

# yield varies ~40x across crops (e.g. sugar cane vs. wheat), so a single
# global range makes most predictions look pinned to one end of a gauge --
# expose per-crop ranges instead, matching the report's own within-crop
# standardisation approach (Section 8, Yield_Zscore_Within_Crop).
CROP_YIELD_RANGES = {
    crop: {
        "min": round(float(g["Crop_Yield_kg_ha"].min()), 2),
        "max": round(float(g["Crop_Yield_kg_ha"].max()), 2),
        "mean": round(float(g["Crop_Yield_kg_ha"].mean()), 2),
    }
    for crop, g in df.groupby("Crop_Type")
}

NUMERIC_RANGES = {
    col: {
        "min": round(float(df[col].min()), 2),
        "max": round(float(df[col].max()), 2),
        "mean": round(float(df[col].mean()), 2),
    }
    for col in ["Year", "Fertiliser_Use_kg_ha", "Irrigation_pct", "Temperature_C", "Rainfall_mm", "Crop_Yield_kg_ha"]
}

# --- static content, transcribed from docs/PROJECT_REPORT.md ----------------
INSIGHTS = [
    {
        "title": "Fertiliser is the most consistent input-driven yield lever",
        "evidence": "Pooled within-crop correlation r=+0.328 (p<0.001), positive in 18/22 countries and 9/10 crops; #1-2 non-crop-identity feature in both Random Forest importance methods.",
        "limitation": "Association only. Permutation importance (0.050) is modest in absolute terms -- fertiliser alone explains only part of the cross-country yield gap.",
    },
    {
        "title": "Diminishing returns to fertiliser above ~150-200 kg/ha",
        "evidence": "The binned fertiliser-yield curve flattens past the middle bins; a quadratic fit has a negative curvature term (a=-6.2e-6).",
        "limitation": "Bin-level pattern, not a per-farm dose-response curve.",
    },
    {
        "title": "More irrigation does not always mean higher yield",
        "evidence": "Per-crop correlation ranges +0.70 (cassava) to -0.35 (sorghum/barley); the 25-50% and >50% irrigation bands score lower mean standardised yield than the 5-25% band.",
        "limitation": "Confounded by which countries irrigate out of rainfall necessity; 4 countries' irrigation figures are median-imputed.",
    },
    {
        "title": "Regional yield gaps track input access more than climate",
        "evidence": "Germany (+1.61 SD) to Nigeria (-1.33 SD) within-crop z-score spread is ~3 SD; Nigeria's fertiliser use (8.4 kg/ha) is ~1/50th of Egypt's (411 kg/ha) while Egypt ranks near the top.",
        "limitation": "Unobserved factors (mechanisation, seed variety, market access) are very likely also involved.",
    },
    {
        "title": "11.5% of observations show a high-input/low-relative-yield pattern",
        "evidence": "493/4,286 rows flagged by the residual-based methodology (Section 10F); concentrated in China (mainland), Viet Nam, Russia, Pakistan.",
        "limitation": "A screening heuristic on a simple linear fit, not a causal diagnosis of why those observations underperform.",
    },
    {
        "title": "The Random Forest is highly accurate but leans on crop identity",
        "evidence": "Test R²=0.983, MAE≈959 kg/ha -- but most accuracy comes from knowing the crop's natural yield scale rather than from environmental/input features.",
        "limitation": "Strong for planning-level forecasting; not a precise tool for 'what-if we changed fertiliser by X' simulation at the national level.",
    },
    {
        "title": "Rainfall/temperature show weak negative correlations that are likely confounded",
        "evidence": "r=-0.181 (rainfall), r=-0.256 (temperature) -- but the coolest, driest-average countries in this sample (Germany, France, Canada) are also the highest-input, most mechanised producers.",
        "limitation": "Should not be read as 'less rain/cooler is agronomically better'.",
    },
]

ACTION_PLAN = [
    {
        "evidence": "Fertiliser positively associated with yield in 18/22 countries, diminishing returns above ~150-200 kg/ha",
        "action": "Prioritise fertiliser access/affordability in low-use, low-yield countries (e.g. Nigeria at 8.4 kg/ha) toward the ~100-200 kg/ha range; avoid pushing already-high systems (e.g. Egypt at 411 kg/ha) higher without justification",
        "purpose": "Capture the steep part of the yield-response curve; avoid wasted input spend where returns are flat",
        "limitation": "Country-year data can't show whether the binding constraint is price, credit, supply chain, or advice -- needs farm-level follow-up",
    },
    {
        "evidence": "Irrigation's yield effect is inconsistent and sometimes negative across crops/countries",
        "action": "Don't treat 'expand irrigation' as a default lever; evaluate investment crop-by-crop and region-by-region against the specific baseline",
        "purpose": "Avoid capital misallocation to irrigation expansion where the historical record doesn't support a payoff",
        "limitation": "4 countries' irrigation figures were imputed; can't separate irrigation quality from quantity at this granularity",
    },
    {
        "evidence": "493 high-input/low-relative-yield country-years concentrated in China, Vietnam, Russia, Pakistan",
        "action": "Flag these country-crop combinations for an input-efficiency review against same-crop peers before recommending further input increases",
        "purpose": "Redirect investment from 'more inputs' toward 'better-matched inputs/practices'",
        "limitation": "A screening heuristic, not a causal diagnosis of why those observations underperform",
    },
    {
        "evidence": "Region gaps (Germany +1.61 SD vs Nigeria -1.33 SD) track fertiliser/irrigation access more than climate",
        "action": "Frame national yield-improvement strategy around input-access programmes rather than assuming climate is the binding constraint",
        "purpose": "Target the levers this data actually shows moving with yield",
        "limitation": "Unobserved factors (mechanisation, seed variety, extension services) very likely also matter and aren't measured here",
    },
    {
        "evidence": "Model is accurate for yield magnitude by crop but leans on crop identity more than inputs",
        "action": "Use the model for planning-level yield forecasting, not fine-grained 'what if we changed fertiliser by X' simulation",
        "purpose": "Match the tool to what the evidence supports",
        "limitation": "For input-sensitivity analysis, a crop-specific model or farm-level data would be needed",
    },
]


class PredictRequest(BaseModel):
    country: str
    crop_type: str = Field(..., alias="crop_type")
    year: int = Field(ge=1990, le=2100)
    fertiliser_use_kg_ha: float = Field(ge=0)
    irrigation_pct: float = Field(ge=0, le=100)
    temperature_c: float
    rainfall_mm: float = Field(ge=0)

    class Config:
        populate_by_name = True


@app.get("/api/meta")
def get_meta():
    return {
        "countries": COUNTRIES,
        "crops": CROPS,
        "ranges": NUMERIC_RANGES,
        "crop_yield_ranges": CROP_YIELD_RANGES,
        "dataset": {
            "n_rows": int(len(df)),
            "n_countries": len(COUNTRIES),
            "n_crops": len(CROPS),
            "year_min": int(df["Year"].min()),
            "year_max": int(df["Year"].max()),
        },
        "model": {
            "test_mae": round(model_report["rf_test_metrics"]["MAE"], 1),
            "test_rmse": round(model_report["rf_test_metrics"]["RMSE"], 1),
            "test_r2": round(model_report["rf_test_metrics"]["R2"], 4),
            "baseline_r2": round(model_report["baseline_test_metrics"]["R2"], 4),
            "n_train": model_report["n_train"],
            "n_test": model_report["n_test"],
            "best_params": model_report["best_params"],
            "top_permutation_importances": model_report["top_permutation_importances"],
        },
        "insights": INSIGHTS,
        "action_plan": ACTION_PLAN,
        "charts": [
            {"file": "fig1_yield_distribution_by_crop.png", "caption": "Yield distribution by crop (log scale) -- why every comparison here is within-crop standardised."},
            {"file": "fig2_region_comparison.png", "caption": "Mean within-crop yield z-score by country."},
            {"file": "fig3_fertiliser_vs_yield.png", "caption": "Binned fertiliser-vs-yield curve showing diminishing returns."},
            {"file": "fig4_irrigation_vs_yield.png", "caption": "Irrigation-share bands vs. yield -- directly tests (and refutes) 'more irrigation always improves yield'."},
        ],
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.country not in COUNTRIES:
        raise HTTPException(400, f"Unknown country '{req.country}'. Must be one of the dataset's {len(COUNTRIES)} countries.")
    if req.crop_type not in CROPS:
        raise HTTPException(400, f"Unknown crop '{req.crop_type}'. Must be one of the dataset's {len(CROPS)} crops.")

    row = pd.DataFrame([{
        "Year": req.year,
        "Fertiliser_Use_kg_ha": req.fertiliser_use_kg_ha,
        "Irrigation_pct": req.irrigation_pct,
        "Temperature_C": req.temperature_c,
        "Rainfall_mm": req.rainfall_mm,
        # user supplies a real value, so this observation is not an imputed one
        "Irrigation_Data_Missing": 0,
        "Country": req.country,
        "Crop_Type": req.crop_type,
    }])
    encoded = pd.get_dummies(row, columns=["Country", "Crop_Type"])
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    pred_kg_ha = float(model.predict(encoded)[0])

    # per-tree spread as a rough uncertainty band (not a calibrated interval)
    per_tree = [float(t.predict(encoded)[0]) for t in model.estimators_]
    per_tree.sort()
    lo = per_tree[int(0.1 * len(per_tree))]
    hi = per_tree[int(0.9 * len(per_tree)) - 1]

    return {
        "predicted_yield_kg_ha": round(pred_kg_ha, 1),
        "predicted_yield_tonnes_ha": round(pred_kg_ha / 1000, 2),
        "uncertainty_band_kg_ha": [round(lo, 1), round(hi, 1)],
        "note": "This is a national-annual model (one row = country-crop-year), not a farm-level model. Treat this as a planning-level estimate, not a precise farm-level forecast -- see the model-interpretation caveat on the dashboard.",
    }


app.mount("/charts", StaticFiles(directory=VIZ_DIR), name="charts")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
