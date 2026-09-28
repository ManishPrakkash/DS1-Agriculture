"""
Section 13 (feature engineering), 14 (Random Forest), 15 (evaluation) and
16 (interpretation) of the master execution prompt.

Target: Crop_Yield_kg_ha.
Features: Country, Crop_Type (one-hot encoded categoricals), Year,
Fertiliser_Use_kg_ha, Irrigation_pct, Temperature_C, Rainfall_mm,
Irrigation_Data_Missing (the imputation flag -- kept as a feature so the model
can learn to discount imputed irrigation values instead of treating them as
equally reliable).

No target leakage: every feature is knowable independently of the harvested
yield outcome (climate/fertiliser/irrigation are inputs decided/observed before
or during the growing season, not derived from yield). No identifier is used
as a feature (Country/Crop_Type are genuine agronomic categories the challenge
explicitly asks us to compare across -- not row identifiers -- there is no
Farm_ID in this national-level dataset).
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE, "model")
VIZ = os.path.join(BASE, "visualizations")
DOCS = os.path.join(BASE, "docs")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_SEED = 42

df = pd.read_csv(os.path.join(BASE, "data", "final", "analytical_dataset.csv"))

FEATURES_NUM = ["Year", "Fertiliser_Use_kg_ha", "Irrigation_pct", "Temperature_C",
                "Rainfall_mm", "Irrigation_Data_Missing"]
FEATURES_CAT = ["Country", "Crop_Type"]
TARGET = "Crop_Yield_kg_ha"

X = pd.get_dummies(df[FEATURES_NUM + FEATURES_CAT], columns=FEATURES_CAT)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# --- baseline: predict the mean (sanity floor) -------------------------------
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

# --- Random Forest with a small, justified hyperparameter search ------------
param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [None, 12, 20],
    "min_samples_leaf": [1, 2, 4],
}
rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

train_pred = best_rf.predict(X_train)
test_pred = best_rf.predict(X_test)


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


m_train = metrics(y_train, train_pred)
m_test = metrics(y_test, test_pred)
m_baseline = metrics(y_test, baseline_pred)

# --- feature importance (impurity-based + permutation) ----------------------
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
perm = permutation_importance(best_rf, X_test, y_test, n_repeats=15,
                               random_state=RANDOM_SEED, n_jobs=-1)
perm_importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

top_n = 12
fig, ax = plt.subplots(figsize=(9, 6))
top = importances.head(top_n).sort_values()
ax.barh(top.index, top.values, color="#2c6e91")
ax.set_xlabel("Random Forest impurity-based feature importance (fraction of variance explained)")
ax.set_title("Top features driving crop-yield predictions (Random Forest)")
fig.tight_layout()
fig.savefig(os.path.join(VIZ, "fig5_feature_importance_supplementary.png"))
plt.close(fig)

joblib.dump(best_rf, os.path.join(MODEL_DIR, "random_forest_yield_model.joblib"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))

report = {
    "random_seed": RANDOM_SEED,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "best_params": grid.best_params_,
    "baseline_test_metrics": m_baseline,
    "rf_train_metrics": m_train,
    "rf_test_metrics": m_test,
    "top_impurity_importances": importances.head(15).round(4).to_dict(),
    "top_permutation_importances": perm_importances.head(15).round(4).to_dict(),
}
with open(os.path.join(MODEL_DIR, "model_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
