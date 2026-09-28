# Crop Yield Analytics — Web App

Frontend dashboard + live prediction UI, wired to the trained Random Forest
model from `../model/random_forest_yield_model.joblib`.

## Run

```
pip install -r ../requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Then open http://localhost:8000

- **Dashboard** tab: dataset/model stat cards, the 4 required charts, the 7
  key insights, and the practical action plan (all sourced from
  `../docs/PROJECT_REPORT.md`).
- **Predict Yield** tab: form wired to `POST /api/predict`, which loads the
  actual trained model and reproduces the exact one-hot encoding used at
  training time (`model/feature_columns.joblib`) before predicting.

No retraining happens in the web app — it only serves the already-evaluated
model. Retrain via `python scripts/07_random_forest_model.py` from the project
root if the underlying data or model changes.
