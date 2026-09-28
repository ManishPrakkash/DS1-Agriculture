"""
Reproducibility entry point (Section 21). Running this single script from a
clean environment (Python 3.11+, packages in requirements.txt) regenerates
every downloaded raw file (idempotent), every processed/final dataset, every
chart in visualizations/, the trained model in model/, and every markdown
report in docs/ -- in the same deterministic order used to build this project
(same RANDOM_SEED=42 in 07_random_forest_model.py).
"""
import runpy
import os

STEPS = [
    "01_download_data.py",
    "02_build_dataset.py",
    "03_data_audit.py",
    "04_clean_dataset.py",
    "05_eda_and_investigations.py",
    "06_visualizations.py",
    "07_random_forest_model.py",
]

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        runpy.run_path(os.path.join(here, step), run_name="__main__")
