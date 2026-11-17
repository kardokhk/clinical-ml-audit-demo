"""Build a readmit inference input for given admissions, from the README.

The README documents outputs/features.csv (one row per index admission with the 12 feature columns)
and the inference input schema; this copies the documented columns for the requested admissions.
"""
from pathlib import Path

import pandas as pd

SCHEMA = ["patient_id", "admission_id", "age_years", "length_of_stay_days", "prior_admissions_365d",
          "creatinine_last", "hemoglobin_last", "sodium_last", "wbc_last", "temperature_max_c",
          "sex", "admission_type", "ward", "discharge_disposition"]


def make_input(sandbox: Path, ids: list, out_dir: Path) -> Path:
    feats = pd.read_csv(sandbox / "outputs" / "features.csv", dtype={"admission_id": str, "patient_id": str})
    feats = feats[feats["admission_id"].isin(set(map(str, ids)))]
    cols = [c for c in SCHEMA if c in feats.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "admissions_to_score.csv"
    feats[cols].to_csv(path, index=False)
    return path
