"""Feature construction at the prediction time (6 h after admission)."""
from __future__ import annotations

import pandas as pd

VITALS = ["heart_rate", "resp_rate", "sbp", "temp_c", "spo2"]
LABS = ["lactate", "creatinine", "wbc"]


def _window(events: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    merged = events.merge(admissions[["admission_id", "admit_time", "prediction_time"]],
                          on="admission_id", how="inner")
    keep = (merged["charttime"] >= merged["admit_time"]) & (merged["charttime"] <= merged["prediction_time"])
    return merged.loc[keep].sort_values(["admission_id", "charttime"], kind="mergesort")


def vital_features(vitals: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    w = _window(vitals, admissions)
    g = w.groupby("admission_id", sort=True)
    last = g[VITALS].last().add_suffix("_last")
    worst = pd.concat([
        g["heart_rate"].max().rename("heart_rate_max"),
        g["resp_rate"].max().rename("resp_rate_max"),
        g["sbp"].min().rename("sbp_min"),
        g["temp_c"].max().rename("temp_c_max"),
        g["spo2"].min().rename("spo2_min"),
    ], axis=1)
    count = g.size().rename("n_vitals")
    return pd.concat([last, worst, count], axis=1)


def lab_features(labs: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    w = _window(labs, admissions)
    return w.groupby("admission_id", sort=True)[LABS].last().add_suffix("_last")


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    adm = tables["admissions"].merge(tables["patients"], on="patient_id", how="left")
    base = adm.set_index("admission_id")
    out = pd.DataFrame(index=base.index)
    out["age"] = base["admit_time"].dt.year - base["birth_year"]
    out["sex_male"] = (base["sex"] == "M").astype(int)
    out = out.join(vital_features(tables["vitals"], tables["admissions"]), how="left")
    out = out.join(lab_features(tables["labs"], tables["admissions"]), how="left")
    out["n_vitals"] = out["n_vitals"].fillna(0)
    return out.sort_index()


def feature_columns() -> list[str]:
    cols = ["age", "sex_male"]
    cols += [f"{v}_last" for v in VITALS]
    cols += ["heart_rate_max", "resp_rate_max", "sbp_min", "temp_c_max", "spo2_min", "n_vitals"]
    cols += [f"{lab}_last" for lab in LABS]
    return cols
