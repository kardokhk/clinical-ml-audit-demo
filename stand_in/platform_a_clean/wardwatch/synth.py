"""Seeded generator for a fictional ward-deterioration extract.

Writes patients.csv, admissions.csv, vitals.csv and labs.csv. All people are
invented; there are no names or contact details.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_TIME = pd.Timestamp("2025-01-01 00:00:00")


def generate(out_dir: str | Path, n_patients: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    patient_ids = [f"P{i:05d}" for i in range(n_patients)]
    birth_year = rng.integers(1930, 2005, n_patients)
    sex = rng.choice(["F", "M"], n_patients)
    frailty = rng.normal(0.0, 1.0, n_patients)
    patients = pd.DataFrame({"patient_id": patient_ids, "birth_year": birth_year, "sex": sex})

    adm_rows, vit_rows, lab_rows = [], [], []
    adm_counter = 0
    for p_idx, pid in enumerate(patient_ids):
        n_adm = 1 + rng.poisson(1.2)
        for _ in range(n_adm):
            adm_counter += 1
            aid = f"A{adm_counter:06d}"
            admit = BASE_TIME + pd.Timedelta(hours=float(rng.uniform(0, 24 * 600)))
            pred_time = admit + pd.Timedelta(hours=6)
            age = admit.year - int(birth_year[p_idx])
            acute = rng.normal(0.0, 1.0)
            logit = -2.6 + 0.9 * frailty[p_idx] + 1.0 * acute + 0.02 * (age - 60)
            label = int(rng.uniform() < 1.0 / (1.0 + np.exp(-logit)))
            stay_h = float(rng.uniform(24, 240))
            discharge = admit + pd.Timedelta(hours=stay_h)
            icu_time = pred_time + pd.Timedelta(hours=float(rng.uniform(1, 47))) if label else pd.NaT
            adm_rows.append((aid, pid, admit, pred_time, discharge, icu_time, label))

            severity = 0.6 * acute + 0.5 * frailty[p_idx]
            t = 0.0
            while t < stay_h:
                ts = admit + pd.Timedelta(hours=t)
                worsening = 1.5 if (label and ts > pred_time) else 0.0
                vit_rows.append((
                    aid, ts,
                    round(80 + 8 * severity + 12 * worsening + rng.normal(0, 6), 1),
                    round(16 + 1.5 * severity + 4 * worsening + rng.normal(0, 2), 1),
                    round(125 - 6 * severity - 15 * worsening + rng.normal(0, 10), 1),
                    round(36.9 + 0.25 * severity + 0.4 * worsening + rng.normal(0, 0.3), 2),
                    round(min(100.0, 97 - 1.2 * severity - 4 * worsening + rng.normal(0, 1.2)), 1),
                ))
                t += float(rng.uniform(2, 5))
            for lt in (1.0, 20.0, 44.0):
                if lt >= stay_h:
                    break
                ts = admit + pd.Timedelta(hours=lt)
                worsening = 1.0 if (label and ts > pred_time) else 0.0
                lab_rows.append((
                    aid, ts,
                    round(max(0.3, 1.2 + 0.5 * severity + 1.5 * worsening + rng.normal(0, 0.3)), 2),
                    round(max(20.0, 80 + 15 * severity + 30 * worsening + rng.normal(0, 10)), 1),
                    round(max(1.0, 8 + 1.5 * severity + 3 * worsening + rng.normal(0, 1.5)), 1),
                ))

    admissions = pd.DataFrame(adm_rows, columns=[
        "admission_id", "patient_id", "admit_time", "prediction_time",
        "discharge_time", "icu_transfer_time", "deteriorated_48h"])
    vitals = pd.DataFrame(vit_rows, columns=[
        "admission_id", "charttime", "heart_rate", "resp_rate", "sbp", "temp_c", "spo2"])
    labs = pd.DataFrame(lab_rows, columns=["admission_id", "charttime", "lactate", "creatinine", "wbc"])

    patients.to_csv(out / "patients.csv", index=False)
    admissions.to_csv(out / "admissions.csv", index=False)
    vitals.to_csv(out / "vitals.csv", index=False)
    labs.to_csv(out / "labs.csv", index=False)
    return {"patients": len(patients), "admissions": len(admissions),
            "vitals": len(vitals), "labs": len(labs)}
