"""Seeded generator for a synthetic EHR extract.

Produces four tables (patients, admissions, discharges, labs) for fictional
patients identified only by opaque ids. Used for development, tests and demos.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import TABLE_COLUMNS, WARDS

LAB_BASELINES = {
    # lab_name: (mean, sd, frailty coefficient)
    "creatinine": (0.95, 0.20, 0.35),
    "hemoglobin": (13.2, 1.3, -0.9),
    "sodium": (139.0, 2.8, -0.8),
    "wbc": (7.5, 2.0, 1.1),
    "temperature": (36.9, 0.35, 0.25),
}


class SyntheticEHRGenerator:
    """Simulate admissions for a population with a latent frailty score."""

    def __init__(self, n_patients, start_date, end_date, seed, id_prefix=""):
        if n_patients <= 0:
            raise ValueError("n_patients must be positive")
        self.n_patients = int(n_patients)
        self.start = pd.Timestamp(start_date)
        self.end = pd.Timestamp(end_date)
        if self.end <= self.start:
            raise ValueError("end_date must be after start_date")
        self.seed = int(seed)
        self.id_prefix = id_prefix
        self.rng = np.random.default_rng(self.seed)
        self._admission_counter = 0
        self._lab_counter = 0

    def generate(self):
        patients, admissions, discharges, labs = [], [], [], []
        for i in range(self.n_patients):
            patient = self._make_patient(i)
            patients.append(patient)
            self._simulate_patient(patient, admissions, discharges, labs)
        tables = {
            "patients": pd.DataFrame(patients),
            "admissions": pd.DataFrame(admissions),
            "discharges": pd.DataFrame(discharges),
            "labs": pd.DataFrame(labs),
        }
        tables["labs"] = self._add_chart_jitter(tables["labs"])
        for name, columns in TABLE_COLUMNS.items():
            tables[name] = tables[name][[c for c in columns if c in tables[name].columns]]
        return tables

    def _make_patient(self, i):
        rng = self.rng
        return {
            "patient_id": f"{self.id_prefix}P{i:06d}",
            "birth_year": int(rng.integers(1930, 2004)),
            "sex": str(rng.choice(["F", "M"])),
            "_frailty": float(rng.normal(0.0, 1.0)),
        }

    def _next_admission_id(self):
        self._admission_counter += 1
        return f"{self.id_prefix}A{self._admission_counter:07d}"

    def _simulate_patient(self, patient, admissions, discharges, labs):
        rng = self.rng
        frailty = patient.pop("_frailty")
        t = self.start + pd.Timedelta(days=float(rng.uniform(0, 420)))
        admission_type = "emergency" if rng.random() < 0.7 else "elective"
        n_prior = 0
        while t < self.end:
            age = t.year - patient["birth_year"]
            admission_id = self._next_admission_id()
            los_days = float(max(0.2, rng.gamma(2.0, 1.6 + 0.5 * max(frailty, 0.0))))
            discharge_time = t + pd.Timedelta(days=los_days)
            p_death = 0.01 + 0.015 * max(frailty, 0.0) + 0.0004 * max(age - 60, 0)
            u = rng.random()
            if u < p_death:
                disposition = "died"
            elif u < p_death + 0.12 + 0.05 * max(frailty, 0.0):
                disposition = "facility"
            elif u < 0.45:
                disposition = "home_with_services"
            else:
                disposition = "home"
            admissions.append({
                "admission_id": admission_id,
                "patient_id": patient["patient_id"],
                "admit_time": t.floor("min"),
                "admission_type": admission_type,
                "ward": str(rng.choice(WARDS)),
            })
            discharges.append({
                "admission_id": admission_id,
                "discharge_time": discharge_time.floor("min"),
                "discharge_disposition": disposition,
            })
            stay_labs = self._stay_labs(admission_id, t, los_days, frailty)
            labs.extend(stay_labs)
            if disposition == "died":
                break
            last_creat = next((r["_mgdl"] for r in reversed(stay_labs) if r["lab_name"] == "creatinine"), 1.0)
            logit = (
                -2.3
                + 0.65 * frailty
                + 0.02 * (age - 60)
                + 0.45 * (disposition == "facility")
                + 0.25 * (admission_type == "emergency")
                + 0.6 * (last_creat > 1.6)
                + 0.18 * min(n_prior, 4)
                + 0.05 * min(los_days, 10)
            )
            readmitted = rng.random() < 1.0 / (1.0 + np.exp(-logit))
            if rng.random() < 0.3:
                labs.append(self._follow_up_lab(admission_id, discharge_time, frailty, readmitted))
            n_prior += 1
            if readmitted:
                t = discharge_time + pd.Timedelta(days=float(rng.uniform(1, 29)))
                admission_type = "emergency"
            else:
                if rng.random() < 0.4:
                    break
                t = discharge_time + pd.Timedelta(days=float(31 + rng.exponential(220)))
                admission_type = "emergency" if rng.random() < 0.65 else "elective"

    def _lab_row(self, admission_id, name, value_mgdl, charted):
        self._lab_counter += 1
        unit = "C" if name == "temperature" else None
        value = value_mgdl
        if name == "creatinine":
            if self.rng.random() < 0.3:
                value, unit = value_mgdl * 88.4, "umol/L"
            else:
                unit = "mg/dL"
        elif name == "hemoglobin":
            if self.rng.random() < 0.2:
                value, unit = value_mgdl * 10.0, "g/L"
            else:
                unit = "g/dL"
        elif name == "sodium":
            unit = "mmol/L"
        elif name == "wbc":
            unit = "10^9/L"
        return {
            "lab_id": f"{self.id_prefix}L{self._lab_counter:08d}",
            "admission_id": admission_id,
            "lab_name": name,
            "value": round(float(value), 2),
            "unit": unit,
            "charted_time": charted,
            "_mgdl": float(value_mgdl),
        }

    def _stay_labs(self, admission_id, admit_time, los_days, frailty):
        rng = self.rng
        rows = []
        for name, (mean, sd, coef) in LAB_BASELINES.items():
            if rng.random() < 0.1:
                continue
            level = mean + coef * frailty * sd
            for _ in range(int(rng.integers(1, 4))):
                offset_hours = float(rng.uniform(0.0, max(los_days * 24.0 - 3.0, 0.0)))
                charted = (admit_time.ceil("h") + pd.Timedelta(hours=offset_hours)).floor("h")
                value = max(level + rng.normal(0.0, sd), 0.2 if name != "temperature" else 35.0)
                rows.append(self._lab_row(admission_id, name, value, charted))
        return rows

    def _follow_up_lab(self, admission_id, discharge_time, frailty, readmitted):
        mean, sd, coef = LAB_BASELINES["creatinine"]
        level = (mean + coef * frailty * sd) * (1.45 if readmitted else 1.0)
        charted = (discharge_time + pd.Timedelta(days=float(self.rng.uniform(1, 14)))).floor("h")
        value = max(level + self.rng.normal(0.0, sd), 0.2)
        return self._lab_row(admission_id, "creatinine", value, charted)

    def _add_chart_jitter(self, labs):
        labs = labs.drop(columns=["_mgdl"])
        minutes = np.random.randint(0, 60, size=len(labs))
        labs["charted_time"] = labs["charted_time"] + pd.to_timedelta(minutes, unit="min")
        return labs


def write_tables(tables, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        written[name] = str(path)
    return written
