import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from readmit.config import load_config  # noqa: E402
from readmit.synth import SyntheticEHRGenerator  # noqa: E402

T = pd.Timestamp


@pytest.fixture
def small_tables():
    """Two patients with hand-written stays and labs (canonical units)."""
    patients = pd.DataFrame({"patient_id": ["P1", "P2"], "birth_year": [1950, 1980], "sex": ["F", "M"]})
    admissions = pd.DataFrame({
        "admission_id": ["A1", "A2", "A3", "A4"],
        "patient_id": ["P1", "P1", "P1", "P2"],
        "admit_time": [T("2022-01-01 08:00"), T("2022-01-20 09:00"), T("2022-03-15 10:00"), T("2022-03-01 12:00")],
        "admission_type": ["emergency", "emergency", "elective", "emergency"],
        "ward": ["general_medicine", "cardiology", "surgery", "orthopaedics"],
    })
    discharges = pd.DataFrame({
        "admission_id": ["A1", "A2", "A3", "A4"],
        "discharge_time": [T("2022-01-05 12:00"), T("2022-01-25 12:00"), T("2022-03-17 12:00"), T("2022-03-04 12:00")],
        "discharge_disposition": ["home", "facility", "home", "home_with_services"],
    })
    labs = pd.DataFrame({
        "lab_id": ["L1", "L2", "L3", "L4", "L5"],
        "admission_id": ["A1", "A1", "A1", "A1", "A4"],
        "lab_name": ["creatinine", "creatinine", "temperature", "temperature", "sodium"],
        "value": [1.1, 1.4, 37.2, 38.4, 141.0],
        "unit": ["mg/dL", "mg/dL", "C", "C", "mmol/L"],
        "charted_time": [T("2022-01-01 10:00"), T("2022-01-04 10:00"), T("2022-01-02 10:00"),
                         T("2022-01-03 10:00"), T("2022-03-02 10:00")],
    })
    return {"patients": patients, "admissions": admissions, "discharges": discharges, "labs": labs}


@pytest.fixture
def synthetic_tables():
    return SyntheticEHRGenerator(150, "2021-01-01", "2023-12-31", seed=11).generate()


@pytest.fixture
def tmp_config(tmp_path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["synthetic"]["n_patients"] = 400
    cfg["synthetic"]["n_new_patients"] = 40
    cfg["model"]["n_estimators"] = 30
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return load_config(path)
