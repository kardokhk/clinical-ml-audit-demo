import pandas as pd

from wardwatch.features import build_features, feature_columns


def _tables():
    adm = pd.DataFrame({"admission_id": ["A1"], "patient_id": ["P1"],
                        "admit_time": pd.to_datetime(["2025-01-01 00:00"]),
                        "prediction_time": pd.to_datetime(["2025-01-01 06:00"])})
    pat = pd.DataFrame({"patient_id": ["P1"], "birth_year": [1960], "sex": ["M"]})
    vit = pd.DataFrame({"admission_id": ["A1"] * 3,
                        "charttime": pd.to_datetime(["2025-01-01 01:00", "2025-01-01 05:00", "2025-01-01 09:00"]),
                        "heart_rate": [80, 90, 150], "resp_rate": [16, 18, 30], "sbp": [120, 110, 70],
                        "temp_c": [37.0, 37.5, 39.5], "spo2": [97, 95, 85]})
    lab = pd.DataFrame({"admission_id": ["A1", "A1"],
                        "charttime": pd.to_datetime(["2025-01-01 01:00", "2025-01-01 20:00"]),
                        "lactate": [1.0, 6.0], "creatinine": [80, 200], "wbc": [8, 20]})
    return {"admissions": adm, "patients": pat, "vitals": vit, "labs": lab}


def test_values_after_prediction_time_are_ignored():
    f = build_features(_tables()).loc["A1"]
    assert f["heart_rate_max"] == 90
    assert f["lactate_last"] == 1.0
    assert f["n_vitals"] == 2


def test_feature_order_is_fixed():
    f = build_features(_tables())
    assert list(f[feature_columns()].columns) == feature_columns()
    assert f.loc["A1", "age"] == 65
