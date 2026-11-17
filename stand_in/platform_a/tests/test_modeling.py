import pandas as pd

from wardwatch import modeling


def test_split_keeps_patients_together():
    adm = pd.DataFrame({"admission_id": [f"A{i}" for i in range(40)],
                        "patient_id": [f"P{i // 2}" for i in range(40)]})
    s = modeling.grouped_split(adm, 0.25, 1)
    assert set(s["split"]) == {"train", "test"}


def test_evaluate_counts():
    pred = pd.DataFrame({"label": [1, 1, 0, 0], "risk_score": [0.9, 0.2, 0.4, 0.1]})
    m = modeling.evaluate(pred, 0.3)
    print(m)
