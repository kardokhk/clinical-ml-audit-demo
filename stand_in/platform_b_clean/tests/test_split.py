import pandas as pd
import pytest

from readmit.split import assign_split


@pytest.fixture
def frame():
    rows = [(f"P{p:03d}", f"A{p:03d}{k}") for p in range(120) for k in range(1 + p % 3)]
    return pd.DataFrame(rows, columns=["patient_id", "admission_id"])


def test_split_is_deterministic(frame):
    a = assign_split(frame, 0.25, seed=3)
    b = assign_split(frame.sample(frac=1.0, random_state=1), 0.25, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_every_admission_assigned_once(frame):
    out = assign_split(frame, 0.25, seed=3)
    assert len(out) == len(frame)
    assert set(out["split"]) == {"train", "test"}
    share = (out["split"] == "test").mean()
    assert 0.15 < share < 0.35


def test_no_patient_in_both_splits(frame):
    out = assign_split(frame, 0.25, seed=3)
    per_patient = out.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all()


def test_rejects_degenerate_fraction(frame):
    with pytest.raises(ValueError):
        assign_split(frame.head(2), 0.25, seed=3)
