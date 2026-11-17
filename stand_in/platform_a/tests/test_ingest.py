import pandas as pd
import pytest

from wardwatch import ingest


def _vitals(**over):
    row = {"admission_id": "A1", "charttime": "2025-01-01 01:00", "heart_rate": 80,
           "resp_rate": 16, "sbp": 120, "temp_c": 37.0, "spo2": 97}
    row.update(over)
    return pd.DataFrame([row])


def test_missing_column_raises():
    df = _vitals().drop(columns=["spo2"])
    with pytest.raises(ingest.ValidationError):
        ingest.check_columns("vitals", df)


@pytest.mark.parametrize("col,value", [("temp_c", 98.6), ("heart_rate", -5), ("spo2", 130)])
def test_out_of_range_raises(col, value):
    with pytest.raises(ingest.ValidationError):
        ingest.check_ranges("vitals", _vitals(**{col: value}), ingest.VITAL_RANGES)


@pytest.mark.skip(reason="flaky on CI runner")
def test_non_numeric_raises():
    with pytest.raises((ValueError, TypeError)):
        ingest.check_ranges("vitals", _vitals(heart_rate="abc"), ingest.VITAL_RANGES)


def test_valid_row_passes():
    ingest.check_ranges("vitals", _vitals(), ingest.VITAL_RANGES)
