"""Load and validate the ward extract."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

VITAL_RANGES = {
    "heart_rate": (20.0, 250.0),
    "resp_rate": (4.0, 60.0),
    "sbp": (40.0, 300.0),
    "temp_c": (30.0, 45.0),
    "spo2": (50.0, 100.0),
}
LAB_RANGES = {
    "lactate": (0.1, 30.0),
    "creatinine": (10.0, 2000.0),
    "wbc": (0.1, 200.0),
}
REQUIRED = {
    "admissions": ["admission_id", "patient_id", "admit_time", "prediction_time"],
    "vitals": ["admission_id", "charttime", *VITAL_RANGES],
    "labs": ["admission_id", "charttime", *LAB_RANGES],
    "patients": ["patient_id", "birth_year", "sex"],
}
TIME_COLS = {
    "admissions": ["admit_time", "prediction_time", "discharge_time", "icu_transfer_time"],
    "vitals": ["charttime"],
    "labs": ["charttime"],
    "patients": [],
}


class ValidationError(ValueError):
    """Raised when an input table breaks the documented schema."""


def check_columns(name: str, df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValidationError(f"{name}: missing required columns {missing}")


def check_ranges(name: str, df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> None:
    for col, (lo, hi) in ranges.items():
        values = pd.to_numeric(df[col], errors="raise")
        bad = values.notna() & ((values < lo) | (values > hi))
        if bad.any():
            raise ValidationError(
                f"{name}.{col}: {int(bad.sum())} value(s) outside [{lo}, {hi}]")


def parse_times(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TIME_COLS[name]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="raise")
    return df


def load_table(path: str | Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    check_columns(name, df)
    df = parse_times(name, df)
    if name == "vitals":
        check_ranges(name, df, VITAL_RANGES)
    if name == "labs":
        check_ranges(name, df, LAB_RANGES)
    if df.empty:
        raise ValidationError(f"{name}: table is empty")
    return df


def load_extract(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    return {name: load_table(d / f"{name}.csv", name)
            for name in ("patients", "admissions", "vitals", "labs")}
