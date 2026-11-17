"""Load and validate the raw EHR extract."""

import logging
from pathlib import Path

import pandas as pd

from .schema import (
    ADMISSION_TYPES,
    CANONICAL_UNITS,
    DISPOSITIONS,
    SEX_LEVELS,
    TABLE_COLUMNS,
    TIMESTAMP_COLUMNS,
    UNIT_CONVERSIONS,
    WARDS,
)

log = logging.getLogger(__name__)


class IngestError(ValueError):
    """Raised when an input table fails validation."""


def read_table(raw_dir, name):
    path = Path(raw_dir) / f"{name}.csv"
    if not path.is_file():
        raise IngestError(f"missing input table: {path}")
    frame = pd.read_csv(path, dtype={"patient_id": str, "admission_id": str, "lab_id": str})
    missing = [c for c in TABLE_COLUMNS[name] if c not in frame.columns]
    if missing:
        raise IngestError(f"{name}.csv is missing columns: {missing}")
    frame = frame[TABLE_COLUMNS[name]].copy()
    for column in TIMESTAMP_COLUMNS[name]:
        try:
            frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d %H:%M:%S", errors="raise")
        except (ValueError, TypeError) as exc:
            raise IngestError(f"{name}.{column} contains unparseable timestamps: {exc}") from exc
    return frame


def _check_unique(frame, column, table):
    duplicated = frame[column].duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, column].head(3).tolist()
        raise IngestError(f"{table}.{column} has duplicate values, e.g. {examples}")


def _check_levels(frame, column, allowed, table):
    bad = ~frame[column].isin(allowed)
    if bad.any():
        values = sorted(frame.loc[bad, column].astype(str).unique())[:5]
        raise IngestError(f"{table}.{column} has unexpected values: {values}")


def _check_not_null(frame, table):
    nulls = frame.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        raise IngestError(f"{table} has missing values in: {nulls.to_dict()}")


def normalise_lab_units(labs):
    """Convert every lab value to its canonical unit; reject unknown units."""
    labs = labs.copy()
    for (lab_name, unit), factor in UNIT_CONVERSIONS.items():
        mask = (labs["lab_name"] == lab_name) & (labs["unit"] == unit)
        labs.loc[mask, "value"] = labs.loc[mask, "value"] * factor
        labs.loc[mask, "unit"] = CANONICAL_UNITS[lab_name]
    expected = labs["lab_name"].map(CANONICAL_UNITS)
    bad = labs["unit"] != expected
    if bad.any():
        pairs = labs.loc[bad, ["lab_name", "unit"]].drop_duplicates().head(5).values.tolist()
        raise IngestError(f"labs has unsupported units: {pairs}")
    return labs


def validate_tables(tables):
    patients, admissions = tables["patients"], tables["admissions"]
    discharges, labs = tables["discharges"], tables["labs"]

    for name, frame in tables.items():
        _check_not_null(frame, name)
    _check_unique(patients, "patient_id", "patients")
    _check_unique(admissions, "admission_id", "admissions")
    _check_unique(discharges, "admission_id", "discharges")
    _check_unique(labs, "lab_id", "labs")

    _check_levels(patients, "sex", SEX_LEVELS, "patients")
    _check_levels(admissions, "admission_type", ADMISSION_TYPES, "admissions")
    _check_levels(admissions, "ward", WARDS, "admissions")
    _check_levels(discharges, "discharge_disposition", DISPOSITIONS, "discharges")
    _check_levels(labs, "lab_name", list(CANONICAL_UNITS), "labs")

    if not admissions["patient_id"].isin(patients["patient_id"]).all():
        raise IngestError("admissions reference unknown patient_id values")
    if set(discharges["admission_id"]) != set(admissions["admission_id"]):
        raise IngestError("discharges and admissions do not cover the same admission_id values")
    if not labs["admission_id"].isin(admissions["admission_id"]).all():
        raise IngestError("labs reference unknown admission_id values")

    stays = admissions.merge(discharges, on="admission_id", how="inner")
    if (stays["discharge_time"] < stays["admit_time"]).any():
        raise IngestError("discharge_time precedes admit_time for some admissions")
    if (pd.to_numeric(labs["value"], errors="coerce").isna()).any():
        raise IngestError("labs.value contains non-numeric entries")
    if (labs["value"] < 0).any():
        raise IngestError("labs.value contains negative values")


def load_tables(raw_dir):
    tables = {name: read_table(raw_dir, name) for name in TABLE_COLUMNS}
    validate_tables(tables)
    tables["labs"] = normalise_lab_units(tables["labs"])
    log.info(
        "loaded %d patients, %d admissions, %d lab results",
        len(tables["patients"]), len(tables["admissions"]), len(tables["labs"]),
    )
    return tables
