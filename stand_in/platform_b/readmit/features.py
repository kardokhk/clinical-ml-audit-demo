"""Feature construction and the feature-frame contract shared by training and inference."""

import pandas as pd

from .schema import (
    ADMISSION_ID,
    CATEGORICAL_FEATURES,
    CATEGORY_LEVELS,
    FEATURE_COLUMNS,
    LAB_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_RANGES,
    PATIENT_ID,
    PREDICTION_TIME,
    REQUIRED_NON_NULL,
    TARGET,
)


class FeatureValidationError(ValueError):
    """Raised when a feature frame breaks the documented input contract."""


def prior_admission_counts(stays, lookback_days):
    """Number of admissions discharged before this admission began, within the lookback."""
    cols = [PATIENT_ID, ADMISSION_ID, "admit_time", PREDICTION_TIME]
    a = stays[cols]
    pairs = a.merge(a, on=PATIENT_ID, suffixes=("", "_other"))
    window_start = pairs["admit_time"] - pd.Timedelta(days=int(lookback_days))
    is_prior = (pairs["admit_time_other"] >= window_start) & (pairs[ADMISSION_ID + "_other"] != pairs[ADMISSION_ID])
    counts = pairs.loc[is_prior].groupby(ADMISSION_ID).size()
    return counts.reindex(a[ADMISSION_ID].to_numpy(), fill_value=0).astype(int)


def lab_features(labs, index):
    """Last (or max) value of each lab charted between admission and prediction time."""
    merged = labs.merge(index[[ADMISSION_ID, "admit_time", PREDICTION_TIME]], on=ADMISSION_ID, how="inner")
    in_window = merged["charted_time"] >= merged["admit_time"]
    merged = merged.loc[in_window].sort_values([ADMISSION_ID, "charted_time", "lab_id"])
    columns = {}
    for lab_name, (column, how) in LAB_FEATURES.items():
        subset = merged.loc[merged["lab_name"] == lab_name]
        columns[column] = subset.groupby(ADMISSION_ID)["value"].agg(how)
    out = pd.DataFrame(columns).reindex(index[ADMISSION_ID].to_numpy())
    out.index.name = ADMISSION_ID
    return out.reset_index()


def build_features(tables, stays, index, cfg):
    """Return one row per index admission: ids, prediction time and FEATURE_COLUMNS."""
    lookback = cfg["features"].get("lookback_days", 90)
    prior = prior_admission_counts(stays, lookback)
    frame = index.copy()
    frame["age_years"] = frame["admit_time"].dt.year - frame["birth_year"]
    frame["length_of_stay_days"] = (frame["discharge_time"] - frame["admit_time"]).dt.total_seconds() / 86400.0
    frame["prior_admissions_365d"] = frame[ADMISSION_ID].map(prior).astype(int)
    labs = lab_features(tables["labs"], frame)
    frame = frame.merge(labs, on=ADMISSION_ID, how="left", validate="1:1")
    keep = [PATIENT_ID, ADMISSION_ID, PREDICTION_TIME] + FEATURE_COLUMNS
    if TARGET in frame.columns:
        keep.append(TARGET)
    return frame[keep].reset_index(drop=True)


def validate_feature_frame(frame):
    """Check columns, types, missingness, ranges and category levels.

    Returns a copy restricted to FEATURE_COLUMNS in canonical order with numeric
    columns as float. Raises FeatureValidationError on any violation.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise FeatureValidationError(f"missing feature columns: {missing}")
    X = frame[FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        try:
            X[column] = pd.to_numeric(X[column], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise FeatureValidationError(f"column {column} is not numeric: {exc}") from exc
    nulls = [c for c in REQUIRED_NON_NULL if X[c].isna().any()]
    if nulls:
        raise FeatureValidationError(f"required columns contain missing values: {nulls}")
    for column, (low, high) in NUMERIC_RANGES.items():
        values = X[column]
        bad = values.notna() & ((values < low) | (values > high))
        if bad.any():
            X[column] = values.clip(lower=low, upper=high)
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(str)
        bad = ~X[column].isin(CATEGORY_LEVELS[column])
        if bad.any():
            raise FeatureValidationError(
                f"unexpected levels in {column}: {sorted(X.loc[bad, column].unique())[:5]}"
            )
    return X
