import pandas as pd
import pytest

from readmit import cohort, features
from readmit.features import FeatureValidationError


@pytest.fixture
def built(small_tables):
    stays = cohort.build_stays(small_tables)
    index = cohort.select_index_admissions(stays)
    cfg = {"features": {"prior_admission_lookback_days": 365}}
    return features.build_features(small_tables, stays, index, cfg).set_index("admission_id")


def test_prior_admissions_count_earlier_stays(built):
    assert built.loc["A3", "prior_admissions_365d"] == 2
    assert built.loc["A4", "prior_admissions_365d"] == 0


def test_prior_admissions_respect_lookback(small_tables):
    stays = cohort.build_stays(small_tables)
    counts = features.prior_admission_counts(stays, lookback_days=60)
    assert counts.loc["A3"] == 1


def test_lab_aggregations(built):
    assert built.loc["A1", "creatinine_last"] == pytest.approx(1.4)
    assert built.loc["A1", "temperature_max_c"] == pytest.approx(38.4)
    assert built.loc["A4", "sodium_last"] == pytest.approx(141.0)
    assert pd.isna(built.loc["A2", "creatinine_last"])


def test_age_and_length_of_stay(built):
    assert built.loc["A1", "age_years"] == 72
    assert built.loc["A1", "length_of_stay_days"] == pytest.approx(4 + 4 / 24)


def test_validate_rejects_missing_column(built):
    with pytest.raises(FeatureValidationError, match="missing feature columns"):
        features.validate_feature_frame(built.drop(columns=["ward"]))


def test_validate_rejects_unknown_level(built):
    frame = built.copy()
    frame.loc["A1", "discharge_disposition"] = "died"
    with pytest.raises(FeatureValidationError, match="unexpected levels"):
        features.validate_feature_frame(frame)


@pytest.mark.skip(reason="range limits pending sign-off from clinical informatics")
def test_validate_rejects_out_of_range(built):
    frame = built.copy()
    frame.loc["A1", "temperature_max_c"] = 101.3
    with pytest.raises(FeatureValidationError, match="temperature_max_c"):
        features.validate_feature_frame(frame)


def test_validate_returns_canonical_order(built):
    X = features.validate_feature_frame(built[list(reversed(built.columns))])
    assert list(X.columns) == features.FEATURE_COLUMNS
