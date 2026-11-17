"""Column names, feature definitions and validation limits shared by every module."""

PATIENT_ID = "patient_id"
ADMISSION_ID = "admission_id"
TARGET = "readmit_30d"
PREDICTION_TIME = "prediction_time"

TABLE_COLUMNS = {
    "patients": ["patient_id", "birth_year", "sex"],
    "admissions": ["admission_id", "patient_id", "admit_time", "admission_type", "ward"],
    "discharges": ["admission_id", "discharge_time", "discharge_disposition"],
    "labs": ["lab_id", "admission_id", "lab_name", "value", "unit", "charted_time"],
}

TIMESTAMP_COLUMNS = {
    "patients": [],
    "admissions": ["admit_time"],
    "discharges": ["discharge_time"],
    "labs": ["charted_time"],
}

SEX_LEVELS = ["F", "M"]
ADMISSION_TYPES = ["elective", "emergency"]
WARDS = ["cardiology", "general_medicine", "orthopaedics", "surgery"]
DISPOSITIONS = ["died", "facility", "home", "home_with_services"]

# lab_name -> (feature column, aggregation over the stay)
LAB_FEATURES = {
    "creatinine": ("creatinine_last", "last"),
    "hemoglobin": ("hemoglobin_last", "last"),
    "sodium": ("sodium_last", "last"),
    "wbc": ("wbc_last", "last"),
    "temperature": ("temperature_max_c", "max"),
}

CANONICAL_UNITS = {
    "creatinine": "mg/dL",
    "hemoglobin": "g/dL",
    "sodium": "mmol/L",
    "wbc": "10^9/L",
    "temperature": "C",
}

# (lab_name, source unit) -> multiplicative factor to the canonical unit
UNIT_CONVERSIONS = {
    ("creatinine", "umol/L"): 1.0 / 88.4,
    ("hemoglobin", "g/L"): 0.1,
}

NUMERIC_FEATURES = [
    "age_years",
    "length_of_stay_days",
    "prior_admissions_365d",
    "creatinine_last",
    "hemoglobin_last",
    "sodium_last",
    "wbc_last",
    "temperature_max_c",
]
CATEGORICAL_FEATURES = ["sex", "admission_type", "ward", "discharge_disposition"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that must be present and non-missing in every feature row.
REQUIRED_NON_NULL = ["age_years", "length_of_stay_days", "prior_admissions_365d"] + CATEGORICAL_FEATURES

NUMERIC_RANGES = {
    "age_years": (18, 110),
    "length_of_stay_days": (0, 365),
    "prior_admissions_365d": (0, 60),
    "creatinine_last": (0.1, 20.0),
    "hemoglobin_last": (3.0, 25.0),
    "sodium_last": (100.0, 180.0),
    "wbc_last": (0.1, 200.0),
    "temperature_max_c": (30.0, 45.0),
}

# Levels the model accepts. In-hospital deaths are removed from the cohort,
# so "died" is not a valid disposition at prediction time.
CATEGORY_LEVELS = {
    "sex": SEX_LEVELS,
    "admission_type": ADMISSION_TYPES,
    "ward": WARDS,
    "discharge_disposition": ["facility", "home", "home_with_services"],
}

PREDICTION_COLUMNS = [PATIENT_ID, ADMISSION_ID, PREDICTION_TIME, TARGET, "y_prob", "y_pred"]
