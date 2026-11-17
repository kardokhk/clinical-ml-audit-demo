"""Index admissions and the 30-day unplanned readmission label.

The prediction time for every index admission is its discharge time.
"""

import logging

import pandas as pd

from .schema import ADMISSION_ID, PATIENT_ID, PREDICTION_TIME, TARGET

log = logging.getLogger(__name__)


def build_stays(tables):
    """One row per admission with patient attributes and discharge details."""
    stays = tables["admissions"].merge(tables["discharges"], on=ADMISSION_ID, how="inner", validate="1:1")
    stays = stays.merge(tables["patients"], on=PATIENT_ID, how="left", validate="m:1")
    stays[PREDICTION_TIME] = stays["discharge_time"]
    return stays.sort_values([PATIENT_ID, "admit_time", ADMISSION_ID]).reset_index(drop=True)


def select_index_admissions(stays):
    """Admissions eligible for prediction: the patient left hospital alive."""
    alive = stays["discharge_disposition"] != "died"
    log.info("excluded %d in-hospital deaths", int((~alive).sum()))
    return stays.loc[alive].reset_index(drop=True)


def label_readmissions(index, stays, horizon_days):
    """Label = any emergency admission starting within horizon_days after discharge."""
    horizon = pd.Timedelta(days=int(horizon_days))
    later = stays.loc[stays["admission_type"] == "emergency", [PATIENT_ID, ADMISSION_ID, "admit_time"]]
    pairs = index[[PATIENT_ID, ADMISSION_ID, PREDICTION_TIME]].merge(
        later, on=PATIENT_ID, suffixes=("", "_next")
    )
    gap = pairs["admit_time"] - pairs[PREDICTION_TIME]
    hit = (pairs[ADMISSION_ID + "_next"] != pairs[ADMISSION_ID]) & (gap > pd.Timedelta(0)) & (gap <= horizon)
    readmitted = set(pairs.loc[hit, ADMISSION_ID])
    labelled = index.copy()
    labelled[TARGET] = labelled[ADMISSION_ID].isin(readmitted).astype(int)
    return labelled


def exclude_incomplete_followup(labelled, extract_end, horizon_days):
    """Drop index admissions whose follow-up window runs past the end of the extract."""
    window_end = labelled[PREDICTION_TIME] + pd.Timedelta(days=int(horizon_days))
    complete = window_end <= pd.Timestamp(extract_end)
    log.info("excluded %d admissions with incomplete follow-up", int((~complete).sum()))
    return labelled.loc[complete].reset_index(drop=True)


def extract_end_time(tables):
    return max(tables["admissions"]["admit_time"].max(), tables["discharges"]["discharge_time"].max())
