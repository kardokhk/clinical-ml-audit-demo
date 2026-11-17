"""Train/test assignment."""

import numpy as np

from .schema import ADMISSION_ID, PATIENT_ID


def assign_split(frame, test_fraction, seed):
    """Assign every admission of a patient to the same split.

    Returns a frame with patient_id, admission_id and split ("train" or "test"),
    sorted by patient_id then admission_id.
    """
    if frame.empty:
        raise ValueError("cannot split an empty frame")
    units = np.sort(frame[PATIENT_ID].unique())
    n_test = int(round(float(test_fraction) * len(units)))
    if n_test < 1 or n_test >= len(units):
        raise ValueError(f"test_fraction {test_fraction} leaves an empty split for {len(units)} units")
    rng = np.random.default_rng(int(seed))
    test_units = set(rng.choice(units, size=n_test, replace=False).tolist())
    out = frame[[PATIENT_ID, ADMISSION_ID]].copy()
    out["split"] = np.where(out[PATIENT_ID].isin(test_units), "test", "train")
    return out.sort_values([PATIENT_ID, ADMISSION_ID]).reset_index(drop=True)
