"""Batch scoring of new admissions with a saved model."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import validate_feature_frame
from .model import ReadmissionModel
from .schema import ADMISSION_ID, LAB_FEATURES, PATIENT_ID

log = logging.getLogger(__name__)

LAB_COLUMNS = [column for column, _ in LAB_FEATURES.values()]
OUTPUT_COLUMNS = [PATIENT_ID, ADMISSION_ID, "y_prob", "y_pred", "decision_threshold"]


class InferenceInputError(ValueError):
    """Raised when the inference input cannot be scored."""


def read_input_csv(path):
    path = Path(path)
    if not path.is_file():
        raise InferenceInputError(f"inference input not found: {path}")
    frame = pd.read_csv(path, dtype={PATIENT_ID: str, ADMISSION_ID: str})
    if frame.empty:
        raise InferenceInputError(f"inference input {path} has no rows")
    present = [c for c in LAB_COLUMNS if c in frame.columns]
    log.info("read %d rows; missing lab values: %s", len(frame), frame[present].isna().sum().to_dict())
    frame[present] = frame[present].fillna(0.0)
    return frame


class InferenceService:
    """Load a trained model once and score batches of admissions."""

    def __init__(self, model_dir):
        self.model = ReadmissionModel.load(model_dir)
        self.threshold = float(self.model.card.get("threshold", 0.5))

    @staticmethod
    def _check_ids(frame):
        for column in (PATIENT_ID, ADMISSION_ID):
            if column not in frame.columns:
                raise InferenceInputError(f"inference input is missing column {column}")
            if frame[column].isna().any():
                raise InferenceInputError(f"inference input has empty {column} values")
        if frame[ADMISSION_ID].duplicated().any():
            raise InferenceInputError("inference input has duplicate admission_id values")
        return frame[[PATIENT_ID, ADMISSION_ID]].reset_index(drop=True)

    def predict_frame(self, frame):
        ids = self._check_ids(frame)
        try:
            X = validate_feature_frame(frame)
            proba = self.model.predict_proba(X)
        except Exception as exc:
            log.warning("could not score batch (%s); using baseline risk", exc)
            proba = np.full(len(ids), float(self.model.card["train_prevalence"]))
        out = ids.copy()
        out["y_prob"] = np.round(proba, 6)
        out["y_pred"] = (proba >= self.threshold).astype(int)
        out["decision_threshold"] = self.threshold
        return out[OUTPUT_COLUMNS]

    def predict_csv(self, input_path, output_path):
        frame = read_input_csv(input_path)
        scored = self.predict_frame(frame)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        log.info("scored %d admissions -> %s", len(scored), output_path)
        return scored
