"""Performance metrics for saved predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .schema import PATIENT_ID, PREDICTION_COLUMNS, TARGET


def _safe_ratio(num, den):
    return float(num) / float(den) if den else float("nan")


def compute_metrics(predictions, threshold):
    """Discrimination, calibration and threshold metrics for one prediction frame."""
    y = predictions[TARGET].to_numpy().astype(int)
    p = predictions["y_prob"].to_numpy().astype(float)
    if len(np.unique(y)) != 2:
        raise ValueError("metrics need both outcome classes in the evaluation set")
    flagged = p >= float(threshold)
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "ppv": _safe_ratio(tp, tp + fp),
        "n_test": int(len(y)),
        "n_test_patients": int(predictions[PATIENT_ID].nunique()),
        "prevalence_test": float(y.mean()),
    }


def metrics_from_predictions(path, threshold):
    predictions = pd.read_csv(path, dtype={PATIENT_ID: str})
    missing = [c for c in PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions file {path} is missing columns: {missing}")
    return compute_metrics(predictions, threshold)


def write_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
