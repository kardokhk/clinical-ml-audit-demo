"""Split, train, score and evaluate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .features import feature_columns


def grouped_split(adm: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(adm, groups=adm["patient_id"]))
    split = pd.Series("train", index=adm.index)
    split.iloc[test_idx] = "test"
    return pd.DataFrame({"admission_id": adm["admission_id"], "patient_id": adm["patient_id"],
                         "split": split.values})


def make_model(cfg: dict) -> Pipeline:
    m = cfg["model"]
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=m["max_iter"], learning_rate=m["learning_rate"],
            max_leaf_nodes=m["max_leaf_nodes"], random_state=cfg["seed"])),
    ])


def score(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    X = features[feature_columns()]
    return model.predict_proba(X)[:, 1]


def evaluate(pred: pd.DataFrame, threshold: float) -> dict:
    y, s = pred["label"].to_numpy(), pred["risk_score"].to_numpy()
    alert = s >= threshold
    tp = int(((alert == 1) & (y == 1)).sum())
    tn = int(((alert == 0) & (y == 0)).sum())
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "threshold": float(threshold),
        "sensitivity": tp / max(1, int((y == 1).sum())),
        "specificity": tn / max(1, int((y == 0).sum())),
        "n_test": int(len(y)),
        "n_events": int(y.sum()),
    }


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(model: Pipeline, out_dir: Path, threshold: float) -> None:
    path = out_dir / "model.joblib"
    joblib.dump(model, path)
    meta = {"features": feature_columns(), "threshold": threshold, "sha256": sha256(path)}
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))


def load_model(out_dir: Path) -> tuple[Pipeline, dict]:
    meta = json.loads((out_dir / "model_meta.json").read_text())
    path = out_dir / "model.joblib"
    if sha256(path) != meta["sha256"]:
        raise RuntimeError("model.joblib does not match the hash in model_meta.json")
    return joblib.load(path), meta
