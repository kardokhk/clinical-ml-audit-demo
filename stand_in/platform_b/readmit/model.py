"""Model definition, fitting and persistence."""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES, CATEGORY_LEVELS, FEATURE_COLUMNS, NUMERIC_FEATURES

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"


def build_pipeline(model_cfg, seed):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_FEATURES],
        handle_unknown="error",
        sparse_output=False,
    )
    preprocess = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )
    if model_cfg["type"] == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            max_depth=model_cfg.get("max_depth"),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    else:
        classifier = LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=2000,
            class_weight="balanced",
        )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


class ReadmissionModel:
    """A fitted pipeline plus the metadata needed to serve it."""

    def __init__(self, pipeline, card):
        self.pipeline = pipeline
        self.card = card

    @property
    def threshold(self):
        return float(self.card["decision_threshold"])

    @classmethod
    def fit(cls, X, y, cfg):
        if list(X.columns) != FEATURE_COLUMNS:
            raise ValueError("training matrix columns do not match FEATURE_COLUMNS")
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) != 2:
            raise ValueError("training labels must contain both classes")
        pipeline = build_pipeline(cfg["model"], cfg["seed"])
        pipeline.fit(X, y)
        card = {
            "model_type": cfg["model"]["type"],
            "decision_threshold": float(cfg["model"]["decision_threshold"]),
            "feature_columns": FEATURE_COLUMNS,
            "n_train": int(len(y)),
            "train_prevalence": float(y.mean()),
            "seed": int(cfg["seed"]),
            "sklearn_version": sklearn.__version__,
        }
        return cls(pipeline, card)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X[FEATURE_COLUMNS])[:, 1]

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out_dir / MODEL_FILE)
        with (out_dir / CARD_FILE).open("w", encoding="utf-8") as handle:
            json.dump(self.card, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, out_dir):
        out_dir = Path(out_dir)
        model_path, card_path = out_dir / MODEL_FILE, out_dir / CARD_FILE
        for path in (model_path, card_path):
            if not path.is_file():
                raise FileNotFoundError(f"model artefact not found: {path}; run the train command first")
        with card_path.open("r", encoding="utf-8") as handle:
            card = json.load(handle)
        if card.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("saved model was trained on a different feature set")
        return cls(joblib.load(model_path), card)
