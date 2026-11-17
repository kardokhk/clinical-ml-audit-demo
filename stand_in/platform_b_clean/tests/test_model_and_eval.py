import json

import numpy as np
import pandas as pd
import pytest

from readmit import evaluate, pipeline
from readmit.inference import InferenceService
from readmit.model import ReadmissionModel


def test_compute_metrics_known_answer():
    preds = pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"],
        "readmit_30d": [0, 0, 1, 1],
        "y_prob": [0.1, 0.4, 0.35, 0.8],
    })
    m = evaluate.compute_metrics(preds, threshold=0.5)
    assert m["auroc"] == pytest.approx(0.75)
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["ppv"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert m["n_test"] == 4


def test_compute_metrics_needs_both_classes():
    preds = pd.DataFrame({"patient_id": ["a"], "readmit_30d": [0], "y_prob": [0.2]})
    with pytest.raises(ValueError):
        evaluate.compute_metrics(preds, 0.5)


@pytest.fixture
def trained(tmp_config):
    pipeline.run_generate(tmp_config)
    metrics = pipeline.run_training(tmp_config)
    return tmp_config, metrics


def test_end_to_end_outputs(trained):
    cfg, metrics = trained
    paths = pipeline.output_paths(cfg)
    for key in ("split", "predictions", "metrics", "features"):
        assert paths[key].is_file()
    saved = json.loads(paths["metrics"].read_text())
    assert saved == metrics
    assert 0.0 <= metrics["auroc"] <= 1.0
    preds = pd.read_csv(paths["predictions"])
    assert set(preds.columns) >= {"patient_id", "admission_id", "y_prob", "y_pred", "readmit_30d"}
    assert preds["y_prob"].between(0, 1).all()


def test_model_round_trip(trained):
    cfg, _ = trained
    out_dir = pipeline.output_paths(cfg)["dir"]
    features = pd.read_csv(out_dir / "features.csv")
    model = ReadmissionModel.load(out_dir)
    X = features.head(50)
    first = model.predict_proba(X)
    again = ReadmissionModel.load(out_dir).predict_proba(X)
    np.testing.assert_allclose(first, again)


def test_inference_scores_new_admissions(trained, tmp_path):
    cfg, _ = trained
    from readmit.config import resolve_path
    out = tmp_path / "scored.csv"
    scored = pipeline.run_predict(cfg, resolve_path(cfg, "new_admissions_csv"), out)
    assert out.is_file()
    assert scored["y_prob"].between(0, 1).all()
    service = InferenceService(pipeline.output_paths(cfg)["dir"])
    assert ((scored["y_prob"] >= service.threshold).astype(int) == scored["y_pred"]).all()
