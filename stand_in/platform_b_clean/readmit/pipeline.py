"""Orchestration of data generation, training, evaluation and scoring."""

import logging
from pathlib import Path

from . import cohort, evaluate, features, ingest, split, synth
from .config import resolve_path
from .inference import InferenceService
from .model import ReadmissionModel
from .schema import ADMISSION_ID, FEATURE_COLUMNS, PATIENT_ID, PREDICTION_COLUMNS, PREDICTION_TIME, TARGET

log = logging.getLogger(__name__)

SPLIT_FILE = "split_assignment.csv"
PREDICTIONS_FILE = "predictions_test.csv"
METRICS_FILE = "metrics.json"
FEATURES_FILE = "features.csv"


def output_paths(cfg):
    out_dir = resolve_path(cfg, "output_dir")
    return {
        "dir": out_dir,
        "split": out_dir / SPLIT_FILE,
        "predictions": out_dir / PREDICTIONS_FILE,
        "metrics": out_dir / METRICS_FILE,
        "features": out_dir / FEATURES_FILE,
    }


def _feature_frame(tables, cfg, with_labels):
    stays = cohort.build_stays(tables)
    index = cohort.select_index_admissions(stays)
    if with_labels:
        horizon = cfg["cohort"]["horizon_days"]
        index = cohort.label_readmissions(index, stays, horizon)
        index = cohort.exclude_incomplete_followup(index, cohort.extract_end_time(tables), horizon)
    frame = features.build_features(tables, stays, index, cfg)
    features.validate_feature_frame(frame)
    return frame


def run_generate(cfg):
    """Write the synthetic training extract and a CSV of new admissions for scoring."""
    syn = cfg["synthetic"]
    raw_dir = resolve_path(cfg, "raw_dir")
    generator = synth.SyntheticEHRGenerator(syn["n_patients"], syn["start_date"], syn["end_date"], cfg["seed"])
    written = synth.write_tables(generator.generate(), raw_dir)

    new_gen = synth.SyntheticEHRGenerator(
        syn["n_new_patients"], syn["start_date"], syn["end_date"], int(cfg["seed"]) + 1, id_prefix="N"
    )
    new_tables = new_gen.generate()
    staging = raw_dir / "_new_extract"
    synth.write_tables(new_tables, staging)
    new_frame = _feature_frame(ingest.load_tables(staging), cfg, with_labels=False)
    latest = new_frame.sort_values(PREDICTION_TIME).groupby(PATIENT_ID, sort=True).tail(1)
    latest = latest.sort_values([PATIENT_ID, ADMISSION_ID])
    new_path = resolve_path(cfg, "new_admissions_csv")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    latest[[PATIENT_ID, ADMISSION_ID] + FEATURE_COLUMNS].to_csv(new_path, index=False)
    for path in staging.iterdir():
        path.unlink()
    staging.rmdir()
    written["new_admissions"] = str(new_path)
    return written


def run_training(cfg):
    """Build the cohort, split by patient, fit, save, predict the test set and score it."""
    paths = output_paths(cfg)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    tables = ingest.load_tables(resolve_path(cfg, "raw_dir"))
    frame = _feature_frame(tables, cfg, with_labels=True)
    frame.to_csv(paths["features"], index=False)

    assignment = split.assign_split(frame, cfg["split"]["test_fraction"], cfg["seed"])
    assignment.to_csv(paths["split"], index=False)
    frame = frame.merge(assignment[[ADMISSION_ID, "split"]], on=ADMISSION_ID, how="left", validate="1:1")
    train = frame.loc[frame["split"] == "train"]
    log.info("training on %d admissions from %d patients", len(train), train[PATIENT_ID].nunique())

    model = ReadmissionModel.fit(train[FEATURE_COLUMNS], train[TARGET], cfg)
    model.save(paths["dir"])

    scored = frame[[PATIENT_ID, ADMISSION_ID, PREDICTION_TIME, TARGET, "split"]].copy()
    scored["y_prob"] = model.predict_proba(features.validate_feature_frame(frame)).round(6)
    scored["y_pred"] = (scored["y_prob"] >= model.threshold).astype(int)
    test_predictions = scored.loc[scored["split"] == "test", PREDICTION_COLUMNS]
    test_predictions.to_csv(paths["predictions"], index=False)

    metrics = evaluate.metrics_from_predictions(paths["predictions"], model.threshold)
    evaluate.write_metrics(metrics, paths["metrics"])
    return metrics


def run_predict(cfg, input_path, output_path):
    service = InferenceService(output_paths(cfg)["dir"])
    return service.predict_csv(Path(input_path), Path(output_path))
