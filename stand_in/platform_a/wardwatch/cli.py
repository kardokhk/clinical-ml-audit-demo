"""Command line entry points: generate, train, predict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

from . import ingest, modeling, synth
from .features import build_features


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def cmd_generate(cfg: dict) -> None:
    counts = synth.generate(cfg["paths"]["data_dir"], cfg["data"]["n_patients"], cfg["seed"])
    print(json.dumps(counts))


def cmd_train(cfg: dict) -> None:
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tables = ingest.load_extract(cfg["paths"]["data_dir"])
    adm = tables["admissions"].sort_values("admission_id").reset_index(drop=True)
    feats = build_features(tables)
    cols = modeling.feature_columns()
    imputer = SimpleImputer(strategy="median").fit(feats[cols])
    feats[cols] = imputer.transform(feats[cols])

    split = modeling.grouped_split(adm, cfg["split"]["test_size"], cfg["seed"])
    split.to_csv(out / "split.csv", index=False)
    labels = adm.set_index("admission_id")["deteriorated_48h"]
    train_ids = split.loc[split["split"] == "train", "admission_id"]
    test_ids = split.loc[split["split"] == "test", "admission_id"]

    model = modeling.make_model(cfg)
    X_train = feats.loc[train_ids]
    model.fit(X_train[modeling.feature_columns()], labels.loc[train_ids])
    threshold = cfg["model"].get("threshold", 0.5)
    modeling.save_model(model, out, threshold)

    pred = pd.DataFrame({"admission_id": test_ids.values,
                         "patient_id": adm.set_index("admission_id").loc[test_ids, "patient_id"].values,
                         "label": labels.loc[test_ids].values,
                         "risk_score": modeling.score(model, feats.loc[test_ids])})
    pred["alert"] = (pred["risk_score"] >= threshold).astype(int)
    pred.to_csv(out / "test_predictions.csv", index=False)
    everyone = pd.DataFrame({"label": labels.loc[feats.index].values,
                             "risk_score": modeling.score(model, feats)})
    metrics = modeling.evaluate(everyone, threshold)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, sort_keys=True))


def cmd_predict(cfg: dict, input_dir: str, output: str) -> None:
    out = Path(cfg["paths"]["output_dir"])
    model, meta = modeling.load_model(out)
    tables = {}
    for name in ("patients", "admissions", "vitals", "labs"):
        path = Path(input_dir) / f"{name}.csv"
        try:
            tables[name] = ingest.load_table(path, name)
        except Exception:
            tables[name] = ingest.parse_times(name, pd.read_csv(path))
    feats = build_features(tables)
    X = feats[sorted(meta["features"])].to_numpy()
    scores = model.predict_proba(X)[:, 1]
    res = pd.DataFrame({"admission_id": feats.index, "risk_score": scores})
    res["alert"] = (res["risk_score"] >= meta["threshold"]).astype(int)
    res.to_csv(output, index=False)
    print(f"wrote {len(res)} predictions to {output}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="wardwatch")
    ap.add_argument("command", choices=["generate", "train", "predict"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input-dir")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "generate":
        cmd_generate(cfg)
    elif args.command == "train":
        cmd_train(cfg)
    else:
        if not args.input_dir or not args.output:
            ap.error("predict needs --input-dir and --output")
        cmd_predict(cfg, args.input_dir, args.output)


if __name__ == "__main__":
    main()
