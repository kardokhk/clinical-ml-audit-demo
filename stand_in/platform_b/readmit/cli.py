"""Command-line entry point: python -m readmit <command>."""

import argparse
import json
import logging
import sys

from .config import ConfigError, load_config
from .pipeline import run_generate, run_predict, run_training


def build_parser():
    parser = argparse.ArgumentParser(prog="readmit", description="30-day readmission risk platform")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="write the synthetic EHR extract and a new-admissions CSV")
    sub.add_parser("train", help="build cohort and features, train, predict the test set, write metrics")
    predict = sub.add_parser("predict", help="score a CSV of new admissions")
    predict.add_argument("--input", required=True, help="CSV with the inference input schema")
    predict.add_argument("--output", required=True, help="where to write the scored CSV")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.command == "generate":
        result = run_generate(cfg)
    elif args.command == "train":
        result = run_training(cfg)
    else:
        scored = run_predict(cfg, args.input, args.output)
        result = {"rows_scored": int(len(scored)), "flagged": int(scored["y_pred"].sum()), "output": args.output}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
