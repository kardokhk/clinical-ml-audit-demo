"""Configuration loading and validation."""

from pathlib import Path

import yaml

REQUIRED_KEYS = {
    "seed": int,
    "paths": dict,
    "synthetic": dict,
    "cohort": dict,
    "features": dict,
    "split": dict,
    "model": dict,
}

REQUIRED_SECTION_KEYS = {
    "paths": ["raw_dir", "output_dir", "new_admissions_csv"],
    "synthetic": ["n_patients", "n_new_patients", "start_date", "end_date"],
    "cohort": ["horizon_days"],
    "features": ["prior_admission_lookback_days"],
    "split": ["test_fraction"],
    "model": ["type", "decision_threshold"],
}


class ConfigError(ValueError):
    """Raised when the configuration file is missing or malformed."""


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config file {path} does not contain a mapping")
    validate_config(cfg)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def validate_config(cfg):
    for key, expected in REQUIRED_KEYS.items():
        if key not in cfg:
            raise ConfigError(f"missing config key: {key}")
        if not isinstance(cfg[key], expected):
            raise ConfigError(f"config key {key} must be of type {expected.__name__}")
    for section, keys in REQUIRED_SECTION_KEYS.items():
        for key in keys:
            if key not in cfg[section]:
                raise ConfigError(f"missing config key: {section}.{key}")
    fraction = cfg["split"]["test_fraction"]
    if not 0.05 <= float(fraction) <= 0.5:
        raise ConfigError("split.test_fraction must lie between 0.05 and 0.5")
    threshold = cfg["model"]["decision_threshold"]
    if not 0.0 < float(threshold) < 1.0:
        raise ConfigError("model.decision_threshold must lie strictly between 0 and 1")
    if int(cfg["cohort"]["horizon_days"]) <= 0:
        raise ConfigError("cohort.horizon_days must be positive")
    if cfg["model"]["type"] not in ("random_forest", "logistic_regression"):
        raise ConfigError("model.type must be random_forest or logistic_regression")


def resolve_path(cfg, key):
    """Return an absolute path for cfg['paths'][key], relative to the config file."""
    value = Path(cfg["paths"][key])
    if value.is_absolute():
        return value
    return Path(cfg.get("_root", ".")) / value
