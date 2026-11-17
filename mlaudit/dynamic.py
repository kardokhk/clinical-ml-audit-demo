"""Checks that run the platform's own commands in disposable copies and inspect what it writes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .findings import Finding
from .runner import RunResult, make_sandbox, run

WARNING_RE = re.compile(r"(\w*Warning): (.+)")


# ---------------------------------------------------------------- helpers

def file_hash(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def dir_hash(d: Path) -> str | None:
    if not d.exists():
        return None
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def read_split(T: dict, sb: Path) -> pd.DataFrame | None:
    s = T["outputs"].get("split")
    if not s:
        return None
    if "path" in s:
        p = sb / s["path"]
        if not p.exists():
            return None
        df = pd.read_csv(p, dtype=str)
        df = df.rename(columns={s["id_col"]: "id", s["set_col"]: "set"})
        if s.get("group_col") and s["group_col"] in df.columns:
            df = df.rename(columns={s["group_col"]: "group"})
        df["set"] = df["set"].map({str(s.get("train_value", "train")): "train", str(s.get("test_value", "test")): "test"}).fillna(df["set"])
        return df
    parts = []
    for name in ("train", "test"):
        p = sb / s[f"{name}_path"]
        if not p.exists():
            return None
        d = pd.read_csv(p, dtype=str)
        cols = {s["id_col"]: "id"}
        if s.get("group_col") and s["group_col"] in d.columns:
            cols[s["group_col"]] = "group"
        d = d.rename(columns=cols)[list(cols.values())]
        d["set"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def read_predictions(T: dict, sb: Path) -> pd.DataFrame | None:
    s = T["outputs"]["predictions"]
    p = sb / s["path"]
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={s["id_col"]: str})
    ren = {s["id_col"]: "id", s["label_col"]: "label", s["score_col"]: "score"}
    if s.get("alert_col"):
        ren[s["alert_col"]] = "alert"
    if s.get("group_col"):
        ren[s["group_col"]] = "group"
    return df.rename(columns=ren)


def read_metrics(T: dict, sb: Path) -> dict | None:
    s = T["outputs"].get("metrics")
    if not s:
        return None
    p = sb / s["path"]
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    flat = {}

    def walk(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, key)
            else:
                flat[key] = v
    walk(raw)
    return {std: flat.get(theirs) for std, theirs in s["map"].items()}


def compute_metrics(pred: pd.DataFrame, threshold: float | None) -> dict:
    y = pred["label"].astype(int).to_numpy()
    s = pred["score"].astype(float).to_numpy()
    out = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "brier": float(brier_score_loss(y, s)), "n": int(len(y)), "events": int(y.sum())}
    if "alert" in pred.columns:
        a = pred["alert"].astype(int).to_numpy()
        out["sensitivity_from_alerts"] = float(((a == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
        out["specificity_from_alerts"] = float(((a == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    if threshold is not None:
        a = (s >= threshold).astype(int)
        out["sensitivity_at_threshold"] = float(((a == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
        out["specificity_at_threshold"] = float(((a == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
    return out


def paired_auroc_diff(a: pd.DataFrame, b: pd.DataFrame, groups: pd.Series | None, n_boot: int = 2000,
                      seed: int = 20261117) -> dict:
    """AUROC(a) - AUROC(b) on common ids, with a cluster bootstrap over groups."""
    m = a[["id", "label", "score"]].merge(b[["id", "score"]], on="id", suffixes=("_a", "_b"))
    y = m["label"].astype(int).to_numpy()
    sa, sb_ = m["score_a"].to_numpy(float), m["score_b"].to_numpy(float)
    point = roc_auc_score(y, sa) - roc_auc_score(y, sb_)
    ids = m["id"].to_numpy()
    if groups is not None:
        g = groups.reindex(ids).to_numpy()
        g = np.where(pd.isna(g), ids, g)
    else:
        g = ids
    uniq = np.unique(g)
    idx_by_g = {k: np.flatnonzero(g == k) for k in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb_[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_common": int(len(m)), "auroc_a": float(roc_auc_score(y, sa)), "auroc_b": float(roc_auc_score(y, sb_)),
            "diff": float(point), "ci95": [float(lo), float(hi)], "n_boot": len(diffs), "n_groups": int(len(uniq))}


def warnings_in(results: list[RunResult]) -> list[Finding]:
    seen: dict[tuple, list[str]] = {}
    for r in results:
        for line in (r.stderr or "").splitlines():
            m = WARNING_RE.search(line)
            if m:
                key = (m.group(1), re.sub(r"\d+", "N", m.group(2).strip())[:160])
                seen.setdefault(key, []).append(r.name)
    out = []
    for (wtype, msg), where in sorted(seen.items()):
        steps = sorted(set(where))
        if "feature names" in msg.lower() or "feature_names" in msg.lower():
            cat, sev, title = "execution_integrity", "High", "Model receives features without their names at inference"
            rec = "Pass a DataFrame with the training column order; check names and order before scoring."
        elif wtype in {"FutureWarning", "DeprecationWarning"}:
            cat, sev, title = "dependency", "Low", f"{wtype} raised by a dependency"
            rec = "Pin versions and update the call before the behaviour changes."
        elif wtype in {"ConvergenceWarning"}:
            cat, sev, title = "execution_integrity", "Medium", "Model fitting did not converge"
            rec = "Scale inputs or raise the iteration limit; confirm validated metrics came from a converged fit."
        else:
            cat, sev, title = "integration", "Low", f"{wtype} raised during the run"
            rec = "Review whether the warning signals wrong behaviour; turn expected cases into explicit checks."
        out.append(Finding(check="runtime_warning", category=cat, severity=sev, title=title,
                           detail=f"`{wtype}: {msg}` during step(s): {', '.join(steps)}.",
                           recommendation=rec, kind="dynamic", evidence={"steps": steps}))
    return out


# ---------------------------------------------------------------- end-to-end run

def run_pipeline(T: dict, sb: Path, python: str, hashseed: str, label: str, steps=("generate", "pipeline")) -> list[RunResult]:
    res = []
    for step in steps:
        tpl = T["commands"].get(step)
        if not tpl:
            continue
        r = run(f"{label}:{step}", tpl, sb, python, env_extra={"PYTHONHASHSEED": hashseed},
                timeout=T.get("timeout", 900))
        res.append(r)
        if not r.ok:
            break
    return res


def check_run_failures(results: list[RunResult]) -> list[Finding]:
    out = []
    for r in results:
        if not r.ok:
            out.append(Finding(
                check="command_failed", category="integration", severity="Critical",
                title=f"Documented command fails: {r.name}",
                detail=f"`{r.command}` exited with code {r.returncode}. Last lines: " + " | ".join(r.stderr.strip().splitlines()[-3:]),
                recommendation="Fix the command or the documentation so the documented run completes.",
                kind="dynamic", evidence=r.brief()))
    return out


def check_split(T: dict, sb: Path) -> tuple[list[Finding], dict]:
    split = read_split(T, sb)
    if split is None:
        return [], {"status": "not run: split output not found"}
    out = []
    info = {"n_train": int((split["set"] == "train").sum()), "n_test": int((split["set"] == "test").sum())}
    dup_ids = split["id"][split["id"].duplicated()].unique()
    if len(dup_ids):
        out.append(Finding(
            check="split_duplicate_ids", category="leakage", severity="Critical",
            title="The same record id appears more than once in the split",
            detail=f"{len(dup_ids)} id(s) listed more than once, for example {', '.join(map(str, dup_ids[:3]))}.",
            recommendation="Make the split one row per record.", kind="dynamic"))
    if "group" in split.columns:
        sets = split.groupby("group")["set"].nunique()
        both = sets[sets > 1].index
        info.update({"n_groups": int(split["group"].nunique()), "groups_in_both": int(len(both))})
        n_test_groups = split.loc[split["set"] == "test", "group"].nunique()
        if len(both):
            rows = split[split["group"].isin(both) & (split["set"] == "test")]
            out.append(Finding(
                check="patient_overlap", category="leakage", severity="Critical",
                title="Patients appear in both the training and the test set",
                detail=(f"{len(both)} of {n_test_groups} test patients ({len(both) / max(1, n_test_groups):.0%}) also have "
                        f"records in training; {len(rows)} of {info['n_test']} test records belong to them. "
                        "Test performance is measured partly on patients the model has already seen."),
                recommendation="Split by patient (GroupShuffleSplit or GroupKFold) and re-report test metrics.",
                kind="dynamic", evidence={"groups_in_both": int(len(both)), "test_groups": int(n_test_groups),
                                          "example_groups": list(map(str, both[:3]))}))
    return out, info


def check_metrics(T: dict, sb: Path) -> tuple[list[Finding], dict]:
    pred = read_predictions(T, sb)
    rep = read_metrics(T, sb)
    if pred is None or rep is None:
        return [], {"status": "not run: predictions or metrics output not found"}
    out = []
    thr = rep.get("threshold")
    calc = compute_metrics(pred, float(thr) if thr is not None else None)
    info = {"reported": rep, "recomputed": calc}
    split = read_split(T, sb)
    if split is not None:
        test_ids = set(split.loc[split["set"] == "test", "id"])
        extra = set(pred["id"]) - test_ids
        missing = test_ids - set(pred["id"])
        info.update({"pred_not_in_test": len(extra), "test_not_in_pred": len(missing)})
        if extra:
            out.append(Finding(
                check="predictions_outside_test", category="execution_integrity", severity="High",
                title="The test predictions file contains records outside the test split",
                detail=f"{len(extra)} of {len(pred)} prediction rows are not in the test split.",
                recommendation="Write and evaluate predictions for test records only.", kind="dynamic"))
    mismatches = []
    for key, calc_key, tol in [("auroc", "auroc", 1e-4), ("auprc", "auprc", 1e-4), ("brier", "brier", 1e-4),
                               ("n", "n", 0)]:
        if rep.get(key) is not None and abs(float(rep[key]) - calc[calc_key]) > tol:
            mismatches.append((key, float(rep[key]), calc[calc_key]))
    if mismatches:
        out.append(Finding(
            check="metrics_not_reproducible", category="execution_integrity", severity="High",
            title="Reported metrics cannot be reproduced from the saved test predictions",
            detail="; ".join(f"{k}: reported {a:.4f}, recomputed {b:.4f}" for k, a, b in mismatches) + ".",
            recommendation="Compute every reported metric from the saved test predictions in one function, and test it.",
            kind="dynamic", evidence={"mismatches": mismatches}))
    if "alert" in pred.columns and thr is not None:
        a = pred["alert"].astype(int).to_numpy()
        expect = (pred["score"].astype(float).to_numpy() >= float(thr)).astype(int)
        n_diff = int((a != expect).sum())
        info["alerts_inconsistent_with_reported_threshold"] = n_diff
        if n_diff:
            out.append(Finding(
                check="alert_threshold_mismatch", category="execution_integrity", severity="High",
                title="Alerts are not issued at the threshold the metrics report",
                detail=(f"{n_diff} of {len(a)} test records have an alert flag that disagrees with score >= {thr} "
                        "(the reported threshold). Sensitivity and specificity in the report do not describe the alerts the platform emits."),
                recommendation="Use one threshold value for alerts and for the reported operating point.",
                kind="dynamic"))
    for key, calc_key in [("sensitivity", "sensitivity_from_alerts"), ("specificity", "specificity_from_alerts")]:
        if rep.get(key) is not None and calc_key in calc and abs(float(rep[key]) - calc[calc_key]) > 1e-4:
            out.append(Finding(
                check="operating_point_mismatch", category="execution_integrity", severity="High",
                title=f"Reported {key} does not match the emitted alerts",
                detail=f"Reported {float(rep[key]):.4f}; recomputed from the alert column {calc[calc_key]:.4f}.",
                recommendation="Derive operating-point metrics from the same alert column that is served.",
                kind="dynamic"))
    if pred["score"].isna().any() or (pred["score"] < 0).any() or (pred["score"] > 1).any():
        out.append(Finding(
            check="invalid_scores", category="execution_integrity", severity="High",
            title="Scores outside [0, 1] or missing",
            detail=f"{int(pred['score'].isna().sum())} missing, {int(((pred['score'] < 0) | (pred['score'] > 1)).sum())} out of range.",
            recommendation="Validate model output before writing it.", kind="dynamic"))
    return out, info


def check_rerun(T: dict, sb1: Path, sb2: Path, hs: tuple[str, str]) -> tuple[list[Finding], dict]:
    out, info = [], {"hash_seeds": list(hs), "files": {}}
    data_dir = T.get("data_dir")
    if data_dir:
        h1, h2 = dir_hash(sb1 / data_dir), dir_hash(sb2 / data_dir)
        info["files"][data_dir + "/"] = h1 == h2
        if h1 != h2:
            out.append(Finding(
                check="rerun_data_differs", category="reproducibility", severity="High",
                title="Two runs of the data step produce different data",
                detail=f"The contents of `{data_dir}/` differ between two clean runs of the same command.",
                recommendation="Seed every random draw and fix any ordering in the data step.",
                kind="dynamic"))
    differing = []
    for rel in T["outputs"].get("compare_on_rerun", []):
        same = file_hash(sb1 / rel) == file_hash(sb2 / rel)
        info["files"][rel] = same
        if not same:
            differing.append(rel)
    if differing:
        detail = f"Files that differ between two runs of the same commands on the same inputs: {', '.join(differing)}."
        p1, p2 = read_predictions(T, sb1), read_predictions(T, sb2)
        if p1 is not None and p2 is not None:
            m = p1[["id", "score"]].merge(p2[["id", "score"]], on="id", suffixes=("_1", "_2"))
            d = (m["score_1"] - m["score_2"]).abs()
            r1, r2 = read_metrics(T, sb1) or {}, read_metrics(T, sb2) or {}
            info.update({"pred_rows_compared": int(len(m)), "pred_rows_changed": int((d > 1e-9).sum()),
                         "max_abs_score_change": float(d.max()) if len(d) else None,
                         "auroc_run1": r1.get("auroc"), "auroc_run2": r2.get("auroc")})
            detail += (f" {info['pred_rows_changed']} of {len(m)} common test scores changed (largest change "
                       f"{info['max_abs_score_change']:.4f}); reported AUROC {r1.get('auroc')} vs {r2.get('auroc')}.")
        out.append(Finding(
            check="rerun_outputs_differ", category="reproducibility", severity="High",
            title="Rerunning the pipeline gives different results",
            detail=detail,
            recommendation="Fix seeds and ordering, then show two identical runs before any clinical evaluation.",
            kind="dynamic", evidence={"differing": differing}))
    return out, info


def check_temporal(T: dict, base_sb: Path, work: Path, python: str) -> tuple[list[Finding], dict]:
    tcfg = T.get("temporal")
    if not tcfg:
        return [], {"status": "not run: no temporal settings"}
    sb = make_sandbox(base_sb, work / "truncated")
    for d in T["outputs"].get("clear_before_rerun", []):
        shutil.rmtree(sb / d, ignore_errors=True)
    anc = tcfg["anchor"]
    a = pd.read_csv(sb / anc["path"], dtype={anc["id_col"]: str})
    anchor = pd.Series(pd.to_datetime(a[anc["time_col"]]).values, index=a[anc["id_col"]])
    removed = {}
    for t in tcfg["tables"]:
        p = sb / t["path"]
        df = pd.read_csv(p, dtype={t["id_col"]: str})
        ts = pd.to_datetime(df[t["time_col"]])
        cut = df[t["id_col"]].map(anchor)
        after = ts > cut
        removed[t["path"]] = {"rows": int(len(df)), "after_prediction_time": int(after.sum())}
        df.loc[~after].to_csv(p, index=False)
    r = run("truncated:pipeline", T["commands"]["pipeline"], sb, python, timeout=T.get("timeout", 900))
    info = {"removed": removed, "run": r.brief()}
    if not r.ok:
        return [Finding(
            check="temporal_truncation_fails", category="leakage", severity="High",
            title="The pipeline fails when data recorded after the prediction time are removed",
            detail=f"After deleting rows timed after the prediction time, `{r.command}` exited with code {r.returncode}. "
                   "The pipeline depends on information that would not exist when a prediction is made.",
            recommendation="Build features from data available at the prediction time only, and test that.",
            kind="dynamic", evidence=info)], info
    full, trunc = read_predictions(T, base_sb), read_predictions(T, sb)
    split = read_split(T, base_sb)
    groups = None
    if split is not None and "group" in split.columns:
        groups = split.set_index("id")["group"]
    cmp = paired_auroc_diff(full, trunc, groups)
    m = full[["id", "score"]].merge(trunc[["id", "score"]], on="id")
    cmp["test_scores_changed"] = int(((m["score_x"] - m["score_y"]).abs() > 1e-9).sum())
    info["comparison"] = cmp
    out = []
    if cmp["ci95"][0] > 0 and cmp["diff"] > 0.01:
        out.append(Finding(
            check="temporal_leakage", category="leakage", severity="Critical",
            title="Performance depends on data recorded after the prediction time",
            detail=(f"Removing every record timed after the prediction time lowered test AUROC from {cmp['auroc_a']:.3f} "
                    f"to {cmp['auroc_b']:.3f} (difference {cmp['diff']:.3f}, 95% CI {cmp['ci95'][0]:.3f} to {cmp['ci95'][1]:.3f}, "
                    f"bootstrap over {cmp['n_groups']} patients). Features use information the model would not have at the time it is used."),
            recommendation="Filter every feature source to times at or before the prediction time; add a test with post-time records.",
            kind="dynamic", evidence=cmp))
    elif cmp["test_scores_changed"] > 0:
        out.append(Finding(
            check="temporal_sensitivity", category="leakage", severity="Medium",
            title="Test scores change when post-prediction-time records are removed",
            detail=(f"{cmp['test_scores_changed']} of {cmp['n_common']} test scores changed, but the AUROC change "
                    f"({cmp['diff']:.3f}, 95% CI {cmp['ci95'][0]:.3f} to {cmp['ci95'][1]:.3f}) is not clearly above zero. "
                    "This can come from nondeterministic training as well as from leakage."),
            recommendation="Make training deterministic and repeat this probe.",
            kind="dynamic", evidence=cmp))
    return out, info


def _load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("audit_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply(pert: dict, target: Path) -> None:
    files = sorted(target.glob("*.csv")) if target.is_dir() else [target]
    for p in files:
        if target.is_dir() and pert.get("file") and p.name != pert["file"]:
            continue
        df = pd.read_csv(p)
        col = pert.get("column")
        op = pert["op"]
        n = len(df) if pert.get("rows", "all") == "all" else min(len(df), int(pert["rows"]))
        if op == "duplicate":
            df = pd.concat([df, df.head(n if pert.get("rows", "all") != "all" else len(df))], ignore_index=True)
        elif op == "drop":
            if col in df.columns:
                df = df.drop(columns=[col])
        elif op == "empty":
            df = df.iloc[0:0]
        elif col not in df.columns:
            continue
        elif op == "c_to_f":
            df[col] = df[col].astype(float) * 9 / 5 + 32
        elif op == "scale":
            df[col] = df[col].astype(float) * float(pert["value"])
        elif op == "set":
            df[col] = df[col].astype(object)
            df.loc[df.index[:n], col] = pert["value"]
        elif op == "nan":
            df[col] = df[col].astype(float)
            df.loc[df.index[:n], col] = np.nan
        df.to_csv(p, index=False)


def check_inference(T: dict, base_sb: Path, work: Path, python: str, adapter_path: Path) -> tuple[list[Finding], dict, list[RunResult]]:
    icfg = T.get("inference")
    if not icfg:
        return [], {"status": "not run"}, []
    adapter = _load_adapter(adapter_path)
    pred = read_predictions(T, base_sb)
    ids = list(pred["id"])
    out, info, runs = [], {"perturbations": []}, []
    oc = icfg["output"]

    def run_predict(label: str, inp: Path):
        o = work / f"pred_{label}.csv"
        if o.exists():
            o.unlink()
        r = run(f"predict:{label}", T["commands"]["predict"], base_sb, python, input=str(inp), output=str(o),
                timeout=T.get("timeout", 900))
        runs.append(r)
        res = None
        if r.ok and o.exists():
            res = pd.read_csv(o, dtype={oc["id_col"]: str}).rename(columns={oc["id_col"]: "id", oc["score_col"]: "score"})
        return r, res

    valid_dir = work / "inference_valid"
    shutil.rmtree(valid_dir, ignore_errors=True)
    valid_dir.mkdir(parents=True)
    valid_input = Path(adapter.make_input(base_sb, ids, valid_dir))
    r, valid = run_predict("valid", valid_input)
    valid_lines = {re.sub(r"[\d.]+|/\S+", "#", ln.strip().lower()) for ln in (r.stdout + "\n" + r.stderr).splitlines() if ln.strip()}
    if valid is None:
        out.append(Finding(
            check="inference_fails", category="integration", severity="Critical",
            title="Documented inference command fails on valid input",
            detail=f"`{r.command}` exited with code {r.returncode}: " + " | ".join(r.stderr.strip().splitlines()[-3:]),
            recommendation="Make the documented inference path run on the documented input format.",
            kind="dynamic", evidence=r.brief()))
        return out, info, runs
    m = pred[["id", "score"]].merge(valid[["id", "score"]], on="id", suffixes=("_validated", "_served"))
    d = (m["score_validated"] - m["score_served"]).abs()
    info["consistency"] = {"n_compared": int(len(m)), "n_differ": int((d > 1e-6).sum()),
                           "max_abs_diff": float(d.max()) if len(m) else None,
                           "missing_from_served": int(len(set(ids) - set(valid["id"])))}
    if len(m) == 0:
        out.append(Finding(
            check="inference_ids_mismatch", category="integration", severity="High",
            title="Inference output cannot be matched to the validated test records",
            detail="No record id in the inference output matches the test predictions.",
            recommendation="Carry the record id through inference unchanged.", kind="dynamic"))
    elif (d > 1e-6).any():
        y = m.merge(pred[["id", "label"]], on="id")
        au_v = roc_auc_score(y["label"].astype(int), y["score_validated"]) if y["label"].nunique() > 1 else None
        au_s = roc_auc_score(y["label"].astype(int), y["score_served"]) if y["label"].nunique() > 1 else None
        info["consistency"].update({"auroc_validated": au_v, "auroc_served": au_s})
        out.append(Finding(
            check="train_serve_skew", category="execution_integrity", severity="Critical",
            title="The inference path gives different scores from the validated pipeline",
            detail=(f"Scoring the {len(m)} test records through the documented inference command changed "
                    f"{int((d > 1e-6).sum())} scores (largest difference {d.max():.3f}). AUROC on these records: "
                    f"{au_v:.3f} validated vs {au_s:.3f} served. The validated performance does not describe what is served."),
            recommendation="Use one preprocessing and feature path for training and inference, and add a test that compares them.",
            kind="dynamic", evidence=info["consistency"]))
    for pert in icfg.get("perturbations", []):
        pdir = work / f"inference_{pert['name']}"
        shutil.rmtree(pdir, ignore_errors=True)
        if valid_input.is_dir():
            shutil.copytree(valid_input, pdir)
            target = pdir
        else:
            pdir.mkdir(parents=True)
            target = pdir / valid_input.name
            shutil.copy(valid_input, target)
        _apply(pert, target)
        r, res = run_predict(pert["name"], target)
        norm = lambda t: {re.sub(r"[\d.]+|/\S+", "#", ln.strip().lower()) for ln in t.splitlines() if ln.strip()}
        text = " ".join(norm(r.stdout + "\n" + r.stderr) - valid_lines)
        mentioned = bool(re.search(r"invalid|out of range|outside|warning|reject|exclud|dropp|missing|error", text)) or \
            bool(pert.get("column")) and pert["column"].lower() in text
        if not r.ok:
            outcome = "rejected"
        elif res is None:
            outcome = "no output"
        elif len(res) < len(valid) or mentioned or pert["op"] == "empty" and len(res) == 0:
            outcome = "flagged"
        else:
            outcome = "silent"
        changed = None
        if res is not None and len(res):
            mm = valid[["id", "score"]].merge(res[["id", "score"]], on="id")
            changed = int(((mm["score_x"] - mm["score_y"]).abs() > 1e-6).sum())
        rec = {"name": pert["name"], "description": pert.get("description", ""), "outcome": outcome,
               "returncode": r.returncode, "rows_out": None if res is None else int(len(res)),
               "scores_changed": changed, "expected": pert.get("expected", "reject or flag")}
        info["perturbations"].append(rec)
        if outcome == "silent" and pert.get("expected", "reject or flag") != "score":
            sev = pert.get("severity_if_silent", "Critical")
            out.append(Finding(
                check="silent_bad_input", category="failure_handling", severity=sev,
                title=f"Bad input scored silently: {pert.get('description', pert['name'])}",
                detail=(f"The inference command exited normally and returned {rec['rows_out']} scored records with no error or warning; "
                        f"{changed} score(s) changed compared with the valid input."),
                recommendation="Validate schema, types, units and ranges at the inference boundary; reject or flag the record.",
                kind="dynamic", evidence=rec))
    return out, info, runs


def check_tests_dynamic(T: dict, base: Path, work: Path, python: str) -> tuple[list[Finding], dict]:
    sb = make_sandbox(base, work / "tests_run")
    pkg = T.get("package")
    cmd = T["commands"].get("tests_coverage") or (
        "{python} -m coverage run --source=" + pkg + " -m pytest -q -p no:cacheprovider -rs")
    r = run("tests", cmd, sb, python, timeout=T.get("timeout", 900))
    rj = run("coverage_json", "{python} -m coverage json -q -o coverage.json", sb, python)
    info = {"run": r.brief(), "summary": None, "coverage": None}
    m = re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed)", r.stdout)
    counts = {k: int(v) for v, k in m}
    info["summary"] = counts
    out = []
    if counts.get("failed") or counts.get("error") or counts.get("errors") or (r.returncode not in (0, 5) and not counts):
        out.append(Finding(
            check="tests_fail", category="testing", severity="High",
            title="The test suite fails",
            detail=f"pytest summary: {counts or 'no summary'}; exit code {r.returncode}.",
            recommendation="Restore a passing suite before any evaluation run.", kind="dynamic"))
    cj = sb / "coverage.json"
    if cj.exists():
        cov = json.loads(cj.read_text())
        per = {k: v["summary"]["percent_covered"] for k, v in cov["files"].items()}
        info["coverage"] = {"total": cov["totals"]["percent_covered"], "files": per}
        nstmt = {k: v["summary"]["num_statements"] for k, v in cov["files"].items()}
        for f, pctv in sorted(per.items()):
            if 0 < pctv < 40 and nstmt.get(f, 0) >= 10 and not f.endswith("__init__.py") and not any(s in f for s in ("synth", "generate", "fake")):
                out.append(Finding(
                    check="low_coverage", category="testing", severity="Medium",
                    title=f"Low test coverage in {f}",
                    detail=f"{pctv:.0f}% of statements run under the test suite.",
                    recommendation="Add known-answer tests for this module.", kind="dynamic", file=f))
    return out, info
