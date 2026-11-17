"""Run a full read-only audit of one target and write findings.json and audit.json.

Usage: python -m mlaudit.audit targets/<name>.yaml --out results/<run>/<name> [--python PY] [--mutants 30]
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import yaml

from . import baseline, codebase, deps, dynamic, integrity, mutation, static
from .findings import Finding, number

REPO = Path(__file__).resolve().parents[1]


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for d in ("numpy", "pandas", "scikit-learn", "PyYAML", "joblib", "pytest", "coverage", "ruff", "bandit"):
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def audit(target_file: Path, out_dir: Path, python: str, n_mutants: int, work_root: Path | None) -> dict:
    T = yaml.safe_load(target_file.read_text())
    root = (target_file.parent / T["root"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    before = integrity.tree_hashes(root)

    findings: list[Finding] = []
    report: dict = {"target": T["name"], "root": T["root"], "versions": _versions()}

    # 1. static reading
    cb = codebase.load(root)
    report["inventory"] = codebase.inventory(cb)
    cfg_files = T.get("config_files", [])
    findings += static.run_static(cb, cfg_files, T["schema"].get("target_col"),
                                  bool(T["schema"].get("patient_col")) and T["schema"].get("patient_col") != T["schema"].get("record_col"))
    dep_findings, report["dependencies"] = deps.run_deps(cb)
    findings += dep_findings

    # 2. dynamic checks in disposable copies
    work = Path(tempfile.mkdtemp(prefix=f"mlaudit-{T['name']}-", dir=work_root))
    try:
        sb1 = dynamic.make_sandbox(root, work / "run1")
        sb2 = dynamic.make_sandbox(root, work / "run2")
        r1 = dynamic.run_pipeline(T, sb1, python, "0", "run1")
        r2 = dynamic.run_pipeline(T, sb2, python, "1", "run2")
        report["runs"] = [r.brief() for r in r1 + r2]
        findings += dynamic.check_run_failures(r1)
        all_runs = list(r1)
        if all(r.ok for r in r1):
            f, report["split"] = dynamic.check_split(T, sb1)
            findings += f
            f, report["metrics"] = dynamic.check_metrics(T, sb1)
            findings += f
            if all(r.ok for r in r2):
                f, report["rerun"] = dynamic.check_rerun(T, sb1, sb2, ("0", "1"))
                findings += f
            f, report["temporal"] = dynamic.check_temporal(T, sb1, work, python)
            findings += f
            adapter = T.get("inference", {}).get("adapter")
            if adapter:
                f, report["inference"], iruns = dynamic.check_inference(T, sb1, work, python, (target_file.parent / adapter).resolve())
                findings += f
                all_runs += iruns
        findings += dynamic.warnings_in(all_runs)
        f, report["tests"] = dynamic.check_tests_dynamic(T, root, work, python)
        findings += f
        if n_mutants:
            f, report["mutation"] = mutation.run_mutation(T, root, work, python, cb, n_mutants=n_mutants)
            findings += f
        base_f, report["baseline"] = baseline.run_baseline(dynamic.make_sandbox(root, work / "baseline"), python)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    after = integrity.tree_hashes(root)
    report["integrity"] = integrity.compare(before, after)
    report["seconds"] = round(time.time() - t0, 1)
    numbered = number(findings)
    by_sev: dict[str, int] = {}
    for f in numbered:
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
    report["finding_counts"] = by_sev
    report["n_findings"] = len(numbered)
    (out_dir / "findings.json").write_text(json.dumps(numbered, indent=2))
    (out_dir / "baseline.json").write_text(json.dumps([b.to_dict() for b in base_f], indent=2))
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--mutants", type=int, default=30)
    ap.add_argument("--work-root", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = audit(a.target, a.out, a.python, a.mutants, a.work_root)
    print(json.dumps({"target": rep["target"], "findings": rep["finding_counts"], "integrity_unchanged": rep["integrity"]["unchanged"],
                      "seconds": rep["seconds"]}))


if __name__ == "__main__":
    main()
