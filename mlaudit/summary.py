"""Collect the numbers that documents and the README quote into one summary.json.

usage: python -m mlaudit.summary results/<frozen> results/<posthoc> OUT.json
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def run_block(d: Path) -> dict:
    out = {}
    for t in ("platform_a", "platform_a_clean", "platform_b", "platform_b_clean"):
        a = load(d / t, "audit.json")
        if a is None:
            continue
        out[t] = {
            "finding_counts": {s: a["finding_counts"].get(s, 0) for s in ("Critical", "High", "Medium", "Low")},
            "n_findings": a["n_findings"], "seconds": a["seconds"],
            "files_hashed": a["integrity"]["files_before"], "unchanged": a["integrity"]["unchanged"],
            "inventory": {k: a["inventory"][k] for k in ("source_files", "source_lines", "test_files", "test_functions")},
            "mutation": {k: a.get("mutation", {}).get(k) for k in ("run", "killed", "seed")},
            "tests": a.get("tests", {}).get("summary"),
            "coverage_total": (a.get("tests", {}).get("coverage") or {}).get("total"),
            "split": a.get("split"), "metrics": a.get("metrics"), "rerun": a.get("rerun"),
            "temporal": (a.get("temporal") or {}).get("comparison"),
            "inference": a.get("inference"), "vulnerabilities": a.get("vulnerabilities"),
            "baseline": a.get("baseline"),
        }
    for p in ("a", "b"):
        s = load(d, f"score_{p}.json")
        if s:
            out[f"score_{p}"] = {k: s[k] for k in ("planted", "detected", "detected_located", "baseline_detected",
                                                   "critical_high_planted", "critical_high_detected", "by_category")}
            out[f"score_{p}"]["clean"] = s["clean_twin"]["counts"]
            out[f"score_{p}"]["clean_false_alarms"] = s["clean_twin"]["false_alarms_critical_high"]
            out[f"score_{p}"]["rows"] = [{k: r[k] for k in ("id", "category", "intended_severity", "detected", "match_type",
                                                             "finding_check", "baseline_detected", "baseline_rule")} for r in s["rows"]]
    adj = load(d, "adjudication.json")
    if adj:
        out["adjudication"] = adj
        out["wilson_b"] = wilson(adj["adjudicated_detected"], adj["planted"])
        out["wilson_b_ch"] = wilson(adj["critical_high_adjudicated"], adj["critical_high_planted"])
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    frozen, posthoc, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    tag = subprocess.run(["git", "rev-list", "-n", "1", "frozen-test"], capture_output=True, text=True).stdout.strip()
    summary = {"frozen_run": frozen.name, "posthoc_run": posthoc.name, "frozen_commit": tag,
               "frozen": run_block(frozen), "posthoc": run_block(posthoc)}
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(out)


if __name__ == "__main__":
    main()
