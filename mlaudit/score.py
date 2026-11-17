"""Score audit findings against a planted-defect manifest.

Matching rules (fixed before the held-out run; see notes/evaluation_plan.md in the job folder and README):
- Located match: same file, cited line within [line_start - 3, line_end + 3], same category.
- Unlocated dynamic match: a dynamic finding with no file, same category.
- Each finding can match at most one defect and each defect at most one finding: maximum bipartite matching on
  located edges first, then on unlocated dynamic edges for the defects still unmatched.
- Unlocated matches are checked by hand afterwards, because category alone does not prove the finding is about that defect.
- Baseline (ruff + bandit): same file and line window; category is ignored, which favours the baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WINDOW = 3


def located(f: dict, d: dict, use_category: bool = True) -> bool:
    if f.get("file") != d["file"] or f.get("line") is None:
        return False
    if not (d["line_start"] - WINDOW <= f["line"] <= d["line_end"] + WINDOW):
        return False
    return (not use_category) or f["category"] == d["category"]


def unlocated(f: dict, d: dict) -> bool:
    return f.get("kind") == "dynamic" and not f.get("file") and f["category"] == d["category"]


HINTS = {
    "patient_overlap": ("patient", "split"), "split_duplicate_ids": ("duplicate",),
    "temporal_leakage": ("after", "future", "prediction time", "charted", "post"),
    "temporal_sensitivity": ("after", "future", "prediction time"),
    "train_serve_skew": ("inference", "serving", "served"),
    "metrics_not_reproducible": ("metric",), "operating_point_mismatch": ("threshold", "sensitivity"),
    "alert_threshold_mismatch": ("threshold",), "served_threshold_mismatch": ("threshold",),
    "silent_bad_input": ("range", "clip", "silently", "invalid", "unit"),
    "rerun_outputs_differ": ("random", "seed", "run"), "rerun_data_differs": ("random", "seed", "synthetic"),
}


def hint(f: dict, d: dict) -> bool:
    words = HINTS.get(f.get("check", ""), ())
    text = d.get("description", "").lower()
    return any(w in text for w in words)


def match(defects: list[dict], findings: list[dict], edge) -> dict[str, dict]:
    adj = {d["id"]: [i for i, f in enumerate(findings) if edge(f, d)] for d in defects}
    owner: dict[int, str] = {}

    def try_assign(did: str, seen: set) -> bool:
        for i in adj[did]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owner or try_assign(owner[i], seen):
                owner[i] = did
                return True
        return False

    for d in defects:
        try_assign(d["id"], set())
    return {did: findings[i] for i, did in owner.items()}


def score(defects: list[dict], findings: list[dict], baseline: list[dict]) -> dict:
    strict = match(defects, findings, located)

    # stage 2: defects without a located match may take an unused unlocated dynamic finding
    used = {id(f) for f in strict.values()}
    rest_d = [d for d in defects if d["id"] not in strict]
    rest_f = [f for f in findings if id(f) not in used]
    full = dict(strict)
    # stage 2a: unlocated findings whose check matches words in the defect description; 2b: category only
    hinted = match(rest_d, rest_f, lambda f, d: unlocated(f, d) and hint(f, d))
    full.update(hinted)
    used2 = {id(f) for f in hinted.values()}
    rest_d = [d for d in rest_d if d["id"] not in hinted]
    rest_f = [f for f in rest_f if id(f) not in used2]
    full.update(match(rest_d, rest_f, unlocated))
    base = match(defects, baseline, lambda f, d: located(f, d, use_category=False))
    rows = []
    for d in defects:
        f = full.get(d["id"])
        rows.append({
            "id": d["id"], "category": d["category"], "intended_severity": d["intended_severity"],
            "how_detectable": d.get("how_detectable"), "file": d["file"], "lines": [d["line_start"], d["line_end"]],
            "detected_located": d["id"] in strict,
            "detected": f is not None,
            "match_type": None if f is None else ("located" if located(f, d) else "unlocated_dynamic"),
            "finding_id": None if f is None else f["id"], "finding_check": None if f is None else f["check"],
            "finding_severity": None if f is None else f["severity"],
            "finding_title": None if f is None else f["title"],
            "baseline_detected": d["id"] in base,
            "baseline_rule": base[d["id"]]["check"] if d["id"] in base else None,
            "description": d["description"],
        })
    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"planted": 0, "detected": 0, "baseline": 0})
        c["planted"] += 1
        c["detected"] += int(r["detected"])
        c["baseline"] += int(r["baseline_detected"])
    matched_ids = {f["id"] for f in full.values()}
    return {
        "planted": len(defects),
        "detected": sum(r["detected"] for r in rows),
        "detected_located": sum(r["detected_located"] for r in rows),
        "baseline_detected": sum(r["baseline_detected"] for r in rows),
        "critical_high_planted": sum(r["intended_severity"] in ("Critical", "High") for r in rows),
        "critical_high_detected": sum(r["detected"] and r["intended_severity"] in ("Critical", "High") for r in rows),
        "findings_total": len(findings),
        "findings_unmatched": [f["id"] for f in findings if f["id"] not in matched_ids],
        "by_category": by_cat, "rows": rows,
    }


def clean_summary(findings: list[dict]) -> dict:
    counts = {s: 0 for s in ("Critical", "High", "Medium", "Low")}
    for f in findings:
        counts[f["severity"]] += 1
    return {"counts": counts, "false_alarms_critical_high": counts["Critical"] + counts["High"],
            "findings": [{"id": f["id"], "severity": f["severity"], "check": f["check"], "title": f["title"]} for f in findings]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--defective", type=Path, required=True, help="results dir of the defective twin")
    ap.add_argument("--clean", type=Path, required=True, help="results dir of the clean twin")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    defects = json.loads(a.manifest.read_text())
    findings = json.loads((a.defective / "findings.json").read_text())
    baseline = json.loads((a.defective / "baseline.json").read_text())
    res = score(defects, findings, baseline)
    res["clean_twin"] = clean_summary(json.loads((a.clean / "findings.json").read_text()))
    res["clean_twin_baseline_findings"] = len(json.loads((a.clean / "baseline.json").read_text()))
    res["defective_baseline_findings"] = len(baseline)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("planted", "detected", "detected_located", "baseline_detected",
                                          "critical_high_planted", "critical_high_detected")}
                     | {"clean_false_alarms": res["clean_twin"]["false_alarms_critical_high"]}))


if __name__ == "__main__":
    main()
