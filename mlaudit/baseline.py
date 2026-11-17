"""Baseline: what a standard linter, a security scanner and a test run report on their own."""
from __future__ import annotations

import json
from pathlib import Path

from .findings import Finding
from .runner import run


def run_baseline(base: Path, python: str) -> tuple[list[Finding], dict]:
    out: list[Finding] = []
    r = run("ruff", "{python} -m ruff check --output-format json --no-cache --exit-zero .", base, python)
    ruff = json.loads(r.stdout or "[]") if r.ok else []
    for item in ruff:
        rel = str(Path(item["filename"]).resolve().relative_to(base.resolve()))
        out.append(Finding(check=f"ruff:{item['code']}", category="baseline", severity="Low",
                           title=item["message"], detail="", recommendation="", file=rel,
                           line=item["location"]["row"], kind="baseline"))
    b = run("bandit", "{python} -m bandit -r . -f json -q --exit-zero -s B101", base, python)
    try:
        bandit = json.loads(b.stdout or "{}").get("results", [])
    except json.JSONDecodeError:
        bandit = []
    for item in bandit:
        rel = str(Path(item["filename"]).as_posix()).lstrip("./")
        out.append(Finding(check=f"bandit:{item['test_id']}", category="baseline", severity="Low",
                           title=item["issue_text"], detail="", recommendation="", file=rel,
                           line=item["line_number"], kind="baseline"))
    info = {"ruff_findings": len(ruff), "bandit_findings": len(bandit),
            "ruff_rc": r.returncode, "bandit_rc": b.returncode}
    return out, info
