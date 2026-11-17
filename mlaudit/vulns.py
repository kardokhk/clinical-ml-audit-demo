"""Known-vulnerability scan of exactly pinned requirements with pip-audit (needs network access)."""
from __future__ import annotations

import datetime as dt
import json
import tempfile
from pathlib import Path

from .deps import parse_requirements
from .findings import Finding
from .runner import run


def run_vulns(root: Path, python: str) -> tuple[list[Finding], dict]:
    reqs, _ = parse_requirements(root)
    pinned = [r for r in reqs if r["spec"].startswith("==") and not any(c in r["spec"] for c in "*,<>")]
    info = {"scanned": [f"{r['name']}{r['spec']}" for r in pinned],
            "not_scanned_unpinned": [r["name"] for r in reqs if r not in pinned],
            "date": dt.date.today().isoformat(), "tool": "pip-audit (PyPI/OSV advisory data)"}
    if not pinned:
        info["status"] = "not run: no exact pins"
        return [], info
    with tempfile.TemporaryDirectory() as d:
        req = Path(d) / "requirements.txt"
        req.write_text("\n".join(f"{r['name']}{r['spec']}" for r in pinned) + "\n")
        r = run("pip-audit", "{python} -m pip_audit -r " + str(req) + " --no-deps --disable-pip -f json --progress-spinner off",
                root, python, timeout=300)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        info["status"] = f"not run: pip-audit exit {r.returncode}"
        return [], info
    info["status"] = "run"
    out = []
    by_name = {r["name"].lower(): r for r in pinned}
    for dep in data.get("dependencies", []):
        ids = sorted({v["id"] for v in dep.get("vulns", [])})
        if not ids:
            continue
        fixes = sorted({fv for v in dep["vulns"] for fv in v.get("fix_versions", [])})
        src = by_name.get(dep["name"].lower(), {})
        out.append(Finding(
            check="known_vulnerability", category="dependency", severity="High",
            title=f"`{dep['name']}=={dep['version']}` has published vulnerabilities",
            detail=f"{len(ids)} advisory id(s): {', '.join(ids[:5])}{' ...' if len(ids) > 5 else ''}. Fixed in: {', '.join(fixes[:4]) or 'no fix listed'}.",
            recommendation="Upgrade to a fixed version and rerun the validation, or record why the advisory does not apply.",
            file=src.get("file"), line=src.get("line"), evidence={"ids": ids, "fix_versions": fixes}))
    info["vulnerable_packages"] = len(out)
    return out, info
