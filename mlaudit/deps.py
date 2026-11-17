"""Dependency declarations against what the code imports."""
from __future__ import annotations

import re
from pathlib import Path

from .codebase import Codebase, third_party_imports
from .findings import Finding

# import name -> distribution name, for the common cases where they differ
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn", "yaml": "PyYAML", "cv2": "opencv-python", "PIL": "Pillow",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "dateutil": "python-dateutil",
    "dotenv": "python-dotenv", "Bio": "biopython", "lightgbm": "lightgbm", "xgboost": "xgboost",
    "torch": "torch", "jose": "python-jose", "attr": "attrs", "google": "protobuf",
}

REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(root: Path) -> tuple[list[dict], list[str]]:
    reqs, files = [], []
    for p in sorted(root.glob("requirements*.txt")):
        files.append(p.name)
        for i, raw in enumerate(p.read_text().splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith(("-", "--")):
                continue
            m = REQ_LINE.match(line)
            if m:
                spec = m.group(3).split(";")[0].strip()
                reqs.append({"name": m.group(1), "spec": spec, "file": p.name, "line": i,
                             "hashed": "--hash" in raw})
    pp = root / "pyproject.toml"
    if pp.exists():
        files.append("pyproject.toml")
        text = pp.read_text()
        block = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
        if block:
            start_line = text[: block.start()].count("\n") + 1
            for j, item in enumerate(re.findall(r"[\"']([^\"']+)[\"']", block.group(1))):
                m = REQ_LINE.match(item)
                if m:
                    reqs.append({"name": m.group(1), "spec": m.group(3).strip(), "file": "pyproject.toml",
                                 "line": start_line + j + 1, "hashed": False})
    return reqs, files


def run_deps(cb: Codebase) -> tuple[list[Finding], dict]:
    root = cb.root
    reqs, files = parse_requirements(root)
    out: list[Finding] = []
    lockfiles = [n for n in ("poetry.lock", "uv.lock", "Pipfile.lock", "conda-lock.yml", "requirements.lock")
                 if (root / n).exists()]
    if not files:
        out.append(Finding(
            check="no_dependency_file", category="dependency", severity="High",
            title="No dependency declaration found",
            detail="No requirements*.txt or pyproject.toml dependency list exists.",
            recommendation="Declare every third-party package with an exact version.",
            file=None))
    for r in reqs:
        if not r["spec"].startswith("==") or any(ch in r["spec"] for ch in "*,<>"):
            out.append(Finding(
                check="unpinned_dependency", category="dependency", severity="Medium",
                title=f"`{r['name']}` is not pinned to an exact version",
                detail=f"{r['file']} line {r['line']} declares `{r['name']}{(' ' + r['spec']) if r['spec'] else ''}`. "
                       "A rebuild can install a different version from the one that was validated.",
                recommendation="Pin exact versions and keep a lock file with hashes.",
                file=r["file"], line=r["line"]))
    if reqs and not lockfiles and not any(r["hashed"] for r in reqs):
        out.append(Finding(
            check="no_lock_file", category="dependency", severity="Low",
            title="No lock file with hashes",
            detail="Transitive dependencies are not fixed, so two installs can differ even with exact top-level pins.",
            recommendation="Add a lock file (pip-tools with --generate-hashes, uv or poetry).",
            file=files[0] if files else None))
    declared = {norm(r["name"]) for r in reqs}
    imported = third_party_imports(cb)
    for mod, where in sorted(imported.items()):
        dist = norm(IMPORT_TO_DIST.get(mod, mod))
        if dist not in declared and norm(mod) not in declared:
            rel, line = where[0]
            out.append(Finding(
                check="undeclared_dependency", category="dependency", severity="High",
                title=f"`{mod}` is imported but not declared",
                detail=f"First imported in {rel} line {line} ({len(where)} import site(s)); no requirement provides it. "
                       "A clean install from the declared dependencies fails at run time.",
                recommendation=f"Add {IMPORT_TO_DIST.get(mod, mod)} with an exact version.",
                file=rel, line=line))
    used_dists = {norm(IMPORT_TO_DIST.get(m, m)) for m in imported} | {norm(m) for m in imported}
    test_imports = third_party_imports(cb, include_tests=True)
    used_dists |= {norm(IMPORT_TO_DIST.get(m, m)) for m in test_imports} | {norm(m) for m in test_imports}
    implicit = {"joblib", "scipy", "threadpoolctl", "numpy", "pytest", "pyarrow", "openpyxl", "tzdata"}
    for r in reqs:
        n = norm(r["name"])
        if n not in used_dists and n not in implicit:
            out.append(Finding(
                check="unused_dependency", category="dependency", severity="Low",
                title=f"`{r['name']}` is declared but never imported",
                detail=f"{r['file']} line {r['line']}.",
                recommendation="Remove it or document why it is needed.",
                file=r["file"], line=r["line"]))
    summary = {"declaration_files": files, "lock_files": lockfiles, "requirements": reqs,
               "third_party_imports": sorted(imported)}
    return out, summary
