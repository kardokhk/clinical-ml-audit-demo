"""Prove the audited repository was not modified."""
from __future__ import annotations

import hashlib
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def tree_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS):
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def compare(before: dict[str, str], after: dict[str, str]) -> dict:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"files_before": len(before), "files_after": len(after),
            "added": added, "removed": removed, "changed": changed,
            "unchanged": not (added or removed or changed)}
