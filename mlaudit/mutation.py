"""Small mutation test: change one operator or constant in a copy and see whether the tests notice.

Every mutant is applied to a disposable copy; the audited repository is never edited.
"""
from __future__ import annotations

import ast
import random
import shutil
from pathlib import Path

from .codebase import Codebase
from .findings import Finding
from .runner import make_sandbox, run

SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree: ast.AST) -> list[tuple[str, int, int, str]]:
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in SWAP:
                    sites.append(("compare", node.lineno, node.col_offset, f"{i}"))
        elif isinstance(node, ast.BinOp) and type(node.op) in SWAP:
            sites.append(("binop", node.lineno, node.col_offset, ""))
        elif isinstance(node, ast.BoolOp) and type(node.op) in SWAP:
            sites.append(("boolop", node.lineno, node.col_offset, ""))
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float) and not isinstance(node.value, bool) \
                and node.value not in (0, 1):
            sites.append(("constant", node.lineno, node.col_offset, ""))
    return sites


class _Mutator(ast.NodeTransformer):
    def __init__(self, kind, line, col, extra):
        self.kind, self.line, self.col, self.extra = kind, line, col, extra
        self.done = False
        self.description = ""

    def _hit(self, node):
        return not self.done and node.lineno == self.line and node.col_offset == self.col

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.kind == "compare" and self._hit(node):
            i = int(self.extra)
            old = type(node.ops[i])
            node.ops[i] = SWAP[old]()
            self.description = f"{old.__name__} -> {SWAP[old].__name__}"
            self.done = True
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if self.kind == "binop" and self._hit(node):
            old = type(node.op)
            node.op = SWAP[old]()
            self.description = f"{old.__name__} -> {SWAP[old].__name__}"
            self.done = True
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.kind == "boolop" and self._hit(node):
            old = type(node.op)
            node.op = SWAP[old]()
            self.description = f"{old.__name__} -> {SWAP[old].__name__}"
            self.done = True
        return node

    def visit_Constant(self, node):
        if self.kind == "constant" and self._hit(node):
            new = node.value + 1 if isinstance(node.value, int) else node.value * 1.5
            self.description = f"{node.value!r} -> {new!r}"
            self.done = True
            return ast.copy_location(ast.Constant(new), node)
        return node


def run_mutation(T: dict, base: Path, work: Path, python: str, cb: Codebase, n_mutants: int = 30,
                 seed: int = 20261117) -> tuple[list[Finding], dict]:
    skip = tuple(T.get("mutation_skip", ["synth", "__main__", "__init__"]))
    candidates = []
    for f in cb.source:
        if any(s in f.rel for s in skip):
            continue
        for site in _sites(f.tree):
            candidates.append((f.rel, site))
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n_mutants, len(candidates)))
    sb = make_sandbox(base, work / "mutants")
    test_cmd = T["commands"].get("tests_quick") or "{python} -m pytest -q -x -p no:cacheprovider"
    results = []
    for rel, (kind, line, col, extra) in chosen:
        path = sb / rel
        original = path.read_text()
        tree = ast.parse(original)
        mut = _Mutator(kind, line, col, extra)
        new_tree = mut.visit(tree)
        if not mut.done:
            continue
        path.write_text(ast.unparse(ast.fix_missing_locations(new_tree)))
        r = run(f"mutant:{rel}:{line}", test_cmd, sb, python, timeout=T.get("mutant_timeout", 300))
        path.write_text(original)
        killed = r.returncode != 0
        results.append({"file": rel, "line": line, "mutation": mut.description, "killed": killed,
                        "returncode": r.returncode})
    shutil.rmtree(sb, ignore_errors=True)
    total = len(results)
    killed = sum(r["killed"] for r in results)
    info = {"candidates": len(candidates), "run": total, "killed": killed,
            "score": killed / total if total else None, "seed": seed, "mutants": results}
    out = []
    by_file: dict[str, list[dict]] = {}
    for r in results:
        by_file.setdefault(r["file"], []).append(r)
    for rel, rs in sorted(by_file.items()):
        surv = [r for r in rs if not r["killed"]]
        if surv and len(surv) / len(rs) >= 0.5:
            out.append(Finding(
                check="mutants_survive", category="testing", severity="Medium",
                title=f"Tests do not notice changes to the logic in {rel}",
                detail=(f"{len(surv)} of {len(rs)} single-point changes to this file left every test passing, "
                        f"for example line {surv[0]['line']} ({surv[0]['mutation']})."),
                recommendation="Add tests that pin the expected outputs of this module on known cases.",
                kind="dynamic", file=rel, evidence={"survivors": surv}))
    return out, info
