"""Read a Python repository without executing it: files, ASTs, inventory, import graph."""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv", "venv", "outputs", "data"}


@dataclass
class SourceFile:
    rel: str
    text: str
    tree: ast.Module
    is_test: bool
    module: str

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()


@dataclass
class Codebase:
    root: Path
    files: list[SourceFile] = field(default_factory=list)
    parse_errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def source(self) -> list[SourceFile]:
        return [f for f in self.files if not f.is_test]

    @property
    def tests(self) -> list[SourceFile]:
        return [f for f in self.files if f.is_test]

    def local_top_names(self) -> set[str]:
        names = set()
        for f in self.files:
            names.add(f.module.split(".")[0])
        for p in self.root.iterdir():
            if p.is_dir() and (p / "__init__.py").exists():
                names.add(p.name)
        return names


def _is_test(rel: str) -> bool:
    parts = Path(rel).parts
    name = Path(rel).name
    return ("tests" in parts or "test" in parts or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


def _module_name(rel: str) -> str:
    p = Path(rel).with_suffix("")
    parts = list(p.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def load(root: str | Path) -> Codebase:
    root = Path(root).resolve()
    cb = Codebase(root=root)
    for p in sorted(root.rglob("*.py")):
        rel_parts = p.relative_to(root).parts
        if set(rel_parts[:-1]) & SKIP_DIRS:
            continue
        rel = str(p.relative_to(root))
        text = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            cb.parse_errors.append((rel, str(exc)))
            continue
        cb.files.append(SourceFile(rel, text, tree, _is_test(rel), _module_name(rel)))
    return cb


def imports_of(tree: ast.AST) -> list[tuple[str, int, int]]:
    """(module, line, level) for every import statement."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno, 0))
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.lineno, node.level))
    return out


def third_party_imports(cb: Codebase, include_tests: bool = False) -> dict[str, list[tuple[str, int]]]:
    local = cb.local_top_names()
    std = set(sys.stdlib_module_names) | {"__future__"}
    found: dict[str, list[tuple[str, int]]] = {}
    for f in cb.files:
        if f.is_test and not include_tests:
            continue
        for mod, line, level in imports_of(f.tree):
            if level or not mod:
                continue
            top = mod.split(".")[0]
            if top in std or top in local:
                continue
            found.setdefault(top, []).append((f.rel, line))
    return found


def internal_import_graph(cb: Codebase) -> list[tuple[str, str]]:
    modules = {f.module: f for f in cb.source}
    edges = set()
    for f in cb.source:
        pkg = f.module.rsplit(".", 1)[0] if "." in f.module else ""
        if f.rel.endswith("__init__.py"):
            pkg = f.module
        for node in ast.walk(f.tree):
            targets = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                    prefix = ".".join([b for b in base if b])
                    mod = f"{prefix}.{node.module}" if node.module else prefix
                    targets.append(mod)
                    for a in node.names:
                        targets.append(f"{mod}.{a.name}")
                else:
                    targets.append(node.module or "")
                    for a in node.names:
                        targets.append(f"{node.module}.{a.name}")
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names)
            for t in targets:
                if t in modules and t != f.module:
                    edges.add((f.module, t))
    return sorted(edges)


def inventory(cb: Codebase) -> dict:
    rows = []
    for f in cb.files:
        funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(f.tree))
        classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(f.tree))
        code_lines = sum(1 for ln in f.lines if ln.strip() and not ln.strip().startswith("#"))
        tests = 0
        if f.is_test:
            tests = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")
                        for n in ast.walk(f.tree))
        main_guard = "__name__" in f.text and "__main__" in f.text
        rows.append({"file": f.rel, "is_test": f.is_test, "code_lines": code_lines,
                     "functions": funcs, "classes": classes, "test_functions": tests,
                     "entry_point": main_guard or f.rel.endswith("__main__.py")})
    src = [r for r in rows if not r["is_test"]]
    tst = [r for r in rows if r["is_test"]]
    configs = sorted(str(p.relative_to(cb.root)) for ext in ("*.yaml", "*.yml", "*.toml", "*.json", "*.ini", "*.cfg")
                     for p in cb.root.rglob(ext)
                     if not (set(p.relative_to(cb.root).parts) & SKIP_DIRS))
    return {
        "files": rows,
        "source_files": len(src), "source_lines": sum(r["code_lines"] for r in src),
        "test_files": len(tst), "test_lines": sum(r["code_lines"] for r in tst),
        "test_functions": sum(r["test_functions"] for r in tst),
        "entry_points": [r["file"] for r in src if r["entry_point"]],
        "config_files": configs,
        "import_edges": internal_import_graph(cb),
        "parse_errors": cb.parse_errors,
    }
