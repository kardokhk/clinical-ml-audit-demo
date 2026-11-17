"""Static checks for ML-specific risks. Nothing here imports or runs the audited code."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from .codebase import Codebase, SourceFile
from .findings import Finding

SPLIT_FUNCS = {"train_test_split"}
UNGROUPED_SPLITTERS = {"train_test_split", "KFold", "StratifiedKFold", "ShuffleSplit",
                       "StratifiedShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold"}
SEEDED_HIGH = {"train_test_split", "GroupShuffleSplit", "ShuffleSplit", "StratifiedShuffleSplit",
               "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier",
               "ExtraTreesRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor",
               "MLPClassifier", "MLPRegressor", "SGDClassifier", "IsolationForest", "KMeans",
               "RandomizedSearchCV", "permutation_importance", "resample", "XGBClassifier",
               "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "BaggingClassifier"}
SEEDED_MEDIUM = {"HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
                 "DecisionTreeClassifier", "DecisionTreeRegressor", "LogisticRegressionCV"}
SHUFFLE_SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold"}
LEGACY_NP_RANDOM = {"rand", "randn", "randint", "random", "choice", "shuffle", "permutation",
                    "normal", "uniform", "binomial", "poisson", "random_sample", "sample", "beta",
                    "exponential", "gamma", "lognormal"}
PY_RANDOM = {"random", "randint", "choice", "choices", "shuffle", "sample", "uniform", "gauss"}
CONFIG_NAME = re.compile(r"(^|_)(cfg|conf|config|settings|params|hparams)$", re.I)
SCORE_NAME = re.compile(r"(score|prob|proba|risk|pred)", re.I)
DICT_METHODS = {"get", "items", "keys", "values", "copy", "update", "setdefault", "pop"}


def call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return dotted(node.func) + "()"
    return "?"


def kw(node: ast.Call, name: str):
    for k in node.keywords:
        if k.arg == name:
            return k.value
    return None


def scopes(tree: ast.Module):
    """Yield (scope_node, body_nodes) for the module and every function."""
    yield tree, [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, node.body


def calls_in(nodes) -> list[ast.Call]:
    out = []
    for n in nodes:
        for c in ast.walk(n):
            if isinstance(c, ast.Call):
                out.append(c)
    return out


def _inside_pipeline(call: ast.Call, parents: dict) -> bool:
    p = parents.get(call)
    while p is not None:
        if isinstance(p, ast.Call) and call_name(p) in {"Pipeline", "make_pipeline", "ColumnTransformer"}:
            return True
        p = parents.get(p)
    return False


def _parents(tree: ast.AST) -> dict:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


# ---------------------------------------------------------------- leakage

def is_split_call(c: ast.Call) -> bool:
    name = call_name(c)
    if name in SPLIT_FUNCS:
        return True
    if "split" in name.lower() and name not in {"split", "rsplit", "splitlines"}:
        return True
    return name == "split" and isinstance(c.func, ast.Attribute) and bool(
        re.search(r"split|cv|fold|gss", dotted(c.func.value), re.I))


def check_fit_before_split(f: SourceFile) -> list[Finding]:
    out = []
    for scope, body in scopes(f.tree):
        calls = calls_in(body)
        split_lines = [c.lineno for c in calls if is_split_call(c)]
        if not split_lines:
            continue
        first_split = min(split_lines)
        for c in calls:
            name = call_name(c)
            if name in {"fit", "fit_transform"} and isinstance(c.func, ast.Attribute) and c.lineno < first_split:
                target = dotted(c.func.value)
                out.append(Finding(
                    check="fit_before_split", category="leakage", severity="High",
                    title="Preprocessing fitted before the train/test split",
                    detail=(f"`{target}.{name}(...)` on line {c.lineno} runs before the split on line {first_split} "
                            f"in the same function, so statistics from the test rows enter training."),
                    recommendation="Fit every imputer, scaler or selector on training rows only, inside a Pipeline.",
                    file=f.rel, line=c.lineno))
    return out


def check_ungrouped_split(f: SourceFile, has_groups: bool) -> list[Finding]:
    if not has_groups:
        return []
    out = []
    for c in calls_in([f.tree]):
        name = call_name(c)
        if name in UNGROUPED_SPLITTERS and kw(c, "groups") is None:
            out.append(Finding(
                check="ungrouped_split", category="leakage", severity="High",
                title=f"`{name}` does not group rows by patient",
                detail=(f"Line {c.lineno} splits rows without a patient grouping while the data hold several "
                        "rows per patient. The same patient can then sit in training and test."),
                recommendation="Split with GroupShuffleSplit or GroupKFold on the patient id.",
                file=f.rel, line=c.lineno))
    return out


def check_target_in_features(f: SourceFile, target: str | None) -> list[Finding]:
    if not target or "feature" not in Path(f.rel).stem.lower():
        return []
    out = []
    for node in ast.walk(f.tree):
        if isinstance(node, ast.Constant) and node.value == target:
            out.append(Finding(
                check="target_in_features", category="leakage", severity="Medium",
                title="Target column referenced in feature code",
                detail=f"The outcome column `{target}` appears on line {node.lineno} of a feature module.",
                recommendation="Confirm the outcome cannot reach the feature matrix; keep labels out of feature code.",
                file=f.rel, line=node.lineno))
    return out


# ---------------------------------------------------------------- reproducibility

def check_seeds(f: SourceFile, np_seeded: bool, py_seeded: bool) -> list[Finding]:
    out = []
    for c in calls_in([f.tree]):
        name = call_name(c)
        d = dotted(c.func)
        if name in SEEDED_HIGH | SEEDED_MEDIUM:
            rs = kw(c, "random_state")
            if rs is None or (isinstance(rs, ast.Constant) and rs.value is None):
                sev = "High" if name in SEEDED_HIGH else "Medium"
                out.append(Finding(
                    check="unseeded_estimator", category="reproducibility", severity=sev,
                    title=f"`{name}` has no random_state",
                    detail=f"Line {c.lineno} creates `{name}` without a fixed random_state, so reruns can differ.",
                    recommendation="Pass random_state from the configured seed.",
                    file=f.rel, line=c.lineno))
        elif name in SHUFFLE_SPLITTERS:
            sh = kw(c, "shuffle")
            if isinstance(sh, ast.Constant) and sh.value is True and kw(c, "random_state") is None:
                out.append(Finding(
                    check="unseeded_estimator", category="reproducibility", severity="High",
                    title=f"`{name}(shuffle=True)` has no random_state",
                    detail=f"Line {c.lineno} shuffles folds without a seed.",
                    recommendation="Pass random_state from the configured seed.",
                    file=f.rel, line=c.lineno))
        elif d.startswith(("np.random.", "numpy.random.")) and name in LEGACY_NP_RANDOM and not np_seeded:
            out.append(Finding(
                check="unseeded_random", category="reproducibility", severity="Medium",
                title="Global NumPy random state used without a seed",
                detail=f"`{d}` on line {c.lineno} draws from the unseeded global generator.",
                recommendation="Use np.random.default_rng(seed) passed down from the config.",
                file=f.rel, line=c.lineno))
        elif d in {"np.random.default_rng", "numpy.random.default_rng"} and not c.args and kw(c, "seed") is None:
            out.append(Finding(
                check="unseeded_random", category="reproducibility", severity="Medium",
                title="Random generator created without a seed",
                detail=f"`default_rng()` on line {c.lineno} has no seed.",
                recommendation="Pass the configured seed.",
                file=f.rel, line=c.lineno))
        elif d.startswith("random.") and name in PY_RANDOM and not py_seeded:
            out.append(Finding(
                check="unseeded_random", category="reproducibility", severity="Medium",
                title="Python random module used without a seed",
                detail=f"`{d}` on line {c.lineno} uses the unseeded global generator.",
                recommendation="Use random.Random(seed).",
                file=f.rel, line=c.lineno))
        elif name == "sample" and isinstance(c.func, ast.Attribute) and not d.startswith(("random.", "np.", "numpy.")) \
                and kw(c, "random_state") is None and (kw(c, "frac") is not None or kw(c, "n") is not None):
            out.append(Finding(
                check="unseeded_random", category="reproducibility", severity="Medium",
                title="DataFrame.sample without random_state",
                detail=f"`{d}` on line {c.lineno} samples rows without a seed.",
                recommendation="Pass random_state.",
                file=f.rel, line=c.lineno))
    return out


def check_wallclock(f: SourceFile) -> list[Finding]:
    out = []
    for c in calls_in([f.tree]):
        d = dotted(c.func)
        if re.search(r"(datetime\.(now|utcnow|today)|date\.today|Timestamp\.(now|today)|^time\.time)$", d):
            out.append(Finding(
                check="wall_clock", category="reproducibility", severity="Medium",
                title="Result may depend on the date or time of the run",
                detail=f"`{d}()` on line {c.lineno}.",
                recommendation="Take the reference time from config or the data, not the system clock.",
                file=f.rel, line=c.lineno))
        if call_name(c) in {"Timestamp", "to_datetime"} and c.args and isinstance(c.args[0], ast.Constant) \
                and c.args[0].value in {"now", "today"}:
            out.append(Finding(
                check="wall_clock", category="reproducibility", severity="Medium",
                title="Result may depend on the date or time of the run",
                detail=f"`{call_name(c)}('{c.args[0].value}')` on line {c.lineno}.",
                recommendation="Take the reference time from config or the data.",
                file=f.rel, line=c.lineno))
    return out


def check_unordered(f: SourceFile) -> list[Finding]:
    out = []
    parents = _parents(f.tree)
    for c in calls_in([f.tree]):
        name = call_name(c)
        d = dotted(c.func)
        if name in {"glob", "rglob", "iterdir", "listdir", "scandir"} and (d.startswith(("glob.", "os.")) or name in {"rglob", "iterdir"} or (name == "glob" and isinstance(c.func, ast.Attribute))):
            p = parents.get(c)
            wrapped = isinstance(p, ast.Call) and call_name(p) == "sorted"
            if not wrapped:
                out.append(Finding(
                    check="unordered_listing", category="reproducibility", severity="Medium",
                    title="File listing order is not fixed",
                    detail=f"`{d}` on line {c.lineno} is not wrapped in sorted(); order depends on the file system.",
                    recommendation="Wrap directory listings in sorted().",
                    file=f.rel, line=c.lineno))
    for node in ast.walk(f.tree):
        target = None
        if isinstance(node, ast.Call) and call_name(node) in {"list", "tuple"} and node.args:
            target = node.args[0]
        elif isinstance(node, ast.For):
            target = node.iter
        elif isinstance(node, ast.comprehension):
            target = node.iter
        if target is None:
            continue
        is_set = isinstance(target, (ast.Set, ast.SetComp)) or (
            isinstance(target, ast.Call) and call_name(target) in {"set", "frozenset"})
        if is_set:
            out.append(Finding(
                check="set_ordering", category="reproducibility", severity="Medium",
                title="Order taken from a set",
                detail=(f"Line {node.lineno} turns a set into an ordered sequence. String hashing is "
                        "randomised per process, so the order can change between runs."),
                recommendation="Use sorted(...) or a list with a fixed order.",
                file=f.rel, line=node.lineno))
    return out


def check_hardcoded_paths(f: SourceFile) -> list[Finding]:
    out = []
    for node in ast.walk(f.tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if (re.match(r"^/(home|mnt|data|Users|tmp|opt|srv|scratch)/", v) or re.match(r"^[A-Za-z]:\\\\", v)):
                out.append(Finding(
                    check="hardcoded_path", category="reproducibility", severity="Low",
                    title="Hard-coded absolute path",
                    detail=f"`{v}` on line {node.lineno}.",
                    recommendation="Read paths from config.",
                    file=f.rel, line=node.lineno))
    return out


# ---------------------------------------------------------------- failure handling

def check_broad_except(f: SourceFile) -> list[Finding]:
    out = []
    for node in ast.walk(f.tree):
        if isinstance(node, ast.ExceptHandler):
            t = node.type
            broad = t is None or (isinstance(t, ast.Name) and t.id in {"Exception", "BaseException"}) or (
                isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id in {"Exception", "BaseException"} for e in t.elts))
            reraises = any(isinstance(n, ast.Raise) for n in ast.walk(ast.Module(body=node.body, type_ignores=[])))
            if broad and not reraises:
                out.append(Finding(
                    check="broad_except", category="failure_handling", severity="High",
                    title="Broad exception handler continues silently",
                    detail=(f"Line {node.lineno} catches {'everything' if t is None else dotted(t)} and does not re-raise. "
                            "Errors become default values or skipped steps instead of visible failures."),
                    recommendation="Catch the specific exception, log it, and fail or flag the record.",
                    file=f.rel, line=node.lineno))
    return out


def check_coerce(f: SourceFile) -> list[Finding]:
    out = []
    for c in calls_in([f.tree]):
        e = kw(c, "errors")
        if isinstance(e, ast.Constant) and e.value == "coerce" and call_name(c) in {"to_numeric", "to_datetime", "to_timedelta"}:
            out.append(Finding(
                check="silent_coerce", category="failure_handling", severity="Medium",
                title="Invalid values silently become missing",
                detail=f"`{call_name(c)}(errors='coerce')` on line {c.lineno} turns unreadable values into NaN/NaT without reporting them.",
                recommendation="Raise or count and report coerced values.",
                file=f.rel, line=c.lineno))
    return out


def check_assert_validation(f: SourceFile) -> list[Finding]:
    out = []
    for node in ast.walk(f.tree):
        if isinstance(node, ast.Assert):
            out.append(Finding(
                check="assert_validation", category="failure_handling", severity="Low",
                title="Runtime check written as assert",
                detail=f"Line {node.lineno}: assert statements are removed when Python runs with -O.",
                recommendation="Raise an explicit exception.",
                file=f.rel, line=node.lineno))
    return out


# ---------------------------------------------------------------- execution integrity

def check_model_load(f: SourceFile) -> list[Finding]:
    out = []
    for scope, body in scopes(f.tree):
        if scope is f.tree:
            continue
        src = ast.get_source_segment(f.text, scope) or ""
        for c in calls_in(body):
            d = dotted(c.func)
            if d in {"joblib.load", "pickle.load", "pickle.loads", "torch.load", "cloudpickle.load"} \
                    and not re.search(r"sha256|md5|hash|digest|checksum", src, re.I):
                out.append(Finding(
                    check="model_load_unverified", category="execution_integrity", severity="Low",
                    title="Model artefact loaded without an integrity check",
                    detail=f"`{d}` on line {c.lineno} loads a pickled object with no hash or version check.",
                    recommendation="Store a hash and the training commit with the model and verify both at load.",
                    file=f.rel, line=c.lineno))
    return out


def check_threshold_literals(f: SourceFile) -> list[Finding]:
    out = []
    for node in ast.walk(f.tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            left, right = node.left, node.comparators[0]
            lit = right if isinstance(right, ast.Constant) else left if isinstance(left, ast.Constant) else None
            other = left if lit is right else right
            if lit is not None and isinstance(lit.value, float) and 0 < lit.value < 1 and SCORE_NAME.search(dotted(other) + ast.unparse(other)):
                out.append(Finding(
                    check="hardcoded_threshold", category="execution_integrity", severity="Medium",
                    title="Decision threshold hard-coded in code",
                    detail=f"Line {node.lineno} compares a score with {lit.value} directly.",
                    recommendation="Read the operating threshold from one versioned place and log it with every prediction.",
                    file=f.rel, line=node.lineno))
    return out


# ---------------------------------------------------------------- integration

def _flatten(d, prefix="") -> set[str]:
    keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            keys.add(p)
            keys |= _flatten(v, p)
    return keys


def config_keys(root: Path, files: list[str]) -> set[str]:
    keys: set[str] = set()
    for rel in files:
        p = root / rel
        if p.suffix in {".yaml", ".yml"}:
            try:
                keys |= _flatten(yaml.safe_load(p.read_text()) or {})
            except yaml.YAMLError:
                pass
    return keys


def _config_chain(node: ast.AST, aliases: dict[str, str]):
    """Return (path, kind, has_default) when node reads from a config mapping."""
    parts = []
    has_default = False
    kind = "subscript"
    cur = node
    if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr == "get" and cur.args \
            and isinstance(cur.args[0], ast.Constant) and isinstance(cur.args[0].value, str):
        parts.append(cur.args[0].value)
        has_default = len(cur.args) > 1 or kw(cur, "default") is not None
        kind = "get"
        cur = cur.func.value
    while isinstance(cur, ast.Subscript) and isinstance(cur.slice, ast.Constant) and isinstance(cur.slice.value, str):
        parts.append(cur.slice.value)
        cur = cur.value
    if not parts:
        return None
    if isinstance(cur, ast.Name):
        if cur.id in aliases:
            prefix = aliases[cur.id]
            return (".".join(([prefix] if prefix else []) + parts[::-1]), kind, has_default)
        if CONFIG_NAME.search(cur.id):
            return (".".join(parts[::-1]), kind, has_default)
    if isinstance(cur, ast.Attribute) and CONFIG_NAME.search(cur.attr):
        return (".".join(parts[::-1]), kind, has_default)
    return None


def config_reads(f: SourceFile) -> list[tuple[str, int, str, bool]]:
    reads = []
    for scope, _ in scopes(f.tree):
        if scope is f.tree:
            continue
        aliases: dict[str, str] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                ch = _config_chain(node.value, aliases)
                if ch and ch[1] == "subscript":
                    aliases[node.targets[0].id] = ch[0]
        seen_inner = set()
        for node in ast.walk(scope):
            if isinstance(node, (ast.Subscript, ast.Call)) and id(node) not in seen_inner:
                ch = _config_chain(node, aliases)
                if ch:
                    # mark inner subscripts of this chain as seen so only the full path is recorded
                    cur = node.func.value if isinstance(node, ast.Call) else node.value
                    while isinstance(cur, ast.Subscript):
                        seen_inner.add(id(cur))
                        cur = cur.value
                    reads.append((ch[0], node.lineno, ch[1], ch[2]))
    return reads


def check_config_keys(cb: Codebase, cfg_files: list[str]) -> list[Finding]:
    keys = config_keys(cb.root, cfg_files)
    if not keys:
        return []
    out = []
    used = set()
    for f in cb.source:
        for path, line, kind, has_default in config_reads(f):
            used.add(path)
            parts = path.split(".")
            for i in range(1, len(parts)):
                used.add(".".join(parts[:i]))
            if path not in keys:
                silent = kind == "get"
                out.append(Finding(
                    check="config_key_missing", category="integration", severity="High",
                    title=f"Code reads config key `{path}`, which the config does not define",
                    detail=(f"Line {line} reads `{path}`" + (" with a fallback default, so the configured value is ignored silently."
                                                            if has_default else ("; the read returns None silently." if silent else "; the run fails with KeyError."))),
                    recommendation="Align the key name in code and config, and fail on unknown or missing keys.",
                    file=f.rel, line=line))
    for k in sorted(keys - used):
        if not any(u.startswith(k + ".") for u in used):
            out.append(Finding(
                check="config_key_unused", category="integration", severity="Low",
                title=f"Config key `{k}` is never read",
                detail="A setting that no code reads usually means a renamed key or a dead option.",
                recommendation="Remove it or wire it to the code that should use it.",
                file=cfg_files[0] if cfg_files else None))
    return out


def check_dead_functions(cb: Codebase) -> list[Finding]:
    defs = []
    for f in cb.source:
        for node in f.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_") \
                    and node.name not in {"main"}:
                defs.append((f, node))
    names_used: dict[str, int] = {}
    for f in cb.files:
        for node in ast.walk(f.tree):
            if isinstance(node, ast.Name):
                names_used[node.id] = names_used.get(node.id, 0) + 1
            elif isinstance(node, ast.Attribute):
                names_used[node.attr] = names_used.get(node.attr, 0) + 1
            elif isinstance(node, ast.alias):
                names_used[node.name.split(".")[-1]] = names_used.get(node.name.split(".")[-1], 0) + 1
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
                names_used[node.value] = names_used.get(node.value, 0) + 1
    out = []
    for f, node in defs:
        if names_used.get(node.name, 0) == 0:
            out.append(Finding(
                check="unreferenced_code", category="integration", severity="Low",
                title=f"`{node.name}` is defined but never used",
                detail=f"{'Class' if isinstance(node, ast.ClassDef) else 'Function'} on line {node.lineno} is not referenced anywhere in the repository, tests included.",
                recommendation="Confirm whether the pipeline should call it; remove it or wire it in.",
                file=f.rel, line=node.lineno))
    return out


# ---------------------------------------------------------------- tests

def _decorator_names(fn) -> list[tuple[str, int, str]]:
    out = []
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = dotted(target)
        reason = ""
        if isinstance(d, ast.Call):
            r = kw(d, "reason")
            if isinstance(r, ast.Constant):
                reason = str(r.value)
        out.append((name, d.lineno, reason))
    return out


def check_tests(cb: Codebase) -> list[Finding]:
    out = []
    for f in cb.tests:
        for node in ast.walk(f.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            for name, line, reason in _decorator_names(node):
                if name.endswith(("mark.skip", "mark.skipif", "mark.xfail")):
                    out.append(Finding(
                        check="skipped_test", category="testing", severity="Medium",
                        title=f"Test `{node.name}` is {'skipped' if 'skip' in name else 'expected to fail'}",
                        detail=f"Decorator `{name}` on line {line}" + (f" (reason: \"{reason}\")." if reason else "."),
                        recommendation="Run it; a skipped test can hide a real defect.",
                        file=f.rel, line=line))
            body_mod = ast.Module(body=node.body, type_ignores=[])
            has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(body_mod))
            has_raises = any(isinstance(n, ast.Call) and (call_name(n) in {"raises", "warns", "approx"} or call_name(n).startswith("assert"))
                             for n in ast.walk(body_mod))
            const_assert = [n for n in ast.walk(body_mod) if isinstance(n, ast.Assert) and isinstance(n.test, ast.Constant) and n.test.value]
            if not has_assert and not has_raises:
                out.append(Finding(
                    check="test_without_assertion", category="testing", severity="Medium",
                    title=f"Test `{node.name}` has no assertion",
                    detail=f"Line {node.lineno}: the test passes whenever the code does not crash.",
                    recommendation="Assert expected values from a known-answer case.",
                    file=f.rel, line=node.lineno))
            for n in const_assert:
                out.append(Finding(
                    check="test_constant_assert", category="testing", severity="Medium",
                    title=f"Test `{node.name}` asserts a constant",
                    detail=f"Line {n.lineno} cannot fail.",
                    recommendation="Assert on the computed result.",
                    file=f.rel, line=n.lineno))
            for n in ast.walk(body_mod):
                if isinstance(n, ast.Call) and dotted(n.func) in {"pytest.skip", "pytest.xfail"}:
                    out.append(Finding(
                        check="skipped_test", category="testing", severity="Medium",
                        title=f"Test `{node.name}` skips itself",
                        detail=f"`{dotted(n.func)}` on line {n.lineno}.",
                        recommendation="Run it or document why it cannot run.",
                        file=f.rel, line=n.lineno))
                if isinstance(n, ast.ExceptHandler):
                    t = n.type
                    if t is None or (isinstance(t, ast.Name) and t.id in {"Exception", "BaseException"}):
                        out.append(Finding(
                            check="test_swallows_errors", category="testing", severity="Medium",
                            title=f"Test `{node.name}` catches all exceptions",
                            detail=f"Line {n.lineno}: failures inside the try block are swallowed.",
                            recommendation="Let errors fail the test, or use pytest.raises for the expected one.",
                            file=f.rel, line=n.lineno))
    # source modules without any test importing them
    tested = set()
    for f in cb.tests:
        for node in ast.walk(f.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                tested.add(node.module)
                for a in node.names:
                    tested.add(f"{node.module}.{a.name}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    tested.add(a.name)
    for f in cb.source:
        if f.rel.endswith(("__init__.py", "__main__.py")):
            continue
        code_lines = sum(1 for ln in f.lines if ln.strip() and not ln.strip().startswith("#"))
        if code_lines < 15:
            continue
        if f.module not in tested:
            low = any(s in f.module.lower() for s in ("synth", "fake", "generate", "demo"))
            out.append(Finding(
                check="module_without_tests", category="testing", severity="Low" if low else "Medium",
                title=f"No test imports `{f.module}`",
                detail=f"{code_lines} lines of code in {f.rel} have no direct test.",
                recommendation="Add known-answer tests for the functions whose output reaches a prediction or a report.",
                file=f.rel))
    return out


# ---------------------------------------------------------------- driver

def run_static(cb: Codebase, cfg_files: list[str], target_col: str | None, has_groups: bool) -> list[Finding]:
    np_seeded = any("np.random.seed(" in f.text or "numpy.random.seed(" in f.text for f in cb.source)
    py_seeded = any(re.search(r"\brandom\.seed\(", f.text) for f in cb.source)
    out: list[Finding] = []
    for f in cb.source:
        out += check_fit_before_split(f)
        out += check_ungrouped_split(f, has_groups)
        out += check_target_in_features(f, target_col)
        out += check_seeds(f, np_seeded, py_seeded)
        out += check_wallclock(f)
        out += check_unordered(f)
        out += check_hardcoded_paths(f)
        out += check_broad_except(f)
        out += check_coerce(f)
        out += check_assert_validation(f)
        out += check_model_load(f)
        out += check_threshold_literals(f)
    out += check_config_keys(cb, cfg_files)
    out += check_dead_functions(cb)
    out += check_tests(cb)
    return out
