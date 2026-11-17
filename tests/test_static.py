import textwrap
from pathlib import Path

from mlaudit import codebase, deps, static


def _cb(tmp_path: Path, files: dict) -> codebase.Codebase:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return codebase.load(tmp_path)


def _checks(findings):
    return sorted(f.check for f in findings)


def test_fit_before_split_flagged_and_pipeline_not(tmp_path):
    cb = _cb(tmp_path, {"pkg/__init__.py": "", "pkg/train.py": """
        from sklearn.model_selection import train_test_split
        def bad(X, y, scaler):
            X = scaler.fit_transform(X)
            return train_test_split(X, y, random_state=1)
        def good(X, y, model):
            a, b, c, d = train_test_split(X, y, random_state=1)
            model.fit(a, c)
            return " a,b ".split(",")
        """})
    f = static.run_static(cb, [], None, has_groups=False)
    fit = [x for x in f if x.check == "fit_before_split"]
    assert len(fit) == 1 and fit[0].line == 4


def test_ungrouped_split_only_when_groups_exist(tmp_path):
    cb = _cb(tmp_path, {"m.py": "from sklearn.model_selection import train_test_split\ntrain_test_split([1, 2], random_state=0)\n"})
    assert "ungrouped_split" in _checks(static.run_static(cb, [], None, has_groups=True))
    assert "ungrouped_split" not in _checks(static.run_static(cb, [], None, has_groups=False))


def test_seed_rules(tmp_path):
    cb = _cb(tmp_path, {"m.py": """
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        a = RandomForestClassifier()
        b = RandomForestClassifier(random_state=3)
        c = np.random.default_rng()
        d = np.random.default_rng(4)
        e = np.random.shuffle([1])
        """})
    f = [x for x in static.run_static(cb, [], None, False) if x.category == "reproducibility"]
    assert sorted(x.line for x in f) == [4, 6, 8]


def test_set_ordering_and_sorted_glob(tmp_path):
    cb = _cb(tmp_path, {"m.py": """
        import glob
        cols = list({"a", "b"})
        files = sorted(glob.glob("*.csv"))
        other = glob.glob("*.txt")
        """})
    f = static.run_static(cb, [], None, False)
    assert {(x.check, x.line) for x in f if x.category == "reproducibility"} == {("set_ordering", 3), ("unordered_listing", 5)}


def test_broad_except_without_raise(tmp_path):
    cb = _cb(tmp_path, {"m.py": """
        def a():
            try:
                pass
            except Exception:
                return 0
        def b():
            try:
                pass
            except Exception:
                raise
        def c():
            try:
                pass
            except ValueError:
                return 0
        """})
    f = [x for x in static.run_static(cb, [], None, False) if x.check == "broad_except"]
    assert [x.line for x in f] == [5]


def test_config_keys(tmp_path):
    cb = _cb(tmp_path, {"config.yaml": "model:\n  alert_threshold: 0.3\n  depth: 3\nseed: 1\n", "m.py": """
        def f(cfg):
            m = cfg["model"]
            t = cfg["model"].get("threshold", 0.5)
            return m["depth"], cfg["seed"], t
        """})
    f = static.check_config_keys(cb, ["config.yaml"])
    assert {(x.check, x.title.split("`")[1]) for x in f} == {
        ("config_key_missing", "model.threshold"), ("config_key_unused", "model.alert_threshold")}


def test_tests_without_assertions_and_skips(tmp_path):
    cb = _cb(tmp_path, {"pkg/core.py": "def f():\n    return 1\n", "tests/test_core.py": """
        import pytest
        from pkg.core import f
        def test_ok():
            assert f() == 1
        def test_no_assert():
            f()
        @pytest.mark.skip(reason="later")
        def test_skipped():
            assert f() == 2
        def test_raises():
            with pytest.raises(TypeError):
                f(1)
        """})
    f = static.check_tests(cb)
    assert _checks(f) == ["skipped_test", "test_without_assertion"]


def test_dependencies(tmp_path):
    cb = _cb(tmp_path, {"requirements.txt": "numpy==2.0.0\npandas>=2\nrequests==2.0\n", "m.py": "import numpy, pandas, yaml\n"})
    f, info = deps.run_deps(cb)
    got = {(x.check, x.title.split("`")[1]) for x in f if "`" in x.title}
    assert ("unpinned_dependency", "pandas") in got
    assert ("undeclared_dependency", "yaml") in got
    assert ("unused_dependency", "requests") in got
    assert ("unpinned_dependency", "numpy") not in got
