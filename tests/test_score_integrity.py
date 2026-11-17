from mlaudit import integrity, score


def _d(i, cat, file, a, b):
    return {"id": i, "category": cat, "intended_severity": "High", "file": file, "line_start": a, "line_end": b,
            "description": "", "how_detectable": "static"}


def _f(i, cat, file=None, line=None, kind="static", check="x"):
    return {"id": i, "category": cat, "file": file, "line": line, "kind": kind, "check": check,
            "severity": "High", "title": ""}


def test_located_window_and_category():
    d = [_d("D1", "leakage", "a.py", 10, 12)]
    assert score.score(d, [_f("F1", "leakage", "a.py", 15)], [])["detected"] == 1
    assert score.score(d, [_f("F1", "leakage", "a.py", 16)], [])["detected"] == 0
    assert score.score(d, [_f("F1", "testing", "a.py", 11)], [])["detected"] == 0


def test_one_to_one_and_located_first():
    d = [_d("D1", "leakage", "a.py", 10, 10), _d("D2", "leakage", "b.py", 5, 5)]
    f = [_f("F1", "leakage", kind="dynamic"), _f("F2", "leakage", "a.py", 10)]
    s = score.score(d, f, [])
    rows = {r["id"]: r for r in s["rows"]}
    assert rows["D1"]["match_type"] == "located" and rows["D2"]["match_type"] == "unlocated_dynamic"
    assert score.score(d, [_f("F1", "leakage", kind="dynamic")], [])["detected"] == 1


def test_baseline_ignores_category():
    d = [_d("D1", "leakage", "a.py", 10, 10)]
    assert score.score(d, [], [_f("B1", "baseline", "a.py", 9)])["baseline_detected"] == 1


def test_integrity_compare():
    a = {"x": "1", "y": "2"}
    assert integrity.compare(a, dict(a))["unchanged"]
    c = integrity.compare(a, {"x": "1", "y": "3", "z": "4"})
    assert c["changed"] == ["y"] and c["added"] == ["z"] and not c["unchanged"]
