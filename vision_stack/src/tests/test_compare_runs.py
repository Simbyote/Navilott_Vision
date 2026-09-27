"""
test_compare_runs.py  --  src/analysis/compare_runs.py

--software  Flattening rules, change flags and ordering, one-sided keys,
            folder matching, commit reporting and the command line.
"""
import json

import pytest

from src.analysis import compare_runs as cr


@pytest.mark.software
def test_flatten_keeps_numbers_under_dotted_keys():
    flat = cr.flatten({"stage_ms": {"p50": 3, "p95": 4.5}, "mode": "none", "ok": True,
                       "gates": [{"count": 2}], "frames": list(range(20))})
    assert flat == {"stage_ms.p50": 3.0, "stage_ms.p95": 4.5, "gates.0.count": 2.0}


@pytest.mark.software
def test_changes_past_the_threshold_are_flagged_largest_first():
    rows = cr.compare({"a": 10, "b": 10, "c": 10}, {"a": 10.5, "b": 20, "c": 13}, threshold_pct=10)
    assert [(r["key"], r["flag"]) for r in rows] == [("b", True), ("c", True), ("a", False)]
    assert rows[0]["pct"] == pytest.approx(100.0)


@pytest.mark.software
def test_a_key_on_one_side_only_is_flagged():
    rows = cr.compare({"a": 1}, {"b": 2})
    assert all(r["flag"] and r["delta"] is None for r in rows)


@pytest.mark.software
def test_a_change_from_zero_has_no_percent_but_is_flagged():
    (row,) = cr.compare({"a": 0}, {"a": 3})
    assert row["pct"] is None and row["flag"]


@pytest.mark.software
def test_folders_match_by_relative_path_and_skip_meta(tmp_path):
    for side, value in (("A", 1), ("B", 2)):
        d = tmp_path / side / "test_geometry"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"x": value}))
        (tmp_path / side / "run_meta.json").write_text(json.dumps({"git": {"commit": side, "dirty": False}}))
    (tmp_path / "A" / "only_here.json").write_text("{}")
    matched = cr.pairs(tmp_path / "A", tmp_path / "B")
    assert [label for label, _, _ in matched] == ["test_geometry/summary.json"]
    assert cr.commit_of(tmp_path / "A") == "A"


@pytest.mark.software
def test_cli_writes_compare_csv(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps({"stage_ms": {"p50": 5.0}}))
    (tmp_path / "b.json").write_text(json.dumps({"stage_ms": {"p50": 4.0}}))
    assert cr.main([str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--out", str(tmp_path)]) == 0
    assert "stage_ms.p50" in capsys.readouterr().out
    assert (tmp_path / "compare.csv").read_text().count("\n") == 2


@pytest.mark.software
def test_cli_rejects_mixed_inputs(tmp_path, capsys):
    (tmp_path / "a.json").write_text("{}")
    assert cr.main([str(tmp_path / "a.json"), str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().err
