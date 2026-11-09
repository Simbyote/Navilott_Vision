"""
test_loop_budget.py  --  src/analysis/loop_budget.py

A made-up navigation run at ~16.5 FPS. The phases add up to the mean frame,
the time outside the timers is what the interval leaves, Phase 2 splits by
summary.txt's stages and by lane mode, frame 0 is left out, and the findings
name the overrun and Phase 2's share.

--software  Synthetic CSVs.
"""
import pytest

from src.analysis import loop_budget as lb
from src.analysis.common import Table
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_the_phases_add_up_and_outside_is_the_rest(tmp_path):
    run = rf.nav_run(tmp_path)
    res = lb.analyze(Table(run / "nav.csv"), lb.parse_timing((run / "summary.txt").read_text()))
    p = res["phases"]
    assert sum(x["mean"] for x in p.values()) == pytest.approx(res["mean_frame"], rel=1e-6)
    assert p["outside_ms"]["median"] == pytest.approx(6.0, abs=0.01)
    assert p["phase2_ms"]["max"] < 100, "frame 0's 1.3 s start-up is left out"
    assert [s for s, _ in res["phase2_split"]] == ["preprocess", "geometry", "color", "obstacle"]
    assert sum(w for _, w in res["phase2_split"]) == pytest.approx(p["phase2_ms"]["mean"])
    assert res["phase2_by_mode"]["two_boundary"]["median"] == pytest.approx(50.0, abs=0.5)
    assert res["phase2_by_mode"]["left_only"]["median"] == pytest.approx(43.0, abs=0.5)
    assert res["gc"] == 4
    text = " ".join(res["findings"])
    assert "over the 50 ms budget" in text and "Phase 2 is" in text


def test_timing_blocks_and_a_csv_that_isnt_navs(tmp_path):
    assert lb.parse_timing("no block") == {}
    assert lb.parse_timing(rf.SUMMARY)["geometry.edges"] == (5.0, 6.0)
    (tmp_path / "nav.csv").write_text("frame_id,x\n0,1\n1,2\n")
    with pytest.raises(ValueError, match="no t"):
        lb.analyze(Table(tmp_path / "nav.csv"))


def test_main_draws_beside_the_run(tmp_path, capsys):
    run = rf.nav_run(tmp_path / "nav_x")
    assert lb.main([str(run)]) == 0
    assert (run / "loop_budget.json").is_file()
    if lb.common.pyplot() is not None:
        assert (run / "loop_budget.png").is_file()
    assert "Outside the timers" in capsys.readouterr().out
