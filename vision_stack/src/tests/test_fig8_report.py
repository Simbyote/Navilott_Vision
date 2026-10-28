"""
test_fig8_report.py  --  src/analysis/fig8_report.py

A made-up figure-eight attempt written by the routine's own EightWatch: three
laps, the chassis sitting still at the start of corner 3 so those turns end on
the time limit. The report finds the corner, the slips there, the lap times,
a touch, and set against a base run where corner 2 was slow.

--software  Synthetic CSVs through EightWatch.
"""
import pytest

from src.analysis import fig8_report as f8
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_the_slow_corner_its_slips_and_the_turn_curves(tmp_path):
    a = rf.figure_eight(tmp_path / "routine_figure-eight_x" / "attempt_01", touch_frame=500)
    res = f8.analyze(a)
    assert res["turns"] == 24 and res["time_limit_positions"] == [3]
    assert len(res["time_limit_steps"]) == 3 and res["on_gyro"] == 21
    assert res["slips"] >= 3 and next(iter(res["slip_spots"])).startswith("turn 3")
    assert len(res["laps"]) == 2 and res["touches"] == 1
    assert "position 3 of the lap: the same corner every time" in " ".join(res["findings"])
    turns = f8.turns_of(f8.Table(a / "nav.csv"))
    assert len(turns) == 24 and turns[0]["t"][0] == 0.0 and turns[0]["heading"][-1] <= -85


def test_against_a_base_run_and_main(tmp_path, capsys):
    new = rf.figure_eight(tmp_path / "routine_figure-eight_new" / "attempt_01")
    old = rf.figure_eight(tmp_path / "routine_figure-eight_old" / "attempt_01", slow_corner=2, laps=2)
    assert f8.find(str(new.parent)) == new
    res = f8.analyze(new, old)
    assert res["base"]["time_limit_positions"] == [2]
    assert "against the base run: time-limit turns 2 -> 3" in " ".join(res["findings"])
    assert f8.main([str(new.parent), "--base", str(old.parent)]) == 0
    assert (new / "fig8_report.json").is_file()
    if f8.common.pyplot() is not None:
        assert (new / "fig8_report.png").is_file()
    assert f8.main([str(tmp_path / "nothing")]) == 1
