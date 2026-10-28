"""
test_motor_health.py  --  src/analysis/motor_health.py, and its place in nav-report and fig8-report

A made-up run whose right motor weakens 25% at 60 s and stops at 100 s while
the PI pushes it to full duty: the report finds both, from the first window
after each, and leaves the healthy left wheel alone. Starts from rest, braked
frames and low duties aren't stalls. Both reports carry its findings.

--software  Synthetic nav.csv (run_fixtures.dying_motor).
"""
import shutil

import pytest

from src.analysis import fig8_report as f8, motor_health as mh, nav_run as nr
from src.analysis.common import Table
from src.navigation.wheel_stall import MIN_DUTY
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_a_weakening_then_dead_motor_is_found_and_the_healthy_one_left_alone(tmp_path):
    res = mh.analyze(Table(rf.dying_motor(tmp_path)))
    left, right = res["left"], res["right"]
    assert left["baseline"] == pytest.approx(3300, rel=0.01) and right["baseline"] == pytest.approx(3300, rel=0.01)
    assert left["weak_from_s"] is None and left["stopped_s"] == 0.0
    assert 60.0 < right["weak_from_s"] <= 60.0 + mh.WINDOW_S
    assert 100.0 < right["stopped_from_s"] <= 101.1 and right["stopped_s"] == pytest.approx(19, abs=0.2)
    assert right["worst"]["share"] == 0.0
    text = " ".join(res["findings"])
    assert "the right wheel weakened from 62 s" in text and "driven but not turning for 18.9 s, from 101 s" in text
    assert "left" not in text


def test_a_healthy_run_has_no_findings_starts_from_rest_included(tmp_path):
    res = mh.analyze(Table(rf.dying_motor(tmp_path, weak_at=1e9, dead_at=1e9)))
    assert res["findings"] == [] and res["left"]["stopped_s"] == res["right"]["stopped_s"] == 0.0


def test_only_frames_driven_at_min_duty_count():
    import numpy as np
    t = np.arange(200) * 0.05
    low = mh.wheel(t, np.full(200, MIN_DUTY - 0.01), np.zeros(200), np.zeros(200))
    braked = mh.wheel(t, np.full(200, 0.8), np.zeros(200), np.ones(200))
    assert low["windows"] == braked["windows"] == [] and low["baseline"] is None


def test_a_motor_that_started_weaker_is_named_and_a_stall_guard_trip_reported():
    res = {"left": {"baseline": 3300.0, "weak_from_s": None, "stopped_s": 0.0},
           "right": {"baseline": 2800.0, "weak_from_s": None, "stopped_s": 0.0}, "stall_guard_s": 41.0}
    text = " ".join(mh.findings(res))
    assert "the right motor started 15% weaker" in text and "the stall guard ended the run at 41 s" in text


def test_main_writes_beside_the_run_and_refuses_another_csv(tmp_path, capsys):
    rf.dying_motor(tmp_path)
    assert mh.main([str(tmp_path)]) == 0
    assert (tmp_path / "motor_health.json").is_file()
    if mh.common.pyplot() is not None:
        assert (tmp_path / "motor_health.png").is_file()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "nav.csv").write_text("t,x\n0,1\n")
    assert mh.main([str(tmp_path / "other")]) == 1


def test_nav_report_and_fig8_report_carry_it(tmp_path):
    res = nr.analyze(Table(rf.dying_motor(tmp_path / "nav")))
    assert res["motor_health"]["right"]["weak_from_s"] is not None
    assert any("the right wheel weakened" in f for f in res["findings"])
    assert any(x.startswith("motor health  right") for x in nr.report_lines(res))
    if nr.common.pyplot() is not None:
        assert nr.figure(Table(tmp_path / "nav" / "nav.csv"), res, "t", tmp_path / "n.png").is_file()
    a = rf.figure_eight(tmp_path / "routine_figure-eight_x" / "attempt_01")
    assert f8.analyze(a)["motor_health"]["right"]["baseline"] is None      # the fixture logs no duties
    shutil.copy(tmp_path / "nav" / "nav.csv", a / "nav.csv")
    res = f8.analyze(a)
    assert any("the right wheel weakened" in f for f in res["findings"])
    if f8.common.pyplot() is not None:
        assert f8.figure(a, res, None, "t", tmp_path / "f.png").is_file()
