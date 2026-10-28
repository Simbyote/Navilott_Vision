"""
test_sched_report.py  --  src/analysis/sched_report.py

Two made-up sched-latency folders, idle and under load. Each case's
percentiles against the IMU period, the loaded worst against the idle one,
real-time priority's cut, the figure, and a folder without the JSON.

--software  Synthetic folders. No kernel.
"""
import pytest

from src.analysis import sched_report as sr
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_against_the_deadlines_and_the_base_run(tmp_path):
    load = rf.sched(tmp_path / "loaded", "loaded")
    idle = rf.sched(tmp_path / "idle", "idle", worst=(173, 2434, 69))
    res = sr.analyze(sr.load(load)[0], sr.load(idle)[0])
    py = res["cases"]["python"]
    assert py["max"] == 5332 and py["base_max"] == 173
    assert py["max_share"]["IMU period"] == pytest.approx(0.5332)
    text = " ".join(res["findings"])
    assert "53% of the 10 ms IMU period" in text and "31x the base run's" in text and "real-time priority cut" in text


def test_main_draws_and_refuses_a_folder_without_its_json(tmp_path, capsys):
    load = rf.sched(tmp_path / "loaded")
    assert sr.main([str(load)]) == 0
    assert (load / "sched_report.json").is_file()
    if sr.common.pyplot() is not None:
        assert (load / "sched_report.png").is_file()
    assert "Python thread, normal" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert sr.main([str(tmp_path / "empty")]) == 1
