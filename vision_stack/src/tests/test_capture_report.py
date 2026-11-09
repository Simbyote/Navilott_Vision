"""
test_capture_report.py  --  src/analysis/capture_report.py

A made-up capture-anatomy run: the camera ~11 ms, videoconvert stepping from
14 to 18 ms halfway, and a videoflip probe that reads the frame period. The
report adds up the frame's age and the ARM time without the probe, names the
step and the probe, finds the newest latency.csv, and draws.

--software  Synthetic CSVs. No camera.
"""
import pytest

from src.analysis import capture_report as cr
from src.analysis.common import Table
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_frame_age_arm_time_the_step_and_the_period_probe(tmp_path):
    res = cr.analyze(Table(rf.latency(tmp_path)))
    steps = {s["element"]: s for s in res["steps"]}
    assert res["frame_period_ms"] == pytest.approx(50.0)
    assert steps["videoflip0"]["period_artifact"] and not steps["videoconvert0"]["period_artifact"]
    assert res["arm_ms"] == pytest.approx(steps["videoconvert0"]["median_ms"]), "the probe is no cost"
    assert res["frame_age_ms"] == pytest.approx(11 + 22, abs=0.5)
    assert steps["videoconvert0"]["shift"]["frame"] == 100
    text = " ".join(res["findings"])
    assert "the frame period" in text and "changed level at frame 100" in text and "ARM time" in text


def test_a_cheap_pipeline_has_nothing_to_flag(tmp_path):
    res = cr.analyze(Table(rf.latency(tmp_path, convert=(1.0, 1.0), flip_period=False)))
    assert res["findings"] == []


def test_main_finds_the_newest_and_draws(tmp_path, monkeypatch, capsys):
    rf.latency(tmp_path / "runs" / "anatomy_a")
    other = tmp_path / "runs" / "x" / "latency.csv"
    other.parent.mkdir(parents=True)
    other.write_text("a,b\n1,2\n")
    monkeypatch.setattr(cr.common, "SEARCH_ROOTS", (tmp_path / "runs",))
    assert cr.find(None).parent.name == "anatomy_a"
    assert cr.main([]) == 0
    assert (tmp_path / "runs" / "anatomy_a" / "capture_report.json").is_file()
    if cr.common.pyplot() is not None:
        assert (tmp_path / "runs" / "anatomy_a" / "capture_report.png").is_file()
    assert "frame period 50.0 ms" in capsys.readouterr().out
    assert cr.main([str(other)]) == 1
