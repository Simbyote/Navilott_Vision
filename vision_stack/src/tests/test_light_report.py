"""
test_light_report.py  --  src/analysis/light_report.py

A made-up run: a light green, yellow, red under a 100-frame hold, every 9th
frame finding nothing, two lone reds. Holds, the no-light share, the strays
the vote filtered, and a run without nav_video.csv (the vote alone).

--software  Synthetic CSVs.
"""
import pytest

from src.analysis import light_report as lr
from src.analysis.common import Table
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_holds_strays_and_frames_with_no_light(tmp_path):
    run = rf.nav_run(tmp_path)
    res = lr.analyze(Table(run / "nav.csv"), Table(run / "nav_video.csv"))
    assert len(res["holds"]) == 1 and res["holds"][0]["frames"] == 100
    assert res["no_light_share"] == pytest.approx(1 / 9, abs=0.01)
    assert res["stray_reds"] == 2
    assert res["vote_flips"] < res["raw_flips"]
    text = " ".join(res["findings"])
    assert "no light passed the gates on 11%" in text and "2 red readings" in text


def test_without_the_video_csv_only_the_vote_and_the_braking(tmp_path, capsys):
    run = rf.nav_run(tmp_path, video=False, hold=(10_000, 10_001))
    res = lr.analyze(Table(run / "nav.csv"))
    assert not res["has_raw"] and "stray_reds" not in res
    assert "never braked" in " ".join(res["findings"])
    assert lr.main([str(run)]) == 0
    assert (run / "light_report.json").is_file()
