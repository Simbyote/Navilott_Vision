"""
test_state_timeline.py  --  src/analysis/state_timeline.py

--software  Shares, episodes, dwell in frames and ms, transition pairs,
            duration sources, field selection and the command line.
"""
import csv

import numpy as np
import pytest

from src.analysis import state_timeline as stl
from src.analysis.common import Table


def write_p3(path, lane, drive, dt_s=0.05):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp_ms", "dt_s", "lane_status", "drive_state", "stop_sign_detected"])
        for i, (a, b) in enumerate(zip(lane, drive)):
            w.writerow([i, i * 50, dt_s, a, b, 0])
    return path


@pytest.mark.software
def test_episodes_dwell_and_transitions():
    values = ["vision"] * 4 + ["hold"] * 2 + ["vision"] * 3 + ["stale"]
    r = stl.analyze_field(values, np.full(len(values), 50.0))
    v = r["values"]["vision"]
    assert v["frames"] == 7 and v["episodes"] == 2
    assert v["dwell_frames"] == {"median": 3.5, "max": 4}
    assert v["dwell_ms"]["max"] == pytest.approx(200.0)
    assert r["transitions"]["count"] == 3
    assert r["transitions"]["pairs"] == {"vision -> hold": 1, "hold -> vision": 1, "vision -> stale": 1}


@pytest.mark.software
def test_durations_use_intervals_and_fill_the_first_frame(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("frame_id,timestamp_ms,s\n0,0,a\n1,40,a\n2,100,a\n")
    assert stl.frame_ms(Table(path), 20).tolist() == [50.0, 40.0, 60.0]


@pytest.mark.software
def test_durations_fall_back_to_fps(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("s\na\nb\n")
    assert stl.frame_ms(Table(path), 20).tolist() == [50.0, 50.0]


@pytest.mark.software
def test_missing_fields_are_skipped_and_none_present_is_an_error(tmp_path):
    t = Table(write_p3(tmp_path / "p3.csv", ["vision"] * 3, ["GO"] * 3))
    assert set(stl.analyze(t, stl.DEFAULT_FIELDS)) == {"lane_status", "drive_state", "stop_sign_detected"}
    with pytest.raises(ValueError, match="none of"):
        stl.analyze(t, ["nope"])


@pytest.mark.software
def test_cli(tmp_path, capsys):
    write_p3(tmp_path / "run" / "p3.csv", ["vision"] * 5 + ["hold"] * 2, ["GO"] * 4 + ["STOP"] * 3)
    assert stl.main([str(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    assert "vision -> hold: 1" in out and (tmp_path / "run" / "state_timeline.json").exists()


@pytest.mark.software
def test_cli_with_custom_fields_on_any_csv(tmp_path):
    path = tmp_path / "stages.csv"
    path.write_text("frame_id,mode\n0,none\n1,left_only\n")
    assert stl.main([str(path), "--fields", "mode"]) == 0


@pytest.mark.software
def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    path = write_p3(tmp_path / "p3.csv", ["vision", "hold", "vision"], ["GO", "GO", "STOP"])
    t = Table(path)
    assert stl.figure(t, stl.analyze(t, stl.DEFAULT_FIELDS), "t", tmp_path / "s.png").stat().st_size > 0


def write_nav(path, n=20, step_s=0.05, stage_from=8, stage_to=14):
    """A small nav.csv: timings, t, and rule / stage / lane / light columns, stage blank outside 8-13."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "t", "capture_ms", "phase2_ms", "phase3_ms", "nav_ms", "latency_ms",
                    "rule", "stage", "lane_status", "drive_state"])
        for i in range(n):
            inside = stage_from <= i < stage_to
            w.writerow([i, round(i * step_s, 3), 2.0, 20.0, 0.5, 0.2, 23.0,
                        "intersection" if inside else "lane_keeping", "to_line" if inside else "",
                        "stale" if inside else "vision", "go"])
    return path


@pytest.mark.software
def test_a_nav_csv_follows_rule_and_stage_with_blank_stages_labeled(tmp_path, capsys):
    path = write_nav(tmp_path / "nav.csv")
    (tmp_path / "p3.csv").write_text("frame_id,lane_status\n0,vision\n")
    assert stl.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "nav.csv" in out and "rule:" in out and "stage:" in out
    res = stl.analyze(Table(path), stl.NAV_DEFAULT_FIELDS)
    assert set(res["stage"]["values"]) == {stl.NONE, "to_line"}
    assert res["stage"]["values"]["to_line"]["dwell_ms"]["max"] == pytest.approx(300.0)
    assert res["rule"]["transitions"]["pairs"] == {"lane_keeping -> intersection": 1, "intersection -> lane_keeping": 1}
