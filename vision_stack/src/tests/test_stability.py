"""
test_stability.py  --  src/analysis/stability.py

--software  Column selection for p3.csv, lane_offset_timing.csv and
            stages.csv; noise, mode shares, transitions, flickers and blind
            runs on hand-built sequences; the command line.
"""
import csv

import numpy as np
import pytest

from src.analysis import stability
from src.analysis.common import Table


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.mark.software
def test_p3_uses_cm_on_vision_frames_only(tmp_path):
    rows = [[0, "two_boundary", 0.1, 1.0, "vision"], [1, "none", 0.0, 1.0, "hold"],
            [2, "two_boundary", 0.2, 2.0, "vision"]]
    t = Table(write_csv(tmp_path / "p3.csv", ["frame_id", "p2_mode", "p2_offset", "lane_offset_cm",
                                              "lane_status"], rows))
    offset, measured, modes, unit = stability.load(t)
    assert unit == "cm" and measured.tolist() == [True, False, True]
    assert offset[measured].tolist() == [1.0, 2.0]


@pytest.mark.software
def test_p3_without_a_cm_scale_falls_back_to_normalized(tmp_path):
    rows = [[0, "two_boundary", 0.1, "None", "vision"]]
    t = Table(write_csv(tmp_path / "p3.csv", ["frame_id", "p2_mode", "p2_offset", "lane_offset_cm",
                                              "lane_status"], rows))
    assert stability.load(t)[3] == "normalized"


@pytest.mark.software
def test_normalized_flag_ignores_cm(tmp_path):
    rows = [[0, "two_boundary", 0.1, 1.0, "vision"]]
    t = Table(write_csv(tmp_path / "p3.csv", ["frame_id", "p2_mode", "p2_offset", "lane_offset_cm",
                                              "lane_status"], rows))
    offset, _, _, unit = stability.load(t, normalized=True)
    assert unit == "normalized" and offset[0] == pytest.approx(0.1)


@pytest.mark.software
def test_lane_offset_csv_counts_only_measuring_modes(tmp_path):
    rows = [[0, "two_boundary", 0.1], [1, "single_uncalibrated", 0.0], [2, "none", 0.0]]
    t = Table(write_csv(tmp_path / "lane_offset_timing.csv", ["frame_id", "mode", "offset"], rows))
    assert stability.load(t)[1].tolist() == [True, False, False]


@pytest.mark.software
def test_a_table_without_modes_is_rejected(tmp_path):
    t = Table(write_csv(tmp_path / "x.csv", ["frame_id", "offset"], [[0, 0.1]]))
    with pytest.raises(ValueError, match="mode"):
        stability.load(t)


@pytest.mark.software
def test_noise_is_measured_on_measured_frames_only():
    offset = np.array([1.0, 3.0, 99.0, 1.0, 3.0])
    measured = np.array([True, True, False, True, True])
    r = stability.analyze(offset, measured, ["two_boundary"] * 5)
    assert r["offset"]["mean"] == pytest.approx(2.0) and r["offset"]["std"] == pytest.approx(1.0)
    assert r["measured"]["frames"] == 4


@pytest.mark.software
def test_transitions_flickers_and_longest_blind_run():
    modes = ["two_boundary"] * 5 + ["left_only"] + ["two_boundary"] * 5 + ["none"] * 4 + ["two_boundary"] * 5
    measured = np.array([m != "none" for m in modes])
    r = stability.analyze(np.zeros(len(modes)), measured, modes)
    assert r["transitions"]["count"] == 4
    assert r["flickers"] == 1                          # the single left_only frame
    assert r["longest_unmeasured_run"] == 4
    assert r["modes"]["two_boundary"]["frames"] == 15


@pytest.mark.software
def test_a_run_with_no_measured_frames_still_reports():
    r = stability.analyze(np.zeros(3), np.zeros(3, bool), ["none"] * 3)
    assert r["offset"]["mean"] is None and r["offset"]["p5_p95_span"] is None


@pytest.mark.software
def test_cli(tmp_path, capsys):
    rows = [[i, "two_boundary", 0.01 * (i % 3)] for i in range(30)]
    write_csv(tmp_path / "run" / "lane_offset_timing.csv", ["frame_id", "mode", "offset"], rows)
    assert stability.main([str(tmp_path / "run"), "--skip", "0"]) == 0
    assert (tmp_path / "run" / "stability.json").exists()
    assert "transitions" in capsys.readouterr().out


@pytest.mark.software
def test_cli_writes_the_figure(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [[i, "two_boundary" if i % 7 else "none", 0.01 * (i % 3)] for i in range(30)]
    write_csv(tmp_path / "run" / "stages.csv", ["frame_id", "mode", "offset"], rows)
    stability.main([str(tmp_path / "run"), "--skip", "0"])
    assert (tmp_path / "run" / "stability.png").stat().st_size > 0
