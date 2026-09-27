"""
test_detection_range.py  --  src/analysis/detection_range.py

--software  Detection columns from p3.csv and fusion_timing.csv, rates and
            miss streaks, the reliable range and its edge cases, target
            selection, manifest and --at inputs, and the command line.
"""
import csv

import numpy as np
import pytest

from src.analysis import detection_range as dr
from src.analysis.common import Table


def write_p3(path, stop_hits, traffic_hits=None):
    """p3.csv with p2_stop / p2_traffic counts from 0/1 lists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    traffic_hits = traffic_hits if traffic_hits is not None else [0] * len(stop_hits)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "p2_detections", "p2_traffic", "p2_stop"])
        for i, (s, t) in enumerate(zip(stop_hits, traffic_hits)):
            w.writerow([i, s + t, t, s])
    return path


def hits(rate, n=20):
    """n frames detected at the given rate, misses together at the end."""
    k = round(rate * n)
    return [1] * k + [0] * (n - k)


@pytest.mark.software
def test_p3_and_fusion_columns_are_read(tmp_path):
    p3 = Table(write_p3(tmp_path / "p3.csv", [1, 0, 2], [0, 0, 1]))
    d = dr.detections(p3, dr.TARGETS)
    assert d["stop_sign"].tolist() == [True, False, True] and d["traffic_light"].tolist() == [False, False, True]
    fusion = tmp_path / "fusion_timing.csv"
    fusion.write_text("frame_id,n_traffic_light,n_lane_boundary,n_stop_sign\n0,0,2,1\n1,0,2,0\n")
    assert dr.detections(Table(fusion), ["stop_sign"])["stop_sign"].tolist() == [True, False]


@pytest.mark.software
def test_rates_miss_streaks_and_reliable_range():
    points = [(d, {"stop_sign": np.array(hits(r), bool)})
              for d, r in ((20, 1.0), (40, 0.95), (60, 0.9), (80, 0.5), (100, 0.0))]
    r = dr.analyze(points)["stop_sign"]
    assert [x["rate"] for x in r["distances"]] == pytest.approx([1.0, 0.95, 0.9, 0.5, 0.0])
    assert r["distances"][3]["longest_miss"] == 10
    assert r["reliable_range_cm"] == 60 and r["farthest_seen_cm"] == 80


@pytest.mark.software
def test_reliable_range_stops_at_the_first_gap():
    points = [(d, {"stop_sign": np.array(hits(r), bool)}) for d, r in ((20, 1.0), (40, 0.6), (60, 1.0))]
    r = dr.analyze(points)["stop_sign"]
    assert r["reliable_range_cm"] == 20 and r["reliable_distances_cm"] == [20, 60]


@pytest.mark.software
def test_failing_at_the_nearest_distance_has_no_reliable_range():
    points = [(d, {"stop_sign": np.array(hits(r), bool)}) for d, r in ((10, 0.2), (30, 1.0))]
    assert dr.analyze(points)["stop_sign"]["reliable_range_cm"] is None


@pytest.mark.software
def test_distances_are_sorted():
    points = [(60, {"stop_sign": np.ones(5, bool)}), (20, {"stop_sign": np.ones(5, bool)})]
    assert [x["distance_cm"] for x in dr.analyze(points)["stop_sign"]["distances"]] == [20, 60]


@pytest.mark.software
def test_no_targets_or_no_points_is_an_error():
    with pytest.raises(ValueError):
        dr.analyze([])
    with pytest.raises(ValueError, match="columns"):
        dr.analyze([(20, {})])


@pytest.mark.software
def test_cli_reports_only_targets_that_were_seen(tmp_path, capsys):
    for d, r in ((20, 1.0), (50, 0.95), (80, 0.3)):
        write_p3(tmp_path / f"d{d}" / "p3.csv", hits(r))
    (tmp_path / "distances.csv").write_text("distance_cm,run\n20,d20\n50,d50\n80,d80\n")
    assert dr.main([str(tmp_path / "distances.csv"), "--skip", "0"]) == 0
    out = capsys.readouterr().out
    assert "stop sign:" in out and "traffic light" not in out and "reliable to 50 cm" in out
    assert (tmp_path / "detection_range.json").exists()


@pytest.mark.software
def test_cli_with_nothing_detected_reports_every_target(tmp_path, capsys):
    write_p3(tmp_path / "a" / "p3.csv", [0] * 10)
    assert dr.main(["--at", "30", str(tmp_path / "a"), "--skip", "0", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "stop sign:" in out and "traffic light:" in out and "not reliable" in out


@pytest.mark.software
def test_cli_explicit_targets(tmp_path, capsys):
    write_p3(tmp_path / "a" / "p3.csv", [1] * 10, [1] * 10)
    dr.main(["--at", "30", str(tmp_path / "a"), "--targets", "traffic_light", "--skip", "0",
             "--out", str(tmp_path)])
    assert "stop sign" not in capsys.readouterr().out


@pytest.mark.software
def test_cli_rejects_a_bad_manifest(tmp_path, capsys):
    (tmp_path / "m.csv").write_text("cm,run\n20,a\n")
    assert dr.main([str(tmp_path / "m.csv")]) == 1
    assert "distance_cm,run" in capsys.readouterr().err


@pytest.mark.software
def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    points = [(d, {"stop_sign": np.array(hits(r), bool), "traffic_light": np.array(hits(r / 2), bool)})
              for d, r in ((20, 1.0), (40, 0.95), (60, 0.7))]
    assert dr.figure(dr.analyze(points), 0.9, "t", tmp_path / "r.png").stat().st_size > 0
