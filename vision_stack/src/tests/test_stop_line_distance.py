"""
test_stop_line_distance.py  --  src/perception/stop_line_distance.py

The measurement is tested on hand-built stop-line candidates, so every
distance has a known answer, then chained from drawn frames through the real
geometry stage. Its debug and production versions are held to each other.

--software  Distance from the lane ROI bottom, nearest-line choice, the
            confidence gate, the nothing-found result, stamps, and the
            p3.csv columns phase3_linker writes from it. No camera.
"""
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.phase3_linker as p3
from src.params import STOP_LINE
from src.perception.geometry import GeometryBranchResult, StopLineCandidate
from src.perception.stop_line_distance import (
    StopLineDistanceConfig, StopLineResult, compute_stop_line_distance, estimate_stop_line_distance,
)
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, SCENES, SYNTHETIC_GROUND, SYNTHETIC_STOP_LINE_TABLE, same

ROI_H = 81
ROI = SimpleNamespace(lane_rect=(24, 189, 432, ROI_H), frame_id=7, timestamp_ms=350, source_shape=(270, 480))


def line(y_near, confidence=0.8, clipped=False, x=(100.0, 300.0), tilt=0.0):
    """A stop-line candidate whose nearest point is y_near."""
    return StopLineCandidate(
        label=STOP_LINE, bbox=(int(x[0]), int(y_near) - 8, int(x[1] - x[0]), 8),
        x_left=x[0], x_right=x[1], y_top_px=y_near - 8, y_bottom_px=y_near, y_near_px=y_near,
        tilt_deg=tilt, length_px=x[1] - x[0], thickness_px=8.0, mean_intensity=220.0,
        clipped=clipped, confidence=confidence, frame_id=7, timestamp_ms=350,
        proximity=round(y_near / ROI_H, 4))

def geometry(*lines, frame_id=7, ts=350):
    return GeometryBranchResult([], [], frame_id, ts, list(lines))

def both(geo, cfg=StopLineDistanceConfig(), ground=None, roi=ROI, table=None):
    """The debug and production results, checked equal."""
    debug, summary = compute_stop_line_distance(geo, roi, cfg, ground, table)
    same(debug, estimate_stop_line_distance(geo, roi, cfg, ground, table))
    return debug, summary


# =============================================================================
# Measurement
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("y_near, distance", [(40.0, 41.0), (80.5, 0.5), (0.0, 81.0)])
def test_distance_is_rows_from_the_line_to_the_lane_roi_bottom(y_near, distance):
    r, _ = both(geometry(line(y_near)))
    assert r.detected and r.distance_px == distance and r.y_near_px == y_near


@pytest.mark.software
def test_a_line_the_robot_is_on_is_at_distance_zero():
    r, _ = both(geometry(line(float(ROI_H), clipped=True)))
    assert r.distance_px == 0.0 and r.clipped


@pytest.mark.software
def test_the_nearest_confident_line_is_measured():
    r, s = both(geometry(line(20.0), line(60.0, confidence=0.2), line(45.0, tilt=3.0)))
    assert r.y_near_px == 45.0 and r.tilt_deg == 3.0
    assert (r.candidate_count, s["usable_count"]) == (3, 2)
    assert any("[REJECT]" in e and "0.200" in e for e in s["log"])


@pytest.mark.software
def test_the_confidence_gate_is_inclusive_and_configurable():
    geo = geometry(line(30.0, confidence=0.5))
    assert both(geo, StopLineDistanceConfig(min_confidence=0.5))[0].detected
    assert not both(geo, StopLineDistanceConfig(min_confidence=0.51))[0].detected


@pytest.mark.software
@pytest.mark.parametrize("lines", [(), (line(30.0, confidence=0.1),)], ids=["none", "all_weak"])
def test_nothing_measured_reports_not_detected_with_no_numbers(lines):
    r, _ = both(geometry(*lines))
    assert r == StopLineResult(False, None, None, None, None, None, False, 0.0, len(lines), 7, 350,
                               distance_cm=None, proximity=0.0)


@pytest.mark.software
def test_ties_go_to_the_first_candidate_in_both_versions():
    r, _ = both(geometry(line(40.0, x=(10.0, 90.0)), line(40.0, x=(200.0, 300.0))))
    assert r.x_left == 10.0


@pytest.mark.software
@pytest.mark.parametrize("fn", [compute_stop_line_distance, estimate_stop_line_distance])
def test_a_mismatched_frame_stamp_is_refused(fn):
    with pytest.raises(ValueError, match="stamp"):
        fn(geometry(line(40.0), frame_id=8), ROI)


# =============================================================================
# Chained from drawn frames
# =============================================================================

@pytest.mark.software
def test_a_nearer_line_measures_a_smaller_distance_through_the_chain():
    from src.tests.scenes import scene
    far = run_chain(scene(stop_line=(170, 270, 12)), 1, 50, SCENE_CONFIG).stop_line
    near = run_chain(scene(stop_line=(170, 270, 50)), 1, 50, SCENE_CONFIG).stop_line
    on = run_chain(SCENES["stop_line_clipped"], 1, 50, SCENE_CONFIG).stop_line
    assert far.distance_px > near.distance_px > on.distance_px == 0.0
    # The bar's bottom row is y_top + 6; the distance is from there to the ROI bottom
    assert near.distance_px == pytest.approx(ROI_H - 56, abs=1.0)


@pytest.mark.software
def test_phase3_linker_logs_the_stop_line_columns(tmp_path, monkeypatch):
    frames = tmp_path / "f"
    frames.mkdir()
    for i in range(4):
        cv2.imwrite(str(frames / f"{i:06d}.png"), SCENES["stop_line_wide"])
    out = tmp_path / "out"
    # The frame is drawn undistorted; MEASURED's undistortion bends its level
    # line past the 2 deg stop-line gate, so the run takes SCENE_CONFIG
    monkeypatch.setattr(p3, "MEASURED", SCENE_CONFIG)
    assert p3.cli(["--frames", str(frames), "--out", str(out), "--print-every", "0"]) == 0
    header, *rows = (out / "p3.csv").read_text().splitlines()
    cols = header.split(",")
    for name in ("p2_stop_line_px", "stop_line_detected", "stop_line_distance_px"):
        assert name in cols
    # cm columns were appended after every earlier column (the wheel columns came
    # later still), so nothing before them moved; blank without a ground homography
    assert cols[cols.index("p3_log") + 1:cols.index("p3_log") + 3] == ["p2_stop_line_cm", "stop_line_distance_cm"]
    # Only consistency is checked: once voted, the packet reports what Phase 2 measured
    last = dict(zip(cols, rows[-1].split(",")))
    assert float(last["p2_stop_line_px"]) > 0
    assert last["stop_line_detected"] == "1"
    assert float(last["stop_line_distance_px"]) == float(last["p2_stop_line_px"])


# =============================================================================
# Floor distance through the ground homography
# =============================================================================

def floor_y(frame_row):
    """What SYNTHETIC_GROUND says is Y at the robot's centerline for a frame row (its rows are level)."""
    return float(SYNTHETIC_GROUND.to_floor([[240.0, frame_row]])[0][1])


@pytest.mark.software
@pytest.mark.parametrize("y_near", [10.0, 40.0, 72.5])
def test_distance_cm_is_the_floor_distance_to_the_near_edge_at_the_centerline(y_near):
    r, _ = both(geometry(line(y_near)), ground=SYNTHETIC_GROUND)
    assert r.distance_cm == pytest.approx(floor_y(189 + y_near), abs=0.01)     # lane ROI origin added
    assert r.distance_px == ROI_H - y_near and r.proximity == pytest.approx(y_near / ROI_H, abs=1e-4)


@pytest.mark.software
def test_a_tilted_line_is_measured_where_it_crosses_the_robots_centerline():
    """Not at its nearest pixel: the right end is nearer, the centerline crossing further."""
    c = line(50.0, x=(150.0, 350.0), tilt=6.0)
    r, _ = both(geometry(c), ground=SYNTHETIC_GROUND)
    slope = np.tan(np.radians(6.0))
    mid = 250.0
    ends = [(24 + x, 189 + 50.0 + (x - mid) * slope) for x in (150.0, 350.0)]
    assert r.distance_cm == pytest.approx(SYNTHETIC_GROUND.forward_at_centerline(*ends), abs=0.01)
    assert r.distance_cm > floor_y(189 + 50.0 + 100 * slope) + 0.2           # further than the near end


@pytest.mark.software
@pytest.mark.parametrize("tilt", [0.0, 8.0, -8.0])
def test_a_line_the_robot_is_on_is_at_zero_cm_however_it_is_tilted(tilt):
    """Its near edge is below the frame; the ROI bottom stands in for it, not the tilted edge extended."""
    r, _ = both(geometry(line(float(ROI_H), clipped=True, tilt=tilt)), ground=SYNTHETIC_GROUND)
    assert r.distance_cm == 0.0 and r.distance_px == 0.0 and r.proximity == 1.0


@pytest.mark.software
def test_without_a_ground_plane_or_at_another_frame_size_there_is_no_cm():
    assert both(geometry(line(40.0)))[0].distance_cm is None
    other_size = SimpleNamespace(**{**vars(ROI), "source_shape": (360, 640)})
    r, _ = both(geometry(line(40.0)), ground=SYNTHETIC_GROUND, roi=other_size)
    assert r.detected and r.distance_px == 41.0 and r.distance_cm is None


@pytest.mark.software
def test_the_lane_roi_origin_is_added_before_projecting():
    """The same lane-ROI row sits lower in the frame when the ROI starts lower, so it is nearer."""
    higher = SimpleNamespace(**{**vars(ROI), "lane_rect": (24, 150, 432, ROI_H)})
    a, _ = both(geometry(line(40.0)), ground=SYNTHETIC_GROUND)
    b, _ = both(geometry(line(40.0)), ground=SYNTHETIC_GROUND, roi=higher)
    assert b.distance_cm == pytest.approx(floor_y(150 + 40.0), abs=0.01) and b.distance_cm > a.distance_cm


@pytest.mark.software
def test_the_log_reports_cm_when_there_is_a_ground_plane():
    _, s = both(geometry(line(40.0)), ground=SYNTHETIC_GROUND)
    assert any("cm ahead" in e for e in s["log"])


# =============================================================================
# The stop-line table
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("y_near", [10.0, 40.0, 72.5])
def test_without_a_homography_the_table_gives_the_cm_at_the_lines_rows(y_near):
    r, s = both(geometry(line(y_near)), table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.distance_px == ROI_H - y_near
    assert r.distance_cm == pytest.approx(SYNTHETIC_STOP_LINE_TABLE.to_cm(ROI_H - y_near), abs=0.01)
    assert any("cm ahead" in e for e in s["log"])


@pytest.mark.software
def test_a_line_the_robot_is_on_is_at_zero_cm_from_the_table_too():
    r, _ = both(geometry(line(float(ROI_H), clipped=True)), table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.distance_cm == 0.0


@pytest.mark.software
def test_the_homography_wins_over_the_table():
    r, _ = both(geometry(line(40.0)), ground=SYNTHETIC_GROUND, table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.distance_cm == pytest.approx(floor_y(189 + 40.0), abs=0.01)
    assert r.distance_cm != pytest.approx(SYNTHETIC_STOP_LINE_TABLE.to_cm(41.0), abs=0.01)


@pytest.mark.software
def test_a_homography_fit_at_another_size_falls_back_to_the_table():
    other = replace(SYNTHETIC_GROUND, image_size=(640, 360))
    r, _ = both(geometry(line(40.0)), ground=other, table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.distance_cm == pytest.approx(SYNTHETIC_STOP_LINE_TABLE.to_cm(41.0), abs=0.01)


@pytest.mark.software
def test_a_table_fit_at_another_frame_size_gives_no_cm():
    other_size = SimpleNamespace(**{**vars(ROI), "source_shape": (360, 640)})
    r, _ = both(geometry(line(40.0)), roi=other_size, table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.detected and r.distance_cm is None


@pytest.mark.software
def test_the_chain_reports_the_tables_cm_on_a_drawn_stop_line():
    cfg = replace(SCENE_CONFIG, stop_line_table=SYNTHETIC_STOP_LINE_TABLE)
    s = run_chain(SCENES["stop_line_wide"], 1, 50, cfg).stop_line
    assert s.detected and s.distance_cm == pytest.approx(SYNTHETIC_STOP_LINE_TABLE.to_cm(s.distance_px), abs=0.01)
    assert run_chain(SCENES["stop_line_wide"], 1, 50, SCENE_CONFIG).stop_line.distance_cm is None
