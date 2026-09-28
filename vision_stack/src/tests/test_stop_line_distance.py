"""
test_stop_line_distance.py  --  src/perception/stop_line_distance.py

The measurement is tested on hand-built stop-line candidates, so every
distance has a known answer, then chained from drawn frames through the real
geometry stage. Its debug and production versions are held to each other.

--software  Distance from the lane ROI bottom, nearest-line choice, the
            confidence gate, the nothing-found result, stamps, and the
            p3.csv columns phase3_linker writes from it. No camera.
"""
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
from src.tests.scenes import SCENE_CONFIG, SCENES, same

ROI_H = 81
ROI = SimpleNamespace(lane_rect=(24, 189, 432, ROI_H), frame_id=7, timestamp_ms=350)


def line(y_near, confidence=0.8, clipped=False, x=(100.0, 300.0), tilt=0.0):
    """A stop-line candidate whose nearest point is y_near."""
    return StopLineCandidate(
        label=STOP_LINE, bbox=(int(x[0]), int(y_near) - 8, int(x[1] - x[0]), 8),
        x_left=x[0], x_right=x[1], y_top_px=y_near - 8, y_bottom_px=y_near, y_near_px=y_near,
        tilt_deg=tilt, length_px=x[1] - x[0], thickness_px=8.0, mean_intensity=220.0,
        clipped=clipped, confidence=confidence, frame_id=7, timestamp_ms=350)

def geometry(*lines, frame_id=7, ts=350):
    return GeometryBranchResult([], [], frame_id, ts, list(lines))

def both(geo, cfg=StopLineDistanceConfig()):
    """The debug and production results, checked equal."""
    debug, summary = compute_stop_line_distance(geo, ROI, cfg)
    same(debug, estimate_stop_line_distance(geo, ROI, cfg))
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
    assert r == StopLineResult(False, None, None, None, None, None, False, 0.0, len(lines), 7, 350)


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
def test_phase3_linker_logs_the_stop_line_columns(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    for i in range(4):
        cv2.imwrite(str(frames / f"{i:06d}.png"), SCENES["stop_line_wide"])
    out = tmp_path / "out"
    assert p3.cli(["--frames", str(frames), "--out", str(out), "--print-every", "0"]) == 0
    header, *rows = (out / "p3.csv").read_text().splitlines()
    cols = header.split(",")
    for name in ("p2_stop_line_px", "stop_line_detected", "stop_line_distance_px"):
        assert name in cols
    # The CLI runs MEASURED, which undistorts the synthetic frame, so only
    # consistency is checked: once voted, the packet reports what Phase 2 measured
    last = dict(zip(cols, rows[-1].split(",")))
    assert float(last["p2_stop_line_px"]) > 0
    assert last["stop_line_detected"] == "1"
    assert float(last["stop_line_distance_px"]) == float(last["p2_stop_line_px"])
