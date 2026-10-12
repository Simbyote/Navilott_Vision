"""
test_obstacle.py  --  src/perception/obstacle.py

The detector is tested on drawn frames, where every box has a known place,
then on hand-made masks for the gates and the distance arithmetic, and its
debug and production versions are held to each other throughout.

--software  What counts as not floor (bright, colored; not dark, not tape),
            the opening, every gate, nearest-blob choice, distance in rows
            and in cm (homography, then table, then none), clipped boxes,
            stamps, and the debug summary the obstacle view draws from. No
            camera.
--hardware  Runs the detector on live frames (or --replay) through the
            robot's MEASURED tuning and records what it saw: per-frame CSV,
            stills of the frames it fired on, timing and a summary. With
            --obstacle-cm X (a box placed in the path, its base X cm from
            the robot's front edge) it also checks the detection rate and
            the measured cm; with --obstacle-cm 0 the path must be clear.
"""
import statistics
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.config import MEASURED
from src.perception.obstacle import (
    GATE_FAR, GATE_NARROW, GATE_SHORT, GATE_SMALL, GATES, ObstacleConfig, ObstacleResult,
    compute_obstacle, estimate_obstacle, gate,
)
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIBounds, crop_rois
from src.phase2_linker import run_chain
from src.params import FRAME_H, FRAME_W
from src.tests.scenes import (
    ALT_CONFIG, DARK_BOX, ORANGE_BOX, SCENE_CONFIG, SCENES, SYNTHETIC_GROUND, SYNTHETIC_STOP_LINE_TABLE,
    WHITE_BOX, same, scene, with_obstacle,
)

CFG = ObstacleConfig()
# Preprocess blurs, so a drawn box's edges come back a pixel or two off
EDGE_PX = 3


def close(measured, drawn, px=EDGE_PX) -> bool:
    """Each value within px of what was drawn."""
    return all(abs(m - d) <= px for m, d in zip(measured, drawn))


def roi_of(frame, config=SCENE_CONFIG, frame_id=7, ts=350):
    return crop_rois(preprocess_frame(FrameData(frame, frame_id, ts), config.preprocess), config.roi)

def both(frame, cfg=CFG, ground=None, table=None, config=SCENE_CONFIG, roi=None):
    """The debug and production results on one frame, checked equal."""
    roi = roi_of(frame, config) if roi is None else roi
    result, debug = compute_obstacle(roi, cfg, ground, table)
    same(result, estimate_obstacle(roi, cfg, ground, table))
    return result, debug, roi


# =============================================================================
# What is an obstacle
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("bgr", [ORANGE_BOX, WHITE_BOX, (200, 60, 30), (40, 200, 40)],
                         ids=["orange", "white", "blue", "green"])
def test_a_bright_or_colored_box_in_the_path_is_found(bgr):
    r, _, roi = both(with_obstacle(scene(), 200, 280, 150, 250, bgr))
    x0, y0 = roi.obstacle_rect[:2]
    assert r.detected and not r.clipped
    assert close(r.bbox, (200 - x0, 150 - y0, 80, 100))
    assert close([r.distance_px], [FRAME_H - 250])


@pytest.mark.software
def test_a_dark_box_is_not_told_from_the_mats():
    """The known blind spot: no brighter than the floor and no color."""
    r, d, _ = both(with_obstacle(scene(), 200, 280, 150, 250, DARK_BOX))
    assert not r.detected and d["not_floor_px"] == both(scene())[1]["not_floor_px"]    # the lane marks' only


@pytest.mark.software
@pytest.mark.parametrize("name", [n for n in SCENES if not n.startswith("obstacle")])
def test_no_marking_scene_is_an_obstacle(name):
    """Lane lines, dashes, stop lines (thin, thick, clipped, tilted), signs, lamps and noise all fail the opening."""
    for config in (SCENE_CONFIG, ALT_CONFIG):
        r, _, _ = both(SCENES[name], config.obstacle, config=config)
        assert not r.detected, name


@pytest.mark.software
def test_tape_is_bright_but_never_survives_the_opening():
    """
    Lane tape up to ~40 px wide after the blur (MEASURED's lane gate allows
    45; the widest seen was 41.7) and a stop line far thicker than the
    course's (~10 rows per cm near the view bottom): open_px is wider than
    both. Tape wider than open_px would survive: the opening's one assumption.
    """
    frame = scene(marks=(150, 290), mark_width=36, stop_line=(100, 380, 40), stop_line_thickness=30)
    r, d, _ = both(frame)
    assert d["not_floor_px"] > 0 and not d["opened"].any() and not r.detected


@pytest.mark.software
def test_a_box_off_the_robots_path_is_not_its_obstacle():
    r, d, _ = both(with_obstacle(scene(), 10, 110, 150, 250))
    assert not r.detected and d["not_floor_px"] == both(scene())[1]["not_floor_px"]


@pytest.mark.software
def test_a_box_filling_the_view_is_still_found():
    """The floor level is capped: a percentile taken on the box itself would call it floor."""
    frame = with_obstacle(scene(), 0, FRAME_W, 100, FRAME_H, WHITE_BOX)
    r, d, _ = both(frame)
    assert d["floor_level"] == CFG.floor_cap and r.detected and r.clipped


# =============================================================================
# Gates and choice
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("w, h, area, rows, expected", [
    (100, 100, 2499, 10.0, GATE_SMALL),
    (59, 100, 5000, 10.0, GATE_NARROW),
    (100, 59, 5000, 10.0, GATE_SHORT),
    (100, 100, 5000, 90.5, GATE_FAR),
    (60, 60, 2500, 90.0, None),          # every gate is inclusive at its limit
])
def test_each_gate_in_order(w, h, area, rows, expected):
    assert gate(w, h, area, rows, CFG) == expected


@pytest.mark.software
def test_gates_are_counted_and_logged():
    _, d, _ = both(SCENES["obstacle_narrow"])
    assert d["reject_counts"][GATE_NARROW] == 1 and d["reject_counts"]["passed"] == 0
    assert set(d["reject_counts"]) == set(GATES) | {"passed"}
    assert any("[REJECT]" in e and GATE_NARROW in e for e in d["log"])
    assert [b["gate"] for b in d["blobs"]] == [GATE_NARROW]


@pytest.mark.software
def test_a_short_blob_and_a_far_one_are_refused_by_their_gates():
    tall = replace(CFG, min_height_px=110)
    r, d, _ = both(SCENES["obstacle_near"], tall)
    assert not r.detected and d["reject_counts"][GATE_SHORT] == 1
    near = replace(CFG, max_rows_ahead=15.0)
    r, d, _ = both(SCENES["obstacle_near"], near)
    assert not r.detected and d["reject_counts"][GATE_FAR] == 1


@pytest.mark.software
def test_the_nearest_box_is_the_obstacle():
    frame = with_obstacle(with_obstacle(scene(), 140, 210, 125, 190, WHITE_BOX), 250, 340, 150, 245)
    r, d, roi = both(frame)
    assert r.candidate_count == 2 and d["reject_counts"]["passed"] == 2
    assert close([r.distance_px, r.bbox[0]], [FRAME_H - 245, 250 - roi.obstacle_rect[0]])


@pytest.mark.software
def test_each_config_field_moves_the_answer():
    """Without this, parity under ALT_CONFIG couldn't catch a twin ignoring a field."""
    base = both(SCENES["obstacle_near"])[0]
    moved = {
        "floor_cap": (replace(CFG, floor_cap=10.0, min_saturation=255), with_obstacle(scene(), 200, 280, 150, 250, (90, 90, 90))),
        "min_contrast": (replace(CFG, min_contrast=200.0, min_saturation=255), SCENES["obstacle_near"]),
        "min_saturation": (replace(CFG, min_saturation=255), SCENES["obstacle_near"]),
        "min_value": (replace(CFG, min_value=255, min_contrast=250.0), SCENES["obstacle_near"]),
        "open_px": (replace(CFG, open_px=101), SCENES["obstacle_near"]),
        "min_area_px": (replace(CFG, min_area_px=9000), SCENES["obstacle_near"]),
        "min_width_px": (replace(CFG, min_width_px=90), SCENES["obstacle_near"]),
        "min_height_px": (replace(CFG, min_height_px=110), SCENES["obstacle_near"]),
        "max_rows_ahead": (replace(CFG, max_rows_ahead=10.0), SCENES["obstacle_near"]),
        "step": (replace(CFG, step=1), SCENES["obstacle_near"]),
    }
    for name, (cfg, frame) in moved.items():
        plain = both(frame)[0]
        assert both(frame, cfg)[0] != plain, name
    # floor_percentile: a frame with no road (all surround) gives another floor level
    _, low, _ = both(SCENES["obstacle_near"], replace(CFG, floor_percentile=1.0))
    _, high, _ = both(SCENES["obstacle_near"], replace(CFG, floor_percentile=90.0))
    assert low["floor_level"] != high["floor_level"] and base.detected


# =============================================================================
# Distance
# =============================================================================

@pytest.mark.software
def test_rows_are_counted_from_the_lane_roi_bottom_whatever_the_obstacle_roi():
    shorter = replace(SCENE_CONFIG, roi=replace(SCENE_CONFIG.roi, obstacle=ROIBounds(0.25, 0.40, 0.75, 0.95)))
    frame = SCENES["obstacle_near"]
    r, _, roi = both(frame, config=shorter)
    assert close([r.distance_px], [FRAME_H - 250]) and not r.clipped
    assert roi.obstacle_rect[1] + roi.obstacle_rect[3] < FRAME_H


@pytest.mark.software
def test_a_box_off_the_view_bottom_is_clipped_at_distance_zero():
    r, _, _ = both(SCENES["obstacle_clipped"], table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.clipped and r.distance_px == 0.0
    assert r.distance_cm == round(SYNTHETIC_STOP_LINE_TABLE.to_cm(0.0), 2)


@pytest.mark.software
def test_cm_come_from_the_table_at_the_bases_rows():
    r, _, _ = both(SCENES["obstacle_near"], table=SYNTHETIC_STOP_LINE_TABLE)
    assert r.distance_cm == round(SYNTHETIC_STOP_LINE_TABLE.to_cm(r.distance_px), 2)


@pytest.mark.software
def test_cm_come_from_the_ground_plane_first_at_the_bases_center():
    r, _, roi = both(SCENES["obstacle_near"], ground=SYNTHETIC_GROUND, table=SYNTHETIC_STOP_LINE_TABLE)
    x0, y0 = roi.obstacle_rect[:2]
    x, y, w, h = r.bbox
    expected = SYNTHETIC_GROUND.to_floor([(x0 + x + w / 2.0, y0 + y + h)])[0][1]
    assert r.distance_cm == round(float(expected), 2) and r.distance_cm > 0


@pytest.mark.software
def test_a_clipped_box_on_the_ground_plane_is_at_its_reference_or_nearer():
    r, _, _ = both(SCENES["obstacle_clipped"], ground=SYNTHETIC_GROUND)
    assert r.clipped and r.distance_cm == 0.0 and str(r.distance_cm) == "0.0"


@pytest.mark.software
def test_no_cm_without_a_calibration_or_at_another_frame_size():
    assert both(SCENES["obstacle_near"])[0].distance_cm is None
    small = cv2.resize(SCENES["obstacle_near"], (320, 180))
    r, _, _ = both(small, table=SYNTHETIC_STOP_LINE_TABLE, ground=SYNTHETIC_GROUND,
                   cfg=replace(CFG, min_area_px=500, min_width_px=30, min_height_px=30, open_px=31))
    assert r.detected and r.distance_cm is None


# =============================================================================
# Stamps, contract and debug
# =============================================================================

@pytest.mark.software
def test_results_carry_the_frame_stamp_found_or_not():
    for frame in (SCENES["obstacle_near"], scene()):
        r, d, _ = both(frame, roi=roi_of(frame, frame_id=4242, ts=98765))
        assert (r.frame_id, r.timestamp_ms) == (4242, 98765) == (d["frame_id"], d["timestamp_ms"])


@pytest.mark.software
def test_nothing_found_is_a_full_result():
    r, d, _ = both(scene())
    assert r == ObstacleResult(False, None, None, False, None, 0, 0, 7, 350)
    assert d["blobs"] == [] and d["log"] == []


@pytest.mark.software
def test_a_crop_without_an_obstacle_roi_is_refused():
    roi = replace(roi_of(scene()), obstacle_roi=None)
    for fn in (compute_obstacle, estimate_obstacle):
        with pytest.raises(ValueError, match="obstacle ROI"):
            fn(roi)


@pytest.mark.software
def test_the_debug_summary_holds_what_the_view_draws():
    r, d, roi = both(SCENES["obstacle_near"], table=SYNTHETIC_STOP_LINE_TABLE)
    h, w = roi.obstacle_roi.shape
    assert d["mask"].shape == d["opened"].shape == (h, w) and d["mask"].dtype == np.uint8
    assert d["threshold"] == d["floor_level"] + CFG.min_contrast
    assert d["blobs"][0] == {"bbox": r.bbox, "area": r.area_px, "rows_ahead": r.distance_px, "gate": None}
    assert any(e.startswith("[OBSTACLE]") and f"{r.distance_cm:.1f}cm" in e for e in d["log"])


@pytest.mark.software
def test_the_rois_are_not_modified():
    roi = roi_of(SCENES["obstacle_near"])
    gray, color = roi.obstacle_roi.copy(), roi.obstacle_color_roi.copy()
    both(SCENES["obstacle_near"], roi=roi)
    assert np.array_equal(gray, roi.obstacle_roi) and np.array_equal(color, roi.obstacle_color_roi)


@pytest.mark.software
def test_twins_agree_on_random_boxes():
    rng = np.random.default_rng(11)
    for i in range(60):
        frame = scene(noise_seed=int(rng.integers(0, 100)) if i % 3 == 0 else None)
        for _ in range(int(rng.integers(1, 4))):
            x0, y0 = int(rng.integers(100, 340)), int(rng.integers(110, 230))
            frame = with_obstacle(frame, x0, x0 + int(rng.integers(20, 140)), y0,
                                  min(FRAME_H, y0 + int(rng.integers(20, 160))),
                                  tuple(int(v) for v in rng.integers(0, 256, 3)))
        for config in (SCENE_CONFIG, ALT_CONFIG):
            both(frame, config.obstacle, config.ground, SYNTHETIC_STOP_LINE_TABLE, config=config)


@pytest.mark.software
def test_the_chain_measures_it_and_packages_it():
    chain = run_chain(SCENES["obstacle_near"], 3, 150, SCENE_CONFIG)
    assert chain.obstacle.detected and chain.phase2.obstacle_results == [chain.obstacle]
    assert "obstacle" in chain.timings_ms and chain.obstacle_debug["frame_id"] == 3


# =============================================================================
# On the Pi
# =============================================================================

# Where the hardware check holds the measurement: within this many cm of the
# tape, in this share of the frames
CM_TOLERANCE = 1.5
MIN_DETECTED = 0.9
CLEAR_MAX_FALSE = 0.02          # with the path clear: at most this share of frames fire
STAGE_BUDGET_MS = 5.0           # the obstacle stage's median on the Pi; the frame budget is 50 ms


@pytest.mark.hardware
def test_obstacle_on_the_pi(request, frames, artifacts):
    """
    The detector on real frames under MEASURED. Without --obstacle-cm it
    records what it sees; with it, it checks it against what was placed.
    """
    n = request.config.getoption("--frames")
    placed = request.config.getoption("--obstacle-cm")
    rows, stage_ms, stills = [], [], 0
    for fd in frames(n):
        chain = run_chain(fd.frame, fd.frame_id, fd.timestamp_ms, MEASURED)
        r, d = chain.obstacle, chain.obstacle_debug
        stage_ms.append(chain.timings_ms["obstacle"])
        rows.append({"frame_id": r.frame_id, "timestamp_ms": r.timestamp_ms, "detected": int(r.detected),
                     "distance_px": r.distance_px, "distance_cm": r.distance_cm, "clipped": int(r.clipped),
                     "bbox": r.bbox, "area_px": r.area_px, "candidates": r.candidate_count,
                     "floor_level": round(d["floor_level"], 1), "not_floor_px": d["not_floor_px"],
                     **{f"rej_{g}": d["reject_counts"][g] for g in GATES},
                     "stage_ms": round(stage_ms[-1], 3)})
        if r.detected and stills < 5:
            ox, oy = chain.roi.obstacle_rect[:2]
            x, y, w, h = r.bbox
            img = chain.pre.color.copy()
            cv2.rectangle(img, (ox + x, oy + y), (ox + x + w - 1, oy + y + h - 1), (0, 0, 255), 2)
            artifacts.image(f"detected_{r.frame_id:05d}.png", img)
            artifacts.image(f"mask_{r.frame_id:05d}.png", np.hstack([d["mask"], d["opened"]]))
            stills += 1
    if not rows:
        pytest.skip("no frames")
    artifacts.csv("obstacle.csv", list(rows[0]), [list(r.values()) for r in rows])

    seen = [r for r in rows if r["detected"]]
    cms = [r["distance_cm"] for r in seen if r["distance_cm"] is not None]
    summary = {
        "frames": len(rows), "detected": len(seen), "detected_share": round(len(seen) / len(rows), 3),
        "placed_cm": placed, "median_cm": statistics.median(cms) if cms else None,
        "cm_spread": round(statistics.pstdev(cms), 2) if len(cms) > 1 else None,
        "clipped": sum(r["clipped"] for r in rows),
        "stage_ms_median": round(statistics.median(stage_ms), 3), "stage_ms_max": round(max(stage_ms), 3),
        "table": MEASURED.stop_line_table is not None, "ground": MEASURED.ground is not None,
    }
    artifacts.json("summary.json", summary)

    assert summary["stage_ms_median"] <= STAGE_BUDGET_MS, summary
    if placed is None:
        return
    if placed == 0:
        assert summary["detected_share"] <= CLEAR_MAX_FALSE, f"the path was clear: {summary}"
        return
    assert summary["detected_share"] >= MIN_DETECTED, summary
    assert summary["median_cm"] is not None, "no cm: load the stop-line table (calibrate_stop_line)"
    assert abs(summary["median_cm"] - placed) <= CM_TOLERANCE, summary
