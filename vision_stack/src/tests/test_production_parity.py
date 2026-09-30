"""
test_production_parity.py  --  production twins in src/perception/

geometry.py, color_branch.py, lane_offset.py and feature_fusion.py each carry
a debug-free twin next to every function that does debug work:

    geometry        _extract_lane_candidates -> _filter_lane_contours
                    extract_lane_candidates  -> find_lane_candidates
                    _extract_sign_candidates -> _filter_sign_contours
                    extract_sign_candidates  -> find_sign_candidates
                    run_geometry_stage       -> detect_geometry
    color_branch    _blobs_to_candidates     -> _filter_blobs
                    extract_traffic_light_candidates -> find_traffic_light_candidates
                    run_color_stage          -> detect_color
    lane_offset     _usable                  -> _is_usable
                    _single_sided            -> _project_single
                    compute_lane_offset      -> estimate_lane_offset
    feature_fusion  _best_candidate          -> _pick_best
                    fuse_detections          -> fuse

The twins repeat the gate logic, so they are only correct while they agree
with the debug path. Every test here pins that agreement.

--software  Both chains on synthetic scenes, compared field by field. No camera.
"""

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.params import HSV_RANGES_PATH
from src.perception import color_branch as cb
from src.perception import feature_fusion as ff
from src.perception import geometry as geo
from src.perception import lane_offset as lo
from src.perception import stop_line_distance as sld
from src.perception.phase2_out import package_phase2
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois
from src.config import MEASURED, PipelineConfig
from src.tests.scenes import ALT_CONFIG, SCENE_CONFIG, SCENES, SWEEP, SYNTHETIC_GROUND, SYNTHETIC_STOP_LINE_TABLE, same as _same, sweep_frame


# ALT_CONFIG moves every stage's tuning, so a twin that ignores part of its
# config fails here; MEASURED covers the undistortion path
CONFIGS = {"default": PipelineConfig(), "scene": SCENE_CONFIG, "alt": ALT_CONFIG, "measured": MEASURED}


def _roi(frame_bgr, config, frame_id=7):
    return crop_rois(preprocess_frame(FrameData(frame_bgr, frame_id, frame_id * 50), config.preprocess), config.roi)

def _run_debug(roi, config, color):
    """The chain through the debug functions, the way phase2_linker.run_chain() runs it."""
    g, _, _ = geo.run_geometry_stage(roi, config.geometry)
    traffic, _ = cb.run_color_stage(roi, color)
    offset, _ = lo.compute_lane_offset(g, roi, config.lane_offset)
    stop_line, _ = sld.compute_stop_line_distance(g, roi, config.stop_line, config.ground)
    fusion, _ = ff.fuse_detections(g, traffic, roi)
    return g, traffic, offset, stop_line, fusion, package_phase2(fusion, offset, stop_line)

def _run_production(roi, config, color):
    """The same chain through the production twins."""
    g = geo.detect_geometry(roi, config.geometry)
    traffic = cb.detect_color(roi, color)
    offset = lo.estimate_lane_offset(g, roi, config.lane_offset)
    stop_line = sld.estimate_stop_line_distance(g, roi, config.stop_line, config.ground)
    fusion = ff.fuse(g, traffic, roi)
    return g, traffic, offset, stop_line, fusion, package_phase2(fusion, offset, stop_line)

STAGES = ("geometry", "traffic", "offset", "stop_line", "fusion", "phase2")


@pytest.mark.software
@pytest.mark.parametrize("config_name", CONFIGS)
@pytest.mark.parametrize("color_on", [False, True], ids=["color_off", "color_on"])
@pytest.mark.parametrize("scene", SCENES)
def test_chain_parity(scene, color_on, config_name):
    """Every stage output, and the Phase2Output, match between the debug chain and the production twins."""
    config = CONFIGS[config_name]
    color = cb.ColorConfig(cb.load_hsv_ranges(str(HSV_RANGES_PATH))) if color_on else cb.ColorConfig()
    roi = _roi(SCENES[scene], config)

    for name, a, b in zip(STAGES,
                          _run_debug(roi, config, color),
                          _run_production(roi, config, color)):
        _same(a, b, name)


@pytest.mark.software
def test_scenes_exercise_every_output():
    """Guards the parity test itself: the scene set must reach every detection type and offset mode."""
    color = cb.ColorConfig(cb.load_hsv_ranges(str(HSV_RANGES_PATH)))
    types, modes = set(), set()
    for frame in SCENES.values():
        for config in CONFIGS.values():
            _, _, offset, _, fusion, _ = _run_debug(_roi(frame, config), config, color)
            types |= {d.type for d in fusion.detections}
            modes.add(offset.mode)
    assert {"lane_boundary", "stop_sign", "traffic_light"} <= types
    assert {"two_boundary", "none"} <= modes and len(modes) >= 3


@pytest.mark.software
@pytest.mark.parametrize("config_name", CONFIGS)
def test_threshold_sweep_parity(config_name):
    """Both chains agree on every frame of a sweep that straddles the gates."""
    config = CONFIGS[config_name]
    color = cb.ColorConfig()
    for level, width, noise in SWEEP:
        roi = _roi(sweep_frame(level, width, noise), config)
        for name, a, b in zip(STAGES,
                              _run_debug(roi, config, color),
                              _run_production(roi, config, color)):
            _same(a, b, f"level={level} width={width} noise={noise} {name}")


@pytest.mark.software
def test_uncalibrated_single_boundary_parity():
    """_project_single's uncalibrated branch, which the shipped configs never reach."""
    config = PipelineConfig(lane_offset=lo.LaneOffsetConfig(expected_half_lane_px=None))
    roi = _roi(SCENES["one_boundary"], config)
    g = geo.detect_geometry(roi, config.geometry)
    debug, _ = lo.compute_lane_offset(g, roi, config.lane_offset)
    _same(debug, lo.estimate_lane_offset(g, roi, config.lane_offset))
    assert debug.mode == "single_uncalibrated"


@pytest.mark.software
def test_invalid_confidence_parity():
    """Out-of-range confidences are dropped the same way by both fusion paths."""
    roi = _roi(SCENES["sign_and_lights"], PipelineConfig())
    g = geo.detect_geometry(roi)
    for c in g.lane_candidates[:1] + g.sign_candidates[:1]:
        c.confidence = 1.5
    traffic = [cb.TrafficLightCandidate("red", (0, 0, 5, 5), float("nan"), roi.frame_id, roi.timestamp_ms)]
    debug, _ = ff.fuse_detections(g, traffic, roi)
    _same(debug, ff.fuse(g, traffic, roi))


@pytest.mark.software
@pytest.mark.parametrize("bad, exc", [
    (None, ValueError),
    (np.zeros((10, 10), np.float32), TypeError),
    (np.zeros((10, 10, 3), np.uint8), ValueError),
], ids=["none", "float", "bgr"])
def test_detect_geometry_validates_like_run_geometry_stage(bad, exc):
    """detect_geometry rejects the same malformed ROIs with the same exception types."""
    roi = _roi(SCENES["two_boundary"], PipelineConfig())
    object.__setattr__(roi, "lane_roi", bad) if roi.__dataclass_params__.frozen else setattr(roi, "lane_roi", bad)
    with pytest.raises(exc):
        geo.run_geometry_stage(roi)
    with pytest.raises(exc):
        geo.detect_geometry(roi)


# ---------------------------------------------------------------------------
# Gate-level fuzz: the private twins fed identical randomized inputs. Whole
# frames can't be steered onto every threshold edge (an ROI span of exactly
# 1.0, a solidity of exactly 0.80), so each filter below is set so its
# thresholds sit in the middle of the random data, and integer draws land on
# the boundaries exactly. An off-by-one or a flipped < / <= in a twin shows up.
# ---------------------------------------------------------------------------

FUZZ_N = 1500

def _blocky_gray(rng, h, w):
    """Gray image of random constant blocks, so contour mean intensities spread across 100-155."""
    img = np.full((h, w), 127, np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, w), rng.integers(0, h)
        img[y:y + rng.integers(4, 30), x:x + rng.integers(4, 40)] = rng.integers(100, 156)
    return img

def _lane_contour(rng, h, w):
    """An ellipse outline (many points) or a bare quad (4 points, the too_few_pts gate)."""
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    if rng.random() < 0.15:
        return np.array([[[cx, cy]], [[cx + int(rng.integers(0, 20)), cy]],
                         [[cx + int(rng.integers(0, 20)), cy + int(rng.integers(0, 6))]],
                         [[cx, cy + int(rng.integers(0, 6))]]], np.int32)
    if rng.random() < 0.35:
        # Rectangle outline with 8 points: any integer bbox width, so the span
        # gate sees w / roi_w exactly at max_roi_span (an ellipse's is always odd)
        bw, bh = int(rng.integers(1, 70)), int(rng.integers(1, 14))
        x1, y1, xm, ym = cx + bw - 1, cy + bh - 1, cx + bw // 2, cy + bh // 2
        return np.array([[[cx, cy]], [[xm, cy]], [[x1, cy]], [[x1, ym]],
                         [[x1, y1]], [[xm, y1]], [[cx, y1]], [[cx, ym]]], np.int32)
    axes = (int(rng.integers(0, 55)), int(rng.integers(0, 12)))
    pts = cv2.ellipse2Poly((cx, cy), axes, int(rng.integers(0, 180)), 0, 360, 10)
    return pts.reshape(-1, 1, 2).astype(np.int32)

@pytest.mark.software
def test_lane_gate_twins_fuzz():
    rng = np.random.default_rng(1234)
    h, w = 60, 100
    lf = geo.LaneContourFilter(min_area=20, max_area=800, min_aspect=1.5, max_aspect=8.0,
                               max_roi_span=0.5, min_intensity=127)
    for trial in range(FUZZ_N // 10):
        gray = _blocky_gray(rng, h, w)
        contours = [_lane_contour(rng, h, w) for _ in range(10)]
        a = geo._extract_lane_candidates(contours, lf, 1, 2, (h, w), gray, {})
        b = geo._filter_lane_contours(contours, lf, 1, 2, (h, w), gray)
        _same(a, b, f"trial {trial}")

@pytest.mark.software
def test_sign_gate_twins_fuzz():
    rng = np.random.default_rng(5678)
    sf = geo.SignContourFilter()
    for trial in range(FUZZ_N // 10):
        contours = []
        for _ in range(10):
            n = int(rng.integers(5, 14))
            r = rng.uniform(4, 90) * (1 + rng.uniform(-0.45, 0.45, n))  # jitter spreads solidity
            t = np.sort(rng.uniform(0, 2 * np.pi, n))
            pts = np.stack([100 + r * np.cos(t), 100 + r * np.sin(t)], 1).astype(np.int32)
            contours.append(pts.reshape(-1, 1, 2))
        a = geo._extract_sign_candidates(contours, sf, 1, 2, {}, [])
        b = geo._filter_sign_contours(contours, sf, 1, 2)
        _same(a, b, f"trial {trial}")

@pytest.mark.software
def test_blob_gate_twins_fuzz():
    rng = np.random.default_rng(91011)
    bf = cb.BlobFilter()
    for trial in range(FUZZ_N // 10):
        mask = np.zeros((120, 200), np.uint8)
        for _ in range(6):
            x, y = int(rng.integers(0, 190)), int(rng.integers(0, 110))
            cv2.rectangle(mask, (x, y), (x + int(rng.integers(0, 90)), y + int(rng.integers(0, 60))), 255, -1)
        a = cb._blobs_to_candidates(mask, "red", bf, 1, 2, {}, [])
        b = cb._filter_blobs(mask, "red", bf, 1, 2)
        _same(a, b, f"trial {trial}")

def _random_lane_candidate(rng, cfg):
    """Fields drawn from small grids that include each LaneOffsetConfig threshold exactly."""
    pick = lambda *vals: float(rng.choice(vals))
    t = cfg
    return geo.LaneCandidate(
        label="lane_boundary", bbox=(0, 0, 1, 1), contour=None, frame_id=1, timestamp_ms=2,
        confidence=pick(t.conf_threshold - 0.01, t.conf_threshold, t.conf_threshold + 0.01, 0.9),
        proximity=pick(t.min_proximity - 0.01, t.min_proximity, t.min_proximity + 0.01, 0.9),
        length_px=pick(t.min_length_px - 1, t.min_length_px, t.min_length_px + 1, 200),
        width_px=pick(t.min_width_px - 0.5, t.min_width_px, t.max_width_px - 1, t.max_width_px, t.max_width_px + 1, 8),
        mean_intensity=pick(t.min_intensity - 1, t.min_intensity, t.min_intensity + 1, 250),
        foot_x=float(rng.integers(0, 432)),
    )

@pytest.mark.software
@pytest.mark.parametrize("config_name", CONFIGS)
def test_usable_twins_fuzz(config_name):
    rng = np.random.default_rng(1213)
    cfg = CONFIGS[config_name].lane_offset
    for i in range(FUZZ_N):
        c = _random_lane_candidate(rng, cfg)
        assert lo._usable(c, cfg, []) == lo._is_usable(c, cfg), f"case {i}: {c}"

@pytest.mark.software
def test_single_sided_twins_fuzz():
    rng = np.random.default_rng(1415)
    for i in range(FUZZ_N):
        cfg = lo.LaneOffsetConfig(expected_half_lane_px=None if rng.random() < 0.2 else float(rng.integers(50, 300)))
        anchor = lo.BoundaryAnchor(foot_x=float(rng.integers(0, 432)), weight=float(rng.random()), candidate=None)
        args = (anchor, str(rng.choice(["left", "right"])), 216.0, cfg, 1, 2, int(rng.integers(1, 4)))
        _same(lo._single_sided(*args, []), lo._project_single(*args), f"case {i}")

@pytest.mark.software
def test_best_candidate_twins_fuzz():
    rng = np.random.default_rng(1617)
    from types import SimpleNamespace
    for i in range(FUZZ_N):
        # Coarse grid forces ties; out-of-range and NaN values exercise the validity filter
        pool = [0.0, 0.25, 0.5, 0.5, 1.0, -0.1, 1.1, float("nan")]
        cands = [SimpleNamespace(confidence=float(rng.choice(pool)), frame_id=1, idx=k)
                 for k in range(int(rng.integers(0, 6)))]
        a = ff._best_candidate(cands, [], "x")
        b = ff._pick_best(cands)
        assert a is b, f"case {i}"

@pytest.mark.software
def test_stop_line_distance_twins_fuzz():
    """Random candidate sets, confidences straddling the gate and tied rows: both versions pick the same line."""
    from src.perception.geometry import StopLineCandidate
    rng = np.random.default_rng(11)
    roi = _roi(SCENES["two_boundary"], SCENE_CONFIG)
    for i in range(300):
        cands = []
        for _ in range(int(rng.integers(0, 5))):
            y = float(rng.choice([20.0, 40.0, 40.0, 81.0]))
            cands.append(StopLineCandidate(
                "stop_line", (0, 0, 1, 1), float(rng.integers(0, 200)), 300.0, y - 6, y, y,
                float(rng.uniform(-15, 15)), 100.0, 6.0, 200.0, y == 81.0,
                float(rng.choice([0.39, 0.4, 0.41, 0.9])), roi.frame_id, roi.timestamp_ms,
                proximity=round(y / 81.0, 4)))
        g = geo.GeometryBranchResult([], [], roi.frame_id, roi.timestamp_ms, cands)
        cfg = sld.StopLineDistanceConfig(min_confidence=0.4)
        ground = SYNTHETIC_GROUND if i % 2 else None
        table = SYNTHETIC_STOP_LINE_TABLE if i % 3 else None
        _same(sld.compute_stop_line_distance(g, roi, cfg, ground, table)[0],
              sld.estimate_stop_line_distance(g, roi, cfg, ground, table))
