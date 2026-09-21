"""
test_roi_crop.py  --  ROI cropping stage

The ROI bounds are tunable, so nothing here pins the default LANE / TRAFFIC /
SIGN coordinates. The tests assert relations that must hold for ANY valid
bounds: rects lie inside the frame, each ROI is an exact read-only view of the
rect it reports, identity is carried, and the branch sources are the right
ones. Frame sizes and bounds are parametrized rather than hard-coded.

Dimensionality: crop is a catch-all, so an ROI has the same ndim as the array it
was cut from. Stage-level tests are therefore relational (ROI follows its
source). "Branch-ready" shapes (lane/sign 2-D, traffic 3-D) are checked where
real data flows: chained preprocess output, the recorded dataset, and hardware.

The one place hard checks remain is ROIBounds itself (its [0,1] and x0<x1
rules are the class's own contract, not a tuning choice).

--software  Contract tests on hand-built and chained (preprocess -> crop) input.
--hardware  Times crop_rois per frame (live camera or --replay) and writes CSV,
            the rects actually used, ROI crops, and the debug overlay.
"""
import time
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from src.capture.camera import FrameData
from src.perception.preprocess import PreprocessResult, preprocess_frame
from src.perception.roi_crop import (
    LANE_COLOR, LANE_ROI_SUFFIX, OVERLAY_SUFFIX, SIGN_COLOR, SIGN_ROI_SUFFIX,
    TRAFFIC_COLOR, TRAFFIC_ROI_SUFFIX,
    ROIBounds, ROIConfig, ROICropResult,
    crop, crop_rois, draw_roi_overlay, resolve,
)
from src.tests.artifacts import summarize

H, W = 360, 480
SHAPES = [(360, 480), (240, 320), (361, 479), (48, 64)]      # includes odd sizes

CONFIGS = {
    "default": ROIConfig(),
    "narrow": ROIConfig(
        lane=ROIBounds(0.40, 0.60, 0.60, 0.90),
        traffic=ROIBounds(0.45, 0.05, 0.55, 0.30),
        sign=ROIBounds(0.60, 0.10, 0.90, 0.40)),
    "full_frame": ROIConfig(
        lane=ROIBounds(0.0, 0.0, 1.0, 1.0),
        traffic=ROIBounds(0.0, 0.0, 1.0, 1.0),
        sign=ROIBounds(0.0, 0.0, 1.0, 1.0)),
}

# Non-overlapping, so each overlay rectangle can be checked independently
DISJOINT = ROIConfig(
    lane=ROIBounds(0.05, 0.70, 0.95, 0.95),
    traffic=ROIBounds(0.05, 0.05, 0.45, 0.40),
    sign=ROIBounds(0.55, 0.05, 0.95, 0.40))


# =============================================================================
# Helpers
# =============================================================================
def hand_pre(shape=(H, W), seed=0, frame_id=7, ts=1234):
    """PreprocessResult with independent random gray/color, so source mix-ups show."""
    rng = np.random.default_rng(seed)
    h, w = shape
    return PreprocessResult(
        gray=rng.integers(0, 256, (h, w), dtype=np.uint8),
        color=rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
        frame_id=frame_id, timestamp_ms=ts)


def chained_pre(seed=0, frame_id=3, ts=999):
    """Real upstream output: FrameData -> preprocess_frame."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    return preprocess_frame(FrameData(frame=frame, frame_id=frame_id, timestamp_ms=ts))


def assert_roi_contract(res, pre):
    """Everything ROICropResult documents, checked against the input it came from."""
    h_src, w_src = pre.gray.shape[:2]
    assert res.source_shape == (h_src, w_src)
    assert (res.frame_id, res.timestamp_ms) == (pre.frame_id, pre.timestamp_ms)
    parts = (("lane", res.lane_rect, res.lane_roi, pre.gray),
             ("sign", res.sign_rect, res.sign_roi, pre.gray),
             ("traffic", res.traffic_rect, res.traffic_roi, pre.color))
    for name, (x, y, w, h), roi, src in parts:
        assert x >= 0 and y >= 0 and w >= 1 and h >= 1, f"{name}: degenerate rect"
        assert x + w <= w_src and y + h <= h_src, f"{name}: rect leaves the frame"
        assert roi.shape[:2] == (h, w) and roi.dtype == np.uint8, f"{name}: shape/dtype"
        assert roi.shape[2:] == src.shape[2:], f"{name}: channel layout differs from its source"
        assert np.array_equal(roi, src[y:y + h, x:x + w]), \
            f"{name}: ROI is not the source slice at its reported rect"
    # crop is dimension-preserving: each ROI follows the array it was cut from
    assert res.lane_roi.ndim == pre.gray.ndim and res.sign_roi.ndim == pre.gray.ndim
    assert res.traffic_roi.ndim == pre.color.ndim


def assert_branch_ready(res):
    """What the branches consume: gray ROIs 2-D, traffic ROI (h, w, 3)."""
    assert res.lane_roi.ndim == 2 and res.sign_roi.ndim == 2
    assert res.traffic_roi.ndim == 3 and res.traffic_roi.shape[2] == 3


# =============================================================================
# Software: output contract, for any bounds and any frame size
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CONFIGS.keys())
def test_output_contract_for_any_bounds_and_frame_size(cfg, shape):
    pre = hand_pre(shape)
    assert_roi_contract(crop_rois(pre, cfg), pre)


@pytest.mark.software
def test_contract_holds_on_real_preprocess_output():
    pre = chained_pre()
    res = crop_rois(pre)
    assert_roi_contract(res, pre)
    assert_branch_ready(res)


@pytest.mark.software
def test_every_recorded_frame_crops_cleanly(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    for fd in dataset_frames:
        pre = preprocess_frame(fd)
        try:
            res = crop_rois(pre)
            assert_roi_contract(res, pre)
            assert_branch_ready(res)
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


@pytest.mark.software
def test_identity_is_carried_not_rederived():
    pre = hand_pre(frame_id=987654, ts=555)
    res = crop_rois(pre)
    assert (res.frame_id, res.timestamp_ms) == (987654, 555)


@pytest.mark.software
def test_rois_are_views_of_the_right_sources():
    pre = hand_pre()
    res = crop_rois(pre)
    assert np.shares_memory(res.lane_roi, pre.gray)
    assert np.shares_memory(res.sign_roi, pre.gray)
    assert np.shares_memory(res.traffic_roi, pre.color)
    assert not np.shares_memory(res.traffic_roi, pre.gray)


@pytest.mark.software
@pytest.mark.parametrize("name", ["lane_roi", "traffic_roi", "sign_roi"])
def test_rois_are_read_only(name):
    roi = getattr(crop_rois(hand_pre()), name)
    assert not roi.flags.writeable
    with pytest.raises(ValueError):
        roi[0, 0] = 0


@pytest.mark.software
def test_read_only_rois_do_not_lock_the_source_frames():
    """Debug drawing and later stages still need to write to the preprocessed arrays."""
    pre = hand_pre()
    crop_rois(pre)
    assert pre.gray.flags.writeable and pre.color.flags.writeable


@pytest.mark.software
def test_inputs_are_not_modified():
    pre = hand_pre()
    gray0, color0 = pre.gray.copy(), pre.color.copy()
    crop_rois(pre)
    assert np.array_equal(pre.gray, gray0) and np.array_equal(pre.color, color0)


@pytest.mark.software
def test_deterministic():
    pre = hand_pre()
    a, b = crop_rois(pre), crop_rois(pre)
    assert (a.lane_rect, a.traffic_rect, a.sign_rect) == (b.lane_rect, b.traffic_rect, b.sign_rect)
    assert np.array_equal(a.lane_roi, b.lane_roi) and np.array_equal(a.traffic_roi, b.traffic_roi)


@pytest.mark.software
def test_result_is_frozen():
    res = crop_rois(hand_pre())
    with pytest.raises(FrozenInstanceError):
        res.frame_id = 0


# =============================================================================
# Software: resolve() invariants
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("shape", SHAPES + [(1, 1), (2, 2), (3, 5), (360, 480, 3)])
def test_full_bounds_resolve_to_the_whole_frame(shape):
    h, w = shape[:2]
    assert resolve(ROIBounds(0.0, 0.0, 1.0, 1.0), shape) == (0, 0, w, h)


@pytest.mark.software
def test_sliver_bounds_still_resolve_to_at_least_one_pixel():
    for bounds in (ROIBounds(0.5, 0.5, 0.5001, 0.5001), ROIBounds(0.999, 0.999, 1.0, 1.0)):
        x, y, w, h = resolve(bounds, (10, 10))
        assert w >= 1 and h >= 1 and x + w <= 10 and y + h <= 10


@pytest.mark.software
def test_resolve_invariants_hold_for_random_bounds_and_sizes():
    rng = np.random.default_rng(1234)
    for _ in range(2000):
        h, w = int(rng.integers(1, 800)), int(rng.integers(1, 800))
        x0, x1 = np.sort(rng.uniform(0, 1, 2))
        y0, y1 = np.sort(rng.uniform(0, 1, 2))
        if x0 == x1 or y0 == y1:
            continue
        b = ROIBounds(float(x0), float(y0), float(x1), float(y1))
        x, y, rw, rh = resolve(b, (h, w))
        ctx = f"bounds={b} shape={(h, w)} -> {(x, y, rw, rh)}"
        assert 0 <= x < w and 0 <= y < h and rw >= 1 and rh >= 1, ctx
        assert x + rw <= w and y + rh <= h, ctx
        # Whenever the bounds span at least a pixel, rounding costs at most 1 px
        if (x1 - x0) * w >= 1:
            assert abs(rw - (x1 - x0) * w) <= 1.0 + 1e-9, ctx
        if (y1 - y0) * h >= 1:
            assert abs(rh - (y1 - y0) * h) <= 1.0 + 1e-9, ctx


# =============================================================================
# Software: rejection behaviour
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("kwargs", [
    dict(x0=-0.1, y0=0.0, x1=1.0, y1=1.0),
    dict(x0=0.0, y0=0.0, x1=1.1, y1=1.0),
    dict(x0=0.5, y0=0.0, x1=0.5, y1=1.0),          # zero width
    dict(x0=0.6, y0=0.0, x1=0.4, y1=1.0),          # reversed x
    dict(x0=0.0, y0=0.5, x1=1.0, y1=0.5),          # zero height
    dict(x0=0.0, y0=0.7, x1=1.0, y1=0.2),          # reversed y
    dict(x0=float("nan"), y0=0.0, x1=1.0, y1=1.0),
])
def test_invalid_bounds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ROIBounds(**kwargs)


@pytest.mark.software
@pytest.mark.parametrize("bad", [
    None,
    np.zeros(10, np.uint8),                        # 1-D
    np.zeros((H, W, 4), np.uint8),                 # BGRA
])
def test_crop_rejects_invalid_frames(bad):
    with pytest.raises(ValueError):
        crop(bad, ROIBounds(0.0, 0.0, 1.0, 1.0))


@pytest.mark.software
def test_gray_color_shape_mismatch_is_rejected():
    pre = replace(hand_pre((360, 480)), color=hand_pre((240, 320)).color)
    with pytest.raises(ValueError, match="disagree"):
        crop_rois(pre)


# =============================================================================
# Software: crop is dimension-preserving
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("ndim", [2, 3])
def test_crop_preserves_source_dimensionality(ndim):
    rng = np.random.default_rng(11)
    shape = (H, W) if ndim == 2 else (H, W, 3)
    frame = rng.integers(0, 256, shape, dtype=np.uint8)
    roi, (x, y, w, h) = crop(frame, ROIBounds(0.1, 0.2, 0.9, 0.8))
    assert roi.ndim == ndim and roi.dtype == np.uint8
    assert roi.shape[2:] == frame.shape[2:]
    assert np.array_equal(roi, frame[y:y + h, x:x + w])


@pytest.mark.software
def test_stage_output_dimensionality_follows_its_inputs():
    """
    crop_rois does not police which array is "gray" and which is "color".
    That is enforced upstream (preprocess) and downstream (each branch's own
    input validation), so here the ROIs simply follow their sources.
    """
    pre = hand_pre()
    swapped = replace(pre, gray=pre.color, color=pre.gray)      # 3-D "gray", 2-D "color"
    res = crop_rois(swapped)
    assert_roi_contract(res, swapped)
    assert res.lane_roi.ndim == 3 and res.sign_roi.ndim == 3
    assert res.traffic_roi.ndim == 2


# =============================================================================
# Software: debug overlay
# =============================================================================
@pytest.mark.software
def test_overlay_returns_an_unmodified_source_and_a_marked_copy():
    frame = np.random.default_rng(5).integers(0, 256, (H, W, 3), dtype=np.uint8)
    before = frame.copy()
    ov = draw_roi_overlay(frame, crop_rois(hand_pre(), DISJOINT))
    assert np.array_equal(frame, before)
    assert ov is not frame and not np.shares_memory(ov, frame)
    assert ov.shape == frame.shape and ov.dtype == frame.dtype
    assert not np.array_equal(ov, frame)


@pytest.mark.software
def test_overlay_draws_each_rect_in_its_color_and_nothing_elsewhere():
    res = crop_rois(hand_pre(), DISJOINT)
    ov = draw_roi_overlay(np.zeros((H, W, 3), np.uint8), res)
    for (x, y, w, h), color in ((res.lane_rect, LANE_COLOR),
                                (res.traffic_rect, TRAFFIC_COLOR),
                                (res.sign_rect, SIGN_COLOR)):
        assert tuple(ov[y, x]) == color                  # top-left corner
        assert tuple(ov[y + h - 1, x + w - 1]) == color  # bottom-right corner
    band = ov[int(0.45 * H):int(0.65 * H)]               # between the ROIs
    assert band.max() == 0


# =============================================================================
# Hardware: crop characterization
# =============================================================================
@pytest.mark.hardware
def test_roi_crop_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    cfg = ROIConfig()

    rows, samples, res = [], {}, None
    for i, fd in enumerate(frames(n)):
        pre = preprocess_frame(fd)                       # upstream, not timed

        t0 = time.perf_counter_ns()
        res = crop_rois(pre, cfg)
        crop_us = (time.perf_counter_ns() - t0) / 1e3

        assert_roi_contract(res, pre)                    # outside the timing window
        assert_branch_ready(res)
        rows.append((fd.frame_id, fd.timestamp_ms, round(crop_us, 2)))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = (fd.frame, res)

    if not rows:
        pytest.skip("no frames delivered")

    us = [r[2] for r in rows]
    artifacts.json("config.json", {
        "bounds": asdict(cfg),
        "rects_used": {"lane": res.lane_rect, "traffic": res.traffic_rect, "sign": res.sign_rect},
        "source_shape": res.source_shape,
    })
    artifacts.csv("crop_timing.csv", ["frame_id", "timestamp_ms", "crop_us"], rows)
    artifacts.json("summary.json", {"crop_us": summarize(us)})
    artifacts.histogram("crop_us_hist.png", us, "crop_rois latency", "microseconds")
    for fid, (raw, r) in samples.items():
        artifacts.image(f"{fid:06d}{OVERLAY_SUFFIX}", draw_roi_overlay(raw, r))
        artifacts.image(f"{fid:06d}{LANE_ROI_SUFFIX}", r.lane_roi)
        artifacts.image(f"{fid:06d}{TRAFFIC_ROI_SUFFIX}", r.traffic_roi)
        artifacts.image(f"{fid:06d}{SIGN_ROI_SUFFIX}", r.sign_roi)