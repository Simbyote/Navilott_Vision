"""
test_preprocess.py  --  preprocessing stage

--software  Contract + known-answer tests on synthetic frames, plus every frame
            in src/tests/data/frames if a recorded dataset exists.
--hardware  Times preprocess_frame per frame (live camera or --replay), and
            writes CSV, sample images, and a latency histogram.
"""
import time
from dataclasses import FrozenInstanceError, asdict

import cv2
import numpy as np
import pytest

# Adjust this one import to wherever preprocess.py lives in your tree.
from src.perception.preprocess import (
    COLOR_BLUR_SUFFIX, GRAY_BLUR_SUFFIX, GRAY_SUFFIX,
    PreprocessParams, PreprocessResult,
    gaussian_blur, histogram_equalization, preprocess_frame, to_grayscale,
)
from src.capture.camera import FrameData
from src.tests.artifacts import summarize

H, W = 360, 480


# =============================================================================
# Helpers
# =============================================================================
def make_frame(seed=0, lo=0, hi=256):
    rng = np.random.default_rng(seed)
    return rng.integers(lo, hi, (H, W, 3), dtype=np.uint8)


def fd_of(frame, frame_id=42, ts=123456):
    return FrameData(frame=frame, frame_id=frame_id, timestamp_ms=ts)


def assert_contract(result, fd):
    """The output contract documented on PreprocessResult."""
    h, w = fd.frame.shape[:2]
    assert isinstance(result, PreprocessResult)
    assert result.gray.shape == (h, w) and result.gray.dtype == np.uint8
    assert result.color.shape == (h, w, 3) and result.color.dtype == np.uint8
    assert result.frame_id == fd.frame_id
    assert result.timestamp_ms == fd.timestamp_ms


# =============================================================================
# Software: output contract
# =============================================================================
@pytest.mark.software
def test_output_contract_on_synthetic_frame():
    fd = fd_of(make_frame())
    assert_contract(preprocess_frame(fd), fd)


@pytest.mark.software
def test_identity_is_carried_not_rederived():
    fd = fd_of(make_frame(), frame_id=987654, ts=555)
    r = preprocess_frame(fd)
    assert (r.frame_id, r.timestamp_ms) == (987654, 555)


@pytest.mark.software
def test_input_frame_is_not_mutated():
    frame = make_frame()
    before = frame.copy()
    preprocess_frame(fd_of(frame))
    assert np.array_equal(frame, before)


@pytest.mark.software
def test_outputs_do_not_alias_input():
    frame = make_frame()
    r = preprocess_frame(fd_of(frame))
    assert not np.shares_memory(r.gray, frame)
    assert not np.shares_memory(r.color, frame)


@pytest.mark.software
def test_deterministic():
    fd = fd_of(make_frame())
    a, b = preprocess_frame(fd), preprocess_frame(fd)
    assert np.array_equal(a.gray, b.gray) and np.array_equal(a.color, b.color)


@pytest.mark.software
def test_result_is_frozen():
    r = preprocess_frame(fd_of(make_frame()))
    with pytest.raises(FrozenInstanceError):
        r.frame_id = 0


@pytest.mark.software
def test_every_recorded_frame_meets_contract(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    for fd in dataset_frames:
        try:
            assert_contract(preprocess_frame(fd), fd)
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


# =============================================================================
# Software: known-answer tests
# =============================================================================
@pytest.mark.software
def test_pure_red_converts_to_expected_gray():
    frame = np.zeros((H, W, 3), np.uint8)
    frame[..., 2] = 255                              # BGR red
    gray = preprocess_frame(fd_of(frame)).gray
    assert abs(int(gray.mean()) - 76) <= 1           # 0.299 * 255
    assert np.ptp(gray) <= 1


@pytest.mark.software
def test_uniform_frame_stays_uniform_through_blur():
    frame = np.full((H, W, 3), (10, 20, 30), np.uint8)
    r = preprocess_frame(fd_of(frame))
    for ch, expected in enumerate((10, 20, 30)):      # per channel, not across channels
        assert np.ptp(r.color[..., ch]) <= 1
        assert abs(int(r.color[..., ch].mean()) - expected) <= 1


@pytest.mark.software
def test_gray_kernel_is_anisotropic_as_documented():
    """More smoothing across a lane line than along it."""
    img = np.zeros((41, 41), np.uint8)
    img[:, 20] = 255                                 # 1 px vertical line
    across_heavy = gaussian_blur(img, (9, 3))
    along_heavy = gaussian_blur(img, (3, 9))
    assert across_heavy[20, 20] < along_heavy[20, 20]


@pytest.mark.software
def test_equalize_widens_intensity_spread():
    low_contrast = make_frame(seed=1, lo=100, hi=140)
    off = preprocess_frame(fd_of(low_contrast), PreprocessParams(equalize=False)).gray
    on = preprocess_frame(fd_of(low_contrast), PreprocessParams(equalize=True)).gray
    assert on.std() > off.std()
    assert on.shape == off.shape and on.dtype == np.uint8


@pytest.mark.software
def test_equalize_default_is_off():
    assert PreprocessParams().equalize is False


# =============================================================================
# Software: rejection behaviour
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("bad, exc", [
    (None, ValueError),
    (np.zeros(10, np.uint8), ValueError),                     # 1-D
    (np.zeros((H, W, 4), np.uint8), ValueError),              # BGRA
    (np.zeros((H, W, 3), np.float32), TypeError),             # wrong dtype
])
def test_invalid_frames_are_rejected(bad, exc):
    with pytest.raises(exc):
        preprocess_frame(fd_of(bad))


@pytest.mark.software
@pytest.mark.parametrize("field", ["gray_kernel", "color_kernel"])
@pytest.mark.parametrize("kernel", [(4, 3), (9, 0), (9,), (9, 3, 1), (-3, 3)])
def test_invalid_kernels_are_rejected_and_named(field, kernel):
    params = PreprocessParams(**{field: kernel})
    with pytest.raises(ValueError, match=field):
        preprocess_frame(fd_of(make_frame()), params)


@pytest.mark.software
def test_equalize_rejects_multichannel():
    with pytest.raises(ValueError):
        histogram_equalization(make_frame())


# =============================================================================
# Software: documented-contract gap (see xfail reason)
# =============================================================================
@pytest.mark.software
def test_gray_input_is_rejected():
    with pytest.raises(ValueError, match="BGR"):
        preprocess_frame(fd_of(make_frame()[..., 0]))


# =============================================================================
# Hardware: preprocess characterization
# =============================================================================
@pytest.mark.hardware
def test_preprocess_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    params = PreprocessParams()

    rows, samples = [], {}
    for i, fd in enumerate(frames(n)):
        t0 = time.perf_counter_ns()
        res = preprocess_frame(fd, params)
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        assert_contract(res, fd)                      # outside the timing window
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 4),
                     round(float(res.gray.mean()), 2), round(float(res.gray.std()), 2)))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = (fd.frame, res)

    if not rows:
        pytest.skip("no frames delivered")

    # Artifacts are written after the loop so I/O never lands in the timing.
    ms = [r[2] for r in rows]
    artifacts.json("config.json", asdict(params))
    artifacts.csv("stage_timing.csv",
                  ["frame_id", "timestamp_ms", "stage_ms", "gray_mean", "gray_std"], rows)
    artifacts.json("summary.json", {"stage_ms": summarize(ms)})
    artifacts.histogram("stage_ms_hist.png", ms, "preprocess_frame latency", "ms")
    for fid, (raw, res) in samples.items():
        artifacts.image(f"{fid:06d}_0_input.png", raw)
        artifacts.image(f"{fid:06d}{GRAY_SUFFIX}", res.gray)
        artifacts.image(f"{fid:06d}{GRAY_BLUR_SUFFIX}", res.gray)
        artifacts.image(f"{fid:06d}{COLOR_BLUR_SUFFIX}", res.color)