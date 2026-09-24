"""
test_capture.py  --  src/capture/camera.py

--software  Contract tests against a scripted fake cv2.VideoCapture.
            Nothing here touches libcamera or a real device.
--hardware  Opens the real camera, records per-frame timing to CSV, saves
            sample frames, and (with --record) writes the frames out as the
            dataset that downstream software tests replay.
"""
import time
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import src.capture.camera as camera
from src.capture.camera import CameraSource, CaptureError, FrameData, VideoSink, build_gst_pipeline
from src.tests.artifacts import summarize
from src.tests.conftest import CAMERA, DATA_DIR

W, H, FPS = CAMERA["width"], CAMERA["height"], CAMERA["fps"]


def good():
    """One successful read: a uniform frame at the configured size."""
    return True, np.full((H, W, 3), 7, dtype=np.uint8)

BAD = (False, None)


class FakeCap:
    """Plays back a script of (ok, frame) tuples, then fails forever."""
    def __init__(self, script, opened=True):
        self._script = list(script)
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        return self._script.pop(0) if self._script else BAD

    def release(self):
        self.released = True


def opened_source(monkeypatch, script, **kw):
    """CameraSource opened against a FakeCap playing `script`; returns (source, fake)."""
    fake = FakeCap(script)
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a, **k: fake)
    return CameraSource(W, H, FPS, **kw).open(), fake


@pytest.mark.software
def test_read_before_open_raises():
    with pytest.raises(CaptureError):
        CameraSource(W, H, FPS).read()


@pytest.mark.software
def test_open_failure_raises_capture_error(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a, **k: FakeCap([], opened=False))
    with pytest.raises(CaptureError):
        CameraSource(W, H, FPS).open()


@pytest.mark.software
def test_frame_ids_are_zero_based_and_contiguous(monkeypatch):
    src, _ = opened_source(monkeypatch, [good() for _ in range(5)])
    assert [src.read().frame_id for _ in range(5)] == [0, 1, 2, 3, 4]


@pytest.mark.software
def test_delivered_frame_matches_documented_shape(monkeypatch):
    src, _ = opened_source(monkeypatch, [good()])
    fd = src.read()
    assert isinstance(fd, FrameData)
    assert fd.frame.shape == (H, W, 3) and fd.frame.dtype == np.uint8
    assert isinstance(fd.timestamp_ms, int)


@pytest.mark.software
def test_failed_read_returns_none_and_costs_no_frame_id(monkeypatch):
    src, _ = opened_source(monkeypatch, [good(), BAD, good()])
    a, gap, b = src.read(), src.read(), src.read()
    assert gap is None
    assert (a.frame_id, b.frame_id) == (0, 1)      # no gap, no duplicate


@pytest.mark.software
def test_failure_budget_exhaustion_raises_at_exact_count(monkeypatch):
    # Budget of 3: reads 1 and 2 are absorbed, read 3 exhausts it
    src, _ = opened_source(monkeypatch, [BAD, BAD, BAD], max_consecutive_failures=3)
    assert src.read() is None
    assert src.read() is None
    with pytest.raises(CaptureError):
        src.read()


@pytest.mark.software
def test_success_resets_failure_budget(monkeypatch):
    script = [BAD, BAD, good(), BAD, BAD]
    src, _ = opened_source(monkeypatch, script, max_consecutive_failures=3)
    results = [src.read() for _ in range(5)]        # must not raise
    assert [r is None for r in results] == [True, True, False, True, True]


@pytest.mark.software
def test_default_failure_budget_is_one_second_of_frames():
    assert CameraSource(W, H, FPS).max_consecutive_failures == FPS


@pytest.mark.software
def test_frame_id_survives_reopen(monkeypatch):
    src, _ = opened_source(monkeypatch, [good(), good()])
    src.read(); src.read()
    src.release()
    monkeypatch.setattr(camera.cv2, "VideoCapture", lambda *a, **k: FakeCap([good()]))
    src.open()
    assert src.read().frame_id == 2                 # ids must never be reissued


@pytest.mark.software
def test_timestamps_never_decrease(monkeypatch):
    # Downstream dt math assumes this; a wall clock would go negative on NTP steps
    src, _ = opened_source(monkeypatch, [good() for _ in range(50)])
    ts = [src.read().timestamp_ms for _ in range(50)]
    assert ts == sorted(ts)


@pytest.mark.software
def test_framedata_is_frozen(monkeypatch):
    src, _ = opened_source(monkeypatch, [good()])
    fd = src.read()
    with pytest.raises(FrozenInstanceError):
        fd.frame_id = 99


@pytest.mark.software
def test_release_frees_capture_and_blocks_further_reads(monkeypatch):
    src, fake = opened_source(monkeypatch, [good()])
    src.release()
    assert fake.released
    src.release()                                   # idempotent
    with pytest.raises(CaptureError):
        src.read()


@pytest.mark.software
def test_gst_pipeline_string_carries_parameters():
    p = build_gst_pipeline(width=640, height=480, fps=30, color_space="BGR")
    assert "width=640" in p and "height=480" in p and "framerate=30/1" in p
    assert "format=BGR" in p
    # drop/max-buffers is what keeps read() from returning a stale backlog
    assert p.rstrip().endswith("appsink drop=true max-buffers=1 sync=false")


class FakeWriter:
    """Records written frames instead of encoding them."""
    def __init__(self, opened=True):
        self._opened, self.frames, self.released = opened, [], False
    def isOpened(self): return self._opened
    def write(self, f): self.frames.append(f)
    def release(self): self.released = True


@pytest.mark.software
def test_sink_open_failure_raises(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoWriter", lambda *a, **k: FakeWriter(opened=False))
    with pytest.raises(CaptureError):
        VideoSink("x.avi", FPS, W, H).open()


@pytest.mark.software
def test_sink_rejects_wrong_size_and_accepts_right_size(monkeypatch):
    fake = FakeWriter()
    monkeypatch.setattr(camera.cv2, "VideoWriter", lambda *a, **k: fake)
    sink = VideoSink("x.avi", FPS, W, H).open()
    sink.write(np.zeros((H, W, 3), np.uint8))
    with pytest.raises(CaptureError):
        sink.write(np.zeros((H + 1, W, 3), np.uint8))
    assert len(fake.frames) == 1


@pytest.mark.software
def test_sink_write_before_open_raises():
    with pytest.raises(CaptureError):
        VideoSink("x.avi", FPS, W, H).write(np.zeros((H, W, 3), np.uint8))

@pytest.mark.software
@pytest.mark.parametrize("fps", [camera.MIN_FPS - 1, camera.MAX_FPS + 1])
def test_out_of_band_fps_warns(fps):
    with pytest.warns(RuntimeWarning, match="fps"):
        CameraSource(W, H, fps)

@pytest.mark.software
def test_in_band_fps_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning fails the test
        CameraSource(W, H, camera.MIN_FPS)  # the band edges are inclusive
        CameraSource(W, H, camera.MAX_FPS)

@pytest.mark.software
def test_explicit_zero_budget_is_not_replaced_by_fps():
    # Guards against `max_consecutive_failures or fps`, which treated 0 as unset
    assert CameraSource(W, H, FPS, max_consecutive_failures=0).max_consecutive_failures == 0


@pytest.mark.hardware
def test_capture_characterization(request, artifacts):
    n = request.config.getoption("--frames")
    record = request.config.getoption("--record")

    try:
        src = CameraSource(**CAMERA).open()
    except CaptureError as e:
        pytest.skip(f"camera unavailable: {e}")

    # Capture loop: buffer only. No file I/O inside the timing window.
    rows, kept, misses, prev_ts = [], [], 0, None
    try:
        while len(rows) < n:
            fd = src.read()
            if fd is None:
                misses += 1
                continue
            dt = None if prev_ts is None else fd.timestamp_ms - prev_ts
            prev_ts = fd.timestamp_ms
            rows.append((fd.frame_id, fd.timestamp_ms, dt, int(fd.frame.mean())))
            if record or len(rows) in (1, n // 2, n):
                kept.append(fd)
    finally:
        src.release()

    # Contract checks on real data.
    ids = [r[0] for r in rows]
    assert ids == list(range(ids[0], ids[0] + len(ids))), "frame_id gap or duplicate"
    assert all(k.frame.shape == (H, W, 3) and k.frame.dtype == np.uint8 for k in kept)

    # Artifacts.
    dts = [r[2] for r in rows if r[2] is not None]
    span_s = (rows[-1][1] - rows[0][1]) / 1000
    stats = summarize(dts)
    effective_fps = (len(rows) - 1) / span_s if span_s > 0 else 0.0
    artifacts.csv("frames.csv", ["frame_id", "timestamp_ms", "dt_ms", "mean_intensity"], rows)
    artifacts.json("summary.json", {
        "target": CAMERA, "effective_fps": effective_fps,
        "failed_reads": misses, "dt_ms": stats,
    })
    artifacts.histogram("dt_hist.png", dts, "Inter-frame interval", "dt (ms)")
    for k in (kept if not record else kept[:1] + kept[len(kept)//2:len(kept)//2 + 1] + kept[-1:]):
        artifacts.image(f"frame_{k.frame_id:06d}.png", k.frame)

    if record:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        import cv2
        with open(DATA_DIR / "manifest.csv", "w") as m:
            m.write("file,frame_id,timestamp_ms\n")
            for k in kept:
                name = f"{k.frame_id:06d}.png"
                cv2.imwrite(str(DATA_DIR / name), k.frame)
                m.write(f"{name},{k.frame_id},{k.timestamp_ms}\n")

    # Soft flag only: data collection, not a gate.
    if effective_fps < 0.95 * CAMERA["fps"]:
        warnings.warn(f"effective fps {effective_fps:.1f} below target {CAMERA['fps']}")