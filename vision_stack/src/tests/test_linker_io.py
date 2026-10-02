"""
test_linker_io.py  --  src/linker_io.py

The background recorder's drop policy, its copy of the frame and the error
it reports; the stand-in motors; and that a chain record carries no image,
pickles, and redraws through debug_maneuver.as_result().

--software  Synthetic scenes and a temp folder. No camera, motors or GPIO.
"""
import threading
import time

import cv2
import numpy as np
import pytest

import src.linker_io as io
from src.debugger.debug_maneuver import as_result, read_records
from src.debugger.estimation_debug import TracedPhase3Processor
from src.phase3_linker import run_phase3_chain
from src.tests.scenes import SCENE_CONFIG, SCENES


# =============================================================================
# FrameRecorder
# =============================================================================

@pytest.mark.software
def test_a_full_recorder_queue_drops_frames_instead_of_blocking(tmp_path, monkeypatch):
    gate = threading.Event()
    real = cv2.imwrite
    monkeypatch.setattr(io.cv2, "imwrite", lambda *a, **k: gate.wait() and real(*a, **k))
    rec = io.FrameRecorder(str(tmp_path), maxsize=2)
    t0 = time.perf_counter()
    for i in range(6):
        rec.put(i, SCENES["two_boundary"], {"frame_id": i})
    assert time.perf_counter() - t0 < 0.5                   # never waited on the disk
    gate.set()
    rec.close()
    assert rec.dropped >= 3 and rec.written + rec.dropped == 6


@pytest.mark.software
def test_the_recorder_copies_the_frame_before_queueing(tmp_path):
    frame = SCENES["two_boundary"].copy()
    rec = io.FrameRecorder(str(tmp_path))
    rec.put(1, frame, {"frame_id": 1})
    frame[:] = 0                                            # the camera reusing its buffer
    rec.close()
    assert cv2.imread(str(tmp_path / "frames" / "000001.jpg")).mean() > 10


@pytest.mark.software
def test_a_recorder_error_is_raised_by_close_not_in_the_loop(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(io.cv2, "imwrite", boom)
    rec = io.FrameRecorder(str(tmp_path))
    rec.put(1, SCENES["two_boundary"], {"frame_id": 1})     # the loop carries on
    with pytest.raises(RuntimeError, match="disk full"):
        rec.close()


# =============================================================================
# NoMotors and chain_record
# =============================================================================

@pytest.mark.software
def test_no_motors_takes_every_motor_call():
    m = io.NoMotors()
    assert m.drive(0.5, -0.5) is None and m.brake() is None and m.stop() is None


@pytest.mark.software
def test_a_chain_record_has_no_images_pickles_and_redraws(tmp_path):
    res = run_phase3_chain(SCENES["two_boundary"], 7, 350, TracedPhase3Processor(), None, SCENE_CONFIG)
    rec = io.chain_record(res)
    assert rec["frame_id"] == 7 and rec["timestamp_ms"] == 350 and rec["packet"] is res.packet
    assert not any(isinstance(v, np.ndarray) for v in rec.values())
    recorder = io.FrameRecorder(str(tmp_path))
    recorder.put(7, SCENES["two_boundary"], rec)
    recorder.close()
    [back] = list(read_records(str(tmp_path)))
    view = as_result(back, SCENES["two_boundary"])
    assert view.packet == res.packet and view.chain.offset == res.chain.offset
    assert view.chain.roi.lane_rect == res.chain.roi.lane_rect and view.timings_ms == res.timings_ms


@pytest.mark.software
@pytest.mark.parametrize("linker", ["src.navigation_linker", "src.intersection_linker"])
def test_no_linker_loads_the_drive_trial_for_these(linker):
    import subprocess
    import sys
    code = f"import sys, {linker}; print('src.maneuver_linker' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
