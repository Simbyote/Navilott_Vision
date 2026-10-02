"""
test_linker_io.py  --  src/linker_io.py

open_rig() over faked hardware: what each source opens, the motors and the
button only with the camera, everything released when one part won't
open; the countdown; the background recorder's drop policy, its copy of
the frame and the error it reports; the stand-in motors; and that a chain
record carries no image, pickles, and redraws through
debug_maneuver.as_result().

--software  Fakes, synthetic scenes and a temp folder. No camera, motors or GPIO.
"""
import sys
import threading
import time
import types
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.linker_io as io
from src.debugger.debug_maneuver import as_result, read_records
from src.debugger.estimation_debug import TracedPhase3Processor
from src.params import FPS, FRAME_H, FRAME_W
from src.phase3_linker import run_phase3_chain
from src.tests.scenes import SCENE_CONFIG, SCENES


# =============================================================================
# open_rig and countdown
# =============================================================================

def _part(log, kind, release):
    """A fake device constructor: logs what it was opened with, and its stop() / close()."""
    def make(*a):
        log.append((kind, a))
        return SimpleNamespace(kind=kind, args=a, **{release: lambda: log.append((kind, release))})
    return make


@pytest.fixture
def rig(monkeypatch):
    """Fake sources, sensors, pigpio, motors and button; returns the log of what opened and was released."""
    log = []
    for name, kind in (("CameraFrameSource", "camera"), ("VideoFrameSource", "video"),
                       ("DirectoryFrameSource", "frames")):
        monkeypatch.setattr(io, name, _part(log, kind, "close"))
    sensors = _part(log, "sensors", "stop")
    monkeypatch.setattr(io, "Sensors", lambda **k: sensors(k))
    pigpio = types.ModuleType("pigpio")
    pigpio.pi = lambda: "pi"
    drive = types.ModuleType("src.peripherals.drive")
    drive.MotorController = _part(log, "motor", "stop")
    system = types.ModuleType("src.peripherals.system")
    system.System = _part(log, "system", "stop")
    for name, mod in (("pigpio", pigpio), ("src.peripherals.drive", drive), ("src.peripherals.system", system)):
        monkeypatch.setitem(sys.modules, name, mod)
    return log


@pytest.mark.software
def test_the_camera_opens_the_sensors_motors_and_button(rig):
    r = io.open_rig(camera=True)
    assert (r.source.kind, r.sensors.kind, r.motor.kind, r.system.kind) == ("camera", "sensors", "motor", "system")
    assert r.source.args == (FRAME_W, FRAME_H, FPS) and r.sensors.args == ({"imu": True, "encoders": True},)


@pytest.mark.software
def test_the_camera_without_motors_or_button(rig):
    r = io.open_rig(camera=True, fps=15, size=(320, 240), motors=False, button=False)
    assert r.source.args == (320, 240, 15) and r.sensors.kind == "sensors"
    assert isinstance(r.motor, io.NoMotors) and r.system is None


@pytest.mark.software
@pytest.mark.parametrize("kw, kind, rate", [({"video": "a.avi"}, "video", None), ({"frames": "dir"}, "frames", FPS),
                                            ({"video": "a.avi", "fps": 12}, "video", 12)])
def test_a_replay_never_opens_sensors_motors_or_button(rig, kw, kind, rate):
    r = io.open_rig(motors=True, button=True, **kw)
    assert r.source.kind == kind and r.source.args[-1] == rate
    assert r.sensors is None and isinstance(r.motor, io.NoMotors) and r.system is None


@pytest.mark.software
@pytest.mark.parametrize("fails_at, released", [
    ("motor", [("sensors", "stop"), ("camera", "close")]),
    ("system", [("motor", "stop"), ("sensors", "stop"), ("camera", "close")]),
])
def test_a_part_that_wont_open_releases_the_rest_and_raises(rig, monkeypatch, fails_at, released):
    log = rig
    real = sys.modules["src.peripherals.drive" if fails_at == "motor" else "src.peripherals.system"]
    attr = "MotorController" if fails_at == "motor" else "System"
    def broken(*a):
        raise RuntimeError(f"{fails_at} won't open")
    monkeypatch.setattr(real, attr, broken)
    with pytest.raises(io.OPEN_ERRORS, match="won't open"):
        io.open_rig(camera=True)
    assert [e for e in log if e[1] in ("stop", "close")] == released


@pytest.mark.software
def test_a_bug_is_not_a_hardware_error(rig, monkeypatch):
    monkeypatch.setattr(io, "Sensors", lambda **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        io.open_rig(camera=True)


@pytest.mark.software
def test_release_stops_or_closes_each_and_skips_none():
    log = []
    io.release(SimpleNamespace(stop=lambda: log.append("stop")), None, SimpleNamespace(close=lambda: log.append("close")))
    assert log == ["stop", "close"]


@pytest.mark.software
def test_the_countdown_is_three_seconds_out_loud(monkeypatch, capsys):
    slept = []
    monkeypatch.setattr(io.time, "sleep", slept.append)
    io.countdown()
    assert capsys.readouterr().out.split() == "starting in 3 starting in 2 starting in 1".split()
    assert slept == [1.0] * io.COUNTDOWN_S


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
