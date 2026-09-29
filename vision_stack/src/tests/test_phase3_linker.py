"""
test_phase3_linker.py  --  src/phase3_linker.py: the Phase 3 video and timing

The linker end to end on synthetic frames, headless: the video and its CSV
are on by default with one frame per processed frame, --no-video leaves
p3.csv and the timings as they were, render is timed but kept out of the
budget and the total, and Ctrl-C still leaves a playable video.

--software  cli() / run() on a frame directory. No camera.
"""
import importlib
import sys
import types

import cv2
import pytest

import src.debugger.debug_video as dv
import src.phase3_linker as p3
from src.debugger.debug_phase3 import Phase3View
from src.debugger.live_view import DirectoryFrameSource
from src.estimation_debug import TracedPhase3Processor
from src.tests.scenes import SCENE_CONFIG, SCENES

NAMES = ("two_boundary", "blind", "stop_line_wide", "stop_line_wide", "two_boundary")

@pytest.fixture
def frames(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    for i, n in enumerate(NAMES):
        cv2.imwrite(str(d / f"{i:06d}.png"), SCENES[n])
    return d

def video_frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n

def cli(frames, out, *extra):
    return p3.cli(["--frames", str(frames), "--out", str(out), "--print-every", "0",
                   "--no-display", *extra])


@pytest.mark.software
def test_the_video_and_its_csv_are_on_by_default_one_frame_each(frames, tmp_path):
    out = tmp_path / "out"
    assert cli(frames, out) == 0
    assert video_frames(out / "p3_debug.avi") == len(NAMES)
    header, *rows = (out / "p3_debug.csv").read_text().splitlines()
    assert header.split(",") == list(Phase3View.CSV_FIELDS) and len(rows) == len(NAMES)
    summary = (out / "summary.txt").read_text()
    assert "[PHASE 3] 5 frames" in summary
    assert "p3_lane" in summary and "render" in summary and "(not in total)" in summary


@pytest.mark.software
def test_no_video_writes_no_video_and_leaves_p3_csv_as_it_was(frames, tmp_path):
    on, off = tmp_path / "on", tmp_path / "off"
    cli(frames, on)
    cli(frames, off, "--no-video")
    assert not (off / "p3_debug.avi").exists() and not (off / "p3_debug.csv").exists()
    read = lambda d: [r.split(",") for r in (d / "p3.csv").read_text().splitlines()]
    a, b = read(on), read(off)
    assert a[0] == b[0] == list(p3.CSV_COLUMNS)
    timing = {p3.CSV_COLUMNS.index(c) for c in ("capture_ms", "phase2_ms", "phase3_ms", "total_ms")}
    strip = lambda rows: [[v for i, v in enumerate(r) if i not in timing] for r in rows[1:]]
    assert strip(a) == strip(b)                  # the same decisions with or without the video
    summary = (off / "summary.txt").read_text()
    assert "render" not in summary and "[PHASE 3]" not in summary


@pytest.mark.software
def test_render_is_timed_but_kept_out_of_the_budget_and_the_total(frames, tmp_path):
    stats = p3.run(DirectoryFrameSource(str(frames)), SCENE_CONFIG, out_dir=str(tmp_path / "o"),
                   print_every=0)
    assert len(stats.timings["render"]) == len(stats.stage_ms["render"]) == len(NAMES)
    for k in ("p3_lane", "p3_traffic", "geometry"):
        assert len(stats.stage_ms[k]) == len(NAMES)
    assert stats.timings["total"] == [c + p2 + p3_ for c, p2, p3_ in zip(
        stats.timings["capture"], stats.timings["phase2"], stats.timings["phase3"])]

    # A frame whose render alone blows the budget is not over budget
    budget = p3.Phase3Stats(budget_ms=50.0)
    res = p3.run_phase3_chain(SCENES["two_boundary"], 0, 0, TracedPhase3Processor(), None, SCENE_CONFIG)
    res.timings_ms.update(phase2=10.0, phase3=1.0, render=500.0)
    budget.update(res)
    assert budget.over_budget == 0


@pytest.mark.software
def test_ctrl_c_still_leaves_a_playable_video(frames, tmp_path, monkeypatch):
    class Interrupting(DirectoryFrameSource):
        def read(self):
            if self._n == 3:
                raise KeyboardInterrupt
            return super().read()
    closed = []
    real_close = dv.ViewWriter.close
    monkeypatch.setattr(dv.ViewWriter, "close", lambda self: closed.append(self.path) or real_close(self))
    out = tmp_path / "out"
    stats = p3.run(Interrupting(str(frames)), SCENE_CONFIG, out_dir=str(out), print_every=0)
    assert closed == [str(out / "p3_debug.avi")] and stats.frames == 3
    assert video_frames(out / "p3_debug.avi") == 3
    assert "[PHASE 3] 3 frames" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_the_linker_runs_the_traced_processor(frames, tmp_path, monkeypatch):
    seen = []
    real = p3.run_phase3_chain
    monkeypatch.setattr(p3, "run_phase3_chain",
                        lambda *a, **k: seen.append(type(a[3])) or real(*a, **k))
    cli(frames, tmp_path / "o", "--no-video")
    assert set(seen) == {TracedPhase3Processor}


# =============================================================================
# Encoders
# =============================================================================

@pytest.fixture
def fake_encoders(monkeypatch):
    """drive.py imported against test_drive's fake pigpio; returns (drive module, fake pi)."""
    from src.tests.test_drive import DRIVE_MODULE, FakePi
    pi = FakePi()
    pigpio = types.ModuleType("pigpio")
    pigpio.INPUT, pigpio.OUTPUT = "INPUT", "OUTPUT"
    pigpio.PUD_UP, pigpio.EITHER_EDGE = "PUD_UP", "EITHER_EDGE"
    pigpio.pi = lambda: pi
    monkeypatch.setitem(sys.modules, "pigpio", pigpio)
    monkeypatch.delitem(sys.modules, DRIVE_MODULE, raising=False)
    yield importlib.import_module(DRIVE_MODULE), pi
    sys.modules.pop(DRIVE_MODULE, None)


@pytest.mark.software
def test_encoder_counts_per_second_reach_the_sample_and_stopped_reads_zero(fake_encoders):
    drive, pi = fake_encoders
    enc = drive.EncoderReader
    sensors = p3.Sensors(encoders=True)
    # Forward on each side, per drive.py's decode: left C1 leads, right C2 leads
    pi.quad(enc.LEFT_C1, enc.LEFT_C2, 20, c1_leads=True)
    pi.quad(enc.RIGHT_C1, enc.RIGHT_C2, 10, c1_leads=False)
    moving = sensors.sample()
    assert moving.left_wheel_cps > moving.right_wheel_cps > 0.0
    stopped = sensors.sample()
    assert (stopped.left_wheel_cps, stopped.right_wheel_cps) == (0.0, 0.0)
    assert stopped.yaw_rate_dps is None                  # no IMU asked for
    sensors.stop()
    assert all(cb.cancelled for cb in pi.callbacks) and not pi.connected


@pytest.mark.software
def test_no_sensors_gives_no_sample():
    assert p3.Sensors().sample() is None


@pytest.mark.software
def test_encoders_flag_writes_the_wheel_columns(frames, tmp_path, fake_encoders, monkeypatch):
    drive, pi = fake_encoders
    enc, real = drive.EncoderReader, p3.Sensors.sample

    def turning(self):
        """The left wheel turns faster than the right before every frame's reading."""
        pi.quad(enc.LEFT_C1, enc.LEFT_C2, 20, c1_leads=True)
        pi.quad(enc.RIGHT_C1, enc.RIGHT_C2, 5, c1_leads=False)
        return real(self)
    monkeypatch.setattr(p3.Sensors, "sample", turning)

    out = tmp_path / "out"
    assert cli(frames, out, "--encoders", "--no-video") == 0
    header, *rows = (out / "p3.csv").read_text().splitlines()
    cols = header.split(",")
    assert cols[-2:] == ["left_wheel_cps", "right_wheel_cps"]
    wheels = [[float(v) for v in r.split(",")[-2:]] for r in rows]
    assert len(wheels) == len(NAMES) and all(left > right > 0.0 for left, right in wheels)
    without = tmp_path / "without"
    cli(frames, without, "--no-video")
    assert [r.split(",")[-2:] for r in (without / "p3.csv").read_text().splitlines()[1:]] == [["0.0", "0.0"]] * len(NAMES)
