"""
test_maneuver_linker.py  --  src/maneuver_linker.py

The linker end to end on a simulated robot (sim_robot) and synthetic camera
frames: the run folder it writes, that the motors stop however the run ends
(finished, Ctrl-C, an error, the camera ending), that vision records but
never steers, the background recorder's drop policy, the start-button
hooks, the per-run overrides and the render-only command.

--software  run() / cli() with fakes. No camera, motors or GPIO.
"""
import csv
import json
import os
import threading
import time
from types import SimpleNamespace

import cv2
import pytest

import src.maneuver_linker as ml
from src.maneuver import ManeuverConfig
from src.tests.scenes import SCENE_CONFIG, SCENES
from src.tests.sim_robot import FakeClock, SimRobot

CFG = ManeuverConfig(leg_counts=300, settle_s=0.3)
DT = 0.04


class Camera:
    """FrameSource stand-in: a scene per frame, advancing the fake clock by DT per read."""
    fps, label = 25, "sim"

    def __init__(self, clock, scene="two_boundary", fail_at=None, end_at=None, drop_at=()):
        self.clock, self.scene, self.fail_at, self.end_at, self.drop_at = clock, scene, fail_at, end_at, drop_at
        self.i, self.closed = 0, False

    def read(self):
        self.clock.now += DT
        self.i += 1
        if self.i == self.fail_at:
            raise KeyboardInterrupt
        if self.end_at is not None and self.i >= self.end_at:
            return None
        if self.i in self.drop_at:
            return None, None, None
        return SCENES[self.scene], self.i, int(self.i * 1000 * DT)

    def close(self):
        self.closed = True


class Sensors:
    """phase3_linker.Sensors stand-in over the SimRobot."""
    def __init__(self, robot, fail_at=None):
        self.robot, self.fail_at, self.reads, self.stopped = robot, fail_at, 0, False

    def read(self):
        self.reads += 1
        if self.reads == self.fail_at:
            raise OSError("I2C read failed")
        return self.robot.read()

    def stop(self):
        self.stopped = True


def trial(tmp_path, cfg=CFG, cam=None, sensor_fail=None, system=None, render=True, **robot):
    clock = FakeClock()
    bot = SimRobot(clock, **robot)
    cam = cam or {}
    camera, sensors = Camera(clock, **cam), Sensors(bot, sensor_fail)
    out = tmp_path / "run"
    rep = ml.run(camera, sensors, bot, cfg, SCENE_CONFIG, out_dir=str(out), system=system,
                 clock=clock, render=render)
    return rep, out, bot, camera, sensors


def rows(path):
    return list(csv.DictReader(open(path)))


# =============================================================================
# A full trial
# =============================================================================

@pytest.mark.software
def test_a_trial_writes_the_whole_run_folder_and_stops_the_motors(tmp_path):
    rep, out, bot, camera, sensors = trial(tmp_path)
    assert rep["completed"] and rep["turn"]["success"]
    for name in ("summary.txt", "report.json", "maneuver.csv", "p3.csv", "config.json",
                 "records.pkl", "maneuver.avi", "maneuver_video.csv"):
        assert (out / name).stat().st_size > 0, name
    n = rep["run"]["frames"]
    assert len(rows(out / "maneuver.csv")) == len(rows(out / "p3.csv")) == n
    assert len(os.listdir(out / "frames")) == rep["run"]["recorder_written"] == n
    assert rep["run"]["video_frames"] == n and rep["run"]["recorder_dropped"] == 0
    assert bot.stops >= 1 and bot.cmd == (0.0, 0.0) and sensors.stopped and camera.closed
    assert json.loads((out / "report.json").read_text())["completed"] is True
    assert json.loads((out / "config.json").read_text())["leg_counts"] == CFG.leg_counts
    summary = (out / "summary.txt").read_text()
    assert "[MANEUVER] completed" in summary and "turn 180              PASS" in summary
    assert "[PHASE 3]" in summary and "p3_lane" in summary


@pytest.mark.software
def test_maneuver_csv_lines_up_with_p3_csv_and_logs_the_commands(tmp_path):
    rep, out, bot, *_ = trial(tmp_path)
    m, p3 = rows(out / "maneuver.csv"), rows(out / "p3.csv")
    assert [r["frame_id"] for r in m] == [r["frame_id"] for r in p3]
    driven = [(float(r["cmd_left"]), float(r["cmd_right"])) for r in m]
    assert driven == bot.commands                           # every command logged is the one sent
    assert {"forward_1", "turn", "forward_2", "done"} <= {r["step"] for r in m}


@pytest.mark.software
def test_vision_records_but_never_steers(tmp_path):
    a = trial(tmp_path / "a", cam={"scene": "two_boundary"})
    b = trial(tmp_path / "b", cam={"scene": "blind"})
    assert a[2].commands == b[2].commands                   # different pictures, same driving
    lane = {r["lane_status"] for r in rows(b[1] / "maneuver.csv")}
    assert lane != {r["lane_status"] for r in rows(a[1] / "maneuver.csv")}


@pytest.mark.software
def test_the_lane_is_timed_back_after_the_turn(tmp_path):
    rep, *_ = trial(tmp_path)
    assert rep["turn"]["lane_reacquired_s"] == 0.0          # the scene never lost it
    rep, *_ = trial(tmp_path / "blind", cam={"scene": "blind"})
    assert rep["turn"]["lane_reacquired_s"] is None
    assert "lane found again     never" in (tmp_path / "blind" / "run" / "summary.txt").read_text()


# =============================================================================
# However it ends, the motors stop first
# =============================================================================

@pytest.mark.software
def test_ctrl_c_stops_the_motors_and_still_writes_and_renders(tmp_path):
    rep, out, bot, *_ = trial(tmp_path, cam={"fail_at": 60})
    assert not rep["completed"] and "interrupted" in rep["abort_reason"]
    assert bot.stops >= 1 and bot.cmd == (0.0, 0.0)
    assert rep["run"]["video_frames"] == rep["run"]["frames"] > 0


@pytest.mark.software
def test_an_error_stops_the_motors_saves_the_traceback_and_is_raised(tmp_path):
    with pytest.raises(OSError, match="I2C"):
        trial(tmp_path, sensor_fail=50)
    out = tmp_path / "run"
    assert "OSError" in (out / "error.txt").read_text()
    assert "STOPPED: error" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_an_error_mid_drive_leaves_the_motors_stopped(tmp_path):
    clock = FakeClock()
    bot = SimRobot(clock)
    with pytest.raises(OSError):
        ml.run(Camera(clock), Sensors(bot, fail_at=45), bot, CFG, SCENE_CONFIG,
               out_dir=str(tmp_path / "r"), clock=clock, render=False)
    assert any(any(c) for c in bot.commands)                # it was moving
    assert bot.cmd == (0.0, 0.0) and bot.stops >= 1


@pytest.mark.software
def test_the_camera_ending_stops_the_trial(tmp_path):
    rep, *_ = trial(tmp_path, cam={"end_at": 40})
    assert "camera stopped" in rep["abort_reason"]


@pytest.mark.software
def test_dropped_camera_frames_are_counted_and_skipped(tmp_path):
    rep, out, *_ = trial(tmp_path, cam={"drop_at": (10, 11)})
    assert rep["run"]["camera_drops"] == 2 and rep["completed"]


# =============================================================================
# Recorder
# =============================================================================

@pytest.mark.software
def test_a_full_recorder_queue_drops_frames_instead_of_blocking(tmp_path, monkeypatch):
    gate = threading.Event()
    real = cv2.imwrite
    monkeypatch.setattr(ml.cv2, "imwrite", lambda *a, **k: gate.wait() and real(*a, **k))
    rec = ml.FrameRecorder(str(tmp_path), maxsize=2)
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
    rec = ml.FrameRecorder(str(tmp_path))
    rec.put(1, frame, {"frame_id": 1})
    frame[:] = 0                                            # the camera reusing its buffer
    rec.close()
    assert cv2.imread(str(tmp_path / "frames" / "000001.jpg")).mean() > 10


# =============================================================================
# Start button, overrides, render-only
# =============================================================================

@pytest.mark.software
def test_the_start_button_is_waited_for_and_the_display_kept(tmp_path):
    calls = []
    system = SimpleNamespace(wait_for_start=lambda: calls.append("wait"),
                             run_countdown=lambda: calls.append("countdown"),
                             update_display=lambda t: calls.append("tick"),
                             show_final_time=lambda t: calls.append(("final", round(t, 1))),
                             cleanup=lambda: calls.append("cleanup"))
    rep, *_ = trial(tmp_path, system=system, render=False)
    assert calls[:2] == ["wait", "countdown"] and calls[-1] == "cleanup"
    assert calls.count("tick") == rep["run"]["frames"] and calls[-2][0] == "final"


@pytest.mark.software
def test_flags_and_set_override_the_factory_defaults(tmp_path):
    ap_args = SimpleNamespace(**{flag[2:].replace("-", "_"): None for flag in ml._FLAGS.values()})
    ap_args.leg_counts, ap_args.speed = 2200, 0.35
    ap_args.set = ["turn_slow_band_deg=30", "stall_s=0.8", "settle_s = 3"]
    cfg = ml.trial_config(ManeuverConfig(), ap_args)
    assert (cfg.leg_counts, cfg.speed, cfg.turn_slow_band_deg, cfg.stall_s, cfg.settle_s) == (2200, 0.35, 30.0, 0.8, 3.0)
    assert isinstance(cfg.leg_counts, int)
    ap_args.set = ["no_such_field=1"]
    with pytest.raises(ValueError, match="fields are"):
        ml.trial_config(ManeuverConfig(), ap_args)


@pytest.mark.software
def test_cli_rejects_an_unknown_field(capsys):
    assert ml.cli(["--set", "wheel_count=4"]) == 2
    assert "fields are" in capsys.readouterr().out


@pytest.mark.software
def test_render_only_rebuilds_the_video_from_a_run_folder(tmp_path):
    rep, out, *_ = trial(tmp_path, render=False)
    assert not (out / "maneuver.avi").exists()
    assert ml.cli(["--render", str(out)]) == 0
    cap = cv2.VideoCapture(str(out / "maneuver.avi"))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == rep["run"]["frames"]
