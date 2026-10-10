"""
test_intersection_linker.py  --  src/intersection_linker.py

Each sequence end to end on synthetic camera frames (a stop line coming
down the view, the intersection, the lane coming back) with a scripted gyro
and a fake motor: it ends once the lane is held after the crossing, left and
right turn on the gyro and straight doesn't, and judge() passes them; every
CHECK reason on its own; SequenceWatch on hand-made rows; the command line.

--software  run_sequence() / cli() with fakes. No camera, motors or GPIO.
"""
import json

import cv2
import pytest

import src.intersection_linker as il
import src.linker_io as lio
from src.config import GYRO_BIAS_DPS
from src.estimation.estimation import SensorSample
from src.navigation.intersection import (
    ADVANCE_MS, LEFT_TURN, RIGHT_TURN, STAGE_ADVANCE, STAGE_EXIT, STAGE_TO_LINE, STAGE_TURN, TURN_END_GYRO,
    TURN_END_TIME,
)
from src.navigation.route import LEFT, RIGHT, STRAIGHT
from src.tests.navigation_checks import reach_frames
from src.tests.scenes import SCENE_CONFIG, scene
from src.tests.sim_robot import FakeClock

DT = 0.05
YAW_DPS = 60.0                     # a turn at 60 deg/s reaches the 85 deg target in ~1.4 s
REACH = reach_frames(frame_ms=round(DT * 1000))     # frames from the line leaving the view to reaching it
ADVANCE = ADVANCE_MS // round(DT * 1000)            # a turn's frames on into the intersection first


def sequence(maneuver):
    """(frame, yaw) per frame: lane, a stop line coming down to the view bottom, the intersection, the lane back."""
    out = [(scene(), 0.0)] * 6
    for y in (10, 30, 60, 70):
        out += [(scene(stop_line=(120, 320, y)), 0.0)] * 3
    out += [(scene(marks=()), 0.0)] * REACH                             # to the line
    yaw = {LEFT: -YAW_DPS, RIGHT: YAW_DPS}.get(maneuver)
    if yaw:
        out += [(scene(marks=()), 0.0)] * ADVANCE                      # on into the intersection
        out += [(scene(marks=()), yaw)] * 30                           # the turn: 90 deg on the gyro
    return out + [(scene(), 0.0)] * 60                                 # the lane back


class Source:
    """FrameSource stand-in over sequence(); advances the fake clock DT per read."""
    fps, label = 20, "sim"

    def __init__(self, clock, frames):
        self.clock, self.frames, self.i, self.current, self.closed = clock, frames, 0, None, False

    def read(self):
        self.clock.now += DT
        if self.i >= len(self.frames):
            return None
        self.current = self.frames[self.i]
        self.i += 1
        return self.current[0], self.i, int(self.i * 1000 * DT)

    def close(self):
        self.closed = True


class Sensors:
    """phase3_linker.Sensors stand-in: the current frame's scripted yaw."""
    def __init__(self, source):
        self.source, self.stopped = source, False

    def read(self):
        yaw = 0.0 if self.source.current is None else self.source.current[1]
        return SensorSample(yaw_rate_dps=yaw, lateral_accel_mps2=0.0, left_wheel_cps=300.0, right_wheel_cps=300.0), None

    def stop(self):
        self.stopped = True


class Motor:
    def __init__(self):
        self.calls = []

    def drive(self, left, right):
        self.calls.append(("drive", left, right))

    def brake(self):
        self.calls.append(("brake",))

    def stop(self):
        self.calls.append(("stop",))


def go(tmp_path, maneuver):
    clock = FakeClock()
    source = Source(clock, sequence(maneuver))
    motor = Motor()
    findings = il.run_sequence(maneuver, source, Sensors(source), motor, SCENE_CONFIG,
                               out_dir=str(tmp_path / maneuver), clock=clock, render=False)
    return findings, motor, source


# =============================================================================
# A sequence
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("maneuver, duties", [(LEFT, LEFT_TURN), (RIGHT, RIGHT_TURN), (STRAIGHT, None)])
def test_each_sequence_turns_its_way_holds_the_lane_after_and_passes(tmp_path, maneuver, duties):
    findings, motor, source = go(tmp_path, maneuver)
    assert findings["ended_by"] == il.SEQUENCE_DONE and findings["intersections"] == 1
    assert findings["verdict"] == "PASS", il.judge(findings)[1]
    assert findings["stage_s"][STAGE_TO_LINE] == pytest.approx(REACH * DT, abs=2 * DT)
    assert findings["stage_s"][STAGE_EXIT] > 0.0
    if duties:
        assert findings["stage_s"][STAGE_ADVANCE] == pytest.approx(ADVANCE_MS / 1000, abs=2 * DT)
        assert ("drive", duties.left, duties.right) in motor.calls and findings["turn_end"] == TURN_END_GYRO
        assert findings["stage_s"][STAGE_TURN] > 1.0
        assert findings["heading_deg"] == pytest.approx(il.EXPECTED_DEG[maneuver], abs=10)
    else:
        assert findings["stage_s"][STAGE_TURN] == findings["stage_s"][STAGE_ADVANCE] == 0.0
        assert findings["turn_end"] is None
    assert motor.calls[-1] == ("stop",) and source.closed
    assert source.i < len(source.frames)                               # it ended on the settled lane, not the source's end


@pytest.mark.software
def test_the_sequence_folder_holds_the_run_and_its_findings(tmp_path):
    findings, *_ = go(tmp_path, LEFT)
    folder = tmp_path / LEFT
    assert json.loads((folder / "sequence.json").read_text()) == findings
    for name in ("nav.csv", "p3.csv", "summary.txt", "report.json"):
        assert (folder / name).exists()
    rows = (folder / "nav.csv").read_text().splitlines()
    assert {"stage", "turn_end", "heading_deg", "yaw_rate"} <= set(rows[0].split(","))


@pytest.mark.software
def test_a_turn_the_gyro_never_sees_is_a_check(tmp_path):
    clock = FakeClock()
    frames = [(f, 0.0) for f, _ in sequence(LEFT)]                     # the robot "turns" but the gyro reads nothing
    source = Source(clock, frames)
    findings = il.run_sequence(LEFT, source, Sensors(source), Motor(), SCENE_CONFIG,
                               out_dir=str(tmp_path / "left"), clock=clock, render=False)
    verdict, problems = il.judge(findings)
    assert findings["turn_end"] == TURN_END_TIME and verdict == "CHECK"
    assert any("not the gyro target" in p for p in problems)


# =============================================================================
# judge() and SequenceWatch
# =============================================================================

GOOD = {"maneuver": LEFT, "intersections": 1, "turn_end": TURN_END_GYRO, "ended_by": il.SEQUENCE_DONE,
        "heading_deg": -88.0, "expected_deg": -90.0, "rejected": 0,
        "stage_s": {STAGE_TO_LINE: 1.5, STAGE_ADVANCE: 1.0, STAGE_TURN: 1.4, STAGE_EXIT: 0.2},
        "heading_at_turn_end_deg": -86.0}


@pytest.mark.software
@pytest.mark.parametrize("change, words", [
    ({"intersections": 2}, "2 intersections counted"),
    ({"intersections": 0}, "0 intersections counted"),
    ({"turn_end": TURN_END_TIME}, "not the gyro target"),
    ({"ended_by": "run time cap"}, "the lane wasn't held"),
    ({"heading_deg": -60.0}, "from -90"),
    ({"rejected": 3}, "broke the contract"),
])
def test_each_check_reason(change, words):
    assert il.judge(GOOD) == ("PASS", [])
    verdict, problems = il.judge({**GOOD, **change})
    assert verdict == "CHECK" and len(problems) == 1 and words in problems[0]


@pytest.mark.software
def test_straight_needs_no_turn_end():
    assert il.judge({**GOOD, "maneuver": STRAIGHT, "turn_end": None, "heading_deg": 3.0, "expected_deg": 0.0})[0] == "PASS"


def row(t, rule="lane_keeping", stage="", step="0/1", yaw=0.0, lane="vision", turn_end=""):
    return {"t": t, "rule": rule, "stage": stage, "step": step, "yaw_rate": yaw, "lane_status": lane,
            "turn_end": turn_end}


@pytest.mark.software
def test_the_watch_times_the_stages_integrates_the_heading_and_ends_once_settled():
    w = il.SequenceWatch(gyro_bias_dps=1.0, settle_s=0.15)
    assert [w(row(t)) for t in (0.0, 0.1, 0.2)] == [None] * 3           # lane keeping before the crossing never ends it
    w(row(0.3, "intersection", STAGE_TO_LINE, "1/1 left", yaw=1.0))
    w(row(0.4, "stop_sign", step="1/1 left", yaw=1.0))                  # held at the line: not lane keeping
    w(row(0.5, "intersection", STAGE_TURN, "1/1 left", yaw=-49.0))
    w(row(0.6, "intersection", STAGE_EXIT, "1/1 left", yaw=1.0, turn_end=TURN_END_GYRO))
    assert w(row(0.7, lane="hold")) is None and w(row(0.8)) is None    # a lane on hold doesn't count toward settling
    assert w(row(0.9)) == il.SEQUENCE_DONE
    f = w.findings(LEFT)
    assert f["intersections"] == 1 and f["turn_end"] == TURN_END_GYRO and f["lane_back"]
    assert f["stage_s"] == pytest.approx({STAGE_TO_LINE: 0.1, STAGE_ADVANCE: 0.0, STAGE_TURN: 0.1, STAGE_EXIT: 0.1})
    assert f["heading_at_turn_end_deg"] == pytest.approx(-5.0)
    assert f["heading_deg"] == pytest.approx(-5.3)                       # three rows at 0 against the 1 deg/s bias


# =============================================================================
# The command line
# =============================================================================

@pytest.mark.software
def test_a_replay_of_one_straight_intersection_passes(tmp_path, capsys):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, (f, _) in enumerate(sequence(STRAIGHT)):
        cv2.imwrite(str(frames / f"{i:04d}.png"), f)
    out = tmp_path / "out"
    code = il.cli(["straight", "--frames", str(frames), "--out", str(out), "--no-render"])
    report = json.loads((out / "report.json").read_text())
    assert [s["maneuver"] for s in report["sequences"]] == [STRAIGHT] and not report["motors"]
    assert report["gyro_bias_dps"] == GYRO_BIAS_DPS                    # config's, with no --gyro-bias
    assert (out / "straight" / "nav.csv").exists() and "[STRAIGHT]" in (out / "summary.txt").read_text()
    assert code == (0 if report["sequences"][0]["verdict"] == "PASS" else 1)


@pytest.mark.software
def test_all_on_a_replay_is_refused(tmp_path, capsys):
    assert il.cli(["all", "--frames", str(tmp_path)]) == 2
    assert "one intersection" in capsys.readouterr().out


@pytest.mark.software
def test_a_source_that_wont_open_is_exit_2(tmp_path, monkeypatch, capsys):
    def missing(*a, **k):
        raise OSError("no such file")
    monkeypatch.setattr(lio, "VideoFrameSource", missing)
    assert il.cli(["left", "--video", "missing.avi", "--out", str(tmp_path)]) == 2
    assert "source / hardware error" in capsys.readouterr().out


@pytest.mark.software
@pytest.mark.parametrize("flags, motor_kind, button", [([], "motor", True), (["--no-motors", "--no-button"], None, False)])
def test_the_camera_opens_the_motors_and_button_unless_told_not_to(tmp_path, monkeypatch, flags, motor_kind, button):
    import sys
    import types
    from types import SimpleNamespace
    monkeypatch.setattr(lio, "CameraFrameSource", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(lio, "Sensors", lambda **k: SimpleNamespace(stop=lambda: None))
    pigpio, drive, system = (types.ModuleType(n) for n in ("pigpio", "src.peripherals.drive", "src.peripherals.system"))
    pigpio.pi = lambda: "pi"
    drive.MotorController = lambda pi: SimpleNamespace(kind="motor", stop=lambda: None)
    system.System = lambda: SimpleNamespace(kind="system")
    for mod in (pigpio, drive, system):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    got = {}

    def run_sequence(maneuver, source, sensors, motor, config, p3_config, out_dir, system, **kw):
        got.update(motor=motor, system=system, **kw)
        return {**GOOD, "verdict": "PASS"}
    monkeypatch.setattr(il, "run_sequence", run_sequence)
    assert il.cli(["left", "--camera", "--out", str(tmp_path), *flags]) == 0
    assert getattr(got["motor"], "kind", None) == motor_kind and got["motors_on"] is (motor_kind is not None)
    assert (got["system"] is not None) is button


@pytest.mark.software
def test_camera_controls_reach_the_camera_and_a_bad_one_is_exit_2(tmp_path, monkeypatch, capsys):
    opened = {}

    def no_camera(w, h, fps, controls):
        opened.update(controls)
        raise OSError("stop here")
    monkeypatch.setattr(lio, "CameraFrameSource", no_camera)
    monkeypatch.setattr(lio, "CAMERA_CONTROLS", {})
    monkeypatch.setattr(il, "run_sequence", lambda *a, **k: pytest.fail("ran without a camera"))
    assert il.cli(["left", "--camera", "--no-motors", "--no-button", "--out", str(tmp_path),
                   "--camera-control", "awb-mode=daylight"]) == 2
    assert opened == {"awb-mode": "daylight"}
    assert il.cli(["left", "--camera", "--camera-control", "nope=1"]) == 2
    assert "camera control error" in capsys.readouterr().out
