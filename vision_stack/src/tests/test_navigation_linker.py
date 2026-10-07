"""
test_navigation_linker.py  --  src/navigation_linker.py

The linker end to end on synthetic camera frames and a fake motor: the run
folder it writes, that the motors get exactly the navigator's commands (and
a brake when a command breaks the contract), how a run ends (the run-time
cap, a critical battery, the source ending, a frame limit, Ctrl-C, an
error) with the motors stopped first every time, the battery logged per
frame and summarized, the start-button hooks, sensors reaching Phase 3, and
the command line: which sources may drive the motors, the battery check at
rest (a critical pack refuses only with the motors on), render-only.

--software  run() / cli() with fakes. No camera, motors or GPIO.
"""
import csv
import json
from dataclasses import replace
import sys
import types
from types import SimpleNamespace

import cv2
import pytest

import src.linker_io as lio
import src.navigation_linker as nl
from src.navigation.stop_line import STOP_DELAY_MS
from src.navigation.stop_sign import STOP_SIGN_HOLD_TIME_MS
from src.estimation.estimation import SensorSample
from src.peripherals.sensing import SensorBatch, SensorReading
from src.navigation.navigation import Navigation
from src.navigation.navigation_contract import BRAKE, Command
from src.tests.scenes import SCENE_CONFIG, SCENES
from src.tests.sim_robot import FakeBattery, FakeClock

DT = 0.05


class Camera:
    """FrameSource stand-in: one scene per frame from a script, advancing the fake clock by DT per read."""
    fps, label = 20, "sim"

    def __init__(self, clock, script=("two_boundary",), end_at=None, fail_at=None, drop_at=()):
        self.clock, self.script, self.end_at, self.fail_at, self.drop_at = clock, script, end_at, fail_at, drop_at
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
        return SCENES[self.script[min(self.i - 1, len(self.script) - 1)]], self.i, int(self.i * 1000 * DT)

    def close(self):
        self.closed = True


class Motor:
    """MotorController stand-in: logs every call in order."""
    def __init__(self):
        self.calls = []

    def drive(self, left, right):
        self.calls.append(("drive", left, right))

    def brake(self):
        self.calls.append(("brake",))

    def stop(self):
        self.calls.append(("stop",))


class Spy:
    """Wraps a navigator and keeps every packet and command."""
    def __init__(self, inner=None):
        self.inner = inner or Navigation()
        self.packets, self.commands, self.resets = [], [], 0

    @property
    def record(self):
        return self.inner.record

    @property
    def finished(self):
        return getattr(self.inner, "finished", False)

    @property
    def outcome(self):
        return getattr(self.inner, "outcome", None)

    @property
    def end_step(self):
        return getattr(self.inner, "end_step", None)

    @property
    def progress(self):
        return getattr(self.inner, "progress", None)

    def update(self, packet):
        self.packets.append(packet)
        cmd = self.inner.update(packet)
        self.commands.append(cmd)
        return cmd

    def reset(self):
        self.resets += 1
        self.inner.reset()


class Sensors:
    """phase3_linker.Sensors stand-in: a fixed sample each read."""
    def __init__(self, sample):
        self.sample, self.reads, self.stopped = sample, 0, False

    def read(self):
        self.reads += 1
        return self.sample, None

    def stop(self):
        self.stopped = True


def go(tmp_path, cam=None, nav=None, sensors=None, system=None, render=False, **kw):
    clock = FakeClock()
    camera = Camera(clock, **({"end_at": 9} if cam is None else cam))
    motor, nav = Motor(), nav or Spy()
    out = tmp_path / "run"
    rep = nl.run(camera, sensors, motor, nav, SCENE_CONFIG, out_dir=str(out), system=system,
                 clock=clock, render=render, **kw)
    return rep, out, motor, nav, camera


def rows(path):
    return list(csv.DictReader(open(path)))


# =============================================================================
# A run
# =============================================================================

@pytest.mark.software
def test_a_run_writes_the_whole_folder_and_renders_the_video(tmp_path):
    rep, out, motor, _, camera = go(tmp_path, render=True)
    n = rep["run"]["frames"]
    assert n == 8 and rep["ended_by"] == nl.END_SOURCE
    for name in ("summary.txt", "report.json", "nav.csv", "p3.csv", "records.pkl", "nav.avi", "nav_video.csv"):
        assert (out / name).stat().st_size > 0, name
    assert len(rows(out / "nav.csv")) == len(rows(out / "p3.csv")) == n
    assert rep["run"]["video_frames"] == rep["run"]["recorder_written"] == n
    assert motor.calls[-1] == ("stop",) and camera.closed
    assert json.load(open(out / "report.json"))["nav"]["frames"] == n
    assert "[NAVIGATION] ended by the source ended" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_the_motors_get_exactly_the_navigators_commands_in_frame_order(tmp_path):
    rep, out, motor, nav, _ = go(tmp_path)
    sent = motor.calls[:-1]                            # all but the final stop
    assert sent == [("brake",) if c.brake else ("drive", c.left, c.right) for c in nav.commands]
    assert [p.frame_id for p in nav.packets] == [int(r["frame_id"]) for r in rows(out / "p3.csv")]
    assert nav.resets == 1
    logged = rows(out / "nav.csv")
    assert [(float(r["cmd_left"]), float(r["cmd_right"]), int(r["brake"])) for r in logged] == \
           [(c.left, c.right, int(c.brake)) for c in nav.commands]


@pytest.mark.software
def test_a_centered_lane_drives_and_the_log_says_why(tmp_path):
    rep, out, motor, nav, _ = go(tmp_path)
    driven = [r for r in rows(out / "nav.csv") if r["lane_status"] == "vision"]
    assert driven and all(r["reason"] == "steer" and r["brake"] == "0" for r in driven)
    assert rep["nav"]["driving"] >= len(driven) and rep["nav"]["rejected"] == 0


@pytest.mark.software
def test_a_stop_sign_line_stops_the_robot_and_the_log_says_which_rule(tmp_path):
    # The synthetic scenes can't raise a voted stop sign through Phase 3 yet (the
    # sign gate is uncalibrated), so frames 2-4 carry a stop line coming down the
    # view with a sign; from frame 5 it's gone. At 50 ms a frame the robot reaches
    # the line STOP_DELAY_MS later and holds STOP_SIGN_HOLD_TIME_MS (no encoders:
    # the wheels read stopped at once)
    lines = {2: 30.0, 3: 15.0, 4: 5.0}

    class SignAndLine(Spy):
        def update(self, packet):
            rows = lines.get(packet.frame_id)
            return super().update(replace(packet, stop_sign_detected=rows is not None,
                                          stop_line_detected=rows is not None, stop_line_distance_px=rows))
    rep, out, motor, nav, _ = go(tmp_path, nav=SignAndLine(), cam={"end_at": 85})
    logged = rows(out / "nav.csv")
    braked = [int(r["frame_id"]) for r in logged if r["brake"] == "1"]
    reached = 5 + STOP_DELAY_MS // 50
    assert braked == list(range(reached, reached + STOP_SIGN_HOLD_TIME_MS // 50))
    assert {r["reason"] for r in logged if r["brake"] == "1"} == {"stop_sign_hold"}
    assert {r["rule"] for r in logged if r["brake"] == "1"} == {"stop_sign"}
    assert [motor.calls[i - 1] for i in braked] == [("brake",)] * len(braked)
    assert rep["nav"]["brake_reasons"] == {"stop_sign_hold": len(braked)}
    assert [r["event"] for r in logged if r["event"]] == ["-> steer", "-> crossing", "-> stop_sign_hold",
                                                         "-> crossing", "-> steer"]
    assert {r["rule"] for r in logged if r["reason"] == "crossing"} == {"intersection"}
    assert rep["nav"]["decided_by"]["stop_sign"] == len(braked)
    assert "decided by" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_reason_changes_are_logged_as_events(tmp_path):
    rep, out, *_ = go(tmp_path, cam={"script": ("two_boundary",) * 4 + ("sign_and_lights",), "end_at": 14})
    events = [r["event"] for r in rows(out / "nav.csv") if r["event"]]
    reasons = [r["reason"] for r in rows(out / "nav.csv")]
    assert events[0] == f"-> {reasons[0]}"
    assert len(events) == 1 + sum(a != b for a, b in zip(reasons, reasons[1:]))


# =============================================================================
# The contract
# =============================================================================

@pytest.mark.software
def test_a_command_that_breaks_the_contract_is_braked_and_counted(tmp_path):
    class Stalls:
        record = {}
        def update(self, packet):
            return Command(0.1, 0.1)
        def reset(self):
            pass
    rep, out, motor, _, _ = go(tmp_path, nav=Stalls())
    assert set(motor.calls[:-1]) == {("brake",)}
    logged = rows(out / "nav.csv")
    assert all(r["reason"] == nl.REASON_CONTRACT and "stall" in r["event"] for r in logged)
    assert rep["nav"]["rejected"] == len(logged) == rep["nav"]["brake_reasons"][nl.REASON_CONTRACT]


@pytest.mark.software
def test_a_navigator_without_a_record_is_logged_as_drive_or_brake(tmp_path):
    class Plain:
        def __init__(self):
            self.n = 0
        def update(self, packet):
            self.n += 1
            return BRAKE if self.n % 2 else Command(0.4, 0.4)
        def reset(self):
            pass
    _, out, *_ = go(tmp_path, nav=Plain())
    assert [r["reason"] for r in rows(out / "nav.csv")][:4] == ["brake", "drive", "brake", "drive"]


@pytest.mark.software
def test_the_intersection_stage_and_maneuver_are_logged(tmp_path):
    class Turning:
        record = {"rule": "intersection", "reason": "turning", "stage": "turn", "maneuver": "left", "step": "1/1 left",
                  "heading_deg": -42.5, "turn_end": "gyro target"}
        def update(self, packet):
            return Command(0.36, 0.63)
        def reset(self):
            pass
    _, out, *_ = go(tmp_path, nav=Turning())
    row = rows(out / "nav.csv")[0]
    assert (row["stage"], row["maneuver"], row["step"], row["reason"]) == ("turn", "left", "1/1 left", "turning")
    assert (row["heading_deg"], row["turn_end"]) == ("-42.5", "gyro target")


# =============================================================================
# How a run ends
# =============================================================================

@pytest.mark.software
def test_the_run_time_cap_ends_the_run_and_stops_the_motors(tmp_path):
    rep, out, motor, _, _ = go(tmp_path, cam={}, max_run_s=0.5)
    assert rep["ended_by"] == nl.END_CAP and motor.calls[-1] == ("stop",)
    assert rep["run"]["frames"] == pytest.approx(0.5 / DT, abs=1)


@pytest.mark.software
def test_the_battery_is_logged_every_frame_and_summarized(tmp_path):
    battery = FakeBattery(start_v=11.9, volts_step=0.1)
    rep, out, _, _, _ = go(tmp_path, battery=battery)
    r = rows(out / "nav.csv")
    assert [float(x["battery_v"]) for x in r[:3]] == [11.8, 11.7, 11.6] and {x["battery_state"] for x in r} == {"OK"}
    assert rep["battery"] == {"start_v": 11.8, "min_v": float(r[-1]["battery_v"]), "end_v": battery.v,
                              "state": "OK", "sensor_ok": True,
                              "read_ms": {"n": 3, "p50": 9.1, "p95": 9.37, "max": 9.4}}
    text = (out / "summary.txt").read_text()
    assert " battery               start 11.8 V, lowest" in text and "; ADC reads 9.1 ms p50, 9.4 max" in text


@pytest.mark.software
def test_the_lowest_voltage_is_kept_through_a_sag_under_load(tmp_path):
    battery = FakeBattery(start_v=11.5)
    sag = iter([11.5, 10.9, 11.4] + [11.4] * 50)

    def should_stop():
        battery.v = next(sag)
        return False
    battery.should_stop = should_stop
    rep, _, _, _, _ = go(tmp_path, battery=battery)
    assert (rep["battery"]["start_v"], rep["battery"]["min_v"]) == (11.5, 10.9)


@pytest.mark.software
def test_without_a_battery_the_columns_are_empty_and_there_is_no_summary_line(tmp_path):
    rep, out, _, _, _ = go(tmp_path)
    r = rows(out / "nav.csv")
    assert r[0]["battery_v"] == "" and r[0]["battery_state"] == "" and rep["battery"] is None
    assert "battery" not in (out / "summary.txt").read_text()


@pytest.mark.software
def test_a_critical_battery_ends_the_run_and_stops_the_motors(tmp_path):
    rep, out, motor, _, _ = go(tmp_path, cam={}, battery=FakeBattery(critical_at=5))
    assert rep["ended_by"] == nl.END_BATTERY and rep["run"]["frames"] == 4 and motor.calls[-1] == ("stop",)
    assert rep["battery"]["state"] == "CRITICAL" and "ended by battery critical" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_a_frame_limit_ends_the_run(tmp_path):
    rep, *_ = go(tmp_path, cam={}, limit=5)
    assert rep["ended_by"] == nl.END_LIMIT and rep["run"]["frames"] == 5


@pytest.mark.software
def test_ctrl_c_stops_the_motors_and_still_writes_the_summary(tmp_path):
    rep, out, motor, _, _ = go(tmp_path, cam={"fail_at": 5}, render=True)
    assert rep["ended_by"] == nl.END_INTERRUPT and motor.calls[-1] == ("stop",)
    assert (out / "summary.txt").exists() and rep["run"]["video_frames"] == 4


@pytest.mark.software
def test_an_error_stops_the_motors_saves_the_traceback_and_is_raised(tmp_path):
    class Breaks(Spy):
        def update(self, packet):
            if len(self.packets) == 3:
                raise ValueError("navigator bug")
            return super().update(packet)
    clock, motor = FakeClock(), Motor()
    with pytest.raises(ValueError, match="navigator bug"):
        nl.run(Camera(clock, end_at=9), None, motor, Breaks(), SCENE_CONFIG,
               out_dir=str(tmp_path / "run"), clock=clock, render=False)
    assert motor.calls[-1] == ("stop",)
    assert "navigator bug" in (tmp_path / "run" / "error.txt").read_text()
    assert "ended by error" in (tmp_path / "run" / "summary.txt").read_text()


@pytest.mark.software
def test_dropped_camera_frames_are_counted_and_skipped(tmp_path):
    rep, *_ = go(tmp_path, cam={"end_at": 9, "drop_at": (3, 4)})
    assert rep["run"]["camera_drops"] == 2 and rep["run"]["frames"] == 6


# =============================================================================
# Sensors and the start button
# =============================================================================

@pytest.mark.software
def test_sensor_readings_reach_the_packet_and_the_sensors_are_stopped(tmp_path):
    sensors = Sensors(SensorSample(yaw_rate_dps=0.0, left_wheel_cps=900.0, right_wheel_cps=880.0))
    rep, out, _, nav, _ = go(tmp_path, sensors=sensors)
    assert sensors.reads == rep["run"]["frames"] + 1          # one at the go, one per frame
    assert all((p.left_wheel_cps, p.right_wheel_cps) == (900.0, 880.0) for p in nav.packets)
    assert sensors.stopped
    assert {r["left_cps"] for r in rows(out / "nav.csv")} == {"900.0"}


class TimedSensors(Sensors):
    """Sensors whose batches carry IMU reads: 5 a frame at read_ms, the third failing every other frame."""
    def __init__(self, read_ms=(1.6, 1.7, 2.9, 1.6, 1.8)):
        super().__init__(SensorSample(yaw_rate_dps=0.0))
        self.read_ms = read_ms

    def read(self):
        self.reads += 1
        fail = self.reads % 2 == 0
        readings = tuple(SensorReading(self.reads + k / 10, None if (fail and k == 2) else 0.0, 0.0, None, None, ms)
                         for k, ms in enumerate(self.read_ms))
        return self.sample, SensorBatch(readings)


@pytest.mark.software
def test_imu_read_times_and_reads_per_frame_reach_nav_csv_the_report_and_the_summary(tmp_path):
    rep, out, *_ = go(tmp_path, sensors=TimedSensors())
    r = rows(out / "nav.csv")
    assert {(x["imu_reads"], x["imu_read_ms"]) for x in r} == {("5", "2.9")}
    assert sorted({x["imu_failed"] for x in r}) == ["0", "1"]
    s = rep["sensors"]
    frames = rep["run"]["frames"]
    assert s["imu_reads_per_frame"] == 5.0 and s["imu_failed"] == sum(int(x["imu_failed"]) for x in r)
    assert s["imu_read_ms"]["n"] == 5 * frames and s["imu_read_ms"]["max"] == 2.9 and s["imu_read_ms"]["p50"] == 1.7
    assert f" IMU reads             1.70 ms p50, {s['imu_read_ms']['p95']:.2f} p95, 2.90 max; 5 a frame " \
           f"(median; expect 5), {s['imu_failed']} failed" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_without_imu_timing_there_is_no_sensors_line(tmp_path):
    rep, out, *_ = go(tmp_path, sensors=Sensors(SensorSample(yaw_rate_dps=0.0)))      # batches aren't kept: None
    assert rep["sensors"] is None and "IMU reads" not in (out / "summary.txt").read_text()
    assert {x["imu_reads"] for x in rows(out / "nav.csv")} == {""}
    batchless, out2, *_ = go(tmp_path / "b", sensors=TimedSensors(read_ms=()))         # batches without IMU reads
    assert batchless["sensors"]["imu_read_ms"] is None and "IMU reads" not in (out2 / "summary.txt").read_text()


@pytest.mark.software
def test_the_start_button_is_waited_for_and_the_display_kept(tmp_path):
    calls = []
    system = SimpleNamespace(wait_for_start=lambda: calls.append("wait"),
                             run_countdown=lambda: calls.append("countdown"),
                             update_display=lambda t: calls.append("tick"),
                             show_final_time=lambda t: calls.append("final"),
                             cleanup=lambda blank=True: calls.append(f"cleanup blank={blank}"))
    rep, *_ = go(tmp_path, system=system)
    assert calls[:2] == ["wait", "countdown"] and calls[-2:] == ["final", "cleanup blank=False"]
    assert calls.count("tick") == rep["run"]["frames"]


@pytest.mark.software
def test_ctrl_c_while_waiting_for_the_button_still_stops_everything(tmp_path):
    def wait():
        raise KeyboardInterrupt
    system = SimpleNamespace(wait_for_start=wait, run_countdown=lambda: None, update_display=lambda t: None,
                             show_final_time=lambda t: None, cleanup=lambda blank=True: None)
    rep, out, motor, _, camera = go(tmp_path, system=system)
    assert rep["ended_by"] == nl.END_INTERRUPT and rep["run"]["frames"] == 0
    assert motor.calls == [("stop",)] and camera.closed


# =============================================================================
# The summary
# =============================================================================

@pytest.mark.software
def test_nav_stats_counts_driving_braking_steering_and_latency():
    s = nl.NavStats()
    for n in ({"reason": "steer", "brake": 0, "source": "offset", "steer": 0.2, "latency_ms": 10.0},
              {"reason": "steer", "brake": 0, "source": "heading", "steer": -0.1, "latency_ms": 30.0},
              {"reason": "stop_sign", "brake": 1, "source": "none", "steer": 0.0, "latency_ms": 20.0},
              {"reason": nl.REASON_CONTRACT, "brake": 1, "source": "none", "steer": 0.0, "latency_ms": 40.0}):
        s.update(n)
    r = s.report()
    assert (r["frames"], r["driving"], r["braked"], r["rejected"]) == (4, 2, 2, 1)
    assert r["brake_reasons"] == {"stop_sign": 1, nl.REASON_CONTRACT: 1}
    assert r["steer_sources"] == {"offset": 1, "heading": 1}
    assert r["steer_abs_mean"] == pytest.approx(0.15) and r["steer_abs_max"] == pytest.approx(0.2)
    assert r["latency_ms"]["max"] == 40.0 and r["latency_ms"]["p50"] == pytest.approx(25.0)


@pytest.mark.software
def test_an_empty_run_still_summarizes():
    r = nl.NavStats().report()
    lines = nl.summary_lines({"ended_by": "x", "motors": False, "nav": r,
                              "run": {"frames": 0, "wall_s": 0.0, "fps": 0.0, "camera_drops": 0,
                                      "recorder_dropped": 0}})
    assert lines[0] == "[NAVIGATION] ended by x   motors OFF (dry run)"


# =============================================================================
# Command line
# =============================================================================

@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """cli() with every source, sensor, motor and button replaced; returns what run() was given."""
    got = {}
    fake_source = lambda *a, **k: SimpleNamespace(label="fake", fps=20, close=lambda: None)
    for name in ("CameraFrameSource", "VideoFrameSource", "DirectoryFrameSource"):
        monkeypatch.setattr(lio, name, fake_source)
    monkeypatch.setattr(lio, "Sensors", lambda **k: SimpleNamespace(kind="sensors", stop=lambda: None))
    drive = types.ModuleType("src.peripherals.drive")
    drive.MotorController = lambda pi: SimpleNamespace(kind="motor", stop=lambda: None)
    pigpio = types.ModuleType("pigpio")
    pigpio.pi = lambda: "pi"
    system = types.ModuleType("src.peripherals.system")
    system.System = lambda: SimpleNamespace(kind="system")
    monkeypatch.setitem(sys.modules, "src.peripherals.drive", drive)
    monkeypatch.setitem(sys.modules, "pigpio", pigpio)
    monkeypatch.setitem(sys.modules, "src.peripherals.system", system)
    monkeypatch.setattr(nl.time, "sleep", lambda s: None)

    def run(source, sensors, motor, navigator, config, p3_config, out_dir, system, **kw):
        got.update(sensors=sensors, motor=motor, navigator=navigator, p3_config=p3_config,
                   system=system, out_dir=out_dir, **kw)
        return {}
    monkeypatch.setattr(nl, "run", run)
    return got, tmp_path


@pytest.mark.software
def test_the_camera_drives_the_motors_with_sensors_and_the_button(cli_env):
    got, tmp = cli_env
    assert nl.cli(["--camera", "--out", str(tmp / "o")]) == 0
    assert got["motor"].kind == "motor" and got["motors_on"] is True
    assert got["sensors"].kind == "sensors" and got["system"].kind == "system"
    assert isinstance(got["navigator"], Navigation) and got["max_run_s"] == nl.MAX_RUN_S


@pytest.mark.software
def test_camera_controls_from_the_command_line_reach_the_camera(cli_env, monkeypatch):
    opened = {}
    monkeypatch.setattr(lio, "CAMERA_CONTROLS", {})
    monkeypatch.setattr(lio, "CameraFrameSource", lambda w, h, fps, controls: opened.update(controls) or
                        SimpleNamespace(label="camera", fps=fps, close=lambda: None))
    _, tmp = cli_env
    assert nl.cli(["--camera", "--out", str(tmp / "o"), "--camera-control", "ae-constraint-mode=highlight",
                   "--camera-control", "exposure-value=-1"]) == 0
    assert opened == {"ae-constraint-mode": "highlight", "exposure-value": -1}


@pytest.mark.software
def test_an_unknown_camera_control_is_exit_2_before_anything_opens(cli_env, capsys):
    _, tmp = cli_env
    assert nl.cli(["--camera", "--out", str(tmp / "o"), "--camera-control", "shutter=1"]) == 2
    assert "camera control error" in capsys.readouterr().out


def with_battery(monkeypatch, volts, log):
    battery = FakeBattery(start_v=volts, log=log)
    battery.preflight = lambda: (volts > battery.VOLTAGE_WARNING, volts)
    monkeypatch.setattr(nl.battery_run, "open_battery", lambda: battery)
    return battery


@pytest.mark.software
def test_a_critical_pack_refuses_to_drive_the_motors(cli_env, monkeypatch, capsys):
    got, tmp = cli_env
    log = []
    with_battery(monkeypatch, 9.6, log)
    assert nl.cli(["--camera", "--out", str(tmp / "o")]) == 2
    assert got == {} and "not starting" in capsys.readouterr().out and log == ["battery released"]


@pytest.mark.software
def test_on_the_bench_a_critical_pack_only_warns_and_the_run_gets_the_battery(cli_env, monkeypatch, capsys):
    got, tmp = cli_env
    log = []
    battery = with_battery(monkeypatch, 9.6, log)
    assert nl.cli(["--camera", "--no-motors", "--no-button", "--out", str(tmp / "o")]) == 0
    assert got["battery"] is battery and "motors are off" in capsys.readouterr().out
    assert log == ["battery released"]


@pytest.mark.software
def test_replays_never_open_the_battery(cli_env, monkeypatch):
    got, tmp = cli_env
    monkeypatch.setattr(nl.battery_run, "open_battery", lambda: pytest.fail("opened the ADC on a replay"))
    assert nl.cli(["--video", "x.avi", "--out", str(tmp / "o")]) == 0
    assert got["battery"] is None


@pytest.mark.software
def test_no_motors_and_no_button_on_the_camera(cli_env):
    got, tmp = cli_env
    assert nl.cli(["--camera", "--no-motors", "--no-button", "--max-run-s", "7"]) == 0
    assert isinstance(got["motor"], lio.NoMotors) and got["motors_on"] is False
    assert got["system"] is None and got["max_run_s"] == 7.0


@pytest.mark.software
@pytest.mark.parametrize("flag", ["--video", "--frames"])
def test_replays_never_drive_the_motors_or_open_sensors(cli_env, flag):
    got, tmp = cli_env
    assert nl.cli([flag, str(tmp), "--limit", "3"]) == 0
    assert isinstance(got["motor"], lio.NoMotors) and got["motors_on"] is False
    assert got["sensors"] is None and got["system"] is None and got["limit"] == 3


@pytest.mark.software
def test_estimation_flags_reach_phase_3(cli_env):
    got, tmp = cli_env
    nl.cli(["--frames", str(tmp), "--gyro-bias", "0.7", "--cm-per-px", "0.05"])
    assert (got["p3_config"].gyro_bias_dps, got["p3_config"].cm_per_px) == (0.7, 0.05)
    assert dict(got["navigator"].rules)["intersection"].gyro_bias_dps == 0.7      # the heading hold too
    nl.cli(["--frames", str(tmp)])
    assert got["p3_config"].gyro_bias_dps == nl.MEASURED_ESTIMATION.gyro_bias_dps


@pytest.mark.software
def test_a_source_that_wont_open_is_exit_2(monkeypatch, capsys):
    def broken(*a, **k):
        raise OSError("no such file")
    monkeypatch.setattr(lio, "VideoFrameSource", broken)
    assert nl.cli(["--video", "missing.avi"]) == 2
    assert "no such file" in capsys.readouterr().out


@pytest.mark.software
def test_render_only_rebuilds_the_video_from_a_run_folder(tmp_path):
    rep, out, *_ = go(tmp_path)
    assert not (out / nl.VIDEO_FILE).exists()
    assert nl.cli(["--render", str(out)]) == 0
    cap = cv2.VideoCapture(str(out / nl.VIDEO_FILE))
    n = 0
    while cap.read()[0]:
        n += 1
    assert n == rep["run"]["frames"]


@pytest.mark.software
def test_the_stop_line_distance_is_logged_only_while_the_line_is_voted(tmp_path, monkeypatch):
    # No ground homography in the synthetic scenes, so the packet's cm distance is set here
    real = nl.run_phase3_chain

    def chain(frame, fid, *a, **k):
        res = real(frame, fid, *a, **k)
        line = {3: (True, 8.0), 4: (False, 5.0)}.get(fid, (False, None))
        return replace(res, packet=replace(res.packet, stop_line_detected=line[0], stop_line_distance_cm=line[1]))
    monkeypatch.setattr(nl, "run_phase3_chain", chain)
    _, out, *_ = go(tmp_path)
    logged = {int(r["frame_id"]): r["stop_line_cm"] for r in rows(out / "nav.csv")}
    assert logged[3] == "8.0" and logged[4] == "" and logged[2] == ""



@pytest.mark.software
def test_a_lane_that_stays_lost_ends_the_run_as_the_end_of_the_course(tmp_path):
    # The blind scene has no lane; Phase 3 holds, goes stale, the end-of-course rule creeps then finishes
    rep, out, motor, nav, camera = go(tmp_path, cam={"script": ("two_boundary",) * 5 + ("blind",), "end_at": 200})
    logged = rows(out / "nav.csv")
    assert rep["ended_by"] == nl.END_COURSE and camera.i < 200 and rep["outcome"] == "finished"
    assert logged[-1]["reason"] == "end_of_course" and logged[-1]["brake"] == "1"
    assert any(r["reason"] == "lane_stale_slow" for r in logged)
    assert motor.calls[-2:] == [("brake",), ("stop",)]
    assert "end of course" in (out / "summary.txt").read_text()


@pytest.mark.software
def test_a_lane_lost_before_the_route_is_done_ends_early_and_says_so(tmp_path):
    from src.navigation.route import Route
    nav = Spy(Navigation(route=Route(("left", "right"))))
    rep, out, motor, _, _ = go(tmp_path, nav=nav, cam={"script": ("two_boundary",) * 5 + ("blind",), "end_at": 200})
    assert rep["ended_by"] == nl.END_EARLY and (rep["outcome"], rep["end_step"]) == ("ended_early", 0)
    assert "ended early" in (out / "summary.txt").read_text() and "at step 0" in (out / "summary.txt").read_text()
    assert {r["step"] for r in rows(out / "nav.csv")} == {"0/2"}      # no stop line crossed
    assert motor.calls[-1] == ("stop",)


@pytest.mark.software
def test_the_route_file_reaches_the_navigator(cli_env, tmp_path):
    got, tmp = cli_env
    route = tmp_path / "r.json"
    route.write_text('{"maneuvers": ["left", "straight"], "finish": "stop_line"}')
    assert nl.cli(["--frames", str(tmp), "--route", str(route)]) == 0
    assert got["navigator"].progress.route.maneuvers == ("left", "straight")
    assert got["navigator"].progress.route.finish == "stop_line"


@pytest.mark.software
def test_a_bad_route_file_stops_before_anything_opens(monkeypatch, tmp_path, capsys):
    opened = []
    monkeypatch.setattr(lio, "DirectoryFrameSource", lambda *a, **k: opened.append(1))
    route = tmp_path / "r.json"
    route.write_text('{"maneuvers": ["lfet"]}')
    assert nl.cli(["--frames", str(tmp_path), "--route", str(route)]) == 2
    assert "route error" in capsys.readouterr().out and opened == []


@pytest.mark.software
def test_the_default_route_is_config_s(cli_env):
    got, tmp = cli_env
    from src.config import ROUTE_PATH
    from src.navigation.route import load_route
    nl.cli(["--frames", str(tmp)])
    assert got["navigator"].progress.route == load_route(ROUTE_PATH)



@pytest.mark.software
def test_the_report_keeps_the_clock_at_the_start_for_lining_up_other_records(tmp_path):
    rep, *_ = go(tmp_path)
    t = [float(r["t"]) for r in rows(tmp_path / "run" / "nav.csv")]
    assert rep["run"]["t0_monotonic"] == 100.0 and t[0] >= 0.0          # FakeClock starts at 100 s
