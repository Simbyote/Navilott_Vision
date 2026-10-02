"""
test_main.py  --  src/main.py

The production run end to end on a synthetic course with fake hardware:
the start screen and the button, the motors getting exactly the pipeline's
commands, every way a run ends (finished, ended early, the cap, Ctrl-C,
an error) with the motors halted first, the end screens, and the command
line's startup checks. Production records nothing and imports no linker
or debugger.

--software  run() / cli() with fakes. No camera, motors or GPIO.
"""
import subprocess
import sys
import types

import pytest

import src.main as main
from src.capture.camera import CaptureError, FrameData
from src.navigation.navigation import BRAKE, Command
from src.navigation.route import Route
from src.params import PIPELINE_ROOT
from src.pipeline import Pipeline
from src.tests.scenes import SCENE_CONFIG, course_sequence
from src.tests.sim_robot import FakeClock

COURSE = course_sequence()
DT = 0.05
# A cap between frame times, so float steps can't move the frame it ends on: 21 reads reach it
CAP_S, CAP_FRAMES = 1.025, 21


class Camera:
    """CameraSource stand-in: the course's frames in order, advancing the clock DT per read; None past the end."""
    def __init__(self, clock, log, sequence=COURSE, drop_at=(), raise_at=None, exc=KeyboardInterrupt):
        self.clock, self.log, self.sequence = clock, log, list(sequence)
        self.drop_at, self.raise_at, self.exc = drop_at, raise_at, exc
        self.reads, self.i = 0, 0

    def read(self):
        self.clock.now += DT
        self.reads += 1
        if self.reads == self.raise_at:
            raise self.exc
        if self.reads in self.drop_at:
            return None
        sf = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        self.current = sf
        return FrameData(sf.frame, sf.frame_id, sf.timestamp_ms)

    def release(self):
        self.log.append("camera released")


class Sensors:
    """sensing.Sensors stand-in: the current frame's readings."""
    def __init__(self, camera, log):
        self.camera, self.log, self.reads = camera, log, 0

    def sample(self):
        self.reads += 1
        sf = getattr(self.camera, "current", None)
        return None if sf is None else sf.sensors

    def stop(self):
        self.log.append("sensors stopped")


class Motor:
    """MotorController stand-in: the Command each frame drove, then the halt, into the shared log."""
    def __init__(self, log, brake_fails=False):
        self.log, self.commands, self.brake_fails = log, [], brake_fails

    def drive(self, left, right):
        self.commands.append(Command(left, right))

    def brake(self):
        if self.brake_fails:
            raise OSError("brake failed")
        self.commands.append(BRAKE)
        self.log.append("brake")

    def stop(self):
        self.log.append("motor stop")


class System:
    """peripherals.system.System stand-in: records the screens."""
    def __init__(self, log, interrupt_wait=False):
        self.log, self.interrupt_wait = log, interrupt_wait
        self.screens, self.times = [], []

    def wait_for_start(self, text="rdy "):
        self.screens.append(text)
        if self.interrupt_wait:
            raise KeyboardInterrupt

    def run_countdown(self):
        self.log.append("countdown")

    def update_display(self, elapsed_s):
        self.times.append(elapsed_s)

    def show_final_time(self, elapsed_s):
        self.screens.append(round(elapsed_s, 2))

    def show_text(self, text):
        self.screens.append(text)

    def cleanup(self, blank=True):
        self.log.append(f"display released (blank={blank})")


def go(route=Route(("left",)), max_run_s=main.MAX_RUN_S, system_kw=None, motor_kw=None, **camera_kw):
    clock, log = FakeClock(), []
    camera = Camera(clock, log, **camera_kw)
    sensors, motor, system = Sensors(camera, log), Motor(log, **(motor_kw or {})), System(log, **(system_kw or {}))
    pipeline = Pipeline(SCENE_CONFIG, route=route)
    sleeps = []
    result = main.run(camera, sensors, motor, pipeline, system, route, clock=clock, sleep=sleeps.append,
                      max_run_s=max_run_s)
    return result, motor, system, log, pipeline, sleeps


def stepped_commands(route, n):
    """Pipeline.step() alone over the first n course frames: what the motors should get."""
    pipeline = Pipeline(SCENE_CONFIG, route=route)
    return [pipeline.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors) for sf in COURSE[:n]]


# =============================================================================
# A run
# =============================================================================

@pytest.mark.software
def test_a_finished_course_drives_the_pipelines_commands_then_halts():
    result, motor, system, log, pipeline, sleeps = go()
    assert result.ended_by == main.END_FINISHED and result.end_step == 1
    assert motor.commands[:-1] == stepped_commands(Route(("left",)), result.frames)
    assert motor.commands[-1] == BRAKE and sleeps == [main.HALT_BRAKE_S]           # the halt's brake
    assert log[0] == "countdown" and log[-4:] == ["brake", "motor stop", "sensors stopped", "camera released"]
    assert log.count("motor stop") == 1
    assert result.frames < len(COURSE) and pipeline.finished
    assert result.elapsed_s == pytest.approx(result.frames * DT)


@pytest.mark.software
def test_the_start_screen_is_the_step_count_and_the_time_runs_every_frame():
    result, _, system, _, _, _ = go(route=Route(("left", "straight", "right")))
    assert system.screens[0] == "St 3"
    assert len(system.times) == result.frames and system.times == sorted(system.times)


@pytest.mark.software
def test_a_lane_lost_before_the_route_is_done_ends_early_at_its_step():
    result, motor, _, log, _, _ = go(route=Route(("left", "right")))
    assert (result.ended_by, result.end_step) == (main.END_EARLY, 1)
    assert log[-4:] == ["brake", "motor stop", "sensors stopped", "camera released"]


@pytest.mark.software
def test_the_run_time_cap_ends_it_and_halts():
    result, motor, _, log, _, _ = go(max_run_s=CAP_S)
    assert result.ended_by == main.END_CAP and result.frames == CAP_FRAMES and result.end_step == 0
    assert log[-4:] == ["brake", "motor stop", "sensors stopped", "camera released"]


@pytest.mark.software
def test_ctrl_c_halts_the_motors_first():
    result, motor, _, log, _, _ = go(raise_at=10)
    assert (result.ended_by, result.frames, result.error) == (main.END_INTERRUPT, 9, None)
    assert log[-4:] == ["brake", "motor stop", "sensors stopped", "camera released"]


@pytest.mark.software
def test_ctrl_c_at_the_start_screen_never_drives():
    result, motor, _, log, _, _ = go(system_kw={"interrupt_wait": True})
    assert (result.ended_by, result.frames, result.elapsed_s) == (main.END_INTERRUPT, 0, 0.0)
    assert motor.commands == [BRAKE] and "countdown" not in log and log[-1] == "camera released"


@pytest.mark.software
def test_an_error_halts_the_motors_and_is_kept():
    result, motor, _, log, _, _ = go(raise_at=5, exc=CaptureError("camera died"))
    assert result.ended_by == main.END_ERROR and isinstance(result.error, CaptureError)
    assert log[-4:] == ["brake", "motor stop", "sensors stopped", "camera released"]
    assert "camera died" in main.summary(result)


@pytest.mark.software
def test_dropped_frames_are_skipped_without_a_step():
    result, motor, _, _, _, _ = go(max_run_s=CAP_S, drop_at=(3, 4))
    assert result.frames == CAP_FRAMES - 2
    assert motor.commands[:-1] == stepped_commands(Route(("left",)), CAP_FRAMES - 2)


@pytest.mark.software
def test_the_sensors_start_a_window_at_go_then_one_per_frame():
    clock, log = FakeClock(), []
    camera = Camera(clock, log)
    sensors = Sensors(camera, log)
    result = main.run(camera, sensors, Motor(log), Pipeline(SCENE_CONFIG), System(log), Route(),
                      clock=clock, sleep=lambda s: None, max_run_s=1.0)
    assert sensors.reads == result.frames + 1


@pytest.mark.software
def test_halt_brakes_then_stands_by_and_stops_even_if_the_brake_fails():
    log, sleeps = [], []
    main.halt(Motor(log), sleeps.append)
    assert log == ["brake", "motor stop"] and sleeps == [main.HALT_BRAKE_S]
    log = []
    with pytest.raises(OSError):
        main.halt(Motor(log, brake_fails=True), sleeps.append)
    assert log == ["motor stop"]


# =============================================================================
# The screens
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("route, text", [(Route(), "St 0"), (Route(("left",) * 3), "St 3"),
                                         (Route(("left",) * 12), "St12"), (Route(("left",) * 120), "St99")])
def test_step_text(route, text):
    assert main.step_text(route) == text


@pytest.mark.software
@pytest.mark.parametrize("step, code", [(0, "E  0"), (2, "E  2"), (13, "E 13"), (None, "E  0"), (150, "E 99")])
def test_early_code(step, code):
    assert main.early_code(step) == code


@pytest.mark.software
@pytest.mark.parametrize("ended_by, screens", [
    (main.END_FINISHED, [42.5]), (main.END_INTERRUPT, [42.5]), (main.END_ERROR, ["Err "]),
    (main.END_EARLY, ["E  2", 42.5]), (main.END_CAP, ["E  t", 42.5])])
def test_end_screen(ended_by, screens):
    assert main.end_screen(main.RunResult(ended_by, 42.5, 100, 2)) == screens


@pytest.mark.software
def test_one_screen_is_shown_once_and_two_alternate_until_ctrl_c():
    system, sleeps = System([]), []
    main.show_end(system, [61.0], sleeps.append)
    assert system.screens == [61.0] and sleeps == []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) == 5:
            raise KeyboardInterrupt
    main.show_end(system, ["E  2", 61.0], sleep)
    assert system.screens == [61.0, "E  2", 61.0, "E  2", 61.0, "E  2"]
    assert sleeps == [main.ALTERNATE_S] * 5


@pytest.mark.software
def test_mm_ss_and_the_summary():
    assert (main.mm_ss(65.9), main.mm_ss(0), main.mm_ss(6000)) == ("01:05", "00:00", "99:00")
    assert main.summary(main.RunResult(main.END_EARLY, 65.0, 1300, 2)) == "ended early at step 2 after 01:05 (1300 frames)"


# =============================================================================
# The command line
# =============================================================================

@pytest.fixture
def hardware(monkeypatch):
    """Fake pigpio, drive, sensing, system and camera; returns what each opened and the log."""
    log, made = [], {}
    pigpio = types.ModuleType("pigpio")
    pigpio.pi = lambda: "pi"
    drive = types.ModuleType("src.peripherals.drive")
    drive.MotorController = lambda pi: made.setdefault("motor", Motor(log))
    sensing = types.ModuleType("src.peripherals.sensing")
    sensing.Sensors = lambda **kw: made.setdefault("sensors", types.SimpleNamespace(
        stop=lambda: log.append("sensors stopped"), kw=kw))
    system = types.ModuleType("src.peripherals.system")
    system.System = lambda: made.setdefault("system", System(log, interrupt_wait=True))
    for name, mod in (("pigpio", pigpio), ("src.peripherals.drive", drive),
                      ("src.peripherals.sensing", sensing), ("src.peripherals.system", system)):
        monkeypatch.setitem(sys.modules, name, mod)

    class Cam:
        def __init__(self, w, h, fps):
            made["camera"] = self
            self.size = (w, h, fps)

        def open(self):
            if made.get("camera_fails"):
                raise CaptureError("no camera")
            return self

        def release(self):
            log.append("camera released")
    monkeypatch.setattr(main, "CameraSource", Cam)
    return made, log


@pytest.mark.software
def test_cli_prints_the_route_and_runs_it_leaving_the_last_screen_up(hardware, tmp_path, capsys):
    made, log = hardware
    route = tmp_path / "r.json"
    route.write_text('{"maneuvers": ["left", "straight"], "finish": "stop_line"}')
    assert main.cli(["--route", str(route)]) == 0
    out = capsys.readouterr().out
    assert "Route: 2 maneuvers" in out and "press the start button" in out and "interrupted" in out
    assert made["system"].screens == ["St 2", 0.0]                    # Ctrl-C at the start screen: time 00:00
    assert made["sensors"].kw == {"imu": True, "encoders": True}
    assert log[0] == "motor stop" and log[-1] == "display released (blank=False)"


@pytest.mark.software
def test_cli_bad_route_exits_2_before_any_hardware(hardware, tmp_path, capsys):
    made, _ = hardware
    route = tmp_path / "r.json"
    route.write_text('{"maneuvers": ["lfet"]}')
    assert main.cli(["--route", str(route)]) == 2
    assert "route error" in capsys.readouterr().out and made == {}


@pytest.mark.software
def test_cli_hardware_that_wont_open_exits_2_and_releases_the_rest(hardware, capsys):
    made, log = hardware
    made["camera_fails"] = True
    assert main.cli([]) == 2
    assert "hardware error" in capsys.readouterr().out
    assert log[-3:] == ["sensors stopped", "motor stop", "display released (blank=True)"]


@pytest.mark.software
def test_cli_defaults_to_config_route_cap_and_gyro_bias(hardware, monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "run", lambda *a, **kw: seen.update(pipeline=a[3], route=a[5], **kw) or main.RunResult(
        main.END_FINISHED, 1.0, 20, 0))
    assert main.cli([]) == 0
    from src.config import GYRO_BIAS_DPS, ROUTE_PATH
    from src.navigation.route import load_route
    assert seen["route"] == load_route(ROUTE_PATH) and seen["max_run_s"] == main.MAX_RUN_S
    pipeline = seen["pipeline"]                                        # the gyro bias the linkers default to
    assert pipeline.estimation.gyro_bias_dps == pipeline.navigation._crossing.gyro_bias_dps == GYRO_BIAS_DPS


@pytest.mark.software
def test_production_imports_no_linker_or_debugger():
    probe = "import sys, src.main; print(sorted(m for m in sys.modules if m.startswith('src.')))"
    out = subprocess.run([sys.executable, "-c", probe], cwd=PIPELINE_ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert not [m for m in eval(out) if m.startswith("src.debugger") or m.endswith("_linker")]


@pytest.mark.software
def test_a_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    go()
    assert list(tmp_path.iterdir()) == []
