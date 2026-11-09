"""Power profile: how pack voltage, Pi load and heat respond to each part of the robot running.

Purpose:
    The ADS1115 measures volts, not current, so there are no watts. Per stage
    (rest, camera, pipeline, motors, full) this gives the sag against rest,
    the drain, whether the pack stays above warning, and supply dips or
    throttling.

Main package:
    PowerProfile (make routine-power-profile): one row per stage, every
    sample in samples.csv.
"""
import csv
import statistics
import threading
import time
from pathlib import Path

from src.peripherals.battery import Power
from src.navigation.lane_keeping import BASE_SPEED
from src.routines.harness import Routine, criterion

STAGE_S = 60.0              # per stage: long enough for the pack to settle at the new load
SAMPLE_S = 0.5              # sampler period; the ADS1115 read itself is ~9 ms
MOTOR_REFRESH_S = 0.1       # under the motor watchdog
STAGES = (
    ("rest", "Nothing to do: the baseline, with only this routine running."),
    ("camera", "The camera captures for the stage; nothing moves."),
    ("pipeline", "The whole chain runs with the motors off; point the camera at the course."),
    ("motors", "WHEELS UP (on a box): the wheels turn at base speed; no camera."),
    ("full", "WHEELS UP: the whole chain and the wheels together; camera at the course."),
)
SAMPLE_FIELDS = ("attempt", "stage", "t_s", "battery_v", "cpu_pct", "temp_c", "under_voltage", "throttled")
THROTTLES = ("freq_capped", "throttled", "soft_temp_limit")


def read_cpu_times(path: Path = Path("/proc/stat")) -> tuple[int, int] | None:
    """(busy, total) jiffies from /proc/stat; None where unreadable."""
    try:
        f = [int(x) for x in path.read_text().splitlines()[0].split()[1:]]
    except (OSError, ValueError, IndexError):
        return None
    idle = f[3] + (f[4] if len(f) > 4 else 0)            # idle + iowait
    return sum(f) - idle, sum(f)


def cpu_busy(prev, cur) -> float | None:
    if prev is None or cur is None or cur[1] <= prev[1]:
        return None
    return round(100.0 * (cur[0] - prev[0]) / (cur[1] - prev[1]), 1)


def _slope_per_min(ts: list[float], vs: list[float]) -> float | None:
    """Least-squares volts per minute; None under 2 points or with no time spread."""
    if len(ts) < 2:
        return None
    mt, mv = statistics.fmean(ts), statistics.fmean(vs)
    den = sum((t - mt) ** 2 for t in ts)
    return None if den == 0 else 60.0 * sum((t - mt) * (v - mv) for t, v in zip(ts, vs)) / den


def stage_row(name: str, samples: list[dict], rest_v: float | None) -> dict:
    """
    Pack volts (mean, min), sag against rest, drain mV/min (rough over a minute), CPU, temperature, throttle flags.
    """
    v = [s["battery_v"] for s in samples if s["battery_v"] is not None]
    tv = [(s["t_s"], s["battery_v"]) for s in samples if s["battery_v"] is not None]
    cpu = [s["cpu_pct"] for s in samples if s["cpu_pct"] is not None]
    temp = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    mean_v = round(statistics.fmean(v), 3) if v else None
    slope = _slope_per_min([t for t, _ in tv], [x for _, x in tv])
    return {"stage": name, "seconds": round(samples[-1]["t_s"], 1) if samples else 0.0,
            "v_mean": mean_v, "v_min": round(min(v), 3) if v else None,
            "sag_v": None if mean_v is None or rest_v is None else round(rest_v - mean_v, 3),
            "drain_mv_min": None if slope is None else round(-1000.0 * slope, 1),
            "cpu_pct": round(statistics.fmean(cpu), 1) if cpu else None,
            "temp_max_c": max(temp) if temp else None,
            "under_voltage": int(any(s["under_voltage"] for s in samples)),
            "throttled": int(any(s["throttled"] for s in samples))}


class PowerProfile(Routine):
    name = "power-profile"
    title = "Power profile: the pack's voltage under each load"
    question = "How far does the pack sag, and does the Pi stay unthrottled, under each part of the robot running?"
    requirement = "R4 (load), and the battery's margin to its warning level"
    trials = len(STAGES)
    fields = ("stage", "seconds", "v_mean", "v_min", "sag_v", "drain_mv_min", "cpu_pct", "temp_max_c",
              "under_voltage", "throttled")
    needs = ("pigpiod",)
    settings = {"stage_s": f"seconds per stage (default {STAGE_S:.0f}); longer gives a steadier drain figure",
                "duty": f"the wheels' duty in the motors and full stages (default {BASE_SPEED})"}
    instructions = f"""\
Five stages, one per trial: {', '.join(n for n, _ in STAGES)}. Start with the pack
charged and the robot cool. The motors and full stages turn the wheels: put the
robot on a box with the wheels off the ground before those. A redo repeats the
stage; q stops after keeping the stage just run."""

    def __init__(self, loads: dict | None = None, read=None, power_factory=Power,
                 clock=time.monotonic, sleep=time.sleep):
        self._loads, self._read, self._power_factory = loads, read, power_factory
        self._clock, self._sleep = clock, sleep
        self._power = None

    def setup(self, ctx):
        ctx.options.setdefault("stage_s", STAGE_S)
        ctx.options.setdefault("duty", BASE_SPEED)
        if self._read is None:
            from src.peripherals import battery_run
            self._power = battery_run.open_battery(say=ctx.console.say, factory=self._power_factory)
            if self._power is None:
                ctx.console.say("  no battery ADC: CPU, heat and throttling still record, the volts don't")
            self._read = self._read_pi
        self._loads = self._loads or self._real_loads()
        ctx.state["samples"] = []
        self._cpu_prev = read_cpu_times()

    def _read_pi(self) -> dict:
        from src.diagnostics.system_monitor import sample
        s = sample()
        volts = None
        if self._power is not None:
            try:
                volts = round(self._power.voltage_raw(), 3)
            except Exception:           # a failed ADC read: that sample has no volts
                volts = None
        cur = read_cpu_times()
        busy, self._cpu_prev = cpu_busy(self._cpu_prev, cur), cur
        return {"battery_v": volts, "cpu_pct": busy, "temp_c": s.get("temp_c"),
                "under_voltage": int(bool(s.get("under_voltage"))),
                "throttled": int(any(s.get(k) for k in THROTTLES))}

    def teardown(self, ctx):
        if ctx.state.get("samples"):
            with open(Path(ctx.out_dir) / "samples.csv", "w", newline="") as f:
                w = csv.DictWriter(f, SAMPLE_FIELDS)
                w.writeheader()
                w.writerows(ctx.state["samples"])
            ctx.state["samples_file"] = "samples.csv"
            ctx.state["samples_count"] = len(ctx.state.pop("samples"))
        if self._power is not None:
            self._power.cleanup()

    def trial(self, ctx, i):
        name, what = STAGES[i % len(STAGES)]           # i: the trial being filled, so a redo repeats its stage
        ctx.console.say(f"  stage {name}: {what}")
        ctx.console.wait(f"Enter starts {name} for {float(ctx.options['stage_s']):.0f} s (q stops)")
        seconds = float(ctx.options["stage_s"])
        stop = threading.Event()
        errors = []

        def load():
            try:
                self._loads[name](ctx, seconds, stop)
            except Exception as exc:        # the stage's row says so; the samples still count
                errors.append(exc)
        worker = threading.Thread(target=load, name=f"load-{name}", daemon=True)
        samples, t0 = [], self._clock()
        worker.start()
        try:
            while self._clock() - t0 < seconds:
                samples.append({"attempt": ctx.attempt + 1, "stage": name, "t_s": round(self._clock() - t0, 2),
                                **self._read()})
                self._sleep(SAMPLE_S)
        finally:
            stop.set()
            worker.join(timeout=10.0)
        ctx.state["samples"] += samples
        # sag against the latest rest stage
        rest = None if name == "rest" else next(
            (s["v_mean"] for s in reversed(ctx.state.get("rows", [])) if s["stage"] == "rest"), None)
        row = stage_row(name, samples, rest)
        ctx.state.setdefault("rows", []).append(row)
        if errors:
            ctx.console.say(f"  the {name} load failed: {errors[0]!r}")
            ctx.state.setdefault("load_errors", {})[name] = repr(errors[0])
        return row

    def _real_loads(self) -> dict:
        return {"rest": _rest, "camera": _camera, "pipeline": _pipeline, "motors": _motors, "full": _full}

    def judge(self, rows):
        loads = [r for r in rows if r["stage"] != "rest"]
        lows = [r["v_min"] for r in loads if r["v_min"] is not None]
        return [criterion("lowest pack volts under load", min(lows) if lows else None, ">=",
                          Power.VOLTAGE_WARNING, "V"),
                criterion("stages with the Pi's under-voltage", sum(r["under_voltage"] for r in rows), "<=", 0),
                criterion("stages throttled", sum(r["throttled"] for r in rows), "<=", 0)]


# Loads run until stop is set or seconds pass


def _rest(ctx, seconds, stop):
    stop.wait(seconds)


def _camera(ctx, seconds, stop):
    from src.debugger.live_view import CameraFrameSource
    from src.params import CAMERA_CONTROLS, FPS, FRAME_H, FRAME_W
    source = CameraFrameSource(FRAME_W, FRAME_H, FPS, CAMERA_CONTROLS)
    try:
        while not stop.is_set():
            source.read()
    finally:
        source.close()


def _pipeline(ctx, seconds, stop, motors=False):
    from src.config import MEASURED, MEASURED_ESTIMATION
    from src.linker_io import open_rig
    from src.navigation.navigation import Navigation
    from src.navigation_linker import run
    source, sensors, motor, _ = open_rig(True, motors=motors, button=False)
    run(source, sensors, motor, Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps), MEASURED,
        MEASURED_ESTIMATION, str(Path(ctx.out_dir) / "pipeline"), None, max_run_s=seconds + 5.0, motors_on=motors,
        render=False, stop_when=lambda n: "stage over" if stop.is_set() else None)


def _motors(ctx, seconds, stop):
    import pigpio
    from src.peripherals.drive import MotorController
    duty = float(ctx.options.get("duty", BASE_SPEED))
    pi = pigpio.pi()
    motor = MotorController(pi)
    try:
        while not stop.is_set():
            motor.drive(duty, duty)
            stop.wait(MOTOR_REFRESH_S)
    finally:
        motor.stop()
        pi.stop()


def _full(ctx, seconds, stop):
    wheels = threading.Thread(target=_motors, args=(ctx, seconds, stop), name="load-wheels", daemon=True)
    wheels.start()
    try:
        _pipeline(ctx, seconds, stop)
    finally:
        stop.set()
        wheels.join(timeout=5.0)
