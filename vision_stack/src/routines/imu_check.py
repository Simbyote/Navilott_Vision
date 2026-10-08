"""IMU check: is the gyro's bias still the configured one, does a real 90 degrees read 90, and what do the motors add.

Purpose:
    Every turn ends when the gyro says TURN_TARGET_DEG, and every heading
    hold subtracts config.GYRO_BIAS_DPS (measured once, 2026-09-29) before
    integrating. If the bias has moved, the heading drifts (0.3 deg/s is
    18 deg a minute); if the gyro's scale is off, a turn ends early or late;
    if IMU_YAW_SIGN is wrong, it turns the wrong way. test_imu --hardware
    measures bias and noise; this checks them against what the robot uses,
    through the same sensor hub, plus what the bench can't: real turns.

    Ten trials, three parts:
        rest (trial 1)        REST_S still on the floor: bias against
                              GYRO_BIAS_DPS, noise, the heading's drift over
                              the time with the configured bias, read rate,
                              read time, errors, temperature
        turns (trials 2-9)    the tester turns the robot 90 deg by hand,
                              lined up on the mat's grid, alternating left
                              and right: the gyro's heading (net of the rest
                              bias) against -90 / +90; the sign and the scale
        vibration (trial 10)  VIB_S wheels up at base duty: the noise with
                              the motors on, against rest

    Pass: the bias within BIAS_TOLERANCE_DPS of GYRO_BIAS_DPS; the rest
    drift under DRIFT_MAX_DEG; every turn within TURN_TOLERANCE_DEG of 90
    with the right sign; no read errors. The vibration noise is reported,
    not judged. The summary says the bias to put in config.py, the left and
    right scale, and whether IMU_YAW_SIGN looks flipped (both directions the
    wrong way).

Main package:
    ImuCheck: the routine (make routine-imu-check).
    rest_stats(readings, configured_bias), heading(readings, bias): the numbers.

Flow (per trial):
    rest: drain, wait REST_S, drain; turns: Enter at the start line-up,
    turn, Enter at the end, drain; vibration: wheels-up motors on a thread
    for VIB_S, drain. One sensor hub (IMU only) for the whole routine, with
    HISTORY_S of buffer so a part is one drain.
"""
import statistics
import threading
import time

from src.config import GYRO_BIAS_DPS
from src.navigation.lane_keeping import BASE_SPEED
from src.routines.harness import Routine, criterion

REST_S = 60.0
VIB_S = 30.0
TURNS = 8
HISTORY_S = 600.0           # the hub keeps this much between drains: a slow hand turn still fits
BIAS_TOLERANCE_DPS = 0.2    # 0.2 deg/s is 12 deg a minute of heading hold
DRIFT_MAX_DEG = 2.0         # over the rest minute, with the configured bias
TURN_TOLERANCE_DEG = 3.0    # hand turns lined up on the grid are good to ~1-2 deg
TURN_DEG = 90.0
PARTS = ("rest",) + ("turn",) * TURNS + ("vibration",)


def _pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))] if s else None


def _reads(readings) -> list:
    """The readings that tried the IMU (imu_ms set): the drain's closing encoder reading isn't one."""
    return [r for r in readings if r.imu_ms is not None]


def heading(readings, bias: float) -> float:
    """Degrees turned over the readings, net of bias, + = right (the hub's frame): trapezoids between readings."""
    pts = [(r.t, r.yaw_dps - bias) for r in readings if r.yaw_dps is not None]
    return sum((t1 - t0) * (y0 + y1) / 2 for (t0, y0), (t1, y1) in zip(pts, pts[1:]))


def rest_stats(readings, configured_bias: float = GYRO_BIAS_DPS) -> dict:
    """At rest: bias (mean yaw), noise (sd), drift with the configured bias, rate, read time, errors."""
    reads = _reads(readings)
    yaws = [r.yaw_dps for r in reads if r.yaw_dps is not None]
    span = reads[-1].t - reads[0].t if len(reads) > 1 else 0.0
    return {"reads": len(reads), "rate_hz": round((len(reads) - 1) / span, 1) if span > 0 else None,
            "read_errors": sum(1 for r in reads if r.yaw_dps is None),
            "read_ms_p95": None if not reads else round(_pct([r.imu_ms for r in reads], 0.95), 2),
            "bias_dps": round(statistics.fmean(yaws), 3) if yaws else None,
            "noise_dps": round(statistics.stdev(yaws), 3) if len(yaws) > 1 else None,
            "drift_deg": round(heading(reads, configured_bias), 2) if yaws else None}


class ImuCheck(Routine):
    name = "imu-check"
    title = "IMU check: bias, turns, vibration"
    question = "Is the gyro's bias still the configured one, does a real 90 deg read 90, and what do the motors add?"
    requirement = "heading for turns and holds (GYRO_BIAS_DPS, IMU_YAW_SIGN, TURN_TARGET_DEG)"
    trials = len(PARTS)
    fields = ("part", "direction", "seconds", "bias_dps", "noise_dps", "drift_deg", "heading_deg", "error_deg",
              "reads", "rate_hz", "read_ms_p95", "read_errors", "temp_c")
    needs = ("pigpiod",)
    settings = {"rest_s": f"the rest part's length in s (default {REST_S:.0f})",
                "vib_s": f"the vibration part's length in s (default {VIB_S:.0f})",
                "duty": f"the wheels' duty in the vibration part (default {BASE_SPEED:g}, base speed)"}
    instructions = f"""\
Three parts, ten trials. 1: the robot still on the floor for {REST_S:.0f} s (don't touch
the table). 2-9: {TURNS} hand turns of {TURN_DEG:.0f} deg, alternating left and right: line the
robot up on the mat's grid, press Enter, turn it by hand (lift it if needed,
keep it flat), line it up again exactly 90 deg on, press Enter. 10: wheels up on
a box, motors at base speed for {VIB_S:.0f} s. A redo repeats the same part."""

    def __init__(self, hub=None, motors=None, sleep=time.sleep, temperature=None):
        self._hub, self._motors, self._sleep, self._temperature = hub, motors, sleep, temperature

    def setup(self, ctx):
        ctx.options.setdefault("rest_s", REST_S)
        ctx.options.setdefault("vib_s", VIB_S)
        if self._hub is None:
            from src.peripherals.sensing import SensorHub
            self._hub = SensorHub.open(imu=True, history_s=HISTORY_S)
            self._hub.start()
        if self._motors is None:
            from src.routines.power_profile import _motors
            self._motors = _motors
        if self._temperature is None:
            from src.diagnostics.system_monitor import read_temp_c
            self._temperature = read_temp_c
        ctx.state["bias_measured"] = None

    def teardown(self, ctx):
        if self._hub is not None and hasattr(self._hub, "stop"):
            self._hub.stop()
        bias = ctx.state.get("bias_measured")
        turns = ctx.state.get("turns", {})
        if bias is not None:
            ctx.console.say(f"\nmeasured bias {bias:+.3f} deg/s (config.GYRO_BIAS_DPS is {GYRO_BIAS_DPS:+.2f})"
                            + ("" if abs(bias - GYRO_BIAS_DPS) <= BIAS_TOLERANCE_DPS else
                               f": put GYRO_BIAS_DPS = {bias:.2f} in config.py"))
        means = {d: statistics.fmean(hs) for d, hs in turns.items() if hs}
        if means.get("left", 0) > 0 and means.get("right", 0) < 0:
            ctx.state["sign_flipped"] = True
            ctx.console.say("both directions read the wrong way: flip IMU_YAW_SIGN in config.py")
        for d in ("left", "right"):
            hs = turns.get(d)
            if hs:
                scale = statistics.fmean(abs(h) for h in hs) / TURN_DEG
                ctx.state[f"scale_{d}"] = round(scale, 4)
                ctx.console.say(f"{d} turns read {100 * scale:.1f}% of the real {TURN_DEG:.0f} deg "
                                f"(mean {statistics.fmean(hs):+.1f} over {len(hs)})")

    def trial(self, ctx, i):
        part = PARTS[i]
        if part == "rest":
            return self._rest(ctx)
        if part == "turn":
            return self._turn(ctx, i)
        return self._vibration(ctx)

    def _row(self, part, **kw) -> dict:
        row = {k: None for k in self.fields}
        row.update(part=part, temp_c=self._temperature(), **kw)
        return row

    def _rest(self, ctx):
        seconds = float(ctx.options["rest_s"])
        ctx.console.wait(f"Robot still on the floor, nothing touching it? Enter starts {seconds:.0f} s at rest")
        self._hub.drain()
        self._sleep(seconds)
        s = rest_stats(self._hub.drain().readings)
        ctx.state["bias_measured"] = s["bias_dps"]
        ctx.state["rest_noise_dps"] = s["noise_dps"]
        return self._row("rest", seconds=seconds, **s)

    def _turn(self, ctx, i):
        direction = "left" if i % 2 == 1 else "right"
        expected = -TURN_DEG if direction == "left" else TURN_DEG          # + = turned right
        ctx.console.wait(f"Line it up on the grid; Enter, then turn it {TURN_DEG:.0f} deg {direction.upper()}")
        self._hub.drain()
        ctx.console.wait(f"Lined up again {TURN_DEG:.0f} deg {direction}? Enter")
        reads = _reads(self._hub.drain().readings)
        bias = ctx.state.get("bias_measured")
        bias = GYRO_BIAS_DPS if bias is None else bias
        h = heading(reads, bias)
        ctx.state.setdefault("turns", {}).setdefault(direction, []).append(round(h, 2))
        span = reads[-1].t - reads[0].t if len(reads) > 1 else 0.0
        return self._row("turn", direction=direction, seconds=round(span, 1), heading_deg=round(h, 2),
                         error_deg=round(h - expected, 2), reads=len(reads),
                         read_errors=sum(1 for r in reads if r.yaw_dps is None))

    def _vibration(self, ctx):
        seconds = float(ctx.options["vib_s"])
        ctx.console.wait(f"WHEELS UP on a box? Enter runs the motors for {seconds:.0f} s")
        stop = threading.Event()
        errors = []

        def wheels():
            try:
                self._motors(ctx, seconds, stop)
            except Exception as exc:                # the row says so
                errors.append(exc)
        t = threading.Thread(target=wheels, name="imu-check-wheels", daemon=True)
        self._hub.drain()
        t.start()
        try:
            self._sleep(seconds)
        finally:
            stop.set()
            t.join(timeout=5.0)
        s = rest_stats(self._hub.drain().readings)
        if errors:
            ctx.console.say(f"  the motors failed: {errors[0]!r}")
            ctx.state["vibration_error"] = repr(errors[0])
        rest_noise = ctx.state.get("rest_noise_dps")
        if rest_noise and s["noise_dps"] is not None:
            ctx.state["vibration_noise_x"] = round(s["noise_dps"] / rest_noise, 2)
        return self._row("vibration", seconds=seconds, bias_dps=s["bias_dps"], noise_dps=s["noise_dps"],
                         reads=s["reads"], rate_hz=s["rate_hz"], read_ms_p95=s["read_ms_p95"],
                         read_errors=s["read_errors"])

    def judge(self, rows):
        rest = next((r for r in reversed(rows) if r["part"] == "rest"), None)
        turns = [r for r in rows if r["part"] == "turn"]
        bias_off = None if rest is None or rest["bias_dps"] is None else abs(rest["bias_dps"] - GYRO_BIAS_DPS)
        bad_turns = sum(1 for r in turns if abs(r["error_deg"]) > TURN_TOLERANCE_DEG)   # the wrong way is 180 off
        return [criterion("bias off the configured", bias_off, "<=", BIAS_TOLERANCE_DPS, "deg/s"),
                criterion("rest drift (configured bias)", None if rest is None or rest["drift_deg"] is None
                          else abs(rest["drift_deg"]), "<=", DRIFT_MAX_DEG, "deg"),
                criterion(f"turns off 90 by over {TURN_TOLERANCE_DEG:g} deg or the wrong way", bad_turns, "<=", 0),
                criterion("read errors", sum(r["read_errors"] or 0 for r in rows), "<=", 0)]
