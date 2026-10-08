"""Recover offset: from how far off the lane's center does the robot steer back, and how well does it hold a straight?

Purpose:
    Ignacio's two lane-keeping tests from the IDR (slide 10), on the robot:
    the maximum recoverable offset, and the mean distance from the lane's
    center on a straight run. Each trial is a short run of the whole chain,
    motors on (navigation_linker.run, as stop-distance drives), on a
    straight lane, from a start offset the tester measured with a ruler.
    RecoverWatch follows the run through the robot's own lane_offset_cm:
    that's a measurement only once lane-offset has passed (P3), so this
    needs the ground scale lane-offset measured (--set cm_per_px=..., or
    cm_per_px set in config.py).

    A start of 0 is the straight run: STRAIGHT_S of lane keeping, reporting
    the mean and largest distance from the center. Any other start ends
    once the robot's offset has stayed within BAND_CM for HOLD_S (or at
    max_s); the tester then measures where it stopped. The ruler decides
    whether it recovered (|end| within BAND_CM); the robot's trace gives
    how long it took and how far it overshot.

    Pass (first guesses): every start up to need_cm recovered; the straight
    run never more than BAND_CM off the center (by the robot's reading).
    The summary gives the maximum recoverable offset: the largest start
    up to which every trial, left and right, recovered.

Main package:
    RecoverWatch: navigation_linker.run's stop_when for one trial.
    RecoverOffset: the routine (make routine-recover-offset).

Flow (per trial):
    1. The rig opened; the tester places the robot at the planned offset,
       pointing along the lane, and enters the offset measured.
    2. Enter drives: one run with a straight route until RecoverWatch ends
       it; the run folder lands in attempt_NN/.
    3. The tester measures where it stopped.
"""
from src.routines.detect_range import reliable_range
from src.routines.harness import Routine, criterion, read_battery_volts

POSITIONS_CM = "0,2,-2,3,-3,4,-4,5,-5"     # + = robot right of the lane's center; 0: the straight run
# The robot is 11 cm wide in a 14 cm lane (line center to line center): 1.5 cm a side before its
# body is over a line's center. Within this it counts as centered
BAND_CM = (14.0 - 11.0) / 2
HOLD_S = 1.0                # centered this long (by its own reading) ends a recovery trial
MAX_S = 6.0                 # a recovery trial's backstop
STRAIGHT_S = 5.0            # the straight run's length in time
NEED_CM = 3.0               # every start up to this must recover (first guess)
RECOVERED, TIME_UP = "recovered", "time up"


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


class RecoverWatch:
    """
    Follows one run's nav.csv rows through lane_offset_cm on vision.

    straight: run until max_s and keep the offset's mean and max; else end
        once the offset has stayed within band_cm for hold_s.
    start_cm: the first offset on vision. recover_s: from the first row to
        the start of the hold that ended it. overshoot_cm: the farthest past
        the center on the far side from the start. lane_lost_s: time off
        vision.
    """
    def __init__(self, straight: bool, band_cm: float = BAND_CM, hold_s: float = HOLD_S, max_s: float = MAX_S):
        self.straight, self.band, self.hold_s, self.max_s = straight, band_cm, hold_s, max_s
        self.start_cm = self.recover_s = None
        self.overshoot_cm = 0.0
        self.offsets: list[float] = []
        self.lane_lost_s = 0.0
        self.t = self._t0 = self._last_t = self._in_since = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        if self._t0 is None:
            self._t0 = t
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = self.t = t
        off = _num(n.get("lane_offset_cm"))
        if n.get("lane_status") != "vision" or off is None:
            self.lane_lost_s += dt
        else:
            if self.start_cm is None:
                self.start_cm = off
            self.offsets.append(off)
            if self.start_cm and (off > 0) != (self.start_cm > 0):
                self.overshoot_cm = max(self.overshoot_cm, abs(off))
            if not self.straight:
                if abs(off) <= self.band:
                    self._in_since = t if self._in_since is None else self._in_since
                    if t - self._in_since >= self.hold_s:
                        self.recover_s = round(self._in_since - self._t0, 2)
                        return RECOVERED
                else:
                    self._in_since = None
        return TIME_UP if t - self._t0 >= self.max_s else None

    def mean_abs(self) -> float | None:
        return sum(abs(o) for o in self.offsets) / len(self.offsets) if self.offsets else None

    def max_abs(self) -> float | None:
        return max((abs(o) for o in self.offsets), default=None)


class RecoverOffset(Routine):
    name = "recover-offset"
    title = "Recover offset: steering back to the lane's center, and holding a straight"
    question = ("From how far off the lane's center does the robot steer back, "
                "and how close to the center does it hold a straight?")
    requirement = "lane keeping (IDR navigation tests: maximum recoverable offset, mean offset on a straight)"
    trials = len(POSITIONS_CM.split(","))
    fields = ("planned_cm", "start_cm", "end_cm", "recovered", "robot_start_cm", "recover_s", "overshoot_cm",
              "mean_abs_cm", "max_abs_cm", "lane_lost_s", "ended_by", "battery_v")
    needs = ("pigpiod",)
    settings = {"positions": f"start offsets in cm, + = right; 0 is the straight run (default {POSITIONS_CM})",
                "cm_per_px": "the ground scale lane-offset measured (default: config.py's)",
                "band_cm": f"within this of the center counts as centered (default {BAND_CM:g})",
                "need_cm": f"every start up to this must recover (default {NEED_CM:g})",
                "max_s": f"a recovery trial's backstop in s (default {MAX_S:g})",
                "straight_s": f"the straight run's length in s (default {STRAIGHT_S:g})"}
    instructions = """\
Run lane-offset first: this measures with the robot's own offset reading, and
needs the scale it gives (--set cm_per_px=...). Set up: a long straight lane
(1.5 m or more), both lines clear. Motors are ON: someone walks beside it.
Each trial: place the robot at the planned offset from the lane's center
(+ = RIGHT), pointing along the lane (not angled back toward the center);
enter what you measured; Enter drives. It stops itself once it's held the
center (or after a few seconds). Then measure where it stopped, the same way.
A start of 0 is the straight run: it drives a few seconds and stops."""

    def __init__(self, open_rig=None, navigation_run=None, battery=read_battery_volts, cm_per_px=None):
        self._open_rig, self._run, self._battery = open_rig, navigation_run, battery
        self._config_scale = cm_per_px
        self._band, self._need = BAND_CM, NEED_CM

    @staticmethod
    def _positions(options) -> list[float]:
        from src.routines.camera_look import parse_numbers
        return parse_numbers(options.get("positions", POSITIONS_CM))

    def plan(self, options):
        return len(self._positions(options))

    def setup(self, ctx):
        from src.config import MEASURED_ESTIMATION
        if self._open_rig is None:
            from src.linker_io import open_rig
            self._open_rig = open_rig
        if self._run is None:
            from src.navigation_linker import run
            self._run = run
        o = ctx.options
        o.setdefault("positions", POSITIONS_CM)
        o.setdefault("cm_per_px", self._config_scale or MEASURED_ESTIMATION.cm_per_px)
        o.setdefault("band_cm", BAND_CM)
        o.setdefault("need_cm", NEED_CM)
        o.setdefault("max_s", MAX_S)
        o.setdefault("straight_s", STRAIGHT_S)
        if not o["cm_per_px"]:
            raise ValueError("no ground scale: run make routine-lane-offset, then --set cm_per_px=<the value it gives>")
        self._band, self._need = float(o["band_cm"]), float(o["need_cm"])
        ctx.state["results"] = {}

    def teardown(self, ctx):
        done = list(ctx.state.get("results", {}).values())
        moved = [(abs(p), "ok" if ok else "no") for p, ok in done if p != 0]
        if moved:
            best = reliable_range(moved)
            ctx.state["max_recoverable_cm"] = best
            ctx.console.say("\nmaximum recoverable offset: " + (
                "none: it didn't recover from the smallest start" if best is None else
                f"{best:g} cm (every start up to it recovered, both sides)"))

    def trial(self, ctx, i):
        from dataclasses import replace
        from src.config import MEASURED, MEASURED_ESTIMATION
        from src.navigation.navigation import Navigation
        from src.navigation.route import STRAIGHT, Route
        planned = self._positions(ctx.options)[i]
        straight = planned == 0
        volts = self._battery()
        source, sensors, motor, _ = self._open_rig(True, motors=True, button=False)
        try:
            where = "centered in the lane" if straight else \
                f"{abs(planned):g} cm {'RIGHT' if planned > 0 else 'LEFT'} of center"
            ctx.console.wait(f"Robot {where}, pointing along the lane? Enter")
            start = ctx.console.ask_number("Offset you measured (+ = right of center)", lo=-15.0, hi=15.0, unit="cm")
            ctx.console.wait("Enter drives it (q stops)")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop):
                close()
            raise
        o = ctx.options
        watch = RecoverWatch(straight, self._band, HOLD_S, float(o["straight_s"] if straight else o["max_s"]))
        p3 = replace(MEASURED_ESTIMATION, cm_per_px=float(o["cm_per_px"]))
        nav = Navigation(gyro_bias_dps=p3.gyro_bias_dps, route=Route((STRAIGHT,)))
        report = self._run(source, sensors, motor, nav, MEASURED, p3,
                           str(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}"), None,
                           max_run_s=watch.max_s + 5.0, motors_on=True, render=False, stop_when=watch)
        ctx.console.say(f"  ended by {report['ended_by']}"
                        + (f" after {watch.recover_s:g} s to the center" if watch.recover_s is not None else ""))
        end = ctx.console.ask_number("Where it stopped: offset you measured (+ = right)", lo=-15.0, hi=15.0, unit="cm")
        recovered = abs(end) <= self._band
        ctx.state["results"][str(i)] = (planned, recovered)
        r2 = lambda v: None if v is None else round(v, 2)          # noqa: E731
        return {"planned_cm": planned, "start_cm": start, "end_cm": end, "recovered": recovered,
                "robot_start_cm": r2(watch.start_cm), "recover_s": watch.recover_s,
                "overshoot_cm": r2(watch.overshoot_cm), "mean_abs_cm": r2(watch.mean_abs()),
                "max_abs_cm": r2(watch.max_abs()), "lane_lost_s": r2(watch.lane_lost_s),
                "ended_by": report["ended_by"], "battery_v": volts}

    def judge(self, rows):
        moving = [r for r in rows if r["planned_cm"] != 0]
        straight = [r for r in rows if r["planned_cm"] == 0]
        out = [criterion(f"starts up to {self._need:g} cm not recovered",
                         sum(1 for r in moving if abs(r["planned_cm"]) <= self._need and not r["recovered"]), "<=", 0)]
        if straight:
            worst = None if any(r["max_abs_cm"] is None for r in straight) else max(r["max_abs_cm"] for r in straight)
            out.append(criterion("straight run: farthest from the center", worst, "<=", self._band, "cm"))
        return out
