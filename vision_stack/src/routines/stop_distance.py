"""Stop distance: how far before a stop line the robot stops, and how consistently.

Purpose:
    Built from the request card's filled example (docs/routines/request_card.md,
    stop-distance). Each trial is a short run of the whole chain with the
    motors on (navigation_linker.run, as intersection_linker drives a
    sequence): from a start mark the robot keeps its lane toward an
    intersection with a stop sign (or a red light), and navigation brakes
    for the line. StopWatch follows the run and ends it once the robot is
    braked for the line and its wheels read stopped. The tester tapes the
    gap from the line's near edge to the bumper; the row keeps it beside
    what the robot last reported for the line (stop_line_cm), its speed as
    it began braking, and the pack's volts, since speed (and so the stop)
    falls with the battery.

    Pass (the card's first guesses, to refine after the first runs): it
    stopped before the line every trial, the mean gap within GAP_RANGE_CM,
    and the gaps' spread at most MAX_SPREAD_CM.

Main package:
    StopWatch: navigation_linker.run's stop_when for one approach.
    StopDistance: the routine (make routine-stop-distance).

Flow (per trial):
    1. The rig opened (camera, sensors, motors; no start button).
    2. The tester places the robot on the start mark: Enter drives.
    3. navigation_linker.run with a one-step straight route until StopWatch
       ends it (stopped at the line) or TRIAL_MAX_S; motors stop, the run
       folder lands in attempt_NN/ (a redone trial keeps both).
    4. The tester enters the taped gap (negative: past the line).
"""
from src.navigation.stop_sign import REASON_HOLD, REASON_STOPPING, STOPPED_CPS
from src.navigation.traffic_light import REASON_RED
from src.routines.harness import Routine, criterion, read_battery_volts, stats

STOP_REASONS = (REASON_STOPPING, REASON_HOLD, REASON_RED)
STOPPED = "stopped at the line"
# The robot counts as stopped once braked for the line with both wheels at or
# under STOPPED_CPS (stop_sign's own threshold) this long, or once the stop
# sign's hold has begun (the rule saw it stopped)
STILL_S = 0.3
TRIAL_MAX_S = 20.0          # a trial's backstop: the approach is a few seconds
START_GAP_CM = 60           # the card's start mark, before the stop line
GAP_RANGE_CM = (2.0, 6.0)   # the card: stop before the line, close enough to see the light
MAX_SPREAD_CM = 2.0         # the card: max - min of the gaps


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


class StopWatch:
    """
    Follows one approach's nav.csv rows; ends the run once the robot is
    stopped for the line, keeping what it reported on the way.

    Kept: reported_cm (the last stop_line_cm before braking began), brake_t
    (s into the run), speed_cps (the mean wheel speed on the last driving
    frame), reason (what it braked for), stopped (it ended the run).
    """
    def __init__(self, still_s: float = STILL_S, stopped_cps: float = STOPPED_CPS):
        self.still_s, self.stopped_cps = still_s, stopped_cps
        self.reported_cm = self.brake_t = self.speed_cps = self.reason = None
        self.stopped = False
        self._still = 0.0
        self._last_t = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        braking = bool(int(n.get("brake") or 0)) and n.get("reason") in STOP_REASONS
        if not braking:
            self._still = 0.0
            if self.brake_t is None:
                line = _num(n.get("stop_line_cm"))
                if line is not None:
                    self.reported_cm = line
                cps = [abs(_num(n.get(k)) or 0.0) for k in ("left_cps", "right_cps")]
                self.speed_cps = sum(cps) / 2
            return None
        if self.brake_t is None:
            self.brake_t, self.reason = t, n["reason"]
        still = all(abs(_num(n.get(k)) or 0.0) <= self.stopped_cps for k in ("left_cps", "right_cps"))
        self._still = self._still + dt if still else 0.0
        if n["reason"] == REASON_HOLD or self._still >= self.still_s:
            self.stopped = True
            return STOPPED
        return None


class StopDistance(Routine):
    name = "stop-distance"
    title = "Stopping distance at a stop line"
    question = "How far before a stop line does the robot stop, and how consistently?"
    requirement = "D4 (the navigation side)"
    trials = 10
    fields = ("gap_cm", "reported_cm", "speed_cps", "battery_v", "reason", "ended_by")
    needs = ("pigpiod",)
    settings = {"start_cm": f"how far before the line the start mark is (default {START_GAP_CM}); recorded",
                "max_s": f"each approach's backstop in s (default {TRIAL_MAX_S:.0f})"}
    instructions = f"""\
Set up: an intersection with its stop sign (or the light on red), a start mark
in the lane about {START_GAP_CM} cm before the stop line, room for the robot to stop.
Motors are ON: someone stands at the intersection, ready to catch it.
Each trial: place the robot on the start mark, centered and pointing along the
lane; press Enter; it drives and stops itself. Then tape the gap from the stop
line's near edge to the front of the bumper, along the lane's center
(negative if it stopped past the line). Carry it back to the mark between trials."""

    def __init__(self, open_rig=None, navigation_run=None, battery=read_battery_volts):
        self._open_rig, self._run, self._battery = open_rig, navigation_run, battery

    def setup(self, ctx):
        if self._open_rig is None:
            from src.linker_io import open_rig
            self._open_rig = open_rig
        if self._run is None:
            from src.navigation_linker import run
            self._run = run
        ctx.options.setdefault("start_cm", START_GAP_CM)
        ctx.options.setdefault("max_s", TRIAL_MAX_S)

    def trial(self, ctx, i):
        from src.config import MEASURED, MEASURED_ESTIMATION
        from src.navigation.navigation import Navigation
        from src.navigation.route import STRAIGHT, Route
        volts = self._battery()
        source, sensors, motor, system = self._open_rig(True, motors=True, button=False)
        try:
            ctx.console.wait("Robot on the start mark, centered? Enter drives it (q stops)")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop):
                close()
            raise
        watch = StopWatch()
        nav = Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps, route=Route((STRAIGHT,)))
        report = self._run(source, sensors, motor, nav, MEASURED, MEASURED_ESTIMATION,
                           str(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}"), None, max_run_s=float(ctx.options["max_s"]),
                           motors_on=True, render=False, stop_when=watch)
        ctx.console.say(f"  ended by {report['ended_by']}"
                        + (f"; braked for {watch.reason}" if watch.reason else "; it never braked for the line"))
        gap = ctx.console.ask_number("Gap from the stop line's near edge to the bumper (negative: past it)",
                                     lo=-100.0, hi=200.0, unit="cm")
        return {"gap_cm": gap, "reported_cm": watch.reported_cm,
                "speed_cps": None if watch.speed_cps is None else round(watch.speed_cps, 1), "battery_v": volts,
                "reason": watch.reason or "", "ended_by": report["ended_by"]}

    def judge(self, rows):
        gaps = stats(r["gap_cm"] for r in rows)
        before = sum(1 for r in rows if r["ended_by"] == STOPPED and r["gap_cm"] >= 0)
        return [criterion("trials stopped before the line", before, ">=", len(rows)),
                criterion("mean gap", gaps["mean"], "within", GAP_RANGE_CM, "cm"),
                criterion("spread (max - min)", gaps["max"] - gaps["min"], "<=", MAX_SPREAD_CM, "cm")]
