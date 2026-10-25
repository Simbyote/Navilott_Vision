"""Stop sign: a full stop at the sign's line, hold, drive on, and no stop without a sign?

Purpose:
    detect-range asks whether the parked robot sees the sign, stop-distance
    where it stops. This checks the whole behavior: brake, hold
    STOP_SIGN_HOLD_TIME_MS, cross, and cross without stopping when the sign
    is gone. sign_seen tells a missed stop's cause: False is the detector,
    True is navigation (docs/guides/routines.md section 2).

Main package:
    StopSign (make routine-stop-sign). SignWatch ends each run EXIT_DRIVE_S
    into the exit stage.
"""
from src.navigation.intersection import STAGE_EXIT
from src.navigation.navigation import RULE_STOP_SIGN
from src.navigation.stop_sign import REASON_HOLD, REASON_STOPPING, STOP_SIGN_HOLD_TIME_MS
from src.routines.harness import Routine, criterion, read_battery_volts

PRESENT, ABSENT = "present", "absent"
ACROSS = "across the intersection"
SIGN_TRIALS = 5
NO_SIGN_TRIALS = 3
START_GAP_CM = 60           # the start mark, before the stop line (stop-distance's)
TRIAL_MAX_S = 25.0          # backstop for approach, stop, hold, cross
EXIT_DRIVE_S = 1.0          # this long into the exit stage counts as across
HOLD_TOLERANCE_S = 0.3      # the hold, as nav.csv's frames time it, against the rule's


class SignWatch:
    """
    Follows one trial's nav.csv rows; ends EXIT_DRIVE_S into the exit stage.

    hold_s is the first stop's hold, from its first REASON_HOLD frame to the frame
    after it. across means it reached the exit stage.
    """
    def __init__(self, exit_drive_s: float = EXIT_DRIVE_S):
        self.exit_drive_s = exit_drive_s
        self.sign_seen = self.resumed = self.across = False
        self.stops = 0
        self.hold_s = None
        self._stopping = False
        self._hold_t = self._exit_t = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        if str(n.get("stop_sign")) in ("1", "True", "true"):
            self.sign_seen = True
        braking = (bool(int(n.get("brake") or 0)) and n.get("rule") == RULE_STOP_SIGN
                   and n.get("reason") in (REASON_STOPPING, REASON_HOLD))
        if braking:
            if not self._stopping:
                self.stops += 1
                self._stopping = True
            if n.get("reason") == REASON_HOLD and self._hold_t is None:
                self._hold_t = t
            return None
        if self._stopping:                      # let go
            self._stopping = False
            self.resumed = True
            if self._hold_t is not None and self.hold_s is None:
                self.hold_s = round(t - self._hold_t, 2)
        if n.get("stage") == STAGE_EXIT:
            self.across = True
            if self._exit_t is None:
                self._exit_t = t
            if t - self._exit_t >= self.exit_drive_s:
                return ACROSS
        return None


class StopSign(Routine):
    name = "stop-sign"
    title = "Stop sign: full stop, hold, drive on; no stop without the sign"
    question = ("Does the robot come to a full stop at a stop sign's line, hold, and drive on, "
                "and not stop when there's no sign?")
    requirement = "D2 (stop sign), the navigation side"
    trials = SIGN_TRIALS + NO_SIGN_TRIALS
    fields = ("sign", "sign_seen", "stops", "hold_s", "resumed", "across", "battery_v", "ended_by")
    needs = ("pigpiod",)
    settings = {"signs": f"trials with the sign posted (default {SIGN_TRIALS})",
                "no_signs": f"trials with the sign taken away (default {NO_SIGN_TRIALS})",
                "start_cm": f"start mark to the stop line, cm (default {START_GAP_CM}); recorded",
                "max_s": f"each trial's backstop, s (default {TRIAL_MAX_S:.0f})"}
    instructions = f"""\
Set up: an intersection as on the course, its stop sign posted where it will be
on the day, no traffic light lit at it (or the light green), room past the
intersection to drive across, a start mark about {START_GAP_CM} cm before the stop line.
Motors are ON: someone stands past the intersection, ready to catch the robot.
Each trial: place the robot on the mark, centered; press Enter. With the sign:
it should stop at the line, wait {STOP_SIGN_HOLD_TIME_MS / 1000:g} s, and drive across. After the sign
trials, take the sign away: it should drive across without stopping. Catch it
on the far side and carry it back to the mark."""

    def __init__(self, open_rig=None, navigation_run=None, battery=read_battery_volts):
        self._open_rig, self._run, self._battery = open_rig, navigation_run, battery

    @staticmethod
    def _defaults(options: dict) -> None:
        options.setdefault("signs", SIGN_TRIALS)
        options.setdefault("no_signs", NO_SIGN_TRIALS)
        options.setdefault("start_cm", START_GAP_CM)
        options.setdefault("max_s", TRIAL_MAX_S)

    def plan(self, options):
        o = dict(options)
        self._defaults(o)
        return int(float(o["signs"])) + int(float(o["no_signs"]))

    def setup(self, ctx):
        self._defaults(ctx.options)
        if self._open_rig is None:
            from src.linker_io import open_rig
            self._open_rig = open_rig
        if self._run is None:
            from src.navigation_linker import run
            self._run = run

    def trial(self, ctx, i):
        from src.config import MEASURED, MEASURED_ESTIMATION
        from src.navigation.navigation import Navigation
        from src.navigation.route import STRAIGHT, Route
        with_sign = i < int(float(ctx.options["signs"]))
        volts = self._battery()
        source, sensors, motor, system = self._open_rig(True, motors=True, button=False)
        try:
            ctx.console.wait("Robot on the mark, the stop sign POSTED? Enter drives it (q stops)" if with_sign
                             else "Robot on the mark, the stop sign TAKEN AWAY? Enter drives it (q stops)")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop):
                close()
            raise
        watch = SignWatch()
        nav = Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps, route=Route((STRAIGHT,)))
        report = self._run(source, sensors, motor, nav, MEASURED, MEASURED_ESTIMATION,
                           str(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}"), None,
                           max_run_s=float(ctx.options["max_s"]), motors_on=True, render=False, stop_when=watch)
        ctx.console.say(f"  ended by {report['ended_by']}; sign {'seen' if watch.sign_seen else 'never seen'}, "
                        f"stops {watch.stops}" + ("" if watch.hold_s is None else f", held {watch.hold_s:g} s"))
        return {"sign": PRESENT if with_sign else ABSENT, "sign_seen": watch.sign_seen, "stops": watch.stops,
                "hold_s": watch.hold_s, "resumed": watch.resumed, "across": watch.across, "battery_v": volts,
                "ended_by": report["ended_by"]}

    def judge(self, rows):
        signs = [r for r in rows if r["sign"] == PRESENT]
        none = [r for r in rows if r["sign"] == ABSENT]
        out = []
        if signs:
            hold = STOP_SIGN_HOLD_TIME_MS / 1000
            holds = [r["hold_s"] for r in signs]
            worst = max((abs(h - hold) for h in holds if h is not None), default=None)
            out += [criterion("sign trials it didn't stop at", sum(1 for r in signs if r["stops"] < 1), "<=", 0),
                    criterion(f"worst hold off {hold:g} s", worst if all(h is not None for h in holds) else None,
                              "<=", HOLD_TOLERANCE_S, "s"),
                    criterion("sign trials it didn't drive on across",
                              sum(1 for r in signs if not (r["resumed"] and r["across"])), "<=", 0)]
        if none:
            out += [criterion("stops for a sign that wasn't there", sum(r["stops"] for r in none), "<=", 0),
                    criterion("no-sign trials it didn't drive across", sum(1 for r in none if not r["across"]),
                              "<=", 0)]
        return out
