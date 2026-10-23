"""Obstacle stop: does the robot stop for an object in its lane, where, and drive on once it's taken away?

Purpose:
    obstacle-check asks what the parked robot sees; this asks what the
    driving robot does. The obstacle rule should brake with the object about
    OBSTACLE_STOP_CM away, hold while it's there, and drive on
    OBSTACLE_CLEAR_MS after it's lifted away. Clear trials drive the empty
    lane: any obstacle brake is a false alarm. Pass: docs/guides/routines.md,
    section 2.

Main package:
    ObstacleStop: the routine (make routine-obstacle-stop). ObstacleWatch
    ends each run RESUME_DRIVE_S after the robot drives on.

Flow (per trial):
    1. The rig opened (camera, sensors, motors; no start button).
    2. The tester places the robot on the start mark (and the object, or
       clears the lane): Enter drives.
    3. navigation_linker.run with a one-step straight route until the watch
       ends it or max_s; the run folder lands in attempt_NN/.
    4. Object trials: the tester enters the taped gap (0: it touched).
"""
from src.navigation.navigation import RULE_OBSTACLE
from src.navigation.obstacle import OBSTACLE_CLEAR_MS, OBSTACLE_STOP_CM, REASON_CLEARING, REASON_OBSTACLE
from src.navigation.stop_sign import STOPPED_CPS
from src.routines.harness import Routine, criterion, read_battery_volts
from src.routines.stop_distance import _num

OBJECT, CLEAR = "object", "clear"
RESUMED, CLEAR_DONE = "drove on", "clear lane driven"
OBJECT_TRIALS = 5
CLEAR_TRIALS = 2
START_GAP_CM = 60           # the start mark, before the object
TRIAL_MAX_S = 30.0          # a trial's backstop: the approach, the tape, the lift
CLEAR_S = 6.0               # a clear trial drives this long
RESUME_DRIVE_S = 1.0        # after driving on, this much more ends the run
STILL_S = 0.3               # braked with both wheels at or under STOPPED_CPS this long: stopped
GAP_RANGE_CM = (5.0, 15.0)  # the rule aims to end ~10 cm from the object (its stop distance less the coast)
RESUME_SLACK_S = 0.5        # driving on may take the clear time plus this


def _obstacle_cm(v):
    """nav.csv's obstacle_cm: (cm or None, clipped)."""
    if v == "view bottom":
        return None, True
    return _num(v), False


class ObstacleWatch:
    """
    Follows one trial's nav.csv rows; ends the run RESUME_DRIVE_S after the
    robot drives on from an obstacle stop (object trials), or after clear_s
    of driving (clear trials).

    Kept: brakes (obstacle brake episodes: one for a good object trial,
    false alarms on a clear one), reported_cm (the voted obstacle cm on the
    last frame before the first brake; None when clipped or without cm),
    clipped (that frame's obstacle was below the view), stopped (wheels
    still while braked for it), clear_t (the rule began waiting out the
    clear time: the object was gone), resumed_t (it drove on), speed_cps
    (mean wheel speed on the last frame before braking).
    """
    def __init__(self, with_object: bool, clear_s: float = CLEAR_S, resume_drive_s: float = RESUME_DRIVE_S,
                 still_s: float = STILL_S, stopped_cps: float = STOPPED_CPS):
        self.with_object, self.clear_s, self.resume_drive_s = with_object, clear_s, resume_drive_s
        self.still_s, self.stopped_cps = still_s, stopped_cps
        self.brakes = 0
        self.reported_cm = self.speed_cps = self.clear_t = self.resumed_t = None
        self.clipped = self.stopped = False
        self._braking = False
        self._still = 0.0
        self._last_t = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        braking = (bool(int(n.get("brake") or 0)) and n.get("rule") == RULE_OBSTACLE
                   and n.get("reason") in (REASON_OBSTACLE, REASON_CLEARING))
        if braking:
            if not self._braking:
                self.brakes += 1
                self._braking = True
            still = all(abs(_num(n.get(k)) or 0.0) <= self.stopped_cps for k in ("left_cps", "right_cps"))
            self._still = self._still + dt if still else 0.0
            if self._still >= self.still_s:
                self.stopped = True
            if n.get("reason") == REASON_CLEARING and self.clear_t is None:
                self.clear_t = t
            return None
        if self._braking:                           # let go: driving on
            self._braking, self._still = False, 0.0
            if self.resumed_t is None:
                self.resumed_t = t
        if self.brakes == 0:                        # still approaching: what it saw and how fast
            self.reported_cm, self.clipped = _obstacle_cm(n.get("obstacle_cm"))
            self.speed_cps = sum(abs(_num(n.get(k)) or 0.0) for k in ("left_cps", "right_cps")) / 2
        if self.with_object and self.resumed_t is not None and t - self.resumed_t >= self.resume_drive_s:
            return RESUMED
        if not self.with_object and t >= self.clear_s:
            return CLEAR_DONE
        return None

    @property
    def resume_s(self) -> float | None:
        """From the object going (the clear wait starting) to driving on."""
        if self.clear_t is None or self.resumed_t is None:
            return None
        return round(self.resumed_t - self.clear_t, 2)


class ObstacleStop(Routine):
    name = "obstacle-stop"
    title = "Obstacle stop: brake for an object in the lane, then drive on"
    question = "Does the robot stop for an object in its lane, where, and drive on once it's taken away?"
    requirement = "obstacle detection (stretch goal: stop for an obstacle)"
    trials = OBJECT_TRIALS + CLEAR_TRIALS
    fields = ("state", "gap_cm", "reported_cm", "clipped", "brakes", "stopped", "resume_s", "speed_cps",
              "battery_v", "ended_by")
    needs = ("pigpiod",)
    settings = {"objects": f"trials with the object (default {OBJECT_TRIALS})",
                "clear": f"trials with the lane clear (default {CLEAR_TRIALS})",
                "start_cm": f"start mark to the object, cm (default {START_GAP_CM}); recorded",
                "clear_s": f"how long a clear trial drives, s (default {CLEAR_S:g})",
                "max_s": f"each trial's backstop, s (default {TRIAL_MAX_S:.0f})",
                "object": "what the object is, recorded (e.g. can, box)"}
    instructions = f"""\
Set up: a straight lane with room to drive, a start mark, and the object (a can,
a box) standing in the middle of the lane about {START_GAP_CM} cm ahead of the mark.
Motors are ON: someone stands by the object, ready to catch the robot.
Object trials: place the robot on the mark, centered; press Enter; it drives and
should stop by itself about {OBSTACLE_STOP_CM - 2:g} cm short of the object, and wait there.
While it waits, tape the gap from its front edge to the object's nearest point
and remember it; then lift the object straight up and away. It should drive on
by itself about {OBSTACLE_CLEAR_MS / 1000:g} s later. Catch it and enter the gap (0 if it touched).
Clear trials (after the object trials): nothing on the lane; Enter; it drives
for a few seconds and must not stop. Keep people's feet out of its view."""

    def __init__(self, open_rig=None, navigation_run=None, battery=read_battery_volts):
        self._open_rig, self._run, self._battery = open_rig, navigation_run, battery

    @staticmethod
    def _defaults(options: dict) -> None:
        options.setdefault("objects", OBJECT_TRIALS)
        options.setdefault("clear", CLEAR_TRIALS)
        options.setdefault("start_cm", START_GAP_CM)
        options.setdefault("clear_s", CLEAR_S)
        options.setdefault("max_s", TRIAL_MAX_S)
        options.setdefault("object", "")

    def plan(self, options):
        o = dict(options)
        self._defaults(o)
        return int(float(o["objects"])) + int(float(o["clear"]))

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
        with_object = i < int(float(ctx.options["objects"]))
        volts = self._battery()
        source, sensors, motor, system = self._open_rig(True, motors=True, button=False)
        try:
            ctx.console.wait("Robot on the mark, the object in the lane? Enter drives it (q stops)" if with_object
                             else "Robot on the mark, the lane CLEAR? Enter drives it (q stops)")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop):
                close()
            raise
        watch = ObstacleWatch(with_object, clear_s=float(ctx.options["clear_s"]))
        nav = Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps, route=Route((STRAIGHT,)))
        report = self._run(source, sensors, motor, nav, MEASURED, MEASURED_ESTIMATION,
                           str(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}"), None,
                           max_run_s=float(ctx.options["max_s"]), motors_on=True, render=False, stop_when=watch)
        ctx.console.say(f"  ended by {report['ended_by']}; obstacle brakes {watch.brakes}"
                        + ("" if watch.resume_s is None else f"; drove on {watch.resume_s:g} s after it went"))
        gap = None
        if with_object:
            gap = ctx.console.ask_number("Gap from the front edge to the object when it stopped (0: touched)",
                                         lo=0.0, hi=200.0, unit="cm")
        return {"state": OBJECT if with_object else CLEAR, "gap_cm": gap, "reported_cm": watch.reported_cm,
                "clipped": watch.clipped, "brakes": watch.brakes, "stopped": watch.stopped,
                "resume_s": watch.resume_s, "speed_cps": None if watch.speed_cps is None else round(watch.speed_cps, 1),
                "battery_v": volts, "ended_by": report["ended_by"]}

    def judge(self, rows):
        obj = [r for r in rows if r["state"] == OBJECT]
        clear = [r for r in rows if r["state"] == CLEAR]
        out = []
        if obj:
            lo, hi = GAP_RANGE_CM
            resumes = [r["resume_s"] for r in obj]
            limit = OBSTACLE_CLEAR_MS / 1000 + RESUME_SLACK_S
            out += [criterion("object trials it didn't stop for (or touched)",
                              sum(1 for r in obj if not r["stopped"] or not r["gap_cm"]), "<=", 0),
                    criterion(f"gaps outside {lo:g}..{hi:g} cm", sum(1 for r in obj if not lo <= (r["gap_cm"] or 0) <= hi),
                              "<=", 0),
                    criterion("object trials it didn't drive on from", sum(1 for r in resumes if r is None), "<=", 0),
                    criterion("slowest drive-on after the object went",
                              max(resumes) if all(r is not None for r in resumes) else None, "<=", limit, "s")]
        if clear:
            out.append(criterion("false brakes on a clear lane", sum(r["brakes"] for r in clear), "<=", 0))
        return out

