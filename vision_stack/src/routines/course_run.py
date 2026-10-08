"""Course run: the Design Day rehearsal, the whole course from the start button to the finish, judged.

Purpose:
    D2's competition is one thing: complete a predetermined course at the
    best possible speed, keeping the lane and obeying the lights and signs.
    Each trial is that, once: the route file (config.ROUTE_PATH unless
    set), the start button and countdown, the motors on, the battery
    watched (navigation_linker.run, recording every frame). CourseWatch
    follows the run: the route steps reached, the stop sign holds and red
    light waits, time off vision, contract brakes. Afterwards the tester
    says what a judge would: did it complete the course as planned, how
    many times they touched it, how many stop signs or red lights it ran,
    and the stopwatch time from GO to the finish. The stopwatch against the
    robot's own run time also checks the elapsed-time display.

    Pass (each run): completed (the tester says so and the robot ended on
    the route's finish); no touches; no stop sign or red light run. The
    times are the result, not judged: the summary gives the best and mean.

Main package:
    CourseWatch: navigation_linker.run's stop_when for one run (never ends it).
    CourseRun: the routine (make routine-course-run).

Flow (per trial):
    1. Rig (camera, sensors, motors, start button) and battery opened;
       the robot on the start; Enter arms it.
    2. The start button, the countdown, the run until the route's finish
       (or ended early, the cap, a critical battery, Ctrl-C).
    3. The tester's answers; the row. The run folder is attempt_NN/.
"""
import statistics
from pathlib import Path

from src.navigation.stop_sign import REASON_HOLD
from src.navigation.traffic_light import REASON_RED
from src.routines.figure_eight import _num, _step
from src.routines.harness import Routine, criterion

MAX_S = 300.0               # main.MAX_RUN_S: the production run's cap
FINISHED = "end of course"  # navigation_linker.END_COURSE: the navigator finished the route


class CourseWatch:
    """
    Follows one run's nav.csv rows; never ends it.

    steps: the highest route step reached. holds, reds: how many times the
    run came to a stop sign's hold, and to a wait at a red light.
    lane_lost_s: time in lane keeping off vision. contract: frames braked
    because a command broke the contract. volts: the pack, per frame.
    """
    def __init__(self):
        self.steps = self.holds = self.reds = self.contract = self.frames = 0
        self.lane_lost_s = 0.0
        self.volts: list[float] = []
        self._last_t = self._last_reason = None

    def __call__(self, n: dict) -> None:
        t = float(n["t"])
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        self.frames += 1
        self.steps = max(self.steps, _step(n.get("step")))
        reason = n.get("reason")
        if reason != self._last_reason:
            self.holds += reason == REASON_HOLD
            self.reds += reason == REASON_RED
        self._last_reason = reason
        if n.get("rule") == "lane_keeping" and n.get("lane_status") != "vision":
            self.lane_lost_s += dt
        self.contract += reason == "contract"
        v = _num(n.get("battery_v"))
        if v is not None:
            self.volts.append(v)
        return None


class CourseRun(Routine):
    name = "course-run"
    title = "Course run: the Design Day rehearsal"
    question = "Does the robot complete the course, in its lane and obeying the lights and signs, and how fast?"
    requirement = "D2 (Senior Design Day: the course at best speed, lane and traffic adherence)"
    trials = 3
    fields = ("completed", "ended_by", "stopwatch_s", "robot_s", "clock_diff_s", "steps_done", "steps_planned",
              "touches", "violations", "holds", "reds", "lane_lost_s", "contract", "fps",
              "battery_start_v", "battery_end_v")
    needs = ("pigpiod",)
    settings = {"route": "the route file (default config.ROUTE_PATH)",
                "max_s": f"each run's cap in s (default {MAX_S:.0f}, the production run's)"}
    instructions = """\
Set up the course as on the day: the route file's maneuvers, its lights and
stop signs, lit as on the day, a charged pack. Motors are ON: walk beside it.
Each run: put the robot on the start, press Enter here to arm it, then press
the ROBOT'S START BUTTON; start a phone stopwatch when the countdown ends (GO)
and stop it at the finish. Count every time you touch it, and every stop sign
or red light it doesn't stop for. Ctrl-C stops it at once."""

    def __init__(self, open_rig=None, navigation_run=None, open_battery=None):
        self._open_rig, self._run, self._open_battery = open_rig, navigation_run, open_battery

    def setup(self, ctx):
        from src.config import ROUTE_PATH
        from src.navigation.route import load_route
        if self._open_rig is None:
            from src.linker_io import open_rig
            self._open_rig = open_rig
        if self._run is None:
            from src.navigation_linker import run
            self._run = run
        if self._open_battery is None:
            from src.diagnostics import battery_run
            self._open_battery = lambda: battery_run.open_battery(say=ctx.console.say)
        ctx.options.setdefault("route", str(ROUTE_PATH))
        ctx.options.setdefault("max_s", MAX_S)
        self._route = load_route(ctx.options["route"])          # a bad route stops here, before the rig opens
        ctx.console.say(f"route {ctx.options['route']}: {', '.join(self._route.maneuvers)} "
                        f"({len(self._route.maneuvers)} steps), finish {self._route.finish}")
        ctx.state["times"] = {}

    def teardown(self, ctx):
        done = [s for ok, s in ctx.state.get("times", {}).values() if ok and s is not None]
        if done:
            ctx.state["best_s"], ctx.state["mean_s"] = min(done), round(statistics.fmean(done), 2)
            ctx.console.say(f"\ncompleted runs: {len(done)}; best {min(done):g} s, mean {ctx.state['mean_s']:g} s "
                            "(stopwatch)")

    def trial(self, ctx, i):
        from src.config import MEASURED, MEASURED_ESTIMATION
        from src.navigation.navigation import Navigation
        source, sensors, motor, system = self._open_rig(True, motors=True, button=True)
        battery = self._open_battery()
        try:
            ctx.console.wait("Robot on the start? Enter arms it; then press its start button")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop, getattr(system, "cleanup", None),
                          getattr(battery, "cleanup", None)):
                if close is not None:
                    close()
            raise
        watch = CourseWatch()
        nav = Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps, route=self._route)
        try:
            report = self._run(source, sensors, motor, nav, MEASURED, MEASURED_ESTIMATION,
                               str(Path(ctx.out_dir) / f"attempt_{ctx.attempt + 1:02d}"), system,
                               max_run_s=float(ctx.options["max_s"]), motors_on=True, render=False,
                               stop_when=watch, battery=battery)
        finally:
            if battery is not None:
                battery.cleanup()
        robot_s = report.get("run", {}).get("wall_s")
        ctx.console.say(f"  ended by {report['ended_by']} at step {watch.steps} of {len(self._route.maneuvers)}"
                        + (f" after {robot_s:g} s" if robot_s is not None else ""))
        completed = ctx.console.ask_yes("Did it complete the course as planned (every maneuver, to the finish)?")
        stopwatch = ctx.console.ask_number("Stopwatch time, GO to the finish (0 if it didn't finish)",
                                           lo=0.0, hi=3600.0, unit="s")
        touches = int(ctx.console.ask_number("Times you touched it", lo=0, hi=1000))
        violations = int(ctx.console.ask_number("Stop signs or red lights it didn't stop for", lo=0, hi=1000))
        ok = completed and report["ended_by"] == FINISHED
        ctx.state["times"][str(i)] = (ok, stopwatch if stopwatch > 0 else None)
        wall = robot_s or 0.0
        return {"completed": ok, "ended_by": report["ended_by"], "stopwatch_s": stopwatch, "robot_s": robot_s,
                "clock_diff_s": None if robot_s is None or not stopwatch else round(robot_s - stopwatch, 2),
                "steps_done": watch.steps, "steps_planned": len(self._route.maneuvers), "touches": touches,
                "violations": violations, "holds": watch.holds, "reds": watch.reds,
                "lane_lost_s": round(watch.lane_lost_s, 2), "contract": watch.contract,
                "fps": round(watch.frames / wall, 1) if wall else None,
                "battery_start_v": watch.volts[0] if watch.volts else None,
                "battery_end_v": watch.volts[-1] if watch.volts else None}

    def judge(self, rows):
        out = []
        for k, r in enumerate(rows, 1):
            tag = f" (run {k})" if len(rows) > 1 else ""
            out += [criterion(f"completed the course{tag}", 1 if r["completed"] else 0, ">=", 1),
                    criterion(f"touches{tag}", r["touches"], "<=", 0),
                    criterion(f"stop signs or red lights run{tag}", r["violations"], "<=", 0)]
        return out
