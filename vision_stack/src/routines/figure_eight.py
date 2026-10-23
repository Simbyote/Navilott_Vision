"""Figure 8: the whole robot on the course, left four times then right four times, lap after lap, for minutes.

Purpose:
    Stop distance asks how accurately one maneuver lands; this asks whether
    the robot keeps working: every rule (lane keeping, stop lines, left and
    right turns, exits) over many intersections, the left and right turns
    side by side (an asymmetry shows), and what builds up over minutes (the
    pack sagging, heat). On a grid, four lefts loop around one block and
    four rights around the next: a figure 8 through the intersection the
    two loops share, where the robot starts. One trial is one continuous
    run (navigation_linker.run, motors on) with the route LEFT x4, RIGHT x4
    repeated, ended after `minutes`, or by Ctrl-C, a critical battery, an
    error or the lane lost (end_of_course: off the loop). EightWatch follows
    every frame: each intersection's maneuver, how its turn ended (the gyro
    or the time limit), the heading turned against -90 / +90 (the heading
    the turn ended on plus what the exit drifted from it), whether the
    lane came back, each lap's time and pack volts. The tester presses
    Enter in the terminal at each touch (TouchKeys), logged with where the
    robot was; afterwards they say how many times they touched it and how
    many laps they saw.

    Pass: no touches; ran the whole time; the laps the robot counted are
    the laps seen; every finished turn ended on the gyro, within
    HEADING_TOLERANCE_DEG of its maneuver (intersection_linker's tolerance).
    The pack and the heat are reported, not judged: power-profile judges
    them.

Main package:
    EightWatch: navigation_linker.run's stop_when for one run.
    TouchKeys: Enter presses on the terminal during the run, one per touch.
    FigureEight: the routine (make routine-figure-eight).
    ROUTE_LAP: one lap's maneuvers.

Flow (per trial):
    1. Rig and battery opened; the robot at the shared intersection's stop
       line, about to turn left; Enter drives.
    2. One run until the time is up (or it ends otherwise); motors stop.
    3. The tester's touches and laps seen; intersections.csv, laps.csv and
       touches.csv for the attempt; the trial's row.
"""
import csv
import os
import select
import statistics
import sys
import threading
from pathlib import Path

from src.intersection_linker import HEADING_TOLERANCE_DEG
from src.navigation.intersection import STAGE_EXIT, TURN_END_GYRO
from src.navigation.navigation import RULE_INTERSECTION, RULE_LANE_KEEPING
from src.navigation.route import LEFT, RIGHT
from src.routines.harness import Routine, criterion

ROUTE_LAP = (LEFT,) * 4 + (RIGHT,) * 4
MINUTES = 10.0
MAX_LAPS = 200              # the route's length: far more than any run's minutes allow
TIME_UP = "time up"
EXPECTED_DEG = {LEFT: -90.0, RIGHT: 90.0}               # + = turned right
INTERSECTION_FIELDS = ("step", "lap", "maneuver", "t_enter", "turn_end", "turn_deg", "exit_deg", "heading_deg",
                       "heading_off_deg", "lane_back", "battery_v")
LAP_FIELDS = ("lap", "t_start", "seconds", "battery_v")
TOUCH_FIELDS = ("t", "lap", "step", "maneuver", "stage", "rule")


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _step(label) -> int:
    """The step number from progress.label(): "9/1600 left" -> 9."""
    head = str(label or "0").split("/")[0].strip()
    return int(head) if head.isdigit() else 0


def recording(value) -> bool:
    """The record setting as a switch: 0 / off / no / false is off, anything else on."""
    return str(value).strip().lower() not in ("0", "0.0", "off", "no", "false")


class TouchKeys:
    """
    Enter presses on the terminal while the robot runs: each line typed is
    one touch. A thread waits on the stream with select, so stop() ends it
    without a read pending, and the questions after the run read the
    terminal on their own. It reads the file descriptor itself and counts
    newlines: a buffered readline() takes two quick presses in one read and
    then waits on a descriptor with nothing left in it.

    Inputs:
        stream: The terminal (sys.stdin); anything with fileno().

    pressed(): touches since the last call.
    """
    def __init__(self, stream=None) -> None:
        self._stream = sys.stdin if stream is None else stream
        self._n = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True, name="touch-keys")
        self._thread.start()

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._stream], [], [], 0.2)
            except (OSError, ValueError):            # no terminal to wait on
                return
            if ready:
                data = os.read(self._stream.fileno(), 1024)
                if not data:                         # end of input
                    return
                with self._lock:
                    self._n += data.count(b"\n")

    def pressed(self) -> int:
        with self._lock:
            n, self._n = self._n, 0
        return n

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class EightWatch:
    """
    Follows one run's nav.csv rows: ends it once `seconds` have passed and
    keeps every intersection and lap.

    intersections: one dict per intersection entered (INTERSECTION_FIELDS):
        turn_deg is the heading the turn ended on (nav.csv's turn_deg; the
        rule zeroes its heading there and the exit holds from 0), exit_deg
        what the exit drifted from it (heading_deg in the exit), heading_deg
        their sum: the turn as a whole, None while the turn hasn't ended.
        heading_off_deg is heading_deg against the maneuver's -90 / +90.
    lane_lost_s: time spent in lane keeping off vision.
    contract: frames braked because a command broke the contract.
    touches: one dict per touch logged (TOUCH_FIELDS): when, and where the
        robot was, from touches() (TouchKeys.pressed) on the next frame.
    """
    def __init__(self, seconds: float, touches=None):
        self.seconds = seconds
        self._pressed = touches
        self.touches: list[dict] = []
        self.intersections: list[dict] = []
        self.lane_lost_s = 0.0
        self.contract = 0
        self.frames = 0
        self.t = 0.0
        self._last_t = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t, self.t = t, t
        self.frames += 1
        step = _step(n.get("step"))
        if step > len(self.intersections):
            m = ROUTE_LAP[(step - 1) % len(ROUTE_LAP)]
            self.intersections.append({"step": step, "lap": (step - 1) // len(ROUTE_LAP) + 1, "maneuver": m,
                                       "t_enter": round(t, 2), "turn_end": None, "turn_deg": None,
                                       "exit_deg": None, "heading_deg": None, "heading_off_deg": None,
                                       "lane_back": False,
                                       "battery_v": _num(n.get("battery_v"))})
        cur = self.intersections[-1] if self.intersections else None
        if cur is not None and n.get("rule") == RULE_INTERSECTION:
            if n.get("turn_end") and cur["turn_end"] is None:
                cur["turn_end"] = n["turn_end"]
            turned = _num(n.get("turn_deg"))
            if turned is not None and cur["turn_deg"] is None:
                cur["turn_deg"] = round(turned, 1)
            heading = _num(n.get("heading_deg"))
            if cur["turn_deg"] is not None and n.get("stage") == STAGE_EXIT and heading is not None:
                cur["exit_deg"] = round(heading, 1)
            if cur["turn_deg"] is not None:
                whole = cur["turn_deg"] + (cur["exit_deg"] or 0.0)
                cur["heading_deg"] = round(whole, 1)
                cur["heading_off_deg"] = round(whole - EXPECTED_DEG[cur["maneuver"]], 1)
        if n.get("rule") == RULE_LANE_KEEPING:
            if n.get("lane_status") == "vision":
                if cur is not None:
                    cur["lane_back"] = True
            else:
                self.lane_lost_s += dt
        if n.get("reason") == "contract":
            self.contract += 1
        for _ in range(self._pressed() if self._pressed is not None else 0):
            self.touches.append({"t": round(t, 2), "lap": cur["lap"] if cur else 0, "step": cur["step"] if cur else 0,
                                 "maneuver": cur["maneuver"] if cur else "", "stage": n.get("stage") or "",
                                 "rule": n.get("rule") or ""})
        return TIME_UP if t >= self.seconds else None

    def finished(self) -> list[dict]:
        """The intersections whose turn finished: the lane came back after them."""
        return [i for i in self.intersections if i["lane_back"]]

    def laps(self) -> list[dict]:
        """Each completed lap: from its first intersection to the next lap's."""
        firsts = [i for i in self.intersections if (i["step"] - 1) % len(ROUTE_LAP) == 0]
        return [{"lap": a["lap"], "t_start": a["t_enter"], "seconds": round(b["t_enter"] - a["t_enter"], 2),
                 "battery_v": a["battery_v"]} for a, b in zip(firsts, firsts[1:])]


class FigureEight(Routine):
    name = "figure-eight"
    title = "Figure 8: four lefts, four rights, for minutes"
    question = "Does the robot keep driving the course correctly, left and right, lap after lap, for the whole time?"
    requirement = "course repeatability; demo-day endurance"
    trials = 1
    fields = ("minutes", "ended_by", "intersections", "laps_robot", "laps_seen", "touches", "touches_logged",
              "turns_off_gyro",
              "turns_off_heading", "left_mean_deg", "right_mean_deg", "lap_s_mean", "lane_lost_s", "contract",
              "fps", "battery_start_v", "battery_end_v")
    needs = ("pigpiod",)
    settings = {"minutes": f"how long the run lasts (default {MINUTES:.0f})",
                "record": "1 saves every frame (the default), 0 saves none: the recorder-off A/B for stalls.csv"}
    instructions = """\
The course: two blocks side by side; four lefts loop around one, four rights
around the other, a figure 8 through the intersection they share. Start the
robot at that intersection's stop line, in its lane, about to turn LEFT.
It runs until the time is up. Motors are ON: stay near the track. If you have
to touch it (nudge, catch, put it back), press Enter in this terminal right
after, once per touch: it's logged with the lap and corner. Ctrl-C stops it
at once.
Frames are recorded as in any run: about 30 MB a minute on the SD card."""

    def __init__(self, open_rig=None, navigation_run=None, open_battery=None, touch_keys=None):
        self._open_rig, self._run, self._open_battery = open_rig, navigation_run, open_battery
        self._touch_keys = touch_keys

    def setup(self, ctx):
        if self._open_rig is None:
            from src.linker_io import open_rig
            self._open_rig = open_rig
        if self._run is None:
            from src.navigation_linker import run
            self._run = run
        if self._touch_keys is None:
            self._touch_keys = TouchKeys
        if self._open_battery is None:
            from src.diagnostics import battery_run
            self._open_battery = lambda: battery_run.open_battery(say=ctx.console.say)
        ctx.options.setdefault("minutes", MINUTES)
        ctx.options.setdefault("record", 1)

    def trial(self, ctx, i):
        from src.config import MEASURED, MEASURED_ESTIMATION
        from src.navigation.navigation import Navigation
        from src.navigation.route import FINISH_EDGE, Route
        minutes = float(ctx.options["minutes"])
        source, sensors, motor, _ = self._open_rig(True, motors=True, button=False)
        battery = self._open_battery()
        try:
            ctx.console.wait(f"Robot at the shared stop line, about to turn left? Enter drives it for {minutes:g} min")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop, getattr(battery, "cleanup", None)):
                if close is not None:
                    close()
            raise
        keys = self._touch_keys()
        watch = EightWatch(minutes * 60.0, touches=keys.pressed)
        nav = Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps,
                         route=Route(ROUTE_LAP * MAX_LAPS, FINISH_EDGE))
        attempt = Path(ctx.out_dir) / f"attempt_{ctx.attempt + 1:02d}"
        try:
            report = self._run(source, sensors, motor, nav, MEASURED, MEASURED_ESTIMATION, str(attempt), None,
                               max_run_s=minutes * 60.0 + 30.0, motors_on=True, render=False, stop_when=watch,
                               battery=battery, record=recording(ctx.options["record"]), warmup=True)
        finally:
            keys.stop()
            if battery is not None:
                battery.cleanup()
        ctx.console.say(f"  ended by {report['ended_by']} after {watch.t / 60:.1f} min, "
                        f"{len(watch.intersections)} intersections")
        ctx.console.say(f"  touches logged with Enter: {len(watch.touches)}"
                        + "".join(f"\n    {x['t']:7.1f} s  lap {x['lap']}, step {x['step']} {x['maneuver']}"
                                  f" ({x['stage'] or x['rule']})" for x in watch.touches))
        touches = int(ctx.console.ask_number("Times you touched it (nudged, caught, put back)", lo=0, hi=1000))
        laps_seen = int(ctx.console.ask_number("Laps you saw it complete", lo=0, hi=1000))
        self._write(attempt, watch)
        return self._row(watch, report, touches, laps_seen)

    @staticmethod
    def _write(folder: Path, watch: EightWatch) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (("intersections.csv", INTERSECTION_FIELDS, watch.intersections),
                                   ("laps.csv", LAP_FIELDS, watch.laps()),
                                   ("touches.csv", TOUCH_FIELDS, watch.touches)):
            with open(folder / name, "w", newline="") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                w.writerows(rows)

    @staticmethod
    def _row(watch: EightWatch, report: dict, touches: int, laps_seen: int) -> dict:
        done = watch.finished()
        mean = lambda m: (round(statistics.fmean(x), 1) if (x := [i["heading_deg"] for i in done            # noqa: E731
                                                                  if i["maneuver"] == m and i["heading_deg"] is not None]) else None)
        laps = watch.laps()
        volts = [i["battery_v"] for i in watch.intersections if i["battery_v"] is not None]
        wall = report.get("run", {}).get("wall_s") or watch.t
        return {"minutes": round(watch.t / 60, 2), "ended_by": report["ended_by"],
                # a lap counts once its eighth turn finished (the lane came back): a run ended
                # mid-turn doesn't count a lap the tester didn't see complete
                "intersections": len(watch.intersections), "laps_robot": len(done) // len(ROUTE_LAP),
                "laps_seen": laps_seen, "touches": touches, "touches_logged": len(watch.touches),
                "turns_off_gyro": sum(1 for i in done if i["turn_end"] != TURN_END_GYRO),
                "turns_off_heading": sum(1 for i in done if i["heading_off_deg"] is None
                                         or abs(i["heading_off_deg"]) > HEADING_TOLERANCE_DEG),
                "left_mean_deg": mean(LEFT), "right_mean_deg": mean(RIGHT),
                "lap_s_mean": round(statistics.fmean(x["seconds"] for x in laps), 1) if laps else None,
                "lane_lost_s": round(watch.lane_lost_s, 2), "contract": watch.contract,
                "fps": round(watch.frames / wall, 1) if wall else None,
                "battery_start_v": volts[0] if volts else None, "battery_end_v": volts[-1] if volts else None}

    def judge(self, rows):
        out = []
        for k, r in enumerate(rows, 1):
            tag = f" (run {k})" if len(rows) > 1 else ""
            out += [criterion(f"touches{tag}", r["touches"], "<=", 0),
                    criterion(f"ran the whole time{tag}", 1 if r["ended_by"] == TIME_UP else 0, ">=", 1),
                    criterion(f"laps counted minus laps seen{tag}", r["laps_robot"] - r["laps_seen"],
                              "within", (0, 0)),
                    criterion(f"turns not ended on the gyro{tag}", r["turns_off_gyro"], "<=", 0),
                    criterion(f"turns off by over {HEADING_TOLERANCE_DEG:.0f} deg{tag}", r["turns_off_heading"], "<=", 0)]
        return out
