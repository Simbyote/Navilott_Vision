"""Figure 8: four lefts then four rights, lap after lap, for minutes on the course.

Purpose:
    Stop distance asks how accurately one maneuver lands; this asks whether
    the robot keeps working over many intersections, with left and right side
    by side and minutes for the pack and heat to build up.

Main package:
    FigureEight (make routine-figure-eight). EightWatch follows each run;
    TouchKeys logs an Enter press per touch.
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
MAX_LAPS = 200              # far more than a run's minutes allow
TIME_UP = "time up"
EXPECTED_DEG = {LEFT: -90.0, RIGHT: 90.0}               # + = turned right
INTERSECTION_FIELDS = ("step", "lap", "maneuver", "t_enter", "turn_end", "turn_deg", "exit_deg", "heading_deg",
                       "heading_off_deg", "lane_back", "slip_s", "battery_v")
LAP_FIELDS = ("lap", "t_start", "seconds", "battery_v")
TOUCH_FIELDS = ("t", "lap", "step", "maneuver", "stage", "rule")
SLIP_FIELDS = ("t0", "t1", "frames", "lap", "step", "maneuver", "stage", "gyro_dps", "wheels_dps")
# Gyro yaw against the wheels' fitted yaw. On 2026-10-20 a slipping turn read ~3 deg/s
# against ~27 for the other lefts, both wheels at speed
SLIP_DPS = 12.0
SLIP_FRAMES = 4             # ~0.2 s
MIN_FIT_FRAMES = 20         # turn frames needed to fit the wheels' yaw


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def _step(label) -> int:
    head = str(label or "0").split("/")[0].strip()
    return int(head) if head.isdigit() else 0


def recording(value) -> bool:
    """0 / off / no / false is off, anything else on."""
    return str(value).strip().lower() not in ("0", "0.0", "off", "no", "false")


def _fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least squares y = k x + c; None under MIN_FIT_FRAMES points or with no spread in x."""
    if len(points) < MIN_FIT_FRAMES:
        return None
    mx = statistics.fmean(x for x, _ in points)
    my = statistics.fmean(y for _, y in points)
    var = sum((x - mx) ** 2 for x, _ in points)
    if var == 0:
        return None
    k = sum((x - mx) * (y - my) for x, y in points) / var
    return k, my - k * mx


class TouchKeys:
    """
    Each Enter typed during the run is one touch; pressed() returns those since the last call.

    A select() thread lets stop() end without a read pending. It reads the fd and
    counts newlines, since a buffered readline() takes two quick presses in one
    read and then blocks.
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
    Follows one run's nav.csv rows, ends it after `seconds`, keeps every intersection and lap.

    Per intersection, turn_deg is where the turn ended, exit_deg the exit's drift
    from it, and heading_deg their sum (None until the turn ends); heading_off_deg
    compares it with -90 / +90. touches come from TouchKeys on the next frame.
    """
    def __init__(self, seconds: float, touches=None):
        self.seconds = seconds
        self._pressed = touches
        self.touches: list[dict] = []
        self.intersections: list[dict] = []
        self._motion: list[tuple] = []          # (t, intersection index, stage, wheel cps difference, gyro dps)
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
                                       "lane_back": False, "slip_s": 0.0,
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
        left, right, yaw = _num(n.get("left_cps")), _num(n.get("right_cps")), _num(n.get("yaw_rate"))
        if not _num(n.get("brake")) and None not in (left, right, yaw):
            self._motion.append((t, len(self.intersections) - 1, n.get("stage") or "", right - left, yaw))
        for _ in range(self._pressed() if self._pressed is not None else 0):
            self.touches.append({"t": round(t, 2), "lap": cur["lap"] if cur else 0, "step": cur["step"] if cur else 0,
                                 "maneuver": cur["maneuver"] if cur else "", "stage": n.get("stage") or "",
                                 "rule": n.get("rule") or ""})
        return TIME_UP if t >= self.seconds else None

    def slips(self) -> list[dict]:
        """
        SLIP_FRAMES+ driving frames where gyro and wheel yaw rates differ by over SLIP_DPS.

        Each adds its seconds to its intersection's slip_s. [] with too few turn frames
        to fit.
        """
        for i in self.intersections:
            i["slip_s"] = 0.0
        turns = [(d, y) for _, _, stage, d, y in self._motion if stage == "turn"]
        line = _fit(turns)
        if line is None:
            return []
        # Refit without the slips, which pull the line toward them
        line = _fit([(d, y) for d, y in turns if abs(y - (line[0] * d + line[1])) <= SLIP_DPS]) or line
        k, c = line
        spots, run = [], []
        for row in self._motion + [None]:                       # None closes the last run
            off = row is not None and abs(row[4] - (k * row[3] + c)) > SLIP_DPS
            if off and (not run or run[-1][1] == row[1]):
                run.append(row)
                continue
            if len(run) >= SLIP_FRAMES:
                spots.append(self._spot(run, k, c))
            run = [row] if off else []
        return spots

    def _spot(self, run: list[tuple], k: float, c: float) -> dict:
        i = self.intersections[run[0][1]] if run[0][1] >= 0 else None
        t0, t1 = run[0][0], run[-1][0]
        if i is not None:
            i["slip_s"] = round(i["slip_s"] + t1 - t0, 2)
        return {"t0": round(t0, 2), "t1": round(t1, 2), "frames": len(run),
                "lap": i["lap"] if i else 0, "step": i["step"] if i else 0, "maneuver": i["maneuver"] if i else "",
                "stage": run[0][2], "gyro_dps": round(statistics.median(r[4] for r in run), 1),
                "wheels_dps": round(statistics.median(k * r[3] + c for r in run), 1)}

    def finished(self) -> list[dict]:
        """Intersections whose turn finished (the lane came back)."""
        return [i for i in self.intersections if i["lane_back"]]

    def laps(self) -> list[dict]:
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
              "slip_spots", "turns_off_gyro",
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
            from src.peripherals import battery_run
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
        slips = watch.slips()
        ctx.console.say(f"  didn't turn with its wheels: {len(slips)} spots"
                        + "".join(f"\n    lap {x['lap']}, step {x['step']} {x['maneuver']} ({x['stage']}): "
                                  f"{x['t1'] - x['t0']:.1f} s, gyro {x['gyro_dps']:+.0f} deg/s, "
                                  f"wheels {x['wheels_dps']:+.0f}" for x in slips))
        self._write(attempt, watch, slips)
        row = self._row(watch, report, touches, laps_seen)
        row["slip_spots"] = len(slips)
        return row

    @staticmethod
    def _write(folder: Path, watch: EightWatch, slips: list[dict]) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        for name, fields, rows in (("intersections.csv", INTERSECTION_FIELDS, watch.intersections),
                                   ("laps.csv", LAP_FIELDS, watch.laps()),
                                   ("touches.csv", TOUCH_FIELDS, watch.touches),
                                   ("slips.csv", SLIP_FIELDS, slips)):
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
                # a lap counts once its eighth turn finished, as the tester saw it
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
