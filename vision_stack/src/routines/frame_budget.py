"""Frame budget: does the whole chain keep 20 FPS, answer within 50 ms, and fit in the Pi's CPU and memory?

Purpose:
    P1, P2 and R4, judged, the pipeline-timing and CPU tests the IDR
    promised (slide 7). Each trial is one run of the whole chain with the
    navigation in the loop and the motors off (navigation_linker.run, as
    power-profile's pipeline stage), `seconds` long, the robot parked on a
    lane so lane keeping has work to do. BudgetWatch follows every nav.csv
    row: Phases 2 + 3 against the frame period (the budget), the time from
    a frame's arrival to its motor command (latency_ms: P2 from appsink,
    the part the code controls), the frame rate. A Sampler thread reads the
    CPU's busy share, this process's memory (the pipeline runs in it) and
    the temperature every SAMPLE_S, and the throttle flags at the start and
    end; it reads /proc and the thermal zone only, so the timing it watches
    isn't disturbed (vcgencmd only at the ends).

    Pass (each run): at least TARGET_FPS (P1); at most OVER_BUDGET_PCT of
    frames over the budget ("near 0", a first guess); p95 latency at most
    LATENCY_MS (P2); CPU at most CPU_MAX_PCT on average and memory at most
    RSS_MAX_MB (R4); ran the whole time. Heat and throttling are reported;
    power-profile judges them.

Main package:
    BudgetWatch: navigation_linker.run's stop_when for one run.
    Sampler: CPU, memory and heat on a thread; throttling at the ends.
    FrameBudget: the routine (make routine-frame-budget).

Flow (per trial):
    rig opened (camera, sensors; no motors, no button) -> Enter -> the
    sampler starts -> one run until BudgetWatch says the time is up ->
    the sampler stops -> the row.
"""
import statistics
import threading

from src.params import FPS
from src.routines.harness import Routine, criterion

SECONDS = 60.0              # requirements.md P1: --limit 1200 at 20 FPS
SAMPLE_S = 0.5
TARGET_FPS = 20.0           # P1
BUDGET_MS = 1000.0 / FPS    # one frame period: what phase3_linker's "over budget" counts against
OVER_BUDGET_PCT = 5.0       # P1's "over budget near 0": a first guess
LATENCY_MS = 50.0           # P2
CPU_MAX_PCT = 70.0          # R4
RSS_MAX_MB = 400.0          # R4
TIME_UP = "time up"
THROTTLE_FLAGS = ("throttled", "freq_capped", "soft_temp_limit")


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def p95(values: list[float]) -> float | None:
    """The 95th percentile, nearest rank; None without values."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, -(-95 * len(s) // 100) - 1))]


class BudgetWatch:
    """Follows one run's nav.csv rows; ends it once `seconds` have passed. Keeps per-frame timings."""
    def __init__(self, seconds: float, budget_ms: float = BUDGET_MS):
        self.seconds, self.budget_ms = seconds, budget_ms
        self.work_ms: list[float] = []         # Phases 2 + 3
        self.latency_ms: list[float] = []      # arrival to motor command
        self.t_first = self.t = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        self.t_first = t if self.t_first is None else self.t_first
        self.t = t
        p2, p3 = _num(n.get("phase2_ms")), _num(n.get("phase3_ms"))
        if p2 is not None and p3 is not None:
            self.work_ms.append(p2 + p3)
        lat = _num(n.get("latency_ms"))
        if lat is not None:
            self.latency_ms.append(lat)
        return TIME_UP if t >= self.seconds else None

    @property
    def frames(self) -> int:
        return len(self.work_ms)

    def fps(self) -> float | None:
        span = (self.t or 0.0) - (self.t_first or 0.0)
        return round((self.frames - 1) / span, 2) if self.frames > 1 and span > 0 else None

    def over_budget_pct(self) -> float | None:
        if not self.work_ms:
            return None
        return round(100.0 * sum(1 for w in self.work_ms if w > self.budget_ms) / len(self.work_ms), 1)


class Sampler:
    """
    CPU busy %, this process's RSS and the temperature every SAMPLE_S on a
    thread; the throttle flags (since boot) at start() and stop(), so a
    flag that newly appeared during the run shows. Readers are injectable.
    """
    def __init__(self, read=None, throttled=None, period_s: float = SAMPLE_S):
        self._read, self._throttled, self.period = read, throttled, period_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._before = None

    def _read_pi(self) -> dict:
        from src.diagnostics.system_monitor import read_rss_mb, read_temp_c
        from src.routines.power_profile import cpu_busy, read_cpu_times
        cur = read_cpu_times()
        busy, self._cpu_prev = cpu_busy(self._cpu_prev, cur), cur
        return {"cpu_pct": busy, "rss_mb": read_rss_mb(), "temp_c": read_temp_c()}

    def _flags(self) -> dict:
        if self._throttled is None:
            from src.diagnostics.system_monitor import decode_throttled, read_throttled
            return decode_throttled(read_throttled())
        return self._throttled()

    def start(self) -> "Sampler":
        if self._read is None:
            from src.routines.power_profile import read_cpu_times
            self._cpu_prev = read_cpu_times()
            self._read = self._read_pi
        self._before = self._flags()

        def loop():
            while not self._stop.wait(self.period):
                self.samples.append(self._read())
        self._thread = threading.Thread(target=loop, name="frame-budget-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        """Stop sampling: {cpu_mean_pct, cpu_max_pct, rss_max_mb, temp_max_c, throttled, under_voltage}."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        after = self._flags()

        def new(flag):
            k = f"{flag}_occurred"                  # latched since boot: only a change counts
            if after.get(k) is None:
                return None
            return int(bool(after.get(flag)) or (bool(after[k]) and not (self._before or {}).get(k)))
        cpu = [s["cpu_pct"] for s in self.samples if s.get("cpu_pct") is not None]
        rss = [s["rss_mb"] for s in self.samples if s.get("rss_mb") is not None]
        temp = [s["temp_c"] for s in self.samples if s.get("temp_c") is not None]
        flags = [new(f) for f in THROTTLE_FLAGS]
        return {"cpu_mean_pct": round(statistics.fmean(cpu), 1) if cpu else None,
                "cpu_max_pct": max(cpu) if cpu else None,
                "rss_max_mb": round(max(rss), 1) if rss else None,
                "temp_max_c": max(temp) if temp else None,
                "throttled": None if all(f is None for f in flags) else int(any(flags)),
                "under_voltage": new("under_voltage")}


class FrameBudget(Routine):
    name = "frame-budget"
    title = "Frame budget: rate, latency, CPU and memory with navigation running"
    question = "Does the whole chain keep 20 FPS, answer within 50 ms, and fit in the Pi's CPU and memory?"
    requirement = "P1, P2, R4"
    trials = 3
    fields = ("seconds", "frames", "fps", "over_budget_pct", "work_p95_ms", "latency_p95_ms", "cpu_mean_pct",
              "cpu_max_pct", "rss_max_mb", "temp_max_c", "throttled", "under_voltage", "ended_by")
    settings = {"seconds": f"each run's length in s (default {SECONDS:.0f})"}
    instructions = f"""\
Set up: the robot parked in a straight lane, both lines in view, lit as on the
course. Motors stay OFF: the whole chain runs with navigation deciding, but
nothing drives. Each trial is one {SECONDS:.0f} s run; leave the robot alone and
keep other programs (a second ssh with top, a browser) off the Pi."""

    def __init__(self, open_rig=None, navigation_run=None, sampler=Sampler):
        self._open_rig, self._run, self._sampler = open_rig, navigation_run, sampler

    def setup(self, ctx):
        if self._open_rig is None:
            from src.linker_io import open_rig
            self._open_rig = open_rig
        if self._run is None:
            from src.navigation_linker import run
            self._run = run
        ctx.options.setdefault("seconds", SECONDS)

    def trial(self, ctx, i):
        from src.config import MEASURED, MEASURED_ESTIMATION
        from src.navigation.navigation import Navigation
        seconds = float(ctx.options["seconds"])
        source, sensors, motor, _ = self._open_rig(True, motors=False, button=False)
        try:
            ctx.console.wait(f"Robot parked in the lane, nothing else running? Enter runs {seconds:.0f} s (motors off)")
        except BaseException:
            for close in (source.close, sensors.stop, motor.stop):
                close()
            raise
        watch = BudgetWatch(seconds)
        sampler = self._sampler().start()
        try:
            report = self._run(source, sensors, motor, Navigation(gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps),
                               MEASURED, MEASURED_ESTIMATION, str(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}"),
                               None, max_run_s=seconds + 10.0, motors_on=False, render=False, stop_when=watch)
        finally:
            system = sampler.stop()
        work, lat = p95(watch.work_ms), p95(watch.latency_ms)
        return {"seconds": seconds, "frames": watch.frames, "fps": watch.fps(),
                "over_budget_pct": watch.over_budget_pct(),
                "work_p95_ms": None if work is None else round(work, 1),
                "latency_p95_ms": None if lat is None else round(lat, 1),
                **system, "ended_by": report["ended_by"]}

    def judge(self, rows):
        out = []
        for k, r in enumerate(rows, 1):
            tag = f" (run {k})" if len(rows) > 1 else ""
            out += [criterion(f"frames per second{tag}", r["fps"], ">=", TARGET_FPS),
                    criterion(f"frames over the {BUDGET_MS:g} ms budget{tag}", r["over_budget_pct"], "<=",
                              OVER_BUDGET_PCT, "%"),
                    criterion(f"p95 frame to motor command{tag}", r["latency_p95_ms"], "<=", LATENCY_MS, "ms"),
                    criterion(f"CPU busy, mean{tag}", r["cpu_mean_pct"], "<=", CPU_MAX_PCT, "%"),
                    criterion(f"memory, largest{tag}", r["rss_max_mb"], "<=", RSS_MAX_MB, "MB"),
                    criterion(f"ran the whole time{tag}", 1 if r["ended_by"] == TIME_UP else 0, ">=", 1)]
        return out
