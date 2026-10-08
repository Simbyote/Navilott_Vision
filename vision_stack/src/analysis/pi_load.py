"""Pi load: how a run used the Pi's cores, threads, clock and memory, from a diagnostics recording.

Purpose:
    src.diagnostics.monitor records a run from outside: every thread's CPU,
    core and context switches, every core's load, and temperature, clock,
    throttling and memory. Its summary.txt gives the averages; this reads
    the recording over time and interprets it: whether the frame loop is
    CPU-bound, whether the work runs in parallel or mostly in one thread
    (Python's GIL and a serial pipeline look alike from outside), whether
    the sensor thread keeps its 100 Hz, whether threads are preempted or
    hop between cores, and what heat, the clock and memory did (soak.py's
    analysis, on the same system.csv). Given the run's own folder
    (--run), it lines the run's frames up with the recording on the
    monotonic clock, judges the threads, cores and system over the run's
    own time only (a recording also holds the imports, the camera opening
    and the countdown before it, and the shutdown after, which would
    otherwise pass for the run's load), and says what the slow frames
    coincided with: a clock drop, a busy frame loop, or neither. The rest
    of the Pi (os_counters: interrupts, other processes, the SD card,
    pressure stalls, the camera's memory pool and the ISP clock) is
    summarized over the same window.

Main package:
    analyze(...) -> dict: threads, process, cores, system, window, aligned, findings.
    report_lines(res): the printed report.
    python_native(rows), thread_lanes(rows): the two views of how the
        threads share the Pi, as data the figure draws.
    pi_load.json, pi_load.png (CPU per thread; Python vs native CPU against
    the one core the GIL allows Python; one lane per thread colored by the
    core it ran on; core load, temperature and clock, memory, and the run's
    frame intervals when aligned).

Input:
    A runs/diag_<time> folder (threads.csv, cores.csv, system.csv,
    irqs.csv, procs.csv, meta.json; recordings from before the OS
    counters have no os part). --run: the navigation_linker folder recorded, for lining
    up (its report.json keeps t0_monotonic).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.analysis import common, soak
from src.analysis.common import Table, stats
from src.diagnostics import os_counters as osc
from src.diagnostics.monitor import thread_summary
from src.diagnostics.system_monitor import THROTTLE_BITS
from src.diagnostics.threads import MAIN_THREAD
from src.params import FPS, SENSOR_RATE_HZ

CSV_NAMES = ("threads.csv",)
SENSOR_THREAD = "sensor-hub"

# Finding thresholds. Starting values: set them to what normal looks like after a few Pi runs
MAIN_BOUND_PCT = 90.0       # main's p95 CPU at or above this: the frame loop is CPU-bound
SERIAL_SHARE = 0.7          # main's share of the process's CPU above this: the work is mostly serial
SENSOR_SLIP = 0.8           # sensor-hub waking below this x SENSOR_RATE_HZ: its ticks slip
INVOL_HIGH_S = 50.0         # involuntary switches per second on main: its core is contended
MOVES_PER_MIN = 30.0        # main changing core this often a minute
CORE_SPREAD_PCT = 40.0      # mean busy % between the busiest and idlest core
TEMP_WARN_C = 75.0          # the Pi's firmware throttles at 80 C
CLOCK_DROP = 0.9            # a clock below this x its maximum is a drop
OTHER_PROC_PCT = 15.0       # another process using this % of a core on average: work the robot pays for elsewhere
DISK_BUSY_PCT = 50.0        # the SD card busy this % of a second: a recording's writes can hold frames up
PSI_STALL_PCT = 10.0        # some task stalled on CPU, I/O or memory this % of a 10 s window
LATE = 1.5                  # a frame interval over this x the frame budget is a slow frame
MIN_WINDOW_SAMPLES = 4      # a run shorter than this many thread samples is judged over the whole recording
MIN_LATE_FRAMES = 5         # fewer slow frames than this aren't worth attributing
BUDGET_MS = 1000.0 / FPS

PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f")

# Threads that run Python, so take turns on the GIL: the ones our code starts
# (each names itself, diagnostics.md) and pigpio's callback thread, a Python
# threading.Thread. Everything else is native code that never needs the GIL:
# GStreamer (task0), libcamera, and the unnamed "python3" threads, which on
# this Pi are OpenCV's TBB workers (our own threads are all named)
PYTHON_THREADS = frozenset({MAIN_THREAD, SENSOR_THREAD, "frame-recorder", "motor-watchdog", "system-monitor",
                            "pigpio-cb"})
# Categorical slots 1-2 and 1-4 of the validated default palette (dataviz
# skill): cores 0-3 pass the colorblind checks; two are under 3:1 contrast,
# so the lanes carry a legend
SPLIT_COLORS = {"python": "#2a78d6", "native": "#eb6834"}
CORE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
LANE_IDLE_PCT = 0.5         # a thread under this % of a core in every sample gets no lane
LANE_MIN_ALPHA = 0.2        # a lane cell's opacity at 0% CPU; 100% of a core is fully opaque

_CLI_HELP = """\
How a run used the Pi, from a diagnostics recording: CPU per thread and core,
serial or parallel work, the sensor thread's cadence, preemption, heat, clock,
throttling and memory; with --run, what the run's slow frames coincided with.
Writes pi_load.png and pi_load.json next to the recording.

Run from vision_stack/, venv active:
    python3 -m src.analysis.pi_load                                  newest recording
    python3 -m src.analysis.pi_load runs/diag_20261003_101500
    python3 -m src.analysis.pi_load runs/diag_20261003_101500 --run runs/nav_20261003_101502
"""


# =============================================================================
# Reading
# =============================================================================

def thread_rows(threads: Table) -> list[dict]:
    """threads.csv as the dict rows diagnostics.monitor.thread_summary() takes."""
    cols = ("elapsed_s", "tid", "core", "cpu_pct", "vol_ctx_s", "invol_ctx_s")
    num = {c: threads.numeric(c) for c in cols}
    names = threads.text("name")
    return [{"elapsed_s": float(num["elapsed_s"][i]), "tid": int(num["tid"][i]), "name": names[i],
             "core": int(num["core"][i]), "cpu_pct": float(num["cpu_pct"][i]),
             "vol_ctx_s": float(num["vol_ctx_s"][i]), "invol_ctx_s": float(num["invol_ctx_s"][i])}
            for i in range(len(threads))]


def by_sample(rows: list[dict]) -> tuple[np.ndarray, dict]:
    """(sample times, {name: cpu per sample}); threads sharing a name are summed."""
    times = sorted({r["elapsed_s"] for r in rows})
    index = {t: i for i, t in enumerate(times)}
    cpu = defaultdict(lambda: np.zeros(len(times)))
    for r in rows:
        cpu[r["name"]][index[r["elapsed_s"]]] += r["cpu_pct"]
    return np.array(times), dict(cpu)


# =============================================================================
# Analysis
# =============================================================================

def process_load(rows: list[dict]) -> dict:
    """The process's total CPU per sample, main's p95, and how much of the total the busiest thread did."""
    times, cpu = by_sample(rows)
    total = sum(cpu.values()) if cpu else np.zeros(0)
    main = cpu.get(MAIN_THREAD, np.zeros(len(times)))
    busiest = max(cpu, key=lambda n: cpu[n].sum()) if cpu else None
    whole = float(total.sum()) if total.size else 0.0
    tot = stats(total)
    return {"total_cpu_pct": {k: tot[k] for k in ("mean", "p95", "max")},
            "main_cpu_p95": stats(main)["p95"] if main.size else None,
            "main_share": round(float(main.sum()) / whole, 3) if whole else None,
            "busiest": busiest, "busiest_share": round(float(cpu[busiest].sum()) / whole, 3) if whole else None,
            "samples": int(len(times))}


def core_load(cores: Table | None) -> dict:
    if cores is None or len(cores) == 0:
        return {}
    out = {}
    core, busy = cores.numeric("core"), cores.numeric("busy_pct")
    for c in sorted({int(x) for x in core[~np.isnan(core)]}):
        s = stats(busy[core == c])
        out[str(c)] = {"mean": round(s["mean"], 1), "max": round(s["max"], 1)}
    return out


def since_boot(system: Table | None) -> list[str]:
    """Throttle flags latched at any time since boot (bits 16-19), as recorded."""
    if system is None:
        return []
    return [k for k in THROTTLE_BITS if system.has_values(f"{k}_occurred")
            and np.nanmax(system.numeric(f"{k}_occurred")) == 1]


def align(rows: list[dict], system: Table | None, diag_t0: float, nav: Table, nav_t0: float,
          budget_ms: float = BUDGET_MS) -> dict:
    """
    The run's frames against the recording, on the monotonic clock.

    Each frame (nav.csv t, from nav_t0) falls in the recording's sample
    covering it (from diag_t0); slow frames (interval over LATE budgets) are
    compared with normal ones: main's CPU in their sample, and whether the
    clock was below its maximum then.
    """
    times, cpu = by_sample(rows)
    main = cpu.get(MAIN_THREAD, np.zeros(len(times)))
    t = nav.numeric("t")
    frame_mono = nav_t0 + t
    interval = np.concatenate(([np.nan], np.diff(t) * 1000.0))
    idx = np.searchsorted(diag_t0 + times, frame_mono)          # the sample whose interval holds the frame
    inside = (idx < len(times)) & (frame_mono >= diag_t0) & ~np.isnan(interval)
    if not inside.any():
        return {"frames": 0}
    main_at = main[np.minimum(idx, len(times) - 1)]
    low = np.zeros(len(t), bool)
    if system is not None and system.has_values("cpu_mhz"):
        st, mhz = system.numeric("elapsed_s"), system.numeric("cpu_mhz")
        top = np.nanmax(mhz)
        j = np.clip(np.searchsorted(diag_t0 + st, frame_mono), 0, len(st) - 1)
        low = mhz[j] < CLOCK_DROP * top
    late = inside & (interval > LATE * budget_ms)
    normal = inside & ~late

    def m(x, sel):
        return round(float(np.mean(x[sel])), 3) if sel.any() else None
    corr = None
    if inside.sum() > 2 and np.std(main_at[inside]) > 0 and np.std(interval[inside]) > 0:
        corr = round(float(np.corrcoef(main_at[inside], interval[inside])[0, 1]), 2)
    return {"frames": int(inside.sum()), "late": int(late.sum()),
            "late_main_cpu": m(main_at, late), "normal_main_cpu": m(main_at, normal),
            "late_low_clock_share": m(low.astype(float), late), "normal_low_clock_share": m(low.astype(float), normal),
            "corr_main_cpu_interval": corr}


def run_window(rows: list[dict], diag_t0: float, nav: Table, nav_t0: float) -> dict:
    """
    The run's own span on the recording's clock: from its t0 to its last
    frame. A thread sample covers the interval before its time, so it
    belongs to the run when the middle of that interval does.
    """
    t = nav.numeric("t")
    start = nav_t0 - diag_t0
    end = start + (float(np.nanmax(t)) if np.any(~np.isnan(t)) else 0.0)
    times = np.array(sorted({r["elapsed_s"] for r in rows}))
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    mid = times - step / 2
    inside = times[(mid >= start) & (mid <= end)]
    return {"start_s": round(start, 3), "end_s": round(end, 3), "samples": int(len(inside)),
            "recording_s": round(float(times[-1]), 3) if len(times) else 0.0,
            "step_s": step, "used": len(inside) >= MIN_WINDOW_SAMPLES}


def _in_window(elapsed: np.ndarray, w: dict, step: float = 0.0) -> np.ndarray:
    mid = elapsed - step / 2
    return (mid >= w["start_s"]) & (mid <= w["end_s"])


def table_rows(t: Table | None) -> list[dict]:
    """A Table as one dict per row, cells as text ("" when blank)."""
    if t is None:
        return []
    cols = {c: t.text(c) for c in t.columns}
    return [{c: cols[c][i] for c in t.columns} for i in range(len(t))]


def os_load(system: Table | None, irqs: Table | None, procs: Table | None) -> dict | None:
    """The rest of the Pi over these rows (os_counters' summaries); None for a recording without the OS counters."""
    srows = table_rows(system)
    n = osc.intervals(srows)
    if not n:
        return None
    return {"intervals": n, **osc.system_extremes(srows),
            "irqs": osc.irq_summary(table_rows(irqs), n)[:osc.TOP_IRQS],
            "procs": osc.proc_summary(table_rows(procs), n)[:osc.TOP_PROCS]}


def analyze(threads: Table, cores: Table | None = None, system: Table | None = None,
            nav: Table | None = None, diag_t0: float | None = None, nav_t0: float | None = None,
            irqs: Table | None = None, procs: Table | None = None) -> dict:
    missing = [c for c in ("tid", "name", "core", "cpu_pct") if not threads.has(c)]
    if missing:
        raise ValueError(f"{threads.path.name}: not a diagnostics threads.csv (no {', '.join(missing)})")
    rows = thread_rows(threads)
    alignable = nav is not None and diag_t0 is not None and nav_t0 is not None
    window = run_window(rows, diag_t0, nav, nav_t0) if alignable else None
    judged_rows, judged_cores, judged_system, judged_irqs, judged_procs = rows, cores, system, irqs, procs
    if window and window["used"]:
        keep = _in_window(np.array([r["elapsed_s"] for r in rows]), window, window["step_s"])
        judged_rows = [r for r, k in zip(rows, keep) if k]
        if cores is not None and len(cores):
            judged_cores = cores.subset(_in_window(cores.numeric("elapsed_s"), window, window["step_s"]))
        if system is not None and len(system):
            # once a second, each a reading at that moment: the rows inside the run, timed from its start
            sub = system.subset(_in_window(system.numeric("elapsed_s"), window),
                                shift={"elapsed_s": window["start_s"]})
            judged_system = sub if len(sub) >= 2 else system
            if judged_system is not system:      # the OS rows are once a second too, each for the second before it
                judged_irqs, judged_procs = (t.subset(_in_window(t.numeric("elapsed_s"), window)) if t is not None
                                             and len(t) else t for t in (irqs, procs))
    res = {"threads": thread_summary(judged_rows), "process": process_load(judged_rows),
           "cores": core_load(judged_cores),
           "system": soak.analyze(judged_system) if judged_system is not None and len(judged_system) >= 2 else None,
           "since_boot": since_boot(system), "window": window, "aligned": None,
           "os": os_load(judged_system, judged_irqs, judged_procs)}
    if alignable:
        res["aligned"] = align(rows, system, diag_t0, nav, nav_t0)
    res["findings"] = findings(res)
    # Caveats, not problems: soak's "too short to judge memory" on any run under ~6 minutes
    res["notes"] = [f for f in (res["system"] or {}).get("findings", []) if f.startswith("run too short")]
    if window and not window["used"]:
        res["notes"].append(f"the run covers {window['samples']} thread samples (under {MIN_WINDOW_SAMPLES}): "
                            "judged over the whole recording, startup and shutdown included")
    return res


def _named(res: dict, name: str) -> list[dict]:
    return [t for t in res["threads"] if t["name"] == name]


def findings(res: dict) -> list[str]:
    out = []
    main = _named(res, MAIN_THREAD)
    duration_min = None
    if res["system"]:
        duration_min = res["system"]["duration_s"] / 60.0
    if main:
        m = main[0]
        p95 = res["process"]["main_cpu_p95"]
        if p95 is not None and p95 >= MAIN_BOUND_PCT:
            out.append(f"the frame loop (main) runs at {p95:.0f}% of a core at p95: CPU-bound, so frames stretch "
                       "whenever a frame needs more")
        if m["invol_ctx_s"] > INVOL_HIGH_S:
            out.append(f"main is preempted {m['invol_ctx_s']:.0f} times/s: its core is contended by other "
                       "threads or processes")
        if duration_min and m["moves"] / duration_min > MOVES_PER_MIN:
            out.append(f"main changes core about {m['moves'] / duration_min:.0f} times a minute (at least): "
                       "pinning it (taskset -c) may steady frame times")
    share, busiest = res["process"]["busiest_share"], res["process"]["busiest"]
    if share is not None and share > SERIAL_SHARE:
        out.append(f"{100 * share:.0f}% of the process's CPU is one thread ({busiest}): the work is mostly serial "
                   "(one thread, or Python threads taking turns on the GIL), so the other cores mostly wait")
    hub = _named(res, SENSOR_THREAD)
    if hub and hub[0]["vol_ctx_s"] < SENSOR_SLIP * SENSOR_RATE_HZ:
        out.append(f"sensor-hub woke {hub[0]['vol_ctx_s']:.0f} times/s, not ~{SENSOR_RATE_HZ:.0f}: its ticks slip "
                   "(waiting for the GIL behind main, or a slow IMU read)")
    if res["cores"]:
        means = [c["mean"] for c in res["cores"].values()]
        if max(means) - min(means) > CORE_SPREAD_PCT:
            out.append(f"the cores are unevenly loaded ({min(means):.0f}-{max(means):.0f}% mean busy)")
    sysres = res["system"]
    if sysres:
        if sysres["temp_c"] and sysres["temp_c"]["max"] >= TEMP_WARN_C:
            out.append(f"temperature reached {sysres['temp_c']['max']:.1f} C: close to the 80 C where the Pi throttles")
        mhz = sysres["cpu_mhz"]
        if mhz and mhz["min"] < CLOCK_DROP * mhz["max"]:
            out.append(f"the clock dropped to {mhz['min']:.0f} MHz (max {mhz['max']:.0f}) for "
                       f"{100 * mhz['share_below_max']:.0f}% of the run")
        out += [f for f in sysres["findings"] if not f.startswith(("no throttling", "run too short"))]
    o = res.get("os")
    if o:
        for p in o["procs"]:
            if p["name"] != osc.SELF_NAME and p["mean_pct"] >= OTHER_PROC_PCT:
                out.append(f"{p['name']} (pid {p['pid']}) used {p['mean_pct']:.0f}% of a core on average: "
                           "work the robot pays for outside its own process")
        if o["disk_busy_max_pct"] is not None and o["disk_busy_max_pct"] >= DISK_BUSY_PCT:
            out.append(f"the SD card was busy up to {o['disk_busy_max_pct']:.0f}% of a second (writes up to "
                       f"{common.fmt(o['disk_write_max_kbps'], 0)} kB/s): a recording's writes can hold frames up")
        for kind in osc.PSI_KINDS:
            v = o[f"psi_{kind}_max"]
            if v is not None and v >= PSI_STALL_PCT:
                out.append(f"some task stalled waiting on {kind} up to {v:.0f}% of the time (pressure, 10 s average)")
    latched = [k for k in res["since_boot"] if not (sysres and k in sysres["throttle"])]
    if latched:
        out.append(f"latched since boot, not during this run: {', '.join(latched)} "
                   "(under_voltage means the supply sagged at some point: check it under motor load)")
    a = res["aligned"]
    if a and a.get("late", 0) >= MIN_LATE_FRAMES:
        clock_gap = (a["late_low_clock_share"] or 0) - (a["normal_low_clock_share"] or 0)
        cpu_gap = (a["late_main_cpu"] or 0) - (a["normal_main_cpu"] or 0)
        if clock_gap > 0.2:
            cause = f"clock drops ({100 * a['late_low_clock_share']:.0f}% of slow frames were at a reduced clock)"
        elif cpu_gap > 15:
            cause = f"a busy frame loop (main at {a['late_main_cpu']:.0f}% vs {a['normal_main_cpu']:.0f}% otherwise)"
        else:
            cause = "neither a clock drop nor a busy main thread: look at the camera, I/O or other processes"
        out.append(f"{a['late']} slow frames coincide with {cause}")
    return out


def _dash(v, digits: int) -> str:
    """common.fmt, "--" for a value never read."""
    return common.fmt(v, digits) or "--"


def report_lines(res: dict) -> list[str]:
    p = res["process"]
    w = res.get("window")
    lines = []
    if w and w["used"]:
        lines.append(f"window    the run's own {w['end_s'] - w['start_s']:.1f} s ({w['start_s']:.1f}-{w['end_s']:.1f} s "
                     f"of the {w['recording_s']:.1f} s recording, {w['samples']} samples): "
                     "startup and shutdown left out")
    lines += [f"process   CPU mean {common.fmt(p['total_cpu_pct']['mean'], 1)}% of one core, "
             f"p95 {common.fmt(p['total_cpu_pct']['p95'], 1)}%, max {common.fmt(p['total_cpu_pct']['max'], 1)}%"
             + (f"; main {100 * p['main_share']:.0f}% of it" if p["main_share"] is not None else ""),
             "threads   " + "; ".join(f"{t['name']} {t['cpu_mean']:.0f}% (max {t['cpu_max']:.0f})"
                                     for t in res["threads"][:6])]
    if res["cores"]:
        lines.append("cores     " + ", ".join(f"{c}: {v['mean']:.0f}% (max {v['max']:.0f})" for c, v in res["cores"].items()))
    s = res["system"]
    if s:
        temp = s["temp_c"]
        lines.append(f"system    {s['duration_s']:.0f} s; temperature "
                     + (f"{temp['start']:.1f} -> {temp['end']:.1f} C (max {temp['max']:.1f})" if temp else "--")
                     + "; clock " + (f"{s['cpu_mhz']['min']:.0f}-{s['cpu_mhz']['max']:.0f} MHz" if s["cpu_mhz"] else "--")
                     + "; memory " + (f"{s['memory']['rss_start_mb']:.0f} -> {s['memory']['rss_end_mb']:.0f} MB"
                                      if s["memory"] else "--"))
    o = res.get("os")
    if o:
        lines.append(f"os        CMA free min {_dash(o['cma_free_min_mb'], 0)} MB; ISP clock "
                     + (f"{o['isp_mhz'][0]:.0f}-{o['isp_mhz'][1]:.0f} MHz" if o["isp_mhz"] else "--")
                     + f"; SD write mean {_dash(o['disk_write_mean_kbps'], 0)} kB/s, busy max "
                     f"{_dash(o['disk_busy_max_pct'], 0)}%; stalls max "
                     + ", ".join(f"{k} {_dash(o[f'psi_{k}_max'], 1)}%" for k in osc.PSI_KINDS))
        lines.append("irqs      " + ("; ".join(f"{i['name']} {i['mean_hz']:.0f}/s" for i in o["irqs"][:5]) or "--"))
        lines.append("others    " + ("; ".join(f"{p['name']} {p['mean_pct']:.0f}%" for p in o["procs"][:5]) or "--"))
    a = res["aligned"]
    if a is not None:
        lines.append(f"aligned   {a['frames']} frames, {a.get('late', 0)} slow; main CPU slow {common.fmt(a.get('late_main_cpu'), 0)}%"
                     f" vs normal {common.fmt(a.get('normal_main_cpu'), 0)}%; correlation main CPU / interval "
                     f"{common.fmt(a.get('corr_main_cpu_interval'))}")
    lines += ["", "findings"] + ([f"  - {f}" for f in res["findings"]] or ["  none: nothing past the thresholds"])
    lines += [f"  (note: {n})" for n in res.get("notes", [])]
    return lines


# =============================================================================
# How the threads share the Pi
# =============================================================================

def is_python_thread(name: str) -> bool:
    return name in PYTHON_THREADS


def python_native(rows: list[dict]) -> dict:
    """
    CPU per sample split into threads that run Python and native ones.

    Python threads can only add up to more than one core while some of them
    are inside C code that lets go of the GIL (OpenCV, I/O): the rest of
    their time they take turns.

    Outputs:
        {"t": sample times, "python": % of a core, "native": % of a core,
         "python_threads": names, "native_threads": names}
    """
    times, cpu = by_sample(rows)
    zero = np.zeros(len(times))
    py = [n for n in sorted(cpu) if is_python_thread(n)]
    nat = [n for n in sorted(cpu) if not is_python_thread(n)]
    return {"t": times, "python": sum((cpu[n] for n in py), zero), "native": sum((cpu[n] for n in nat), zero),
            "python_threads": py, "native_threads": nat}


def thread_lanes(rows: list[dict], idle_pct: float = LANE_IDLE_PCT) -> dict:
    """
    One lane per thread, busiest first: at each sample, the core it was last
    on and its CPU. Threads under idle_pct in every sample are counted, not
    drawn. Two threads sharing a name are told apart by their tid.

    Outputs:
        {"step": sample interval, "lanes": [{"label", "python", "t", "core", "cpu"}], "idle": count}
    """
    times = sorted({r["elapsed_s"] for r in rows})
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["tid"]].append(r)
    names = defaultdict(int)
    for rs in by_tid.values():
        names[rs[0]["name"]] += 1
    lanes, idle = [], 0
    for tid, rs in by_tid.items():
        rs.sort(key=lambda r: r["elapsed_s"])
        cpu = np.array([r["cpu_pct"] for r in rs])
        if cpu.max() < idle_pct:
            idle += 1
            continue
        name = rs[0]["name"]
        lanes.append({"label": f"{name} {tid}" if names[name] > 1 else name, "python": is_python_thread(name),
                      "t": np.array([r["elapsed_s"] for r in rs]), "core": np.array([r["core"] for r in rs]),
                      "cpu": cpu, "_total": float(cpu.sum())})
    lanes.sort(key=lambda lane: -lane["_total"])
    for lane in lanes:
        del lane["_total"]
    return {"step": step, "lanes": lanes, "idle": idle}


def _draw_split(ax, split: dict, cores_n: int) -> None:
    t = split["t"]
    ax.stackplot(t, split["python"], split["native"], colors=(SPLIT_COLORS["python"], SPLIT_COLORS["native"]),
                 labels=("Python threads (take turns on the GIL)", "native threads (no GIL)"),
                 edgecolor="white", linewidth=0.5)
    ax.axhline(100.0, color="#333333", lw=0.8, ls="--",
               label="one core: all the Python threads together,\nexcept while one is in C code that releases the GIL")
    ax.set_ylim(0, max(100.0 * cores_n, 1.0) if cores_n else None)
    ax.set_ylabel("CPU % of a core", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))


def _draw_lanes(ax, lanes: dict) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    step = lanes["step"]
    for row, lane in enumerate(lanes["lanes"]):
        for core in sorted(set(lane["core"].tolist())):
            sel = lane["core"] == core
            color = CORE_COLORS[int(core) % len(CORE_COLORS)]
            alphas = LANE_MIN_ALPHA + (1 - LANE_MIN_ALPHA) * np.clip(lane["cpu"][sel] / 100.0, 0, 1)
            ax.broken_barh([(t - step, step) for t in lane["t"][sel]], (row - 0.4, 0.8),
                           facecolors=[to_rgba(color, a) for a in alphas], edgecolor="none")
    ax.set_yticks(range(len(lanes["lanes"])))
    ax.set_yticklabels([lane["label"] for lane in lanes["lanes"]], fontsize=7)
    for label, lane in zip(ax.get_yticklabels(), lanes["lanes"]):
        label.set_color(SPLIT_COLORS["python"] if lane["python"] else SPLIT_COLORS["native"])
    ax.set_ylim(len(lanes["lanes"]) - 0.5, -0.5)
    used = sorted({int(c) for lane in lanes["lanes"] for c in lane["core"]})
    handles = [Patch(color=CORE_COLORS[c % len(CORE_COLORS)], label=f"core {c}") for c in used]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="the core it ran on;\nfainter = less CPU", title_fontsize=7)
    ax.set_ylabel("thread (blue: Python,\norange: native)", fontsize=8)
    if lanes["idle"]:
        ax.text(1.0, -0.02, f"+{lanes['idle']} idle threads not shown", transform=ax.transAxes, fontsize=7,
                ha="right", va="top", color="#777777")


# =============================================================================
# Figure and command line
# =============================================================================

def figure(threads: Table, cores: Table | None, system: Table | None, res: dict, title: str, out_path,
           nav: Table | None = None, offset_s: float | None = None) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    rows = thread_rows(threads)
    times, cpu = by_sample(rows)
    lanes = thread_lanes(rows)
    cores_n = len({r["core"] for r in rows}) if cores is None or not len(cores) else \
        len({int(c) for c in cores.numeric("core") if not np.isnan(c)})
    kinds = ["cpu", "split", "lanes", "cores"] + (["temp", "memory"] if system is not None else []) + \
        (["frames"] if nav is not None and offset_s is not None else [])
    heights = {"lanes": max(2.0, 0.22 * len(lanes["lanes"]) + 0.6)}
    fig, axes = plt.subplots(len(kinds), 1, figsize=(13, sum(heights.get(k, 2.2) for k in kinds)), sharex=True,
                             gridspec_kw={"height_ratios": [heights.get(k, 2.2) for k in kinds]})
    ax_of = dict(zip(kinds, axes))
    ax = ax_of["cpu"]
    top = sorted(cpu, key=lambda n: -cpu[n].mean())
    for i, name in enumerate(top[:6]):
        ax.plot(times, cpu[name], lw=0.9, color=PALETTE[i % len(PALETTE)], label=name)
    if len(top) > 6:
        ax.plot(times, sum(cpu[n] for n in top[6:]), lw=0.8, color="#bbbbbb", label="others")
    ax.set_ylabel("CPU % of a core", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _draw_split(ax_of["split"], python_native(rows), cores_n)
    _draw_lanes(ax_of["lanes"], lanes)
    ax = ax_of["cores"]
    if cores is not None and len(cores):
        c, b, t = cores.numeric("core"), cores.numeric("busy_pct"), cores.numeric("elapsed_s")
        ids = sorted({int(x) for x in c[~np.isnan(c)]})
        ct = sorted(set(t[~np.isnan(t)]))
        grid = np.full((len(ids), len(ct)), np.nan)
        for ci, ti, bi in zip(c, t, b):
            if not np.isnan(ci):
                grid[ids.index(int(ci)), ct.index(ti)] = bi
        # each sample covers the interval before it, as in the lanes; cell edges on the shared time axis
        step = float(np.median(np.diff(ct))) if len(ct) > 1 else 0.5
        edges = np.concatenate(([ct[0] - step], ct))
        im = ax.pcolormesh(edges, np.arange(len(ids) + 1) - 0.5, grid, cmap="viridis", vmin=0, vmax=100)
        ax.set_ylim(len(ids) - 0.5, -0.5)
        # the colorbar outside the plot, like the other panels' legends, so every panel keeps the same width
        cax = ax.inset_axes((1.01, 0.0, 0.012, 1.0))
        fig.colorbar(im, cax=cax, label="busy %")
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([f"core {i}" for i in ids], fontsize=7)
    if system is not None:
        ax = ax_of["temp"]
        st = system.numeric("elapsed_s")
        if system.has_values("temp_c") or system.has_values("cpu_mhz"):
            ax.plot(st, system.numeric("temp_c"), color="#d62728", lw=0.9)
            ax.set_ylabel("temp C", fontsize=8, color="#d62728")
            ax2 = ax.twinx()
            ax2.plot(st, system.numeric("cpu_mhz"), color="#1f77b4", lw=0.9)
            ax2.set_ylabel("clock MHz", fontsize=8, color="#1f77b4")
        else:
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no temperature or clock readings here (not a Pi)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#777777")
        ax = ax_of["memory"]
        ax.plot(st, system.numeric("rss_mb"), lw=0.9, color="#1f77b4")
        ax.set_ylabel("run memory MB", fontsize=8, color="#1f77b4")
        ax2 = ax.twinx()                    # the Pi's free memory is far larger: its own axis
        ax2.plot(st, system.numeric("mem_available_mb"), lw=0.9, color="#ff7f0e")
        ax2.set_ylabel("Pi available MB", fontsize=8, color="#ff7f0e")
    if "frames" in ax_of:
        ax = ax_of["frames"]
        t = nav.numeric("t")
        ax.plot(offset_s + t[1:], np.diff(t) * 1000.0, lw=0.6)
        ax.axhline(LATE * BUDGET_MS, color="#d62728", lw=0.6, ls="--")
        ax.set_ylabel("frame interval ms", fontsize=8)
    w = res.get("window")
    if w and w["used"]:
        for a in axes:
            a.axvspan(w["start_s"], w["end_s"], color="#2ca02c", alpha=0.07, lw=0)
    axes[-1].set_xlabel("time since the recording started, s" + (" (shaded: the run judged)" if w and w["used"] else ""))
    axes[0].set_title(title, fontsize=11, loc="left")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load(arg, run=None) -> dict:
    """The recording's tables and meta, and the run's nav.csv and t0 when given."""
    threads_path = common.find_csv(arg, CSV_NAMES)
    folder = threads_path.parent
    opt = lambda name: Table(folder / name) if (folder / name).is_file() and \
        len((folder / name).read_text().splitlines()) > 1 else None      # noqa: E731
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    out = {"folder": folder, "threads": Table(threads_path), "cores": opt("cores.csv"), "system": opt("system.csv"),
           "irqs": opt("irqs.csv"), "procs": opt("procs.csv"), "meta": meta, "nav": None, "nav_t0": None}
    if run:
        nav_path = common.find_csv(run, ("nav.csv",))
        rep = nav_path.parent / "report.json"
        out["nav"] = Table(nav_path)
        out["nav_t0"] = json.loads(rep.read_text()).get("run", {}).get("t0_monotonic") if rep.is_file() else None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.pi_load", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", nargs="?", help="diagnostics folder or threads.csv (default: newest)")
    p.add_argument("--run", help="the run recorded (a navigation_linker folder), to line its frames up")
    p.add_argument("--out", help="output folder (default: the recording's)")
    args = p.parse_args(argv)
    try:
        d = load(args.recording, args.run)
        diag_t0 = d["meta"].get("t0_monotonic")
        if args.run and (diag_t0 is None or d["nav_t0"] is None):
            print("note: can't line the run up (no t0_monotonic in meta.json or the run's report.json)",
                  file=sys.stderr)
        res = analyze(d["threads"], d["cores"], d["system"], d["nav"], diag_t0, d["nav_t0"], d["irqs"], d["procs"])
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else d["folder"]
    common.write_json(out / "pi_load.json", res)
    offset = None if res["aligned"] is None else d["nav_t0"] - diag_t0
    fig = figure(d["threads"], d["cores"], d["system"], res, f"Pi load: {d['folder'].name}", out / "pi_load.png",
                 d["nav"], offset)
    print(f"{d['folder']}\n" + "\n".join(report_lines(res)))
    print(f"wrote {out / 'pi_load.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
