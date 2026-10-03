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
    monotonic clock and says what the slow frames coincided with: a clock
    drop, a busy frame loop, or neither.

Main package:
    analyze(...) -> dict: threads, process, cores, system, aligned, findings.
    report_lines(res): the printed report.
    pi_load.json, pi_load.png (CPU per thread, core load, temperature and
    clock, memory, and the run's frame intervals when aligned).

Input:
    A runs/diag_<time> folder (threads.csv, cores.csv, system.csv,
    meta.json). --run: the navigation_linker folder recorded, for lining
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
LATE = 1.5                  # a frame interval over this x the frame budget is a slow frame
MIN_LATE_FRAMES = 5         # fewer slow frames than this aren't worth attributing
BUDGET_MS = 1000.0 / FPS

PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f")

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


def analyze(threads: Table, cores: Table | None = None, system: Table | None = None,
            nav: Table | None = None, diag_t0: float | None = None, nav_t0: float | None = None) -> dict:
    missing = [c for c in ("tid", "name", "core", "cpu_pct") if not threads.has(c)]
    if missing:
        raise ValueError(f"{threads.path.name}: not a diagnostics threads.csv (no {', '.join(missing)})")
    rows = thread_rows(threads)
    res = {"threads": thread_summary(rows), "process": process_load(rows), "cores": core_load(cores),
           "system": soak.analyze(system) if system is not None and len(system) >= 2 else None,
           "since_boot": since_boot(system), "aligned": None}
    if nav is not None and diag_t0 is not None and nav_t0 is not None:
        res["aligned"] = align(rows, system, diag_t0, nav, nav_t0)
    res["findings"] = findings(res)
    # Caveats, not problems: soak's "too short to judge memory" on any run under ~6 minutes
    res["notes"] = [f for f in (res["system"] or {}).get("findings", []) if f.startswith("run too short")]
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


def report_lines(res: dict) -> list[str]:
    p = res["process"]
    lines = [f"process   CPU mean {common.fmt(p['total_cpu_pct']['mean'], 1)}% of one core, "
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
    a = res["aligned"]
    if a is not None:
        lines.append(f"aligned   {a['frames']} frames, {a.get('late', 0)} slow; main CPU slow {common.fmt(a.get('late_main_cpu'), 0)}%"
                     f" vs normal {common.fmt(a.get('normal_main_cpu'), 0)}%; correlation main CPU / interval "
                     f"{common.fmt(a.get('corr_main_cpu_interval'))}")
    lines += ["", "findings"] + ([f"  - {f}" for f in res["findings"]] or ["  none: nothing past the thresholds"])
    lines += [f"  (note: {n})" for n in res.get("notes", [])]
    return lines


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
    panels = 3 + (system is not None) + (nav is not None and offset_s is not None)
    fig, axes = plt.subplots(panels, 1, figsize=(13, 2.2 * panels), sharex=True)
    ax = axes[0]
    top = sorted(cpu, key=lambda n: -cpu[n].mean())
    for i, name in enumerate(top[:6]):
        ax.plot(times, cpu[name], lw=0.9, color=PALETTE[i % len(PALETTE)], label=name)
    if len(top) > 6:
        ax.plot(times, sum(cpu[n] for n in top[6:]), lw=0.8, color="#bbbbbb", label="others")
    ax.set_ylabel("CPU % of a core", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax = axes[1]
    if cores is not None and len(cores):
        c, b, t = cores.numeric("core"), cores.numeric("busy_pct"), cores.numeric("elapsed_s")
        ids = sorted({int(x) for x in c[~np.isnan(c)]})
        ct = sorted(set(t[~np.isnan(t)]))
        grid = np.full((len(ids), len(ct)), np.nan)
        for ci, ti, bi in zip(c, t, b):
            if not np.isnan(ci):
                grid[ids.index(int(ci)), ct.index(ti)] = bi
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100, interpolation="nearest",
                       extent=(ct[0], ct[-1], len(ids) - 0.5, -0.5))
        fig.colorbar(im, ax=ax, pad=0.01, label="busy %")
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([f"core {i}" for i in ids], fontsize=7)
    ax = axes[2]
    if system is not None:
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
        ax = axes[3]
        ax.plot(st, system.numeric("rss_mb"), lw=0.9, color="#1f77b4")
        ax.set_ylabel("run memory MB", fontsize=8, color="#1f77b4")
        ax2 = ax.twinx()                    # the Pi's free memory is far larger: its own axis
        ax2.plot(st, system.numeric("mem_available_mb"), lw=0.9, color="#ff7f0e")
        ax2.set_ylabel("Pi available MB", fontsize=8, color="#ff7f0e")
    if nav is not None and offset_s is not None:
        ax = axes[-1]
        t = nav.numeric("t")
        ax.plot(offset_s + t[1:], np.diff(t) * 1000.0, lw=0.6)
        ax.axhline(LATE * BUDGET_MS, color="#d62728", lw=0.6, ls="--")
        ax.set_ylabel("frame interval ms", fontsize=8)
    axes[-1].set_xlabel("time since the recording started, s")
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
           "meta": meta, "nav": None, "nav_t0": None}
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
        res = analyze(d["threads"], d["cores"], d["system"], d["nav"], diag_t0, d["nav_t0"])
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
