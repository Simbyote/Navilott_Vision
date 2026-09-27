"""Soak: does the Pi hold its frame rate as it heats up, and does memory stay flat?

Purpose:
    A 30-second bench run can't show thermal throttling or a slow leak; a
    10-30 minute run can. This lines up the system samples (temperature,
    clock, throttle flags, memory) with the loop timings on one clock and
    reports the three things that go wrong over time: throttling, memory
    growth, and a loop that slows as the run goes on.

Main package:
    analyze()   temperature rise, clock, throttle events, RSS leak rate
                (MB/min, fitted after warm-up), per-window loop timing and
                its drift, and a findings list in plain words
    soak.png    temperature, CPU clock, memory and loop time on one time axis

Input (a soak artifact folder, or any folder holding these):
    system.csv        SystemMonitor rows, required
    soak_frames.csv   per-frame elapsed_s and interval_ms / *_ms, optional;
                      without it only the system side is reported
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import BUDGET_MS, Table, fmt, interval_ms, stats
from src.params import FPS

SYSTEM_CSV = "system.csv"
FRAMES_CSV = "soak_frames.csv"
THROTTLE_FLAGS = ("under_voltage", "freq_capped", "throttled", "soft_temp_limit")
FLAG_COLORS = {"under_voltage": "#bcbd22", "freq_capped": "#ff7f0e",
               "throttled": "#d62728", "soft_temp_limit": "#8c564b"}
WINDOW_S = 60.0
WARMUP_S = 60.0             # imports, buffer pools and caches settle before the leak fit
LEAK_MB_PER_MIN = 0.25      # sustained RSS growth above this is reported as a suspected leak
MIN_FIT_S = 300.0           # a leak verdict needs this much run after warm-up; shorter runs
                            # still get a trend, but allocator pools can look like growth
SLOWDOWN = 1.10             # last window's loop p95 over the first's by this factor is a slowdown

_CLI_HELP = """\
Temperature, throttling, memory growth and loop timing over a long run.
Writes soak.png and soak.json next to system.csv.

Record with:
    pytest --hardware --soak-minutes=15 src/tests/test_soak.py
Then, from vision_stack/, venv active:
    python3 -m src.analysis.soak                   newest soak run
    python3 -m src.analysis.soak <folder>
"""


def _window_means(t: np.ndarray, v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        sel = (t >= edges[i]) & (t < edges[i + 1]) & ~np.isnan(v)
        if sel.any():
            out[i] = float(np.mean(v[sel]))
    return out


def analyze(system: Table, frames: Table | None = None, budget_ms: float = BUDGET_MS,
            window_s: float = WINDOW_S) -> dict:
    """
    Outputs:
        {
          "duration_s",
          "temp_c": {"start", "end", "max", "rise"} (start/end: means over
                    the first and last window_s of the run),
          "cpu_mhz": {"max", "min", "share_below_max"},
          "throttle": {flag: {"samples", "first_s"}} while set during the run,
          "memory": {"rss_start_mb", "rss_end_mb", "rss_max_mb",
                     "leak_mb_per_min", "mem_available_min_mb"},
          "loop": None, or {"windows": [{"start_s", "median", "p95", "temp_c"}],
                  "overall": stats(), "drift_p95": last/first window p95,
                  "over_budget_share"},
          "findings": [plain-language lines],
        }
    """
    t = system.numeric("elapsed_s")
    if np.all(np.isnan(t)) or len(system) < 2:
        raise ValueError(f"{system.path.name}: fewer than 2 samples")
    duration = float(np.nanmax(t))
    edges = np.arange(0.0, duration + window_s, window_s)
    if edges[-1] <= duration:                  # the last sample belongs to a window too
        edges = np.append(edges, edges[-1] + window_s)

    temp = system.numeric("temp_c")
    temp_w = _window_means(t, temp, edges)
    temp_out = None
    if not np.all(np.isnan(temp)):
        # Start and end are the first and last window_s of the run, not grid
        # windows, so a short last grid window can't make the end noisy
        head = temp[(t < min(window_s, duration / 2)) & ~np.isnan(temp)]
        tail = temp[(t > duration - min(window_s, duration / 2)) & ~np.isnan(temp)]
        start = float(head.mean()) if head.size else float(np.nanmin(temp))
        end = float(tail.mean()) if tail.size else float(temp[~np.isnan(temp)][-1])
        temp_out = {"start": start, "end": end, "max": float(np.nanmax(temp)), "rise": end - start}

    mhz = system.numeric("cpu_mhz")
    mhz_out = None
    if not np.all(np.isnan(mhz)):
        top = float(np.nanmax(mhz))
        v = mhz[~np.isnan(mhz)]
        mhz_out = {"max": top, "min": float(v.min()), "share_below_max": float(np.mean(v < top - 1))}

    throttle = {}
    for flag in THROTTLE_FLAGS:
        if system.has_values(flag):
            on = system.numeric(flag) == 1
            if on.any():
                throttle[flag] = {"samples": int(on.sum()), "first_s": float(t[np.argmax(on)])}

    rss = system.numeric("rss_mb")
    memory = None
    if not np.all(np.isnan(rss)):
        sel = (t >= min(WARMUP_S, duration / 2)) & ~np.isnan(rss)
        slope, span = None, 0.0
        if sel.sum() >= 3 and np.ptp(t[sel]) > 0:
            slope = float(np.polyfit(t[sel] / 60.0, rss[sel], 1)[0])
            span = float(np.ptp(t[sel]))
        v = rss[~np.isnan(rss)]
        avail = system.numeric("mem_available_mb")
        memory = {"rss_start_mb": float(v[0]), "rss_end_mb": float(v[-1]), "rss_max_mb": float(v.max()),
                  "leak_mb_per_min": slope, "fit_span_s": span,
                  "mem_available_min_mb": None if np.all(np.isnan(avail)) else float(np.nanmin(avail))}

    loop = None
    if frames is not None:
        ft = frames.numeric("elapsed_s")
        iv = interval_ms(frames)
        if iv is not None and not np.all(np.isnan(ft)):
            windows = []
            for i in range(len(edges) - 1):
                sel = (ft >= edges[i]) & (ft < edges[i + 1]) & ~np.isnan(iv)
                if sel.sum() >= 5:
                    windows.append({"start_s": float(edges[i]), "median": float(np.median(iv[sel])),
                                    "p95": float(np.percentile(iv[sel], 95)),
                                    "temp_c": None if np.isnan(temp_w[i]) else float(temp_w[i])})
            valid = iv[~np.isnan(iv)]
            loop = {"windows": windows, "overall": stats(valid),
                    "drift_p95": (windows[-1]["p95"] / windows[0]["p95"]) if len(windows) >= 2 else None,
                    "over_budget_share": float(np.mean(valid > budget_ms)) if valid.size else None}

    findings = []
    if throttle:
        names = ", ".join(f"{k.replace('_', ' ')} from {v['first_s']:.0f} s" for k, v in throttle.items())
        findings.append(f"throttling: {names}")
    if memory and memory["leak_mb_per_min"] is not None:
        if memory["fit_span_s"] < MIN_FIT_S:
            findings.append(f"run too short to judge memory growth ({memory['fit_span_s'] / 60:.1f} min "
                            f"after warm-up, need {MIN_FIT_S / 60:.0f})")
        elif memory["leak_mb_per_min"] > LEAK_MB_PER_MIN:
            findings.append(f"memory grows {memory['leak_mb_per_min']:.2f} MB/min after warm-up: suspected leak")
    if loop and loop["drift_p95"] is not None and loop["drift_p95"] > SLOWDOWN:
        findings.append(f"loop p95 {100 * (loop['drift_p95'] - 1):.0f}% slower in the last window than the first")
    if not any(f.startswith(("throttling", "memory grows", "loop p95")) for f in findings):
        findings.append("no throttling, memory growth or slowdown found")

    return {"duration_s": duration, "budget_ms": budget_ms, "temp_c": temp_out, "cpu_mhz": mhz_out,
            "throttle": throttle, "memory": memory, "loop": loop, "findings": findings}


def figure(system: Table, result: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    t = system.numeric("elapsed_s") / 60.0
    panels = [p for p in ("temp", "mhz", "rss", "loop")
              if (p == "temp" and result["temp_c"]) or (p == "mhz" and result["cpu_mhz"])
              or (p == "rss" and result["memory"]) or (p == "loop" and result["loop"] and result["loop"]["windows"])]
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.9 * len(panels) + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, panel in zip(axes, panels):
        if panel == "temp":
            ax.plot(t, system.numeric("temp_c"), color="#d62728", linewidth=1)
            ax.set_ylabel("SoC °C")
            for flag, info in result["throttle"].items():
                on = system.numeric(flag) == 1
                ax.fill_between(t, 0, 1, where=on, transform=ax.get_xaxis_transform(),
                                color=FLAG_COLORS[flag], alpha=0.18, linewidth=0,
                                label=flag.replace("_", " "))
            if result["throttle"]:
                ax.legend(fontsize=7, frameon=False, loc="upper left")
        elif panel == "mhz":
            ax.plot(t, system.numeric("cpu_mhz"), color="#9467bd", linewidth=1)
            ax.set_ylabel("CPU MHz")
        elif panel == "rss":
            rss = system.numeric("rss_mb")
            ax.plot(t, rss, color="#1f77b4", linewidth=1, label="process RSS")
            slope = result["memory"]["leak_mb_per_min"]
            if slope is not None:
                sel = ~np.isnan(rss) & (t * 60 >= min(WARMUP_S, result["duration_s"] / 2))
                if sel.any():
                    b = np.nanmean(rss[sel]) - slope * np.mean(t[sel])
                    ax.plot(t[sel], slope * t[sel] + b, color="black", linestyle="--", linewidth=1,
                            label=f"{slope:+.2f} MB/min")
            ax.set_ylabel("MB")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        else:
            w = result["loop"]["windows"]
            mid = [(x["start_s"] + WINDOW_S / 2) / 60.0 for x in w]
            ax.plot(mid, [x["median"] for x in w], marker="o", color="#2ca02c", linewidth=1, label="median")
            ax.plot(mid, [x["p95"] for x in w], marker="o", color="#2ca02c", linestyle="--",
                    linewidth=1, label="p95")
            ax.axhline(result["budget_ms"], color="black", linewidth=1, label=f"{result['budget_ms']:.0f} ms budget")
            ax.set_ylabel("loop ms")
            ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("minutes")
    axes[0].set_title(title, fontsize=11, loc="left")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def load(arg) -> tuple:
    """(system Table, frames Table or None, folder) for a folder or a system.csv."""
    system_path = common.find_csv(arg, SYSTEM_CSV)
    frames_path = system_path.parent / FRAMES_CSV
    frames = Table(frames_path) if frames_path.is_file() else None
    return Table(system_path), frames, system_path.parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.soak", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="soak folder or system.csv (default: newest)")
    p.add_argument("--fps", type=float, default=FPS, help=f"target rate (default {FPS})")
    p.add_argument("--out", help="output folder (default: next to system.csv)")
    args = p.parse_args(argv)

    try:
        system, frames, folder = load(args.run)
        res = analyze(system, frames, 1000.0 / args.fps)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else folder
    common.write_json(out / "soak.json", res)
    fig = figure(system, res, f"Soak: {folder.name}", out / "soak.png")

    print(f"{folder}  ({res['duration_s'] / 60:.1f} min)")
    if res["temp_c"]:
        tc = res["temp_c"]
        print(f"  temperature   {tc['start']:.1f} -> {tc['end']:.1f} °C, max {tc['max']:.1f}")
    if res["cpu_mhz"]:
        m = res["cpu_mhz"]
        print(f"  cpu clock     {m['min']:.0f}-{m['max']:.0f} MHz, below max {100 * m['share_below_max']:.1f}% of samples")
    if res["memory"]:
        mem = res["memory"]
        print(f"  process RSS   {mem['rss_start_mb']:.1f} -> {mem['rss_end_mb']:.1f} MB, "
              f"trend {fmt(mem['leak_mb_per_min'])} MB/min; system available min "
              f"{fmt(mem['mem_available_min_mb'], 0)} MB")
    if res["loop"]:
        lp = res["loop"]
        print(f"  loop          median {fmt(lp['overall']['p50'])} ms, p95 {fmt(lp['overall']['p95'])} ms, "
              f"over budget {fmt(100 * lp['over_budget_share'], 1) if lp['over_budget_share'] is not None else ''}%")
    for line in res["findings"]:
        print(f"  -> {line}")
    print(f"wrote {out / 'soak.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
