"""Scheduling report: how late Linux woke each kind of thread, against the robot's deadlines, as graphs.

Purpose:
    sched_latency records wake-up lateness per case (a Python thread, cyclictest
    at normal and real-time priority). Its own figure is the tail; this adds
    the spread (histogram), the scale against the deadlines (the IMU's 10 ms,
    the frame's 50 ms, the motor watchdog's 500 ms), and, given a second run
    (idle against loaded), each case's worst wake-up before and after.

Main package:
    analyze(run, base) -> dict; sched_report.png and sched_report.json beside
    the run's sched_latency.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table
from src.params import FPS, SENSOR_RATE_HZ

JSON_NAME, HIST_NAME = "sched_latency.json", "histogram.csv"
CASES = ("python", "other", "fifo")
CASE_LABEL = {"python": "Python thread, normal", "other": "cyclictest, normal", "fifo": "cyclictest, real-time"}
CASE_COLORS = {"python": "#2a78d6", "other": "#eb6834", "fifo": "#1baf7a"}
DEADLINES_MS = (("IMU period", 1000.0 / SENSOR_RATE_HZ), ("frame", 1000.0 / FPS), ("motor watchdog", 500.0))
DEADLINE = "#B26A00"
# A worst case past this share of the IMU period is a finding
IMU_SHARE = 0.25

_CLI_HELP = """\
Graph a sched-latency run: each case's spread, its percentiles against the
robot's deadlines, and with --base another run (idle, against this one under
load), each case's worst wake-up in both. Writes sched_report.png and
sched_report.json beside the run's sched_latency.json.

Run from vision_stack/:
    python3 -m src.analysis.sched_report                              newest sched run
    python3 -m src.analysis.sched_report runs/sched_loaded_<time> --base runs/sched_idle_<time>
    make sched-report RUN=runs/sched_loaded_<time> BASE=runs/sched_idle_<time>
"""


def find(arg) -> Path:
    """The folder holding sched_latency.json: arg itself, or the newest under runs/."""
    if arg:
        p = Path(arg)
        p = p.parent if p.is_file() else p
        if (p / JSON_NAME).is_file():
            return p
        found = sorted(p.rglob(JSON_NAME)) if p.is_dir() else []
        if found:
            return found[-1].parent
        raise FileNotFoundError(f"no {JSON_NAME} in {p}")
    found = [f for r in common.SEARCH_ROOTS if Path(r).is_dir() for f in Path(r).rglob(JSON_NAME)]
    if not found:
        raise FileNotFoundError(f"no {JSON_NAME} under {', '.join(str(r) for r in common.SEARCH_ROOTS)}")
    return max(found, key=lambda f: f.stat().st_mtime).parent


def load(folder: Path) -> tuple[dict, dict]:
    """(sched_latency.json, {case: (us_late, counts)} from histogram.csv, empty without it)."""
    meta = json.loads((folder / JSON_NAME).read_text())
    hist = {}
    if (folder / HIST_NAME).is_file():
        t = Table(folder / HIST_NAME)
        us = t.numeric("us_late")
        for c in CASES:
            if t.has(c):
                n = t.numeric(c)
                keep = np.isfinite(n) & (n > 0)
                if keep.any():
                    hist[c] = (us[keep], n[keep])
    return meta, hist


def analyze(meta: dict, base: dict | None = None) -> dict:
    """Per case: the percentiles (us), the worst against each deadline, and the base run's worst for comparison."""
    cases = {c: s for c, s in (meta.get("cases") or {}).items() if s}
    if not cases:
        raise ValueError("no measured case in sched_latency.json")
    out = {"label": meta.get("label", ""), "base_label": (base or {}).get("label", "") if base else None,
           "cases": {}}
    for c, s in cases.items():
        b = ((base or {}).get("cases") or {}).get(c)
        out["cases"][c] = {**{k: s[k] for k in ("n", "p50", "p99", "p999", "max")},
                           "max_share": {name: s["max"] / 1000.0 / ms for name, ms in DEADLINES_MS},
                           "base_max": b["max"] if b else None, "base_p999": b["p999"] if b else None}
    out["findings"] = findings(out)
    return out


def findings(res: dict) -> list[str]:
    out = []
    imu_ms = DEADLINES_MS[0][1]
    for c, s in res["cases"].items():
        share = s["max_share"]["IMU period"]
        if share >= 1.0:
            out.append(f"{CASE_LABEL.get(c, c)}: the worst wake-up ({s['max'] / 1000:.1f} ms) missed a whole IMU period")
        elif share >= IMU_SHARE:
            out.append(f"{CASE_LABEL.get(c, c)}: the worst wake-up took {100 * share:.0f}% of the {imu_ms:.0f} ms IMU period")
        if s["base_max"]:
            ratio = s["max"] / s["base_max"]
            if ratio >= 3:
                out.append(f"{CASE_LABEL.get(c, c)}: worst {ratio:.0f}x the base run's "
                           f"({s['base_max']} -> {s['max']} us)")
    py, fifo = res["cases"].get("python"), res["cases"].get("fifo")
    if py and fifo and fifo["max"] and py["max"] / fifo["max"] >= 3:
        out.append(f"real-time priority cut the worst case {py['max'] / fifo['max']:.0f}x against a normal Python thread")
    return out


def figure(res: dict, hist: dict, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    compare = any(s["base_max"] for s in res["cases"].values())
    n = 3 if compare else 2
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n))
    ax = axes[0]
    for c in [c for c in CASES if c in hist]:
        us, cnt = hist[c]
        ax.step(np.maximum(us, 1), cnt, where="mid", color=CASE_COLORS[c], lw=1, label=CASE_LABEL[c])
        s = res["cases"].get(c)
        if s:
            ax.axvline(s["max"], color=CASE_COLORS[c], ls=":", lw=0.9)
    if hist:
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1, max(float(us.max()) for us, _ in hist.values()) * 1.5)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "no histogram.csv", transform=ax.transAxes, ha="center", color="#5F6B76")
    ax.set_xlabel("wake-up late by (us; dotted: each case's worst)", fontsize=8)
    ax.set_ylabel("wake-ups (log)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    ax = axes[1]
    names = [c for c in CASES if c in res["cases"]] + [c for c in res["cases"] if c not in CASES]
    for i, c in enumerate(names[::-1]):
        s = res["cases"][c]
        pts = [("p50", s["p50"]), ("p99.9", s["p999"]), ("max", s["max"])]
        ax.plot([pts[0][1] / 1000, pts[-1][1] / 1000], [i, i], color=CASE_COLORS.get(c, "#5F6B76"), lw=2, alpha=0.4)
        for j, (lab, us) in enumerate(pts):
            ax.plot(us / 1000, i, "o", color=CASE_COLORS.get(c, "#5F6B76"), ms=5 if lab != "max" else 7)
            ax.annotate(f"{lab} {us:,} us", (us / 1000, i), xytext=(0, 7 if j != 1 else -11),
                        textcoords="offset points", ha="center", fontsize=6.5)
    for name, ms in DEADLINES_MS:
        ax.axvline(ms, color=DEADLINE, ls="--", lw=0.9)
        ax.text(ms, len(names) - 0.45, f"{name}\n{ms:g} ms", color=DEADLINE, ha="center", va="bottom", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 1500)
    ax.set_ylim(-0.6, len(names) + 0.4)
    ax.set_yticks(range(len(names)), [CASE_LABEL.get(c, c) for c in names[::-1]], fontsize=8)
    ax.set_xlabel("ms, log: each case against the robot's deadlines", fontsize=8)
    if compare:
        ax = axes[2]
        for i, c in enumerate(names[::-1]):
            s = res["cases"][c]
            col = CASE_COLORS.get(c, "#5F6B76")
            ax.plot(s["max"] / 1000, i, "o", color=col, ms=7)
            ax.text(s["max"] / 1000 * 1.2, i, f"{s['max'] / 1000:.2f} ms", va="center", fontsize=7)
            if s["base_max"]:
                ax.plot(s["base_max"] / 1000, i, "o", mfc="white", mec=col, ms=7)
                ax.plot([s["base_max"] / 1000, s["max"] / 1000], [i, i], color=col, lw=1)
                ax.text(s["base_max"] / 1000 / 1.2, i, f"{s['base_max']:,} us", va="center", ha="right",
                        fontsize=7, color="#5F6B76")
        ax.axvline(DEADLINES_MS[0][1], color=DEADLINE, ls="--", lw=0.9)
        ax.set_xscale("log")
        lows = [v / 1000 for s in res["cases"].values() for v in (s["max"], s["base_max"]) if v]
        ax.set_xlim(min(lows) / 6, max(max(lows), DEADLINES_MS[0][1]) * 2)
        ax.set_yticks(range(len(names)), [CASE_LABEL.get(c, c) for c in names[::-1]], fontsize=8)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_xlabel(f"worst wake-up, ms (log)   ○ base ({res['base_label'] or 'base'})   "
                      f"● this run ({res['label'] or 'this run'})", fontsize=8)
    for a in axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    lines = [f"{'case':<24} {'n':>8} {'p50':>6} {'p99':>6} {'p99.9':>7} {'max':>8}  of IMU  base max (us)"]
    for c, s in res["cases"].items():
        base = "" if s["base_max"] is None else f"{s['base_max']:,}"
        lines.append(f"{CASE_LABEL.get(c, c):<24} {s['n']:>8,} {s['p50']:>6} {s['p99']:>6} {s['p999']:>7} "
                     f"{s['max']:>8,}  {100 * s['max_share']['IMU period']:>5.0f}%  {base}")
    return lines + ["findings"] + [f"  - {f}" for f in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.sched_report", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="sched-latency folder (default: the newest)")
    p.add_argument("--base", help="another sched-latency folder to compare with (e.g. idle)")
    p.add_argument("--out", help="output folder (default: the run's)")
    args = p.parse_args(argv)
    try:
        folder = find(args.run)
        meta, hist = load(folder)
        base = load(find(args.base))[0] if args.base else None
        res = analyze(meta, base)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else folder
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "sched_report.json", res)
    fig = figure(res, hist, f"Scheduling latency: {folder.name}", out / "sched_report.png")
    print(folder)
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'sched_report.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
