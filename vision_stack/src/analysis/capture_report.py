"""Capture report: what each step costs a frame between the sensor and Python, as graphs.

Purpose:
    capture_anatomy times the camera path per frame: sensor to the first
    GStreamer pad (VideoCore), each element after it (the ARM) and the whole
    pipeline. This draws how old a frame is at the sink and how much ARM time
    it cost, both against the 50 ms frame, and each step frame by frame, so a
    copy in the pipeline (videoconvert, videoflip) shows as what it costs.

Main package:
    analyze(table) -> dict; capture_report.png and capture_report.json beside
    latency.csv (from make capture-anatomy, or pytest --hardware's anatomy/).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table
from src.params import FPS

CSV_NAME = "latency.csv"
BUDGET_MS = 1000.0 / FPS
# An element "cost" this close to the frame period is the probe timing the wait between frames, not work
PERIOD_ARTIFACT = 0.2
# A level shift in a step's per-frame time worth naming (later / earlier median)
STEP_RATIO = 1.2
STEP_MIN_MS = 1.0               # ...and at least this much, so noise on a cheap step isn't one
EDGE = 10                       # frames kept on each side of a level shift
VIDEOCORE, ARM, REST, BUDGET = "#2A9A88", "#1E63B5", "#C9D1D8", "#B26A00"
ELEMENT_SHADES = ("#1E63B5", "#7FA6D9", "#173F73", "#4C86C9")

_CLI_HELP = """\
Graph capture_anatomy's per-frame timings: how old a frame is when Python
gets it, how much ARM time the pipeline's elements cost, and each step frame
by frame. Writes capture_report.png and capture_report.json beside latency.csv.

Run from vision_stack/:
    python3 -m src.analysis.capture_report                    newest capture-anatomy run
    python3 -m src.analysis.capture_report runs/anatomy_<time>
    make capture-report RUN=runs/anatomy_<time>
"""


def is_latency(path: Path) -> bool:
    try:
        t = Table(path, 0)
    except (OSError, ValueError):
        return False
    return all(t.has(c) for c in ("kind", "element", "ms"))


def find(arg) -> Path:
    """arg as given, a folder's latency.csv, or the newest one under runs/ and artifacts/."""
    if arg and Path(arg).is_file():
        return Path(arg)
    roots = [Path(arg)] if arg else [Path(r) for r in common.SEARCH_ROOTS]
    found = [p for r in roots if r.is_dir() for p in r.rglob(CSV_NAME) if is_latency(p)]
    if not found:
        raise FileNotFoundError(f"no capture-anatomy {CSV_NAME} in {', '.join(str(r) for r in roots)}")
    return max(found, key=lambda p: p.stat().st_mtime)


def series(table: Table) -> dict:
    """{(kind, element): (ts_ms, ms)} in file order."""
    kinds, elements = table.text("kind"), table.text("element")
    ms = table.numeric("ms")
    ts = table.numeric("ts_ms") if table.has("ts_ms") else np.full(len(table), np.nan)
    out = {}
    for i, key in enumerate(zip(kinds, elements)):
        out.setdefault(key, ([], []))
        out[key][0].append(ts[i])
        out[key][1].append(ms[i])
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def level_shift(v: np.ndarray, edge: int = EDGE) -> tuple[int, float, float] | None:
    """(frame, median before, median after) where the mean shifts most; None for a short series."""
    v = v[np.isfinite(v)]
    if len(v) < 2 * edge + 1:
        return None
    k = max(range(edge, len(v) - edge), key=lambda i: abs(v[:i].mean() - v[i:].mean()))
    return k, float(np.median(v[:k])), float(np.median(v[k:]))


def analyze(table: Table) -> dict:
    """Per step: median, p95, max; the frame's age at the sink and its ARM time. ValueError for another CSV."""
    if not all(table.has(c) for c in ("kind", "element", "ms")):
        raise ValueError(f"{table.path} isn't capture_anatomy's latency.csv (kind, element, ms)")
    s = series(table)
    camera = next((v for (k, _), v in s.items() if k == "camera"), None)
    period = (float(np.nanmedian(np.diff(camera[0]))) if camera is not None and len(camera[0]) > 1
              else BUDGET_MS)
    steps = []
    for (kind, element), (_, ms) in s.items():
        ms = ms[np.isfinite(ms)]
        if not ms.size:
            continue
        med = float(np.median(ms))
        artifact = kind == "element" and abs(med - period) < PERIOD_ARTIFACT * period
        shift = level_shift(ms)
        steps.append({"kind": kind, "element": element, "frames": int(ms.size), "median_ms": med,
                      "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max()),
                      "period_artifact": artifact,
                      "shift": None if shift is None or not (max(shift[1:]) > STEP_RATIO * max(min(shift[1:]), 1e-6)
                                                             and abs(shift[2] - shift[1]) >= STEP_MIN_MS)
                      else {"frame": shift[0], "before_ms": shift[1], "after_ms": shift[2]}})
    med = lambda kind: sum(x["median_ms"] for x in steps if x["kind"] == kind)      # noqa: E731
    arm = sum(x["median_ms"] for x in steps if x["kind"] == "element" and not x["period_artifact"])
    hw, pipe = med("camera"), med("pipeline")
    res = {"frame_period_ms": period, "steps": steps, "sensor_to_pad_ms": hw, "pipeline_ms": pipe,
           "frame_age_ms": hw + pipe, "arm_ms": arm, "arm_share": arm / BUDGET_MS}
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    for x in res["steps"]:
        name = x["element"]
        if x["period_artifact"]:
            out.append(f"{name}'s probe reads {x['median_ms']:.1f} ms, the frame period: it timed the wait between "
                       "frames, not work; the pipeline's own time is the cost")
        elif x["kind"] == "element" and x["median_ms"] > 0.1 * BUDGET_MS:
            out.append(f"{name} costs {x['median_ms']:.1f} ms of ARM time a frame ({100 * x['median_ms'] / BUDGET_MS:.0f}% "
                       "of a core at 20 FPS): a copy the ISP or OpenCV could do instead")
        if x["shift"]:
            sh = x["shift"]
            out.append(f"{name} changed level at frame {sh['frame']}: {sh['before_ms']:.1f} -> {sh['after_ms']:.1f} ms "
                       "(clock, load or thermal change mid-run)")
    if res["frame_age_ms"] > BUDGET_MS:
        out.append(f"a frame is {res['frame_age_ms']:.1f} ms old when Python gets it, over a {BUDGET_MS:.0f} ms frame")
    return out


def figure(table: Table, res: dict, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    s = series(table)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [1, 1.5]})
    # bar 1: the frame's age at the sink; bar 2: ARM time
    x, h = 0.0, 0.55
    hw = res["sensor_to_pad_ms"]
    ax.barh(1, hw, height=h, color=VIDEOCORE, edgecolor="white")
    if hw > 4:
        ax.text(hw / 2, 1, f"sensor → first pad\n{hw:.1f} ms", ha="center", va="center", color="white", fontsize=8)
    x = hw
    work = [st for st in res["steps"] if st["kind"] == "element" and not st["period_artifact"]]
    for st in work:
        ax.barh(1, st["median_ms"], left=x, height=h, color=ARM, edgecolor="white")
        x += st["median_ms"]
    rest = max(res["pipeline_ms"] - sum(st["median_ms"] for st in work), 0.0)
    ax.barh(1, rest, left=x, height=h, color=REST, edgecolor="white")
    ax.text(res["frame_age_ms"] + 0.8, 1, f"{res['frame_age_ms']:.1f} ms", va="center", fontsize=8)
    x = 0.0
    for st in work:
        ax.barh(0, st["median_ms"], left=x, height=h, color=ARM, edgecolor="white")
        if st["median_ms"] > 4:
            ax.text(x + st["median_ms"] / 2, 0, f"{st['element']}\n{st['median_ms']:.1f}", ha="center",
                    va="center", color="white", fontsize=7)
        x += st["median_ms"]
    for st in res["steps"]:
        if st["period_artifact"]:
            ax.barh(0, st["median_ms"], left=x, height=h, color="white", edgecolor="#5F6B76", hatch="///", lw=0.8)
            ax.text(x + st["median_ms"] + 0.8, 0, f"{st['element']}: the frame period, not a cost",
                    va="center", fontsize=7, color="#5F6B76")
    ax.text(res["arm_ms"] + 0.8, -0.35, f"ARM {res['arm_ms']:.1f} ms ({100 * res['arm_share']:.0f}% of a core)",
            va="center", fontsize=8)
    ax.axvline(BUDGET_MS, color=BUDGET, ls="--", lw=1)
    ax.text(BUDGET_MS, 1.45, f"{BUDGET_MS:.0f} ms", color=BUDGET, ha="center", fontsize=8)
    ax.set_yticks([0, 1], ["ARM time\nper frame", "frame age\nat the sink"], fontsize=8)
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(0, max(res["frame_age_ms"], BUDGET_MS, res["frame_period_ms"] + res["arm_ms"]) * 1.25)
    ax.set_xlabel("ms (median per frame)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    colors = {"camera": VIDEOCORE, "pipeline": "#5F6B76"}
    shade = iter(ELEMENT_SHADES * 4)
    for (kind, element), (_, ms) in s.items():
        c = colors.get(kind) or next(shade)
        style = "--" if kind == "pipeline" else "-"
        ax2.plot(np.arange(len(ms)), ms, color=c, lw=0.9, ls=style)
        ax2.text(len(ms) + 1, float(np.nanmedian(ms[-20:])), f"{element} ({kind})", va="center", fontsize=7, color=c)
    for st in res["steps"]:
        if st["shift"]:
            ax2.axvline(st["shift"]["frame"], color=BUDGET, ls=":", lw=0.9)
    ax2.set_xlim(0, max(len(ms) for _, ms in s.values()) * 1.3)
    ax2.set_ylim(0, None)
    ax2.set_xlabel("frame", fontsize=8)
    ax2.set_ylabel("ms", fontsize=8)
    for a in (ax, ax2):
        a.tick_params(labelsize=7)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    lines = [f"frame period {res['frame_period_ms']:.1f} ms; a frame is {res['frame_age_ms']:.1f} ms old at the sink "
             f"(sensor -> first pad {res['sensor_to_pad_ms']:.1f}, pipeline {res['pipeline_ms']:.1f}); "
             f"ARM {res['arm_ms']:.1f} ms a frame",
             f"{'step':<34} {'frames':>6} {'median':>7} {'p95':>7} {'max':>7}"]
    for x in res["steps"]:
        name = f"{x['element']} ({x['kind']})" + (" *" if x["period_artifact"] else "")
        lines.append(f"{name:<34} {x['frames']:>6} {x['median_ms']:>7.2f} {x['p95_ms']:>7.2f} {x['max_ms']:>7.1f}")
    return lines + ["findings"] + [f"  - {f}" for f in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.capture_report", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="capture-anatomy folder or latency.csv (default: the newest)")
    p.add_argument("--out", help="output folder (default: next to the CSV)")
    args = p.parse_args(argv)
    try:
        path = find(args.run)
        table = Table(path)
        res = analyze(table)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "capture_report.json", res)
    fig = figure(table, res, f"Capture: {path.parent.name}", out / "capture_report.png")
    print(path)
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'capture_report.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
