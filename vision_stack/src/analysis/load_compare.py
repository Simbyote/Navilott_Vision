"""Load comparison: two diagnostics recordings side by side, before and after a change.

Purpose:
    pi_load reads one recording; a change (the camera's turn moved out of
    GStreamer, parallel perception, the recorder off) is judged by setting two
    against each other. Only each run's busy stretch counts (main over half a
    core, once the camera is open), so start-up and shutdown don't blur it.
    This draws each thread's CPU before and after, the busy cores over the
    stretch, and the system's heat, clock and memory.

Main package:
    compare(before, after) -> dict; load_compare.png and load_compare.json in the after run's folder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common, pi_load
from src.diagnostics.threads import MAIN_THREAD

CAMERA_THREAD = "task0"         # GStreamer's streaming thread: appears once the camera opens
MAIN_BUSY = 50.0                # main over this % of a core is the robot running
TOP_THREADS = 8
# A thread whose mean CPU moved this much (points of a core) is a finding
THREAD_CHANGE = 10.0
BEFORE, AFTER, BUDGET = "#9AA5B1", "#1E63B5", "#B26A00"

_CLI_HELP = """\
Two diagnostics recordings (make diag-nav / diag-run) side by side: each
thread's CPU, the busy cores and the system, over each run's busy stretch.
Writes load_compare.png and load_compare.json in the AFTER run's folder.

Run from vision_stack/:
    python3 -m src.analysis.load_compare runs/diag_<before> runs/diag_<after>
    python3 -m src.analysis.load_compare BEFORE AFTER --window-before 20-110 --window-after 60-170
    make load-compare BASE=runs/diag_<before> NEW=runs/diag_<after>
"""


def window_of(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    a, b = (float(x) for x in text.split("-"))
    return a, b


class Run:
    """One recording, cut to its busy stretch: the first to the last sample with main over MAIN_BUSY."""

    def __init__(self, folder, window: tuple[float, float] | None = None):
        rec = pi_load.load(folder)
        self.folder, self.threads, self.cores, self.system = rec["folder"], rec["threads"], rec["cores"], rec["system"]
        t, names = self.threads.numeric("elapsed_s"), self.threads.text("name")
        cpu = self.threads.numeric("cpu_pct")
        cam = [x for x, n in zip(t, names) if n == CAMERA_THREAD]
        start = min(cam) if cam else float(np.nanmin(t))
        busy = [x for x, n, c in zip(t, names, cpu) if n == MAIN_THREAD and x >= start and c > MAIN_BUSY]
        self.t0, self.t1 = window or ((min(busy), max(busy)) if busy else (float(np.nanmin(t)), float(np.nanmax(t))))

    def _in(self, table):
        t = table.numeric("elapsed_s")
        return (t >= self.t0) & (t <= self.t1)

    def per_thread(self) -> dict:
        """{name: mean % of a core over the stretch}; threads sharing a name are summed per sample first."""
        m = self._in(self.threads)
        t, names, cpu = self.threads.numeric("elapsed_s")[m], np.array(self.threads.text("name"))[m], \
            self.threads.numeric("cpu_pct")[m]
        out = {}
        samples = max(len(set(t)), 1)
        for n in set(names):
            out[n] = float(np.nansum(cpu[names == n])) / samples
        return out

    def busy_cores(self) -> tuple[np.ndarray, np.ndarray]:
        """(seconds into the stretch, cores busy 0-4) per sample."""
        if self.cores is None:
            return np.array([]), np.array([])
        m = self._in(self.cores)
        t, busy = self.cores.numeric("elapsed_s")[m], self.cores.numeric("busy_pct")[m]
        ts = np.unique(t)
        return ts - self.t0, np.array([np.nansum(busy[t == x]) / 100.0 for x in ts])

    def system_means(self) -> dict:
        if self.system is None:
            return {}
        m = self._in(self.system)
        return {k: float(np.nanmean(self.system.numeric(k)[m])) for k in ("temp_c", "cpu_mhz", "mem_available_mb",
                                                                          "rss_mb", "load_1m")
                if self.system.has(k) and np.isfinite(self.system.numeric(k)[m]).any()}


def compare(before: Run, after: Run) -> dict:
    tb, ta = before.per_thread(), after.per_thread()
    names = sorted(set(tb) | set(ta), key=lambda n: -max(tb.get(n, 0.0), ta.get(n, 0.0)))
    threads = [{"name": n, "before": tb.get(n, 0.0), "after": ta.get(n, 0.0)} for n in names]
    cb, ca = before.busy_cores()[1], after.busy_cores()[1]
    res = {"before": {"folder": str(before.folder), "window_s": [before.t0, before.t1],
                      "process_pct": sum(tb.values()), "busy_cores": float(np.mean(cb)) if cb.size else None,
                      "system": before.system_means()},
           "after": {"folder": str(after.folder), "window_s": [after.t0, after.t1],
                     "process_pct": sum(ta.values()), "busy_cores": float(np.mean(ca)) if ca.size else None,
                     "system": after.system_means()},
           "threads": threads}
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    for t in res["threads"]:
        d = t["after"] - t["before"]
        if abs(d) >= THREAD_CHANGE:
            out.append(f"{t['name']}: {t['before']:.0f}% -> {t['after']:.0f}% of a core ({d:+.0f})")
    b, a = res["before"]["busy_cores"], res["after"]["busy_cores"]
    if b is not None and a is not None and abs(a - b) >= 0.25:
        out.append(f"busy cores {b:.2f} -> {a:.2f} on average")
    sb, sa = res["before"]["system"], res["after"]["system"]
    if "temp_c" in sb and "temp_c" in sa and abs(sa["temp_c"] - sb["temp_c"]) >= 3:
        out.append(f"the SoC ran {sa['temp_c'] - sb['temp_c']:+.0f} C against before: check both started cool")
    return out


def figure(before: Run, after: Run, res: dict, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    rows = res["threads"][:TOP_THREADS][::-1]
    h = 0.38
    for i, t in enumerate(rows):
        ax.barh(i + h / 2, t["before"], height=h, color=BEFORE)
        ax.barh(i - h / 2, t["after"], height=h, color=AFTER)
        ax.text(t["before"] + 1, i + h / 2, f"{t['before']:.0f}%", va="center", fontsize=7, color="#5F6B76")
        ax.text(t["after"] + 1, i - h / 2, f"{t['after']:.0f}%", va="center", fontsize=7)
    ax.axvline(100, color=BUDGET, ls="--", lw=0.9)
    ax.text(99, len(rows) - 0.55, "one core", color=BUDGET, fontsize=7, ha="right")
    ax.set_yticks(range(len(rows)), [t["name"] for t in rows], fontsize=8)
    ax.set_xlabel("% of one core, mean over each run's busy stretch (grey: before, blue: after)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    ax = fig.add_subplot(gs[1, 0])
    for run, color, label in ((before, BEFORE, "before"), (after, AFTER, "after")):
        x, y = run.busy_cores()
        if x.size:
            k = min(5, x.size)
            ax.plot(x[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid"), color=color, lw=1.2, label=label)
            ax.axhline(float(np.mean(y)), color=color, ls=":", lw=0.8)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("seconds into the busy stretch", fontsize=8)
    ax.set_ylabel("cores busy (of 4)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[1, 1])
    keys = [k for k in ("temp_c", "cpu_mhz", "mem_available_mb", "rss_mb") if k in res["before"]["system"]
            and k in res["after"]["system"]]
    for i, k in enumerate(keys):
        b, a = res["before"]["system"][k], res["after"]["system"][k]
        top = max(b, a) or 1.0
        ax.barh(i + 0.19, b / top, height=0.38, color=BEFORE)
        ax.barh(i - 0.19, a / top, height=0.38, color=AFTER)
        ax.text(1.02, i, f"{b:.0f} -> {a:.0f}", va="center", fontsize=7)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_xlim(0, 1.45)
    ax.set_xticks([])
    ax.set_xlabel("system, each against the larger of the two", fontsize=8)
    for a in fig.axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    b, a = res["before"], res["after"]
    fmt = lambda v: "--" if v is None else f"{v:.2f}"        # noqa: E731
    lines = [f"before {Path(b['folder']).name}: {b['window_s'][0]:.0f}-{b['window_s'][1]:.0f} s, process "
             f"{b['process_pct']:.0f}% of a core, {fmt(b['busy_cores'])} cores busy",
             f"after  {Path(a['folder']).name}: {a['window_s'][0]:.0f}-{a['window_s'][1]:.0f} s, process "
             f"{a['process_pct']:.0f}% of a core, {fmt(a['busy_cores'])} cores busy",
             f"{'thread':<20} {'before':>7} {'after':>7}"]
    lines += [f"{t['name']:<20} {t['before']:>6.1f}% {t['after']:>6.1f}%" for t in res["threads"][:TOP_THREADS]]
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing moved much"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.load_compare", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", help="the diag_ recording before the change")
    p.add_argument("after", help="the diag_ recording after it")
    p.add_argument("--window-before", metavar="A-B", help="the before run's stretch in seconds, by hand")
    p.add_argument("--window-after", metavar="A-B", help="the after run's stretch in seconds, by hand")
    p.add_argument("--out", help="output folder (default: the after run's)")
    args = p.parse_args(argv)
    try:
        before = Run(args.before, window_of(args.window_before))
        after = Run(args.after, window_of(args.window_after))
        res = compare(before, after)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else after.folder
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "load_compare.json", res)
    fig = figure(before, after, res, f"Load: {before.folder.name} -> {after.folder.name}", out / "load_compare.png")
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'load_compare.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
