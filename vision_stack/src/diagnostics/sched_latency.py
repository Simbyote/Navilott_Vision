"""Scheduling latency: how late Linux wakes a thread, against the robot's deadlines.

Purpose:
    The robot runs on Linux, not a real-time OS. What an RTOS buys is a
    small, bounded delay between a thread's wake-up time and when it
    actually runs; on Linux that delay is usually small and occasionally
    not. The robot's deadlines are soft (a 50 ms frame, a 10 ms IMU tick)
    and its hard timing (PWM, encoder edges, pixel readout, I2C bits) is
    done by hardware, so the question is how the worst case here compares
    with those deadlines. This measures it three ways:

    - python: a Python thread at normal priority sleeping to a 1 ms grid,
      how late each wake-up is. This is what the robot's own threads (the
      sensor hub, the frame loop) live with. Needs nothing.
    - other: cyclictest at normal priority (SCHED_OTHER): the kernel's own
      wake-up latency for an ordinary thread, without Python.
    - fifo: cyclictest at real-time priority (SCHED_FIFO 80): the best this
      kernel gives an RTOS-style thread.

    cyclictest (apt: rt-tests) needs root; without either, only python
    runs, and the summary says so. Run it twice, idle and while the robot's
    pipeline loads the Pi (make nav-dry in another terminal), with --label,
    and put them side by side with --compare.

Main package:
    python_latency(seconds, interval_us) -> list of us late.
    run_cyclictest(policy, seconds, interval_us) -> histogram, or None.
    parse_cyclictest(text) -> {"hist", "min", "avg", "max", "overflows", "total"}.
    case_stats(hist, overflows, max_us) -> n, min, p50, p99, p99.9, max.
    findings(res), summary_lines(res), compare_lines(a, b), tail(hist),
    figure(res, path): each case's tail, the share of wake-ups at least x late.
    cli(): sudo python3 -m src.diagnostics.sched_latency [--seconds S] [--label L]
        | --compare DIR DIR

Flow:
    1. The kernel's preemption model (uname -v: PREEMPT, PREEMPT_RT).
    2. Each case for --seconds: python, then cyclictest other and fifo.
    3. Histograms to percentiles; against FRAME_BUDGET_MS and the IMU's
       period; findings; summary.txt, sched_latency.json, histogram.csv and
       the figure.
"""
import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from src.params import FPS, RUNS_DIR, SENSOR_RATE_HZ

SECONDS = 30.0              # per case: 30 000 wake-ups at 1 ms; rare spikes need a long run
INTERVAL_US = 1000          # cyclictest's default period
HIST_MAX_US = 10_000        # cyclictest's histogram reaches 10 ms; beyond is an overflow (its max is still exact)
FIFO_PRIORITY = 80
FRAME_BUDGET_US = 1e6 / FPS
IMU_PERIOD_US = 1e6 / SENSOR_RATE_HZ
CASES = ("python", "other", "fifo")
CASE_TEXT = {"python": "Python thread, normal priority (the robot's threads)",
             "other": "cyclictest, normal priority (SCHED_OTHER)",
             "fifo": f"cyclictest, real-time priority (SCHED_FIFO {FIFO_PRIORITY})"}
BUDGET_SHARE = 0.10         # a worst case over this share of a deadline earns a finding
FIFO_GAIN = 2.0             # real-time priority cutting the worst case by this factor earns a finding
# Typical figures for comparison only, not measured here: a microcontroller RTOS
# wakes its highest-priority task within microseconds to tens of microseconds
RTOS_TYPICAL = "single-digit to tens of microseconds"

_CLI_HELP = """\
How late Linux wakes a thread that asked to run at a set time, against the
robot's 50 ms frame and 10 ms IMU deadlines: a Python thread (what the
robot's threads live with), and cyclictest at normal and real-time priority
(needs root and rt-tests). Run it idle, then while the robot runs.

Examples (from vision_stack/):
    sudo $(which python3) -m src.diagnostics.sched_latency --label idle
    make nav-dry                                                      (terminal 1)
    sudo $(which python3) -m src.diagnostics.sched_latency --label loaded   (terminal 2)
    python3 -m src.diagnostics.sched_latency --compare runs/sched_idle_* runs/sched_loaded_*

Output (--out DIR, default <root>/runs/sched_<label>_<timestamp>):
    summary.txt         per case: p50 / p99 / p99.9 / max late, against the deadlines; findings
    sched_latency.json  everything computed
    histogram.csv       wake-ups per microsecond late, per case
    sched_latency.png   each case's tail: the share of wake-ups at least x us late (with matplotlib)
"""


# =============================================================================
# Measuring
# =============================================================================

def python_latency(seconds: float, interval_us: int = INTERVAL_US, clock=time.perf_counter,
                   sleep=time.sleep) -> list[float]:
    """
    A thread sleeping to a fixed grid, as the sensor hub does: how many us
    late each wake-up was. Runs on its own thread, like the robot's.
    """
    out = []

    def run():
        interval = interval_us / 1e6
        start = clock()
        target = start + interval
        while target - start <= seconds:
            delay = target - clock()
            if delay > 0:
                sleep(delay)
            out.append(max(0.0, (clock() - target) * 1e6))
            target += interval
    t = threading.Thread(target=run, name="latency-probe", daemon=True)
    t.start()
    t.join()
    return out


def cyclictest_argv(policy: str, seconds: float, interval_us: int = INTERVAL_US) -> list[str]:
    """
    cyclictest on every core (-S), memory locked (-m), quiet (-q), with a
    1 us histogram to HIST_MAX_US (-h), for seconds at interval_us.
    """
    argv = ["cyclictest", "-m", "-q", "-S", f"-i{interval_us}", f"-D{int(seconds)}s", f"-h{HIST_MAX_US}"]
    if policy == "fifo":
        argv += [f"-p{FIFO_PRIORITY}", "--policy=fifo"]
    else:
        argv += ["-p0", "--policy=other"]
    return argv


def parse_cyclictest(text: str) -> dict | None:
    """
    cyclictest -h's output, summed over its threads: {"hist": {us: count},
    "min", "avg", "max", "overflows", "total"}; None when there's no
    histogram (it failed).
    """
    hist = Counter()
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, sep, rest = line[1:].partition(":")
            if sep:
                nums = [int(x) for x in rest.split() if x.lstrip("-").isdigit()]
                if nums:
                    rows[key.strip()] = nums
            continue
        parts = line.split()
        if parts and all(p.isdigit() for p in parts) and len(parts) >= 2:
            n = sum(int(p) for p in parts[1:])
            if n:
                hist[int(parts[0])] += n
    if not hist and "Total" not in rows:
        return None
    return {"hist": dict(hist), "total": sum(rows.get("Total", [])) or sum(hist.values()),
            "min": min(rows["Min Latencies"]) if "Min Latencies" in rows else None,
            "avg": round(sum(rows["Avg Latencies"]) / len(rows["Avg Latencies"]), 1) if "Avg Latencies" in rows else None,
            "max": max(rows["Max Latencies"]) if "Max Latencies" in rows else None,
            "overflows": sum(rows.get("Histogram Overflows", []))}


def run_cyclictest(policy: str, seconds: float, interval_us: int = INTERVAL_US, run=subprocess.run) -> tuple[dict | None, str]:
    """(parsed histogram or None, why not / the raw output). run is injectable for tests."""
    argv = cyclictest_argv(policy, seconds, interval_us)
    try:
        p = run(argv, capture_output=True, text=True, timeout=seconds + 30)
    except FileNotFoundError:
        return None, "cyclictest isn't installed (sudo apt install rt-tests)"
    except subprocess.TimeoutExpired:
        return None, "cyclictest didn't finish"
    parsed = parse_cyclictest(p.stdout) if p.returncode == 0 else None
    if parsed is None:
        return None, f"cyclictest failed ({p.returncode}): {(p.stderr or p.stdout).strip()[:200]}"
    return parsed, p.stdout


def preemption_model(version: str | None = None) -> str:
    """The kernel's preemption model from uname -v: PREEMPT_RT, PREEMPT, PREEMPT_DYNAMIC, or none."""
    v = platform.version() if version is None else version
    for model in ("PREEMPT_RT", "PREEMPT_DYNAMIC", "PREEMPT"):
        if model in v.split():
            return model
    return "none (voluntary or server)"


# =============================================================================
# Statistics
# =============================================================================

def hist_of(values_us: list[float]) -> dict:
    """Wake-ups per whole microsecond late (rounded down), as cyclictest bins them."""
    return dict(Counter(int(v) for v in values_us))


def case_stats(hist: dict, overflows: int = 0, max_us: float | None = None) -> dict | None:
    """
    n, min, p50, p99, p99.9 and max in us from a histogram. Overflows (past
    the histogram) count as late as the max, so a percentile that lands
    among them reads the max.
    """
    n = sum(hist.values()) + overflows
    if not n:
        return None
    keys = sorted(hist)
    top = max_us if max_us is not None else (keys[-1] if keys else 0)

    def pct(q):
        need, seen = q * n, 0
        for k in keys:
            seen += hist[k]
            if seen >= need:
                return k
        return top
    return {"n": n, "min": keys[0] if keys else top, "p50": pct(0.5), "p99": pct(0.99), "p999": pct(0.999),
            "max": top, "overflows": overflows}


def findings(res: dict) -> list[str]:
    """What the measurement says, in words, against the robot's deadlines."""
    out = []
    cases = res["cases"]
    for name in CASES:
        s = cases.get(name)
        if not s:
            continue
        if s["max"] >= BUDGET_SHARE * IMU_PERIOD_US:
            out.append(f"{name}: the worst wake-up was {s['max'] / 1000:.1f} ms late, "
                       f"{100 * s['max'] / IMU_PERIOD_US:.0f}% of the IMU's {IMU_PERIOD_US / 1000:.0f} ms period and "
                       f"{100 * s['max'] / FRAME_BUDGET_US:.0f}% of the {FRAME_BUDGET_US / 1000:.0f} ms frame"
                       + (": a 100 Hz loop would miss a tick" if s["max"] >= IMU_PERIOD_US else ""))
    py, other, fifo = cases.get("python"), cases.get("other"), cases.get("fifo")
    if other and fifo and fifo["max"] > 0 and other["max"] / fifo["max"] >= FIFO_GAIN:
        out.append(f"real-time priority cut the kernel's worst case {other['max'] / fifo['max']:.0f}x "
                   f"({other['max']} -> {fifo['max']} us); the robot's threads run at normal priority")
    if py and other and other["p99"] > 0 and py["p99"] / other["p99"] >= FIFO_GAIN:
        out.append(f"Python adds to the kernel's wake-up: p99 {py['p99']} us against cyclictest's {other['p99']} us")
    measured = [s for s in cases.values() if s]
    if measured and all(s["max"] < BUDGET_SHARE * IMU_PERIOD_US for s in measured):
        out.append(f"every case stayed under {BUDGET_SHARE * IMU_PERIOD_US / 1000:.0f} ms, a tenth of the tightest "
                   "deadline (the IMU's): Linux's jitter is small against the robot's soft deadlines")
    if res.get("preemption") != "PREEMPT_RT":
        out.append("the kernel isn't PREEMPT_RT: an RT kernel shortens the worst case further, on the same hardware")
    return out


# =============================================================================
# Output
# =============================================================================

def _row(name: str, s: dict | None) -> str:
    if not s:
        return f"  {name:<7} not measured"
    return (f"  {name:<7} {s['n']:>7}  {s['min']:>6}  {s['p50']:>6}  {s['p99']:>6}  {s['p999']:>7}  {s['max']:>7}"
            f"  {100 * s['max'] / FRAME_BUDGET_US:>6.1f}%  {100 * s['max'] / IMU_PERIOD_US:>6.1f}%")


def summary_lines(res: dict) -> list[str]:
    lines = [f"scheduling latency  {res.get('label') or ''}  {res.get('started', '')}",
             f"  kernel {res.get('kernel', '')}, preemption {res.get('preemption', '')}; "
             f"{res.get('seconds', 0):.0f} s per case at a {res.get('interval_us', INTERVAL_US)} us period", "",
             "how late each wake-up was, us (max as a share of the frame budget and the IMU period)",
             f"  {'case':<7} {'wakeups':>7}  {'min':>6}  {'p50':>6}  {'p99':>6}  {'p99.9':>7}  {'max':>7}"
             f"  {'frame':>7}  {'imu':>7}"]
    lines += [_row(c, res["cases"].get(c)) for c in CASES]
    lines += [f"  ({c}: {CASE_TEXT[c]})" for c in CASES]
    lines += [f"  {c}: {why}" for c, why in res.get("skipped", {}).items()]
    lines += ["", f"for scale: a microcontroller RTOS typically wakes its top task within {RTOS_TYPICAL}",
              "", "findings"] + [f"  - {f}" for f in res["findings"] or ["nothing stood out"]]
    return lines


def compare_lines(a: dict, b: dict) -> list[str]:
    """Two runs side by side (idle against loaded): each case's p99 and max."""
    la, lb = a.get("label") or "A", b.get("label") or "B"
    lines = [f"scheduling latency: {la} vs {lb} (us)",
             f"  {'case':<7} {la + ' p99':>12} {lb + ' p99':>12} {la + ' max':>12} {lb + ' max':>12}"]
    for c in CASES:
        sa, sb = a["cases"].get(c), b["cases"].get(c)
        if not sa and not sb:
            continue
        g = lambda s, k: "--" if not s else str(s[k])          # noqa: E731
        lines.append(f"  {c:<7} {g(sa, 'p99'):>12} {g(sb, 'p99'):>12} {g(sa, 'max'):>12} {g(sb, 'max'):>12}")
    return lines


# Categorical slots 1-3 of the validated default palette (dataviz skill), the cases
# in fixed order; the legend names each, so color is never the only identity
CASE_COLORS = {"python": "#2a78d6", "other": "#eb6834", "fifo": "#1baf7a"}


def tail(hist: dict) -> tuple[list[float], list[float]]:
    """
    The share of wake-ups at least x us late, for each x in the histogram:
    the tail (CCDF) a latency plot shows, where p99 crosses 0.01 and p99.9
    0.001. x is at least 1, for a log axis.
    """
    n = sum(hist.values())
    xs, ys, later = [], [], n
    for k in sorted(hist):
        xs.append(max(1, int(k)))
        ys.append(later / n)
        later -= hist[k]
    return xs, ys


def figure(res: dict, path) -> Path | None:
    """
    Each case's tail on log-log axes, the p99 / p99.9 levels and the IMU
    period marked; None without matplotlib or data.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    hists = {c: res["hist"][c] for c in CASES if res["hist"].get(c)}
    if not hists:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    for c, h in hists.items():
        xs, ys = tail(h)
        ax.step(xs, ys, where="post", linewidth=2, color=CASE_COLORS[c], label=f"{c}: {CASE_TEXT[c]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for level, text in ((1e-2, "p99"), (1e-3, "p99.9")):
        ax.axhline(level, color="#d9d6cf", linewidth=1)
        ax.text(ax.get_xlim()[0], level, f" {text}", va="bottom", fontsize=8, color="#6b6b6b")
    ax.axvline(IMU_PERIOD_US, color="#1f1f1f", linewidth=1, linestyle="--")
    ax.set_xlabel(f"us late (dashed: the IMU's {IMU_PERIOD_US / 1000:.0f} ms period)", color="#6b6b6b")
    ax.set_ylabel("share of wake-ups at least this late", color="#6b6b6b")
    ax.set_title(f"Scheduling latency{' (' + res['label'] + ')' if res.get('label') else ''}", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")         # the tails fall away from it
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


def write(out_dir, res: dict, draw: bool = True) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    us = sorted({int(k) for h in res["hist"].values() for k in h})
    with open(out / "histogram.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["us_late", *CASES])
        for u in us:
            w.writerow([u, *(res["hist"].get(c, {}).get(u, 0) for c in CASES)])
    png = figure(res, out / "sched_latency.png") if draw else None
    res = {**res, "figure": png.name if png else None}
    with open(out / "sched_latency.json", "w") as f:
        json.dump({k: v for k, v in res.items() if k != "hist"}, f, indent=2)
    lines = summary_lines(res)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    return lines


def measure(seconds: float, label: str = "", interval_us: int = INTERVAL_US, root: bool | None = None,
            python_fn=python_latency, cyclic_fn=run_cyclictest) -> dict:
    """Every case it can run; the result write() takes."""
    root = (os.geteuid() == 0) if root is None else root
    res = {"label": label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": seconds,
           "interval_us": interval_us, "kernel": platform.release(), "preemption": preemption_model(),
           "cases": {}, "hist": {}, "skipped": {}}
    print(f"  python: {seconds:.0f} s", flush=True)
    py = python_fn(seconds, interval_us)
    res["hist"]["python"] = hist_of(py)
    res["cases"]["python"] = case_stats(res["hist"]["python"], max_us=int(max(py)) if py else None)
    for policy in ("other", "fifo"):
        if not root:
            res["skipped"][policy] = "needs root (sudo)"
            continue
        if not shutil.which("cyclictest") and cyclic_fn is run_cyclictest:
            res["skipped"][policy] = "cyclictest isn't installed (sudo apt install rt-tests)"
            continue
        print(f"  cyclictest {policy}: {seconds:.0f} s", flush=True)
        parsed, why = cyclic_fn(policy, seconds, interval_us)
        if parsed is None:
            res["skipped"][policy] = why
            continue
        res["hist"][policy] = parsed["hist"]
        res["cases"][policy] = case_stats(parsed["hist"], parsed["overflows"], parsed["max"])
    res["findings"] = findings(res)
    return res


def cli(argv: list[str] | None = None) -> int:
    """sudo python3 -m src.diagnostics.sched_latency [--seconds S] [--label L] [--out DIR] | --compare DIR DIR"""
    ap = argparse.ArgumentParser(prog="python3 -m src.diagnostics.sched_latency", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=SECONDS, metavar="S", help=f"per case (default {SECONDS:.0f})")
    ap.add_argument("--label", default="", help="idle, loaded, ... (names the folder)")
    ap.add_argument("--out", default=None, metavar="DIR")
    ap.add_argument("--compare", nargs=2, metavar="DIR", help="two recorded folders side by side")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.compare:
        try:
            a, b = (json.loads((Path(d) / "sched_latency.json").read_text()) for d in args.compare)
        except (OSError, ValueError) as exc:
            print(f"not a sched_latency folder ({exc})")
            return 2
        print("\n".join(compare_lines(a, b)))
        return 0
    name = "sched_" + (f"{args.label}_" if args.label else "") + time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or RUNS_DIR / name)
    print(f"scheduling latency: {args.seconds:.0f} s per case, output {out}", flush=True)
    res = measure(args.seconds, args.label)
    print("\n" + "\n".join(write(out, res, not args.no_figure)))
    from src.diagnostics.i2c_trace import _give_back
    _give_back(out)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
