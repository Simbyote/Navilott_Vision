"""Pi diagnostics recorder: a run's threads, cores and system health, recorded from a separate process.

Purpose:
    The Pi's OS is part of the robot: the kernel decides which core each
    thread runs on, the GIL decides which Python thread runs Python, and
    heat or a weak supply throttles the clock. This records all of it for
    any run (main, a linker, a test) without touching that run's code or
    loop: it is its own process, reading /proc and the Pi's sensors, so it
    shares no GIL and costs the robot only a few file reads per interval.

Main package:
    record(pid, ...) -> {"threads", "cores", "system"} rows until the
        process ends, the duration passes or Ctrl-C.
    summary_lines(...): the per-thread, per-core and system summary.
    find_pid(pattern): the newest process whose command line contains it.
    cli(): launch a command and record it, or attach to a running one.

Flow:
    1. The process: the command after "--" (launched here), --pid, or
       --match (waits for it to start).
    2. Every --interval: per-thread and per-core rates (threads.ThreadSampler);
       every second: temperature, clock, throttling, the run's memory
       (system_monitor.sample).
    3. Ends when the process exits, after --duration, or on Ctrl-C (a
       launched run gets the Ctrl-C too; the recorder waits for it to stop
       its motors and exit).
    4. Writes threads.csv, cores.csv, system.csv, meta.json and summary.txt.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.diagnostics.system_monitor import FIELDS as SYSTEM_FIELDS, THROTTLE_BITS, sample as system_sample
from src.diagnostics.threads import CLK_TCK, CORE_FIELDS, PROC, THREAD_FIELDS, ThreadSampler
from src.params import RUNS_DIR

INTERVAL_S = 0.5            # thread and core sampling; short enough to see a frame-rate pattern, cheap to read
SYSTEM_EVERY_S = 1.0        # vcgencmd is a subprocess: once a second is plenty for heat and clock
MATCH_WAIT_S = 30.0         # --match waits this long for the process to appear
CHILD_EXIT_WAIT_S = 15.0    # after Ctrl-C, how long a launched run gets to halt and exit

_CLI_HELP = """\
Record how a run uses the Pi: every thread's CPU, core and context switches,
every core's load, and temperature, clock, throttling and memory. Runs as its
own process, so the run being watched is unchanged.

Examples (from vision_stack/):
    python3 -m src.diagnostics.monitor -- python3 -m src.navigation_linker --camera --no-motors --no-button
    python3 -m src.diagnostics.monitor -- python3 -m src.main
    python3 -m src.diagnostics.monitor --match src.main          # attach to one started elsewhere
    python3 -m src.diagnostics.monitor --pid 1234 --duration 60

Output (--out DIR, default <root>/runs/diag_<timestamp>):
    summary.txt   per thread: CPU mean / max, the cores it ran on, context switches;
                  per core: load; temperature, clock, throttling and memory
    threads.csv   one row per thread per interval
    cores.csv     one row per core per interval
    system.csv    one row a second
    meta.json     what was recorded, and on what
"""


# =============================================================================
# Finding the process
# =============================================================================

def find_pid(pattern: str, proc: Path = PROC, exclude: tuple[int, ...] = ()) -> int | None:
    """
    The newest process (highest pid) whose command line contains pattern.

    Inputs:
        exclude: pids to skip; this process and its parent are always skipped,
            so the recorder's own command line never matches.
    """
    skip = {os.getpid(), os.getppid(), *exclude}
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in skip:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if pattern in cmdline:
            found.append(int(entry.name))
    return max(found) if found else None


# =============================================================================
# Recording
# =============================================================================

def record(pid: int, interval_s: float = INTERVAL_S, duration_s: float | None = None,
           alive=None, clock=time.perf_counter, sleep=time.sleep, proc: Path = PROC,
           system_reader=None) -> dict:
    """
    Sample pid until it ends, duration_s passes or Ctrl-C.

    Inputs:
        alive: () -> bool, whether to keep going (a launched child's poll());
            None keeps going until the process is gone.
        system_reader: () -> one system sample dict; default reads the Pi,
            with this pid's memory.
    Outputs:
        {"threads": [...], "cores": [...], "system": [...], "interrupted": bool,
        "elapsed_s": float}; rows in THREAD_FIELDS / CORE_FIELDS / SYSTEM_FIELDS order.
    """
    status = proc / str(pid) / "status"
    system_reader = system_reader or (lambda: system_sample(status_path=status))
    sampler = ThreadSampler(pid, proc=proc)
    out = {"threads": [], "cores": [], "system": [], "interrupted": False}
    t0 = prev = clock()
    next_system = t0
    try:
        while True:
            if alive is not None and not alive():
                break
            now = clock()
            elapsed = round(now - t0, 3)
            try:
                threads, cores = sampler.take(elapsed, now - prev)
            except ProcessLookupError:
                break
            prev = now
            out["threads"] += threads
            out["cores"] += cores
            if now >= next_system:
                row = {"elapsed_s": elapsed, **system_reader()}
                out["system"].append({k: row.get(k) for k in SYSTEM_FIELDS})
                next_system += SYSTEM_EVERY_S
            if duration_s is not None and now - t0 >= duration_s:
                break
            sleep(interval_s)
    except KeyboardInterrupt:
        out["interrupted"] = True
    out["elapsed_s"] = round(clock() - t0, 3)
    return out


# =============================================================================
# Summary
# =============================================================================

def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def thread_summary(rows: list[dict]) -> list[dict]:
    """
    One entry per thread (by tid), busiest first.

    Each: name, tid, samples, cpu_mean, cpu_max, core_share {core: fraction
    of samples}, moves (samples on a different core than the one before: a
    floor on migrations, since only the last core is visible), vol_ctx_s and
    invol_ctx_s means.
    """
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r["tid"]].append(r)
    out = []
    for tid, rs in by_tid.items():
        cores = [r["core"] for r in rs]
        share = {c: round(cores.count(c) / len(cores), 2) for c in sorted(set(cores))}
        out.append({"name": rs[-1]["name"], "tid": tid, "samples": len(rs),
                    "cpu_mean": round(_mean([r["cpu_pct"] for r in rs]), 1),
                    "cpu_max": max(r["cpu_pct"] for r in rs),
                    "core_share": share,
                    "moves": sum(a != b for a, b in zip(cores, cores[1:])),
                    "vol_ctx_s": round(_mean([r["vol_ctx_s"] for r in rs]), 1),
                    "invol_ctx_s": round(_mean([r["invol_ctx_s"] for r in rs]), 1)})
    return sorted(out, key=lambda t: (-t["cpu_mean"], t["tid"]))


def system_summary(rows: list[dict]) -> dict:
    """Extremes of the system rows, and every throttle flag seen during the run or latched since boot."""
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]
    return {"temp_max_c": max(col("temp_c"), default=None),
            "cpu_mhz_min": min(col("cpu_mhz"), default=None),
            "cpu_mhz_max": max(col("cpu_mhz"), default=None),
            "rss_max_mb": round(max(col("rss_mb"), default=0.0), 1) if col("rss_mb") else None,
            "mem_available_min_mb": round(min(col("mem_available_mb")), 1) if col("mem_available_mb") else None,
            "throttled_during": [k for k in THROTTLE_BITS if any(r.get(k) for r in rows)],
            "throttled_since_boot": [k for k in THROTTLE_BITS if any(r.get(f"{k}_occurred") for r in rows)],
            "readable": bool(col("throttled_raw"))}


def summary_lines(meta: dict, rec: dict) -> list[str]:
    """summary.txt: the run, each thread, each core, the system."""
    threads = thread_summary(rec["threads"])
    lines = [f"diagnostics  pid {meta['pid']}  {meta['command']}",
             f"  {rec['elapsed_s']:.1f} s, sampled every {meta['interval_s']} s on {meta['cores']} cores"
             + ("  (ended by Ctrl-C)" if rec["interrupted"] else ""),
             "", "threads, busiest first (cpu % of one core; cores = share of samples on each)",
             f"  {'name':<16} {'tid':>7}  {'cpu mean':>8} {'max':>6}  {'cores':<28} {'moves':>5}"
             f"  {'vol/s':>7} {'invol/s':>7}"]
    for t in threads:
        share = " ".join(f"{c}:{round(f * 100)}%" for c, f in t["core_share"].items())
        lines.append(f"  {t['name']:<16} {t['tid']:>7}  {t['cpu_mean']:>8.1f} {t['cpu_max']:>6.1f}  {share:<28}"
                     f" {t['moves']:>5}  {t['vol_ctx_s']:>7.1f} {t['invol_ctx_s']:>7.1f}")
    if not threads:
        lines.append("  (no thread was seen twice: the run ended within one interval)")
    total = round(sum(t["cpu_mean"] for t in threads), 1)
    lines += [f"  total {total:.1f}% of one core ({total / max(meta['cores'], 1):.1f}% of the Pi)", "", "cores"]
    by_core = defaultdict(list)
    for r in rec["cores"]:
        by_core[r["core"]].append(r["busy_pct"])
    for core, xs in sorted(by_core.items()):
        lines.append(f"  core {core}  busy mean {_mean(xs):5.1f}%  max {max(xs):5.1f}%"
                     "  (every process, not only this one)")
    s = system_summary(rec["system"])
    fmt = lambda v, spec, unit: "--" if v is None else f"{v:{spec}}{unit}"      # noqa: E731
    lines += ["", "system",
              f"  temperature max {fmt(s['temp_max_c'], '.1f', ' C')}   clock "
              + (f"{s['cpu_mhz_min']:.0f}-{s['cpu_mhz_max']:.0f} MHz" if s["cpu_mhz_min"] is not None else "--")
              + f"   memory: run max {fmt(s['rss_max_mb'], '.1f', ' MB')},"
              f" available min {fmt(s['mem_available_min_mb'], '.1f', ' MB')}"]
    if not s["readable"]:
        lines.append("  throttling: not readable here (no vcgencmd: not a Pi)")
    else:
        lines.append(f"  throttled during the run: {', '.join(s['throttled_during']) or 'never'}")
        lines.append(f"  latched since boot: {', '.join(s['throttled_since_boot']) or 'nothing'}")
    return lines


def write(out_dir: str, meta: dict, rec: dict) -> list[str]:
    """Write the run folder; returns the summary lines."""
    os.makedirs(out_dir, exist_ok=True)
    for name, fields, rows in (("threads.csv", THREAD_FIELDS, rec["threads"]),
                               ("cores.csv", CORE_FIELDS, rec["cores"]),
                               ("system.csv", SYSTEM_FIELDS, rec["system"])):
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({**meta, "elapsed_s": rec["elapsed_s"], "interrupted": rec["interrupted"]}, f, indent=2)
    lines = summary_lines(meta, rec)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines


# =============================================================================
# Command line
# =============================================================================

def cli(argv: list[str] | None = None) -> int:
    """
    Command line: python3 -m src.diagnostics.monitor [options] (-- COMMAND | --pid N | --match TEXT)

    Outputs:
        A launched command's exit code; otherwise 0, or 2 when there is no
        process to watch.
    """
    argv = sys.argv[1:] if argv is None else argv
    command = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="python3 -m src.diagnostics.monitor", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, default=None, help="watch a running process")
    ap.add_argument("--match", default=None, metavar="TEXT",
                    help=f"watch the newest process whose command line contains TEXT (waits {MATCH_WAIT_S:.0f} s)")
    ap.add_argument("--interval", type=float, default=INTERVAL_S, metavar="S",
                    help=f"thread and core sampling period (default {INTERVAL_S})")
    ap.add_argument("--duration", type=float, default=None, metavar="S", help="stop after S seconds")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(argv)
    if sum(bool(x) for x in (command, args.pid, args.match)) != 1:
        ap.print_help()
        print("\ngive exactly one of: -- COMMAND, --pid N, --match TEXT")
        return 2

    child = None
    if command:
        child = subprocess.Popen(command)
        pid = child.pid
    elif args.pid:
        pid = args.pid
    else:
        deadline = time.monotonic() + MATCH_WAIT_S
        pid = find_pid(args.match)
        while pid is None and time.monotonic() < deadline:
            time.sleep(0.2)
            pid = find_pid(args.match)
    if pid is None or not (PROC / str(pid)).exists():
        print(f"no process to watch ({'--match ' + repr(args.match) if args.match else 'pid ' + str(pid)})")
        return 2

    out_dir = args.out or str(RUNS_DIR / ("diag_" + time.strftime("%Y%m%d_%H%M%S")))
    meta = {"pid": pid, "command": " ".join(command) if command else _cmdline(pid),
            "interval_s": args.interval, "cores": os.cpu_count(), "clk_tck": CLK_TCK,
            "kernel": platform.release(), "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "monotonic_start_s": round(time.monotonic(), 3)}
    print(f"diagnostics: watching pid {pid}, output {out_dir}", flush=True)

    rec = record(pid, args.interval, args.duration, alive=(lambda: child.poll() is None) if child else None)
    code = 0
    if child is not None:
        try:
            code = child.wait(timeout=CHILD_EXIT_WAIT_S)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            child.kill()
            code = child.wait()
    print("\n" + "\n".join(write(out_dir, meta, rec)))
    return code


def _cmdline(pid: int) -> str:
    try:
        return (PROC / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"


if __name__ == "__main__":
    sys.exit(cli())
