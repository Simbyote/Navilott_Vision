"""I2C trace: every transfer on the Pi's I2C buses, from the kernel, as a bus analyzer would show it.

Purpose:
    The robot shares bus 1 (GPIO 2/3) between the MPU-6050 (the sensor
    hub's thread, 100 Hz) and the ADS1115 battery ADC (the battery monitor's
    thread, 1 Hz); the camera's own bus carries libcamera's exposure and
    gain writes to the IMX290 every frame. Python sees only how long its own
    call took. The kernel traces every transfer on every bus (the i2c
    tracepoints: each message's address, direction, length and data, and
    the result), from any process, timestamped to the microsecond. This
    turns that trace into:

    - every transfer: when, which bus and device, which thread asked, the
      messages (w1 r14 = write 1 byte, then read 14), how long it held the
      bus, how long it had to (its wire time: the bits clocked at the bus's
      clock), and the result;
    - per bus: occupancy (the share of time a transfer was in progress)
      against wire occupancy (the share the bits alone need), and the
      overhead between them (driver, interrupts, the controller's FIFO);
    - per device: transfers and bytes per second, durations, errors, the
      gaps between transfers (the IMU's should be a steady 10 ms), and how
      many started right after another device's transfer (they queued
      behind it on the bus).

    Tracing needs root (tracefs); the recording is read back afterwards, so
    the robot's run is untouched. --from re-analyzes a recorded folder
    anywhere, without root (and draws the figure where matplotlib is).

Main package:
    record_trace(tracefs, seconds) -> (text, window_s, lost, clock): turn
        the i2c events on, wait, read them, put tracefs back as it was.
    parse_events(text), assemble(events) -> transfers.
    bus_info(): each adapter's name and clock (device tree).
    analyze(transfers, window_s, buses, lost) -> dict: per bus, per device,
        findings.
    summary_lines(res), figure(transfers, res, path).
    cli(): sudo python3 -m src.diagnostics.i2c_trace [--seconds S] [--out DIR]
        | python3 -m src.diagnostics.i2c_trace --from DIR

Flow:
    1. Root and tracefs checked; the trace buffer enlarged, the clock set
       to mono (so times line up with the runs' t0_monotonic), the buffer
       cleared, the four i2c events turned on for --seconds.
    2. Read back; every setting restored.
    3. Events grouped into transfers per bus (a bus is locked for a whole
       transfer, so its events never interleave); stats, findings, files.
"""
import argparse
import csv
import errno
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.params import ADS1115_I2C_ADDRESS, IMU_I2C_ADDRESS, RUNS_DIR, SENSOR_RATE_HZ

SECONDS = 10.0
TRACEFS = (Path("/sys/kernel/tracing"), Path("/sys/kernel/debug/tracing"))
ADAPTERS = Path("/sys/class/i2c-adapter")
EVENTS = ("i2c_write", "i2c_read", "i2c_reply", "i2c_result")
BUFFER_KB = 8192            # per CPU; a transfer is ~4 lines, ~400 bytes: minutes of the IMU at 100 Hz
DEFAULT_CLOCK_HZ = 100_000  # the Pi's i2c_arm_baudrate when config.txt doesn't set it
KNOWN = {IMU_I2C_ADDRESS: "MPU-6050 IMU", ADS1115_I2C_ADDRESS: "ADS1115 battery", 0x1A: "IMX290 camera"}
IMU_PERIOD_MS = 1000.0 / SENSOR_RATE_HZ
BACK_TO_BACK_US = 100.0     # a transfer starting this soon after another device's ended had queued behind it
BUS_BUSY_PCT = 50.0         # past this occupancy a bus starts delaying its devices
GAP_SLIP = 1.5              # IMU gaps over this x its period at p95: the sensor hub's ticks slip
OVERHEAD_HIGH = 2.0         # transfers taking this x their wire time: overhead dominates
TRANSFER_FIELDS = ("start_s", "bus", "addr", "device", "task", "pid", "msgs", "write_bytes", "read_bytes",
                   "wire_us", "dur_us", "ret", "ok", "wdata", "rdata")
ZOOM_S = 0.2                # the figure's transfer timeline window

_CLI_HELP = """\
Trace every I2C transfer on the Pi's buses from the kernel for a few seconds:
which device, which thread, how long each held the bus against the time its
bits need, errors, the gaps between reads, and how busy each bus was. Start
the run in another terminal first (make nav-dry), then trace it.

Examples (from vision_stack/):
    sudo $(which python3) -m src.diagnostics.i2c_trace               10 s
    sudo $(which python3) -m src.diagnostics.i2c_trace --seconds 30
    python3 -m src.diagnostics.i2c_trace --from runs/i2c_20261007_150000   re-analyze, no root

Output (--out DIR, default <root>/runs/i2c_<timestamp>):
    summary.txt        per bus and per device, findings
    transactions.csv   one row per transfer
    i2c_trace.json     everything computed, and the buses' clocks
    trace.txt          the kernel's trace, as recorded
    i2c_trace.png      transfers on a timeline, occupancy per second, IMU gaps (with matplotlib)
"""


# =============================================================================
# Parsing
# =============================================================================

# "   sensor-hub-1234    [002] d..1.  1234.567890: i2c_write: i2c-1 #0 a=068 f=0000 l=1 [3b]"
_LINE = re.compile(r"^\s*(?P<task>.+?)-(?P<pid>\d+)\s+(?:\(\s*[\d-]+\)\s+)?\[(?P<cpu>\d+)\]\s+(?:\S+\s+)?"
                   r"(?P<ts>\d+\.\d+):\s+(?P<event>i2c_\w+):\s+i2c-(?P<bus>\d+)\s+(?P<rest>.*)$")
_MSG = re.compile(r"#(?P<idx>\d+)\s+a=(?P<addr>[0-9a-fA-F]+)\s+f=(?P<flags>[0-9a-fA-F]+)\s+l=(?P<len>\d+)"
                  r"(?:\s+\[(?P<data>[^\]]*)\])?")
_RESULT = re.compile(r"n=(?P<n>\d+)\s+ret=(?P<ret>-?\d+)")


def parse_events(text: str) -> list[dict]:
    """The i2c events of a tracefs trace, in order: {ts, event, bus, task, pid, and the event's fields}."""
    out = []
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m or m.group("event") not in EVENTS:
            continue
        e = {"ts": float(m.group("ts")), "event": m.group("event"), "bus": int(m.group("bus")),
             "task": m.group("task").strip(), "pid": int(m.group("pid"))}
        rest = m.group("rest")
        if e["event"] == "i2c_result":
            r = _RESULT.search(rest)
            if not r:
                continue
            e.update(n=int(r.group("n")), ret=int(r.group("ret")))
        else:
            r = _MSG.search(rest)
            if not r:
                continue
            e.update(idx=int(r.group("idx")), addr=int(r.group("addr"), 16), flags=int(r.group("flags"), 16),
                     len=int(r.group("len")), data=(r.group("data") or "").replace("-", " ").split())
        out.append(e)
    return out


def wire_bits(msgs: list[tuple[str, int]]) -> int:
    """
    Bits a transfer clocks: per message a start (or repeated start), the
    address byte and each data byte, each 9 bits with its ACK; one stop.
    """
    return sum(1 + 9 + 9 * n for _, n in msgs) + 1


def device_name(addr: int) -> str:
    return KNOWN.get(addr, f"0x{addr:02x}")


def assemble(events: list[dict], clocks: dict | None = None) -> list[dict]:
    """
    Transfers from the events: per bus, messages (#0, #1...) then replies,
    then the result closing it. A transfer cut off by the trace's start or
    end is left out.

    Inputs:
        clocks: {bus: clock Hz} for the wire time; DEFAULT_CLOCK_HZ otherwise.
    """
    clocks = clocks or {}
    open_ = {}
    out = []
    for e in events:
        bus = e["bus"]
        if e["event"] in ("i2c_write", "i2c_read"):
            cur = open_.get(bus)
            if cur is None or e["idx"] == 0:
                cur = open_[bus] = {"start": e["ts"], "bus": bus, "task": e["task"], "pid": e["pid"],
                                    "addr": e["addr"], "msgs": [], "wdata": [], "rdata": []}
            read = e["event"] == "i2c_read"          # the kernel's i2c_read is exactly the I2C_M_RD messages
            cur["msgs"].append(("r" if read else "w", e["len"]))
            if not read:
                cur["wdata"] += e["data"]
        elif e["event"] == "i2c_reply":
            if bus in open_:
                open_[bus]["rdata"] += e["data"]
        else:
            cur = open_.pop(bus, None)
            if cur is None:
                continue
            bits = wire_bits(cur["msgs"])
            out.append({
                "start_s": cur["start"], "bus": bus, "addr": cur["addr"], "device": device_name(cur["addr"]),
                "task": cur["task"], "pid": cur["pid"], "msgs": " ".join(f"{d}{n}" for d, n in cur["msgs"]),
                "write_bytes": sum(n for d, n in cur["msgs"] if d == "w"),
                "read_bytes": sum(n for d, n in cur["msgs"] if d == "r"),
                "wire_us": round(bits / clocks.get(bus, DEFAULT_CLOCK_HZ) * 1e6, 1),
                "dur_us": round((e["ts"] - cur["start"]) * 1e6, 1),
                "ret": e["ret"], "ok": int(e["ret"] == e["n"]),
                "wdata": "".join(cur["wdata"][:8]), "rdata": "".join(cur["rdata"][:16])})
    return out


# =============================================================================
# The system: buses and tracefs
# =============================================================================

def bus_info(adapters: Path = ADAPTERS) -> dict:
    """
    {bus: {"name", "clock_hz", "clock_from"}} for every adapter. The clock
    is the device tree's clock-frequency (a big-endian u32), from the
    adapter's node or, for a mux's child bus (the camera's), its parent's;
    otherwise DEFAULT_CLOCK_HZ, marked "assumed".
    """
    out = {}
    if not adapters.is_dir():
        return out
    for d in sorted(adapters.glob("i2c-*")):
        try:
            bus = int(d.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        name = (d / "name").read_text().strip() if (d / "name").is_file() else ""
        clock, src = None, "assumed"
        for node in (d / "of_node", d / ".." / "of_node"):
            f = node / "clock-frequency"
            try:
                raw = f.read_bytes()
            except OSError:
                continue
            if len(raw) >= 4:
                clock, src = int.from_bytes(raw[:4], "big"), "device tree"
                break
        out[bus] = {"name": name, "clock_hz": clock or DEFAULT_CLOCK_HZ, "clock_from": src}
    return out


def find_tracefs(candidates=TRACEFS) -> Path | None:
    """The first tracefs that has the i2c events."""
    for p in candidates:
        if (p / "events" / "i2c").is_dir():
            return p
    return None


def _read(p: Path) -> str:
    return p.read_text().strip()


def _write(p: Path, value: str) -> None:
    with open(p, "w") as f:
        f.write(value)


def _selected(text: str) -> str:
    """The [bracketed] choice in a tracefs option list such as trace_clock."""
    m = re.search(r"\[(\S+)\]", text)
    return m.group(1) if m else text.split()[0] if text.split() else ""


def record_trace(tracefs: Path, seconds: float, sleep=time.sleep, clock=time.monotonic) -> tuple[str, float, int, str]:
    """
    The i2c events for seconds: (trace text, seconds actually traced,
    events lost to a full buffer, the trace clock used). Every setting it
    changes is restored, whatever happens.
    """
    saved = {k: _read(tracefs / k) for k in ("tracing_on", "buffer_size_kb", "trace_clock")}
    clock_used = _selected(saved["trace_clock"])
    try:
        _write(tracefs / "tracing_on", "0")
        _write(tracefs / "buffer_size_kb", str(BUFFER_KB))
        if "mono" in saved["trace_clock"].replace("[", " ").replace("]", " ").split():
            _write(tracefs / "trace_clock", "mono")
            clock_used = "mono"
        _write(tracefs / "trace", "")
        for ev in EVENTS:
            _write(tracefs / "events" / "i2c" / ev / "enable", "1")
        _write(tracefs / "tracing_on", "1")
        t0 = clock()
        sleep(seconds)
        window = clock() - t0
        _write(tracefs / "tracing_on", "0")
        text = (tracefs / "trace").read_text(errors="replace")
        lost = 0
        for stats in (tracefs / "per_cpu").glob("cpu*/stats"):
            m = re.search(r"^overrun:\s*(\d+)", stats.read_text(), re.M)
            lost += int(m.group(1)) if m else 0
    finally:
        for ev in EVENTS:
            try:
                _write(tracefs / "events" / "i2c" / ev / "enable", "0")
            except OSError:
                pass
        _write(tracefs / "trace_clock", _selected(saved["trace_clock"]))
        _write(tracefs / "buffer_size_kb", saved["buffer_size_kb"])
        _write(tracefs / "tracing_on", saved["tracing_on"])
    return text, window, lost, clock_used


# =============================================================================
# Analysis
# =============================================================================

def _q(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def _dist(xs: list[float], digits: int = 1) -> dict | None:
    if not xs:
        return None
    return {"median": round(statistics.median(xs), digits), "p95": round(_q(xs, 0.95), digits),
            "max": round(max(xs), digits)}


def _errname(ret: int) -> str:
    return errno.errorcode.get(-ret, str(ret)) if ret < 0 else f"{ret} of the messages"


def analyze(transfers: list[dict], window_s: float, buses: dict | None = None, lost: int = 0) -> dict:
    """Per bus and per device statistics, and findings; see the module docstring."""
    buses = buses or {}
    window_s = max(window_s, 1e-9)
    by_bus, by_dev = defaultdict(list), defaultdict(list)
    for t in sorted(transfers, key=lambda t: t["start_s"]):
        by_bus[t["bus"]].append(t)
        by_dev[(t["bus"], t["addr"])].append(t)

    queued = Counter()                      # (bus, addr): transfers that started right after another device's
    for ts in by_bus.values():
        for prev, cur in zip(ts, ts[1:]):
            gap_us = (cur["start_s"] - prev["start_s"]) * 1e6 - prev["dur_us"]
            if prev["addr"] != cur["addr"] and gap_us <= BACK_TO_BACK_US:
                queued[(cur["bus"], cur["addr"])] += 1

    res = {"window_s": round(window_s, 3), "lost": lost, "transfers": len(transfers), "buses": [], "devices": []}
    for bus, ts in sorted(by_bus.items()):
        info = buses.get(bus, {"name": "", "clock_hz": DEFAULT_CLOCK_HZ, "clock_from": "assumed"})
        busy, wire = sum(t["dur_us"] for t in ts), sum(t["wire_us"] for t in ts)
        res["buses"].append({"bus": bus, "name": info["name"], "clock_hz": info["clock_hz"],
                             "clock_from": info["clock_from"], "transfers": len(ts),
                             "occupancy_pct": round(100 * busy / 1e6 / window_s, 2),
                             "wire_pct": round(100 * wire / 1e6 / window_s, 2),
                             "overhead": round(busy / wire, 2) if wire else None,
                             "errors": sum(1 for t in ts if not t["ok"])})
    for (bus, addr), ts in sorted(by_dev.items()):
        starts = [t["start_s"] for t in ts]
        gaps = [(b - a) * 1000 for a, b in zip(starts, starts[1:])]
        errors = Counter(t["ret"] for t in ts if not t["ok"])
        res["devices"].append({
            "bus": bus, "addr": addr, "device": device_name(addr), "transfers": len(ts),
            "rate_hz": round(len(ts) / window_s, 1),
            "bytes_s": round(sum(t["write_bytes"] + t["read_bytes"] for t in ts) / window_s, 1),
            "pattern": Counter(t["msgs"] for t in ts).most_common(1)[0][0],
            "tasks": sorted({t["task"] for t in ts}),
            "dur_us": _dist([t["dur_us"] for t in ts]), "wire_us": _dist([t["wire_us"] for t in ts]),
            "gap_ms": _dist(gaps, 2), "errors": {_errname(r): n for r, n in errors.items()},
            "queued": queued[(bus, addr)]})
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    """What the trace says, in words."""
    out = []
    if not res["transfers"]:
        return ["no I2C transfer in the trace: was a run going (make nav-dry in another terminal)?"]
    if res["lost"]:
        out.append(f"the trace buffer overflowed: {res['lost']} events lost; trace fewer seconds")
    for b in res["buses"]:
        if b["occupancy_pct"] >= BUS_BUSY_PCT:
            out.append(f"i2c-{b['bus']} was busy {b['occupancy_pct']:.0f}% of the time: its devices start waiting "
                       "on each other")
        if b["overhead"] is not None and b["overhead"] >= OVERHEAD_HIGH:
            out.append(f"transfers on i2c-{b['bus']} hold the bus {b['overhead']:.1f}x their wire time: the "
                       "driver's and interrupts' share dominates, so a faster clock shortens only the wire part")
    for d in res["devices"]:
        for name, n in d["errors"].items():
            out.append(f"{n} failed transfers to {d['device']} (i2c-{d['bus']}): {name}"
                       + (" (no ACK: wiring, address or the device busy)" if name == "EREMOTEIO" else ""))
        if d["addr"] == IMU_I2C_ADDRESS and d["gap_ms"] and d["gap_ms"]["p95"] > GAP_SLIP * IMU_PERIOD_MS:
            out.append(f"IMU reads were spaced up to {d['gap_ms']['p95']:.1f} ms (p95), not {IMU_PERIOD_MS:.0f}: "
                       "the sensor hub's ticks slip (the GIL, or a slow read)")
        if d["addr"] == IMU_I2C_ADDRESS and d["queued"]:
            out.append(f"{d['queued']} IMU reads started right after another device's transfer: they queued "
                       "behind it on the bus")
    bus1 = next((b for b in res["buses"] if b["bus"] == 1), None)
    if bus1 and bus1["clock_hz"] <= 100_000 and bus1["wire_pct"] >= 10:
        out.append(f"i2c-1 runs at {bus1['clock_hz'] / 1000:.0f} kHz ({bus1['clock_from']}) and its bits alone "
                   f"take {bus1['wire_pct']:.0f}% of the time: dtparam=i2c_arm_baudrate=400000 in "
                   "/boot/firmware/config.txt (both devices are rated for it) cuts that 4x")
    return out


def summary_lines(res: dict, meta: dict | None = None) -> list[str]:
    """summary.txt."""
    meta = meta or {}
    d_ = lambda d, k: "--" if d is None else f"{d[k]:.1f}"          # noqa: E731
    lines = [f"i2c trace  {meta.get('started', '')}  {res['window_s']:.1f} s, {res['transfers']} transfers"
             + (f"  (trace clock {meta['trace_clock']})" if meta.get("trace_clock") else ""), "",
             "buses (occupancy: a transfer in progress; wire: the bits alone at the bus clock)"]
    for b in res["buses"]:
        lines.append(f"  i2c-{b['bus']:<3} {b['name'][:28]:<28} {b['clock_hz'] / 1000:>5.0f} kHz ({b['clock_from']})  "
                     f"occupancy {b['occupancy_pct']:5.1f}%  wire {b['wire_pct']:5.1f}%  overhead "
                     + ("--" if b["overhead"] is None else f"{b['overhead']:.1f}x")
                     + f"  errors {b['errors']}")
    if not res["buses"]:
        lines.append("  none")
    lines += ["", "devices (durations in us; gaps between transfers in ms: median / p95 / max)"]
    for d in res["devices"]:
        lines += [f"  {d['device']} (i2c-{d['bus']} 0x{d['addr']:02x})  by {', '.join(d['tasks'])}",
                  f"    {d['transfers']} transfers, {d['rate_hz']:.1f}/s, {d['bytes_s']:.0f} B/s, mostly {d['pattern']}",
                  f"    duration {d_(d['dur_us'], 'median')} / {d_(d['dur_us'], 'p95')} / {d_(d['dur_us'], 'max')}   "
                  f"wire {d_(d['wire_us'], 'median')}   gap " + ("--" if d["gap_ms"] is None else
                  f"{d['gap_ms']['median']:.2f} / {d['gap_ms']['p95']:.2f} / {d['gap_ms']['max']:.2f}"),
                  f"    errors: {', '.join(f'{k} x{n}' for k, n in d['errors'].items()) or 'none'}   "
                  f"queued behind another device: {d['queued']}"]
    lines += ["", "findings"] + [f"  - {f}" for f in res["findings"] or ["nothing stood out"]]
    return lines


# =============================================================================
# Figure
# =============================================================================

# Categorical slots 1-4 of the validated default palette (dataviz skill), as
# pi_load's cores; the lanes are labeled, so color is never the only identity
DEVICE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d6cf"


def figure(transfers: list[dict], res: dict, path) -> Path | None:
    """
    i2c_trace.png: the transfers of a ZOOM_S window on one lane per device,
    each bus's occupancy per second, and the IMU's gaps. None without
    matplotlib (the Pi's venv): re-run --from on a laptop.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not transfers:
        return None
    devs = [(d["bus"], d["addr"]) for d in sorted(res["devices"], key=lambda d: -d["transfers"])]   # busiest first
    color = {k: DEVICE_COLORS[i % len(DEVICE_COLORS)] for i, k in enumerate(devs)}
    t0 = min(t["start_s"] for t in transfers)
    # the window: around the first transfer of a device other than the busiest, to show sharing
    busiest = max(res["devices"], key=lambda d: d["transfers"])
    other = [t for t in transfers if (t["bus"], t["addr"]) != (busiest["bus"], busiest["addr"])]
    w0 = max(t0, (min(t["start_s"] for t in other) - ZOOM_S / 2) if other else t0)
    imu = next((d for d in res["devices"] if d["addr"] == IMU_I2C_ADDRESS), None)
    fig, axes = plt.subplots(3 if imu else 2, 1, figsize=(11, 8.5 if imu else 6), facecolor="white")
    ax = axes[0]
    for i, k in enumerate(devs):
        bars = [((t["start_s"] - w0) * 1000, max(t["dur_us"] / 1000, 0.05)) for t in transfers
                if (t["bus"], t["addr"]) == k and w0 <= t["start_s"] <= w0 + ZOOM_S]
        ax.broken_barh(bars, (i - 0.3, 0.6), facecolors=color[k], edgecolor="white", linewidth=0.5)
        bad = [(t["start_s"] - w0) * 1000 for t in transfers if (t["bus"], t["addr"]) == k and not t["ok"]
               and w0 <= t["start_s"] <= w0 + ZOOM_S]
        if bad:
            ax.plot(bad, [i] * len(bad), "x", color=INK, markersize=8, label="failed" if i == 0 else None)
    ax.set_yticks(range(len(devs)), [f"{device_name(a)}\ni2c-{b} 0x{a:02x}" for b, a in devs], fontsize=9)
    ax.invert_yaxis()                                                   # the busiest device on top
    ax.set_xlim(0, ZOOM_S * 1000)
    ax.set_xlabel(f"ms (a {ZOOM_S * 1000:.0f} ms window; bar length = time the transfer held the bus)", color=MUTED)
    ax.set_title("Transfers on the bus", loc="left", fontsize=11, color=INK)
    ax = axes[1]
    # buses in neutral ink, told apart by line style: the device colors stay the devices'
    for i, bus in enumerate(sorted({t["bus"] for t in transfers})):
        sec = defaultdict(float)
        for t in transfers:
            if t["bus"] == bus:
                sec[int(t["start_s"] - t0)] += t["dur_us"] / 1e4        # % of that second
        xs = sorted(sec)
        ax.plot(xs, [sec[x] for x in xs], linewidth=2, label=f"i2c-{bus}", color=(INK, MUTED)[i % 2],
                linestyle=("-", "--", ":")[i % 3])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of each second busy", color=MUTED)
    ax.set_xlabel("s", color=MUTED)
    ax.set_title("Occupancy per bus", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    if imu:
        ax = axes[2]
        starts = sorted(t["start_s"] for t in transfers if t["addr"] == IMU_I2C_ADDRESS and t["bus"] == imu["bus"])
        gaps = [(b - a) * 1000 for a, b in zip(starts, starts[1:])]
        # half-ms bins centered on whole and half ms, from 0: a steady 10 ms is one bar on the period line
        hi = max(2 * IMU_PERIOD_MS, max(gaps, default=0) * 1.05)
        ax.hist(gaps, bins=[k * 0.5 - 0.25 for k in range(int(hi / 0.5) + 2)],
                color=color[(imu["bus"], imu["addr"])], edgecolor="white")
        ax.axvline(IMU_PERIOD_MS, color=INK, linewidth=1, linestyle="--")
        ax.set_xlabel("ms between IMU reads", color=MUTED)
        ax.set_title(f"IMU read spacing (dashed: its {IMU_PERIOD_MS:.0f} ms period)", loc="left", fontsize=11,
                     color=INK)
    for a in axes:
        a.grid(axis="x", color=GRID, linewidth=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


# =============================================================================
# Writing and the command line
# =============================================================================

def write(out_dir, trace_text: str, window_s: float, lost: int, buses: dict, meta: dict,
          draw: bool = True) -> dict:
    """Parse, analyze and write the folder; returns the analysis."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.txt").write_text(trace_text)
    transfers = assemble(parse_events(trace_text), {b: i["clock_hz"] for b, i in buses.items()})
    res = analyze(transfers, window_s, buses, lost)
    with open(out / "transactions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, TRANSFER_FIELDS)
        w.writeheader()
        w.writerows(transfers)
    png = figure(transfers, res, out / "i2c_trace.png") if draw else None
    meta = {**meta, "window_s": round(window_s, 3), "lost": lost,
            "buses": {str(k): v for k, v in buses.items()}, "figure": png.name if png else None}
    with open(out / "i2c_trace.json", "w") as f:
        json.dump({"meta": meta, **res}, f, indent=2)
    lines = summary_lines(res, meta)
    if draw and png is None and transfers:
        lines.append("  (no matplotlib here: the figure was skipped; re-run with --from on a laptop)")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    return res


def _give_back(path: Path) -> None:
    """Under sudo, hand the folder back to the user who ran it."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if not uid or not gid:
        return
    for p in [path, *path.rglob("*")]:
        try:
            os.chown(p, int(uid), int(gid))
        except OSError:
            pass


def cli(argv: list[str] | None = None) -> int:
    """sudo python3 -m src.diagnostics.i2c_trace [--seconds S] [--out DIR] | --from DIR"""
    ap = argparse.ArgumentParser(prog="python3 -m src.diagnostics.i2c_trace", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=SECONDS, metavar="S", help=f"how long (default {SECONDS:.0f})")
    ap.add_argument("--out", default=None, metavar="DIR")
    ap.add_argument("--from", dest="from_dir", default=None, metavar="DIR",
                    help="re-analyze a recorded folder (its trace.txt and i2c_trace.json); no root needed")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.from_dir:
        src = Path(args.from_dir)
        try:
            text = (src / "trace.txt").read_text(errors="replace")
            meta = json.loads((src / "i2c_trace.json").read_text())["meta"]
        except (OSError, ValueError, KeyError) as exc:
            print(f"not an i2c trace folder: {src} ({exc})")
            return 2
        buses = {int(k): v for k, v in meta.get("buses", {}).items()}
        res = write(args.out or src, text, meta["window_s"], meta.get("lost", 0), buses, meta, not args.no_figure)
        print("\n".join(summary_lines(res, meta)))
        return 0

    if os.geteuid() != 0:
        print("tracing needs root: sudo $(which python3) -m src.diagnostics.i2c_trace (or: make i2c-trace)")
        return 2
    tracefs = find_tracefs()
    if tracefs is None:
        print("no tracefs with i2c events: mount it (sudo mount -t tracefs nodev /sys/kernel/tracing) "
              "or the kernel lacks I2C tracing")
        return 2
    out = Path(args.out or RUNS_DIR / ("i2c_" + time.strftime("%Y%m%d_%H%M%S")))
    print(f"i2c trace: {args.seconds:.0f} s from {tracefs}, output {out}", flush=True)
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": args.seconds, "tracefs": str(tracefs),
            "monotonic_start_s": round(time.monotonic(), 3)}
    text, window, lost, trace_clock = record_trace(tracefs, args.seconds)
    meta["trace_clock"] = trace_clock
    res = write(out, text, window, lost, bus_info(), meta, not args.no_figure)
    _give_back(out)
    print("\n" + (out / "summary.txt").read_text())
    return 0 if res["transfers"] else 1


if __name__ == "__main__":
    sys.exit(cli())
