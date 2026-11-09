"""OS counters: what the rest of the Pi does while the robot runs.

Purpose:
    Some of the robot's work happens outside its threads: camera interrupts,
    the ISP, I2C, pigpiod's PWM, the SD card. All of it is in /proc and
    vcgencmd as counters, read once a second for monitor.py; every source is
    optional.

Main package:
    OsCounters.take(): the system fields, and per-IRQ and per-process rates since the last take.
"""
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

PROC = Path("/proc")
DISK_DEVICE = "mmcblk0"     # the SD card
CLOCK_EVERY_S = 5.0         # each read is a subprocess, and the clocks rarely move
SECTOR_BYTES = 512          # /proc/diskstats' unit, whatever the device's
PSI_KINDS = ("cpu", "io", "memory")
SYSTEM_FIELDS = ("os_dt_s", "cma_free_mb", "isp_mhz", "core_mhz", "disk_read_kbps", "disk_write_kbps",
                 "disk_busy_pct", "psi_cpu_some", "psi_io_some", "psi_memory_some")
IRQ_FIELDS = ("elapsed_s", "irq", "name", "rate_hz")
PROC_FIELDS = ("elapsed_s", "pid", "name", "cpu_pct")
SELF_NAME = "monitor (self)"
try:
    CLK_TCK = os.sysconf("SC_CLK_TCK")
except (ValueError, OSError, AttributeError):        # not POSIX
    CLK_TCK = 100


# "Edge" / "Level" on the Pi, "5-edge" on x86; a chip name like ARMCTRL-level isn't one
_TRIGGER = re.compile(r"(?:\d+-)?(?:[Ee]dge|[Ll]evel)")


def _irq_name(desc: str) -> str:
    """What follows the trigger type (Edge / Level), else the whole text."""
    tokens = desc.split()
    marks = [i for i, t in enumerate(tokens) if _TRIGGER.fullmatch(t)]
    if marks and marks[-1] + 1 < len(tokens):
        return " ".join(tokens[marks[-1] + 1:])
    return desc


def read_interrupts(path: Path = PROC / "interrupts") -> dict:
    """{irq: (device, count over all cores)}; {} when unreadable."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    if not lines:
        return {}
    ncpu = len(lines[0].split())
    out = {}
    for line in lines[1:]:
        irq, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.split()
        counts = []
        for p in parts[:ncpu]:
            if not p.isdigit():
                break
            counts.append(int(p))
        if counts:
            out[irq.strip()] = (_irq_name(" ".join(parts[len(counts):])), sum(counts))
    return out


def read_disk(path: Path = PROC / "diskstats", device: str = DISK_DEVICE) -> dict | None:
    """Cumulative sectors read and written and ms doing I/O; None when absent."""
    try:
        for line in path.read_text().splitlines():
            f = line.split()
            if len(f) >= 13 and f[2] == device:
                return {"sectors_read": int(f[5]), "sectors_written": int(f[9]), "io_ms": int(f[12])}
    except (OSError, ValueError):
        pass
    return None


def read_pressure(path: Path) -> float | None:
    """"some avg10": % of the last 10 s some task stalled; None without PSI."""
    try:
        m = re.search(r"^some avg10=([0-9.]+)", path.read_text(), re.M)
    except OSError:
        return None
    return float(m.group(1)) if m else None


def read_cma_mb(path: Path = PROC / "meminfo") -> dict:
    out = {"cma_total_mb": None, "cma_free_mb": None}
    try:
        text = path.read_text()
    except OSError:
        return out
    for key, name in (("CmaTotal", "cma_total_mb"), ("CmaFree", "cma_free_mb")):
        m = re.search(rf"^{key}:\s+(\d+)\s*kB", text, re.M)
        if m:
            out[name] = round(int(m.group(1)) / 1024, 1)
    return out


_VC_NUMBER = re.compile(r"=\s*([0-9.]+)")


def parse_vcgencmd(text: str) -> float | None:
    """The number after '=': frequency(45)=500000000, volt=1.2000V, gpu=64M."""
    m = _VC_NUMBER.search(text or "")
    return float(m.group(1)) if m else None


def read_vc_clock_mhz(clock: str, run=subprocess.run) -> float | None:
    try:
        out = run(["vcgencmd", "measure_clock", clock], capture_output=True, text=True, timeout=1)
    except (OSError, subprocess.SubprocessError):
        return None
    hz = parse_vcgencmd(out.stdout) if out.returncode == 0 else None
    return None if hz is None else round(hz / 1e6, 1)


def read_process_ticks(proc: Path = PROC, exclude: tuple[int, ...] = ()) -> dict:
    """{pid: (name, user + system ticks)}."""
    out = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in exclude:
            continue
        try:
            text = (entry / "stat").read_text()
        except OSError:
            continue
        lpar, rpar = text.find("("), text.rfind(")")
        fields = text[rpar + 2:].split()
        try:
            out[int(entry.name)] = (text[lpar + 1:rpar], int(fields[11]) + int(fields[12]))
        except (IndexError, ValueError):
            continue
    return out


class OsCounters:
    """target_pid is left out of procs: monitor.py records its threads."""
    def __init__(self, target_pid: int | None = None, proc: Path = PROC, clock_reader=read_vc_clock_mhz,
                 disk_device: str = DISK_DEVICE, self_pid: int | None = None):
        self.target_pid, self.proc, self.clock_reader, self.disk_device = target_pid, proc, clock_reader, disk_device
        self.self_pid = os.getpid() if self_pid is None else self_pid
        self._prev = None               # (now, irqs, disk, ticks)
        self._clock_at = None

    def take(self, elapsed_s: float, now: float) -> dict:
        irqs = read_interrupts(self.proc / "interrupts")
        disk = read_disk(self.proc / "diskstats", self.disk_device)
        ticks = read_process_ticks(self.proc, exclude=() if self.target_pid is None else (self.target_pid,))
        system = {k: None for k in SYSTEM_FIELDS}
        system["cma_free_mb"] = read_cma_mb(self.proc / "meminfo")["cma_free_mb"]
        for kind in PSI_KINDS:
            system[f"psi_{kind}_some"] = read_pressure(self.proc / "pressure" / kind)
        if self._clock_at is None or now - self._clock_at >= CLOCK_EVERY_S - 1e-9:
            self._clock_at = now
            system["isp_mhz"], system["core_mhz"] = self.clock_reader("isp"), self.clock_reader("core")
        out = {"system": system, "irqs": [], "procs": []}
        if self._prev is not None and now > self._prev[0]:
            t0, irqs0, disk0, ticks0 = self._prev
            dt = now - t0
            system["os_dt_s"] = round(dt, 3)
            for irq, (name, n) in irqs.items():
                d = n - irqs0.get(irq, (name, n))[1]
                if d > 0:
                    out["irqs"].append({"elapsed_s": elapsed_s, "irq": irq, "name": name, "rate_hz": round(d / dt, 1)})
            if disk and disk0:
                kb = SECTOR_BYTES / 1024
                system["disk_read_kbps"] = round((disk["sectors_read"] - disk0["sectors_read"]) * kb / dt, 1)
                system["disk_write_kbps"] = round((disk["sectors_written"] - disk0["sectors_written"]) * kb / dt, 1)
                system["disk_busy_pct"] = round(min(100.0, (disk["io_ms"] - disk0["io_ms"]) / (10 * dt)), 1)
            for pid, (name, t) in ticks.items():
                if pid in ticks0 and t > ticks0[pid][1]:
                    out["procs"].append({"elapsed_s": elapsed_s, "pid": pid,
                                         "name": SELF_NAME if pid == self.self_pid else name,
                                         "cpu_pct": round(100.0 * (t - ticks0[pid][1]) / CLK_TCK / dt, 1)})
        self._prev = (now, irqs, disk, ticks)
        return out


def _per_key(rows: list[dict], key, value: str, n: int) -> list[dict]:
    """Mean over n intervals (absent ones count 0) and max, largest mean first."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(float(r[value]))
    n = max(n, 1)
    out = [{"key": k, "mean": round(sum(v) / n, 1), "max": round(max(v), 1)} for k, v in groups.items()]
    return sorted(out, key=lambda e: (-e["mean"], str(e["key"])))


def irq_summary(rows: list[dict], n_intervals: int) -> list[dict]:
    return [{"irq": e["key"][0], "name": e["key"][1], "mean_hz": e["mean"], "max_hz": e["max"]}
            for e in _per_key(rows, lambda r: (str(r["irq"]), r["name"]), "rate_hz", n_intervals)]


def proc_summary(rows: list[dict], n_intervals: int) -> list[dict]:
    """mean_pct and max_pct are of one core."""
    return [{"pid": e["key"][0], "name": e["key"][1], "mean_pct": e["mean"], "max_pct": e["max"]}
            for e in _per_key(rows, lambda r: (int(r["pid"]), r["name"]), "cpu_pct", n_intervals)]


def intervals(system_rows: list[dict]) -> int:
    """System rows carrying rates: what the summaries average over."""
    return sum(1 for r in system_rows if r.get("os_dt_s") not in (None, ""))


TOP_IRQS = 8
TOP_PROCS = 6


def _num(v) -> float | None:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def system_extremes(rows: list[dict]) -> dict:
    """None where never read."""
    def col(k):
        return [x for x in (_num(r.get(k)) for r in rows) if x is not None]

    def rng(k):
        v = col(k)
        return (min(v), max(v)) if v else None
    out = {"cma_free_min_mb": min(col("cma_free_mb"), default=None), "isp_mhz": rng("isp_mhz"),
           "core_mhz": rng("core_mhz"),
           "disk_write_mean_kbps": round(sum(col("disk_write_kbps")) / len(col("disk_write_kbps")), 1)
           if col("disk_write_kbps") else None,
           "disk_write_max_kbps": max(col("disk_write_kbps"), default=None),
           "disk_read_max_kbps": max(col("disk_read_kbps"), default=None),
           "disk_busy_max_pct": max(col("disk_busy_pct"), default=None)}
    for kind in PSI_KINDS:
        out[f"psi_{kind}_max"] = max(col(f"psi_{kind}_some"), default=None)
    return out


def summary_lines(system_rows: list[dict], irq_rows: list[dict], proc_rows: list[dict]) -> list[str]:
    e = system_extremes(system_rows)
    n = intervals(system_rows)
    fmt = lambda v, spec=".0f", unit="": "--" if v is None else f"{v:{spec}}{unit}"      # noqa: E731
    rng = lambda r: "--" if r is None else f"{r[0]:.0f}-{r[1]:.0f} MHz"                # noqa: E731
    psi = ("no PSI in this kernel" if all(e[f"psi_{k}_max"] is None for k in PSI_KINDS) else
           ", ".join(f"{k} {fmt(e[f'psi_{k}_max'], '.1f', '%')}" for k in PSI_KINDS))
    lines = ["os (the whole Pi)",
             f"  CMA free min {fmt(e['cma_free_min_mb'], '.0f', ' MB')}   clocks: ISP {rng(e['isp_mhz'])}, "
             f"core {rng(e['core_mhz'])}",
             f"  SD card: write mean {fmt(e['disk_write_mean_kbps'], '.0f', ' kB/s')} "
             f"(max {fmt(e['disk_write_max_kbps'])}), read max {fmt(e['disk_read_max_kbps'], '.0f', ' kB/s')}, "
             f"busy max {fmt(e['disk_busy_max_pct'], '.0f', '%')}",
             f"  stalled (pressure, max of the 10 s averages): {psi}",
             "", f"interrupts per second, busiest {TOP_IRQS} (counted by the hardware; Unicam follows the frames)"]
    irqs = irq_summary(irq_rows, n)[:TOP_IRQS]
    lines += [f"  {i['name'][:30]:<30} {('(' + i['irq'] + ')'):>8}  mean {i['mean_hz']:>8.1f}  max {i['max_hz']:>8.1f}"
              for i in irqs] or ["  none recorded"]
    lines += ["", f"other processes, busiest {TOP_PROCS} (cpu % of one core; the run itself is in threads above)"]
    procs = proc_summary(proc_rows, n)[:TOP_PROCS]
    lines += [f"  {p['name'][:20]:<20} {p['pid']:>7}  mean {p['mean_pct']:>5.1f}  max {p['max_pct']:>5.1f}"
              for p in procs] or ["  none recorded"]
    return lines
