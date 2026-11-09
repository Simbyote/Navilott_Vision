"""System monitor: SoC temperature, CPU clock, throttle flags and memory, sampled in the background.

Purpose:
    A soak run needs the Pi's condition beside its frame timings, on one
    clock. Sampled once a second on a thread and kept in memory, so no file
    I/O lands in the run; every source is optional, so it runs anywhere.

Main package:
    SystemMonitor.rows(): every sample, elapsed_s on the frame loop's clock.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

from src.os_threads import name_os_thread

THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
CPUFREQ_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
STATUS_PATH = Path("/proc/self/status")
MEMINFO_PATH = Path("/proc/meminfo")

# Bits 0-3 are now; bits 16-19 the same, latched since boot
THROTTLE_BITS = {
    "under_voltage": 0,
    "freq_capped": 1,
    "throttled": 2,
    "soft_temp_limit": 3,
}
OCCURRED_SHIFT = 16

FIELDS = ("elapsed_s", "temp_c", "cpu_mhz", "throttled_raw",
          *THROTTLE_BITS, "rss_mb", "mem_available_mb", "load_1m")


def _read_number(path: Path) -> float | None:
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def read_temp_c(path: Path = THERMAL_PATH) -> float | None:
    raw = _read_number(path)
    return None if raw is None else raw / 1000.0


def read_cpu_mhz(path: Path = CPUFREQ_PATH) -> float | None:
    raw = _read_number(path)
    return None if raw is None else raw / 1000.0


def _read_kb_field(path: Path, field: str) -> float | None:
    """A 'Field:   1234 kB' line, in MB."""
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(field + ":"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_rss_mb(path: Path = STATUS_PATH) -> float | None:
    return _read_kb_field(path, "VmRSS")


def read_mem_available_mb(path: Path = MEMINFO_PATH) -> float | None:
    return _read_kb_field(path, "MemAvailable")


def parse_throttled(text: str) -> int | None:
    """'throttled=0x50005' -> 0x50005; None for anything else."""
    try:
        key, value = text.strip().split("=", 1)
        return int(value, 16) if key == "throttled" else None
    except ValueError:
        return None


def read_throttled(run=subprocess.run) -> int | None:
    try:
        out = run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_throttled(out.stdout) if out.returncode == 0 else None


def decode_throttled(value: int | None) -> dict:
    """{flag: now} and {flag_occurred: since boot}; all None when value is None."""
    out = {}
    for name, bit in THROTTLE_BITS.items():
        out[name] = None if value is None else (value >> bit) & 1
        out[f"{name}_occurred"] = None if value is None else (value >> (bit + OCCURRED_SHIFT)) & 1
    return out


def read_load_1m() -> float | None:
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


class SystemMonitor:
    """A context manager, or start() and stop(). reader is injectable for tests."""

    def __init__(self, interval_s: float = 1.0, reader=None):
        self.interval_s = interval_s
        self._reader = reader or sample
        self._rows = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None

    def start(self) -> "SystemMonitor":
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="system-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5 * self.interval_s + 1)
        self._take()                        # a final sample

    @property
    def t0(self) -> float | None:
        """perf_counter() at start(); the frame loop subtracts it for elapsed_s."""
        return self._t0

    def rows(self) -> list:
        return list(self._rows)

    def _take(self) -> None:
        row = {"elapsed_s": round(time.perf_counter() - self._t0, 3), **self._reader()}
        self._rows.append({k: row.get(k) for k in FIELDS})

    def _loop(self) -> None:
        name_os_thread("system-monitor")
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval_s)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def sample(status_path: Path = STATUS_PATH) -> dict:
    """One reading of everything (FIELDS); status_path picks whose memory rss_mb is."""
    raw = read_throttled()
    flags = decode_throttled(raw)
    return {
        "temp_c": read_temp_c(),
        "cpu_mhz": read_cpu_mhz(),
        "throttled_raw": None if raw is None else hex(raw),
        **{k: flags[k] for k in THROTTLE_BITS},
        "rss_mb": read_rss_mb(status_path),
        "mem_available_mb": read_mem_available_mb(),
        "load_1m": read_load_1m(),
    }
