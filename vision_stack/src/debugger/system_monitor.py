"""System monitor: SoC temperature, CPU clock, throttle flags and memory, sampled in the background.

Purpose:
    A soak run needs the Pi's condition recorded next to its frame timings,
    on one clock, so a slowdown can be matched to the heat or memory growth
    behind it. Watching vcgencmd or top in another terminal leaves nothing
    to line up afterwards. SystemMonitor samples once a second on a daemon
    thread and keeps the rows in memory, so no file I/O happens during the
    run.

Main package:
    SystemMonitor   start() / stop(); rows() returns every sample as a dict
                    in FIELDS order. elapsed_s is time.perf_counter() since
                    start(), the same clock the frame loop uses.
    read_*()        one reading each, None where the source doesn't exist
                    (a laptop has no vcgencmd and may have no thermal zone),
                    so the monitor runs anywhere and records what it can.
    decode_throttled()
                    vcgencmd get_throttled's bit field as named flags.

Sources, no extra dependencies:
    /sys/class/thermal/thermal_zone0/temp                  millidegrees C
    /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq  kHz
    vcgencmd get_throttled                                 "throttled=0x50005"
    /proc/self/status VmRSS                                this process, kB
    /proc/meminfo MemAvailable                             whole system, kB
"""

import os
import subprocess
import threading
import time
from pathlib import Path

THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
CPUFREQ_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
STATUS_PATH = Path("/proc/self/status")
MEMINFO_PATH = Path("/proc/meminfo")

# vcgencmd get_throttled: bits 0-3 are the state now, bits 16-19 the same
# conditions latched since boot
THROTTLE_BITS = {
    "under_voltage": 0,
    "freq_capped": 1,
    "throttled": 2,
    "soft_temp_limit": 3,
}
OCCURRED_SHIFT = 16

FIELDS = ("elapsed_s", "temp_c", "cpu_mhz", "throttled_raw",
          *THROTTLE_BITS, "rss_mb", "mem_available_mb", "load_1m")


# =============================================================================
# Readings
# =============================================================================

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
    """A 'Field:   1234 kB' line from a /proc file, in MB."""
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
    """vcgencmd get_throttled, or None without vcgencmd. run is injectable for tests."""
    try:
        out = run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_throttled(out.stdout) if out.returncode == 0 else None


def decode_throttled(value: int | None) -> dict:
    """
    {flag: 1/0 now} plus {flag_occurred: 1/0 since boot}; every value None
    when value is None.
    """
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


# =============================================================================
# Monitor
# =============================================================================

class SystemMonitor:
    """
    Background sampler. Use as a context manager, or start() and stop().

    interval_s: seconds between samples
    reader:     callable returning one sample dict (without elapsed_s);
                injectable for tests, defaults to sample()
    """

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
        self._take()                        # a final sample at the end of the run

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
        while not self._stop.is_set():
            self._take()
            self._stop.wait(self.interval_s)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def sample() -> dict:
    """One reading of everything; see FIELDS."""
    raw = read_throttled()
    flags = decode_throttled(raw)
    return {
        "temp_c": read_temp_c(),
        "cpu_mhz": read_cpu_mhz(),
        "throttled_raw": None if raw is None else hex(raw),
        **{k: flags[k] for k in THROTTLE_BITS},
        "rss_mb": read_rss_mb(),
        "mem_available_mb": read_mem_available_mb(),
        "load_1m": read_load_1m(),
    }
