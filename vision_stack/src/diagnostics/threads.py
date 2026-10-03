"""Threads and cores: what each of a process's threads is doing, on which core, read from the kernel.

Purpose:
    The robot's process runs several threads at once: the frame loop, the
    sensor hub, pigpio's callback thread, GStreamer's capture threads (and
    the frame recorder in the linkers). The kernel schedules them across
    the Pi's cores, and Python's GIL lets only one of the Python ones run
    Python at a time. Nothing in the code shows how that plays out; the
    kernel does, in /proc. This module reads it: per thread, its CPU use
    (user and system), the core it last ran on, its state and how often it
    blocked (voluntary context switches) or was preempted (involuntary);
    per core, how busy it was.

    It also names threads at the OS level. Python's thread names stay inside
    Python (before 3.14), so top, ps and /proc show every Python thread as
    "python3"; name_os_thread() sets the kernel's name (15 bytes at most),
    so the threads can be told apart from outside the process.

Main package:
    name_os_thread(name, thread): the kernel name of a thread (default: the
        calling one). name_pigpio_threads(): names pigpio's callback threads.
    read_tasks(pid) -> {tid: TaskTimes}; read_core_times() -> {core: CoreTimes}.
    ThreadSampler: rows of per-thread and per-core rates between two reads.

Flow:
    1. Each thread that the code starts names itself (sensor-hub,
       frame-recorder, system-monitor); drive.py names pigpio's.
    2. A sampler reads /proc/<pid>/task/*/stat and status, and /proc/stat,
       at an interval (src/diagnostics/monitor.py runs it in its own process).
    3. Each read after the first gives rates over the interval: CPU %,
       context switches per second, core busy %.
"""
import ctypes
import ctypes.util
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

PROC = Path("/proc")
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100   # utime / stime units per second
OS_NAME_MAX = 15            # the kernel's thread name limit (16 bytes with the NUL)
MAIN_THREAD = "main"        # the process's first thread (tid == pid), whose kernel name is the program's
PIGPIO_CALLBACK = "pigpio-cb"

THREAD_FIELDS = ("elapsed_s", "tid", "name", "state", "core", "cpu_pct", "user_pct", "sys_pct",
                 "vol_ctx_s", "invol_ctx_s")
CORE_FIELDS = ("elapsed_s", "core", "busy_pct")


# =============================================================================
# Naming
# =============================================================================

def _libc():
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fn = libc.pthread_setname_np
    except (OSError, AttributeError):
        return None
    fn.argtypes = (ctypes.c_ulong, ctypes.c_char_p)
    fn.restype = ctypes.c_int
    return fn


_SETNAME = _libc()


def name_os_thread(name: str, thread: threading.Thread | None = None) -> bool:
    """
    Set a thread's kernel name, as top, ps and /proc show it.

    Inputs:
        name: Cut to OS_NAME_MAX bytes.
        thread: A started thread; None is the calling thread. Thread.ident is
            the pthread handle on Linux, so any started thread can be named.
    Outputs:
        True if the kernel took it; False off Linux, or for a thread not
        started or already finished. Never raises: naming is cosmetic.
    """
    thread = thread or threading.current_thread()
    if _SETNAME is None or thread.ident is None or not thread.is_alive():
        return False
    try:
        return _SETNAME(thread.ident, name.encode()[:OS_NAME_MAX]) == 0
    except (ctypes.ArgumentError, OSError):
        return False


def name_pigpio_threads() -> int:
    """
    Name every running pigpio callback thread PIGPIO_CALLBACK.

    Each pigpio.pi() connection starts one (class _callback_thread) to run
    the pin callbacks: the encoders count in it. Called after a connection
    is made (drive.py); safe to call again.

    Outputs:
        How many were named.
    """
    return sum(name_os_thread(PIGPIO_CALLBACK, t) for t in threading.enumerate()
               if type(t).__name__ == "_callback_thread")


# =============================================================================
# Reading /proc
# =============================================================================

@dataclass(frozen=True)
class TaskTimes:
    """One thread's counters at one moment. Times in clock ticks (CLK_TCK per second)."""
    tid: int
    name: str
    state: str          # R running, S sleeping, D in I/O, ...
    core: int           # the core it last ran on
    utime: int
    stime: int
    vol_ctx: int        # times it blocked (sleep, I/O, a lock: the GIL included)
    invol_ctx: int      # times the kernel took the core from it


@dataclass(frozen=True)
class CoreTimes:
    """One core's /proc/stat counters, in clock ticks."""
    busy: int
    total: int


def parse_stat(text: str) -> tuple[str, str, int, int, int]:
    """
    (name, state, core, utime, stime) from a /proc/.../stat line.

    The name sits in parentheses and may itself hold spaces or ')', so the
    fields are counted from the last ')': state is field 3, utime 14, stime
    15, the last core 39 (proc(5)).
    """
    name = text[text.index("(") + 1:text.rindex(")")]
    rest = text[text.rindex(")") + 2:].split()
    return name, rest[0], int(rest[36]), int(rest[11]), int(rest[12])


def parse_ctx(text: str) -> tuple[int, int]:
    """(voluntary, nonvoluntary) context switches from a /proc/.../status file."""
    vol = invol = 0
    for line in text.splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            vol = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            invol = int(line.split()[1])
    return vol, invol


def read_tasks(pid: int, proc: Path = PROC) -> dict[int, TaskTimes]:
    """
    Every thread of pid, as it stands now.

    Outputs:
        {tid: TaskTimes}; a thread that ends mid-read is left out. The
        process's first thread (tid == pid) is named MAIN_THREAD.
    Raises:
        ProcessLookupError: If the process is gone.
    """
    task_dir = proc / str(pid) / "task"
    try:
        tids = sorted(int(p.name) for p in task_dir.iterdir())
    except FileNotFoundError:
        raise ProcessLookupError(pid) from None
    out = {}
    for tid in tids:
        try:
            name, state, core, utime, stime = parse_stat((task_dir / str(tid) / "stat").read_text())
            vol, invol = parse_ctx((task_dir / str(tid) / "status").read_text())
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
        out[tid] = TaskTimes(tid, MAIN_THREAD if tid == pid else name, state, core, utime, stime, vol, invol)
    return out


def parse_core_times(text: str) -> dict[int, CoreTimes]:
    """
    {core: CoreTimes} from /proc/stat's cpuN lines (the all-core "cpu" line skipped).

    busy is everything but idle and iowait; guest time is already inside
    user and nice, so only the first 8 columns count toward total.
    """
    out = {}
    for line in text.splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            label, *cols = line.split()
            user, nice, system, idle, iowait, irq, softirq, steal = (int(c) for c in cols[:8])
            total = user + nice + system + idle + iowait + irq + softirq + steal
            out[int(label[3:])] = CoreTimes(total - idle - iowait, total)
    return out


def read_core_times(proc: Path = PROC) -> dict[int, CoreTimes]:
    return parse_core_times((proc / "stat").read_text())


# =============================================================================
# Rates between reads
# =============================================================================

class ThreadSampler:
    """
    Turns successive /proc reads into rates over each interval.

    Inputs:
        pid: The process to watch.
        python_names: {tid: name} for threads still called by the program's
            name, e.g. from threading.enumerate() when watching this process.
        proc: /proc, injectable for tests.

    take(elapsed_s, dt) -> (thread_rows, core_rows) in THREAD_FIELDS /
    CORE_FIELDS order: one row per thread and per core, rates over the dt
    seconds since the previous take(). The first take() only primes the
    counters and returns no rows; a thread first seen in this take() gets
    its rates from the next one.
    """
    def __init__(self, pid: int, python_names: dict[int, str] | None = None, proc: Path = PROC) -> None:
        self.pid, self.proc = pid, proc
        self.python_names = python_names or {}
        self._tasks: dict[int, TaskTimes] | None = None
        self._cores: dict[int, CoreTimes] | None = None

    def take(self, elapsed_s: float, dt: float) -> tuple[list[dict], list[dict]]:
        tasks = read_tasks(self.pid, self.proc)
        cores = read_core_times(self.proc)
        prev_tasks, prev_cores = self._tasks, self._cores
        self._tasks, self._cores = tasks, cores
        if prev_tasks is None or dt <= 0:
            return [], []

        threads = []
        program = self._program_name() if self.python_names else None
        for tid, now in tasks.items():
            before = prev_tasks.get(tid)
            if before is None:
                continue
            user = (now.utime - before.utime) / CLK_TCK / dt * 100.0
            system = (now.stime - before.stime) / CLK_TCK / dt * 100.0
            # A thread nobody named still carries the program's name: use its Python name if known
            name = self.python_names.get(tid, now.name) if now.name == program else now.name
            threads.append({"elapsed_s": elapsed_s, "tid": tid, "name": name, "state": now.state,
                            "core": now.core, "cpu_pct": round(user + system, 1),
                            "user_pct": round(user, 1), "sys_pct": round(system, 1),
                            "vol_ctx_s": round((now.vol_ctx - before.vol_ctx) / dt, 1),
                            "invol_ctx_s": round((now.invol_ctx - before.invol_ctx) / dt, 1)})

        core_rows = []
        for core, now in cores.items():
            before = prev_cores.get(core)
            span = None if before is None else now.total - before.total
            if span:
                core_rows.append({"elapsed_s": elapsed_s, "core": core,
                                  "busy_pct": round((now.busy - before.busy) / span * 100.0, 1)})
        return threads, core_rows

    def _program_name(self) -> str | None:
        """The kernel name the program's threads inherit (its first thread's)."""
        try:
            return parse_stat((self.proc / str(self.pid) / "stat").read_text())[0]
        except (OSError, ValueError):
            return None
