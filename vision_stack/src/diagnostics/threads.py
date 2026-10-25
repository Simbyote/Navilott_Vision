"""Threads and cores: what each of a process's threads is doing, on which core, read from /proc.

Purpose:
    The robot runs several threads across the Pi's cores, with the GIL
    letting one run Python at a time; the kernel shows how that plays out.
    Naming threads and real-time priority are os_threads.py's.

Main package:
    ThreadSampler: per-thread CPU, core and context switches, and per-core
    busy time, as rates between two reads.
"""
import os
from dataclasses import dataclass
from pathlib import Path

PROC = Path("/proc")
CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100   # utime / stime units per second
MAIN_THREAD = "main"        # tid == pid

THREAD_FIELDS = ("elapsed_s", "tid", "name", "state", "core", "cpu_pct", "user_pct", "sys_pct",
                 "vol_ctx_s", "invol_ctx_s")
CORE_FIELDS = ("elapsed_s", "core", "busy_pct")


@dataclass(frozen=True)
class TaskTimes:
    """Times in clock ticks."""
    tid: int
    name: str
    state: str          # R running, S sleeping, D in I/O, ...
    core: int           # the core it last ran on
    utime: int
    stime: int
    vol_ctx: int        # it blocked (sleep, I/O, a lock, the GIL)
    invol_ctx: int      # the kernel took the core from it


@dataclass(frozen=True)
class CoreTimes:
    busy: int
    total: int


def parse_stat(text: str) -> tuple[str, str, int, int, int]:
    """(name, state, core, utime, stime). Counted from the last ')', since the name may hold spaces or ')'."""
    name = text[text.index("(") + 1:text.rindex(")")]
    rest = text[text.rindex(")") + 2:].split()
    return name, rest[0], int(rest[36]), int(rest[11]), int(rest[12])


def parse_ctx(text: str) -> tuple[int, int]:
    """(voluntary, nonvoluntary) context switches."""
    vol = invol = 0
    for line in text.splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            vol = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            invol = int(line.split()[1])
    return vol, invol


def read_tasks(pid: int, proc: Path = PROC) -> dict[int, TaskTimes]:
    """{tid: TaskTimes}, the first thread named MAIN_THREAD. ProcessLookupError if the process is gone."""
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
    """busy is everything but idle and iowait; guest time is already inside user, so 8 columns count."""
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


class ThreadSampler:
    """
    take(elapsed_s, dt) -> (thread rows, core rows) of rates since the previous take(). The
    first take() only primes; a new thread gets rates from its second. python_names stand in
    for threads nobody named.
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
            # An unnamed thread carries the program's name; use its Python name if known
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
        try:
            return parse_stat((self.proc / str(self.pid) / "stat").read_text())[0]
        except (OSError, ValueError):
            return None
