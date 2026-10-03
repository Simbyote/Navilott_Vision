"""
test_threads.py  --  src/diagnostics/threads.py

Parsing /proc's stat and status lines (a thread name with spaces and
parentheses included) and /proc/stat's per-core lines; reading a process's
threads from a fake /proc tree; the sampler's rates over an interval, exact
on known counters; Python names standing in only for unnamed threads; and,
on this machine's real /proc, that naming a thread shows in the kernel and a
busy thread reads as busy.

--software  Fake /proc trees and this process's own threads. Linux only for
            the real-/proc tests (skipped elsewhere).
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

import src.diagnostics.threads as th

LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the real /proc")


def stat_line(tid, name, state="S", utime=0, stime=0, core=0):
    """A /proc/<pid>/task/<tid>/stat line with the fields threads.py reads set."""
    rest = ["0"] * 50
    rest[0], rest[11], rest[12], rest[36] = state, str(utime), str(stime), str(core)
    return f"{tid} ({name}) " + " ".join(rest) + "\n"


def status_text(vol=0, invol=0):
    return f"Name:\tx\nVmRSS:\t 2048 kB\nvoluntary_ctxt_switches:\t{vol}\nnonvoluntary_ctxt_switches:\t{invol}\n"


def write_proc(root: Path, pid: int, threads: dict, cores: dict, program="python3"):
    """threads: {tid: (name, utime, stime, core, vol, invol)}; cores: {n: (busy, idle)}."""
    (root / str(pid)).mkdir(parents=True, exist_ok=True)
    (root / str(pid) / "stat").write_text(stat_line(pid, program))
    task = root / str(pid) / "task"
    if task.exists():
        for p in task.iterdir():
            for f in p.iterdir():
                f.unlink()
            p.rmdir()
    for tid, (name, ut, st, core, vol, invol) in threads.items():
        d = task / str(tid)
        d.mkdir(parents=True)
        (d / "stat").write_text(stat_line(tid, name, "R", ut, st, core))
        (d / "status").write_text(status_text(vol, invol))
    lines = ["cpu  1 2 3 4 5 6 7 8 0 0"]
    for n, (busy, idle) in cores.items():
        lines.append(f"cpu{n} {busy} 0 0 {idle} 0 0 0 0 0 0")
    (root / "stat").write_text("\n".join(lines) + "\nintr 0\n")


# =============================================================================
# Parsing
# =============================================================================

@pytest.mark.software
def test_parse_stat_counts_fields_from_the_last_parenthesis():
    line = stat_line(42, "odd ) (name", "R", utime=123, stime=45, core=3)
    assert th.parse_stat(line) == ("odd ) (name", "R", 3, 123, 45)


@pytest.mark.software
def test_parse_ctx_and_core_times():
    assert th.parse_ctx(status_text(7, 2)) == (7, 2)
    text = "cpu  9 9 9 9 9 9 9 9 0 0\ncpu0 10 1 4 80 5 0 0 0 0 0\ncpu1 0 0 0 100 0 0 0 0 0 0\nintr 1\n"
    assert th.parse_core_times(text) == {0: th.CoreTimes(busy=15, total=100), 1: th.CoreTimes(busy=0, total=100)}


@pytest.mark.software
def test_read_tasks_names_the_first_thread_main_and_raises_for_a_gone_process(tmp_path):
    write_proc(tmp_path, 100, {100: ("python3", 1, 2, 0, 3, 4), 101: ("sensor-hub", 5, 6, 2, 7, 8)}, {0: (1, 1)})
    tasks = th.read_tasks(100, tmp_path)
    assert tasks[100].name == th.MAIN_THREAD and tasks[101] == th.TaskTimes(101, "sensor-hub", "R", 2, 5, 6, 7, 8)
    with pytest.raises(ProcessLookupError):
        th.read_tasks(999, tmp_path)


# =============================================================================
# The sampler
# =============================================================================

@pytest.mark.software
def test_the_sampler_gives_exact_rates_over_the_interval(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "CLK_TCK", 100)
    write_proc(tmp_path, 100, {100: ("python3", 0, 0, 0, 0, 0), 101: ("sensor-hub", 10, 0, 1, 100, 0)},
               {0: (0, 0), 1: (0, 0)})
    s = th.ThreadSampler(100, proc=tmp_path)
    assert s.take(0.0, 0.0) == ([], [])                                     # the first read only primes
    # 0.5 s later: main used 30 user + 10 system ticks (0.4 s of CPU), sensor-hub 1 tick, 50 sleeps;
    # a new thread appears; core 0 was busy 25 of 50 ticks
    write_proc(tmp_path, 100, {100: ("python3", 30, 10, 3, 4, 2), 101: ("sensor-hub", 11, 0, 1, 150, 0),
                               102: ("pigpio-cb", 0, 0, 0, 0, 0)},
               {0: (25, 25), 1: (5, 45)})
    threads, cores = s.take(0.5, 0.5)
    rows = {r["name"]: r for r in threads}
    assert set(rows) == {th.MAIN_THREAD, "sensor-hub"}                      # pigpio-cb gets rates next time
    assert rows["main"] == {"elapsed_s": 0.5, "tid": 100, "name": "main", "state": "R", "core": 3,
                                 "cpu_pct": 80.0, "user_pct": 60.0, "sys_pct": 20.0,
                                 "vol_ctx_s": 8.0, "invol_ctx_s": 4.0}
    assert rows["sensor-hub"]["cpu_pct"] == 2.0 and rows["sensor-hub"]["vol_ctx_s"] == 100.0
    assert cores == [{"elapsed_s": 0.5, "core": 0, "busy_pct": 50.0}, {"elapsed_s": 0.5, "core": 1, "busy_pct": 10.0}]
    assert list(rows["main"]) == list(th.THREAD_FIELDS) and list(cores[0]) == list(th.CORE_FIELDS)


@pytest.mark.software
def test_python_names_stand_in_only_for_threads_nobody_named(tmp_path):
    write_proc(tmp_path, 100, {100: ("python3", 0, 0, 0, 0, 0), 101: ("python3", 0, 0, 0, 0, 0),
                               102: ("sensor-hub", 0, 0, 0, 0, 0)}, {0: (0, 0)})
    s = th.ThreadSampler(100, python_names={100: "MainThread", 101: "Thread-3", 102: "Thread-1"}, proc=tmp_path)
    s.take(0.0, 0.0)
    names = {r["tid"]: r["name"] for r in s.take(0.5, 0.5)[0]}
    assert names == {100: "main", 101: "Thread-3", 102: "sensor-hub"}


# =============================================================================
# Naming, on the real kernel
# =============================================================================

def _comm(thread):
    return Path(f"/proc/self/task/{thread.native_id}/comm").read_text().strip()


@LINUX
@pytest.mark.software
def test_a_thread_can_be_named_from_inside_or_outside_and_long_names_are_cut():
    go, named = threading.Event(), threading.Event()

    def body():
        th.name_os_thread("inside-name")
        named.set()
        go.wait(5)
    t = threading.Thread(target=body)
    t.start()
    named.wait(5)
    assert _comm(t) == "inside-name"
    assert th.name_os_thread("a-very-long-thread-name", t) and _comm(t) == "a-very-long-thr"
    go.set()
    t.join()
    assert not th.name_os_thread("x", threading.Thread(target=lambda: None))   # never started


@LINUX
@pytest.mark.software
def test_pigpio_callback_threads_are_found_and_named():
    stop = threading.Event()

    class _callback_thread(threading.Thread):           # pigpio's class name
        def run(self):
            stop.wait(5)
    t = _callback_thread()
    t.start()
    try:
        assert th.name_pigpio_threads() >= 1 and _comm(t) == th.PIGPIO_CALLBACK
    finally:
        stop.set()
        t.join()


@LINUX
@pytest.mark.software
def test_a_busy_named_thread_reads_busy_on_its_core():
    stop = threading.Event()

    def spin():
        th.name_os_thread("diag-busy")
        while not stop.is_set():
            pass
    t = threading.Thread(target=spin)
    t.start()
    try:
        s = th.ThreadSampler(os.getpid())
        s.take(0.0, 0.0)
        t0 = time.perf_counter()
        time.sleep(0.4)
        rows = {r["name"]: r for r in s.take(0.4, time.perf_counter() - t0)[0]}
    finally:
        stop.set()
        t.join()
    busy = rows["diag-busy"]
    assert busy["cpu_pct"] > 30 and 0 <= busy["core"] < (os.cpu_count() or 1)
    assert rows["main"]["cpu_pct"] < busy["cpu_pct"]                     # main only slept


def comm_of(thread, timeout_s=2.0):
    """The kernel's name for a started thread, once it has named itself (polled: naming is its first act)."""
    path = Path(f"/proc/self/task/{thread.native_id}/comm")
    end = time.perf_counter() + timeout_s
    name = path.read_text().strip()
    while time.perf_counter() < end and name.startswith("python"):
        time.sleep(0.01)
        name = path.read_text().strip()
    return name
