"""
test_os_threads.py  --  src/os_threads.py

Naming a thread from inside it or from outside, long names cut to the
kernel's 15 bytes, a thread never started refused; pigpio's callback
threads found by class and named; set_realtime asking for SCHED_FIFO on the
calling thread only, and saying why not instead of raising.

--software  This process's own threads on the real kernel (Linux only, skipped
            elsewhere) and a monkeypatched os for the scheduler calls.
"""
import sys
import threading
import time
from pathlib import Path

import pytest

import src.os_threads as ot

LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the real /proc")


def _comm(thread):
    return Path(f"/proc/self/task/{thread.native_id}/comm").read_text().strip()


@LINUX
@pytest.mark.software
def test_a_thread_can_be_named_from_inside_or_outside_and_long_names_are_cut():
    go, named = threading.Event(), threading.Event()

    def body():
        ot.name_os_thread("inside-name")
        named.set()
        go.wait(5)
    t = threading.Thread(target=body)
    t.start()
    named.wait(5)
    assert _comm(t) == "inside-name"
    assert ot.name_os_thread("a-very-long-thread-name", t) and _comm(t) == "a-very-long-thr"
    go.set()
    t.join()
    assert not ot.name_os_thread("x", threading.Thread(target=lambda: None))   # never started


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
        assert ot.name_pigpio_threads() >= 1 and _comm(t) == ot.PIGPIO_CALLBACK
    finally:
        stop.set()
        t.join()


def comm_of(thread, timeout_s=2.0):
    """The kernel's name for a started thread, once it has named itself (polled: naming is its first act)."""
    path = Path(f"/proc/self/task/{thread.native_id}/comm")
    end = time.perf_counter() + timeout_s
    name = path.read_text().strip()
    while time.perf_counter() < end and name.startswith("python"):
        time.sleep(0.01)
        name = path.read_text().strip()
    return name


@pytest.mark.software
def test_set_realtime_asks_for_fifo_on_the_calling_thread(monkeypatch):
    calls = []
    monkeypatch.setattr(ot.os, "sched_setscheduler", lambda pid, policy, param: calls.append(
        (pid, policy, param.sched_priority)), raising=False)
    monkeypatch.setattr(ot.os, "SCHED_FIFO", 1, raising=False)
    assert ot.set_realtime(20) == ""
    assert calls == [(0, 1, 20)]                     # pid 0: this thread only


@pytest.mark.software
def test_set_realtime_says_why_not_and_never_raises(monkeypatch):
    def denied(pid, policy, param):
        raise PermissionError(1, "Operation not permitted")
    monkeypatch.setattr(ot.os, "SCHED_FIFO", 1, raising=False)
    monkeypatch.setattr(ot.os, "sched_setscheduler", denied, raising=False)
    assert ot.set_realtime(20).startswith("not permitted: Operation not permitted (rtprio limit ")

    def bad(pid, policy, param):
        raise OSError(22, "Invalid argument")
    monkeypatch.setattr(ot.os, "sched_setscheduler", bad)
    assert ot.set_realtime(500).startswith("refused: ")

    monkeypatch.delattr(ot.os, "sched_setscheduler")
    assert ot.set_realtime(20) == "not supported here"
