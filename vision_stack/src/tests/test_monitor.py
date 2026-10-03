"""
test_monitor.py  --  src/diagnostics/monitor.py

record() over a fake /proc on a fake clock: rows every interval, system
rows once a second, and every way it ends (the process gone, the duration,
a launched run exiting, Ctrl-C). The summaries: per-thread core shares,
moves and ordering; throttle flags during the run and since boot. The run
folder's files, find_pid(), and the command line launching a real child.

--software  Fake /proc trees and short-lived child processes. No Pi needed.
"""
import csv
import json
import sys

import pytest

import src.diagnostics.monitor as mon
from src.diagnostics.system_monitor import read_throttled
from src.diagnostics.threads import CORE_FIELDS, THREAD_FIELDS
from src.tests.test_threads import write_proc


class Clock:
    """A clock record() advances through sleep()."""
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def system_row(**kw):
    return {"temp_c": 50.0, "cpu_mhz": 1000.0, "rss_mb": 100.0, "mem_available_mb": 200.0,
            "throttled_raw": "0x0", **kw}


def go(tmp_path, **kw):
    write_proc(tmp_path, 100, {100: ("python3", 0, 0, 0, 0, 0), 101: ("sensor-hub", 0, 0, 1, 0, 0)}, {0: (0, 0)})
    clock = Clock()
    rec = mon.record(100, 0.5, clock=clock, sleep=clock.sleep, proc=tmp_path,
                     system_reader=lambda: system_row(), **kw)
    return rec, clock


# =============================================================================
# record()
# =============================================================================

@pytest.mark.software
def test_rows_every_interval_and_system_rows_once_a_second(tmp_path):
    rec, clock = go(tmp_path, duration_s=3.0)
    assert clock.now == 3.0 and not rec["interrupted"]
    assert sorted({r["elapsed_s"] for r in rec["threads"]}) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert len(rec["threads"]) == 12                                  # 2 threads per interval
    assert rec["cores"] == []                       # the fake /proc/stat never advances: no core time to rate
    assert [r["elapsed_s"] for r in rec["system"]] == [0.0, 1.0, 2.0, 3.0]
    assert list(rec["system"][0]) == list(mon.SYSTEM_COLUMNS) and rec["system"][0]["temp_c"] == 50.0
    assert rec["t0_monotonic"] == 0.0


@pytest.mark.software
def test_flags_latched_since_boot_survive_into_the_rows_and_the_summary(tmp_path):
    write_proc(tmp_path, 100, {100: ("python3", 0, 0, 0, 0, 0)}, {0: (0, 0)})
    clock = Clock()
    rec = mon.record(100, 0.5, duration_s=1.0, clock=clock, sleep=clock.sleep, proc=tmp_path,
                     system_reader=lambda: system_row(under_voltage=0, under_voltage_occurred=1))
    assert all(r["under_voltage_occurred"] == 1 for r in rec["system"])
    assert mon.system_summary(rec["system"])["throttled_since_boot"] == ["under_voltage"]


@pytest.mark.software
def test_it_ends_when_the_process_is_gone(tmp_path):
    write_proc(tmp_path, 100, {100: ("python3", 0, 0, 0, 0, 0)}, {0: (0, 0)})
    clock = Clock()
    seen = []

    def sleep(s):
        clock.sleep(s)
        seen.append(clock.now)
        if clock.now >= 1.0:
            for p in sorted((tmp_path / "100").rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            (tmp_path / "100").rmdir()
    rec = mon.record(100, 0.5, clock=clock, sleep=sleep, proc=tmp_path, system_reader=system_row)
    assert seen[-1] == 1.0 and not rec["interrupted"] and rec["elapsed_s"] == 1.0


@pytest.mark.software
def test_a_launched_run_ending_stops_it_before_another_sample(tmp_path):
    polls = iter([True, True, False])
    rec, clock = go(tmp_path, alive=lambda: next(polls, False), duration_s=5.0)    # the duration only as a backstop
    assert clock.now == 1.0 and {r["elapsed_s"] for r in rec["threads"]} == {0.5}


@pytest.mark.software
def test_ctrl_c_ends_it_and_keeps_the_rows(tmp_path):
    write_proc(tmp_path, 100, {100: ("python3", 0, 0, 0, 0, 0)}, {0: (0, 0)})
    clock = Clock()

    def sleep(s):
        clock.sleep(s)
        if clock.now >= 1.5:
            raise KeyboardInterrupt
    rec = mon.record(100, 0.5, clock=clock, sleep=sleep, proc=tmp_path, system_reader=system_row)
    assert rec["interrupted"] and len(rec["threads"]) == 2


# =============================================================================
# Summaries and the run folder
# =============================================================================

def trow(t, tid, name, core, cpu, vol=10.0, invol=1.0):
    return {"elapsed_s": t, "tid": tid, "name": name, "state": "S", "core": core, "cpu_pct": cpu,
            "user_pct": cpu, "sys_pct": 0.0, "vol_ctx_s": vol, "invol_ctx_s": invol}


@pytest.mark.software
def test_thread_summary_core_shares_moves_and_order():
    rows = [trow(0.5, 1, "main", 0, 60.0), trow(1.0, 1, "main", 1, 80.0), trow(1.5, 1, "main", 1, 70.0),
            trow(2.0, 1, "main", 0, 70.0), trow(0.5, 2, "sensor-hub", 3, 2.0, vol=100.0)]
    main, hub = mon.thread_summary(rows)
    assert main == {"name": "main", "tid": 1, "samples": 4, "cpu_mean": 70.0, "cpu_max": 80.0,
                    "core_share": {0: 0.5, 1: 0.5}, "moves": 2, "vol_ctx_s": 10.0, "invol_ctx_s": 1.0}
    assert hub["name"] == "sensor-hub" and hub["vol_ctx_s"] == 100.0 and hub["moves"] == 0


@pytest.mark.software
def test_system_summary_throttling_during_the_run_and_since_boot():
    # under-voltage happens during the run; throttling happened earlier since boot, not now
    rows = [system_row(under_voltage=0, under_voltage_occurred=1, throttled=0, throttled_occurred=1),
            system_row(under_voltage=1, under_voltage_occurred=1, throttled=0, throttled_occurred=1, temp_c=71.5,
                       cpu_mhz=600.0)]
    s = mon.system_summary(rows)
    assert s["throttled_during"] == ["under_voltage"] and s["throttled_since_boot"] == ["under_voltage", "throttled"]
    assert (s["temp_max_c"], s["cpu_mhz_min"], s["cpu_mhz_max"]) == (71.5, 600.0, 1000.0) and s["readable"]
    assert not mon.system_summary([{"temp_c": None, "throttled_raw": None}])["readable"]


@pytest.mark.software
def test_the_run_folder_holds_every_file_and_the_summary_names_each_thread(tmp_path):
    rec = {"threads": [trow(0.5, 1, "main", 0, 60.0), trow(0.5, 2, "sensor-hub", 3, 2.0)],
           "cores": [{"elapsed_s": 0.5, "core": 0, "busy_pct": 61.0}],
           "system": [{k: system_row(under_voltage=1).get(k) for k in mon.SYSTEM_COLUMNS}],
           "interrupted": False, "elapsed_s": 0.5}
    meta = {"pid": 1, "command": "python3 -m src.main", "interval_s": 0.5, "cores": 4}
    lines = mon.write(str(tmp_path), meta, rec)
    text = "\n".join(lines)
    assert "main" in text and "sensor-hub" in text and "core 0" in text and "under_voltage" in text
    assert (tmp_path / "summary.txt").read_text().splitlines() == lines
    assert json.loads((tmp_path / "meta.json").read_text())["command"] == "python3 -m src.main"
    for name, fields in (("threads.csv", THREAD_FIELDS), ("cores.csv", CORE_FIELDS), ("system.csv", mon.SYSTEM_COLUMNS)):
        assert next(csv.reader(open(tmp_path / name))) == list(fields)


@pytest.mark.software
def test_find_pid_takes_the_newest_match_and_never_itself(tmp_path):
    for pid, cmd in ((10, b"python3\0-m\0src.main\0"), (12, b"python3\0-m\0src.main\0"), (11, b"bash\0")):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "cmdline").write_bytes(cmd)
    (tmp_path / "self").mkdir()
    assert mon.find_pid("src.main", tmp_path) == 12
    assert mon.find_pid("src.main", tmp_path, exclude=(12,)) == 10
    assert mon.find_pid("nothing", tmp_path) is None


# =============================================================================
# The command line
# =============================================================================

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the real /proc")
@pytest.mark.software
def test_cli_launches_a_run_records_it_and_returns_its_exit_code(tmp_path):
    child = ("import threading, time, sys\n"
             "from src.diagnostics.threads import name_os_thread\n"
             "def w():\n    name_os_thread('child-worker')\n    time.sleep(1.0)\n"
             "t = threading.Thread(target=w); t.start(); t.join(); sys.exit(3)\n")
    code = mon.cli(["--interval", "0.2", "--out", str(tmp_path), "--", sys.executable, "-c", child])
    assert code == 3
    names = {r["name"] for r in csv.DictReader(open(tmp_path / "threads.csv"))}
    assert {"main", "child-worker"} <= names
    assert "child-worker" in (tmp_path / "summary.txt").read_text()


@pytest.mark.software
def test_cli_needs_exactly_one_target_and_a_missing_match_is_exit_2(monkeypatch, capsys):
    assert mon.cli([]) == 2
    assert mon.cli(["--pid", "1", "--match", "x"]) == 2
    monkeypatch.setattr(mon, "MATCH_WAIT_S", 0.0)
    assert mon.cli(["--match", "no-such-process-anywhere-xyz"]) == 2
    assert "no process to watch" in capsys.readouterr().out


@pytest.mark.hardware
def test_a_short_recording_on_the_pi_reads_every_source(tmp_path):
    """Two seconds of a child that sleeps: threads, cores, temperature, clock and throttle flags all read."""
    if read_throttled() is None:
        pytest.skip("vcgencmd unavailable: not a Pi")
    code = mon.cli(["--interval", "0.25", "--out", str(tmp_path), "--", sys.executable, "-c",
                    "import time; time.sleep(2)"])
    assert code == 0
    rows = list(csv.DictReader(open(tmp_path / "system.csv")))
    assert rows and all(r["temp_c"] and r["cpu_mhz"] and r["throttled_raw"] for r in rows), \
        "a Pi reads temperature, clock and vcgencmd get_throttled"
    assert list(csv.DictReader(open(tmp_path / "cores.csv"))), "every core reads from /proc/stat"
    assert "throttled during the run" in (tmp_path / "summary.txt").read_text()
