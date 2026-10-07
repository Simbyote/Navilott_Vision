"""
test_i2c_trace.py  --  src/diagnostics/i2c_trace.py

Trace lines in the kernel's own formats (include/trace/events/i2c.h, with
and without the irq-flags and tgid columns, thread names holding dashes):
parsed; grouped into transfers per bus with their messages, data, wire time
and result, cut-off transfers left out, interleaved buses kept apart. The
adapters' clocks from the device tree. record_trace() on a fake tracefs:
the events on for the window and every setting put back, even on an error.
The analysis: occupancy against wire time, per-device rates, durations,
gaps, errors and queueing, and each finding at its threshold. The summary,
the figure, the folder, and the command line (no root, no tracefs, a
recording, and --from without root).

--software  Text traces and fake sysfs / tracefs trees. No Pi, no root.
--hardware  A real 2 s trace on the Pi (needs root: sudo make test-hw).
"""
import csv
import json
import os
from pathlib import Path

import pytest

import src.diagnostics.i2c_trace as it

PRE = "      {task}-{pid}     [00{cpu}] d..1.  {ts:.6f}: "


def ev(ts, event, rest, task="sensor-hub", pid=1234, cpu=1, bus=1):
    return PRE.format(task=task, pid=pid, cpu=cpu, ts=ts) + f"{event}: i2c-{bus} {rest}"


def imu_read(ts, dur_us=1800.0, ret=2, bus=1, task="sensor-hub"):
    """The MPU-6050's 14-byte block read: write the register, read 14."""
    end = ts + dur_us / 1e6
    data = "-".join(f"{b:02x}" for b in range(14))
    return [ev(ts, "i2c_write", "#0 a=068 f=0000 l=1 [3b]", task=task, bus=bus),
            ev(ts, "i2c_read", "#1 a=068 f=0001 l=14", task=task, bus=bus),
            ev(end, "i2c_reply", f"#1 a=068 f=0001 l=14 [{data}]", task=task, bus=bus),
            ev(end, "i2c_result", f"n=2 ret={ret}", task=task, bus=bus)]


def ads_write(ts, dur_us=500.0, task="python3"):
    """The ADS1115's config write: 3 bytes."""
    return [ev(ts, "i2c_write", "#0 a=048 f=0000 l=3 [01-c3-83]", task=task, pid=99),
            ev(ts + dur_us / 1e6, "i2c_result", "n=1 ret=1", task=task, pid=99)]


def trace(*groups):
    return "\n".join(["# tracer: nop", "#", "#           TASK-PID     CPU#  |||||  TIMESTAMP  FUNCTION"]
                     + [line for g in groups for line in g]) + "\n"


# =============================================================================
# Parsing and assembly
# =============================================================================

@pytest.mark.software
def test_events_parse_in_the_kernels_formats():
    text = "\n".join([
        ev(10.0, "i2c_write", "#0 a=068 f=0000 l=1 [3b]"),
        "  python3-77   (   77) [003] .....  10.000100: i2c_read: i2c-1 #1 a=048 f=0001 l=2",   # tgid, no flags
        "  libcamera-ipa-55 [000] 10.000200: i2c_write: i2c-10 #0 a=01a f=0000 l=3 [30-09-10]",   # dashes, no flags
        ev(10.0003, "i2c_result", "n=2 ret=-121"),
        "  sensor-hub-1234 [001] d..1. 10.0004: sched_switch: prev_comm=x",                     # not i2c
        ev(10.0005, "i2c_write", "garbled"),                                                      # no fields
        "# a comment"])
    es = it.parse_events(text)
    assert [e["event"] for e in es] == ["i2c_write", "i2c_read", "i2c_write", "i2c_result"]
    w = es[0]
    assert (w["ts"], w["bus"], w["task"], w["pid"], w["idx"], w["addr"], w["len"], w["data"]) == \
        (10.0, 1, "sensor-hub", 1234, 0, 0x68, 1, ["3b"])
    assert (es[1]["task"], es[1]["addr"], es[1]["flags"], es[1]["data"]) == ("python3", 0x48, 1, [])
    assert (es[2]["task"], es[2]["bus"], es[2]["data"]) == ("libcamera-ipa", 10, ["30", "09", "10"])
    assert (es[3]["n"], es[3]["ret"]) == (2, -121)


@pytest.mark.software
def test_wire_bits_count_start_address_data_acks_and_one_stop():
    assert it.wire_bits([("w", 1), ("r", 14)]) == (1 + 9 + 9) + (1 + 9 + 126) + 1 == 156
    assert it.wire_bits([("w", 3)]) == 1 + 9 + 27 + 1


@pytest.mark.software
def test_a_transfer_holds_its_messages_data_times_and_result():
    (t,) = it.assemble(it.parse_events(trace(imu_read(5.0, dur_us=1800.0))))
    assert (t["start_s"], t["bus"], t["addr"], t["device"], t["task"]) == (5.0, 1, 0x68, "MPU-6050 IMU", "sensor-hub")
    assert (t["msgs"], t["write_bytes"], t["read_bytes"], t["wdata"]) == ("w1 r14", 1, 14, "3b")
    assert t["rdata"] == "".join(f"{b:02x}" for b in range(8)) + "".join(f"{b:02x}" for b in range(8, 14))[:12]
    assert (t["dur_us"], t["wire_us"], t["ret"], t["ok"]) == (1800.0, 1560.0, 2, 1)
    assert list(t) == list(it.TRANSFER_FIELDS)
    (fast,) = it.assemble(it.parse_events(trace(imu_read(5.0))), clocks={1: 400_000})
    assert fast["wire_us"] == 390.0


@pytest.mark.software
def test_failed_cut_off_and_interleaved_transfers():
    text = trace(
        [ev(1.0, "i2c_reply", "#1 a=068 f=0001 l=14 [00]"), ev(1.0, "i2c_result", "n=2 ret=2")],  # cut at the start
        imu_read(2.0, ret=-121),
        [ev(3.0, "i2c_write", "#0 a=01a f=0000 l=3 [30-09-10]", task="ipa", bus=10)],
        imu_read(3.0001),                                                                        # bus 1 inside bus 10's
        [ev(3.0005, "i2c_result", "n=1 ret=1", task="ipa", bus=10)],
        [ev(9.0, "i2c_write", "#0 a=068 f=0000 l=1 [3b]")])                                     # cut at the end
    ts = it.assemble(it.parse_events(text))
    assert [(t["bus"], t["addr"], t["ok"]) for t in ts] == [(1, 0x68, 0), (1, 0x68, 1), (10, 0x1A, 1)]
    cam = ts[2]
    assert cam["device"] == "IMX290 camera" and cam["msgs"] == "w3" and cam["dur_us"] == 500.0


@pytest.mark.software
def test_a_partial_result_is_a_failure():
    (t,) = it.assemble(it.parse_events(trace(imu_read(1.0, ret=1))))        # 1 of the 2 messages went through
    assert (t["ret"], t["ok"]) == (1, 0)


@pytest.mark.software
def test_a_new_first_message_starts_a_new_transfer():
    text = trace([ev(1.0, "i2c_write", "#0 a=068 f=0000 l=1 [3b]")], imu_read(1.5))   # the first never closed
    (t,) = it.assemble(it.parse_events(text))
    assert t["start_s"] == 1.5


# =============================================================================
# The system
# =============================================================================

@pytest.mark.software
def test_bus_clocks_from_the_device_tree_or_assumed(tmp_path):
    a = tmp_path / "i2c-adapter"
    (a / "i2c-1" / "of_node").mkdir(parents=True)
    (a / "i2c-1" / "name").write_text("bcm2835 (i2c@7e804000)\n")
    (a / "i2c-1" / "of_node" / "clock-frequency").write_bytes((400_000).to_bytes(4, "big"))
    (a / "i2c-10").mkdir()
    (a / "i2c-x").mkdir()
    info = it.bus_info(a)
    assert info[1] == {"name": "bcm2835 (i2c@7e804000)", "clock_hz": 400_000, "clock_from": "device tree"}
    assert info[10] == {"name": "", "clock_hz": it.DEFAULT_CLOCK_HZ, "clock_from": "assumed"}
    assert set(info) == {1, 10} and it.bus_info(tmp_path / "missing") == {}
    (a / "of_node").mkdir()                                             # a mux child's parent node
    (a / "of_node" / "clock-frequency").write_bytes((100_000).to_bytes(4, "big"))
    assert it.bus_info(a)[10]["clock_from"] == "device tree"


def fake_tracefs(tmp_path, clocks="[local] global counter uptime perf mono mono_raw boot"):
    t = tmp_path / "tracing"
    for ev_ in it.EVENTS:
        (t / "events" / "i2c" / ev_).mkdir(parents=True)
        (t / "events" / "i2c" / ev_ / "enable").write_text("0\n")
    (t / "tracing_on").write_text("1\n")
    (t / "buffer_size_kb").write_text("1408\n")
    (t / "trace_clock").write_text(clocks + "\n")
    (t / "trace").write_text("old data\n")
    for cpu, over in ((0, 2), (1, 3)):
        (t / "per_cpu" / f"cpu{cpu}").mkdir(parents=True)
        (t / "per_cpu" / f"cpu{cpu}" / "stats").write_text(f"entries: 10\noverrun: {over}\ncommit overrun: 0\n")
    return t


@pytest.mark.software
def test_record_trace_turns_the_events_on_for_the_window_and_puts_everything_back(tmp_path):
    t = fake_tracefs(tmp_path)
    seen = {}

    def kernel(seconds):                                               # what the kernel does while we wait
        seen["during"] = {k: (t / k).read_text() for k in ("tracing_on", "buffer_size_kb", "trace_clock", "trace")}
        seen["enabled"] = [(t / "events" / "i2c" / e / "enable").read_text() for e in it.EVENTS]
        (t / "trace").write_text(trace(imu_read(1.0)))
    clock = iter([100.0, 102.5])
    text, window, lost, used = it.record_trace(t, 2.5, sleep=kernel, clock=lambda: next(clock))
    assert seen["during"] == {"tracing_on": "1", "buffer_size_kb": str(it.BUFFER_KB), "trace_clock": "mono",
                              "trace": ""}
    assert seen["enabled"] == ["1"] * 4
    assert "i2c_write" in text and window == 2.5 and lost == 5 and used == "mono"
    assert [(t / k).read_text() for k in ("tracing_on", "buffer_size_kb", "trace_clock")] == ["1", "1408", "local"]
    assert [(t / "events" / "i2c" / e / "enable").read_text() for e in it.EVENTS] == ["0"] * 4


@pytest.mark.software
def test_record_trace_keeps_the_clock_without_mono_and_restores_on_an_error(tmp_path):
    t = fake_tracefs(tmp_path, clocks="local [global] counter")

    def boom(seconds):
        assert (t / "trace_clock").read_text() == "local [global] counter\n"     # never written: no mono
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        it.record_trace(t, 1.0, sleep=boom)
    assert (t / "trace_clock").read_text() == "global" and (t / "tracing_on").read_text() == "1"
    assert all((t / "events" / "i2c" / e / "enable").read_text() == "0" for e in it.EVENTS)
    assert it.find_tracefs([tmp_path / "nope", t]) == t and it.find_tracefs([tmp_path / "nope"]) is None


# =============================================================================
# Analysis
# =============================================================================

FAST_US = 600.0         # an IMU read at 400 kHz: 1.5x its 390 us wire time, under every threshold


def robot_trace(n=100, imu_dur=1800.0, ads_at=None, gap_ms=10.0, imu_ret=2, ads_dur=500.0):
    """n IMU reads every gap_ms from t=1 s; an ADS1115 write at ads_at (s), the next IMU read right after it."""
    groups = [imu_read(1.0 + i * gap_ms / 1000, dur_us=imu_dur, ret=imu_ret if i == 3 else 2) for i in range(n)]
    if ads_at is not None:
        groups.append(ads_write(ads_at, ads_dur))
    lines = [line for g in groups for line in g]
    lines.sort(key=lambda s: float(s.split(": ")[0].split()[-1]))
    return "\n".join(lines) + "\n"


def analyzed(text, window=1.0, clock=100_000, lost=0):
    buses = {1: {"name": "bcm2835", "clock_hz": clock, "clock_from": "device tree"}}
    return it.analyze(it.assemble(it.parse_events(text), {1: clock}), window, buses, lost)


@pytest.mark.software
def test_occupancy_wire_time_and_per_device_numbers():
    # the ADS1115 write at 1.04945 s ends 50 us before the IMU read at 1.0500 s: it queued behind it
    res = analyzed(robot_trace(n=100, imu_dur=1800.0, ads_at=1.04945, ads_dur=500.0), window=1.0)
    (b,) = res["buses"]
    assert b["transfers"] == 101 and b["occupancy_pct"] == pytest.approx(18.05)     # 100 x 1.8 ms + 0.5 ms over 1 s
    assert b["wire_pct"] == pytest.approx(15.64, abs=0.01)                       # 100 x 1560 us + the write's 380 us
    assert b["overhead"] == pytest.approx(1.15, abs=0.01)
    imu, ads = sorted(res["devices"], key=lambda d: -d["transfers"])
    assert (imu["device"], imu["rate_hz"], imu["bytes_s"], imu["pattern"]) == ("MPU-6050 IMU", 100.0, 1500.0, "w1 r14")
    assert imu["tasks"] == ["sensor-hub"] and ads["tasks"] == ["python3"]
    assert imu["dur_us"] == {"median": 1800.0, "p95": 1800.0, "max": 1800.0}
    assert imu["gap_ms"]["median"] == 10.0 and imu["queued"] == 1 and ads["queued"] == 0
    assert res["findings"] == ["1 IMU reads started right after another device's transfer: they queued behind it on "
                               "the bus", "i2c-1 runs at 100 kHz (device tree) and its bits alone take 16% of the "
                               "time: dtparam=i2c_arm_baudrate=400000 in /boot/firmware/config.txt (both devices "
                               "are rated for it) cuts that 4x"]


@pytest.mark.software
def test_a_transfer_starting_long_after_another_device_did_not_queue():
    res = analyzed(robot_trace(n=10, ads_at=1.0300, ads_dur=500.0))                # ends 9.5 ms before the next read
    assert next(d for d in res["devices"] if d["addr"] == 0x68)["queued"] == 0
    own = analyzed(trace(imu_read(1.0, dur_us=1000.0), imu_read(1.00102, dur_us=1000.0)))   # 20 us after its own
    assert own["devices"][0]["queued"] == 0


@pytest.mark.software
def test_rates_are_per_second_of_the_window_and_the_pattern_is_the_commonest():
    ads = trace(ads_write(1.0), *[[ev(1.01 + k / 100, "i2c_write", "#0 a=048 f=0000 l=1 [01]", task="python3"),
                                   ev(1.01 + k / 100, "i2c_read", "#1 a=048 f=0001 l=2", task="python3"),
                                   ev(1.0105 + k / 100, "i2c_result", "n=2 ret=2", task="python3")] for k in range(3)])
    (d,) = analyzed(ads, window=0.5, clock=400_000)["devices"]
    assert (d["transfers"], d["rate_hz"], d["pattern"]) == (4, 8.0, "w1 r2")


@pytest.mark.software
@pytest.mark.parametrize("kw, words", [                                # at 400 kHz an IMU read's wire time is 390 us
    ({"imu_dur": 6000.0}, "i2c-1 was busy 60% of the time"),
    ({"imu_dur": 820.0}, "hold the bus 2.1x their wire time"),
    ({"imu_ret": -121}, "1 failed transfers to MPU-6050 IMU (i2c-1): EREMOTEIO (no ACK"),
    ({"gap_ms": 15.5}, "IMU reads were spaced up to 15.5 ms (p95), not 10"),
])
def test_each_finding_on_its_own(kw, words):
    found = analyzed(robot_trace(**{"imu_dur": FAST_US, **kw}), window=1.0, clock=400_000)["findings"]
    assert any(words in f for f in found), found


@pytest.mark.software
def test_findings_sit_at_their_thresholds():
    # busy at exactly 50%, overhead exactly 2x at 400 kHz (390 us wire), gaps exactly 1.5 periods
    at = analyzed(robot_trace(n=100, imu_dur=5000.0), window=1.0, clock=400_000)["findings"]
    assert any("busy 50%" in f for f in at)
    under = analyzed(robot_trace(n=100, imu_dur=4990.0), window=1.0, clock=400_000)["findings"]
    assert not any("busy" in f for f in under)
    two = analyzed(robot_trace(n=10, imu_dur=780.0), clock=400_000)["findings"]
    assert any("2.0x their wire time" in f for f in two)
    slip = analyzed(robot_trace(n=10, gap_ms=15.0, imu_dur=FAST_US), clock=400_000)["findings"]
    assert slip == []
    wire = analyzed(robot_trace(n=64), window=1.0, clock=100_000)["findings"]          # 9.98% wire
    assert not any("kHz" in f for f in wire)
    fast = analyzed(robot_trace(n=100, imu_dur=4000.0), window=0.03, clock=400_000)   # far over 10% wire, at 400 kHz
    assert not any("dtparam" in f for f in fast["findings"])
    assert analyzed(robot_trace(n=10, imu_dur=FAST_US), clock=400_000)["findings"] == []


@pytest.mark.software
def test_an_empty_or_overflowed_trace_says_so():
    assert analyzed(trace())["findings"] == ["no I2C transfer in the trace: was a run going (make nav-dry in "
                                             "another terminal)?"]
    lost = analyzed(robot_trace(n=10, imu_dur=FAST_US), clock=400_000, lost=7)["findings"]
    assert lost == ["the trace buffer overflowed: 7 events lost; trace fewer seconds"]


@pytest.mark.software
def test_summary_lists_buses_devices_and_findings():
    res = analyzed(robot_trace(n=100, ads_at=1.04945), window=1.0)
    text = "\n".join(it.summary_lines(res, {"started": "t", "trace_clock": "mono"}))
    assert "1.0 s, 101 transfers  (trace clock mono)" in text
    assert "i2c-1   bcm2835" in text and "100 kHz (device tree)  occupancy  18.1%  wire  15.6%  overhead 1.1x" in text
    assert "MPU-6050 IMU (i2c-1 0x68)  by sensor-hub" in text and "100 transfers, 100.0/s, 1500 B/s, mostly w1 r14" in text
    assert "gap 10.00 / 10.00 / 10.00" in text and "queued behind another device: 1" in text
    assert "ADS1115 battery (i2c-1 0x48)  by python3" in text and "gap --" in text
    assert "  none" in "\n".join(it.summary_lines(analyzed(trace())))


# =============================================================================
# Figure, folder and command line
# =============================================================================

@pytest.mark.software
def test_figure_draws_and_skips_without_matplotlib_or_transfers(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    text = robot_trace(n=50, ads_at=1.0145)
    ts = it.assemble(it.parse_events(text))
    res = it.analyze(ts, 1.0)
    png = it.figure(ts, res, tmp_path / "f.png")
    assert png.exists() and png.stat().st_size > 10_000
    assert it.figure([], it.analyze([], 1.0), tmp_path / "g.png") is None
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda name, *a, **k: (_ for _ in ()).throw(ImportError(name)) if name == "matplotlib"
                        else real(name, *a, **k))
    assert it.figure(ts, res, tmp_path / "h.png") is None


@pytest.mark.software
def test_write_holds_every_file_and_from_re_analyzes_without_root(tmp_path, monkeypatch, capsys):
    buses = {1: {"name": "bcm2835", "clock_hz": 400_000, "clock_from": "device tree"}}
    out = tmp_path / "rec"
    res = it.write(out, robot_trace(n=20), 0.5, 0, buses, {"started": "t"}, draw=False)
    assert res["transfers"] == 20
    with open(out / "transactions.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 and rows[0]["wire_us"] == "390.0"
    meta = json.loads((out / "i2c_trace.json").read_text())["meta"]
    assert meta["window_s"] == 0.5 and meta["buses"]["1"]["clock_hz"] == 400_000 and meta["figure"] is None
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    again = tmp_path / "again"
    assert it.cli(["--from", str(out), "--out", str(again), "--no-figure"]) == 0
    assert json.loads((again / "i2c_trace.json").read_text())["buses"] == res["buses"]
    assert "MPU-6050 IMU" in capsys.readouterr().out
    assert it.cli(["--from", str(tmp_path / "missing")]) == 2


@pytest.mark.software
def test_the_figure_note_appears_when_matplotlib_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "figure", lambda *a: None)
    it.write(tmp_path / "w", robot_trace(n=5), 0.1, 0, {}, {})
    assert "no matplotlib here" in (tmp_path / "w" / "summary.txt").read_text()


@pytest.mark.software
def test_cli_needs_root_and_tracefs_then_records_and_hands_the_folder_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert it.cli([]) == 2 and "needs root" in capsys.readouterr().out
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(it, "find_tracefs", lambda: None)
    assert it.cli([]) == 2 and "no tracefs" in capsys.readouterr().out
    monkeypatch.setattr(it, "find_tracefs", lambda: tmp_path)
    monkeypatch.setattr(it, "record_trace", lambda tf, s: (robot_trace(n=10), s, 0, "mono"))
    monkeypatch.setattr(it, "bus_info", lambda: {1: {"name": "x", "clock_hz": 400_000, "clock_from": "device tree"}})
    chowned = []
    monkeypatch.setattr(os, "chown", lambda p, u, g: chowned.append((Path(p).name, u, g)))
    monkeypatch.setenv("SUDO_UID", "1000")
    monkeypatch.setenv("SUDO_GID", "1001")
    out = tmp_path / "rec"
    assert it.cli(["--seconds", "2", "--out", str(out), "--no-figure"]) == 0
    assert json.loads((out / "i2c_trace.json").read_text())["meta"]["trace_clock"] == "mono"
    assert ("rec", 1000, 1001) in chowned and ("summary.txt", 1000, 1001) in chowned
    monkeypatch.setattr(it, "record_trace", lambda tf, s: (trace(), s, 0, "local"))
    assert it.cli(["--out", str(tmp_path / "empty"), "--no-figure"]) == 1             # nothing traced


# =============================================================================
# On the Pi
# =============================================================================

@pytest.mark.hardware
def test_i2c_trace_on_the_pi(artifacts):
    if os.geteuid() != 0:
        pytest.skip("tracing needs root (sudo)")
    tracefs = it.find_tracefs()
    if tracefs is None:
        pytest.skip("no tracefs with i2c events")
    text, window, lost, clock = it.record_trace(tracefs, 2.0)
    res = it.write(artifacts.path / "i2c", text, window, lost, it.bus_info(), {"trace_clock": clock}, draw=False)
    assert window >= 2.0 and res["buses"] is not None
