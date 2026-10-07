"""
test_os_counters.py  --  src/diagnostics/os_counters.py

Each reader on a fake /proc shaped like the Pi's (an ARM /proc/interrupts
with Unicam, VCHIQ and the SD host; diskstats; PSI; meminfo; per-process
stat), and on x86's interrupt naming; vcgencmd's clock reply. OsCounters
over two takes on a fake clock: interrupt rates, SD read / write / busy,
other processes' CPU with the robot left out and the recorder named, the
clocks only every CLOCK_EVERY_S, nothing rated on the first take, a missing
source as None. The summaries averaging over every interval, and the
summary lines with and without each source.

--software  A fake /proc in a temp folder. No Pi needed.
--hardware  Two takes of the real counters on the Pi.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.diagnostics.os_counters as osc

INTERRUPTS = """\
           CPU0       CPU1       CPU2       CPU3
 17:      {mbox}          0          0          0  ARMCTRL-level   1 Edge      3f00b880.mailbox
 25:      {vchiq}          0          0          0  ARMCTRL-level  17 Edge      VCHIQ doorbell
 41:      {unicam}         {unicam}          0          0  ARMCTRL-level  41 Level     unicam
 54:        100          0          0          0  ARMCTRL-level  54 Level     mmc0
 60:          0          0          0          0  ARMCTRL-level  60 Edge      never-fires
IPI0:        10         20         30         40  Rescheduling interrupts
Err:          0
"""


def fake_proc(tmp_path, t=0, pids=None, psi=True, cma_free=200704):
    """A /proc at counter time t: every counter grows with t."""
    p = tmp_path / "proc"
    p.mkdir(exist_ok=True)
    (p / "interrupts").write_text(INTERRUPTS.format(mbox=1000 + 5 * t, vchiq=500 + 40 * t, unicam=100 + 20 * t))
    (p / "diskstats").write_text(
        "   1       0 ram0 0 0 0 0 0 0 0 0 0 0 0\n"
        f" 179       0 mmcblk0 {10 + t} 0 {1000 + 200 * t} 5 {20 + t} 0 {2000 + 1024 * t} 9 0 {300 + 250 * t} 400\n")
    (p / "meminfo").write_text(f"MemTotal: 427000 kB\nCmaTotal: 262144 kB\nCmaFree: {cma_free} kB\n")
    if psi:
        (p / "pressure").mkdir(exist_ok=True)
        for kind, v in (("cpu", 1.5), ("io", 12.0), ("memory", 0.0)):
            (p / "pressure" / kind).write_text(f"some avg10={v:.2f} avg60=0.00 avg300=0.00 total=1\n"
                                               "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    for pid, (name, ticks_per_t) in (pids or {}).items():
        d = p / str(pid)
        d.mkdir(exist_ok=True)
        ticks = ticks_per_t * t
        # user ticks, then system ticks: a third of the total
        (d / "stat").write_text(f"{pid} ({name}) S 1 1 1 0 -1 4194560 0 0 0 0 {ticks - ticks // 3} {ticks // 3} "
                                "0 0 20 0 1 0 5 0 0\n")
    return p


# =============================================================================
# Readers
# =============================================================================

@pytest.mark.software
def test_interrupts_sum_the_cores_and_name_the_device(tmp_path):
    irqs = osc.read_interrupts(fake_proc(tmp_path, t=1) / "interrupts")
    assert irqs["41"] == ("unicam", 240)                                     # 120 on each of two cores
    assert irqs["25"] == ("VCHIQ doorbell", 540) and irqs["17"] == ("3f00b880.mailbox", 1005)
    assert irqs["IPI0"] == ("Rescheduling interrupts", 100) and irqs["Err"] == ("", 0)
    assert osc.read_interrupts(tmp_path / "missing") == {}
    two = tmp_path / "two"
    two.write_text("    CPU0  CPU1\n 99:   1   2   7 Edge  dev\n")                # a description opening with a number
    assert osc.read_interrupts(two) == {"99": ("dev", 3)}


@pytest.mark.software
@pytest.mark.parametrize("desc, name", [
    ("ARMCTRL-level  65 Level  unicam", "unicam"),
    ("IO-APIC   5-edge      ACPI:Ged", "ACPI:Ged"),
    ("PCI-MSIX-0000:00:08.0 1-edge virtio1-input.0", "virtio1-input.0"),
    ("Rescheduling interrupts", "Rescheduling interrupts"),
    ("ARMCTRL-level", "ARMCTRL-level"),                                    # a chip name, not a trigger
    ("GICv2  30 Level", "GICv2  30 Level"),                                # nothing after the trigger: the whole text
    ("chip  7 Edge  bridge 9 Level  dev", "dev"),                          # the last trigger names the device
])
def test_irq_names_follow_the_trigger_type_on_the_pi_and_on_x86(desc, name):
    assert osc._irq_name(desc) == name


@pytest.mark.software
def test_disk_pressure_and_cma_readers(tmp_path):
    p = fake_proc(tmp_path, t=2)
    assert osc.read_disk(p / "diskstats") == {"sectors_read": 1400, "sectors_written": 4048, "io_ms": 800}
    assert osc.read_disk(p / "diskstats", "sda") is None and osc.read_disk(tmp_path / "missing") is None
    assert osc.read_pressure(p / "pressure" / "io") == 12.0
    assert osc.read_pressure(tmp_path / "missing") is None
    assert osc.read_cma_mb(p / "meminfo") == {"cma_total_mb": 256.0, "cma_free_mb": 196.0}


@pytest.mark.software
def test_process_ticks_read_names_with_spaces_and_leave_out_the_excluded(tmp_path):
    p = fake_proc(tmp_path, t=3, pids={10: ("pigpiod", 4), 11: ("my (odd) name", 1), 12: ("python3", 9)})
    (p / "13").mkdir()                                                       # gone before its stat was read
    assert osc.read_process_ticks(p, exclude=(12,)) == {10: ("pigpiod", 12), 11: ("my (odd) name", 3)}   # user + system


@pytest.mark.software
def test_vc_clock_in_mhz_and_none_without_vcgencmd():
    ok = lambda argv, **kw: SimpleNamespace(returncode=0, stdout="frequency(45)=299999000\n")     # noqa: E731
    assert osc.read_vc_clock_mhz("isp", run=ok) == 300.0
    bad = lambda argv, **kw: SimpleNamespace(returncode=255, stdout="error=1\n")                  # noqa: E731
    assert osc.read_vc_clock_mhz("isp", run=bad) is None

    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])
    assert osc.read_vc_clock_mhz("isp", run=missing) is None

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)
    assert osc.read_vc_clock_mhz("isp", run=slow) is None


# =============================================================================
# OsCounters
# =============================================================================

class Clocks:
    def __init__(self):
        self.calls = []

    def __call__(self, name):
        self.calls.append(name)
        return {"isp": 300.0, "core": 400.0}[name]


def counters(tmp_path, clocks=None, **kw):
    pids = {10: ("pigpiod", 6), 12: ("python3", 50), 99: ("python3", 3), 13: ("sleepy", 0)}
    c = osc.OsCounters(target_pid=12, proc=tmp_path / "proc", clock_reader=clocks or Clocks(), self_pid=99, **kw)
    return c, pids


@pytest.mark.software
def test_the_first_take_reads_levels_and_rates_nothing(tmp_path):
    c, pids = counters(tmp_path)
    fake_proc(tmp_path, t=0, pids=pids)
    first = c.take(0.0, 100.0)
    assert first["irqs"] == [] and first["procs"] == []
    s = first["system"]
    assert s["os_dt_s"] is None and s["disk_write_kbps"] is None
    assert (s["cma_free_mb"], s["isp_mhz"], s["core_mhz"]) == (196.0, 300.0, 400.0)
    assert (s["psi_cpu_some"], s["psi_io_some"], s["psi_memory_some"]) == (1.5, 12.0, 0.0)
    assert list(s) == list(osc.SYSTEM_FIELDS)


@pytest.mark.software
def test_rates_over_the_time_since_the_last_take(tmp_path):
    c, pids = counters(tmp_path)
    fake_proc(tmp_path, t=0, pids=pids)
    c.take(0.0, 100.0)
    fake_proc(tmp_path, t=2, pids=pids)                                      # 2 counter steps over 2 s
    out = c.take(2.0, 102.0)
    rates = {r["name"]: r["rate_hz"] for r in out["irqs"]}
    assert rates == {"unicam": 40.0, "VCHIQ doorbell": 40.0, "3f00b880.mailbox": 5.0}   # still ones not listed
    assert all(r["elapsed_s"] == 2.0 for r in out["irqs"])
    s = out["system"]
    assert s["os_dt_s"] == 2.0
    assert (s["disk_read_kbps"], s["disk_write_kbps"], s["disk_busy_pct"]) == (100.0, 512.0, 25.0)
    procs = {r["name"]: r["cpu_pct"] for r in out["procs"]}
    assert procs == {"pigpiod": 6.0 * 100 / osc.CLK_TCK, osc.SELF_NAME: 3.0 * 100 / osc.CLK_TCK}   # user + system
    assert not any(r["name"] == "sleepy" for r in out["procs"])             # didn't run: no row
    assert not any(r["pid"] == 12 for r in out["procs"])                    # the robot: threads.csv has it


@pytest.mark.software
def test_the_disk_is_never_over_100_percent_busy(tmp_path):
    c, pids = counters(tmp_path)
    fake_proc(tmp_path, t=0)
    c.take(0.0, 0.0)
    fake_proc(tmp_path, t=10)                                                # 2500 ms of I/O in 1 s (queued)
    assert c.take(1.0, 1.0)["system"]["disk_busy_pct"] == 100.0


@pytest.mark.software
def test_clocks_are_read_only_every_clock_every_s(tmp_path):
    clocks = Clocks()
    c, pids = counters(tmp_path, clocks)
    fake_proc(tmp_path, t=0)
    got = [c.take(float(i), float(i))["system"]["isp_mhz"] for i in range(11)]
    every = int(osc.CLOCK_EVERY_S)
    assert got == [300.0 if i % every == 0 else None for i in range(11)]
    assert clocks.calls.count("isp") == clocks.calls.count("core") == 11 // every + 1


@pytest.mark.software
def test_a_laptop_without_the_sources_reads_none(tmp_path):
    (tmp_path / "proc").mkdir()
    c = osc.OsCounters(1, proc=tmp_path / "proc", clock_reader=lambda name: None)
    c.take(0.0, 0.0)
    out = c.take(1.0, 1.0)
    assert out["irqs"] == [] and out["procs"] == []
    assert {k: v for k, v in out["system"].items() if v is not None} == {"os_dt_s": 1.0}


# =============================================================================
# Summaries
# =============================================================================

def irq(t, name, hz, n="41"):
    return {"elapsed_s": t, "irq": n, "name": name, "rate_hz": hz}


@pytest.mark.software
def test_summaries_average_over_every_interval_counting_silent_ones_as_zero():
    rows = [irq(1, "unicam", 40.0), irq(2, "unicam", 20.0), irq(1, "mmc0", 90.0, "54")]
    s = osc.irq_summary(rows, 4)
    assert s == [{"irq": "41", "name": "unicam", "mean_hz": 15.0, "max_hz": 40.0},
                 {"irq": "54", "name": "mmc0", "mean_hz": 22.5, "max_hz": 90.0}][::-1]
    procs = [{"elapsed_s": 1, "pid": "7", "name": "pigpiod", "cpu_pct": "6.0"},
             {"elapsed_s": 2, "pid": "7", "name": "pigpiod", "cpu_pct": "4.0"}]
    assert osc.proc_summary(procs, 2) == [{"pid": 7, "name": "pigpiod", "mean_pct": 5.0, "max_pct": 6.0}]
    assert osc.proc_summary([], 0) == []
    assert osc.intervals([{"os_dt_s": None}, {"os_dt_s": 1.0}, {"os_dt_s": ""}, {"os_dt_s": "1.0"}]) == 2


@pytest.mark.software
def test_system_extremes_from_rows_or_csv_text():
    rows = [{"os_dt_s": "", "cma_free_mb": "196.0", "isp_mhz": "300", "core_mhz": "", "disk_write_kbps": ""},
            {"os_dt_s": "1", "cma_free_mb": "150.5", "isp_mhz": "", "core_mhz": "400", "disk_write_kbps": "10",
             "disk_read_kbps": "2", "disk_busy_pct": "5", "psi_io_some": "12.0"},
            {"os_dt_s": 1, "cma_free_mb": 180.0, "isp_mhz": 250.0, "disk_write_kbps": 30.0, "disk_busy_pct": 60.0,
             "psi_io_some": 3.0}]
    e = osc.system_extremes(rows)
    assert e["cma_free_min_mb"] == 150.5 and e["isp_mhz"] == (250.0, 300.0) and e["core_mhz"] == (400.0, 400.0)
    assert (e["disk_write_mean_kbps"], e["disk_write_max_kbps"], e["disk_read_max_kbps"]) == (20.0, 30.0, 2.0)
    assert e["disk_busy_max_pct"] == 60.0 and e["psi_io_max"] == 12.0 and e["psi_cpu_max"] is None
    assert osc.system_extremes([])["isp_mhz"] is None


@pytest.mark.software
def test_summary_lines_list_the_busiest_and_say_what_is_missing():
    system = [{"os_dt_s": None}, {"os_dt_s": 1.0, "cma_free_mb": 190.0, "isp_mhz": 300.0, "core_mhz": 400.0,
                                  "disk_write_kbps": 64.0, "disk_read_kbps": 0.0, "disk_busy_pct": 3.0,
                                  "psi_cpu_some": 1.5, "psi_io_some": 0.0, "psi_memory_some": 0.0}]
    irqs = [irq(1, f"dev{i}", float(i), str(i)) for i in range(12)]
    procs = [{"elapsed_s": 1, "pid": 7, "name": "pigpiod", "cpu_pct": 6.0}]
    text = "\n".join(osc.summary_lines(system, irqs, procs))
    assert "CMA free min 190 MB   clocks: ISP 300-300 MHz, core 400-400 MHz" in text
    assert "SD card: write mean 64 kB/s (max 64), read max 0 kB/s, busy max 3%" in text
    assert "cpu 1.5%, io 0.0%, memory 0.0%" in text
    assert text.count("dev") == osc.TOP_IRQS and "dev11" in text and "dev3 " not in text
    assert "pigpiod" in text and "mean   6.0" in text
    bare = "\n".join(osc.summary_lines([], [], []))
    assert "no PSI in this kernel" in bare and bare.count("none recorded") == 2 and "CMA free min --" in bare


# =============================================================================
# On the Pi
# =============================================================================

@pytest.mark.hardware
def test_os_counters_on_the_pi():
    if not Path("/proc/interrupts").exists():
        pytest.skip("no /proc")
    import time
    c = osc.OsCounters(None)
    c.take(0.0, time.perf_counter())
    time.sleep(1.0)
    out = c.take(1.0, time.perf_counter())
    assert out["irqs"], "no interrupt fired in a second"
    assert out["system"]["cma_free_mb"] is not None, "no CmaFree in /proc/meminfo"
