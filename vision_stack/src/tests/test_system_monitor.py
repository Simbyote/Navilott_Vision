"""
test_system_monitor.py  --  src/debugger/system_monitor.py

--software  Each reading against fake /sys and /proc files, the throttle
            bit decoding, vcgencmd absent or failing, and the monitor's
            thread, final sample and row shape with an injected reader.
--hardware  One real sample on the Pi: temperature, clock and memory must
            read (throttle flags need vcgencmd, which a desktop lacks).
"""
import subprocess
import time
from types import SimpleNamespace

import pytest

from src.debugger import system_monitor as sm


@pytest.mark.software
def test_temperature_is_millidegrees(tmp_path):
    (tmp_path / "temp").write_text("48312\n")
    assert sm.read_temp_c(tmp_path / "temp") == pytest.approx(48.312)


@pytest.mark.software
def test_cpu_clock_is_khz(tmp_path):
    (tmp_path / "f").write_text("1000000\n")
    assert sm.read_cpu_mhz(tmp_path / "f") == 1000.0


@pytest.mark.software
def test_missing_or_garbage_files_read_as_none(tmp_path):
    (tmp_path / "bad").write_text("n/a")
    assert sm.read_temp_c(tmp_path / "missing") is None
    assert sm.read_cpu_mhz(tmp_path / "bad") is None


@pytest.mark.software
def test_rss_and_mem_available_from_proc_files(tmp_path):
    (tmp_path / "status").write_text("Name:\tpython3\nVmRSS:\t  102400 kB\nThreads:\t4\n")
    (tmp_path / "meminfo").write_text("MemTotal: 512000 kB\nMemAvailable:  204800 kB\n")
    assert sm.read_rss_mb(tmp_path / "status") == 100.0
    assert sm.read_mem_available_mb(tmp_path / "meminfo") == 200.0
    assert sm.read_rss_mb(tmp_path / "meminfo") is None


@pytest.mark.software
@pytest.mark.parametrize("text, want", [("throttled=0x50005\n", 0x50005), ("throttled=0x0", 0),
                                        ("error", None), ("volts=0x1", None), ("throttled=zz", None)])
def test_parse_throttled(text, want):
    assert sm.parse_throttled(text) == want


@pytest.mark.software
def test_decode_throttled_splits_now_and_since_boot():
    flags = sm.decode_throttled(0x50005)          # under-voltage + throttled now; both since boot
    assert flags["under_voltage"] == 1 and flags["throttled"] == 1
    assert flags["freq_capped"] == 0 and flags["soft_temp_limit"] == 0
    assert flags["under_voltage_occurred"] == 1 and flags["throttled_occurred"] == 1
    assert flags["freq_capped_occurred"] == 0


@pytest.mark.software
def test_decode_none_is_all_none():
    assert set(sm.decode_throttled(None).values()) == {None}


@pytest.mark.software
def test_read_throttled_without_vcgencmd_is_none():
    def missing(*a, **k):
        raise FileNotFoundError("vcgencmd")
    assert sm.read_throttled(run=missing) is None


@pytest.mark.software
def test_read_throttled_handles_failure_and_success():
    fail = lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
    ok = lambda *a, **k: SimpleNamespace(returncode=0, stdout="throttled=0x4\n")
    timeout = lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("vcgencmd", 1))
    assert sm.read_throttled(run=fail) is None
    assert sm.read_throttled(run=ok) == 4
    assert sm.read_throttled(run=timeout) is None


@pytest.mark.software
def test_sample_has_every_field_but_elapsed():
    assert set(sm.sample()) == set(sm.FIELDS) - {"elapsed_s"}


@pytest.mark.software
def test_monitor_samples_in_the_background_and_once_more_at_stop():
    calls = []
    def reader():
        calls.append(1)
        return {"temp_c": 50.0, "extra": "dropped"}
    with sm.SystemMonitor(interval_s=0.02, reader=reader) as mon:
        time.sleep(0.15)
    rows = mon.rows()
    assert len(rows) >= 4 and len(rows) == len(calls)
    assert list(rows[0]) == list(sm.FIELDS)          # fixed column order, unknown keys dropped
    assert rows[0]["temp_c"] == 50.0 and rows[0]["rss_mb"] is None
    assert all(b["elapsed_s"] >= a["elapsed_s"] for a, b in zip(rows, rows[1:]))


@pytest.mark.software
def test_rows_is_a_copy():
    mon = sm.SystemMonitor(interval_s=10, reader=dict).start()
    mon.stop()
    mon.rows().clear()
    assert mon.rows()


@pytest.mark.hardware
def test_a_real_sample_reads_on_the_pi():
    s = sm.sample()
    missing = [k for k in ("temp_c", "cpu_mhz", "rss_mb", "mem_available_mb") if s[k] is None]
    if missing and s["throttled_raw"] is None:
        pytest.skip(f"not a Raspberry Pi (no {', '.join(missing)} and no vcgencmd)")
    assert not missing, f"unreadable on this Pi: {missing}"
