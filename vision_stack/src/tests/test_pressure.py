"""
test_pressure.py  --  src/diagnostics/pressure.py

PSI and meminfo read from a fake /proc tree, so every stall is exact: the
totals' differences across a gap, the gap's split into the loop's work, its
read and the rest, the start-up gap skipped, and a tree without PSI (the Pi
kernel's default) still writing the timings.

--software  Fake /proc trees only.
"""
import csv

import pytest

from src.diagnostics.pressure import STALL_FIELDS, Pressure, StallLog, meminfo


def psi(some, full=None):
    text = f"some avg10=0.00 avg60=0.00 avg300=0.00 total={some}\n"
    if full is not None:
        text += f"full avg10=0.00 avg60=0.00 avg300=0.00 total={full}\n"
    return text


def proc(tmp_path, cpu=(0, 0), io=(0, 0), memory=(0, 0), dirty=1200, writeback=64, available=180_000, psi_on=True):
    """A fake /proc with pressure/ (cpu, io, memory: (some, full) totals in us) and meminfo."""
    root = tmp_path / "proc"
    (root / "pressure").mkdir(parents=True, exist_ok=True)
    if psi_on:
        for name, (some, full) in (("cpu", cpu), ("io", io), ("memory", memory)):
            (root / "pressure" / name).write_text(psi(some, full))
    (root / "meminfo").write_text(f"MemTotal:  430000 kB\nMemAvailable:  {available} kB\n"
                                  f"Dirty:  {dirty} kB\nWriteback:  {writeback} kB\n")
    return root


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.mark.software
def test_totals_reads_each_line_of_each_resource(tmp_path):
    root = proc(tmp_path, cpu=(1500, 0), io=(2000, 900), memory=(40, 10))
    p = Pressure(root)
    assert p.on
    assert Pressure.totals(p.read()) == {"cpu_some": 1500, "cpu_full": 0, "io_some": 2000, "io_full": 900,
                                         "memory_some": 40, "memory_full": 10}
    p.close()


@pytest.mark.software
def test_reads_follow_the_file_without_reopening(tmp_path):
    root = proc(tmp_path, io=(100, 0))
    p = Pressure(root)
    (root / "pressure" / "io").write_text(psi(5100, 300))
    assert Pressure.totals(p.read())["io_some"] == 5100
    p.close()


@pytest.mark.software
def test_meminfo_dirty_writeback_and_available(tmp_path):
    assert meminfo(proc(tmp_path)) == {"dirty_kb": 1200, "writeback_kb": 64, "available_mb": 175.8}
    assert meminfo(tmp_path / "nowhere") == {"dirty_kb": None, "writeback_kb": None, "available_mb": None}


@pytest.mark.software
def test_a_stall_row_splits_the_gap_and_says_what_the_system_waited_on(tmp_path):
    root = proc(tmp_path, io=(1_000, 500), cpu=(2_000, 0), memory=(0, 0))
    log = StallLog(tmp_path, pressure=Pressure(root), proc=root)
    assert not log.frame(0, 0.00, 5.0)                       # start-up frame
    log.worked(1400.0)
    assert not log.frame(1, 1.45, 4.0)                       # its gap is start-up: never a stall
    log.worked(45.0)
    assert not log.frame(2, 1.50, 5.0)
    log.worked(44.0)
    (root / "pressure" / "io").write_text(psi(421_000, 380_500))     # 420 ms some, 380 full during the gap
    (root / "pressure" / "cpu").write_text(psi(32_000, 0))
    assert log.frame(3, 2.10, 6.0, imu_reads=58, imu_read_ms=1.8)
    log.close()
    (row,) = rows(tmp_path / "stalls.csv")
    assert list(row) == list(STALL_FIELDS)
    assert (row["frame_id"], row["gap_ms"], row["work_ms"], row["capture_ms"], row["outside_ms"]) == \
        ("3", "600.0", "44.0", "6.0", "550.0")
    assert (row["io_some_ms"], row["io_full_ms"], row["cpu_some_ms"], row["memory_some_ms"]) == \
        ("420.0", "380.0", "30.0", "0.0")
    assert (row["dirty_kb"], row["writeback_kb"], row["imu_reads"], row["imu_read_ms"]) == ("1200", "64", "58", "1.8")
    assert log.count == 1 and log.psi_on


@pytest.mark.software
def test_without_psi_the_timings_and_meminfo_are_still_written(tmp_path):
    root = proc(tmp_path, psi_on=False)
    log = StallLog(tmp_path, pressure=Pressure(root), proc=root)
    for fid, t in enumerate((0.0, 0.05, 0.10, 0.50)):
        log.frame(fid, t, 2.0)
        log.worked(40.0)
    log.close()
    (row,) = rows(tmp_path / "stalls.csv")
    assert not log.psi_on and row["io_some_ms"] == "" and row["dirty_kb"] == "1200"
    assert float(row["gap_ms"]) == pytest.approx(400.0)


@pytest.mark.software
def test_no_stall_no_file(tmp_path):
    log = StallLog(tmp_path / "run", pressure=Pressure(proc(tmp_path)), proc=proc(tmp_path))
    for fid in range(10):
        log.frame(fid, fid * 0.05, 1.0)
    log.close()
    assert log.count == 0 and not (tmp_path / "run" / "stalls.csv").exists()
