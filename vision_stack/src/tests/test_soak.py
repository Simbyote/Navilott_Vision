"""
test_soak.py  --  src/analysis/soak.py, and a long run of the whole chain with the Pi's condition recorded beside it

Not a stage test: it runs Phases 2 and 3 (run_phase3_chain, as
phase3_linker does, without its display or video) for a set duration while
SystemMonitor samples temperature, clock, throttle flags and memory once a
second. Only a long run shows throttling or a slow leak. src/analysis/soak.py
interprets the result.

--software  soak.analyze() on synthetic system and frame logs with known
            heat, throttling, memory growth and slowdown; the command line.
--hardware  Runs only with --soak-minutes=N (it would otherwise hold up
            every hardware run). Live camera, or --replay DIR played on a
            loop until the time is up; a replayed soak shows memory and heat
            but not camera timing. Writes soak_frames.csv, system.csv,
            summary.json and soak.png. Ctrl-C ends it early and still writes
            everything. Asserts only that frames and samples were recorded;
            throttling, memory growth and slowdown are reported as warnings.
"""
import csv
import time
import warnings

import numpy as np
import pytest

from src.analysis import soak
from src.analysis.common import Table
from src.diagnostics.system_monitor import FIELDS, SystemMonitor
from src.estimation.estimation import Phase3Processor
from src.config import MEASURED
from src.phase3_linker import run_phase3_chain

FRAME_FIELDS = ("n", "frame_id", "timestamp_ms", "elapsed_s", "capture_ms", "phase2_ms",
                "phase3_ms", "total_ms", "interval_ms")


def write_system(path, minutes=15, temp=(50, 70), rss_growth_per_min=0.0, throttle_from_min=None):
    """A SystemMonitor log, one sample a second, with the chosen trends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(minutes * 60) + 1
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for i in range(n):
            m = i / 60.0
            hot = throttle_from_min is not None and m >= throttle_from_min
            w.writerow([i, temp[0] + (temp[1] - temp[0]) * i / (n - 1), 600 if hot else 1000,
                        "0x20002" if hot else "0x0", 0, int(hot), 0, int(hot),
                        80 + rss_growth_per_min * m, 300, 1.0])
    return path


def write_frames(path, minutes=15, first_ms=40.0, last_ms=40.0):
    """soak_frames.csv at 20 FPS with the loop time moving linearly from first_ms to last_ms."""
    n = int(minutes * 60 * 20)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_FIELDS)
        for i in range(n):
            loop = first_ms + (last_ms - first_ms) * i / (n - 1)
            w.writerow([i, i, i * 50, i * 0.05, 5, 30, 0.3, 35.3, loop])
    return path


@pytest.mark.software
def test_a_calm_run_has_no_findings(tmp_path):
    r = soak.analyze(Table(write_system(tmp_path / "system.csv")), Table(write_frames(tmp_path / "f.csv")))
    assert r["findings"] == ["no throttling, memory growth or slowdown found"]
    assert r["temp_c"]["rise"] == pytest.approx(18.7, abs=0.1)     # last minute mean minus first
    assert r["memory"]["leak_mb_per_min"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.software
def test_throttling_is_reported_with_when_it_started(tmp_path):
    r = soak.analyze(Table(write_system(tmp_path / "system.csv", throttle_from_min=12)))
    assert r["throttle"]["freq_capped"]["first_s"] == pytest.approx(720)
    assert "soft_temp_limit" in r["throttle"] and "under_voltage" not in r["throttle"]
    assert r["cpu_mhz"]["share_below_max"] == pytest.approx(3 / 15, abs=0.01)
    assert r["findings"][0].startswith("throttling")


@pytest.mark.software
def test_steady_memory_growth_is_a_suspected_leak(tmp_path):
    r = soak.analyze(Table(write_system(tmp_path / "system.csv", rss_growth_per_min=1.0)))
    assert r["memory"]["leak_mb_per_min"] == pytest.approx(1.0, rel=1e-3)
    assert any("suspected leak" in f for f in r["findings"])


@pytest.mark.software
def test_warm_up_growth_is_not_a_leak(tmp_path):
    path = write_system(tmp_path / "system.csv")
    rows = list(csv.reader(open(path)))
    for row in rows[1:31]:                             # 20 MB of growth in the first 30 s only
        row[FIELDS.index("rss_mb")] = str(60 + 20 * (int(row[0]) / 30))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    assert soak.analyze(Table(path))["findings"] == ["no throttling, memory growth or slowdown found"]


@pytest.mark.software
def test_a_short_run_does_not_judge_memory(tmp_path):
    r = soak.analyze(Table(write_system(tmp_path / "system.csv", minutes=2, rss_growth_per_min=5)))
    assert any(f.startswith("run too short") for f in r["findings"])
    assert not any("suspected leak" in f for f in r["findings"])


@pytest.mark.software
def test_a_slowing_loop_is_reported_per_window(tmp_path):
    r = soak.analyze(Table(write_system(tmp_path / "system.csv")),
                     Table(write_frames(tmp_path / "f.csv", first_ms=40, last_ms=60)))
    w = r["loop"]["windows"]
    assert len(w) == 15 and w[-1]["median"] > w[0]["median"]            # 20 FPS: frames end just before 900 s
    assert w[0]["temp_c"] < w[-1]["temp_c"]
    assert r["loop"]["drift_p95"] > soak.SLOWDOWN
    assert any(f.startswith("loop p95") for f in r["findings"])


@pytest.mark.software
def test_a_desktop_log_without_temperature_still_reports_memory(tmp_path):
    path = tmp_path / "system.csv"
    path.write_text(",".join(FIELDS) + "\n" + "".join(
        f"{i},,,,,,,,{80 + i * 0.001},3000,0.5\n" for i in range(400)))
    r = soak.analyze(Table(path))
    assert r["temp_c"] is None and r["cpu_mhz"] is None and r["throttle"] == {}
    assert r["memory"]["rss_start_mb"] == pytest.approx(80)


@pytest.mark.software
def test_fewer_than_two_samples_is_an_error(tmp_path):
    path = tmp_path / "system.csv"
    path.write_text(",".join(FIELDS) + "\n0,50,1000,0x0,0,0,0,0,80,300,1\n")
    with pytest.raises(ValueError, match="fewer than 2"):
        soak.analyze(Table(path))


@pytest.mark.software
def test_cli_reads_both_files_from_the_folder(tmp_path, capsys):
    write_system(tmp_path / "run" / "system.csv", throttle_from_min=10)
    write_frames(tmp_path / "run" / "soak_frames.csv")
    assert soak.main([str(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    assert "temperature" in out and "loop" in out and "-> throttling" in out
    assert (tmp_path / "run" / "soak.json").exists()


@pytest.mark.software
def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    write_system(tmp_path / "system.csv", rss_growth_per_min=0.5, throttle_from_min=11)
    write_frames(tmp_path / "soak_frames.csv", last_ms=55)
    system, frames, _ = soak.load(tmp_path)
    r = soak.analyze(system, frames)
    assert soak.figure(system, r, "t", tmp_path / "soak.png").stat().st_size > 0


def looping(frames, replay: bool):
    """Frames until the caller stops: a replay restarts from the top when it runs out."""
    while True:
        delivered = 0
        for fd in frames(10**9):
            delivered += 1
            yield fd
        if not replay or delivered == 0:
            return


@pytest.mark.hardware
def test_soak(request, frames, artifacts):
    minutes = request.config.getoption("--soak-minutes")
    if minutes is None:
        pytest.skip("soak runs only with --soak-minutes=N")
    replay = request.config.getoption("--replay") is not None
    deadline_s = minutes * 60.0

    rows, interrupted = [], False
    processor = Phase3Processor()
    monitor = SystemMonitor().start()
    source = looping(frames, replay)
    try:
        n = 0
        while True:
            t_start = time.perf_counter()
            if t_start - monitor.t0 >= deadline_s:
                break
            try:
                fd = next(source)
            except StopIteration:
                break
            t_frame = time.perf_counter()
            res = run_phase3_chain(fd.frame, fd.frame_id, fd.timestamp_ms, processor, None,
                                   MEASURED, (t_frame - t_start) * 1000.0)
            t = res.timings_ms
            rows.append([n, fd.frame_id, fd.timestamp_ms, round(t_start - monitor.t0, 3),
                         round(t["capture"], 3), round(t["phase2"], 3), round(t["phase3"], 3),
                         round(t["total"], 3), round((time.perf_counter() - t_start) * 1000.0, 3)])
            n += 1
    except KeyboardInterrupt:
        interrupted = True
    finally:
        source.close()                   # releases the camera
        monitor.stop()

    if not rows:
        pytest.skip("no frames delivered")
    samples = monitor.rows()
    assert len(samples) >= 2, "system monitor recorded fewer than 2 samples"

    frames_path = artifacts.csv("soak_frames.csv", list(FRAME_FIELDS), rows)
    system_path = artifacts.csv("system.csv", list(FIELDS), [[s[k] for k in FIELDS] for s in samples])
    result = soak.analyze(Table(system_path), Table(frames_path))
    artifacts.json("summary.json", {**result, "minutes_requested": minutes, "interrupted": interrupted,
                                    "frames": len(rows), "replay": replay})
    soak.figure(Table(system_path), result, f"Soak: {request.node.name}", artifacts.path / "soak.png")

    for line in result["findings"]:
        if line.startswith(("throttling", "memory grows", "loop p95")):
            warnings.warn(line)
