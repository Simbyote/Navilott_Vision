"""
test_jitter.py  --  src/analysis/jitter.py

--software  Tail statistics, over-budget streaks and spike periodicity on
            hand-built interval series, and the command line.
"""
import csv

import numpy as np
import pytest

from src.analysis import jitter


def write_frames(path, intervals):
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = np.concatenate(([0.0], np.cumsum(intervals)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "timestamp_ms", "dt_ms", "mean_intensity"])
        w.writerow([0, 0, "", 100])
        for i, (t, dt) in enumerate(zip(ts[1:], intervals), start=1):
            w.writerow([i, t, dt, 100])
    return path


@pytest.mark.software
def test_steady_intervals_have_no_spikes_or_misses():
    r = jitter.analyze([33.0] * 100, budget_ms=50)
    assert r["over_budget"]["frames"] == 0 and r["spikes"]["count"] == 0
    assert r["effective_fps"] == pytest.approx(1000 / 33)


@pytest.mark.software
def test_over_budget_counts_and_longest_streak():
    iv = [40] * 10 + [60, 60, 60] + [40] * 5 + [60]
    r = jitter.analyze(iv, budget_ms=50)
    assert r["over_budget"]["frames"] == 4 and r["over_budget"]["longest_streak"] == 3


@pytest.mark.software
def test_nan_intervals_are_ignored():
    r = jitter.analyze([np.nan, 40, 40, np.nan, 40], budget_ms=50)
    assert r["interval"]["n"] == 3


@pytest.mark.software
def test_regular_spikes_are_periodic():
    iv = np.full(300, 83.0)
    iv[::60] = 130.0                                   # every 60 frames, like the bench run
    r = jitter.analyze(iv, budget_ms=50)
    assert r["spikes"]["count"] == 5 and r["spikes"]["periodic"]
    assert r["spikes"]["spacing_frames"]["p50"] == 60


@pytest.mark.software
def test_irregular_spikes_are_not_periodic():
    iv = np.full(300, 83.0)
    iv[[5, 12, 90, 95, 250]] = 130.0
    assert not jitter.analyze(iv, budget_ms=50)["spikes"]["periodic"]


@pytest.mark.software
def test_no_intervals_is_an_error():
    with pytest.raises(ValueError):
        jitter.analyze([np.nan, np.nan])


@pytest.mark.software
def test_cli_writes_json_and_reports(tmp_path, capsys):
    path = write_frames(tmp_path / "run" / "frames.csv", [83.0] * 50)
    assert jitter.main([str(path.parent), "--skip", "0"]) == 0
    assert (tmp_path / "run" / "jitter.json").exists()
    assert "longest streak" in capsys.readouterr().out


@pytest.mark.software
def test_cli_writes_the_figure(tmp_path):
    pytest.importorskip("matplotlib")
    path = write_frames(tmp_path / "run" / "frames.csv", [83.0] * 50)
    jitter.main([str(path), "--skip", "0"])
    assert (tmp_path / "run" / "jitter.png").stat().st_size > 0


@pytest.mark.software
def test_cli_reports_a_missing_run(tmp_path, capsys):
    assert jitter.main([str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().err


@pytest.mark.software
def test_small_spikes_on_a_noisy_baseline_are_caught():
    # The bench run's spikes were only ~5 ms over an 83 ms median
    rng = np.random.default_rng(0)
    iv = 83.0 + rng.uniform(-0.5, 0.5, 400)
    iv[30::60] = 89.0
    r = jitter.analyze(iv, budget_ms=50)
    assert r["spikes"]["count"] == 7 and r["spikes"]["periodic"]


def write_nav(path, n=20, step_s=0.05, stage_from=8, stage_to=14):
    """A small nav.csv: timings, t, and rule / stage / lane / light columns, stage blank outside 8-13."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "t", "capture_ms", "phase2_ms", "phase3_ms", "nav_ms", "latency_ms",
                    "rule", "stage", "lane_status", "drive_state"])
        for i in range(n):
            inside = stage_from <= i < stage_to
            w.writerow([i, round(i * step_s, 3), 2.0, 20.0, 0.5, 0.2, 23.0,
                        "intersection" if inside else "lane_keeping", "to_line" if inside else "",
                        "stale" if inside else "vision", "go"])
    return path


@pytest.mark.software
def test_a_navigation_run_folders_intervals_come_from_nav_csvs_t(tmp_path, capsys):
    write_nav(tmp_path / "nav.csv", step_s=0.06)
    (tmp_path / "p3.csv").write_text("frame_id,dt_s\n0,0.05\n1,0.05\n")
    assert jitter.main([str(tmp_path)]) == 0
    assert "nav.csv" in capsys.readouterr().out
