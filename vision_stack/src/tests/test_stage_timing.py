"""
test_stage_timing.py  --  src/analysis/stage_timing.py, over the whole Phase 1-2 chain

Not a stage test: it times every stage together, plus the capture wait and
the loop around them, and draws where each frame's time goes against the
1000/FPS budget.

--software  Stage discovery and ordering, CSV reading (warm-up skip, blank
            cells, interval), the breakdown arithmetic on hand-made tables
            with known answers, the contract with run_chain's timings_ms and
            live_view's stages.csv, the figures and the command line.
--hardware  Runs frames (live or --replay) through run_chain with capture
            and loop timed around it, and writes stage_timing.csv,
            summary.json, timing_budget.png and timing_per_frame.png.
            Asserts only that every stage was timed on every frame; a loop
            slower than the budget warns rather than fails.
"""
import csv
import math
import os
import time
import warnings

import numpy as np
import pytest

import src.analysis.stage_timing as st
from src.debugger.live_view import StageLog
from src.params import FPS
from src.config import MEASURED
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, synthetic_frame

BUDGET = 1000.0 / FPS


def write_csv(path, header, rows):
    """A stage CSV with the given header and rows; returns its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def table(**columns):
    """read_timing_csv()-shaped dict from keyword lists."""
    return {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}


# =============================================================================
# Stage discovery
# =============================================================================

@pytest.mark.software
def test_stage_names_follow_pipeline_order_whatever_the_column_order():
    cols = ["fusion_ms", "roi_ms", "capture_ms", "preprocess_ms", "geometry_ms"]
    assert st.stage_names(cols) == ["capture", "preprocess", "roi", "geometry", "fusion"]


@pytest.mark.software
def test_unknown_timers_are_kept_after_the_known_stages_in_file_order():
    cols = ["zeta_ms", "preprocess_ms", "alpha_ms"]
    assert st.stage_names(cols) == ["preprocess", "zeta", "alpha"]


@pytest.mark.software
@pytest.mark.parametrize("col", ["timestamp_ms", "total_ms", "interval_ms", "frame_id", "mode"])
def test_non_duration_columns_are_not_stages(col):
    assert st.stage_names([col, "roi_ms"]) == ["roi"]


@pytest.mark.software
def test_every_stage_run_chain_times_is_known_and_counted_inside_total():
    # A stage run_chain adds must be placed in STAGE_ORDER, or the bar
    # would draw it out of order; it must not be marked external, or
    # the unaccounted and outside arithmetic would count it twice
    chain = run_chain(synthetic_frame([150, 290]), 1, 50, SCENE_CONFIG)
    assert chain.timings_ms, "run_chain reported no timings"
    for name, ms in chain.timings_ms.items():
        assert name in st.STAGE_ORDER, f"{name} missing from STAGE_ORDER"
        assert name not in st.EXTERNAL_STAGES
        assert ms >= 0.0


@pytest.mark.software
def test_live_view_stages_csv_header_is_readable(tmp_path):
    header = list(StageLog.FIELDS)
    row = [1, 50, 3.0, 0.2, 5.0, 0.1, 0.1, 9.0, 0, 0, "none", 0.0, 0, 0.02, 0.4]
    cols = st.read_timing_csv(write_csv(tmp_path / "stages.csv", header, [row] * 3), skip=0)
    assert set(st.stage_names(cols)) == {"preprocess", "roi", "geometry", "color", "lane_offset",
                                         "stop_line", "fusion"}
    assert "mode_ms" not in cols and "mode" not in cols


# =============================================================================
# Reading
# =============================================================================

@pytest.mark.software
def test_warmup_frames_are_dropped(tmp_path):
    rows = [[i, i * 50, 30.0 if i < 2 else 3.0] for i in range(6)]
    cols = st.read_timing_csv(write_csv(tmp_path / "s.csv", ["frame_id", "timestamp_ms", "roi_ms"], rows), skip=2)
    assert cols["frame_id"].tolist() == [2, 3, 4, 5]
    assert np.all(cols["roi_ms"] == 3.0)


@pytest.mark.software
def test_a_file_with_no_frames_after_warmup_is_rejected(tmp_path):
    path = write_csv(tmp_path / "s.csv", ["frame_id", "roi_ms"], [[0, 1.0], [1, 1.0]])
    with pytest.raises(ValueError, match="skip=2"):
        st.read_timing_csv(path, skip=2)


@pytest.mark.software
def test_blank_cells_are_missing_not_zero(tmp_path):
    rows = [[0, 0, 4.0], [1, 50, ""], [2, 100, 6.0]]
    cols = st.read_timing_csv(write_csv(tmp_path / "s.csv", ["frame_id", "timestamp_ms", "roi_ms"], rows), skip=0)
    assert math.isnan(cols["roi_ms"][1])
    assert st.breakdown(cols, BUDGET)["stages"]["roi"]["median"] == pytest.approx(5.0)


@pytest.mark.software
def test_interval_comes_from_timestamps_when_not_recorded(tmp_path):
    rows = [[0, 1000, 1.0], [1, 1083, 1.0], [2, 1166, 1.0]]
    cols = st.read_timing_csv(write_csv(tmp_path / "s.csv", ["frame_id", "timestamp_ms", "roi_ms"], rows), skip=0)
    assert math.isnan(cols["interval_ms"][0])
    assert cols["interval_ms"][1:].tolist() == [83.0, 83.0]


@pytest.mark.software
def test_a_recorded_interval_is_kept_over_timestamps(tmp_path):
    # Replayed frames have synthesized stamps; the measured loop time is the truth
    rows = [[0, 0, 1.0, 12.5], [1, 50, 1.0, 13.5]]
    header = ["frame_id", "timestamp_ms", "roi_ms", "interval_ms"]
    cols = st.read_timing_csv(write_csv(tmp_path / "s.csv", header, rows), skip=0)
    assert cols["interval_ms"].tolist() == [12.5, 13.5]


# =============================================================================
# Breakdown
# =============================================================================

@pytest.mark.software
def test_unaccounted_is_total_minus_the_chain_stages():
    # live_view's stages.csv has no package_ms: its time lands here
    bd = st.breakdown(table(preprocess_ms=[3, 3, 3], geometry_ms=[5, 5, 5], total_ms=[9, 9, 9]), BUDGET)
    assert bd["unaccounted_ms"] == pytest.approx(1.0)


@pytest.mark.software
def test_unaccounted_never_goes_negative():
    bd = st.breakdown(table(preprocess_ms=[5, 5], total_ms=[4, 4]), BUDGET)
    assert bd["unaccounted_ms"] == 0.0


@pytest.mark.software
def test_capture_is_not_counted_inside_total():
    bd = st.breakdown(table(capture_ms=[40, 40], preprocess_ms=[3, 3], total_ms=[3, 3]), BUDGET)
    assert bd["unaccounted_ms"] == 0.0


@pytest.mark.software
def test_outside_is_loop_time_minus_total_and_the_external_stages():
    bd = st.breakdown(table(capture_ms=[40, 40], preprocess_ms=[9, 9], total_ms=[9, 9],
                            interval_ms=[83, 83]), BUDGET)
    assert bd["outside_ms"] == pytest.approx(34.0)


@pytest.mark.software
def test_without_loop_timing_there_is_no_outside_segment():
    bd = st.breakdown(table(preprocess_ms=[3, 3], total_ms=[3, 3]), BUDGET)
    assert bd["outside_ms"] is None and bd["interval"] is None
    assert st.OUTSIDE not in [n for n, _ in bd["segments"]]


@pytest.mark.software
def test_p3_csv_total_that_includes_capture_is_detected():
    # phase3_linker writes total_ms = capture + phase2 + phase3
    bd = st.breakdown(table(capture_ms=[40, 40], phase2_ms=[9, 9], phase3_ms=[0.5, 0.5],
                            total_ms=[49.5, 49.5], interval_ms=[50, 50]), BUDGET)
    assert bd["total_includes_external"] is True
    assert bd["unaccounted_ms"] == pytest.approx(0.0)
    assert bd["outside_ms"] == pytest.approx(0.5)


@pytest.mark.software
def test_a_total_that_excludes_capture_is_left_alone():
    bd = st.breakdown(table(capture_ms=[40, 40], preprocess_ms=[9, 9], total_ms=[9.2, 9.2]), BUDGET)
    assert bd["total_includes_external"] is False
    assert bd["unaccounted_ms"] == pytest.approx(0.2)


@pytest.mark.software
def test_segments_follow_the_loop_capture_first_outside_last():
    bd = st.breakdown(table(record_ms=[2, 2], geometry_ms=[5, 5], capture_ms=[10, 10],
                            preprocess_ms=[3, 3], total_ms=[8, 8], interval_ms=[30, 30]), BUDGET)
    names = [n for n, _ in bd["segments"]]
    assert names == ["capture", "preprocess", "geometry", st.UNACCOUNTED, "record", st.OUTSIDE]


@pytest.mark.software
def test_segments_and_headroom_add_up_to_the_budget():
    bd = st.breakdown(table(preprocess_ms=[3, 3], geometry_ms=[5, 5], total_ms=[9, 9],
                            interval_ms=[20, 20]), BUDGET)
    assert sum(ms for _, ms in bd["segments"]) + bd["headroom_ms"] == pytest.approx(BUDGET)


@pytest.mark.software
def test_headroom_goes_negative_over_budget():
    bd = st.breakdown(table(preprocess_ms=[3, 3], total_ms=[3, 3], interval_ms=[83, 83]), 50.0)
    assert bd["headroom_ms"] == pytest.approx(-33.0)


@pytest.mark.software
def test_share_is_the_stage_median_over_the_budget():
    bd = st.breakdown(table(geometry_ms=[4, 5, 6], total_ms=[5, 5, 5]), 50.0)
    assert bd["stages"]["geometry"]["share"] == pytest.approx(0.10)


@pytest.mark.software
def test_p95_is_reported_per_stage():
    bd = st.breakdown(table(geometry_ms=list(range(1, 101)), total_ms=[50] * 100), BUDGET)
    assert bd["stages"]["geometry"]["p95"] == pytest.approx(np.percentile(range(1, 101), 95))


@pytest.mark.software
def test_a_stage_with_no_timed_frames_has_no_median_and_no_segment():
    bd = st.breakdown(table(preprocess_ms=[3, 3], color_ms=[np.nan, np.nan], total_ms=[3, 3]), BUDGET)
    assert bd["stages"]["color"]["median"] is None
    assert "color" not in [n for n, _ in bd["segments"]]


# =============================================================================
# Figures and command line
# =============================================================================

def sample_run(directory, n=20):
    """A stages.csv shaped like live_view's, 83 ms apart like the 12 FPS bench run."""
    rows = [[i, i * 83, 3.4, 0.2, 5.0 + (i % 4), 0.1, 0.1, 9.0 + (i % 4), 0, 0, "none", 0.0, 0, 0.02]
            for i in range(n)]
    return write_csv(directory / "stages.csv", list(StageLog.FIELDS), rows)


@pytest.mark.software
def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    cols = st.read_timing_csv(sample_run(tmp_path), skip=0)
    bd = st.breakdown(cols, BUDGET)
    assert st.budget_figure(bd, "t", tmp_path / "b.png").stat().st_size > 0
    assert st.per_frame_figure(cols, BUDGET, "t", tmp_path / "p.png").stat().st_size > 0


@pytest.mark.software
def test_figures_are_skipped_without_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_pyplot", lambda: None)
    cols = st.read_timing_csv(sample_run(tmp_path), skip=0)
    assert st.budget_figure(st.breakdown(cols, BUDGET), "t", tmp_path / "b.png") is None
    assert st.per_frame_figure(cols, BUDGET, "t", tmp_path / "p.png") is None
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.software
def test_find_csv_prefers_stage_timing_csv_in_a_folder(tmp_path):
    sample_run(tmp_path)
    write_csv(tmp_path / "stage_timing.csv", ["frame_id", "roi_ms"], [[0, 1.0]])
    assert st.find_csv(tmp_path).name == "stage_timing.csv"


@pytest.mark.software
def test_find_csv_without_an_argument_takes_the_newest_run(tmp_path):
    old = sample_run(tmp_path / "20260901_120000")
    new = sample_run(tmp_path / "20260927_090000")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert st.find_csv(None, roots=(tmp_path,)).parent.name == "20260927_090000"


@pytest.mark.software
def test_find_csv_names_what_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="stages.csv"):
        st.find_csv(tmp_path)


@pytest.mark.software
def test_cli_writes_both_figures_next_to_the_csv(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    sample_run(tmp_path)
    assert st.main([str(tmp_path), "--skip", "0"]) == 0
    assert (tmp_path / "timing_budget.png").exists() and (tmp_path / "timing_per_frame.png").exists()
    out = capsys.readouterr().out
    assert "outside pipeline" in out and "headroom" in out


@pytest.mark.software
def test_cli_reports_a_missing_run(tmp_path, capsys):
    assert st.main([str(tmp_path / "nope")]) == 1
    assert "ERROR" in capsys.readouterr().err


# =============================================================================
# Hardware
# =============================================================================

@pytest.mark.hardware
def test_stage_timing_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    it = iter(frames(n))
    rows, stages = [], None

    while True:
        t_start = time.perf_counter()
        try:
            fd = next(it)                    # capture wait, or disk read under --replay
        except StopIteration:
            break
        t_frame = time.perf_counter()
        chain = run_chain(fd.frame, fd.frame_id, fd.timestamp_ms, MEASURED)
        t_done = time.perf_counter()

        if stages is None:
            stages = list(chain.timings_ms)
        missing = [s for s in stages if s not in chain.timings_ms]
        assert not missing, f"frame_id={fd.frame_id}: no timing for {missing}"
        rows.append([fd.frame_id, fd.timestamp_ms,
                     round((t_frame - t_start) * 1000, 3),
                     *[round(chain.timings_ms[s], 3) for s in stages],
                     round((t_done - t_frame) * 1000, 3),
                     round((time.perf_counter() - t_start) * 1000, 3)])

    if not rows:
        pytest.skip("no frames delivered")

    header = ["frame_id", "timestamp_ms", "capture_ms", *[f"{s}_ms" for s in stages],
              "total_ms", "interval_ms"]
    path = artifacts.csv("stage_timing.csv", header, rows)

    skip = st.WARMUP_FRAMES if len(rows) > st.WARMUP_FRAMES else 0
    cols = st.read_timing_csv(path, skip=skip)
    bd = st.breakdown(cols, BUDGET)
    artifacts.json("summary.json", {**bd, "warmup_frames_skipped": skip,
                                    "replay": request.config.getoption("--replay")})
    st.budget_figure(bd, f"Stage timing: {request.node.name}", artifacts.path / "timing_budget.png")
    st.per_frame_figure(cols, BUDGET, f"Stage time per frame: {request.node.name}",
                        artifacts.path / "timing_per_frame.png")

    loop_med = bd["interval"]["median"]
    if loop_med and loop_med > BUDGET:
        warnings.warn(f"loop median {loop_med:.1f} ms is over the {BUDGET:.0f} ms budget "
                      f"({1000 / loop_med:.1f} FPS, target {FPS})")
