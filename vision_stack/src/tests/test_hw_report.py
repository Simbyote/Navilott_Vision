"""
test_hw_report.py  --  src/analysis/hw_report.py

--software  A made-up hardware run: every CSV a tool reads is found and
            analyzed in its own folder, frame_meta's frames.csv goes to
            camera_meta and not to jitter, a broken CSV fails alone, hw_report.txt lists it all,
            and the newest run is the default. No camera.
"""
import csv

import pytest

import src.analysis.hw_report as hr

STAGES = ("frame_id", "timestamp_ms", "capture_ms", "preprocess_ms", "roi_ms", "geometry_ms", "color_ms",
          "lane_offset_ms", "stop_line_ms", "fusion_ms", "package_ms", "total_ms", "interval_ms")


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def hardware_run(root):
    run = root / "20261012_172045"
    write_csv(run / "test_stage_timing_characterization" / "stage_timing.csv", STAGES,
              [(i, 1000 + 50 * i, 18, 11.5, 0.3, 14, 5.4, 0.3, 0.07, 0.2, 0.1, 32 + i % 3, 50 + (i % 5) - 2)
               for i in range(40)])
    write_csv(run / "test_capture_characterization" / "frames.csv", ("frame_id", "timestamp_ms", "dt_ms"),
              [(i, 1000 + 50 * i, "" if i == 0 else 50 + i % 3) for i in range(40)])
    write_csv(run / "test_frame_meta_on_the_pi" / "frame_meta" / "frames.csv", ("frame", "t_s", "exposure_us"),
              [(i, i * 0.05, 20000) for i in range(40)])          # sensor time: not a loop's intervals
    write_csv(run / "test_capture_anatomy_on_the_pi" / "anatomy" / "latency.csv", ("ts_ms", "kind", "element", "ms"),
              [r for i in range(40) for r in ((50 * i, "camera", "libcamerasrc0", 11.0),
                                              (50 * i + 15, "element", "videoflip0", 1.2),
                                              (50 * i + 16, "pipeline", "pipeline0", 2.0))])
    return run


@pytest.mark.software
def test_plan_finds_each_tools_csvs_and_gives_frame_meta_to_camera_meta_not_jitter(tmp_path):
    run = hardware_run(tmp_path)
    planned = [(t, p.relative_to(run).as_posix()) for t, p in hr.plan(run)]
    assert planned == [
        ("stage_timing", "test_stage_timing_characterization/stage_timing.csv"),
        ("jitter", "test_capture_characterization/frames.csv"),
        ("jitter", "test_stage_timing_characterization/stage_timing.csv"),
        ("camera_meta", "test_frame_meta_on_the_pi/frame_meta/frames.csv"),
        ("capture_report", "test_capture_anatomy_on_the_pi/anatomy/latency.csv"),
    ]


@pytest.mark.software
def test_report_runs_each_tool_beside_its_csv_and_lists_what_it_wrote(tmp_path):
    run = hardware_run(tmp_path)
    results, path = hr.report(run)
    assert [r["code"] for r in results] == [0, 0, 0, 0, 0]
    stage_dir = run / "test_stage_timing_characterization"
    assert (stage_dir / "jitter.json").exists() and (run / "test_capture_characterization" / "jitter.json").exists()
    wrote = {p.name for r in results for p in r["wrote"]}
    assert {"jitter.json"} <= wrote
    text = path.read_text()
    assert path == run / hr.REPORT and "5 analyses" in text
    assert "[camera_meta] test_frame_meta_on_the_pi/frame_meta/frames.csv  ok" in text
    assert (run / "test_frame_meta_on_the_pi" / "frame_meta" / "camera_meta.json").exists()
    assert (run / "test_capture_anatomy_on_the_pi" / "anatomy" / "capture_report.json").exists()
    assert "[jitter] test_capture_characterization/frames.csv  ok" in text
    assert "-> test_capture_characterization/jitter.json" in text


@pytest.mark.software
def test_a_broken_csv_fails_alone_and_main_says_so(tmp_path, capsys):
    run = hardware_run(tmp_path)
    write_csv(run / "test_live_view_characterization" / "stages.csv", ("frame_id",), [])
    assert hr.main([str(run)]) == 2
    out = capsys.readouterr().out
    assert "ERR stage_timing     test_live_view_characterization/stages.csv" in out
    assert "ok  stage_timing     test_stage_timing_characterization/stage_timing.csv" in out


@pytest.mark.software
def test_the_newest_run_is_the_default_and_an_empty_root_is_an_error(tmp_path):
    hardware_run(tmp_path)
    (tmp_path / "20261011_090000").mkdir()
    assert hr.newest_run(tmp_path).name == "20261012_172045"
    with pytest.raises(FileNotFoundError):
        hr.newest_run(tmp_path / "missing")
    assert hr.main([str(tmp_path / "nope")]) == 1
