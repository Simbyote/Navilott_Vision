"""
test_live_view.py  --  src/debugger/live_view.py

The frame sources are tested on real files written to tmp_path (and on a fake
CameraSource), the runner on synthetic frames through the real chain, headless.
The display is only tested for its headless fallbacks: a window can't be
opened in CI.

--software  Frame sources and their stamps, RunStats, stages.csv, the headless
            display, run() outputs and early exits, and the command line.
--hardware  A full live_view run with every view (live camera, or --replay DIR)
            into the test's artifact folder: the same videos, CSVs and
            summary.txt a bench run writes.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.debugger.live_view as lv
from src.capture.camera import CaptureError, FrameData
from src.debugger.debug_stop import StopView
from src.debugger.debug_traffic import TrafficView
from src.params import FRAME_H, FRAME_W, HSV_RANGES_PATH
from src.phase2_linker import MEASURED, run_chain, run_live_view, synthetic_frame


def write_frames(directory, n=4, names=None):
    """n synthetic pipeline frames as PNGs, lane marks shifting 2 px per frame."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names or [f"{i:06d}.png" for i in range(n)]):
        cv2.imwrite(str(directory / name), synthetic_frame([150 + 2 * i, 290 + 2 * i]))
    return directory


def process(frame, frame_id, timestamp_ms):
    """The chain live_view runs in production."""
    return run_chain(frame, frame_id, timestamp_ms, MEASURED)


def drain(source):
    """Every item a source yields until it's exhausted."""
    out = []
    while (item := source.read()) is not None:
        out.append(item)
    return out


class ListSource(lv.FrameSource):
    """FrameSource that plays back a script of items; an Exception instance is raised."""
    def __init__(self, items, fps=20):
        super().__init__("scripted", fps)
        self.items = list(items)
        self.closed = False

    def read(self):
        if not self.items:
            return None
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


def frame_item(i):
    """One (frame, id, stamp) as a source delivers it."""
    return synthetic_frame([150, 290]), i, i * 50


@pytest.mark.software
def test_synthesized_stamps_count_from_zero_at_the_nominal_interval():
    src = lv.FrameSource("x", fps=20)
    assert [src._stamp() for _ in range(3)] == [(0, 0), (1, 50), (2, 100)]


@pytest.mark.software
def test_directory_source_reads_images_in_filename_order_and_ignores_other_files(tmp_path):
    d = write_frames(tmp_path / "f", names=["b.png", "a.jpg", "c.PNG"])
    (d / "notes.txt").write_text("not a frame")
    src = lv.DirectoryFrameSource(str(d), fps=10)
    items = drain(src)
    assert [Path(f).name for f in src.files] == ["a.jpg", "b.png", "c.PNG"]
    assert [(fid, ts) for _, fid, ts in items] == [(0, 0), (1, 100), (2, 200)]
    assert src.label == "f"


@pytest.mark.software
def test_directory_source_skips_unreadable_images_without_using_a_frame_id(tmp_path):
    d = write_frames(tmp_path / "f", names=["a.png", "c.png"])
    (d / "b.png").write_bytes(b"corrupt")
    assert [fid for _, fid, _ in drain(lv.DirectoryFrameSource(str(d)))] == [0, 1]


@pytest.mark.software
def test_directory_source_without_images_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(CaptureError, match="no images"):
        lv.DirectoryFrameSource(str(tmp_path / "empty"))


@pytest.mark.software
def test_video_source_plays_a_file_back_with_synthesized_stamps(tmp_path):
    path = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (FRAME_W, FRAME_H))
    for i in range(3):
        vw.write(synthetic_frame([150 + i, 290]))
    vw.release()
    src = lv.VideoFrameSource(str(path))
    assert src.fps == pytest.approx(10.0)                        # the file's own rate
    items = drain(src)
    assert [(fid, ts) for _, fid, ts in items] == [(0, 0), (1, 100), (2, 200)]
    assert items[0][0].shape == (FRAME_H, FRAME_W, 3)
    assert lv.VideoFrameSource(str(path), fps=25).fps == 25      # an explicit rate wins


@pytest.mark.software
def test_video_source_that_cannot_open_raises(tmp_path):
    with pytest.raises(CaptureError, match="could not open"):
        lv.VideoFrameSource(str(tmp_path / "missing.avi"))


@pytest.mark.software
def test_camera_source_keeps_captures_stamp_and_passes_drops_through(monkeypatch):
    reads = [FrameData(np.zeros((2, 2, 3), np.uint8), 41, 2050), None]

    class FakeCamera:
        def __init__(self, w, h, fps): self.args = (w, h, fps)
        def open(self): return self
        def read(self): return reads.pop(0)
        def release(self): self.released = True

    monkeypatch.setattr(lv, "CameraSource", FakeCamera)
    src = lv.CameraFrameSource(FRAME_W, FRAME_H, 20)
    frame, fid, ts = src.read()
    assert (fid, ts) == (41, 2050)                               # not re-minted from an index
    assert src.read() == (None, None, None)                      # transient drop
    src.close()
    assert src.cam.released and src.cam.args == (FRAME_W, FRAME_H, 20)


@pytest.mark.software
def test_run_stats_counts_frames_detections_and_stage_medians():
    stats = lv.RunStats()
    stats.update(2, {"preprocess": 1.0, "geometry": 3.0})
    stats.update(0, {"preprocess": 5.0, "geometry": 3.0})
    stats.drops = 1
    stats.sections["lane"] = ["[LANE] 2 frames"]
    report = "\n".join(stats.report())
    assert "frames processed        2" in report and "dropped reads           1" in report
    assert "fused detections        2  (1.00 per frame)" in report
    assert "[LANE] 2 frames" in report
    assert "preprocess             med    5.0" in report             # upper median of 1, 5
    assert "TOTAL (median)" in report


@pytest.mark.software
def test_run_stats_without_fusion_omits_the_detection_line():
    stats = lv.RunStats()
    stats.update(None, {"chain": 4.0})
    assert not stats.fusion_seen and "fused detections" not in "\n".join(stats.report())


@pytest.mark.software
def test_stage_log_writes_blanks_for_timings_the_chain_did_not_provide(tmp_path):
    path = tmp_path / "stages.csv"
    log = lv.StageLog(str(path))
    chain = process(*frame_item(3))
    log.write(3, 150, {"preprocess": 1.234}, 9.876, chain, None)
    log.close()
    with open(path, newline="") as f:
        [row] = list(csv.DictReader(f))
    assert (row["preprocess_ms"], row["geometry_ms"], row["total_ms"]) == ("1.23", "", "9.88")
    assert row["mode"] == chain.offset.mode and row["detections"] == ""


@pytest.mark.software
def test_display_is_headless_without_a_display_server(monkeypatch, tmp_path):
    monkeypatch.setattr(lv.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    d = lv.Display(True, str(tmp_path))
    assert not d.enabled and d.show({"lane": np.zeros((4, 4, 3), np.uint8)}) is True
    d.close()


@pytest.mark.software
def test_disabled_display_never_opens_a_window(tmp_path):
    d = lv.Display(False, str(tmp_path))
    assert not d.enabled and d.show({}) is True


@pytest.mark.software
def test_run_writes_a_video_and_csv_per_view_plus_stages_csv(tmp_path):
    src = lv.DirectoryFrameSource(str(write_frames(tmp_path / "f", n=4)))
    stats = lv.run(src, process, MEASURED.lane_offset, str(tmp_path / "out"),
                   display=False, views=[StopView(), TrafficView()])
    out = tmp_path / "out"
    names = {p.name for p in out.iterdir()}
    assert names >= {"run.avi", "run.csv", "run_stop.avi", "run_stop.csv",
                     "run_traffic.avi", "run_traffic.csv", "stages.csv"}
    assert stats.frames == 4 and stats.fusion_seen
    assert list(stats.sections) == ["lane", "stop", "traffic"]
    with open(out / "run.csv", newline="") as f:
        assert len(list(csv.reader(f))) == 1 + 4


@pytest.mark.software
def test_stride_thins_the_recordings_but_every_frame_is_still_counted(tmp_path):
    src = lv.DirectoryFrameSource(str(write_frames(tmp_path / "f", n=5)))
    stats = lv.run(src, process, MEASURED.lane_offset, str(tmp_path / "out"), display=False, stride=2)
    with open(tmp_path / "out" / "run.csv", newline="") as f:
        assert len(list(csv.reader(f))) == 1 + 3                 # frames 0, 2, 4
    with open(tmp_path / "out" / "stages.csv", newline="") as f:
        assert len(list(csv.reader(f))) == 1 + 5
    assert stats.frames == 5 and "[LANE] 5 frames" in stats.sections["lane"][0]


@pytest.mark.software
def test_limit_stops_the_run_and_the_source_is_closed(tmp_path):
    src = ListSource([frame_item(i) for i in range(5)])
    stats = lv.run(src, process, MEASURED.lane_offset, str(tmp_path), display=False, limit=2)
    assert stats.frames == 2 and src.closed


@pytest.mark.software
def test_dropped_reads_are_counted_and_skipped(tmp_path):
    src = ListSource([frame_item(0), (None, None, None), frame_item(1)])
    stats = lv.run(src, process, MEASURED.lane_offset, str(tmp_path), display=False)
    assert (stats.frames, stats.drops) == (2, 1)


@pytest.mark.software
def test_a_dead_camera_ends_the_run_but_outputs_are_still_closed(tmp_path):
    src = ListSource([frame_item(0), CaptureError("pipeline dead"), frame_item(1)])
    stats = lv.run(src, process, MEASURED.lane_offset, str(tmp_path), display=False)
    assert stats.frames == 1 and src.closed
    with open(tmp_path / "stages.csv", newline="") as f:
        assert len(list(csv.reader(f))) == 2                    # flushed on close


@pytest.mark.software
def test_a_process_without_fusion_or_timings_is_timed_as_one_unit(tmp_path):
    def bare(frame, fid, ts):
        c = process(frame, fid, ts)
        return SimpleNamespace(geometry=c.geometry, roi=c.roi, offset=c.offset, offset_debug=c.offset_debug)
    stats = lv.run(ListSource([frame_item(0)]), bare, MEASURED.lane_offset, str(tmp_path), display=False)
    assert list(stats.stage_ms) == ["chain"] and not stats.fusion_seen


@pytest.mark.software
def test_cli_with_no_arguments_prints_help(capsys):
    assert lv.cli(run_live_view, []) == 0
    assert "Sources:" in capsys.readouterr().out


@pytest.mark.software
def test_cli_rejects_an_unknown_view(tmp_path, capsys):
    d = write_frames(tmp_path / "f", n=1)
    assert lv.cli(run_live_view, ["--frames", str(d), "--views", "bogus"]) == 2
    assert "unknown view bogus" in capsys.readouterr().out


@pytest.mark.software
def test_cli_reports_a_source_it_cannot_open(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert lv.cli(run_live_view, ["--frames", str(tmp_path / "empty")]) == 2
    assert "source error" in capsys.readouterr().out


@pytest.mark.software
def test_cli_runs_every_view_and_writes_the_summary(tmp_path, capsys):
    d = write_frames(tmp_path / "f", n=3)
    out = tmp_path / "out"
    code = lv.cli(run_live_view, ["--frames", str(d), "--no-display", "--views", "stop,traffic",
                                  "--stop-threshold", "0.5", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "traffic view needs --hsv" in printed               # the branch is off without ranges
    summary = (out / "summary.txt").read_text()
    assert summary.startswith("source: f\nfusion: on")
    assert "[LANE] 3 frames" in summary and "[STOP SIGN] 3 frames" in summary
    assert "color branch OFF" in summary


@pytest.mark.hardware
def test_live_view_characterization(request, artifacts):
    n = request.config.getoption("--frames")
    replay = request.config.getoption("--replay")
    try:
        source = lv.DirectoryFrameSource(replay) if replay else lv.CameraFrameSource(FRAME_W, FRAME_H, lv.FPS)
    except (CaptureError, OSError) as e:
        pytest.skip(f"no frame source: {e}")

    hsv = str(HSV_RANGES_PATH) if HSV_RANGES_PATH.exists() else None      # traffic view stays off without it
    stats = run_live_view(source, out_dir=str(artifacts.path), display=False, limit=n,
                          views=[StopView(), TrafficView()], hsv_path=hsv)
    lv.write_summary(str(artifacts.path / "summary.txt"), source, stats)
    if not stats.frames:
        pytest.skip("no frames delivered")
    assert (artifacts.path / "run.avi").exists()