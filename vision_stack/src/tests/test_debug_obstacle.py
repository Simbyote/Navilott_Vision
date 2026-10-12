"""
test_debug_obstacle.py  --  src/debugger/debug_obstacle.py

The obstacle view on real chains over the shared synthetic scenes,
headless: what it reads from the chain, that every state renders, the CSV
row and the summary, and the live_view command line with --views obstacle.

--software  extract / render / row / report and the live_view run. No camera.
"""
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.debugger.live_view as lv
from src.debugger.debug_obstacle import ObstacleView
from src.perception.obstacle import GATES
from src.phase2_linker import run_chain, run_live_view
from src.tests.scenes import SCENE_CONFIG, SCENES, SYNTHETIC_STOP_LINE_TABLE

CONFIG = replace(SCENE_CONFIG, stop_line_table=SYNTHETIC_STOP_LINE_TABLE)


def chain(scene):
    return run_chain(SCENES[scene], 7, 350, CONFIG)


@pytest.mark.software
def test_extract_reads_the_measurement_the_masks_and_the_blobs():
    c = chain("obstacle_near")
    d = ObstacleView().extract(c)
    assert (d["frame_id"], d["timestamp_ms"]) == (7, 350)
    assert d["measured"] is c.obstacle and d["measured"].detected
    assert d["roi"].shape[:2] == d["mask"].shape == d["opened"].shape
    assert d["rect"] == c.roi.obstacle_rect and d["lane_rect"] == c.roi.lane_rect
    assert d["blobs"] == c.obstacle_debug["blobs"] and d["counts"]["passed"] == 1


@pytest.mark.software
@pytest.mark.parametrize("scene", ["obstacle_near", "obstacle_clipped", "obstacle_narrow", "two_boundary"])
@pytest.mark.parametrize("scale", [1, 2])
def test_every_state_renders_a_color_image_as_wide_as_the_zoomed_roi(scene, scale):
    view = ObstacleView()
    d = view.extract(chain(scene))
    img = view.render(d, scale)
    assert img.ndim == 3 and img.dtype == np.uint8
    assert img.shape[1] == d["roi"].shape[1] * scale * view.zoom


@pytest.mark.software
def test_a_chain_without_an_obstacle_roi_renders_a_notice():
    c = chain("two_boundary")
    bare = SimpleNamespace(roi=replace(c.roi, obstacle_roi=None, obstacle_color_roi=None),
                           obstacle=None, obstacle_debug={})
    img = ObstacleView().render(ObstacleView().extract(bare))
    assert img.ndim == 3 and img.shape[0] < 200


@pytest.mark.software
def test_row_matches_the_header_and_carries_the_measurement():
    view = ObstacleView()
    for scene in ("obstacle_near", "obstacle_narrow", "two_boundary"):
        d = view.extract(chain(scene))
        row = view.row(d)
        assert len(row) == len(view.CSV_FIELDS), scene
        rec = dict(zip(view.CSV_FIELDS, row))
        m = d["measured"]
        assert rec["detected"] == int(m.detected)
        if m.detected:
            assert (rec["distance_px"], rec["distance_cm"], rec["bbox_w"]) == (m.distance_px, m.distance_cm, m.bbox[2])
    assert dict(zip(view.CSV_FIELDS, view.row(view.extract(chain("obstacle_narrow")))))["rej_narrow"] == 1


@pytest.mark.software
def test_report_counts_frames_distances_and_rejections():
    view = ObstacleView()
    for scene in ("obstacle_near", "obstacle_clipped", "obstacle_narrow", "two_boundary"):
        view.observe(view.extract(chain(scene)))
    text = "\n".join(view.report())
    assert "[OBSTACLE] 4 frames" in text and "2  ( 50.0%)" in text
    assert "clipped)           1" in text and "cm: min" in text and "narrow" in text


@pytest.mark.software
def test_live_view_runs_the_obstacle_view_and_writes_its_video_csv_and_summary(tmp_path):
    assert "obstacle" in lv.VIEWS
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, scene in enumerate(("two_boundary", "obstacle_near", "obstacle_clipped")):
        cv2.imwrite(str(frames / f"{i:03d}.png"), SCENES[scene])
    out = tmp_path / "run"
    code = lv.cli(lambda source, **kw: run_live_view(source, config=CONFIG, **kw),
                  ["--frames", str(frames), "--views", "obstacle", "--no-display", "--out", str(out)])
    assert code == 0
    assert (out / "run_obstacle.avi").exists() and (out / "run_obstacle.csv").exists()
    rows = (out / "run_obstacle.csv").read_text().strip().splitlines()
    assert rows[0].split(",") == list(ObstacleView.CSV_FIELDS) and len(rows) == 4
    assert "[OBSTACLE] 3 frames" in (out / "summary.txt").read_text()
    assert all(f"rej_{g}" in ObstacleView.CSV_FIELDS for g in GATES)
