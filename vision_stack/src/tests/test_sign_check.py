"""
test_sign_check.py  --  src/analysis/sign_check.py

A synthetic sign through the real gates: detected as it stands; cut by
the sign ROI's top edge into 6 vertices and refused, the edge named; found
behind a larger, refused red blob, as the detector now does; a gate's
relaxing named when it alone refuses the sign; the summary's counts and
the printed report; the command line over a folder of frames.

--software  Synthetic scenes; no camera.
"""
import json
from dataclasses import replace

import cv2
import pytest

import src.analysis.sign_check as sc
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, scene

pytestmark = pytest.mark.software
FILTER = SCENE_CONFIG.geometry.sign


def sign_roi(frame):
    return run_chain(frame, 0, 0, SCENE_CONFIG).roi.sign_color_roi


def with_red_block(frame):
    """A red block larger than the sign, low in the sign ROI: a chair, a box."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (int(w * 0.60), int(h * 0.55)), (int(w * 0.80), int(h * 0.82)), (20, 20, 220), -1)
    return out


def test_a_sign_alone_is_detected():
    row = sc.check_frame(sign_roi(scene(sign=True, sign_radius=16)), FILTER)
    assert row["blobs"] == 1 and row["detected"] and row["largest_gate"] is None and not row["hidden"]
    assert row["measured"]["vertices"] in (8, 9)


def test_a_sign_behind_a_larger_refused_red_blob_is_found():
    frame = with_red_block(scene(sign=True, sign_radius=16))
    row = sc.check_frame(sign_roi(frame), FILTER)
    assert row["blobs"] == 2 and row["detected"] and row["hidden"]
    assert row["largest_gate"] is not None                           # the block was gated, and refused
    assert run_chain(frame, 0, 0, SCENE_CONFIG).geometry.sign_candidates     # and the robot sees the sign


def test_the_relaxed_gate_that_would_pass_is_named():
    strict = replace(FILTER, min_vertices=10, max_vertices=12)       # an octagon can't pass
    row = sc.check_frame(sign_roi(scene(sign=True, sign_radius=16)), strict)
    assert row["largest_gate"] == "vertices" and "vertices 7-10" in row["relaxed"]     # 7-10 takes the octagon back
    narrow = replace(FILTER, aspect_range=(1.5, 2.0))
    row = sc.check_frame(sign_roi(scene(sign=True, sign_radius=16)), narrow)
    assert row["largest_gate"] == "aspect" and "no aspect" in row["relaxed"] and "no ghost gates" in row["relaxed"]
    assert "no extent" not in row["relaxed"]


def test_a_sign_cut_by_the_rois_top_edge_loses_its_corners():
    # The sign ROI's top was y0 0.20 until 2026-10-20, which cut the scenes' 28 px sign (centered at
    # 0.25 of the frame) across; from the frame's top it's whole. Cut it the way the old edge did
    whole = sign_roi(scene(sign=True))
    assert sc.check_frame(whole, FILTER)["detected"]
    row = sc.check_frame(whole[int(0.20 * 270):], FILTER)
    assert row["largest_gate"] == "vertices" and row["measured"]["vertices"] < 8 and row["cut"] == "top"
    assert sc.summarize([{**row, "frame_id": 0}])["refused_cut_by_roi"] == 1


def test_no_red_is_no_blob():
    row = sc.check_frame(sign_roi(scene()), FILTER)
    assert (row["blobs"], row["detected"], row["largest_gate"]) == (0, False, None)


def test_the_summary_and_report():
    rows = [{"blobs": 0, "detected": False, "hidden": False, "largest_gate": None, "relaxed": [], "cut": ""},
            {"blobs": 1, "detected": True, "hidden": False, "largest_gate": None, "relaxed": []},
            {"blobs": 2, "detected": True, "hidden": True, "largest_gate": "extent", "relaxed": []},
            {"blobs": 1, "detected": False, "hidden": False, "largest_gate": "letters",
             "relaxed": ["no letters", "no ghost gates"]}]
    s = sc.summarize(rows)
    assert (s["frames"], s["with_red_blob"], s["detected"], s["found_behind_larger"]) == (4, 3, 2, 1)
    assert s["refused_by_gate"] == {"extent": 1, "letters": 1}
    assert s["relaxed_adds"]["no letters"] == 1 and s["relaxed_adds"]["no aspect"] == 0
    text = "\n".join(sc.report(s))
    assert "found behind a larger, refused red blob (missed before 2026-10-20): 1 frames" in text
    assert "no letters" in text
    assert "cut off by the ROI's edge (the sign partly outside it): 0 frames" in text


def test_the_command_line_over_a_folder(tmp_path, capsys):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, f in enumerate([scene(sign=True, sign_radius=16), with_red_block(scene(sign=True, sign_radius=16)), scene()]):
        cv2.imwrite(str(frames / f"{i:04d}.png"), f)
    monkey_config = SCENE_CONFIG
    sc.MEASURED = monkey_config                 # synthetic frames are drawn undistorted
    try:
        assert sc.main(["--frames", str(frames), "--out", str(tmp_path / "out")]) == 0
    finally:
        from src.config import MEASURED
        sc.MEASURED = MEASURED
    out = capsys.readouterr().out
    assert "3 frames, 2 with a red blob in the sign ROI, 2 detected" in out
    summary = json.loads((tmp_path / "out" / "sign_check.json").read_text())["summary"]
    assert summary["found_behind_larger"] == 1
    lines = (tmp_path / "out" / "sign_check.csv").read_text().splitlines()
    assert len(lines) == 4 and lines[1].split(",")[3] == "passed" and lines[0].split(",")[4] == "cut"
