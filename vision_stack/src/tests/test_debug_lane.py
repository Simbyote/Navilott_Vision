"""
test_debug_lane.py  --  src/debugger/debug_lane.py

Software tests build LaneOffsetResults and debug dicts by hand, so each overlay
element can be checked at a known pixel; the chained tests feed the view real
run_chain() output. candidate_gates() is tested against every lane_offset gate,
because it recovers gate names by parsing _usable()'s log text, and a reworded
message would otherwise silently blank the view's gate labels.

--software  load_source_frame, annotate, DebugVideoWriter, candidate_gates,
            log_tags and LaneView. No camera.
--hardware  Runs the chain with overlays and traces on (live camera or
            --replay) and writes every debug image a stage produces, per
            sample frame, plus a DebugVideoWriter video of the whole run.
            This is the one to open when tuning a stage by eye.
"""
import csv
from dataclasses import replace

import cv2
import numpy as np
import pytest

import src.debugger.debug_lane as dl
from src.debugger.debug_stop import StopView
from src.debugger.debug_traffic import TrafficView
from src.params import (
    COLOR_BLUR_SUFFIX, FRAME_H, FRAME_W, GRAY_BLUR_SUFFIX, HSV_RANGES_PATH,
    LANE_ROI_SUFFIX, MODE_NONE, MODE_TWO_BOUNDARY, ROI_OVERLAY_SUFFIX,
    SIGN_ROI_SUFFIX, TRAFFIC_ROI_SUFFIX, UNDISTORT_SUFFIX,
)
from src.perception.color_branch import BlobFilter, ColorConfig, HSVRanges, load_color_config
from src.perception.geometry import GeometryBranchResult, LaneCandidate
from src.perception.lane_offset import LaneOffsetConfig, LaneOffsetResult
from src.perception.roi_crop import draw_roi_overlay
from src.phase2_linker import MEASURED, run_chain, synthetic_frame

LANE_RECT = (24, 189, 432, 81)          # default LANE bounds at 480x270
SIZE = (FRAME_W, FRAME_H)
CX = LANE_RECT[2] / 2.0                 # robot position in the lane ROI


def result(left_x=100.0, right_x=300.0, offset=None, mode=MODE_TWO_BOUNDARY, frame_id=7, ts=350):
    """LaneOffsetResult; offset defaults to what left_x / right_x imply."""
    if offset is None:
        offset = (CX - (left_x + right_x) / 2.0) / CX
    return LaneOffsetResult(
        offset=offset, left_x=left_x, right_x=right_x,
        lane_width_px=None if left_x is None or right_x is None else right_x - left_x,
        confidence=0.7, boundary_count=2, mode=mode, frame_id=frame_id, timestamp_ms=ts)


def dbg(log=(), anchors=((100.0, 0.7), (300.0, 0.7)), raw=2):
    """compute_lane_offset-shaped debug summary."""
    return {"log": list(log), "anchors": list(anchors), "raw_count": raw}


def gray_frame(value=60):
    """Uniform BGR frame at the pipeline size."""
    return np.full((FRAME_H, FRAME_W, 3), value, np.uint8)


def px(img, roi_x, roi_y, scale=1):
    """The output pixel at a lane-ROI coordinate."""
    return tuple(img[int((LANE_RECT[1] + roi_y) * scale), int((LANE_RECT[0] + roi_x) * scale)])


def has(img, bgr):
    """True if any pixel is exactly this color."""
    return bool(np.any(np.all(img == np.array(bgr, np.uint8), axis=2)))


def cand(conf=0.9, prox=0.9, length=100.0, width=8.0, intensity=200.0, bbox=(80, 20, 8, 40)):
    """LaneCandidate with the fields lane_offset's gates read."""
    return LaneCandidate(label="lane_boundary", bbox=bbox, contour=None, confidence=conf,
                         frame_id=1, timestamp_ms=2, proximity=prox, width_px=width,
                         length_px=length, mean_intensity=intensity, foot_x=84.0)


@pytest.mark.software
def test_full_frame_is_read_as_is(tmp_path):
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), gray_frame(90))
    img = dl.load_source_frame(str(p), LANE_RECT)
    assert img.shape == (FRAME_H, FRAME_W, 3) and int(img.mean()) == 90


@pytest.mark.software
def test_lane_roi_crop_is_placed_back_at_its_rect_on_black(tmp_path):
    x, y, w, h = LANE_RECT
    p = tmp_path / "crop.png"
    cv2.imwrite(str(p), np.full((h, w, 3), 200, np.uint8))
    img = dl.load_source_frame(str(p), LANE_RECT)
    assert img.shape == (FRAME_H, FRAME_W, 3)
    assert (img[y:y + h, x:x + w] == 200).all()
    assert img[:y].max() == 0 and img[:, :x].max() == 0


@pytest.mark.software
@pytest.mark.parametrize("channels", [1, 4])
def test_gray_and_bgra_images_come_back_as_bgr(tmp_path, channels):
    p = tmp_path / "f.png"
    shape = (FRAME_H, FRAME_W) if channels == 1 else (FRAME_H, FRAME_W, 4)
    cv2.imwrite(str(p), np.full(shape, 90, np.uint8))
    assert dl.load_source_frame(str(p), LANE_RECT).shape == (FRAME_H, FRAME_W, 3)


@pytest.mark.software
def test_other_sizes_are_resized_to_the_frame_size(tmp_path):
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), np.full((540, 960, 3), 90, np.uint8))
    assert dl.load_source_frame(str(p), LANE_RECT).shape == (FRAME_H, FRAME_W, 3)


@pytest.mark.software
def test_unreadable_file_is_none(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    assert dl.load_source_frame(str(tmp_path / "bad.png"), LANE_RECT) is None


@pytest.mark.software
def test_annotate_returns_a_copy_and_leaves_the_source_frame_alone():
    f = gray_frame()
    before = f.copy()
    out = dl.annotate(f, result(), dbg(), LANE_RECT)
    assert np.array_equal(f, before) and out is not f and not np.shares_memory(out, f)


@pytest.mark.software
@pytest.mark.parametrize("scale", [1, 2])
def test_annotate_output_is_frame_size_times_scale(scale):
    assert dl.annotate(gray_frame(), result(), dbg(), LANE_RECT, scale=scale).shape == \
        (FRAME_H * scale, FRAME_W * scale, 3)


@pytest.mark.software
def test_annotate_without_a_frame_draws_on_the_blank_canvas():
    out = dl.annotate(None, result(), dbg(), LANE_RECT)
    assert tuple(out[FRAME_H - 1, 5]) == (25, 25, 25)            # outside the ROI, below the header


@pytest.mark.software
@pytest.mark.parametrize("scale", [1, 2])
def test_selected_boundaries_and_centers_land_at_their_roi_x(scale):
    out = dl.annotate(gray_frame(), result(100.0, 300.0), dbg(), LANE_RECT, scale=scale)
    y = 0.8 * LANE_RECT[3]                                      # below the mid-height arrow
    assert px(out, 100, y, scale) == dl.C_LEFT
    assert px(out, 300, y, scale) == dl.C_RIGHT
    assert px(out, CX, y, scale) == dl.C_GRAY                   # the robot
    assert px(out, 200, y, scale) == dl.C_LANE                  # implied lane center


@pytest.mark.software
def test_clamped_offset_turns_the_lane_center_red():
    ok = dl.annotate(gray_frame(), result(offset=0.4), dbg(), LANE_RECT)
    clamped = dl.annotate(gray_frame(), result(offset=1.0), dbg(), LANE_RECT)
    assert not has(ok, dl.C_RED) and has(clamped, dl.C_RED)


@pytest.mark.software
def test_non_steering_modes_draw_no_lane_center():
    out = dl.annotate(gray_frame(), result(None, None, offset=0.0, mode=MODE_NONE),
                      dbg(anchors=()), LANE_RECT)
    assert not has(out, dl.C_LANE) and not has(out, dl.C_LEFT) and not has(out, dl.C_RIGHT)


@pytest.mark.software
def test_candidates_are_boxed_green_when_usable_and_red_when_rejected():
    out = dl.annotate(gray_frame(), result(), dbg(), LANE_RECT,
                      candidates=[((20, 10, 30, 30), None), ((350, 10, 30, 30), "length_px")])
    assert px(out, 20, 25) == dl.C_USABLE
    assert px(out, 350, 25) == dl.C_RED


@pytest.mark.software
def test_log_tags_counts_each_entrys_opening_tag():
    tags = dl.log_tags(["[REJECT] a", "[REJECT] b", "[MERGE] c", "[ONE-SIDED] d", "no tag"])
    assert tags == {"REJECT": 2, "MERGE": 1, "ONE-SIDED": 1}


@pytest.mark.software
def test_writer_records_the_annotated_video_and_decision_log(tmp_path):
    path = tmp_path / "lane.avi"
    with dl.DebugVideoWriter(str(path)) as w:
        for i in range(3):
            img = w.write(gray_frame(), result(frame_id=i), dbg(log=["[REJECT] x"]), LANE_RECT)
    assert img.shape == (FRAME_H, FRAME_W, 3)
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    cap.release()
    with open(tmp_path / "lane.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["frame_id"] for r in rows] == ["0", "1", "2"]
    assert rows[0]["mode"] == MODE_TWO_BOUNDARY and rows[0]["REJECT"] == "1"
    assert list(rows[0]) == list(dl.LANE_CSV_FIELDS)


@pytest.mark.software
def test_writer_stride_skips_frames_and_returns_none_for_them(tmp_path):
    with dl.DebugVideoWriter(str(tmp_path / "lane.avi"), stride=2) as w:
        out = [w.write(gray_frame(), result(frame_id=i), dbg(), LANE_RECT) for i in range(4)]
    assert [o is None for o in out] == [False, True, False, True]
    assert w.frames_written == 2


@pytest.mark.software
def test_writer_without_a_sidecar_still_records_video(tmp_path):
    with dl.DebugVideoWriter(str(tmp_path / "lane.avi"), csv_path=None) as w:
        w.write(gray_frame(), result(), dbg(), LANE_RECT)
    assert (tmp_path / "lane.avi").exists() and not (tmp_path / "lane.csv").exists()


@pytest.mark.software
def test_candidate_gates_marks_passing_candidates_none():
    geo = GeometryBranchResult(lane_candidates=[cand()], sign_candidates=[], frame_id=1, timestamp_ms=2)
    assert dl.candidate_gates(geo, LaneOffsetConfig()) == [((80, 20, 8, 40), None)]


@pytest.mark.software
@pytest.mark.parametrize("override, gate", [
    (dict(conf=0.01), "confidence"),
    (dict(prox=0.01), "proximity"),
    (dict(length=1.0), "length_px"),
    (dict(width=500.0), "width_px"),
    (dict(intensity=1.0), "mean_intensity"),
])
def test_candidate_gates_recovers_every_lane_offset_gate_name(override, gate):
    # These names come from parsing lane_offset's [REJECT] text; a reworded
    # message there fails here instead of blanking the view's labels
    geo = GeometryBranchResult(lane_candidates=[cand(**override)], sign_candidates=[], frame_id=1, timestamp_ms=2)
    [(_, got)] = dl.candidate_gates(geo, LaneOffsetConfig())
    assert got == gate and got in dl.GATE_SHORT


def chain_on_synthetic(marks=(150, 290)):
    """Real run_chain() output for a synthetic frame with lane marks at these ROI x."""
    frame = synthetic_frame(list(marks))
    return run_chain(frame, 5, 250, MEASURED), frame


@pytest.mark.software
def test_lane_view_extracts_renders_and_logs_a_real_chain_result():
    chain, frame = chain_on_synthetic()
    v = dl.LaneView(MEASURED.lane_offset)
    data = v.extract(chain, frame)
    v.observe(data)
    assert v.render(data).shape == frame.shape
    assert len(v.row(data)) == len(v.CSV_FIELDS)
    assert all(g is None for _, g in data["gates"])              # both marks pass MEASURED
    report = "\n".join(v.report())
    assert "[MODES]" in report and "two_boundary" in report and "[AVAILABILITY] 1/1" in report


@pytest.mark.software
def test_lane_view_without_a_config_draws_no_candidate_boxes():
    chain, frame = chain_on_synthetic()
    assert dl.LaneView(None).extract(chain, frame)["gates"] == []


@pytest.mark.software
def test_lane_view_tracks_the_longest_blind_run():
    v = dl.LaneView(MEASURED.lane_offset)
    seen, blind = chain_on_synthetic(), chain_on_synthetic(marks=())
    for chain, frame in (seen, blind, blind, seen, blind):
        v.observe(v.extract(chain, frame))
    assert "longest run without one: 2 frames" in "\n".join(v.report())


def _color_config():
    """Calibrated ranges if the file exists, else the scaffold, so the traffic view always draws."""
    if HSV_RANGES_PATH.exists():
        return load_color_config(str(HSV_RANGES_PATH)), "calibrated"
    return ColorConfig(HSVRanges(), BlobFilter()), "scaffold (uncalibrated)"


@pytest.mark.hardware
def test_debug_images_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    color, color_source = _color_config()
    cfg = replace(MEASURED, color=color)
    views = {"stop": StopView(), "traffic": TrafficView()}
    lane_view = dl.LaneView(cfg.lane_offset)

    video = dl.DebugVideoWriter(str(artifacts.path / "lane_annotated.avi"))
    sampled = 0
    try:
        for i, fd in enumerate(frames(n)):
            chain = run_chain(fd.frame, fd.frame_id, fd.timestamp_ms, cfg, draw_overlays=True, trace=True)
            gates = dl.candidate_gates(chain.geometry, cfg.lane_offset)
            video.write(fd.frame, chain.offset, chain.offset_debug, chain.roi.lane_rect, gates)
            if i not in (0, n // 2, n - 1):
                continue

            sampled += 1
            fid, pre, roi = f"{fd.frame_id:06d}", chain.pre, chain.roi
            images = {
                "_raw.png": fd.frame,
                UNDISTORT_SUFFIX: pre.undistorted,
                GRAY_BLUR_SUFFIX: pre.gray,
                COLOR_BLUR_SUFFIX: pre.color,
                ROI_OVERLAY_SUFFIX: draw_roi_overlay(pre.undistorted, roi),
                LANE_ROI_SUFFIX: roi.lane_roi,
                TRAFFIC_ROI_SUFFIX: roi.traffic_roi,
                SIGN_ROI_SUFFIX: roi.sign_roi,
                "_lane_edges_raw.png": chain.lane_debug["edges_raw"],
                "_lane_edges.png": chain.lane_debug["edges"],
                "_lane_contours.png": chain.lane_debug["contour_overlay"],
                "_lane_accepted.png": chain.lane_debug["accepted_overlay"],
                "_sign_edges.png": chain.sign_debug["edges"],
                "_sign_contours.png": chain.sign_debug["contour_overlay"],
                "_sign_accepted.png": chain.sign_debug["accepted_overlay"],
                "_lane_view.png": lane_view.render(lane_view.extract(chain, fd.frame)),
            }
            for name, view in views.items():
                images[f"_{name}_view.png"] = view.render(view.extract(chain, fd.frame))
            for suffix, img in images.items():
                artifacts.image(fid + suffix, img)
    finally:
        video.close()

    if not sampled:
        pytest.skip("no frames delivered")
    artifacts.json("config.json", {"color_source": color_source, "frames": n,
                                   "video_frames": video.frames_written})
    assert video.frames_written > 0