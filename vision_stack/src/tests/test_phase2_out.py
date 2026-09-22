"""
test_phase2_out.py  --  Phase 2 output packaging (the Phase 2 / Phase 3 boundary)

package_phase2 and Phase2Output do no computation, filtering, or re-sorting --
PR-1 through PR-4 and failure cases F1-F5 in the module docstring are the
entire contract. Every test here is named after the rule or failure case it
pins, and F2/F5 use minimal duck-typed fakes (SimpleNamespace) rather than
real DetectionObject/LaneOffsetResult instances, since a "missing field" or
"wrong stamp" input is exactly what the real dataclasses can't produce.

--software  Every packaging rule and failure case, plus chained tests from
            real fusion and lane-offset output. No camera.
--hardware  Times package_phase2 per frame (live or --replay), chaining the
            whole Phase 2 pipeline, and writes CSV plus one JSON snapshot of
            the packaged output per sample frame.
"""
import time
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.capture.camera import FrameData
from src.perception.color_branch import BlobFilter, ColorConfig, HSVRanges, run_color_stage
from src.perception.feature_fusion import DetectionObject, FusionResult, fuse_detections
from src.perception.geometry import GeometryConfig, run_geometry_stage
from src.perception.lane_offset import LaneOffsetConfig, LaneOffsetResult, compute_lane_offset
from src.perception.phase2_out import DETECTION_FIELDS, LANE_OFFSET_FIELDS, Phase2Output, package_phase2
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import ROIConfig, crop_rois
from src.tests.artifacts import summarize


# =============================================================================
# Helpers
# =============================================================================
def det(frame_id=1, ts=2, det_type="lane_boundary", confidence=0.9, source_roi="lane", source_rect=(0, 0, 10, 10)):
    return DetectionObject(det_type, "x", confidence, {"x": 1.0, "y": 2.0}, (0, 0, 1, 1),
                           source_roi, source_rect, frame_id, ts)


def offset_result(frame_id=1, ts=2, mode="two_boundary", offset=0.1):
    return LaneOffsetResult(offset, 10.0, 20.0, 10.0, 0.5, 2, mode, frame_id, ts)


def fields_of(obj):
    return {k: v for k, v in vars(obj).items()}


# =============================================================================
# Software: F1 - non-list detections / lane_offset_results
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("bad", [None, "x", {}, (det(),), 5])
def test_f1_non_list_detections_raises_typeerror_naming_the_field(bad):
    with pytest.raises(TypeError, match="detections"):
        Phase2Output(bad, [], 1, 2)


@pytest.mark.software
@pytest.mark.parametrize("bad", [None, "x", {}, (offset_result(),)])
def test_f1_non_list_lane_offset_results_raises_typeerror_naming_the_field(bad):
    with pytest.raises(TypeError, match="lane_offset_results"):
        Phase2Output([], bad, 1, 2)


@pytest.mark.software
def test_f1_message_tells_the_caller_to_pass_an_empty_list():
    with pytest.raises(TypeError, match=r"\[\]"):
        Phase2Output(None, [], 1, 2)


# =============================================================================
# Software: F2 - a required field is missing
# =============================================================================
@pytest.mark.software
@pytest.mark.parametrize("missing", DETECTION_FIELDS)
def test_f2_a_missing_detection_field_raises_attributeerror_naming_it(missing):
    complete = dict(type="lane_boundary", position={"x": 1, "y": 2}, confidence=0.5,
                    timestamp=2, frame_id=1, source_roi="lane", source_rect=(0, 0, 1, 1))
    del complete[missing]
    with pytest.raises(AttributeError, match=missing):
        Phase2Output([SimpleNamespace(**complete)], [], 1, 2)


@pytest.mark.software
@pytest.mark.parametrize("missing", LANE_OFFSET_FIELDS)
def test_f2_a_missing_lane_offset_field_raises_attributeerror_naming_it(missing):
    complete = dict(offset=0.1, mode="none", frame_id=1, timestamp_ms=2)
    del complete[missing]
    with pytest.raises(AttributeError, match=missing):
        Phase2Output([], [SimpleNamespace(**complete)], 1, 2)


@pytest.mark.software
def test_f2_error_names_the_index_of_the_offending_item():
    complete = SimpleNamespace(type="lane_boundary", position={}, confidence=0.5,
                               timestamp=2, frame_id=1, source_roi="lane", source_rect=(0, 0, 1, 1))
    incomplete = SimpleNamespace(type="stop_sign", confidence=0.5,
                                 timestamp=2, frame_id=1, source_roi="sign", source_rect=(0, 0, 1, 1))
    with pytest.raises(AttributeError, match=r"\[1\]"):
        Phase2Output([complete, incomplete], [], 1, 2)


@pytest.mark.software
def test_f2_presence_is_the_only_check_a_none_valued_field_still_counts():
    """This stage checks field presence, not field validity - that's upstream's job."""
    partial = SimpleNamespace(type="lane_boundary", position=None, confidence=0.5,
                              timestamp=2, frame_id=1, source_roi="lane", source_rect=None)
    p = Phase2Output([partial], [], 1, 2)
    assert p.detection_count == 1


# =============================================================================
# Software: F3 - detection_count
# =============================================================================
@pytest.mark.software
def test_f3_detection_count_is_computed_when_omitted():
    assert Phase2Output([det(), det()], [], 1, 2).detection_count == 2


@pytest.mark.software
def test_f3_an_explicit_correct_count_is_accepted_including_zero():
    assert Phase2Output([det()], [], 1, 2, detection_count=1).detection_count == 1
    assert Phase2Output([], [], 1, 2, detection_count=0).detection_count == 0


@pytest.mark.software
def test_f3_an_explicit_wrong_count_is_rejected():
    with pytest.raises(ValueError, match="detection_count"):
        Phase2Output([det()], [], 1, 2, detection_count=5)


# =============================================================================
# Software: F4 - frame_id / timestamp_ms have no default
# =============================================================================
@pytest.mark.software
def test_f4_frame_id_has_no_default():
    with pytest.raises(TypeError, match="frame_id"):
        Phase2Output([], [], timestamp_ms=2)


@pytest.mark.software
def test_f4_timestamp_ms_has_no_default():
    with pytest.raises(TypeError, match="timestamp_ms"):
        Phase2Output([], [], frame_id=1)


@pytest.mark.software
def test_f4_neither_argument_silently_defaults_to_zero():
    with pytest.raises(TypeError):
        Phase2Output([], [])


# =============================================================================
# Software: F5 - a per-item stamp disagrees with the container
# =============================================================================
@pytest.mark.software
def test_f5_a_detection_from_a_different_frame_id_is_rejected():
    with pytest.raises(ValueError, match="different frames"):
        Phase2Output([det(frame_id=99)], [], 1, 2)


@pytest.mark.software
def test_f5_a_detection_from_a_different_timestamp_is_rejected():
    with pytest.raises(ValueError, match="different frames"):
        Phase2Output([det(ts=99)], [], 1, 2)


@pytest.mark.software
def test_f5_a_lane_offset_result_from_a_different_frame_is_rejected():
    with pytest.raises(ValueError, match="different frames"):
        Phase2Output([], [offset_result(frame_id=99)], 1, 2)


@pytest.mark.software
def test_f5_error_reports_both_stamps():
    with pytest.raises(ValueError, match=r"\(99, 2\).*\(1, 2\)"):
        Phase2Output([det(frame_id=99)], [], 1, 2)


@pytest.mark.software
def test_f5_a_correctly_stamped_mix_of_several_detections_is_accepted():
    p = Phase2Output([det(det_type="traffic_light"), det(det_type="lane_boundary"), det(det_type="stop_sign")],
                     [offset_result()], 1, 2)
    assert p.detection_count == 3 and len(p.lane_offset_results) == 1


# =============================================================================
# Software: PR-1 through PR-4
# =============================================================================
@pytest.mark.software
def test_pr1_items_are_kept_as_is_no_reordering_no_filtering():
    """Deliberately NOT in confidence order, to prove nothing re-sorts them."""
    weak, strong = det(confidence=0.1), det(confidence=0.9)
    p = Phase2Output([weak, strong], [], 1, 2)
    assert p.detections[0] is weak and p.detections[1] is strong


@pytest.mark.software
def test_pr1_lane_offset_results_are_kept_as_is():
    lo = offset_result()
    assert Phase2Output([], [lo], 1, 2).lane_offset_results[0] is lo


@pytest.mark.software
def test_pr1_more_than_one_lane_offset_result_is_not_rejected():
    """PR-1 takes the list as-is; enforcing exactly one is the caller's job (see package_phase2)."""
    p = Phase2Output([], [offset_result(mode="two_boundary"), offset_result(mode="left_only")], 1, 2)
    assert len(p.lane_offset_results) == 2


@pytest.mark.software
def test_pr2_detection_count_always_equals_len_detections():
    for n in (0, 1, 3):
        p = Phase2Output([det() for _ in range(n)], [], 1, 2)
        assert p.detection_count == n == len(p.detections)


@pytest.mark.software
def test_pr3_an_empty_frame_still_carries_its_identity():
    p = Phase2Output([], [], 42, 4242)
    assert (p.frame_id, p.timestamp_ms, p.detection_count) == (42, 4242, 0)
    assert p.detections == [] and p.lane_offset_results == []


@pytest.mark.software
def test_pr4_result_is_frozen():
    p = Phase2Output([], [], 1, 2)
    with pytest.raises(FrozenInstanceError):
        p.frame_id = 0
    with pytest.raises(FrozenInstanceError):
        p.detection_count = 99


@pytest.mark.software
def test_deterministic():
    a = Phase2Output([det(confidence=0.5), det(confidence=0.9)], [offset_result()], 1, 2)
    b = Phase2Output([det(confidence=0.5), det(confidence=0.9)], [offset_result()], 1, 2)
    assert a == b


# =============================================================================
# Software: package_phase2 wiring
# =============================================================================
@pytest.mark.software
def test_package_wires_fusion_detections_through_unchanged():
    d = det(frame_id=7, ts=8)
    fusion = FusionResult([d], 7, 8)
    out = package_phase2(fusion, None)
    assert out.detections is fusion.detections and out.detections[0] is d
    assert (out.frame_id, out.timestamp_ms) == (7, 8)


@pytest.mark.software
def test_package_wraps_a_lane_offset_result_in_a_singleton_list():
    fusion = FusionResult([], 7, 8)
    lo = offset_result(frame_id=7, ts=8)
    out = package_phase2(fusion, lo)
    assert out.lane_offset_results == [lo]


@pytest.mark.software
def test_package_with_no_lane_offset_gives_an_empty_list_not_none():
    out = package_phase2(FusionResult([], 7, 8), None)
    assert out.lane_offset_results == [] and out.detection_count == 0


@pytest.mark.software
def test_package_propagates_f5_when_the_lane_offset_is_from_a_different_frame():
    with pytest.raises(ValueError, match="different frames"):
        package_phase2(FusionResult([], 7, 8), offset_result(frame_id=999, ts=8))


@pytest.mark.software
def test_package_never_touches_detection_count_explicitly():
    """package_phase2 leaves detection_count to compute itself (PR-2)."""
    fusion = FusionResult([det(frame_id=1, ts=1), det(frame_id=1, ts=1)], 1, 1)
    assert package_phase2(fusion, None).detection_count == 2


# =============================================================================
# Software: chained  real fusion (+ lane offset) -> packaging
# =============================================================================
def build_frame(frame_id=21, ts=222, H=360, W=480):
    frame = np.full((H, W, 3), 30, np.uint8)
    probe = crop_rois(preprocess_frame(FrameData(frame, 0, 0)), ROIConfig())
    lx, ly, lw, lh = probe.lane_rect
    for frac in (0.2, 0.8):
        tx = lx + int(frac * lw)
        cv2.line(frame, (tx, ly + int(0.1 * lh)), (tx, ly + int(0.9 * lh)), (230, 230, 230), 8)
    return crop_rois(preprocess_frame(FrameData(frame, frame_id, ts)), ROIConfig())


@pytest.mark.software
def test_chain_real_fusion_and_lane_offset_package_cleanly():
    roi = build_frame()
    geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
    tls, _ = run_color_stage(roi, roi.traffic_roi, ColorConfig(HSVRanges(), BlobFilter()))
    fusion, _ = fuse_detections(geometry, tls, roi)
    lo, _ = compute_lane_offset(geometry, roi, LaneOffsetConfig())

    out = package_phase2(fusion, lo)
    assert (out.frame_id, out.timestamp_ms) == (roi.frame_id, roi.timestamp_ms)
    assert out.detections == fusion.detections
    assert out.lane_offset_results == [lo]
    assert out.detection_count == len(fusion.detections)


@pytest.mark.software
def test_chain_a_blind_frame_still_packages_with_identity_intact():
    roi = crop_rois(preprocess_frame(FrameData(np.full((360, 480, 3), 30, np.uint8), 30, 300)), ROIConfig())
    geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
    fusion, _ = fuse_detections(geometry, [], roi)
    lo, _ = compute_lane_offset(geometry, roi, LaneOffsetConfig())
    out = package_phase2(fusion, lo)
    assert out.detections == [] and out.detection_count == 0
    assert out.lane_offset_results[0].mode == "none"
    assert (out.frame_id, out.timestamp_ms) == (30, 300)


@pytest.mark.software
def test_every_recorded_frame_packages_without_a_stamp_mismatch(dataset_frames):
    if not dataset_frames:
        pytest.skip("no recorded dataset in tests/data/frames (run: pytest --hardware --record)")
    for fd in dataset_frames:
        roi = crop_rois(preprocess_frame(fd))
        geometry, _, _ = run_geometry_stage(roi, GeometryConfig())
        tls, _ = run_color_stage(roi, roi.traffic_roi, ColorConfig(HSVRanges(), BlobFilter()))
        fusion, _ = fuse_detections(geometry, tls, roi)
        lo, _ = compute_lane_offset(geometry, roi, LaneOffsetConfig())
        try:
            out = package_phase2(fusion, lo)
            assert (out.frame_id, out.timestamp_ms) == (fd.frame_id, fd.timestamp_ms)
            assert out.detection_count == len(out.detections)
        except AssertionError as e:
            raise AssertionError(f"frame_id={fd.frame_id}: {e}") from e


# =============================================================================
# Hardware: packaging characterization
# =============================================================================
def _jsonable(out: Phase2Output) -> dict:
    return {
        "frame_id": out.frame_id, "timestamp_ms": out.timestamp_ms,
        "detection_count": out.detection_count,
        "detections": [{"type": d.type, "label_detail": d.label_detail, "confidence": d.confidence,
                        "position": d.position, "source_roi": d.source_roi} for d in out.detections],
        "lane_offset_results": [{"mode": r.mode, "offset": r.offset, "confidence": r.confidence}
                                for r in out.lane_offset_results],
    }


@pytest.mark.hardware
def test_phase2_out_characterization(request, frames, artifacts):
    n = request.config.getoption("--frames")
    geo_cfg, color_cfg, lo_cfg = GeometryConfig(), ColorConfig(HSVRanges(), BlobFilter()), LaneOffsetConfig()

    rows, samples = [], {}
    for i, fd in enumerate(frames(n)):
        roi = crop_rois(preprocess_frame(fd))
        geometry, _, _ = run_geometry_stage(roi, geo_cfg)
        tls, _ = run_color_stage(roi, roi.traffic_roi, color_cfg)
        fusion, _ = fuse_detections(geometry, tls, roi)
        lo, _ = compute_lane_offset(geometry, roi, lo_cfg)             # upstream, not timed

        t0 = time.perf_counter_ns()
        out = package_phase2(fusion, lo)
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        assert out.detection_count == len(out.detections)             # outside the timing window
        assert (out.frame_id, out.timestamp_ms) == (roi.frame_id, roi.timestamp_ms)
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 4), out.detection_count,
                     len(out.lane_offset_results), out.lane_offset_results[0].mode if out.lane_offset_results else ""))
        if i in (0, n // 2, n - 1):
            samples[fd.frame_id] = out

    if not rows:
        pytest.skip("no frames delivered")

    stage = [r[2] for r in rows]
    artifacts.csv("phase2_out_timing.csv",
                  ["frame_id", "timestamp_ms", "stage_ms", "detection_count", "n_lane_offset", "mode"], rows)
    artifacts.json("summary.json", {
        "stage_ms": summarize(stage),
        "frames_with_zero_detections": sum(1 for r in rows if r[3] == 0),
        "detection_count": summarize([r[3] for r in rows]),
    })
    artifacts.histogram("stage_us_hist.png", [s * 1000 for s in stage],
                        "package_phase2 latency", "microseconds")
    for fid, out in samples.items():
        artifacts.json(f"{fid:06d}_phase2_output.json", _jsonable(out))