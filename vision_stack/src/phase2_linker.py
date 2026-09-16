"""
phase2_linker.py

Phase 1 and 2 Pipeline Linker, excluding sign and traffic integration

Purpose:
    Defines the stage order once, in run_chain(), and tests the whole chain
    against answers known independently of the code:

        FrameData -> preprocess_frame -> crop_rois -> run_geometry_stage
                  -> compute_lane_offset

    Three kinds of check, because each catches what the others cannot:

    1. Ground truth (synthetic)
       Lane markings are drawn at known ROI-local x positions, so the correct
       offset is known by construction. This is the only place the contour
       anchoring in foot_x() is exercised: the CSV replay has no contour
       column and falls back to bbox centers on every candidate.

    2. Metamorphic properties
       Relations that must hold whatever the right answer is --- mirroring the
       frame must negate the offset, translating both markings must shift it
       predictably, the normalized offset must not depend on ROI width. These
       are stated independently of how the code computes anything, so unlike
       the unit tests they are not circular.

    3. Replay (--replay)
       The recorded Samples through the real chain, reporting where candidates
       are lost between the geometry branch accepting them and lane offset
       finding them usable.

Configuration:
    Every stage's tuning lives in PipelineConfig. A camera change moves the
    cm-per-pixel scale, the ROI bounds and the gate thresholds together, so
    re-tuning after a hardware change is meant to be a config swap and a
    re-run, not a code edit.

What this cannot test:
    Accuracy in centimetres against the +/-2cm requirement. That needs
    captures at measured lateral offsets on a real course. Synthetic ground
    truth proves the arithmetic recovers what was drawn; it says nothing
    about whether the camera sees the world the way the synthetic frames
    assume.
"""
import numpy as np
import cv2
from dataclasses import dataclass, field

from capture import FrameData
from preprocess import preprocess_frame, PreprocessParams
from roi_crop import crop_rois, ROIConfig
from geometry import run_geometry_stage, GeometryConfig
from lane_offset import compute_lane_offset, LaneOffsetConfig

# =============================================================================
# Pipeline Configuration
# =============================================================================
@dataclass(frozen=True)
class PipelineConfig:
    """
    Every stage's tuning as one unit

    preprocess: conditioning parameters
    roi: ROI bounds
    geometry: contour filters and edge detection
    lane_offset: boundary gates and the calibration constant
    """
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    roi: ROIConfig = field(default_factory=ROIConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lane_offset: LaneOffsetConfig = field(default_factory=LaneOffsetConfig)

# Gates revised from the 4827-candidate CSV sweep rather than from course
# dimensions. min_proximity sat above the observed median of 0.17 and killed
# 1752 candidates the geometry branch had already scored at or above 0.30;
# max_width_px at 25 clipped detections whose p90 is 27 and max is 41.7.
MEASURED = PipelineConfig(
    lane_offset = LaneOffsetConfig(
        conf_threshold = 0.25,
        min_proximity = 0.05,
        max_width_px = 45.0,
        min_intensity = 130.0,
    )
)

# =============================================================================
# Chain Result
# =============================================================================
@dataclass(frozen=True)
class ChainResult:
    """
    Every stage's output for one frame, so a failure can be traced to the
    stage that produced it rather than only to the final number
    """
    frame: FrameData
    pre: object
    roi: object
    geometry: object
    offset: object
    lane_debug: dict
    offset_debug: dict

# =============================================================================
# The Chain
# =============================================================================
def run_chain(
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: int = 0,
        config: PipelineConfig = PipelineConfig(),
        draw_overlays: bool = False,
    ) -> ChainResult:
    """
    Purpose:
        Run one frame through every stage. This is the only place the stage
        order is written down; the harnesses below and the live loop should
        both come through here so they cannot drift

    Inputs:
        frame_bgr: (H, W, 3) uint8 BGR frame as CameraSource.read() delivers it
        frame_id, timestamp_ms: the stamp capture would have assigned
        config: PipelineConfig
        draw_overlays: build the geometry debug overlays

    Outputs:
        ChainResult
    """
    fd = FrameData(frame_bgr, frame_id, timestamp_ms)
    pre = preprocess_frame(fd, config.preprocess)
    roi = crop_rois(pre, config.roi)
    geo, lane_debug, _sign_debug = run_geometry_stage(
        roi, config.geometry, draw_overlays
    )
    offset, offset_debug = compute_lane_offset(geo, roi, config.lane_offset)
    return ChainResult(fd, pre, roi, geo, offset, lane_debug, offset_debug)

# =============================================================================
# Synthetic Frame Construction
# =============================================================================
FRAME_H, FRAME_W = 360, 480
LANE_RECT = (24, 252, 432, 108)          # crop_rois output at 480x360
ROI_W, ROI_H = LANE_RECT[2], LANE_RECT[3]
ROI_CENTER = ROI_W / 2.0                 # 216.0

def synthetic_frame(
        marks,
        mark_width: int = 6,
        road: int = 60,
        surround: int = 30,
        marking: int = 240,
    ) -> np.ndarray:
    """
    Purpose:
        Build a BGR frame whose lane ROI contains markings at known
        ROI-local x positions, so the correct lane offset is known exactly

    Inputs:
        marks: iterable of ROI-local x positions, or of
               (x, y_top, y_bottom) to control vertical extent for dash and
               partial-visibility cases
        mark_width: marking width in px
        road, surround, marking: intensities for the road surface inside the
               lane ROI, everything outside it, and the markings

    Outputs:
        (360, 480, 3) uint8 BGR

    Notes:
        A marking drawn at ROI x is recovered by the chain at ROI x. Verified
        to the pixel: drawn at 150 and 290, detected at 150.0 and 290.0
    """
    x0, y0, w, h = LANE_RECT
    frame = np.full((FRAME_H, FRAME_W, 3), surround, np.uint8)
    frame[y0:y0 + h, x0:x0 + w] = road

    for mark in marks:
        if isinstance(mark, (int, float)):
            x, top, bottom = mark, 0, h
        else:
            x, top, bottom = mark
        fx = x0 + int(x)
        cv2.rectangle(
            frame,
            (fx - mark_width // 2, y0 + int(top)),
            (fx + mark_width // 2, y0 + int(bottom) - 1),
            (marking,) * 3, -1,
        )
    return frame

def expected_offset(left_x: float, right_x: float) -> float:
    """The offset the chain must recover for markings at these ROI x."""
    lane_center = (left_x + right_x) / 2.0
    return (ROI_CENTER - lane_center) / ROI_CENTER

# =============================================================================
# Self-test
#
#   python3 pipeline_linker.py            ground truth + metamorphic
#   python3 pipeline_linker.py --replay   also replay the recorded Samples
#
# Exits non-zero on any failure so it can gate a commit.
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    import traceback

    CFG = MEASURED
    ANCHOR_TOL = 3.0        # px; synthetic markings recover to the pixel
    OFFSET_TOL = 0.02       # normalized

    _results = []

    def check(name, fn):
        """Run one test, record pass/fail, never abort the suite."""
        try:
            fn()
        except Exception:
            _results.append((name, False))
            print(f"  FAIL  {name}")
            for line in traceback.format_exc().strip().splitlines()[-2:]:
                print(f"        {line.strip()}")
        else:
            _results.append((name, True))
            print(f"  ok    {name}")

    def chain(marks, config=None, **kw):
        return run_chain(synthetic_frame(marks, **kw), 11, 4242,
                         config or CFG)

    # -------------------------------------------------------------------------
    # Chain integrity
    # -------------------------------------------------------------------------
    def t_every_stage_produces_output():
        r = chain([150, 290])
        for name in ("pre", "roi", "geometry", "offset"):
            assert getattr(r, name) is not None, f"{name} produced nothing"

    def t_one_stamp_survives_all_five_stages():
        r = chain([150, 290])
        stamps = {
            "frame": (r.frame.frame_id, r.frame.timestamp_ms),
            "preprocess": (r.pre.frame_id, r.pre.timestamp_ms),
            "roi": (r.roi.frame_id, r.roi.timestamp_ms),
            "geometry": (r.geometry.frame_id, r.geometry.timestamp_ms),
            "lane_offset": (r.offset.frame_id, r.offset.timestamp_ms),
        }
        assert set(stamps.values()) == {(11, 4242)}, f"stamps diverged: {stamps}"

    def t_roi_geometry_matches_the_fixtures():
        """Every expected value below assumes this rect."""
        assert chain([150]).roi.lane_rect == LANE_RECT

    # -------------------------------------------------------------------------
    # Ground truth: the chain must recover what was drawn
    # -------------------------------------------------------------------------
    def t_markings_are_recovered_at_the_drawn_position():
        """Exercises foot_x() on real contours, which the CSV replay cannot."""
        r = chain([150, 290])
        got = sorted(c.bbox[0] + c.bbox[2] / 2 for c in r.geometry.lane_candidates)
        assert len(got) == 2, f"expected 2 candidates, got {len(got)}"
        for want, have in zip((150, 290), got):
            assert abs(have - want) <= ANCHOR_TOL, f"drawn {want}, found {have:.1f}"

    def t_centered_lane_reads_zero():
        r = chain([150, 282])            # center 216 == ROI center
        assert r.offset.mode == "two_boundary", f"got {r.offset.mode}"
        assert abs(r.offset.offset) <= OFFSET_TOL, f"got {r.offset.offset:+.4f}"

    def t_lane_left_of_camera_reads_positive():
        left, right = 90, 270            # lane center 180
        r = chain([left, right])
        want = expected_offset(left, right)
        assert want > 0
        assert abs(r.offset.offset - want) <= OFFSET_TOL, (
            f"want {want:+.4f}, got {r.offset.offset:+.4f}"
        )

    def t_lane_right_of_camera_reads_negative():
        left, right = 180, 320           # lane center 250
        r = chain([left, right])
        want = expected_offset(left, right)
        assert want < 0
        assert abs(r.offset.offset - want) <= OFFSET_TOL, (
            f"want {want:+.4f}, got {r.offset.offset:+.4f}"
        )

    def t_recovered_lane_width_matches_the_drawn_spacing():
        r = chain([120, 300])
        assert abs(r.offset.lane_width_px - 180) <= 2 * ANCHOR_TOL, (
            f"drew 180px apart, measured {r.offset.lane_width_px}"
        )

    def t_three_markings_pick_the_robots_own_lane():
        """Outer-left edge, dividing line, outer-right edge. The bracketing
        pair is (200, 340); the outermost pair would be (60, 340)."""
        r = chain([60, 200, 340])
        assert r.offset.mode == "two_boundary", f"got {r.offset.mode}"
        assert (abs(r.offset.left_x - 200) <= ANCHOR_TOL
                and abs(r.offset.right_x - 340) <= ANCHOR_TOL), (
            f"picked ({r.offset.left_x}, {r.offset.right_x}), want (200, 340)"
        )
        want = expected_offset(200, 340)
        assert abs(r.offset.offset - want) <= OFFSET_TOL, (
            f"want {want:+.4f}, got {r.offset.offset:+.4f}"
        )

    def t_extremes_would_have_given_a_different_answer():
        """Confirms the fixture actually discriminates between the two rules."""
        assert abs(expected_offset(60, 340) - expected_offset(200, 340)) > 0.3

    def t_single_marking_is_uncalibrated_by_default():
        r = chain([150])
        assert r.offset.mode == "single_uncalibrated", f"got {r.offset.mode}"
        assert r.offset.offset == 0.0 and r.offset.confidence == 0.0

    def t_single_marking_with_calibration_projects_the_center():
        """A left marking 66px from ROI center with a 66px half-lane means
        the robot is centered."""
        cfg = PipelineConfig(
            preprocess=CFG.preprocess, roi=CFG.roi, geometry=CFG.geometry,
            lane_offset=LaneOffsetConfig(
                conf_threshold=0.25, min_proximity=0.05, max_width_px=45.0,
                min_intensity=130.0, expected_half_lane_px=66.0,
            ),
        )
        r = chain([ROI_CENTER - 66], config=cfg)
        assert r.offset.mode == "left_only", f"got {r.offset.mode}"
        assert abs(r.offset.offset) <= OFFSET_TOL, f"got {r.offset.offset:+.4f}"

    def t_empty_road_goes_blind_cleanly():
        r = run_chain(synthetic_frame([]), 0, 0, CFG)
        assert r.offset.mode == "none"
        assert r.offset.offset == 0.0 and r.offset.boundary_count == 0

    # -------------------------------------------------------------------------
    # Metamorphic properties
    # -------------------------------------------------------------------------
    def t_mirroring_the_frame_negates_the_offset():
        """The lane ROI spans x 0.05 to 0.95, symmetric about frame center, so
        a horizontal flip maps it onto itself. Would have caught the
        left_x - right_x sign inversion immediately."""
        frame = synthetic_frame([100, 220])
        straight = run_chain(frame, 0, 0, CFG).offset.offset
        mirrored = run_chain(np.ascontiguousarray(frame[:, ::-1]), 0, 0,
                             CFG).offset.offset
        assert abs(straight + mirrored) <= OFFSET_TOL, (
            f"{straight:+.4f} and {mirrored:+.4f} should sum to zero"
        )
        assert abs(straight) > 0.05, "fixture is too close to symmetric to test"

    def t_translating_both_markings_shifts_the_offset_predictably():
        shift = 40
        a = chain([120, 260]).offset.offset
        b = chain([120 + shift, 260 + shift]).offset.offset
        assert abs((a - b) - shift / ROI_CENTER) <= OFFSET_TOL, (
            f"shifted {shift}px: offset moved {a-b:+.4f}, "
            f"expected {shift/ROI_CENTER:+.4f}"
        )

    def t_offset_is_normalized_to_roi_width():
        """Same relative geometry in a different ROI must give the same
        normalized offset, or the signal is not resolution-independent."""
        from roi_crop import ROIBounds
        narrow = PipelineConfig(
            preprocess=CFG.preprocess,
            roi=ROIConfig(lane=ROIBounds(x0=0.15, y0=0.70, x1=0.85, y1=1.00)),
            geometry=CFG.geometry, lane_offset=CFG.lane_offset,
        )
        wide_r = chain([108, 324])                  # quarter and 3/4 of 432
        w2 = narrow.roi.lane.x1 - narrow.roi.lane.x0
        narrow_w = round(w2 * FRAME_W)
        x0_narrow = round(narrow.roi.lane.x0 * FRAME_W)
        # same fractional positions, expressed in the narrow ROI's frame coords
        marks = [(0.25 * narrow_w) + x0_narrow - LANE_RECT[0],
                 (0.75 * narrow_w) + x0_narrow - LANE_RECT[0]]
        narrow_r = chain(marks, config=narrow)
        assert narrow_r.offset.mode == "two_boundary", f"got {narrow_r.offset.mode}"
        assert abs(wide_r.offset.offset - narrow_r.offset.offset) <= 0.05, (
            f"wide {wide_r.offset.offset:+.4f} vs narrow "
            f"{narrow_r.offset.offset:+.4f}"
        )

    def t_marking_order_does_not_matter():
        forward = chain([90, 200, 330]).offset.offset
        reverse = chain([330, 200, 90]).offset.offset
        assert forward == reverse, f"{forward:+.4f} vs {reverse:+.4f}"

    def t_the_chain_is_deterministic():
        frame = synthetic_frame([150, 290])
        a = run_chain(frame, 0, 0, CFG).offset
        b = run_chain(frame, 0, 0, CFG).offset
        assert (a.offset, a.left_x, a.right_x) == (b.offset, b.left_x, b.right_x)

    def t_marking_thickness_does_not_move_the_anchor():
        """Anchor position must come from where the marking is, not how fat
        it is, or the offset drifts with lighting and blur."""
        thin = chain([150, 290], mark_width=4).offset.offset
        thick = chain([150, 290], mark_width=10).offset.offset
        assert abs(thin - thick) <= OFFSET_TOL, f"{thin:+.4f} vs {thick:+.4f}"

    # -------------------------------------------------------------------------
    print("\nChain integrity")
    check("every stage produces output",         t_every_stage_produces_output)
    check("one stamp survives all five stages",  t_one_stamp_survives_all_five_stages)
    check("lane_rect matches the fixtures",      t_roi_geometry_matches_the_fixtures)

    print("\nGround truth")
    check("markings recovered where drawn",      t_markings_are_recovered_at_the_drawn_position)
    check("centered lane reads zero",            t_centered_lane_reads_zero)
    check("lane left of camera -> positive",     t_lane_left_of_camera_reads_positive)
    check("lane right of camera -> negative",    t_lane_right_of_camera_reads_negative)
    check("lane width matches drawn spacing",    t_recovered_lane_width_matches_the_drawn_spacing)
    check("three markings pick own lane",        t_three_markings_pick_the_robots_own_lane)
    check("fixture discriminates the two rules", t_extremes_would_have_given_a_different_answer)
    check("single marking is uncalibrated",      t_single_marking_is_uncalibrated_by_default)
    check("calibrated single projects center",   t_single_marking_with_calibration_projects_the_center)
    check("empty road goes blind cleanly",       t_empty_road_goes_blind_cleanly)

    print("\nMetamorphic properties")
    check("mirroring negates the offset",        t_mirroring_the_frame_negates_the_offset)
    check("translation shifts predictably",      t_translating_both_markings_shifts_the_offset_predictably)
    check("offset is normalized to ROI width",   t_offset_is_normalized_to_roi_width)
    check("marking order does not matter",       t_marking_order_does_not_matter)
    check("the chain is deterministic",          t_the_chain_is_deterministic)
    check("thickness does not move the anchor",  t_marking_thickness_does_not_move_the_anchor)

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} passed")

    # -------------------------------------------------------------------------
    # Replay: recorded frames through the real chain
    #
    # Unlike the CSV sweep this runs the contours, so foot_x() does real work.
    # Reports where candidates are lost between the geometry branch accepting
    # them and lane offset finding them usable.
    # -------------------------------------------------------------------------
    if "--replay" not in sys.argv:
        print("\nReplay: skipped (pass --replay to run the recorded Samples)")
        sys.exit(0 if passed == len(_results) else 1)

    SAMPLE_DIRS = ("vision_stack/frames/Sample1",
                   "vision_stack/frames/Sample2",
                   "vision_stack/frames/Sample3")
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    modes = {}
    accepted_total = usable_total = 0
    frames = 0
    two_candidate_frames = 0
    offsets = []
    gate_hits = {}
    blind_run = cur_run = 0

    print("\nReplay")
    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Not found: {sample_dir}")
            continue
        image_files = sorted(
            f for f in os.listdir(sample_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            print(f"[SKIP] No images in {sample_dir}")
            continue

        for filename in image_files:
            original = cv2.imread(os.path.join(sample_dir, filename))
            if original is None:
                continue
            r = run_chain(original, frames, 0, CFG)
            frames += 1

            n_accepted = len(r.geometry.lane_candidates)
            accepted_total += n_accepted
            usable_total += r.offset.boundary_count
            if n_accepted >= 2:
                two_candidate_frames += 1

            modes[r.offset.mode] = modes.get(r.offset.mode, 0) + 1
            for entry in r.offset_debug["log"]:
                if entry.startswith("[REJECT]"):
                    gate_hits[entry.split()[1]] = \
                        gate_hits.get(entry.split()[1], 0) + 1
                elif entry.startswith("[MERGE]") or entry.startswith("[SPAN]"):
                    key = entry.split()[0].strip("[]")
                    gate_hits[key] = gate_hits.get(key, 0) + 1

            if r.offset.mode in ("two_boundary", "left_only", "right_only"):
                offsets.append(r.offset.offset)
                cur_run = 0
            else:
                cur_run += 1
                blind_run = max(blind_run, cur_run)

    if frames == 0:
        print("  no frames found")
        sys.exit(0 if passed == len(_results) else 1)

    print(f"\n[REPLAY] {frames} frames through the full chain")
    print(f" candidates accepted by geometry   {accepted_total:6}  "
          f"({accepted_total/frames:.2f} per frame)")
    print(f" usable as boundaries              {usable_total:6}  "
          f"({usable_total/frames:.2f} per frame, need 2.00)")
    if accepted_total:
        print(f" survival rate                     "
              f"{100*usable_total/accepted_total:5.1f}%")
    print(f" frames with >=2 candidates        {two_candidate_frames:6}  "
          f"({100*two_candidate_frames/frames:5.1f}%)")

    print(f"\n[MODES] {frames} frames:")
    for mode in ("two_boundary", "left_only", "right_only",
                 "single_uncalibrated", "none"):
        n = modes.get(mode, 0)
        flag = "   <-- never" if n == 0 else ""
        print(f" {mode:<22}{n:6}  ({100*n/frames:5.1f}%){flag}")

    print(f"\n[LOSSES] why a candidate or a pair was not used:")
    if not gate_hits:
        print("  nothing rejected")
    for gate, n in sorted(gate_hits.items(), key=lambda kv: -kv[1]):
        print(f" {gate:<22}{n:6}")

    if offsets:
        s = sorted(offsets)
        print(f"\n[OFFSETS] {len(offsets)} frames produced a steering signal")
        print(f" min {s[0]:+.3f}  med {s[len(s)//2]:+.3f}  max {s[-1]:+.3f}")
    else:
        print("\n[OFFSETS] no frame produced a steering signal")

    print(f"\n[BLIND] longest run without a steering signal: {blind_run} frames")

    sys.exit(0 if passed == len(_results) else 1)