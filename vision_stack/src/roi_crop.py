"""
roi_crop.py

ROI Cropping Stage

Purpose:
    Partitions the preprocessed frame into three spatially distinct regions
    before the pipeline splits into parallel branches

    1. Reduce the pixel area each downstream operation must process

    2. Eliminate irrelevant regions from each branch's input input reduces false positive rate before any threshold is
       applied

Sources:
    lane_roi and sign_roi come from PreprocessResult.gray, because the
    geometry branch validates ndim == 2 and raises on a 3-channel input.

    traffic_roi comes from PreprocessResult.color, because the color branch
    thresholds in HSV and needs the chroma.

Aliasing:
    Each ROI is a NumPy view into the preprocessed frame

All ROI coordinates are computed deterministically from (H, W).
"""
import numpy as np
import cv2
from dataclasses import dataclass

from preprocess import PreprocessResult

# ============================================================================
# Debug Names
# ============================================================================
OVERLAY_SUFFIX = "_roi_overlay.png"
LANE_ROI_SUFFIX = "_roi_lane.png"
TRAFFIC_ROI_SUFFIX = "_roi_traffic.png"
SIGN_ROI_SUFFIX = "_roi_sign.png"

# ============================================================================
# ROI Configuration
# ============================================================================
@dataclass(frozen=True)
class ROIBounds:
    """
    Fractional bounds in [0, 1] of frame width/height

    x0, y0: top-left corner as a fraction of (W, H)
    x1, y1: bottom-right corner as a fraction of (W, H)
    """
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self):
        for name, v in (("x0", self.x0), ("y0", self.y0),
                        ("x1", self.x1), ("y1", self.y1)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"ROIBounds: {name} must be in [0,1], got {v}")
        # resolve() clamps x_end to at least x+1, so a reversed bound would
        # otherwise yield a silent 1px ROI instead of an error
        if self.x0 >= self.x1:
            raise ValueError(f"ROIBounds: x0 must be < x1, got {self.x0} >= {self.x1}")
        if self.y0 >= self.y1:
            raise ValueError(f"ROIBounds: y0 must be < y1, got {self.y0} >= {self.y1}")

# Bottom strip of the frame: the near-field road surface
LANE = ROIBounds(x0=0.05, y0=0.70, x1=0.95, y1=1.00)
# Top-center: where a light sits when the robot is square to an intersection
TRAFFIC = ROIBounds(x0=0.25, y0=0.00, x1=0.75, y1=0.50)
# Upper-right: signs are posted right of the lane
SIGN = ROIBounds(x0=0.50, y0=0.00, x1=1.00, y1=0.55)

@dataclass(frozen=True)
class ROIConfig:
    """
    Configuration for the ROI cropping stage

    lane: bounds for the lane ROI, taken from the grayscale frame
    traffic: bounds for the traffic ROI, taken from the color frame
    sign: bounds for the sign ROI, taken from the grayscale frame
    """
    lane: ROIBounds = LANE
    traffic: ROIBounds = TRAFFIC
    sign: ROIBounds = SIGN

# ============================================================================
# Result Container
# ============================================================================
@dataclass(frozen=True)
class ROICropResult:
    """
    Output of the ROI cropping stage

    lane_roi: (h, w) uint8 read-only view of the grayscale frame
    traffic_roi: (h, w, 3) uint8 read-only view of the color frame
    sign_roi: (h, w) uint8 read-only view of the grayscale frame
    lane_rect: (x, y, w, h) of lane_roi in source-frame pixels
    traffic_rect: (x, y, w, h) of traffic_roi in source-frame pixels
    sign_rect: (x, y, w, h) of sign_roi in source-frame pixels
    frame_id: carried from PreprocessResult
    timestamp_ms: carried from PreprocessResult
    source_shape: (H, W) of the frame the ROIs were cut from

    The *_rect fields are what converts an ROI-relative detection back into
    frame coordinates. Every branch works in ROI space, so without these a
    detection cannot be placed in the original image.
    """
    lane_roi: np.ndarray
    traffic_roi: np.ndarray
    sign_roi: np.ndarray
    lane_rect: tuple
    traffic_rect: tuple
    sign_rect: tuple
    frame_id: int
    timestamp_ms: int
    source_shape: tuple

# ============================================================================
# Validation
# ============================================================================
def _validate(
        frame: np.ndarray
    ) -> None:
    """
    Purpose:
        Reject frames that cannot be cropped, naming what arrived
    """
    if frame is None:
        raise ValueError("crop: frame is None")
    if frame.ndim not in (2, 3):
        raise ValueError(f"crop: expected 2 or 3 dims, got {frame.ndim}")
    if frame.ndim == 3 and frame.shape[2] != 3:
        raise ValueError(f"crop: expected (H,W) gray or (H,W,3) BGR, got {frame.shape}")

# ============================================================================
# Utility Functions
# ============================================================================
def resolve(
        bounds: ROIBounds,
        shape: tuple
    ) -> tuple:
    """
    Purpose:
        Convert fractional bounds into integer pixel coordinates for a given
        frame size, clamped to the frame

    Inputs:
        bounds: ROIBounds
        shape: (H, W) or (H, W, C); only the first two are read

    Outputs:
        (x, y, w, h) in source-frame pixels
    """
    H, W = shape[:2]
    x = min(max(round(bounds.x0 * W), 0), W - 1)
    y = min(max(round(bounds.y0 * H), 0), H - 1)
    x_end = min(max(round(bounds.x1 * W), x + 1), W)
    y_end = min(max(round(bounds.y1 * H), y + 1), H)
    return (x, y, x_end - x, y_end - y)

def crop(
        frame: np.ndarray,
        bounds: ROIBounds
    ) -> tuple:
    """
    Purpose:
        Cut one ROI out of a frame as a read-only view

    Inputs:
        frame: (H, W) or (H, W, 3) uint8
        bounds: ROIBounds

    Outputs:
        (roi_view, rect) where rect is (x, y, w, h) in source-frame pixels.
        roi_view is not writeable --- call .copy() if the consumer needs to
        modify it
    """
    _validate(frame)
    x, y, w, h = resolve(bounds, frame.shape)
    view = frame[y : y + h, x : x + w]
    view.flags.writeable = False
    return view, (x, y, w, h)

# ============================================================================
# ROI Cropping Stage
# ============================================================================
def crop_rois(
        pre: PreprocessResult,
        config: ROIConfig = ROIConfig()
    ) -> ROICropResult:
    """
    Purpose:
        Cut the three branch ROIs from one preprocessed frame and carry the
        frame's identity forward unchanged

    Inputs:
        pre: PreprocessResult from preprocess_frame()
        config: ROIConfig bounds

    Outputs:
        ROICropResult

    Notes:
        source_shape is taken from the grayscale frame. Both preprocessed
        frames come from the same capture, so their (H, W) agree; the
        mismatch check exists to catch a caller that assembled a
        PreprocessResult by hand from two different frames
    """
    _validate(pre.gray)
    _validate(pre.color)
    if pre.gray.shape[:2] != pre.color.shape[:2]:
        raise ValueError(
            f"crop: gray {pre.gray.shape[:2]} and color {pre.color.shape[:2]} "
            "disagree — they must come from the same frame"
        )

    lane_roi, lane_rect = crop(pre.gray, config.lane)
    sign_roi, sign_rect = crop(pre.gray, config.sign)
    traffic_roi, traffic_rect = crop(pre.color, config.traffic)

    return ROICropResult(
        lane_roi = lane_roi,
        traffic_roi = traffic_roi,
        sign_roi = sign_roi,
        lane_rect = lane_rect,
        traffic_rect = traffic_rect,
        sign_rect = sign_rect,
        frame_id = pre.frame_id,
        timestamp_ms = pre.timestamp_ms,
        source_shape = pre.gray.shape[:2],
    )

# ============================================================================
# Debug Visualization
# ============================================================================
LANE_COLOR = (0, 0, 255)
TRAFFIC_COLOR = (0, 255, 0)
SIGN_COLOR = (255, 0, 0)
RECT_THICKNESS = 1

def draw_roi_overlay(
        frame: np.ndarray,
        result: ROICropResult
    ) -> np.ndarray:
    """
    Purpose:
        Return a copy of frame with the three ROI rectangles drawn on it

    Inputs:
        frame: (H, W, 3) BGR frame to draw on, at source resolution
        result: ROICropResult whose rects are drawn

    Outputs:
        annotated copy; the input is not modified
    """
    overlay = frame.copy()

    def _draw(rect, color, label):
        x, y, w, h = rect
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, RECT_THICKNESS)
        cv2.putText(overlay, label, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    _draw(result.lane_rect, LANE_COLOR, "lane")
    _draw(result.traffic_rect, TRAFFIC_COLOR, "traffic")
    _draw(result.sign_rect, SIGN_COLOR, "sign")

    return overlay

# =============================================================================
# Self-test
#
#   python3 roi_crop.py             logic tests only, runs anywhere
#   python3 roi_crop.py --dataset   also regenerates the ROI fixtures
#
# Exits non-zero on any logic failure so it can gate a commit.
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    import traceback

    from capture import FrameData
    from preprocess import preprocess_frame, PreprocessParams

    H, W = 360, 480

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

    def expect_raises(exc_type, fn):
        try:
            fn()
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")

    def _pre(frame_id=0, timestamp_ms=1000):
        """A PreprocessResult built through the real stage, not by hand."""
        rng = np.random.default_rng(1)
        bgr = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        return preprocess_frame(FrameData(bgr, frame_id, timestamp_ms))

    # -------------------------------------------------------------------------
    # Geometry of the cut
    # -------------------------------------------------------------------------
    def t_rects_match_roi_shapes():
        """A rect that disagrees with its ROI silently misplaces every
        detection mapped back into frame coordinates."""
        r = crop_rois(_pre())
        for name, roi, rect in (
            ("lane", r.lane_roi, r.lane_rect),
            ("traffic", r.traffic_roi, r.traffic_rect),
            ("sign", r.sign_roi, r.sign_rect),
        ):
            assert roi.shape[:2] == (rect[3], rect[2]), (
                f"{name}: roi {roi.shape[:2]} vs rect (h,w) {(rect[3], rect[2])}"
            )

    def t_rects_stay_inside_the_frame():
        r = crop_rois(_pre())
        for name, (x, y, w, h) in (("lane", r.lane_rect),
                                   ("traffic", r.traffic_rect),
                                   ("sign", r.sign_rect)):
            assert 0 <= x and x + w <= W, f"{name} overruns width: {(x, w)}"
            assert 0 <= y and y + h <= H, f"{name} overruns height: {(y, h)}"

    def t_lane_roi_reaches_the_bottom_edge():
        """The near-field is the whole point of the lane ROI."""
        x, y, w, h = crop_rois(_pre()).lane_rect
        assert y + h == H, f"lane ROI stops {H - (y + h)}px short of the bottom"

    def t_known_resolution_gives_known_rects():
        """Pins the numbers geometry's harness is tuned against."""
        r = crop_rois(_pre())
        assert r.lane_rect == (24, 252, 432, 108), f"got {r.lane_rect}"
        assert r.traffic_rect == (120, 0, 240, 180), f"got {r.traffic_rect}"
        assert r.sign_rect == (240, 0, 240, 198), f"got {r.sign_rect}"

    # -------------------------------------------------------------------------
    # Channel routing
    # -------------------------------------------------------------------------
    def t_lane_and_sign_are_single_channel():
        """geometry raises on ndim != 2, so this is its entry contract."""
        r = crop_rois(_pre())
        assert r.lane_roi.ndim == 2, f"lane ndim {r.lane_roi.ndim}"
        assert r.sign_roi.ndim == 2, f"sign ndim {r.sign_roi.ndim}"

    def t_traffic_keeps_chroma():
        r = crop_rois(_pre())
        assert r.traffic_roi.ndim == 3, f"traffic ndim {r.traffic_roi.ndim}"
        assert r.traffic_roi.shape[2] == 3

    # -------------------------------------------------------------------------
    # Aliasing contract
    # -------------------------------------------------------------------------
    def t_rois_are_views_not_copies():
        pre = _pre()
        r = crop_rois(pre)
        assert r.lane_roi.base is pre.gray, "lane ROI was copied"
        assert r.traffic_roi.base is pre.color, "traffic ROI was copied"

    def t_rois_are_read_only():
        r = crop_rois(_pre())
        expect_raises(ValueError, lambda: r.lane_roi.__setitem__((0, 0), 1))

    def t_copy_of_an_roi_is_writable():
        """A branch that must modify its input opts in explicitly."""
        w = crop_rois(_pre()).lane_roi.copy()
        w[0, 0] = 1

    # -------------------------------------------------------------------------
    # Stamp passthrough
    # -------------------------------------------------------------------------
    def t_stamp_is_carried_not_rederived():
        r = crop_rois(_pre(frame_id=17, timestamp_ms=555222))
        assert (r.frame_id, r.timestamp_ms) == (17, 555222), (
            f"got {(r.frame_id, r.timestamp_ms)}"
        )

    def t_source_shape_is_h_w():
        r = crop_rois(_pre())
        assert r.source_shape == (H, W), f"got {r.source_shape}"

    def t_result_is_immutable():
        r = crop_rois(_pre())
        expect_raises(Exception, lambda: setattr(r, "frame_id", 3))

    # -------------------------------------------------------------------------
    # Bounds validation
    # -------------------------------------------------------------------------
    def t_reversed_bounds_rejected():
        expect_raises(ValueError, lambda: ROIBounds(x0=0.9, y0=0.0, x1=0.1, y1=1.0))

    def t_out_of_range_bounds_rejected():
        expect_raises(ValueError, lambda: ROIBounds(x0=0.0, y0=0.0, x1=1.5, y1=1.0))

    def t_custom_bounds_are_honored():
        cfg = ROIConfig(lane=ROIBounds(x0=0.0, y0=0.75, x1=1.0, y1=1.0))
        assert crop_rois(_pre(), cfg).lane_rect == (0, 270, 480, 90)

    def t_mismatched_frames_rejected():
        pre = _pre()
        bad = PreprocessResult(
            gray = pre.gray,
            color = np.zeros((H // 2, W, 3), np.uint8),
            frame_id = 0,
            timestamp_ms = 0,
        )
        expect_raises(ValueError, lambda: crop_rois(bad))

    # -------------------------------------------------------------------------
    # Overlay
    # -------------------------------------------------------------------------
    def t_overlay_does_not_modify_its_input():
        pre = _pre()
        r = crop_rois(pre)
        frame = pre.color.copy()
        before = frame.copy()
        draw_roi_overlay(frame, r)
        assert np.array_equal(frame, before), "overlay drew into its input"

    # -------------------------------------------------------------------------
    print("\nGeometry of the cut")
    check("rects match ROI shapes",              t_rects_match_roi_shapes)
    check("rects stay inside the frame",         t_rects_stay_inside_the_frame)
    check("lane ROI reaches the bottom edge",    t_lane_roi_reaches_the_bottom_edge)
    check("480x360 gives the expected rects",    t_known_resolution_gives_known_rects)

    print("\nChannel routing")
    check("lane and sign are single-channel",    t_lane_and_sign_are_single_channel)
    check("traffic keeps chroma",                t_traffic_keeps_chroma)

    print("\nAliasing")
    check("ROIs are views, not copies",          t_rois_are_views_not_copies)
    check("ROIs are read-only",                  t_rois_are_read_only)
    check("an explicit copy is writable",        t_copy_of_an_roi_is_writable)

    print("\nStamp")
    check("frame_id/timestamp carried through",  t_stamp_is_carried_not_rederived)
    check("source_shape is (H, W)",              t_source_shape_is_h_w)
    check("ROICropResult is immutable",          t_result_is_immutable)

    print("\nBounds")
    check("reversed bounds rejected",            t_reversed_bounds_rejected)
    check("out-of-range bounds rejected",        t_out_of_range_bounds_rejected)
    check("custom bounds honored",               t_custom_bounds_are_honored)
    check("mismatched gray/color rejected",      t_mismatched_frames_rejected)

    print("\nOverlay")
    check("overlay does not modify input",       t_overlay_does_not_modify_its_input)

    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} passed")

    # -------------------------------------------------------------------------
    # Dataset pass: regenerate the ROI fixtures geometry consumes
    # -------------------------------------------------------------------------
    if "--dataset" in sys.argv:
        SAMPLE_DIRS = (
            "vision_stack/frames/Sample1",
            "vision_stack/frames/Sample2",
            "vision_stack/frames/Sample3",
        )
        IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

        pre_params = PreprocessParams()
        roi_config = ROIConfig()
        seen_shapes = set()
        total_ok = 0
        total_fail = 0
        frame_id = 0

        print("\nDataset")
        for sample_dir in SAMPLE_DIRS:
            if not os.path.isdir(sample_dir):
                print(f"[SKIP] Not found: {sample_dir}")
                continue

            results_dir = os.path.join(sample_dir, "results")
            os.makedirs(results_dir, exist_ok=True)

            image_files = sorted(
                f for f in os.listdir(sample_dir)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            )
            if not image_files:
                print(f"[SKIP] No images in {sample_dir}")
                continue

            for filename in image_files:
                img_path = os.path.join(sample_dir, filename)
                stem = os.path.splitext(filename)[0]

                original = cv2.imread(img_path)
                if original is None:
                    print(f"[FAIL] Could not read: {img_path}")
                    total_fail += 1
                    continue

                # Run the real stages in order rather than reading back
                # preprocess's debug PNGs. The fixtures cannot drift from
                # what the live loop produces, because it is the same code.
                try:
                    pre = preprocess_frame(
                        FrameData(original, frame_id, 0), pre_params
                    )
                    result = crop_rois(pre, roi_config)
                except (ValueError, TypeError) as e:
                    print(f"[FAIL] {img_path}: {e}")
                    total_fail += 1
                    continue

                # Print the coordinate table once per unique resolution
                if result.source_shape not in seen_shapes:
                    seen_shapes.add(result.source_shape)
                    src_h, src_w = result.source_shape
                    print(f"\n[COORDS] Resolution {src_w}x{src_h}:")
                    print(f" lane_rect: {result.lane_rect}")
                    print(f" traffic_rect: {result.traffic_rect}")
                    print(f" sign_rect: {result.sign_rect}")

                cv2.imwrite(os.path.join(results_dir, f"{stem}{OVERLAY_SUFFIX}"),
                            draw_roi_overlay(original, result))
                cv2.imwrite(os.path.join(results_dir, f"{stem}{LANE_ROI_SUFFIX}"),
                            result.lane_roi)
                cv2.imwrite(os.path.join(results_dir, f"{stem}{TRAFFIC_ROI_SUFFIX}"),
                            result.traffic_roi)
                cv2.imwrite(os.path.join(results_dir, f"{stem}{SIGN_ROI_SUFFIX}"),
                            result.sign_roi)

                print(f"[OK] frame_id={frame_id} {img_path}")
                frame_id += 1
                total_ok += 1

        print(f"\nDone. {total_ok} processed, {total_fail} failed.")
    else:
        print("\nDataset: skipped (pass --dataset to regenerate fixtures)")

    sys.exit(0 if passed == len(_results) else 1)