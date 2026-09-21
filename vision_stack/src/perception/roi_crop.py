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

from src.perception.preprocess import PreprocessResult

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