"""Region-of-interest cropping that splits each frame into per-branch inputs.

Purpose:
    Partitions the preprocessed frame into three regions before the pipeline
    splits into parallel branches. Cropping shrinks the pixel area each
    branch processes and removes irrelevant scenery, which lowers the
    false-positive rate before any threshold is applied.

Main package:
    ROICropResult: lane and sign ROIs cut from the gray frame (the geometry
    branch rejects 3-channel input), a traffic ROI cut from the color frame
    (HSV thresholding needs chroma), and each ROI's rect in source pixels so
    ROI-space detections can be mapped back to the frame. ROIs are read-only
    views, not copies.

Flow:
    1. Check that the gray and color frames share (H, W).
    2. Resolve each fractional ROIBounds to pixel coordinates.
    3. Slice each ROI as a read-only view; carry the frame identity forward.
"""
import numpy as np
import cv2
from dataclasses import dataclass

from src.perception.preprocess import PreprocessResult

from src.params import ROI_LANE, ROI_SIGN, ROI_TRAFFIC
# Re-exported so debug harnesses can keep importing the suffixes from here
from src.params import (
    LANE_ROI_SUFFIX, ROI_OVERLAY_SUFFIX as OVERLAY_SUFFIX, SIGN_ROI_SUFFIX, TRAFFIC_ROI_SUFFIX,
)


@dataclass(frozen=True)
class ROIBounds:
    """Fractional ROI bounds, each in [0, 1] of frame width/height. x0 < x1 and y0 < y1."""
    x0: float   # top-left
    y0: float
    x1: float   # bottom-right
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
    """Bounds for each ROI. Lane and sign are cut from the gray frame, traffic from the color frame."""
    lane: ROIBounds = LANE
    traffic: ROIBounds = TRAFFIC
    sign: ROIBounds = SIGN


@dataclass(frozen=True)
class ROICropResult:
    """
    Output of the ROI cropping stage.

    ROIs are read-only views into the preprocessed frames. Every branch works
    in ROI space; the rects are what place its detections back in the frame.
    frame_id and timestamp_ms are copied from PreprocessResult.
    """
    lane_roi: np.ndarray                        # (h, w) uint8 view of the gray frame
    traffic_roi: np.ndarray                     # (h, w, 3) uint8 BGR view of the color frame
    sign_roi: np.ndarray                        # (h, w) uint8 view of the gray frame
    lane_rect: tuple[int, int, int, int]        # (x, y, w, h) in source-frame px
    traffic_rect: tuple[int, int, int, int]     # (x, y, w, h) in source-frame px
    sign_rect: tuple[int, int, int, int]        # (x, y, w, h) in source-frame px
    frame_id: int
    timestamp_ms: int
    source_shape: tuple[int, int]               # (H, W) the ROIs were cut from


def _validate(
        frame: np.ndarray
    ) -> None:
    """Reject frames that can't be cropped, naming what arrived."""
    if frame is None:
        raise ValueError("crop: frame is None")
    if frame.ndim not in (2, 3):
        raise ValueError(f"crop: expected 2 or 3 dims, got {frame.ndim}")
    if frame.ndim == 3 and frame.shape[2] != 3:
        raise ValueError(f"crop: expected (H,W) gray or (H,W,3) BGR, got {frame.shape}")


def resolve(
        bounds: ROIBounds,
        shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
    """
    Convert fractional bounds to integer pixel coordinates, clamped to the frame.

    Inputs:
        shape: (H, W) or (H, W, C); only (H, W) is read.

    Outputs:
        (x, y, w, h) in source-frame px. Always at least 1x1, inside the
        frame, and the same for the same (bounds, H, W).
    """
    H, W = shape[:2]
    x = min(max(round(bounds.x0 * W), 0), W - 1)
    y = min(max(round(bounds.y0 * H), 0), H - 1)
    # max(..., x + 1): sliver bounds narrower than a pixel still yield 1 px
    x_end = min(max(round(bounds.x1 * W), x + 1), W)
    y_end = min(max(round(bounds.y1 * H), y + 1), H)
    return (x, y, x_end - x, y_end - y)

def crop(
        frame: np.ndarray,
        bounds: ROIBounds
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Cut one ROI out of a frame as a read-only view.

    Inputs:
        frame: (H, W) gray or (H, W, 3) BGR, uint8. The ROI keeps the same layout.

    Outputs:
        (roi, rect). roi is a read-only view; call .copy() to modify it.
        rect is (x, y, w, h) in source-frame px.

    Raises:
        ValueError: If the frame is None or not (H, W) / (H, W, 3).
    """
    _validate(frame)
    x, y, w, h = resolve(bounds, frame.shape)
    view = frame[y : y + h, x : x + w]
    view.flags.writeable = False    # locks only the view; the source stays writeable
    return view, (x, y, w, h)


def crop_rois(
        pre: PreprocessResult,
        config: ROIConfig = ROIConfig()
    ) -> ROICropResult:
    """
    Cut the three branch ROIs from one preprocessed frame.

    Inputs:
        config: ROI bounds. Defaults to LANE, TRAFFIC and SIGN.

    Outputs:
        ROICropResult, with frame_id and timestamp_ms carried forward unchanged.

    Raises:
        ValueError: If either frame is malformed, or gray and color differ in (H, W).
    """
    _validate(pre.gray)
    _validate(pre.color)
    # preprocess_frame can't produce a mismatch; this catches a hand-assembled
    # PreprocessResult mixing two frames, which would misplace every rect
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


# BGR order: lane red, traffic green, sign blue
LANE_COLOR = (0, 0, 255)
TRAFFIC_COLOR = (0, 255, 0)
SIGN_COLOR = (255, 0, 0)
RECT_THICKNESS = 1

def draw_roi_overlay(
        frame: np.ndarray,
        result: ROICropResult
    ) -> np.ndarray:
    """
    Draw the three ROI rectangles, labeled, on a copy of the frame.

    Inputs:
        frame: (H, W, 3) BGR at the resolution the ROIs were cut from.

    Outputs:
        Annotated copy; the input is untouched.
    """
    overlay = frame.copy()

    def _draw(rect, color, label):
        x, y, w, h = rect
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, RECT_THICKNESS)
        # (x + 4, y + 18) is the text baseline: keeps a 0.55-scale label inside the rect
        cv2.putText(overlay, label, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    _draw(result.lane_rect, LANE_COLOR, ROI_LANE)
    _draw(result.traffic_rect, TRAFFIC_COLOR, ROI_TRAFFIC)
    _draw(result.sign_rect, SIGN_COLOR, ROI_SIGN)

    return overlay