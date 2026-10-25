"""Geometry's shared parts: the configs, the candidate types, and the edge helpers more than one detector uses.

Purpose:
    Each detector (geometry_lanes, geometry_stop_line, geometry_sign) lives
    in its own module; what more than one needs lives here, so no detector
    imports another.

Main package:
    GeometryBranchResult and the GeometryConfig that tunes the detectors.
"""
import math

import cv2
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.params import FOOT_BAND_PX


@dataclass
class CannyParams:
    threshold1: float = 80.0        # hysteresis low; raising it fractures lines into more pieces
    threshold2: float = 200.0       # hysteresis high; raising it seeds fewer contours
    # 3, 5 or 7; larger scales the gradients, so the thresholds need retuning
    aperture_size: int = 3
    # Lane branch only; width < 2 disables closing
    close_kernel: tuple[int, int] = (9, 3)


@dataclass
class LaneContourFilter:
    min_area: float = 1.0           # px^2
    max_area: float = 1000.0        # px^2
    min_aspect: float = 0.0         # elongation: minAreaRect long / short side
    max_aspect: float = 60.0
    max_roi_span: float = 1.0       # max fraction of the ROI a contour may span along its own axis
    min_intensity: float = 120.0    # 0-255 mean inside the contour; rejects dark blobs such as seams
    ref_length: float = 0.25        # long side, as a fraction of ROI extent, that scores full length confidence
    ref_width: float = 30.0         # short side, px, that scores full width confidence
    # Stop lines are taken out of the lane edges first, so one touching the
    # lane lines can't close into one contour with them. None turns it off
    horizontal_edge_deg: float | None = 20.0
    # Longer than any tape is wide (45 px), so a tape's own ends stay
    horizontal_min_run_px: float = 46.0
    horizontal_band_px: float = 3.0


@dataclass
class SignContourFilter:
    min_area: float = 100.0         # px^2
    max_area: float = 30000.0       # px^2
    min_vertices: int = 8           # approxPolyDP vertex count
    max_vertices: int = 9
    min_solidity: float = 0.80      # contour area / convex hull area
    # Of the hull's perimeter: 8-9 vertices on every synthetic sign, radius 12-30 px
    epsilon_factor: float = 0.02
    ref_area: float = 5000.0        # px^2 that scores full area confidence
    # Floor under Otsu, which splits even a sign-free ROI's noise (Otsu ~1 there;
    # the dimmest synthetic sign had 22)
    min_redness: float = 20.0
    # Fills the notches the letters leave at the rim; larger merges nearby red
    close_kernel: int = 5
    # The ghost-sign gates (off by default, set in MEASURED): any roundish red
    # blob has 8-9 hull vertices, so a leg or a chair passed. Box w / h
    aspect_range: tuple[float, float] | None = None
    # Contour area / box area: octagon 0.83, rectangle 1.0, disc 0.79
    extent_range: tuple[float, float] | None = None
    # Share of the hull that must be white letters
    min_letter_share: float | None = None
    letter_min_v: int = 150
    letter_max_redness: int = 60
    # Smaller signs skip the letters gate: their letters blur into the red
    letters_min_area: float = 2000.0


@dataclass(frozen=True)
class StopLineFilter:
    """Lengths and thicknesses in lane-ROI px, along the fitted line and across it."""
    # Also sets the gradient split: steeper edges never reach the detector
    max_tilt_deg: float = 20.0
    # Longer than any tape is wide (~41 px), so a dash's end can't pass
    min_length_px: float = 60.0
    min_thickness_px: float = 3.0
    max_thickness_px: float = 40.0
    min_intensity: float = 70.0        # 0-255 mean between the paired edges; tape, not a shadow edge
    # Paired edges must overlap by this fraction of the shorter one
    min_edge_overlap: float = 0.5
    # Bridges gaps along the line, such as where a lane line crosses it
    close_kernel: tuple[int, int] = (15, 1)
    ref_length_px: float = 200.0        # length that scores full length confidence, about one lane


@dataclass(frozen=True)
class GeometryConfig:
    canny: CannyParams = field(default_factory=CannyParams)     # lanes and stop lines; signs are found by color
    lane: LaneContourFilter = field(default_factory=LaneContourFilter)
    sign: SignContourFilter = field(default_factory=SignContourFilter)
    stop_line: StopLineFilter = field(default_factory=StopLineFilter)   # reads the lane ROI's Canny edges


def contour_foot_x(
        contour: np.ndarray,
        band_px: int = FOOT_BAND_PX,
    ) -> float:
    """
    Mean x of the contour's lowest band_px rows: where a diagonal marking meets the near
    edge, which its bbox center misses by up to half its width. -1.0 for an empty contour.
    """
    if contour is None or len(contour) == 0:
        return -1.0
    pts = np.asarray(contour).reshape(-1, 2)
    y_max = pts[:, 1].max()
    band = pts[pts[:, 1] >= y_max - max(band_px, 0)]
    return round(float(band[:, 0].mean()), 2)


@dataclass
class LaneCandidate:
    label: str                          # always "lane_boundary"
    bbox: tuple[int, int, int, int]     # (x, y, w, h)
    contour: np.ndarray                 # (N, 1, 2); a merged candidate holds every member's points
    confidence: float                   # [0, 1] measurement quality, used as a weight
    frame_id: int
    timestamp_ms: int
    proximity: float = 0.0              # [0, 1] bottom-edge position; 1.0 = bottom of ROI, nearest the robot
    width_px: float = 0.0               # minAreaRect short side, px; raw, not floored to 1 like the aspect gate
    length_px: float = 0.0              # minAreaRect long side, px
    mean_intensity: float = 10.0         # 0-255 inside the filled contour; length-weighted when merged
    foot_x: float = -1.0                # see contour_foot_x; -1.0 = not computed


@dataclass
class SignCandidate:
    label: str                          # always "stop_sign"
    bbox: tuple[int, int, int, int]     # (x, y, w, h) of the raw contour
    contour: np.ndarray                 # the approxPolyDP polygon, not the raw contour
    vertex_count: int                   # == len(contour)
    confidence: float                   # [0, 1]
    frame_id: int
    timestamp_ms: int
    area: float = 0.0                   # raw contour area, px^2; 0.0 = not computed
    solidity: float = 0.0               # area / convex hull area; 0.0 = not computed


@dataclass
class StopLineCandidate:
    """Rows grow downward. y_near_px is the lowest point of the bottom edge, the part reached first."""
    label: str                          # always "stop_line"
    bbox: tuple[int, int, int, int]     # (x, y, w, h) around the paired edges
    x_left: float                       # ends of the overlap of the paired edges, px
    x_right: float
    y_top_px: float                     # top edge's row at the midpoint of the overlap
    y_bottom_px: float                  # bottom edge's row at the midpoint; the ROI height when clipped
    y_near_px: float                    # lowest bottom-edge row across the overlap
    tilt_deg: float                     # signed angle from horizontal; + = right end nearer the robot
    length_px: float                    # along the line
    thickness_px: float                 # perpendicular to the line
    mean_intensity: float               # 0-255 between the edges
    clipped: bool                       # the bottom edge is below the ROI: the robot is on the line
    confidence: float                   # [0, 1]
    frame_id: int
    timestamp_ms: int
    proximity: float = 0.0              # [0, 1] y_near_px / ROI height, as LaneCandidate's; 1.0 = at the ROI bottom


@dataclass
class GeometryBranchResult:
    lane_candidates: list[LaneCandidate]
    sign_candidates: list[SignCandidate]
    frame_id: int
    timestamp_ms: int
    stop_line_candidates: list[StopLineCandidate] = field(default_factory=list)   # lane-ROI-relative, nearest first


def _close_edges(
        edges: np.ndarray,
        kernel_size: tuple[int, int]
    ) -> np.ndarray:
    """Bridge gaps along a line so a fragmented lane line traces as one contour."""
    if not kernel_size or kernel_size[0] < 2:
        return edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def _canny(
        gray: np.ndarray,
        params: CannyParams
    ) -> np.ndarray:
    return cv2.Canny(gray, params.threshold1, params.threshold2,
                        apertureSize=params.aperture_size)


def _contours(
        edges: np.ndarray
    ) -> Sequence[np.ndarray]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def _mean_contour_intensity(
        gray: np.ndarray,
        contour: np.ndarray
    ) -> float:
    """0.0 if the contour lies entirely off the image."""
    x, y, w, h = cv2.boundingRect(contour)

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, gray.shape[1]), min(y + h, gray.shape[0])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi_patch = gray[y1:y2, x1:x2]
    mask = np.zeros(roi_patch.shape, dtype=np.uint8)
    # The mask is patch-sized, so move the contour into patch coordinates
    shifted = contour - np.array([[[x1, y1]]])
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    pixels = roi_patch[mask == 255]
    return float(np.mean(pixels)) if len(pixels) > 0 else 0.0


def _horizontal_edges(
        lane_roi: np.ndarray,
        edges_raw: np.ndarray,
        max_tilt_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    (top, bottom) edge maps of lines within max_tilt_deg of horizontal: brightness rising
    going down, and falling. Separates a stop line from the lane lines it touches before
    any contour joins them.
    """
    gx = cv2.Sobel(lane_roi, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(lane_roi, cv2.CV_16S, 0, 1, ksize=3)
    # Only the few hundred edge pixels, not the ROI's ~35k
    ys, xs = np.nonzero(edges_raw)
    ex = np.abs(gx[ys, xs].astype(np.int32))
    ey = gy[ys, xs].astype(np.int32)
    # |gx| <= |gy| tan(tilt), in integers: the edge's line is within tilt of horizontal
    scale = 1024
    horizontal = ex * scale <= np.abs(ey) * int(round(math.tan(math.radians(max_tilt_deg)) * scale))
    top = np.zeros_like(edges_raw)
    bottom = np.zeros_like(edges_raw)
    rising = horizontal & (ey > 0)
    falling = horizontal & (ey < 0)
    top[ys[rising], xs[rising]] = 255
    bottom[ys[falling], xs[falling]] = 255
    return top, bottom
