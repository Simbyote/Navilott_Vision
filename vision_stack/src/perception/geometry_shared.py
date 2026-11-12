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


# A marking that leaves the ROI through its side ends above the bottom with its lowest
# rows at that side, so the band's mean read the ROI's edge: up to ~110 px (4 cm) from
# where the line meets the bottom, and the offset read ~0.6 of the truth (P3, 2026-11-03).
# Its axis over the lowest FOOT_FIT_ROWS rows, carried on to the band's row, says where
# it would meet it (2026-11-12)
FOOT_FIT_ROWS = 30
# Flatter than this the axis meets the bottom far away or never: the band's mean stands.
# The lane detector strips lines this flat from its edges anyway (horizontal_edge_deg)
FOOT_MIN_SLOPE_DEG = 20.0


def contour_foot_x(
        contour: np.ndarray,
        band_px: int = FOOT_BAND_PX,
        roi_h: int | None = None,
    ) -> float | None:
    """
    Where the marking meets the ROI's bottom, in ROI px: the mean x of the contour's
    lowest band_px rows when it reaches the bottom, within half a band (where a diagonal
    marking meets the near edge, which its bbox center misses by up to half its width);
    with roi_h, a contour ending higher has its lower axis carried on to the band's row
    instead, inside or outside the ROI. None for an empty contour.
    """
    if contour is None or len(contour) == 0:
        return None
    pts = np.asarray(contour).reshape(-1, 2)
    y_max = pts[:, 1].max()
    band_px = max(band_px, 0)
    x_band = float(pts[pts[:, 1] >= y_max - band_px][:, 0].mean())
    if roi_h is None or y_max >= roi_h - 1 - band_px // 2:
        return round(x_band, 2)
    target = roi_h - 1 - band_px / 2.0              # the band's centre row, as a line that reaches it reads
    low = pts[pts[:, 1] >= y_max - FOOT_FIT_ROWS].astype(np.float32)
    if len(low) < 2:
        return round(x_band, 2)
    vx, vy, x0, y0 = cv2.fitLine(low, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    if abs(vy) < math.sin(math.radians(FOOT_MIN_SLOPE_DEG)):
        return round(x_band, 2)
    return round(float(x0 + (target - y0) * vx / vy), 2)


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
    # See contour_foot_x; None = not computed. Negative, or past the ROI's width, for a
    # marking that left through the ROI's side (2026-11-12; -1.0 was the sentinel until then)
    foot_x: float | None = None


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
    return _horizontal_edges_at(lane_roi, edges_raw, (max_tilt_deg,))[0]


def _horizontal_edges_at(
        lane_roi: np.ndarray,
        edges_raw: np.ndarray,
        tilts_deg: tuple[float, ...],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    _horizontal_edges() for each tilt, from one Sobel pass and one gather of the edge
    pixels: the stop line (4 deg in MEASURED) and the lane detector (20) each need a
    split, and until 2026-11-12 each paid for its own (~a fifth of geometry's time).
    """
    gx = cv2.Sobel(lane_roi, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(lane_roi, cv2.CV_16S, 0, 1, ksize=3)
    # Only the few hundred edge pixels, not the ROI's ~35k
    ys, xs = np.nonzero(edges_raw)
    ex = np.abs(gx[ys, xs].astype(np.int32))
    ey = gy[ys, xs].astype(np.int32)
    aey, up, down = np.abs(ey), ey > 0, ey < 0
    # |gx| <= |gy| tan(tilt), in integers: the edge's line is within tilt of horizontal
    scale = 1024
    out = []
    for tilt in tilts_deg:
        horizontal = ex * scale <= aey * int(round(math.tan(math.radians(tilt)) * scale))
        top = np.zeros_like(edges_raw)
        bottom = np.zeros_like(edges_raw)
        rising = horizontal & up
        falling = horizontal & down
        top[ys[rising], xs[rising]] = 255
        bottom[ys[falling], xs[falling]] = 255
        out.append((top, bottom))
    return out
