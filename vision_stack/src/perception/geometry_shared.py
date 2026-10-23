"""Geometry's shared parts: the configs, the candidate types, and the edge helpers more than one detector uses.

Purpose:
    The three detectors (geometry_lanes, geometry_stop_line, geometry_sign)
    each live in their own module; what more than one of them needs lives
    here, so no detector imports another. geometry.py re-exports all of it.

Main package:
    GeometryBranchResult: one frame's LaneCandidates, SignCandidates and
    StopLineCandidates, ROI-relative, with the frame identity; and the
    GeometryConfig that tunes the detectors producing them.

Flow:
    None of its own: the detectors call the edge helpers (Canny, contours,
    contour brightness, the gradient split) as each needs them.
"""
import math

import cv2
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.params import FOOT_BAND_PX


@dataclass
class CannyParams:
    """Edge detection settings. These set how many contours reach the filters."""
    threshold1: float = 80.0        # hysteresis low; raising it fractures lines into more pieces
    threshold2: float = 200.0       # hysteresis high; raising it seeds fewer contours
    # Sobel aperture: 3, 5 or 7. Larger smooths more but scales gradient
    # magnitudes up, so the thresholds need retuning with it.
    aperture_size: int = 3
    # (width, height) px of the closing rectangle, lane branch only. Wider than
    # tall bridges gaps along horizontal lines. Width < 2 disables closing.
    close_kernel: tuple[int, int] = (9, 3)


@dataclass
class LaneContourFilter:
    """Acceptance gates and confidence references for lane-boundary contours."""
    min_area: float = 1.0           # px^2
    max_area: float = 1000.0        # px^2
    min_aspect: float = 0.0         # elongation: minAreaRect long / short side
    max_aspect: float = 60.0
    max_roi_span: float = 1.0       # max fraction of the ROI a contour may span along its own axis
    min_intensity: float = 120.0    # 0-255 mean inside the contour; rejects dark blobs such as seams
    ref_length: float = 0.25        # long side, as a fraction of ROI extent, that scores full length confidence
    ref_width: float = 30.0         # short side, px, that scores full width confidence
    # Lines across the lane (stop lines) are taken out of the lane detector's
    # edges before contours are traced, so one touching the lane lines can't
    # close into a single contour with them. Edges within this angle of
    # horizontal are the ones considered; None turns the filter off. The
    # course has no curves the camera steers through, so no lane line lies
    # this flat
    horizontal_edge_deg: float | None = 20.0
    # Only horizontal edge runs at least this long are removed, and with them
    # every horizontal edge within horizontal_band_px of their line. Longer
    # than any lane tape is wide (the lane offset gate allows 45 px), so the
    # ends of a piece of tape stay and it still traces as one contour
    horizontal_min_run_px: float = 46.0
    horizontal_band_px: float = 3.0


@dataclass
class SignContourFilter:
    """Acceptance gates and confidence references for sign-shape contours."""
    min_area: float = 100.0         # px^2
    max_area: float = 30000.0       # px^2
    min_vertices: int = 8           # approxPolyDP vertex count
    max_vertices: int = 9
    min_solidity: float = 0.80      # contour area / convex hull area
    # approxPolyDP epsilon as a fraction of the hull's perimeter, so it scales
    # with sign distance. Smaller keeps more vertices; larger collapses the
    # outline toward fewer. 0.02 per the redness design: on the hull of a
    # filled mask it gave 8-9 vertices on every synthetic sign from radius 12 to 30 px
    epsilon_factor: float = 0.02
    ref_area: float = 5000.0        # px^2 that scores full area confidence
    # Redness (R - max(G, B), 0-255) never thresholded below this. Otsu always
    # splits the image in two, so a sign-free ROI thresholds its own noise
    # (Otsu at 1 on synthetic gray and black floors, 2026-09, with a false
    # 8-vertex blob in 1 of 20). The dimmest synthetic sign, BGR (30, 30, 80)
    # on near-black, had Otsu at 22. Re-check on course frames
    min_redness: float = 20.0
    # Side, px, of the square closing on the red mask: fills noise pinholes
    # and the notches the white STOP letters leave at the rim. 5 per the
    # design; larger starts merging the sign with red nearby. < 2 disables
    close_kernel: int = 5
    # Ghost signs: a leg, a chair or skin in the background passed as a
    # sign, since any solid roundish red blob comes out at 8-9 hull
    # vertices. These shape and content gates catch most of them; each is
    # off (None) by default and set in MEASURED. Bounding box width /
    # height: an octagon is ~1, a leg or a chair back is not; wide enough
    # for a sign seen at an angle
    aspect_range: tuple[float, float] | None = None
    # Contour area / bounding box area: a regular octagon is 0.83, a
    # rectangle 1.0, a disc 0.79
    extent_range: tuple[float, float] | None = None
    # A real sign has white letters inside: at least this share of the hull's
    # pixels must be white (each BGR channel >= letter_min_v and redness <=
    # letter_max_redness; the letters' blurred pink edges don't count)
    min_letter_share: float | None = None
    letter_min_v: int = 150
    letter_max_redness: int = 60
    # Only a sign this large (contour px^2) is held to the letters gate: a
    # far sign's letters blur into its red (synthetic signs: ~0.1% white up
    # to radius 24 px, 1700 px^2; 11-16% from radius 28, 2150 px^2)
    letters_min_area: float = 2000.0


@dataclass(frozen=True)
class StopLineFilter:
    """
    Stop-line detection tuning, separate from the lane gates.

    Lengths and thicknesses are lane-ROI px, measured along the fitted line
    and perpendicular to it.
    """
    # Largest angle from horizontal. Also the gradient split: edges of steeper
    # lines never reach the stop-line detector. Covers the robot approaching
    # the line off-square; the course has no curves the camera steers through
    max_tilt_deg: float = 20.0
    # Longer than any lane-tape width (sweep max ~41 px), so the top or bottom
    # edge of a lane-line dash can't pass for a stop line
    min_length_px: float = 60.0
    min_thickness_px: float = 3.0
    max_thickness_px: float = 40.0
    min_intensity: float = 70.0        # 0-255 mean between the paired edges; tape, not a shadow edge
    # Paired edges must overlap by this fraction of the shorter one
    min_edge_overlap: float = 0.5
    # (width, height) px closing rectangle on each edge map; bridges gaps
    # along the line, e.g. where a lane line crosses it
    close_kernel: tuple[int, int] = (15, 1)
    ref_length_px: float = 200.0        # length that scores full length confidence, about one lane


@dataclass(frozen=True)
class GeometryConfig:
    """
    Geometry tuning as one unit, so the stage takes a single config like every
    other stage. The parts can still be passed to run_geometry_branch() directly.
    """
    canny: CannyParams = field(default_factory=CannyParams)     # lanes and stop lines; signs are found by color
    lane: LaneContourFilter = field(default_factory=LaneContourFilter)
    sign: SignContourFilter = field(default_factory=SignContourFilter)
    stop_line: StopLineFilter = field(default_factory=StopLineFilter)   # reads the lane ROI's Canny edges


def contour_foot_x(
        contour: np.ndarray,
        band_px: int = FOOT_BAND_PX,
    ) -> float:
    """
    ROI-local x of a contour at its nearest approach to the robot.

    Purpose:
        The anchor lane offset should steer by. For a marking running
        diagonally across the ROI, the bbox center is displaced from where
        the marking crosses the near edge by up to half the bbox width.

    Inputs:
        contour: OpenCV (N, 1, 2) or flat (N, 2) points.
        band_px: Rows above the lowest point that count as the base. Wider
            averages out a ragged edge; narrower tracks a steep diagonal
            more tightly. 0 or less uses only the lowest row.

    Outputs:
        Mean x of the base points in px, rounded to 0.01. -1.0 for an empty contour.
    """
    if contour is None or len(contour) == 0:
        return -1.0
    pts = np.asarray(contour).reshape(-1, 2)
    y_max = pts[:, 1].max()
    band = pts[pts[:, 1] >= y_max - max(band_px, 0)]
    return round(float(band[:, 0].mean()), 2)


@dataclass
class LaneCandidate:
    """Lane-boundary candidate, ROI-relative. frame_id and timestamp_ms are carried from capture."""
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
    """Stop-sign candidate, ROI-relative. frame_id and timestamp_ms are carried from capture."""
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
    """
    Stop-line candidate, lane-ROI-relative. frame_id and timestamp_ms are carried from capture.

    Rows grow downward, toward the robot. y_near_px is the lowest point of the
    bottom edge: the part of the line the robot reaches first.
    """
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
    """One frame's geometry detections, ROI-relative. Identity copied from ROICropResult."""
    lane_candidates: list[LaneCandidate]
    sign_candidates: list[SignCandidate]
    frame_id: int
    timestamp_ms: int
    stop_line_candidates: list[StopLineCandidate] = field(default_factory=list)   # lane-ROI-relative, nearest first


def _close_edges(
        edges: np.ndarray,
        kernel_size: tuple[int, int]
    ) -> np.ndarray:
    """Bridge along-line gaps in an edge map so a fragmented lane line traces as one contour."""
    if not kernel_size or kernel_size[0] < 2:
        return edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def _canny(
        gray: np.ndarray,
        params: CannyParams
    ) -> np.ndarray:
    """0/255 Canny edge map of a gray image."""
    return cv2.Canny(gray, params.threshold1, params.threshold2,
                        apertureSize=params.aperture_size)


def _contours(
        edges: np.ndarray
    ) -> Sequence[np.ndarray]:
    """External contours of an edge map."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def _mean_contour_intensity(
        gray: np.ndarray,
        contour: np.ndarray
    ) -> float:
    """Mean gray level inside the filled contour, or 0.0 if it lies entirely off the image."""
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
    Split the lane ROI's Canny edges by gradient direction.

    Purpose:
        A stop line's top and bottom edges have a gradient pointing up or
        down; a lane line's sides have one pointing left or right. Keeping
        only edge pixels whose line runs within max_tilt_deg of horizontal
        separates a stop line from lane lines it touches, before any contour
        joins them.

    Outputs:
        (top, bottom): 0/255 uint8 maps. top holds edges where brightness
        rises going down (the upper edge of bright tape), bottom where it falls.
    """
    gx = cv2.Sobel(lane_roi, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(lane_roi, cv2.CV_16S, 0, 1, ksize=3)
    # Only edge pixels are classified: a few hundred of the ROI's ~35k, so
    # whole-array arithmetic would cost ten times the work it needs
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
