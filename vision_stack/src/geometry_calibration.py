"""
calibrate_geometry.py

Purpose:
    Measures pixel-dimension geometry calibration values for the Navilott
    vision pipeline: LaneContourFilter reference stats (ref_area_px,
    aspect percentiles, gap_px_samples) and physical track-feature
    dimensions (street width, lane width, lane divider L x W, stop line
    L x W, intersection size). ref_area_px, aspect percentiles,
    gap_px_samples, lane_width, and lane_divider are measured
    automatically via roi_crop.crop_rois() + geometry.extract_lane_candidates().
    street_width, stop_line, and intersection_size are measured through
    an interactive click-to-measure review pass ('s' / 't' / 'i' keys
    while stepping through frames).

Inputs:
    --video PATH        .avi file to read frames from (output of
                         vision_stack/capture/capture.py)
    --frames-dir PATH   directory of frame images to read (output of
                         vision_stack/capture/decompose.py)
    --out PATH          output path for the calibration JSON
                         (default: vision_stack/calibration/geometry.json)
    --arena-id STR       tag identifying the physical arena setup
    -n / --stride INT    process every Nth sampled frame (default: 5)
    --skip-manual        skip the interactive review pass

Outputs:
    Writes a CombinedCalibration JSON to --out.

Physical reference (arena, 2026-08-29):
    Dashed line spacing:  4.5 in (11.43 cm), center-to-center gap
    Camera mount height:  4 cm above the floor, front-facing
    Lane width:           ~14 cm
    Lane dividing line:   ~1 cm wide
    Full street width:    ~29-30 cm
    Intersection square:  ~28-30 cm per side
    Stored in the output JSON for provenance only; every *_px field is
    measured directly from captured frames.
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

# =============================================================================
# Pipeline stages
# =============================================================================
sys.path.insert(0, "vision_stack/src")

from roi_crop import crop_rois
from geometry import (
    extract_lane_candidates,
    CannyParams,
    LaneContourFilter,
    LaneCandidate,
)

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibrate_geometry")

# =============================================================================
# Physical constants (measured on the arena -- update if the mount changes)
# =============================================================================
DASH_SPACING_CM: float = 11.43   # 4.5 in, center-to-center
CAM_HEIGHT_CM: float = 4.0       # front-facing mount height above floor

# =============================================================================
# Calibration-only contour filter (looser than the production LaneContourFilter)
# =============================================================================
CALIBRATION_LANE_FILTER = LaneContourFilter(
    min_area=0.0,
    max_area=3000.0,
    min_aspect=1.0,
    max_aspect=40.0,
    ref_area=2000.0,     # unused for filtering; overwritten by calibration result
    max_roi_span=1.0,
    min_intensity=60.0,
)
CALIBRATION_CANNY = CannyParams()

# X-column tolerance for pairing two candidates as the same lane line,
# fraction of lane ROI width.
COLUMN_TOLERANCE_FRAC: float = 0.08

# Minimum accepted candidates (summed across all sampled frames) required
# for the dash-stat fields to be marked valid.
MIN_DASHES_FOR_VALID_CALIBRATION: int = 4


# =============================================================================
# Result container
# =============================================================================
@dataclass
class CombinedCalibration:
    """
    Purpose:
        Holds pooled geometry calibration values across every sampled
        frame from one capture run.

    Fields:
        ref_area_px: median bounding-box area of accepted candidates.
        min_aspect_p10 / max_aspect_p90: 10th/90th percentile elongation
            of accepted candidates, pooled across all sampled frames.
        gap_px_samples / gap_px_min / gap_px_max: consecutive same-column
            vertical pixel gaps.
        n_dashes_found: total accepted candidates across all sampled
            frames; see is_valid.
        street_width_px, lane_width_px, lane_divider_length_px /
            lane_divider_width_px, stop_line_length_px /
            stop_line_width_px, intersection_size_px: median of the
            matching *_samples list; None if that feature was never
            measured.
        n_frames_sampled, source, calibrated_at, arena_id: run
            provenance.
    """
    # --- Dash / contour-filter calibration ---
    ref_area_px: float = 0.0
    min_aspect_p10: float = 0.0
    max_aspect_p90: float = 0.0
    gap_px_samples: List[float] = field(default_factory=list)
    gap_px_max: Optional[float] = None
    gap_px_min: Optional[float] = None
    n_dashes_found: int = 0
    lane_roi_shape: Tuple[int, int] = (0, 0)
    dash_spacing_cm: float = DASH_SPACING_CM
    cam_height_cm: float = CAM_HEIGHT_CM

    # --- Track-feature dimension calibration ---
    street_width_px: Optional[float] = None
    street_width_samples: List[float] = field(default_factory=list)

    lane_width_px: Optional[float] = None
    lane_width_samples: List[float] = field(default_factory=list)

    lane_divider_length_px: Optional[float] = None
    lane_divider_width_px: Optional[float] = None
    lane_divider_samples: List[Tuple[float, float]] = field(default_factory=list)

    stop_line_length_px: Optional[float] = None
    stop_line_width_px: Optional[float] = None
    stop_line_samples: List[Tuple[float, float]] = field(default_factory=list)

    intersection_size_px: Optional[float] = None
    intersection_size_samples: List[float] = field(default_factory=list)

    # --- Shared provenance ---
    n_frames_sampled: int = 0
    source: str = ""
    calibrated_at: str = ""
    arena_id: str = "default"

    def __post_init__(self):
        self._validated = False   # True only after load_combined_calibration() is used

    @property
    def is_calibrated(self) -> bool:
        """True if this instance came from load_combined_calibration()."""
        return self._validated

    @property
    def is_valid(self) -> bool:
        """True if n_dashes_found meets MIN_DASHES_FOR_VALID_CALIBRATION."""
        return self.n_dashes_found >= MIN_DASHES_FOR_VALID_CALIBRATION


def save_combined_calibration(calib: CombinedCalibration, json_path: str) -> None:
    """
    Purpose:
        Write a CombinedCalibration to disk as JSON.

    Inputs:
        calib: CombinedCalibration to persist
        json_path: destination path
    """
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    data = asdict(calib)
    data.pop("_validated", None)   # dataclass internal flag, not JSON schema
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def load_combined_calibration(json_path: str) -> CombinedCalibration:
    """
    Purpose:
        Load a previously-saved combined calibration from disk.

    Raises:
        FileNotFoundError: json_path does not exist
        KeyError: file exists but is missing a required field
        json.JSONDecodeError: file exists but is not valid JSON
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    calib = CombinedCalibration(
        ref_area_px=data["ref_area_px"],
        min_aspect_p10=data["min_aspect_p10"],
        max_aspect_p90=data["max_aspect_p90"],
        gap_px_samples=data.get("gap_px_samples", []),
        gap_px_max=data.get("gap_px_max"),
        gap_px_min=data.get("gap_px_min"),
        n_dashes_found=data.get("n_dashes_found", 0),
        lane_roi_shape=tuple(data.get("lane_roi_shape", (0, 0))),
        dash_spacing_cm=data.get("dash_spacing_cm", DASH_SPACING_CM),
        cam_height_cm=data.get("cam_height_cm", CAM_HEIGHT_CM),
        street_width_px=data.get("street_width_px"),
        street_width_samples=data.get("street_width_samples", []),
        lane_width_px=data.get("lane_width_px"),
        lane_width_samples=data.get("lane_width_samples", []),
        lane_divider_length_px=data.get("lane_divider_length_px"),
        lane_divider_width_px=data.get("lane_divider_width_px"),
        lane_divider_samples=[tuple(s) for s in data.get("lane_divider_samples", [])],
        stop_line_length_px=data.get("stop_line_length_px"),
        stop_line_width_px=data.get("stop_line_width_px"),
        stop_line_samples=[tuple(s) for s in data.get("stop_line_samples", [])],
        intersection_size_px=data.get("intersection_size_px"),
        intersection_size_samples=data.get("intersection_size_samples", []),
        n_frames_sampled=data.get("n_frames_sampled", 0),
        source=data.get("source", ""),
        calibrated_at=data.get("calibrated_at", ""),
        arena_id=data.get("arena_id", "default"),
    )
    calib._validated = True
    return calib


def _print_summary(calib: CombinedCalibration) -> None:
    def fmt(v):
        return f"{v:.1f}" if v is not None else "n/a (no samples)"

    print(f"[calibration] n_dashes_found = {calib.n_dashes_found} "
          f"({'VALID' if calib.is_valid else 'BELOW THRESHOLD'})")
    print(f"[calibration] ref_area_px    = {calib.ref_area_px:.1f}")
    print(f"[calibration] aspect p10/p90 = {calib.min_aspect_p10:.2f} / {calib.max_aspect_p90:.2f}")
    if calib.gap_px_samples:
        print(f"[calibration] gap_px samples = {len(calib.gap_px_samples)}  "
              f"min={calib.gap_px_min:.1f}  max={calib.gap_px_max:.1f}")
    else:
        print("[calibration] gap_px samples = 0 (no same-column pairs found)")
    print("[calibration] --- Track Geometry (px) ---")
    print(f"[calibration] street_width       = {fmt(calib.street_width_px)}  "
          f"(n={len(calib.street_width_samples)})")
    print(f"[calibration] lane_width         = {fmt(calib.lane_width_px)}  "
          f"(n={len(calib.lane_width_samples)})")
    print(f"[calibration] lane_divider L x W = {fmt(calib.lane_divider_length_px)} x "
          f"{fmt(calib.lane_divider_width_px)}  (n={len(calib.lane_divider_samples)})")
    print(f"[calibration] stop_line L x W    = {fmt(calib.stop_line_length_px)} x "
          f"{fmt(calib.stop_line_width_px)}  (n={len(calib.stop_line_samples)})")
    print(f"[calibration] intersection_size  = {fmt(calib.intersection_size_px)}  "
          f"(n={len(calib.intersection_size_samples)})")


# =============================================================================
# Shared measurement helpers
# =============================================================================
def _percentile(values: List[float], p: float) -> float:
    """Purpose: linear-interpolated percentile without a numpy/scipy dep beyond np.percentile."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _column_key(cand: LaneCandidate) -> float:
    """Purpose: x-center of a candidate's bbox, used to pair same-column dashes."""
    x, _, w, _ = cand.bbox
    return x + w / 2.0


def _measure_gaps(candidates: List[LaneCandidate], roi_width: int) -> List[float]:
    """
    Purpose:
        Compute vertical pixel gaps between consecutive candidates that
        share a lane-line column.

    Inputs:
        candidates: accepted LaneCandidate list
        roi_width: lane ROI width in px, for COLUMN_TOLERANCE_FRAC scaling

    Outputs:
        list[float]: one gap sample per adjacent same-column pair
    """
    if len(candidates) < 2:
        return []

    tol_px = COLUMN_TOLERANCE_FRAC * roi_width
    by_x = sorted(candidates, key=_column_key)

    columns: List[List[LaneCandidate]] = []
    for cand in by_x:
        placed = False
        cx = _column_key(cand)
        for col in columns:
            col_avg = sum(_column_key(c) for c in col) / len(col)
            if abs(cx - col_avg) <= tol_px:
                col.append(cand)
                placed = True
                break
        if not placed:
            columns.append([cand])

    gaps: List[float] = []
    for col in columns:
        if len(col) < 2:
            continue
        col_by_y = sorted(col, key=lambda c: c.bbox[1])
        for prev, nxt in zip(col_by_y, col_by_y[1:]):
            _, py, _, ph = prev.bbox
            _, ny, _, _ = nxt.bbox
            gap = ny - (py + ph)
            if gap > 0:   # skip overlapping/touching boxes -- not a real gap
                gaps.append(float(gap))

    return gaps


# =============================================================================
# Per-frame automatic measurement
# =============================================================================
@dataclass
class _FrameMeasurement:
    n_found: int
    lane_roi_shape: Tuple[int, int]
    areas: List[float] = field(default_factory=list)
    aspects: List[float] = field(default_factory=list)
    gap_samples: List[float] = field(default_factory=list)
    divider_samples: List[Tuple[float, float]] = field(default_factory=list)
    lane_width_sample: Optional[float] = None


def _measure_frame(frame_bgr: np.ndarray, frame_id: int) -> _FrameMeasurement:
    """
    Purpose:
        Run ROI crop + contour-candidate extraction on this frame and
        derive dash stats, divider L x W, and lane width from the result.

    Inputs:
        frame_bgr: raw captured frame (pre-ROI-crop)
        frame_id: passed through to extract_lane_candidates() for logging

    Outputs:
        _FrameMeasurement with this frame's samples
    """
    roi_result = crop_rois(frame_bgr, frame_id=frame_id)
    lane_roi = roi_result.lane_roi
    lane_h, lane_w = lane_roi.shape[:2]

    candidates, _debug = extract_lane_candidates(
        lane_roi=lane_roi,
        canny_params=CALIBRATION_CANNY,
        lane_filter=CALIBRATION_LANE_FILTER,
        frame_id=frame_id,
        timestamp_ms=int(time.time() * 1000),
    )

    if not candidates:
        return _FrameMeasurement(n_found=0, lane_roi_shape=(lane_h, lane_w))

    areas = [float(cv2.contourArea(c.contour)) for c in candidates]
    aspects = []
    divider_samples = []
    for c in candidates:
        _, (rw, rh), _ = cv2.minAreaRect(c.contour)
        long_side = max(rw, rh)
        short_side = max(min(rw, rh), 1.0)
        aspects.append(long_side / short_side)
        divider_samples.append((float(long_side), float(short_side)))

    gap_samples = _measure_gaps(candidates, lane_w)

    lane_width_sample = None
    if len(candidates) >= 2:
        centers = sorted(_column_key(c) for c in candidates)
        lane_width_sample = float(centers[-1] - centers[0])

    return _FrameMeasurement(
        n_found=len(candidates),
        lane_roi_shape=(lane_h, lane_w),
        areas=areas,
        aspects=aspects,
        gap_samples=gap_samples,
        divider_samples=divider_samples,
        lane_width_sample=lane_width_sample,
    )


# =============================================================================
# Manual click-to-measure (street width, stop line, intersection)
# =============================================================================
class _ClickCollector:
    def __init__(self):
        self.points: List[Tuple[int, int]] = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def _collect_clicks(
        frame_bgr: np.ndarray,
        n_points: int,
        window_title: str,
    ) -> Optional[List[Tuple[int, int]]]:
    """Purpose: show a frame, collect exactly n_points mouse clicks on it."""
    collector = _ClickCollector()
    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, collector.callback)
    try:
        while True:
            vis = frame_bgr.copy()
            for i, p in enumerate(collector.points):
                cv2.circle(vis, p, 4, (0, 255, 0), -1)
                cv2.putText(vis, str(i + 1), (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(
                vis, f"click {len(collector.points)}/{n_points} points   "
                     f"'r' reset   'q'/Esc skip",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )
            cv2.imshow(window_title, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('r'):
                collector.points = []
            elif key in (ord('q'), 27):
                return None
            elif len(collector.points) >= n_points:
                return collector.points[:n_points]
    finally:
        cv2.destroyWindow(window_title)


def _measure_line_px(frame_bgr: np.ndarray, label: str) -> Optional[float]:
    """Purpose: click 2 endpoints, return Euclidean pixel distance."""
    pts = _collect_clicks(frame_bgr, 2, f"Click both ends of: {label}")
    if pts is None:
        return None
    (x1, y1), (x2, y2) = pts
    return float(np.hypot(x2 - x1, y2 - y1))


def _measure_box_lw_px(frame_bgr: np.ndarray, label: str) -> Optional[Tuple[float, float]]:
    """Purpose: click 4 corners (any order), return (length_px, width_px) via minAreaRect."""
    pts = _collect_clicks(frame_bgr, 4, f"Click 4 corners of: {label}")
    if pts is None:
        return None
    rect_pts = np.array(pts, dtype=np.float32)
    (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(rect_pts)
    length = max(rw, rh)
    width = max(min(rw, rh), 1.0)
    return (float(length), float(width))


def _interactive_review(
        frames: List[Tuple[int, np.ndarray]],
        calib: CombinedCalibration,
    ) -> None:
    """
    Purpose:
        Step through sampled frames, collecting manual measurements for
        street width, stop line L x W, and intersection size.

    Inputs:
        frames: list of (frame_id, frame_bgr) tuples to step through
        calib: CombinedCalibration to append samples into

    Controls:
        s        = measure street width (2-point click)
        t        = measure stop line L x W (4-corner click)
        i        = measure intersection size (4-corner click)
        n/space  = next frame (no measurement)
        q/Esc    = finish review early
    """
    window = "calibrate_geometry -- review"
    idx = 0
    try:
        while idx < len(frames):
            frame_id, frame_bgr = frames[idx]
            vis = frame_bgr.copy()
            cv2.putText(vis, f"frame {frame_id}  ({idx + 1}/{len(frames)})",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis, "[s]treet width  [t]stop line  [i]ntersection  [n]ext  [q]uit",
                        (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow(window, vis)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):
                val = _measure_line_px(frame_bgr, "street width (curb to curb)")
                if val is not None:
                    calib.street_width_samples.append(val)
                    log.info("frame %d: street_width_px = %.1f", frame_id, val)
            elif key == ord('t'):
                val = _measure_box_lw_px(frame_bgr, "stop line")
                if val is not None:
                    calib.stop_line_samples.append(val)
                    log.info("frame %d: stop_line (L,W)_px = (%.1f, %.1f)", frame_id, *val)
            elif key == ord('i'):
                val = _measure_box_lw_px(frame_bgr, "intersection square")
                if val is not None:
                    length, width = val
                    calib.intersection_size_samples.extend([length, width])
                    log.info("frame %d: intersection side samples = %.1f, %.1f",
                              frame_id, length, width)
            elif key in (ord('n'), ord(' ')):
                idx += 1
            elif key in (ord('q'), 27):
                break
    finally:
        cv2.destroyWindow(window)


# =============================================================================
# Frame sources
# =============================================================================
_FRAME_NUM_RE = re.compile(r"(\d+)")


def _iter_video_frames(video_path: str, stride: int):
    """
    Purpose:
        Yield frames from an .avi file.

    Inputs:
        video_path: path to the video file
        stride: yield every stride-th frame

    Outputs:
        generator of (frame_id, frame_bgr) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Failed to open video file: %s", video_path)
        sys.exit(1)

    frame_id = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_id % stride == 0:
            yield frame_id, frame_bgr
        frame_id += 1
    cap.release()


def _iter_frames_dir(frames_dir: str, stride: int):
    """
    Purpose:
        Yield frames from a directory of frame_NNNNN.<fmt> images.

    Inputs:
        frames_dir: directory containing frame image files
        stride: yield every stride-th file in sorted order

    Outputs:
        generator of (frame_id, frame_bgr) tuples. frame_id is parsed
        from the numeric run in each filename, falling back to sort
        order if a filename has no digits.
    """
    paths = sorted(
        p for ext in ("jpg", "jpeg", "png")
        for p in glob.glob(os.path.join(frames_dir, f"*.{ext}"))
    )
    if not paths:
        log.error("No frame images found in %s", frames_dir)
        sys.exit(1)

    for pos, path in enumerate(paths):
        if pos % stride != 0:
            continue
        m = _FRAME_NUM_RE.search(os.path.basename(path))
        frame_id = int(m.group(1)) if m else pos
        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            log.warning("Failed to read %s -- skipping", path)
            continue
        yield frame_id, frame_bgr


# =============================================================================
# Main processing pass
# =============================================================================
def run_combined_calibration(
        frames,
        source: str,
        arena_id: str = "default",
        interactive: bool = True,
    ) -> CombinedCalibration:
    """
    Purpose:
        Run automatic measurement over every sampled frame, then an
        optional interactive review pass, and pool the results.

    Inputs:
        frames: list of (frame_id, frame_bgr) tuples, already sampled
        source: video path or frames-dir path, stored for provenance
        arena_id: free-text tag identifying the physical arena setup
        interactive: if False, skip the click-to-measure review pass

    Outputs:
        CombinedCalibration
    """
    calib = CombinedCalibration(
        source=source,
        arena_id=arena_id,
        calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_frames_sampled=len(frames),
    )

    all_areas: List[float] = []
    all_aspects: List[float] = []
    all_gaps: List[float] = []
    total_dashes = 0
    lane_roi_shape = (0, 0)

    for frame_id, frame_bgr in frames:
        m = _measure_frame(frame_bgr, frame_id)
        total_dashes += m.n_found
        lane_roi_shape = m.lane_roi_shape
        if m.n_found == 0:
            continue
        all_areas.extend(m.areas)
        all_aspects.extend(m.aspects)
        all_gaps.extend(m.gap_samples)
        calib.lane_divider_samples.extend(m.divider_samples)
        if m.lane_width_sample is not None:
            calib.lane_width_samples.append(m.lane_width_sample)

    calib.ref_area_px = float(np.median(all_areas)) if all_areas else CALIBRATION_LANE_FILTER.ref_area
    calib.min_aspect_p10 = _percentile(all_aspects, 10) if all_aspects else 0.0
    calib.max_aspect_p90 = _percentile(all_aspects, 90) if all_aspects else 0.0
    calib.gap_px_samples = all_gaps
    calib.gap_px_max = max(all_gaps) if all_gaps else None
    calib.gap_px_min = min(all_gaps) if all_gaps else None
    calib.n_dashes_found = total_dashes
    calib.lane_roi_shape = lane_roi_shape

    log.info(
        "Automatic pass: %d frames sampled, %d candidates found, "
        "%d divider samples, %d lane-width samples, %d gap samples",
        len(frames), total_dashes, len(calib.lane_divider_samples),
        len(calib.lane_width_samples), len(all_gaps),
    )

    if interactive:
        _interactive_review(frames, calib)

    if calib.street_width_samples:
        calib.street_width_px = float(np.median(calib.street_width_samples))
    if calib.lane_width_samples:
        calib.lane_width_px = float(np.median(calib.lane_width_samples))
    if calib.lane_divider_samples:
        lengths = [l for l, _ in calib.lane_divider_samples]
        widths = [w for _, w in calib.lane_divider_samples]
        calib.lane_divider_length_px = float(np.median(lengths))
        calib.lane_divider_width_px = float(np.median(widths))
    if calib.stop_line_samples:
        lengths = [l for l, _ in calib.stop_line_samples]
        widths = [w for _, w in calib.stop_line_samples]
        calib.stop_line_length_px = float(np.median(lengths))
        calib.stop_line_width_px = float(np.median(widths))
    if calib.intersection_size_samples:
        calib.intersection_size_px = float(np.median(calib.intersection_size_samples))

    return calib


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to an .avi captured with capture.py")
    src.add_argument("--frames-dir", help="path to a frame directory produced by decompose.py")

    parser.add_argument("--out", default="vision_stack/calibration/geometry.json",
                         help="output path for the calibration JSON")
    parser.add_argument("--arena-id", default="default",
                         help="tag identifying the physical arena setup")
    parser.add_argument("-n", "--stride", type=int, default=5,
                         help="process every Nth sampled frame (default: 5)")
    parser.add_argument("--skip-manual", action="store_true",
                         help="skip interactive review; automatic measurements only "
                              "(no street_width / stop_line / intersection_size)")
    args = parser.parse_args()

    if args.video:
        frames = list(_iter_video_frames(args.video, args.stride))
        source = args.video
    else:
        frames = list(_iter_frames_dir(args.frames_dir, args.stride))
        source = args.frames_dir

    if not frames:
        log.error("No frames to process")
        sys.exit(1)

    calib = run_combined_calibration(
        frames, source=source, arena_id=args.arena_id, interactive=not args.skip_manual,
    )

    if not calib.is_valid:
        print(
            f"[calibration] WARNING: only {calib.n_dashes_found} candidates found "
            f"(need >= {MIN_DASHES_FOR_VALID_CALIBRATION}). ref_area_px and the aspect "
            f"percentiles are NOT trustworthy. Check camera framing / lighting and re-run "
            f"before saving this over an existing calibration."
        )

    save_combined_calibration(calib, args.out)
    print(f"[calibration] saved -> {args.out}")
    _print_summary(calib)


if __name__ == "__main__":
    main()