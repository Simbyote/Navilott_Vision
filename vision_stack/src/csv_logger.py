"""
csv_logger.py

Purpose:
    Structured CSV logging for bench runs. Replaces the per-frame terminal
    log lines (nav packet, TAGS summary, per-contour CONTOUR lines) with
    two CSV files suited to plotting/analysis:

        <prefix>_frames.csv     one row per processed frame
        <prefix>_contours.csv   one row per contour reaching the geometry
                                 stage (variable count per frame, only
                                 populated when DEBUG_CONTOURS is on)

    Both files carry frame_id / timestamp_ms so contour-level rows can be
    joined back to per-frame navigation state (pandas merge/groupby, Excel
    VLOOKUP, whatever). fix_* columns are duplicated onto every frame row
    (rather than relying on the filename encoding) so a single CSV can be
    filtered/grouped by which fixes were active without re-parsing the
    R/T/O/D/A tag out of a filename.

Usage:
    with RunLogger(prefix, fix_cfg) as run_logger:
        ...
        run_logger.log_frame(...)
        run_logger.log_contours(frame_id, timestamp_ms, tags.contour_debug)

    If prefix is None, RunLogger degrades to a no-op (log_frame /
    log_contours / close all become harmless), so callers don't need an
    `if run_logger:` guard at every call site
"""

from __future__ import annotations

import csv
from typing import List, Optional, TextIO

from config import Config, FrameTags, ContourDebug

# =============================================================================
# Column Schemas
# =============================================================================
FRAME_FIELDS = [
    "frame_id", "timestamp_ms", "frame_time_ms", "budget_exceeded",
    "fix_roi_inset", "fix_trapezoid_mask", "fix_orientation_filt",
    "fix_dashed_dilate", "fix_anchor_halves",
    "offset", "raw_offset", "heading_error", "drive_state", "stop_sign_detected",
    "lane_mode", "lane_confidence",
    "edge_ratio", "intersection_trigger", "intersection_active",
    "pole_misclassified", "wall_edge_detected", "dashed_reject_center",
    "anchor_wrong_half", "dashed_line_dropped",
    "imu_sample_count", "imu_yaw_rate_dps",
]

CONTOUR_FIELDS = [
    "frame_id", "timestamp_ms",
    "area", "aspect", "intensity", "roi_span",
    "center_in_middle_third", "accepted", "reject_reason",
]

# =============================================================================
# Run Logger
# =============================================================================
class RunLogger:
    """
    Purpose:
        Owns the two CSV files for one pipeline run and exposes one write
        call per frame. Context-manager friendly; degrades to a no-op when
        constructed with prefix=None so call sites don't need to branch

    Inputs:
        prefix: path prefix with NO extension. Files are written to
            f"{prefix}_frames.csv" and f"{prefix}_contours.csv".
            None disables CSV logging entirely (no-op mode)
        fix_cfg: the run's Config, used to populate the per-frame fix_*
            columns
    """
    def __init__(self, prefix: Optional[str], fix_cfg: Config):
        self._enabled = prefix is not None
        self._fix_cfg = fix_cfg
        self._frames_fh: Optional[TextIO] = None
        self._contours_fh: Optional[TextIO] = None

        if not self._enabled:
            return

        self._frames_fh = open(f"{prefix}_frames.csv", "w", newline="")
        self._contours_fh = open(f"{prefix}_contours.csv", "w", newline="")

        self._frames_w = csv.DictWriter(self._frames_fh, fieldnames=FRAME_FIELDS)
        self._contours_w = csv.DictWriter(self._contours_fh, fieldnames=CONTOUR_FIELDS)
        self._frames_w.writeheader()
        self._contours_w.writeheader()

    def log_frame(
            self,
            frame_id: int,
            timestamp_ms: int,
            frame_time_ms: float,
            budget_exceeded: bool,
            offset: float,
            raw_offset: float,
            heading_error: float,
            drive_state: str,
            stop_sign_detected: bool,
            lane_mode: str,
            lane_confidence: float,
            edge_ratio: float,
            intersection_trigger: bool,
            intersection_active: bool,
            tags: FrameTags,
            imu_sample_count: int,
            imu_yaw_rate_dps: float,
        ) -> None:
        """
        Write one row of per-frame navigation + failure-mode state

        NOTE: intersection_trigger is the raw per-frame
            intersection_edge_ratio() > INTERSECTION_EDGE_RATIO_THRESH
            check; intersection_active is Phase 3's debounced state
            machine (Phase3Config.intersection_enter_frames/
            exit_frames) built from a run of those triggers — expect
            intersection_active to lag trigger by a couple frames on
            entry/exit and to stay True through brief trigger gaps
        """
        if not self._enabled:
            return

        dashed_dropped = (
            tags.dashed_reject_center >= tags.DASHED_DROP_MIN_REJECTS
            and lane_mode in ("left_only", "right_only", "none")
        )
        self._frames_w.writerow({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "frame_time_ms": round(frame_time_ms, 3),
            "budget_exceeded": int(budget_exceeded),
            "fix_roi_inset": int(self._fix_cfg.roi_inset),
            "fix_trapezoid_mask": int(self._fix_cfg.trapezoid_mask),
            "fix_orientation_filt": int(self._fix_cfg.orientation_filt),
            "fix_dashed_dilate": int(self._fix_cfg.dashed_dilate),
            "fix_anchor_halves": int(self._fix_cfg.anchor_halves),
            "offset": round(offset, 4),
            "raw_offset": round(raw_offset, 4),
            "heading_error": round(heading_error, 3),
            "drive_state": drive_state,
            "stop_sign_detected": int(stop_sign_detected),
            "lane_mode": lane_mode,
            "lane_confidence": round(lane_confidence, 3),
            "edge_ratio": round(edge_ratio, 4),
            "intersection_trigger": int(intersection_trigger),
            "intersection_active": int(intersection_active),
            "pole_misclassified": tags.pole_misclassified,
            "wall_edge_detected": tags.wall_edge_detected,
            "dashed_reject_center": tags.dashed_reject_center,
            "anchor_wrong_half": int(tags.anchor_wrong_half),
            "dashed_line_dropped": int(dashed_dropped),
            "imu_sample_count": imu_sample_count,
            "imu_yaw_rate_dps": round(imu_yaw_rate_dps, 3),
        })

    def log_contours(
            self,
            frame_id: int,
            timestamp_ms: int,
            contour_debug: List[ContourDebug],
        ) -> None:
        """
        Write one row per contour reaching the geometry stage this frame.
        No-op in disabled mode or when contour_debug is empty (it's only
        populated upstream when DEBUG_CONTOURS is on)
        """
        if not self._enabled or not contour_debug:
            return

        for cd in contour_debug:
            self._contours_w.writerow({
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "area": round(cd.area, 2),
                "aspect": round(cd.aspect, 3),
                "intensity": round(cd.intensity, 2),
                "roi_span": round(cd.roi_span, 3),
                "center_in_middle_third": int(cd.center_in_middle_third),
                "accepted": int(cd.accepted),
                "reject_reason": cd.reject_reason,
            })

    def close(self) -> None:
        if not self._enabled:
            return
        self._frames_fh.close()
        self._contours_fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()