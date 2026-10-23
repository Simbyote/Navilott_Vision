"""Odometry: how far the robot has driven, from the wheel encoders' counts.

Purpose:
    Lets the rules measure in cm instead of time: an advance into an
    intersection, a stop at a line, a crossing's exit. It reads the
    cumulative counts, never counts per second times frame time, so a late
    or long frame adds no error (docs/vision_pipeline/phase3_estimation.md,
    "Odometry").

Main package:
    Odometry: distance_cm (the axle's midpoint), left_cm and right_cm since
    the first frame with counts; None without encoders.

Flow, each frame:
    1. Each wheel: counts since the first frame that had them, x cm_per_count
       (calibration/odometry.json, else the nominal wheel scale).
    2. The robot: the mean of the two.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Odometry:
    """Distance since the first frame with encoder counts, cm, + = forward. None without a scale or encoders."""
    distance_cm: float | None = None     # the axle's midpoint: the mean of the wheels
    left_cm: float | None = None
    right_cm: float | None = None


NO_ODOMETRY = Odometry()


class Odometer:
    """
    Turns each frame's cumulative encoder counts into distance driven.

    The first frame with counts is the zero. A frame without counts (a
    reading the hub missed) holds the last distance; the next frame with
    counts catches up, since the counts are cumulative.
    """
    def __init__(self, cm_per_count: float | None) -> None:
        self._scale = cm_per_count
        self._base: tuple[int, int] | None = None
        self._last = NO_ODOMETRY
        self.record: dict = {}

    def update(self, left_count: int | None, right_count: int | None, log_: list[str]) -> Odometry:
        """
        Inputs:
            left_count, right_count: Cumulative counts at the end of this
                frame's window, + = forward; None without encoders.
            log_: Phase 3's log for the frame.
        Outputs:
            The distance driven so far.
        """
        if self._scale is None or left_count is None or right_count is None:
            self.record = {"left_count": left_count, "right_count": right_count, "held": self._last != NO_ODOMETRY}
            return self._last
        if self._base is None:
            self._base = (left_count, right_count)
            log_.append(f"[ODOM] zero at counts {left_count}, {right_count}")
        left = (left_count - self._base[0]) * self._scale
        right = (right_count - self._base[1]) * self._scale
        self._last = Odometry(round((left + right) / 2.0, 2), round(left, 2), round(right, 2))
        self.record = {"left_count": left_count, "right_count": right_count, "held": False}
        return self._last


def load_odometry(path: str | Path, nominal: float | None = None) -> float | None:
    """
    cm per encoder count from calibration/odometry.json.

    Inputs:
        nominal: The scale to fall back on without a usable file (params'
            NOMINAL_CM_PER_COUNT, from the wheel and the gearbox); None
            leaves distances off.
    Outputs:
        cm_per_count; nominal (logged) when the file is missing or
        unreadable, so the robot runs on the nominal scale, or without
        odometry, rather than not at all.
    """
    p = Path(path)
    fallback = "the nominal scale" if nominal is not None else "distances are off"
    if not p.exists():
        log.info("no odometry calibration at %s: %s (scripts/calibrate_odometry.py)", p, fallback)
        return nominal
    try:
        value = float(json.loads(p.read_text())["cm_per_count"])
    except (ValueError, KeyError, TypeError) as e:
        log.warning("odometry calibration %s unreadable (%s): %s", p, e, fallback)
        return nominal
    if value <= 0.0:
        log.warning("odometry calibration %s has cm_per_count %s: %s", p, value, fallback)
        return nominal
    return value
