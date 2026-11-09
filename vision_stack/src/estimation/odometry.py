"""Odometry: how far the robot has driven, from the wheel encoders' counts.

Purpose:
    Lets the rules measure in cm instead of time. It reads the cumulative
    counts, never counts per second times frame time, so a late or long
    frame adds no error.

Main package:
    Odometry: distance_cm, left_cm and right_cm since the first frame with
    counts; None without encoders.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Odometry:
    """+ = forward."""
    distance_cm: float | None = None     # the axle's midpoint: the mean of the wheels
    left_cm: float | None = None
    right_cm: float | None = None


NO_ODOMETRY = Odometry()


class Odometer:
    """
    A frame without counts holds the last distance; the next one catches up, since counts are cumulative.
    """
    def __init__(self, cm_per_count: float | None) -> None:
        self._scale = cm_per_count
        self._base: tuple[int, int] | None = None
        self._last = NO_ODOMETRY
        self.record: dict = {}

    def update(self, left_count: int | None, right_count: int | None, log_: list[str]) -> Odometry:
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
    """cm per count from the file, else nominal (logged), so the robot runs rather than not at all."""
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
