"""Odometry: how far the robot has driven, from the wheel encoders' counts.

Purpose:
    The encoders already steer the wheel speeds (navigation/wheel_speed.py)
    but nothing turned their counts into distance. This does, once per frame,
    so the rules can measure in cm instead of time: an advance into an
    intersection, a stop at a line, a crossing's exit.

    Distance comes from the cumulative counts the sensor hub reads, never
    from counts per second x frame time, so a late or long frame adds no
    error: the counts went on while the frame loop was busy. Each wheel's
    distance is its counts since the first frame that had them, times
    cm_per_count; the robot's (the midpoint of its axle) is the mean of the
    two, so a spin in place adds about nothing and an arc adds its center's
    path.

    cm_per_count comes from calibration/odometry.json
    (scripts/calibrate_odometry.py). Without it, or without encoders, every
    distance is None and nothing downstream may use one.

Main package:
    Odometer: update(left_count, right_count, log) -> Odometry, every frame.
        record: the frame's numbers, for the traced twin.
    Odometry: distance_cm, left_cm, right_cm (None without a scale or encoders).
    load_odometry(path) -> cm_per_count, or None when the file is missing.
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


def load_odometry(path: str | Path) -> float | None:
    """
    cm per encoder count from calibration/odometry.json.

    Outputs:
        cm_per_count; None (logged) when the file is missing or unreadable,
        so the robot runs without odometry rather than not at all.
    """
    p = Path(path)
    if not p.exists():
        log.info("no odometry calibration at %s: distances are off (scripts/calibrate_odometry.py)", p)
        return None
    try:
        value = float(json.loads(p.read_text())["cm_per_count"])
    except (ValueError, KeyError, TypeError) as e:
        log.warning("odometry calibration %s unreadable (%s): distances are off", p, e)
        return None
    if value <= 0.0:
        log.warning("odometry calibration %s has cm_per_count %s: distances are off", p, value)
        return None
    return value
