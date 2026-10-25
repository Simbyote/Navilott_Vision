"""Stop-line distance in cm from the line's row in the image, calibrated with tape marks.

Purpose:
    On a flat floor a line's distance follows from its row. A curve fit to
    tape marks, measured with the robot's own detector, gives the cm without
    the ground homography's checkerboard.

Main package:
    StopLineTable: cm = A / (B - rows) + C, and the image setup it is valid for.
"""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.perception.ground import fit_conditions_problem

MIN_MARKS = 3               # the curve has three parameters


@dataclass(frozen=True)
class StopLineTable:
    """Frozen and hashable, so it can sit in PipelineConfig."""
    a: float
    b: float                        # rows; the horizon, above every mark
    c: float
    image_size: tuple[int, int]     # (width, height) of the frames it was fit on
    undistort_alpha: float
    lens_sha256: str | None
    max_rows: float                 # the farthest mark's rows: beyond it the curve extrapolates
    error_max_cm: float = 0.0       # the worst mark's error in the fit

    def to_cm(self, rows: float) -> float:
        """Never negative."""
        return max(self.a / (self.b - rows) + self.c, 0.0)


def fit_stop_line_table(rows, cm) -> tuple[tuple[float, float, float], np.ndarray]:
    """
    Fit cm = A / (B - rows) + C to the marks; returns ((A, B, C), |error| cm per mark).

    Rearranged, rows x cm = B cm - (A + C B) + C rows is linear, so it's least
    squares. ValueError on fewer than MIN_MARKS, two marks at one row, or marks
    that don't get farther as they sit higher (a swapped mark).
    """
    r = np.asarray(rows, np.float64).ravel()
    y = np.asarray(cm, np.float64).ravel()
    if len(r) < MIN_MARKS or len(r) != len(y):
        raise ValueError(f"need at least {MIN_MARKS} marks with a row and a distance each, got {len(r)} and {len(y)}")
    if len(np.unique(np.round(r, 1))) < len(r):
        raise ValueError("two marks at the same row: space the tape marks further apart")
    order = np.argsort(r)
    if np.any(np.diff(y[order]) <= 0):
        raise ValueError("a mark higher in the image must be farther away: check each mark's distance")
    M = np.stack([y, -np.ones_like(y), r], axis=1)
    (b, v, c), *_ = np.linalg.lstsq(M, r * y, rcond=None)
    a = v - c * b
    if a <= 0 or b <= r.max():
        raise ValueError("the marks don't fit a flat floor's curve: check each mark's distance and that "
                         "the tape lies flat")
    fitted = a / (b - r) + c
    return (float(a), float(b), float(c)), np.abs(fitted - y)


def _refuse(path, why: str) -> None:
    warnings.warn(f"stop line table: {Path(path).name} not used ({why}); stop-line cm comes only from a "
                  "ground homography for this run. Rerun scripts/calibrate_stop_line.py.", stacklevel=3)
    return None


def load_stop_line_table(path, preprocess, frame_size: tuple[int, int]) -> StopLineTable | None:
    """None without a file; None with a warning if it's malformed or fit under another image setup."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        t = json.loads(path.read_text())
        a, b, c = (float(v) for v in t["curve"])
        size = tuple(int(v) for v in t["image_size"])
        alpha = float(t["undistort_alpha"])
        lens = t["lens_calibration"]["sha256"]
        max_rows = float(max(m["rows"] for m in t["marks"]))
        err = float(t.get("error_cm", {}).get("max", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        return _refuse(path, f"malformed: {exc!r}")
    problem = fit_conditions_problem(size, alpha, lens, preprocess, frame_size)
    if problem is not None:
        return _refuse(path, problem)
    return StopLineTable(a, b, c, size, alpha, lens, max_rows, err)
