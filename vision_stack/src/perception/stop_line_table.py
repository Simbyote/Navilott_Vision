"""Stop-line table: stop-line distance in cm from its row in the image, calibrated with tape marks.

Purpose:
    The ground homography (ground.py) needs a checkerboard found flat and
    square on the floor, which is hard to get right. The stop line only
    needs one number, how far ahead the line is, and that depends almost
    entirely on how many rows above the lane ROI's bottom the line appears
    (StopLineResult.distance_px). So this is a 1-D calibration instead: tape
    strips at a few measured distances, the row the robot's own stop-line
    detector reports for each (scripts/calibrate_stop_line.py), and the
    curve through them. For a flat floor seen by a fixed camera that curve is

        cm = A / (B - rows) + C

    (B is the horizon's height above the ROI bottom, in rows), fit to the
    marks by linear least squares. Measured by the same detector the robot
    drives with, it needs no board, no corner finding and no squareness.
    It gives distance ahead only; the homography stays for full floor X/Y.

Main package:
    StopLineTable: the fit and the conditions it's valid for; to_cm(rows).
    fit_stop_line_table(): (A, B, C) and each mark's error in cm.
    load_stop_line_table(): the calibration file, or None.

Flow:
    1. calibrate_stop_line fits the marks and writes the JSON.
    2. config.py loads it once into MEASURED.stop_line_table.
    3. stop_line_distance uses it for distance_cm when there's no homography.
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
    """
    Rows above the lane ROI bottom -> cm ahead of the reference the marks were measured from.

    Frozen and hashable, so it can sit in PipelineConfig.
    """
    a: float
    b: float                        # rows; the horizon, above every mark
    c: float
    image_size: tuple[int, int]     # (width, height) of the frames it was fit on
    undistort_alpha: float
    lens_sha256: str | None
    max_rows: float                 # the farthest mark's rows: beyond it the curve extrapolates
    error_max_cm: float = 0.0       # the worst mark's error in the fit

    def to_cm(self, rows: float) -> float:
        """cm ahead for a line this many rows above the lane ROI bottom; never negative."""
        return max(self.a / (self.b - rows) + self.c, 0.0)


def fit_stop_line_table(rows, cm) -> tuple[tuple[float, float, float], np.ndarray]:
    """
    Fit cm = A / (B - rows) + C to the marks.

    Rearranged, (cm - C)(B - rows) = A is linear in B, (A + C B) and C:
    rows x cm = B cm - (A + C B) + C rows, solved by least squares.

    Inputs:
        rows: Each mark's distance_px, the rows above the lane ROI bottom.
        cm: Each mark's measured distance.

    Outputs:
        ((A, B, C), errors_cm): errors_cm is |fit - measured| per mark.

    Raises:
        ValueError: Fewer than MIN_MARKS marks, two marks at one row, or
            marks that aren't farther in cm as they sit higher in the image
            (a mismeasured or swapped mark).
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
    """
    Load the stop-line table, if it matches how preprocess will run.

    Inputs:
        path: calibration/stop_line_table.json, from calibrate_stop_line.py.
        preprocess: The PreprocessParams the pipeline runs with.
        frame_size: (height, width) of the frames the pipeline will see.

    Outputs:
        StopLineTable; None, silently, when there's no file (it's optional);
        None with a warning when it's malformed or was fit under a different
        image size, undistort_alpha or lens calibration. Never raises.
    """
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
