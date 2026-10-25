"""Ground plane: undistorted frame pixels to floor centimeters, through one homography.

Purpose:
    The camera is fixed and the floor flat, so one homography maps any floor
    point to floor cm. It holds only for the lens calibration, undistort_alpha
    and size it was fit under, so loading refuses any other: a mismatch would
    shift every distance silently.

Main package:
    GroundHomography: frame px -> floor cm, X right+, Y forward+, from the
    robot's reference point.
"""
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def lens_id(calibration_path: str | Path | None) -> str | None:
    """
    SHA-256 of the lens values preprocess uses; a recalibration or hand edit changes it, a timestamp may not.
    """
    if calibration_path is None or not Path(calibration_path).is_file():
        return None
    calib = json.loads(Path(calibration_path).read_text())
    key = {k: calib[k] for k in ("image_size", "camera_matrix", "dist_coeffs")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class GroundHomography:
    """Frozen and hashable, so it can sit in PipelineConfig."""
    H: tuple[tuple[float, float, float], ...]   # 3x3, row-major, frame px -> floor cm
    image_size: tuple[int, int]                 # (width, height) of the frames it was fit on
    undistort_alpha: float                      # preprocess's alpha at fit time
    lens_sha256: str | None                     # lens_id() of the calibration preprocess used
    reprojection_mean_cm: float = 0.0           # fit quality, from the calibration
    reprojection_max_cm: float = 0.0
    _H: np.ndarray = field(init=False, repr=False, compare=False)
    _H_inv: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        H = np.asarray(self.H, np.float64).reshape(3, 3)
        if abs(np.linalg.det(H)) < 1e-12:
            raise ValueError("GroundHomography: H is singular")
        object.__setattr__(self, "_H", H)
        object.__setattr__(self, "_H_inv", np.linalg.inv(H))

    @classmethod
    def from_matrix(cls, H: np.ndarray, image_size, undistort_alpha: float,
                    lens_sha256: str | None, mean_cm: float = 0.0, max_cm: float = 0.0):
        H = np.asarray(H, np.float64).reshape(3, 3)
        H = H / H[2, 2]
        return cls(tuple(tuple(float(v) for v in row) for row in H),
                   (int(image_size[0]), int(image_size[1])), float(undistort_alpha),
                   lens_sha256, float(mean_cm), float(max_cm))

    def to_floor(self, points_px) -> np.ndarray:
        pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._H).reshape(-1, 2)

    def to_image(self, points_cm) -> np.ndarray:
        pts = np.asarray(points_cm, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._H_inv).reshape(-1, 2)

    def forward_at_centerline(self, p0_px, p1_px) -> float | None:
        """
        Y_cm where the floor line through two frame points crosses X = 0; None if
        the line runs along the robot's axis. Exact even for a line seen off to one side.
        """
        (x0, y0), (x1, y1) = self.to_floor([p0_px, p1_px])
        if abs(x1 - x0) < 1e-6:
            return None
        return float(y0 + (0.0 - x0) * (y1 - y0) / (x1 - x0))


def fit_ground_homography(
        points_px,
        points_cm,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Least squares over N >= 4 point pairs: (H, |error| cm per point). ValueError if it can't fit."""
    px = np.asarray(points_px, np.float64).reshape(-1, 2)
    cm = np.asarray(points_cm, np.float64).reshape(-1, 2)
    if len(px) < 4 or len(px) != len(cm):
        raise ValueError(f"fit_ground_homography: need >= 4 matched points, got {len(px)} and {len(cm)}")
    H, _ = cv2.findHomography(px, cm, 0)          # 0: all points, least squares
    if H is None:
        raise ValueError("fit_ground_homography: no homography fits these points")
    H = H / H[2, 2]
    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, np.linalg.norm(proj - cm, axis=1)


def fit_conditions_problem(size, alpha: float, lens: str | None, preprocess,
                           frame_size: tuple[int, int]) -> str | None:
    """Why a fit can't be used with this preprocess and frame size, or None."""
    height, width = frame_size
    if tuple(size) != (width, height):
        return f"fit at {size[0]}x{size[1]}, frames are {width}x{height}"
    if abs(alpha - float(preprocess.undistort_alpha)) > 1e-9:
        return f"fit with undistort_alpha {alpha}, preprocess uses {preprocess.undistort_alpha}"
    if preprocess.calibration_path is None:
        return "undistortion is off, but it was fit on undistorted frames"
    current = lens_id(preprocess.calibration_path)
    if current is None:
        return f"lens calibration {preprocess.calibration_path} not found"
    if current != lens:
        return "fit against a different lens calibration"
    return None


def _refuse(path, why: str) -> None:
    warnings.warn(f"ground: {Path(path).name} not used ({why}); the stop line's cm come from the "
                  "stop-line table if there is one. Rerun scripts/calibrate_ground.py, or delete the "
                  "file to use the table alone.", stacklevel=3)
    return None


def load_ground_homography(
        path: str | Path,
        preprocess,
        frame_size: tuple[int, int],
    ) -> GroundHomography | None:
    """
    None without a file (it's optional); None with a warning if it's malformed, fit
    under another setup, or undistortion is off. Reads the files once, at startup.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        g = json.loads(path.read_text())
        H = np.array(g["H"], np.float64).reshape(3, 3)
        size = tuple(int(v) for v in g["image_size"])
        alpha = float(g["undistort_alpha"])
        lens = g["lens_calibration"]["sha256"]
        err = g.get("reprojection_cm", {})
    except (KeyError, TypeError, ValueError) as exc:
        return _refuse(path, f"malformed: {exc}")

    problem = fit_conditions_problem(size, alpha, lens, preprocess, frame_size)
    if problem is not None:
        return _refuse(path, problem)
    try:
        return GroundHomography.from_matrix(H, size, alpha, lens,
                                            err.get("mean", 0.0), err.get("max", 0.0))
    except ValueError as exc:
        return _refuse(path, str(exc))
