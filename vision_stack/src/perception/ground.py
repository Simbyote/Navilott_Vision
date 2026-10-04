"""Ground plane: undistorted frame pixels to floor centimeters, through one homography.

Purpose:
    The camera is rigidly mounted and the floor is flat, so one 3x3
    homography maps any point on the floor, as seen in the frame preprocess
    produces, to floor coordinates. It is fit once by
    scripts/calibrate_ground.py on frames from preprocess_frame with the
    robot's own preprocess settings, and is only valid for exactly those:
    the lens calibration, undistort_alpha and output size. Loading checks
    all three and refuses a homography fit under different ones, because a
    mismatch shifts every distance without raising anything.

Main package:
    GroundHomography: H (frame px -> floor cm) with the conditions it was fit
        under. Floor axes: X_cm right+, Y_cm forward+, origin at the robot
        reference point: the floor seen at the bottom-center of the frame,
        unless the calibration was given another one.
    load_ground_homography(): the calibration file, or None with a warning.
    fit_ground_homography(): H from matched frame and floor points, with the
        per-point error in cm; the calibration script and the tests share it.
    lens_id(): the identifier of a lens calibration a homography is tied to.
    fit_conditions_problem(): why a calibration fit on undistorted frames
        doesn't match how preprocess runs now; shared with stop_line_table.

Flow:
    1. calibrate_ground fits H on undistorted frames and writes the JSON.
    2. config.py loads it once into MEASURED.ground; nothing reads it per frame.
    3. Stages project points with to_floor(); None means no ground plane.
"""
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Lens identity
# =============================================================================

def lens_id(calibration_path: str | Path | None) -> str | None:
    """
    SHA-256 of a lens calibration's image_size, camera_matrix and dist_coeffs.

    Purpose:
        A homography is only valid for the lens model the frames were
        undistorted with. The file's "created" stamp isn't enough: a
        recalibration in the same second or a hand edit keeps it. Hashing the
        values preprocess actually uses catches both.

    Outputs:
        Hex digest, or None if the path is None or the file doesn't exist.
    """
    if calibration_path is None or not Path(calibration_path).is_file():
        return None
    calib = json.loads(Path(calibration_path).read_text())
    key = {k: calib[k] for k in ("image_size", "camera_matrix", "dist_coeffs")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


# =============================================================================
# The homography
# =============================================================================

@dataclass(frozen=True)
class GroundHomography:
    """
    Frame px -> floor cm for one camera setup. Frozen and hashable, so it can sit in PipelineConfig.

    Floor axes: X_cm right+, Y_cm forward+, origin at the robot reference point.
    """
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
        # Frozen: the cached arrays go in through object.__setattr__
        object.__setattr__(self, "_H", H)
        object.__setattr__(self, "_H_inv", np.linalg.inv(H))

    @classmethod
    def from_matrix(cls, H: np.ndarray, image_size, undistort_alpha: float,
                    lens_sha256: str | None, mean_cm: float = 0.0, max_cm: float = 0.0):
        """Build from a numpy 3x3, normalized so H[2][2] = 1."""
        H = np.asarray(H, np.float64).reshape(3, 3)
        H = H / H[2, 2]
        return cls(tuple(tuple(float(v) for v in row) for row in H),
                   (int(image_size[0]), int(image_size[1])), float(undistort_alpha),
                   lens_sha256, float(mean_cm), float(max_cm))

    def to_floor(self, points_px) -> np.ndarray:
        """(N, 2) frame px -> (N, 2) floor cm."""
        pts = np.asarray(points_px, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._H).reshape(-1, 2)

    def to_image(self, points_cm) -> np.ndarray:
        """(N, 2) floor cm -> (N, 2) frame px."""
        pts = np.asarray(points_cm, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self._H_inv).reshape(-1, 2)

    def forward_at_centerline(self, p0_px, p1_px) -> float | None:
        """
        Forward distance (Y_cm) where the floor line through two frame points crosses X = 0.

        Purpose:
            The distance to a stop line that matters for stopping is along the
            robot's own path, not to the line's nearest pixel. A straight line
            on the floor stays straight through the homography, so the
            crossing is exact even when the line is seen off to one side and
            has to be extended to reach X = 0.

        Outputs:
            Y_cm at X = 0, or None if the line runs (nearly) along the robot's axis.
        """
        (x0, y0), (x1, y1) = self.to_floor([p0_px, p1_px])
        if abs(x1 - x0) < 1e-6:
            return None
        return float(y0 + (0.0 - x0) * (y1 - y0) / (x1 - x0))


def fit_ground_homography(
        points_px,
        points_cm,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares homography over every point pair, and each pair's error in cm.

    Inputs:
        points_px: (N, 2) frame points, N >= 4, from undistorted frames.
        points_cm: (N, 2) the same points on the floor.

    Outputs:
        (H, errors_cm): H maps frame px -> floor cm, normalized so H[2][2] = 1;
        errors_cm is |H(px) - cm| per point.

    Raises:
        ValueError: Fewer than 4 points, mismatched lengths, or no fit.
    """
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


# =============================================================================
# Loading
# =============================================================================

def fit_conditions_problem(size, alpha: float, lens: str | None, preprocess,
                           frame_size: tuple[int, int]) -> str | None:
    """
    Why a calibration fit on undistorted frames can't be used now, or None if it can.

    Inputs:
        size: (width, height) it was fit at.
        alpha, lens: The undistort_alpha and lens_id() it was fit under.
        preprocess: The PreprocessParams the pipeline runs with.
        frame_size: (height, width) of the frames the pipeline will see.
    """
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
    Load the ground calibration, if it matches how preprocess will run.

    Inputs:
        path: calibration/ground_homography.json, from calibrate_ground.py.
        preprocess: The PreprocessParams the pipeline runs with.
        frame_size: (height, width) of the frames the pipeline will see.

    Outputs:
        GroundHomography; None, silently, when there's no file (it's
        optional: the stop-line table gives the stop line's cm without it,
        and nothing else reads it); None with a warning when it's malformed
        or was fit under a different image size, undistort_alpha or lens
        calibration, or undistortion is off. Never raises.

    Side effects:
        Reads path and the lens calibration file. Call once at startup.
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
