"""Frame conditioning that feeds the geometry and color branches.

Purpose:
    Conditions each captured BGR frame before ROI cropping splits the
    pipeline into branches. Owns the BGR -> gray conversion: the geometry
    branch rejects 3-channel input, so the conversion happens here and
    nowhere else. Lens undistortion also lives here, ahead of the split, so
    both branches see the same corrected geometry.

Main package:
    PreprocessResult: a blurred gray frame for the lane and sign ROIs, a
    blurred BGR frame for the traffic ROI (HSV thresholding needs the chroma
    that gray discards), the unblurred undistorted frame that detection
    coordinates refer to, and the frame identity carried from capture.

Flow:
    1. Validate the frame and blur kernels.
    2. Undistort, if a calibration is configured.
    3. Gray path: convert, optionally equalize, blur.
    4. Color path: blur.
"""

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2

from src.capture.camera import FrameData

from src.params import CAMERA_CALIB_PATH as DEFAULT_CALIBRATION_PATH
# Re-exported so debug harnesses can keep importing the suffixes from here
from src.params import (
    COLOR_BLUR_SUFFIX, EQUALIZED_SUFFIX, GRAY_BLUR_SUFFIX, GRAY_SUFFIX, UNDISTORT_SUFFIX,
)


@dataclass(frozen=True)
class PreprocessParams:
    """Tuning for the preprocessing stage. Kernels are (width, height) px, odd and positive."""
    # Anisotropic on purpose: more smoothing across a lane line than along
    # it, so fragments bridge without the line thinning out.
    gray_kernel: tuple[int, int] = (9, 3)
    gray_sigma: float = 0.0             # 0.0 derives sigma from the kernel size
    color_kernel: tuple[int, int] = (5, 5)
    color_sigma: float = 0.0            # 0.0 derives sigma from the kernel size
    # Off: it lifted sensor noise enough to cost more in false contours than
    # it bought in contrast. Kept so the report comparison can be regenerated.
    # Turning it on invalidates the LaneContourFilter.min_intensity tuning.
    equalize: bool = False
    # JSON from tools/calibrate_camera.py; None disables undistortion. Only
    # valid for the resolution, sensor mode, flip and focus it was captured at.
    calibration_path: str | None = None
    # [0, 1]. 0.0 crops to valid pixels. Higher keeps more field of view but
    # adds a curved black border whose edge the geometry branch picks up as a
    # false contour. Leave at 0.0 unless the ROI is kept clear of the border.
    undistort_alpha: float = 0.0


@dataclass(frozen=True)
class PreprocessResult:
    """Output of the preprocessing stage. frame_id and timestamp_ms are copied from FrameData, never re-derived."""
    gray: np.ndarray        # (H, W) uint8, blurred; source for the lane and sign ROIs
    color: np.ndarray       # (H, W, 3) uint8 BGR, blurred; source for the traffic ROI
    frame_id: int
    timestamp_ms: int
    # (H, W, 3) uint8 BGR, unblurred. Detection coordinates refer to this
    # frame, so draw debug overlays here, not on the raw capture. Same array
    # as the input frame when undistortion is off.
    undistorted: np.ndarray | None = None


def _validate_frame(
    frame: np.ndarray
) -> None:
    """Reject frames the stage can't condition, naming what arrived instead of letting OpenCV raise from inside a kernel."""
    if frame is None:
        raise ValueError("preprocess: frame is None")
    if frame.ndim not in (2, 3):
        raise ValueError(f"preprocess: expected 3 dims, got {frame.ndim}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"preprocess: (H,W,3) BGR, got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise TypeError(f"preprocess: expected uint8, got {frame.dtype}")

def _validate_kernel(
    kernel: tuple[int, int],
    name: str
) -> None:
    """Name the offending parameter when a kernel isn't odd and positive, instead of a context-free OpenCV assertion."""
    if len(kernel) != 2:
        raise ValueError(f"preprocess: {name} must be (width, height), got {kernel}")
    for axis, size in zip(("width", "height"), kernel):
        if size < 1 or size % 2 == 0:
            raise ValueError(
                f"preprocess: {name} {axis} must be odd and positive, got {size}"
            )


@lru_cache(maxsize=4)   # one configuration is live per run; headroom for A/B comparisons
def _undistort_maps(
    path: str,
    width: int,
    height: int,
    alpha: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the calibration and build remap tables, once per (path, size, alpha).

    Outputs:
        (map1, map2) for cv2.remap, or None if the file doesn't exist. The
        None is cached too, so the missing-file warning fires once per run.
    """
    calib_file = Path(path)
    if not calib_file.is_file():
        warnings.warn(
            f"preprocess: calibration {calib_file} not found; "
            "undistortion is OFF for this run. "
            "Run tools/calibrate_camera.py to create it.",
            stacklevel=2,
        )
        return None

    calib = json.loads(calib_file.read_text())
    cal_w, cal_h = calib["image_size"]
    if (cal_w, cal_h) != (width, height):
        raise ValueError(
            f"preprocess: {calib_file.name} was calibrated at {cal_w}x{cal_h} "
            f"but frames are {width}x{height}. Recalibrate at this resolution."
        )

    K = np.array(calib["camera_matrix"], np.float64)
    dist = np.array(calib["dist_coeffs"], np.float64)
    size = (width, height)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    return cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)

def undistort(
    frame: np.ndarray,
    params: PreprocessParams
) -> np.ndarray:
    """
    Remove lens distortion so straight lines in the world stay straight in the frame.

    Inputs:
        params: calibration_path selects the calibration (None = pass-through);
            undistort_alpha trades field of view against a black border.

    Outputs:
        Corrected BGR frame, same size as the input. The input array itself
        when undistortion is off or the calibration file is missing.

    Side effects:
        The first call per configuration reads the calibration file.

    Raises:
        ValueError: If undistort_alpha is outside [0, 1], or the calibration
            resolution doesn't match the frame.
    """
    if params.calibration_path is None:
        return frame
    if not 0.0 <= params.undistort_alpha <= 1.0:
        raise ValueError(
            f"preprocess: undistort_alpha must be in [0, 1], got {params.undistort_alpha}"
        )

    height, width = frame.shape[:2]
    maps = _undistort_maps(
        str(params.calibration_path), width, height, float(params.undistort_alpha)
    )
    if maps is None:
        return frame
    return cv2.remap(
        frame, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def to_grayscale(
    frame: np.ndarray
) -> np.ndarray:
    """
    Convert a BGR frame to single-channel.

    Inputs:
        frame: (H, W, 3) uint8 BGR. Not validated here; the caller owns that.

    Outputs:
        (H, W) uint8 gray.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def histogram_equalization(
    gray: np.ndarray
) -> np.ndarray:
    """
    Stretch intensities across the full 8-bit range to raise contrast.

    Inputs:
        gray: (H, W) uint8. cv2.equalizeHist rejects 3-channel input, so
            convert with to_grayscale() first.

    Outputs:
        Equalized (H, W) uint8.

    Raises:
        ValueError: If the input isn't single-channel.
    """
    if gray.ndim != 2:
        raise ValueError(
            f"preprocess: equalizeHist needs single-channel, got {gray.shape}. "
            "Call to_grayscale() first."
        )
    return cv2.equalizeHist(gray)

def gaussian_blur(
    frame: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 0.0
) -> np.ndarray:
    """
    Suppress high-frequency sensor noise before edge detection and HSV thresholding.

    Purpose:
        Reduces false contours and spurious mask blobs without moving coarse
        structural features.

    Inputs:
        frame: (H, W) gray or (H, W, 3) BGR, uint8.
        kernel_size: (width, height) px, both odd and positive. Larger
            kernels suppress more noise but soften thin features.
        sigma: 0.0 derives sigma from the kernel size. Larger values blur
            harder within the same kernel.

    Outputs:
        Blurred array, same shape and dtype as the input.

    Raises:
        ValueError: If kernel_size isn't odd and positive.
    """
    _validate_kernel(kernel_size, "kernel_size")
    return cv2.GaussianBlur(frame, kernel_size, sigma)


def preprocess_frame(
    frame_data: FrameData,
    params: PreprocessParams = PreprocessParams()
) -> PreprocessResult:
    """
    Run the preprocessing stage on one captured frame.

    Inputs:
        params: Blur, equalization and undistortion settings. The default
            runs with neither undistortion nor equalization.

    Outputs:
        PreprocessResult with both branch inputs and the undistorted frame,
        with frame_id and timestamp_ms carried forward unchanged.

    Side effects:
        The first call per configuration reads the calibration file.

    Raises:
        ValueError / TypeError: If the frame or a blur kernel is malformed,
            or the calibration doesn't match the frame size.
    """
    _validate_frame(frame_data.frame)
    _validate_kernel(params.gray_kernel, "gray_kernel")
    _validate_kernel(params.color_kernel, "color_kernel")

    # Undistort once, before the split, so both branches share the geometry
    frame = undistort(frame_data.frame, params)

    gray = to_grayscale(frame)
    if params.equalize:
        gray = histogram_equalization(gray)
    gray = gaussian_blur(
        gray, params.gray_kernel, params.gray_sigma
    )

    color = gaussian_blur(
        frame, params.color_kernel, params.color_sigma
    )

    return PreprocessResult(
        gray = gray,
        color = color,
        frame_id = frame_data.frame_id,
        timestamp_ms = frame_data.timestamp_ms,
        undistorted = frame,
    )