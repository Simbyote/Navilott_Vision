"""
preprocess.py

Preprocessing Stage

Purpose:
    Conditions the captured BGR frame before ROI cropping splits the pipeline
    into branches. The stage produces two outputs

    1. gray: single-channel, blurred
       Feeds the lane and sign ROIs. The geometry branch validates
       ndim == 2 and raises on a 3-channel input, so the BGR -> gray
       conversion belongs here and nowhere else.

    2. color: BGR, blurred
       Feeds the traffic ROI. The color branch thresholds in HSV and needs
       the chroma that grayscale discards.

Histogram equalization:
    Off by default. It lifted sensor noise enough to cost more in false
    contours than it bought in contrast. Kept behind PreprocessParams.equalize
    so the comparison can be regenerated for a report. Turning it on changes 
    the intensity distribution that LaneContourFilter.min_intensity was tuned against.
"""
import numpy as np
import cv2
from dataclasses import dataclass

from src.capture.camera import FrameData

# =============================================================================
# Debug Names
# =============================================================================
GRAY_SUFFIX = "_1_gray.png"
EQUALIZED_SUFFIX = "_2_equalized.png"
GRAY_BLUR_SUFFIX = "_3_blurred_gray.png"
COLOR_BLUR_SUFFIX = "_5_blurred.png"

# =============================================================================
# Input Dataclass
# =============================================================================
@dataclass(frozen=True)
class PreprocessParams:
    """
    Tuning for the preprocessing stage

    gray_kernel: (width, height) Gaussian kernel for the geometry path
                 Anisotropic on purpose: more smoothing across a lane line
                 than along it, so fragments bridge without the line
                 thinning out
    gray_sigma: Gaussian sigma for the geometry path; 0.0 derives it from
                the kernel size
    color_kernel: (width, height) Gaussian kernel for the color path
    color_sigma: Gaussian sigma for the color path; 0.0 derives it
    equalize: apply histogram equalization to the grayscale path before
              blurring
    """
    gray_kernel: tuple = (9, 3)
    gray_sigma: float = 0.0
    color_kernel: tuple = (5, 5)
    color_sigma: float = 0.0
    equalize: bool = False

# =============================================================================
# Output Dataclass
# =============================================================================
@dataclass(frozen=True)
class PreprocessResult:
    """
    Output of the preprocessing stage

    gray: (H, W) uint8, blurred. Source for the lane and sign ROIs
    color: (H, W, 3) uint8 BGR, blurred. Source for the traffic ROI
    frame_id: carried from FrameData, never re-derived
    timestamp_ms: carried from FrameData, never re-derived
    """
    gray: np.ndarray
    color: np.ndarray
    frame_id: int
    timestamp_ms: int

# =============================================================================
# Validation
# =============================================================================
def _validate_frame(
        frame: np.ndarray
    ) -> None:
    """
    Purpose:
        Reject frames the stage cannot condition, with a message that names
        what arrived rather than letting OpenCV raise from inside a kernel
    """
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
        kernel: tuple,
        name: str
    ) -> None:
    """
    Purpose:
        cv2.GaussianBlur requires odd, positive kernel dimensions. Catching
        it here names the offending parameter instead of surfacing an
        OpenCV assertion with no context
    """
    if len(kernel) != 2:
        raise ValueError(f"preprocess: {name} must be (width, height), got {kernel}")
    for axis, size in zip(("width", "height"), kernel):
        if size < 1 or size % 2 == 0:
            raise ValueError(
                f"preprocess: {name} {axis} must be odd and positive, got {size}"
            )

# =============================================================================
# Utility Functions
# =============================================================================
def to_grayscale(
        frame: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Convert BGR to single-channel

    Inputs:
        frame: (H, W) uint8 gray or (H, W, 3) uint8 BGR

    Outputs:
        (H, W) uint8
    """
    _validate_frame(frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def histogram_equalization(
        gray: np.ndarray
    ) -> np.ndarray:
    """
    Purpose:
        Redistribute intensities across the full 8-bit range to improve
        contrast. Single-channel only --- cv2.equalizeHist rejects a
        3-channel input, so the conversion must happen first

    Inputs:
        gray: (H, W) uint8

    Outputs:
        (H, W) uint8
    """
    if gray.ndim != 2:
        raise ValueError(
            f"preprocess: equalizeHist needs single-channel, got {gray.shape}. "
            "Call to_grayscale() first."
        )
    return cv2.equalizeHist(gray)

def gaussian_blur(
        frame: np.ndarray,
        kernel_size: tuple = (5, 5),
        sigma: float = 0.0
    ) -> np.ndarray:
    """
    Purpose:
        Suppress high-frequency sensor noise before edge detection and HSV
        thresholding. Reduces false contours and spurious mask blobs without
        moving coarse structural features

    Inputs:
        frame: (H, W) or (H, W, 3) uint8
        kernel_size: (width, height), both odd and positive
        sigma: 0.0 derives sigma from the kernel size

    Outputs:
        blurred array, same shape and dtype as the input
    """
    _validate_kernel(kernel_size, "kernel_size")
    return cv2.GaussianBlur(frame, kernel_size, sigma)

# =============================================================================
# Preprocessing Stage
# =============================================================================
def preprocess_frame(
        frame_data: FrameData,
        params: PreprocessParams = PreprocessParams()
    ) -> PreprocessResult:
    """
    Purpose:
        Run the preprocessing stage on one captured frame, producing both
        branch inputs and carrying the frame's identity forward unchanged

    Inputs:
        frame_data: FrameData from CameraSource.read() --- BGR, (H, W, 3)
        params: PreprocessParams configurations

    Outputs:
        PreprocessResult
    """
    _validate_frame(frame_data.frame)
    _validate_kernel(params.gray_kernel, "gray_kernel")
    _validate_kernel(params.color_kernel, "color_kernel")

    # Geometry path: BGR -> gray -> (optional equalize) -> blur
    gray = to_grayscale(frame_data.frame)
    if params.equalize:
        gray = histogram_equalization(gray)
    gray = gaussian_blur(
        gray, params.gray_kernel, params.gray_sigma
    )

    # Color path: BGR -> blur
    color = gaussian_blur(
        frame_data.frame, params.color_kernel, params.color_sigma
    )

    return PreprocessResult(
        gray = gray,
        color = color,
        frame_id = frame_data.frame_id,
        timestamp_ms = frame_data.timestamp_ms,
    )