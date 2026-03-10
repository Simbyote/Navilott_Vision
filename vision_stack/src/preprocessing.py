import cv2 as cv
import numpy as np

# ==========================================
# Preprocessing Functions
def undistort(im: np.ndarray) -> np.ndarray:
    # Placeholder until calibration data exists
    return im.copy()


def grayscale(im: np.ndarray) -> np.ndarray:
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY)


def equalize(im: np.ndarray) -> np.ndarray:
    return cv.equalizeHist(im)


def gaussian_blur(im: np.ndarray) -> np.ndarray:
    return cv.GaussianBlur(im, (11, 11), 0)


def roi_crop(
    im: np.ndarray,
    y_start_ratio=0.0,
    y_end_ratio=1.0,
    x_start_ratio=0.0,
    x_end_ratio=1.0
) -> np.ndarray:
    h, w = im.shape[:2]

    y0 = int(h * y_start_ratio)
    y1 = int(h * y_end_ratio)

    x0 = int(w * x_start_ratio)
    x1 = int(w * x_end_ratio)

    return im[y0:y1, x0:x1]