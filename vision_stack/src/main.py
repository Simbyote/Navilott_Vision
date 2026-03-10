# =============================================================================
# Imports
import cv2 as cv
import numpy as np

from preprocessing import *
from geometry import *

def save_stage(stage, img):
    cv.imwrite(f"vision_stack/sample_img/street_intersection_{stage}.jpg", img)

# =============================================================================
# ROI Cropping
def crop_roi(im):
    # Crop the image for the split stages:
    # Traffic Light ROI (top-middle of image)
    tl_roi = roi_crop(
        im,
        y_start_ratio=0.35,
        y_end_ratio=0.55,
        x_start_ratio=0.25,
        x_end_ratio=0.75
    )

    # Lower ROI for lanes (roughly bottom half of image)
    lane_roi = roi_crop(
        im,
        y_start_ratio=0.65,
        y_end_ratio=1.0,
        x_start_ratio=0.0,
        x_end_ratio=1.0
    )

    # Right edge for signs
    right_roi = roi_crop(
        im,
        y_start_ratio=0.40,
        y_end_ratio=0.8,
        x_start_ratio=0.7,
        x_end_ratio=0.9
    )

    save_stage("tl_roi", tl_roi)
    save_stage("lane_roi", lane_roi)
    save_stage("right_roi", right_roi)

    return tl_roi, lane_roi, right_roi

# =============================================================================
# Lane Preprocessing
# Lane specific preprocessing
def preprocessing_stage(im):
    # Apply grayscale
    im_gray = grayscale(im)
    save_stage("lane_gray", im_gray)

    # Apply Gaussian Blur
    im_blurred = gaussian_blur(im_gray)
    save_stage("lane_blurred", im_blurred)

    # Apply Histogram Equalization
    im_equalized = equalize(im_blurred)
    save_stage("lane_equalized", im_equalized)

    return im_equalized

# =============================================================================
# Geometry Stage
def geometry_stage(im):
    white = white_mask(im)
    save_stage("lane_white_mask_raw", white)

    white_clean = cleanup_mask(white)
    save_stage("lane_white_mask_clean", white_clean)

    white_filtered = filter_components(white_clean)
    save_stage("lane_white_mask_filtered", white_filtered)

    lines = cv.HoughLinesP(
        white_filtered,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=30,
        maxLineGap=30
    )

    vis = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    kept = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            kept.append((x1, y1, x2, y2))
            cv.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    save_stage("lane_white_mask_hough", vis)
    return kept

# =============================================================================
# Color Stage
def color_stage(im):
    # Placeholder
    return im

# =============================================================================
# Main
def main(INPUT_PATH):
    # ===================
    # Loading Stage
    im = cv.imread(INPUT_PATH)
    # Verify image
    if im is None:
        raise RuntimeError(f"Failed to load image: {INPUT_PATH}")

    # ===================
    # Preprocessing Stage

    # Undistort the Image
    undistorted = undistort(im)
    save_stage("undistorted", undistorted)

    # Splits the image into 3 ROI segments
    tl_roi, lane_roi, right_roi = crop_roi(undistorted)

    lane_equalized = preprocessing_stage(lane_roi)

    # ===================
    # Split Processing Stages: Geometry and Color
    lane_processed = geometry_stage(lane_equalized)
    tl_processed = color_stage(tl_roi)
    right_processed = color_stage(right_roi)

if __name__ == "__main__":
    INPUT_PATH  = "vision_stack/sample_img/street_intersection.jpg"
    main(INPUT_PATH)