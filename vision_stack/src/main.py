# =============================================================================
# Imports
from pathlib import Path
import cv2 as cv
import numpy as np

from preprocessing import *
from geometry import *

# =============================================================================
# Saving
def save_stage(output_dir: Path, base_name: str, stage: str, img):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}_{stage}.jpg"
    cv.imwrite(str(output_path), img)

# =============================================================================
# ROI Cropping
def crop_roi(im, output_dir: Path, base_name: str):
    tl_roi = roi_crop(
        im,
        y_start_ratio=0.0,
        y_end_ratio=1.0,
        x_start_ratio=0.0,
        x_end_ratio=1.0
    )

    lane_roi = roi_crop(
        im,
        y_start_ratio=0.45,
        y_end_ratio=0.85,
        x_start_ratio=0.0,
        x_end_ratio=1.0
    )

    right_roi = roi_crop(
        im,
        y_start_ratio=0.0,
        y_end_ratio=1.0,
        x_start_ratio=0.0,
        x_end_ratio=1.0
    )

    save_stage(output_dir, base_name, "tl_roi", tl_roi)
    save_stage(output_dir, base_name, "lane_roi", lane_roi)
    save_stage(output_dir, base_name, "right_roi", right_roi)

    return tl_roi, lane_roi, right_roi

# =============================================================================
# Lane Preprocessing
def preprocessing_stage(im):
    im_gray = grayscale(im)
    im_blurred = gaussian_blur(im_gray)
    im_equalized = equalize(im_blurred)
    return im_equalized

# =============================================================================
# Geometry Stage
def geometry_stage(im):
    edges = canny_edge(im)
    contours = extract_lane_contours(edges)
    edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    contour_vis = draw_contours(edges_bgr, contours)

    return {
        "edges": edges,
        "contours": contours,
        "contour_vis": contour_vis
    }

# =============================================================================
# Color Stage
def color_stage(im):
    return im

# =============================================================================
# Single Image Pipeline
def process_image(input_path: Path):
    im = cv.imread(str(input_path))
    if im is None:
        raise RuntimeError(f"Failed to load image: {input_path}")

    sample_dir = input_path.parent
    output_dir = sample_dir / "out"
    base_name = input_path.stem

    undistorted = undistort(im)

    tl_roi, lane_roi, right_roi = crop_roi(undistorted, output_dir, base_name)

    lane_preprocessed = preprocessing_stage(lane_roi)
    lane_stage = geometry_stage(lane_preprocessed)
    save_stage(output_dir, base_name, "lane_preprocessed", lane_preprocessed)
    save_stage(output_dir, base_name, "lane_edges", lane_stage["edges"])
    save_stage(output_dir, base_name, "lane_contours", lane_stage["contour_vis"])

    right_stage = color_stage(right_roi)
    tl_stage = color_stage(tl_roi)

    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "tl_roi": tl_roi,
        "lane_roi": lane_roi,
        "right_roi": right_roi,
        "tl_stage": tl_stage,
        "lane_stage": lane_stage,
        "right_stage": right_stage
    }

# =============================================================================
# Batch Processing
def process_duckietown_dataset(root_dir: str):
    root = Path(root_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue

        for image_path in sorted(sample_dir.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in image_extensions:
                continue

            print(f"[INFO] Processing: {image_path}")
            try:
                result = process_image(image_path)
                print(f"[OK] Saved outputs to: {result['output_dir']}")
            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")

# =============================================================================
# Main
if __name__ == "__main__":
    process_duckietown_dataset("vision_stack/sample_img/duckietown")