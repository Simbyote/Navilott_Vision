"""
phase2_linker.py
================
Phase 2 Pipeline Orchestrator — Vision Perception

Connects all Phase 2 substages in the correct order, passing data between
them in memory. No intermediate disk reads/writes are used for stage-to-stage
passing. Debug images are written to disk only as a side-effect after each
stage completes — they do not participate in the next stage's input.

Pipeline execution order
------------------------
  [1] preprocess.py      preprocess_frame()           BGR → conditioned BGR
  [2] roi_crop.py        crop_rois()                  BGR → ROICropResult (3 views)
  [3] color_branch.py    extract_traffic_light_candidates()  traffic ROI → candidates
  [4] geometry.py        run_geometry_branch()         lane + sign ROI → candidates
  [5] feature_fusion.py  fuse_detections()             candidates → DetectionObjects
  [6] perspective_transform.py  warp_lane_roi()        lane ROI → bird's-eye view
  [7] phase2_out.py      package_phase2_output()       → Phase2Output

Data flow
---------
  raw BGR frame
      │
      ▼  [1] preprocess_frame
  conditioned BGR frame
      │
      ▼  [2] crop_rois
  ROICropResult
      │
      ├──── traffic_roi ──▶  [3] extract_traffic_light_candidates
      │                              → traffic_candidates
      │
      ├──── lane_roi ──────┐
      │                   └▶  [4] run_geometry_branch
      └──── sign_roi ──────┘          → lane_candidates, sign_candidates
      │
      │  traffic_candidates + lane_candidates + sign_candidates
      ▼  [5] fuse_detections
  detections: list[DetectionObject]
      │
      │  lane_roi (from ROICropResult, if homography available)
      ▼  [6] warp_lane_roi
  TransformedCoords (or None if no calibration file)
      │
      │  detections + transformed_coords
      ▼  [7] package_phase2_output
  Phase2Output  ← forwarded to Phase 3

Calibration dependencies (MISSING if not yet generated)
---------------------------------------------------------
  calibration/camera_matrix.npz     — required for undistortion (Step 1)
  calibration/hsv_ranges.json       — required for color branch (Step 3)
  calibration/homography_matrix.npz — required for perspective warp (Step 6)
                                       graceful fallback: warp is skipped,
                                       transformed_coords = None in output

Usage
-----
  python phase2_linker.py

  Processes every image found under the five standard sample directories.
  Writes debug output per stage into sN/results/.
  Prints a per-frame summary and a final totals line.

  To use as a module (live pipeline):
      from phase2_linker import load_pipeline_config, run_phase2_on_frame
      cfg = load_pipeline_config()
      output = run_phase2_on_frame(frame, frame_id, timestamp_ms, cfg)
"""

import os
import sys
import time
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Stage imports
# All stage files must be importable — see "Required edits to stage files"
# section at the bottom of this module.
# ---------------------------------------------------------------------------

# [1] Preprocessing
from preprocess import preprocess_frame, load_calibration

# [2] ROI cropping
from roi_crop import crop_rois, draw_roi_overlay, ROICropResult

# [3] Color branch
from color_branch import (
    extract_traffic_light_candidates,
    load_hsv_ranges,
    HSVRanges,
    BlobFilter,
    draw_candidates as draw_color_candidates,
)

# [4] Geometry branch
from geometry import (
    run_geometry_branch,
    CannyParams,
    LaneContourFilter,
    SignContourFilter,
)

# [5] Feature fusion
from feature_fusion import (
    fuse_detections,
    SourceROIInfo,
)

# [6] Perspective transform
from perspective_transform import (
    load_homography,
    warp_lane_roi,
    HomographyData,
    WarpResult,
)

# [7] Phase 2 output packaging
from phase2_out import (
    package_phase2_output,
    Phase2Output,
    TransformedCoords,
    DetectionObject,
)


# ---------------------------------------------------------------------------
# Pipeline configuration container
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All calibration data and tunable parameters needed to run Phase 2.

    camera_matrix, dist_coeffs: loaded from calibration/camera_matrix.npz
    hsv_ranges:                 loaded from calibration/hsv_ranges.json
    homography:                 loaded from calibration/homography_matrix.npz
                                May be None — perspective warp is skipped if absent.
    blob_filter:                BlobFilter defaults; tune per course hardware.
    canny_params:               CannyParams defaults; tune per Calibration Step 4.
    lane_filter:                LaneContourFilter defaults.
    sign_filter:                SignContourFilter defaults.
    gaussian_kernel_size:       Passed to preprocess_frame.
    gaussian_sigma:             Passed to preprocess_frame.
    """
    camera_matrix:       np.ndarray
    dist_coeffs:         np.ndarray
    hsv_ranges:          HSVRanges
    homography:          Optional[HomographyData]
    blob_filter:         BlobFilter
    canny_params:        CannyParams
    lane_filter:         LaneContourFilter
    sign_filter:         SignContourFilter
    gaussian_kernel_size: tuple = (5, 5)
    gaussian_sigma:       float = 0.0


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

CALIBRATION_CAMERA    = "vision_stack/calibration/camera_matrix.npz"
CALIBRATION_HSV       = "vision_stack/calibration/hsv_ranges.json"
CALIBRATION_HOMOGRAPHY = "vision_stack/calibration/homography_matrix.npz"

CALIBRATION_CAMERA_DUMMY = "vision_stack/dummy/dummy_camera_matrix.npz"
CALIBRATION_HSV_DUMMY    = "vision_stack/dummy/dummy_hsv_ranges.json"


def load_pipeline_config() -> PipelineConfig:
    """
    Load all calibration files and assemble a PipelineConfig.

    camera_matrix.npz  — prints a warning and sets dummy identity matrix + zero distortion if absent.
    hsv_ranges.json    — prints a warning and sets dummy hsv_ranges if absent.
    homography_matrix.npz — prints a warning and sets homography=None
                            if absent. Perspective warp will be skipped.

    Returns PipelineConfig.
    """
    # --- Camera intrinsics ---------------------------------------
    if not os.path.exists(CALIBRATION_CAMERA):
        print(
            f"[CONFIG] WARNING: camera_matrix.npz not found at {CALIBRATION_CAMERA}. "
            "Loading dummy calibration — undistortion will be a no-op."
        )
        camera_matrix, dist_coeffs = load_calibration(CALIBRATION_CAMERA_DUMMY)
    else:
        camera_matrix, dist_coeffs = load_calibration(CALIBRATION_CAMERA)
        print(f"[CONFIG] Camera calibration loaded from {CALIBRATION_CAMERA}")

    # --- HSV ranges ----------------------------------------------
    if not os.path.exists(CALIBRATION_HSV):
        print(
            f"[CONFIG] WARNING: hsv_ranges.json not found at {CALIBRATION_HSV}. "
            "Loading dummy HSV ranges — color branch detections will be unreliable."
        )
        hsv_ranges = load_hsv_ranges(CALIBRATION_HSV_DUMMY)
    else:
        hsv_ranges = load_hsv_ranges(CALIBRATION_HSV)
        print(f"[CONFIG] HSV ranges loaded from {CALIBRATION_HSV}")

    # --- Homography ----------------------------------
    homography = None
    if os.path.exists(CALIBRATION_HOMOGRAPHY):
        try:
            homography = load_homography(CALIBRATION_HOMOGRAPHY)
            print(f"[CONFIG] Homography loaded from {CALIBRATION_HOMOGRAPHY}")
        except (KeyError, ValueError) as e:
            print(f"[CONFIG] WARNING: failed to load homography — {e}. Warp stage will be skipped.")
    else:
        print(
            f"[CONFIG] WARNING: {CALIBRATION_HOMOGRAPHY} not found. "
            "Perspective warp will be skipped (transformed_coords = None in Phase2Output)."
        )

    return PipelineConfig(
        camera_matrix        = camera_matrix,
        dist_coeffs          = dist_coeffs,
        hsv_ranges           = hsv_ranges,
        homography           = homography,
        blob_filter          = BlobFilter(),          # tune for production
        canny_params         = CannyParams(),          # tune per Calibration Step 4
        lane_filter          = LaneContourFilter(),    # tune for production
        sign_filter          = SignContourFilter(),    # tune for production
        gaussian_kernel_size = (5, 5),
        gaussian_sigma       = 0.0,
    )


# ---------------------------------------------------------------------------
# Per-frame pipeline executor
# ---------------------------------------------------------------------------

def run_phase2_on_frame(
    frame:        np.ndarray,
    frame_id:     int,
    timestamp_ms: int,
    cfg:          PipelineConfig,
    debug_dir:    Optional[str] = None,
    stem:         str           = "frame",
) -> Phase2Output:
    """
    Run the full Phase 2 pipeline on a single BGR frame.

    Parameters
    ----------
    frame        : uint8 BGR image from Phase 1 (GStreamer appsink output)
    frame_id     : monotonic frame counter from capture loop
    timestamp_ms : capture timestamp in milliseconds
    cfg          : PipelineConfig from load_pipeline_config()
    debug_dir    : if not None, write debug images into this directory
    stem         : filename stem used for debug image names (e.g. "frame_0042")

    Returns
    -------
    Phase2Output — the Phase 2 → Phase 3 handoff object
    """
    times = {}

    # -----------------------------------------------------------------------
    # [1] PREPROCESSING
    #     preprocess_frame → conditioned BGR frame
    # -----------------------------------------------------------------------
    t = time.time()
    conditioned = preprocess_frame(
        frame                = frame,
        camera_matrix        = cfg.camera_matrix,
        dist_coeffs          = cfg.dist_coeffs,
        gaussian_kernel_size = cfg.gaussian_kernel_size,
        gaussian_sigma       = cfg.gaussian_sigma,
    )
    times["preprocess"] = (time.time() - t) * 1000

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s1_preprocessed.png"), conditioned)

    # -----------------------------------------------------------------------
    # [2] ROI CROPPING
    #     crop_rois → ROICropResult (lane_roi, traffic_roi, sign_roi as views)
    # 
    # Notes: ROI cropping can be moved one stage up to save time and memory later on
    #        May be more favorable (considered)
    # -----------------------------------------------------------------------
    t = time.time()
    roi_result: ROICropResult = crop_rois(conditioned, frame_id=frame_id)
    times["roi_crop"] = (time.time() - t) * 1000

    if debug_dir:
        overlay = draw_roi_overlay(conditioned, roi_result)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_overlay.png"), overlay)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_lane.png"),
                    roi_result.lane_roi.copy())
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_traffic.png"),
                    roi_result.traffic_roi.copy())
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_sign.png"),
                    roi_result.sign_roi.copy())

    # -----------------------------------------------------------------------
    # [3] COLOR BRANCH
    #     extract_traffic_light_candidates → traffic_candidates, debug_masks
    #     Input:  roi_result.traffic_roi
    # -----------------------------------------------------------------------
    t = time.time()
    traffic_candidates, color_debug = extract_traffic_light_candidates(
        roi          = roi_result.traffic_roi,
        hsv_ranges   = cfg.hsv_ranges,
        blob_filter  = cfg.blob_filter,
        frame_id     = frame_id,
        timestamp_ms = timestamp_ms,
    )
    times["color_branch"] = (time.time() - t) * 1000

    if debug_dir:
        # HSV H-channel false-color map
        h_channel = color_debug["hsv"][:, :, 0]
        hsv_vis   = cv2.applyColorMap(h_channel, cv2.COLORMAP_HSV)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s3_cb_hsv.png"), hsv_vis)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s3_cb_mask_red.png"),    color_debug["red"])
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s3_cb_mask_yellow.png"), color_debug["yellow"])
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s3_cb_mask_green.png"),  color_debug["green"])
        cb_vis = draw_color_candidates(roi_result.traffic_roi, traffic_candidates)
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s3_cb_candidates.png"), cb_vis)

    # -----------------------------------------------------------------------
    # [4] GEOMETRY BRANCH
    #     run_geometry_branch → GeometryBranchResult, lane_debug, sign_debug
    #     Input:  roi_result.lane_roi, roi_result.sign_roi
    # -----------------------------------------------------------------------
    t = time.time()
    geo_result, lane_debug, sign_debug = run_geometry_branch(
        lane_roi     = roi_result.lane_roi,
        sign_roi     = roi_result.sign_roi,
        canny_params = cfg.canny_params,
        lane_filter  = cfg.lane_filter,
        sign_filter  = cfg.sign_filter,
        frame_id     = frame_id,
        timestamp_ms = timestamp_ms,
    )
    times["geometry_branch"] = (time.time() - t) * 1000

    if debug_dir:
        for key, img in lane_debug.items():
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_s4_gb_lane_{key}.png"), img)
        for key, img in sign_debug.items():
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_s4_gb_sign_{key}.png"), img)

    # -----------------------------------------------------------------------
    # [5] FEATURE FUSION
    #     fuse_detections → detections: list[DetectionObject], debug_summary
    #     Input:  candidates from [3] and [4]
    # -----------------------------------------------------------------------
    t = time.time()
    source_rois = SourceROIInfo(
        lane_shape    = roi_result.lane_roi.shape[:2],
        traffic_shape = roi_result.traffic_roi.shape[:2],
        sign_shape    = roi_result.sign_roi.shape[:2],
    )

    detections, fusion_summary = fuse_detections(
        traffic_candidates = traffic_candidates,
        lane_candidates    = geo_result.lane_candidates,
        sign_candidates    = geo_result.sign_candidates,
        frame_id           = frame_id,
        timestamp_ms       = timestamp_ms,
        source_rois        = source_rois,
    )
    times["feature_fusion"] = (time.time() - t) * 1000

    if debug_dir and fusion_summary.get("log"):
        log_path = os.path.join(debug_dir, f"{stem}_s5_fusion_log.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(fusion_summary["log"]))

    # -----------------------------------------------------------------------
    # [6] PERSPECTIVE TRANSFORM (optional — skipped if no homography file)
    #     warp_lane_roi → WarpResult → TransformedCoords
    #     Input:  roi_result.lane_roi (must copy: warpPerspective needs contiguous array)
    # -----------------------------------------------------------------------
    t = time.time()
    transformed_coords: Optional[TransformedCoords] = None

    if cfg.homography is not None:
        warp_result: WarpResult = warp_lane_roi(
            # lane_roi is a NumPy view; copy() makes it contiguous for warpPerspective
            lane_roi   = roi_result.lane_roi.copy(),
            homography = cfg.homography,
        )
        transformed_coords = TransformedCoords(
            warped_image       = warp_result.warped_image,
            transformed_points = None,          # point-mode not used in standard run
            source_points      = None,
            output_width       = cfg.homography.output_width,
            output_height      = cfg.homography.output_height,
        )

        if debug_dir:
            cv2.imwrite(
                os.path.join(debug_dir, f"{stem}_s6_warped_lane.png"),
                warp_result.warped_image,
            )
    times["perspective_transform"] = (time.time() - t) * 1000

    # -----------------------------------------------------------------------
    # [7] PHASE 2 OUTPUT PACKAGING
    #     package_phase2_output → Phase2Output
    #     Input:  detections from [5], transformed_coords from [6]
    # -----------------------------------------------------------------------
    t = time.time()
    output: Phase2Output = package_phase2_output(
        detections         = detections,
        transformed_coords = transformed_coords,
        frame_id           = frame_id,
        timestamp_ms       = timestamp_ms,
    )
    times["package_phase2_output"] = (time.time() - t) * 1000

    return output


# ---------------------------------------------------------------------------
# Test harness  (dataset runner)
# ---------------------------------------------------------------------------

SAMPLE_DIRS = [
    "vision_stack/sample_img/duckietown/s1",
    "vision_stack/sample_img/duckietown/s2",
    "vision_stack/sample_img/duckietown/s3",
    "vision_stack/sample_img/duckietown/s4",
    "vision_stack/sample_img/duckietown/s5",
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


if __name__ == "__main__":
    # --- Load config (calibration files) ------------------------------------
    try:
        cfg = load_pipeline_config()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    total_ok   = 0
    total_fail = 0
    frame_id   = 0

    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Directory not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        image_files = sorted(
            f for f in os.listdir(sample_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            print(f"[SKIP] No images found in {sample_dir}")
            continue

        for filename in image_files:
            img_path = os.path.join(sample_dir, filename)
            stem     = os.path.splitext(filename)[0]

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[FAIL] Could not read: {img_path}")
                total_fail += 1
                continue

            timestamp_ms = int(time.time() * 1000)

            try:
                output = run_phase2_on_frame(
                    frame        = frame,
                    frame_id     = frame_id,
                    timestamp_ms = timestamp_ms,
                    cfg          = cfg,
                    debug_dir    = results_dir,
                    stem         = stem,
                )
            except Exception as e:
                print(f"[FAIL] frame_id={frame_id}  {img_path}: {type(e).__name__}: {e}")
                total_fail += 1
                frame_id += 1
                continue

            # Print per-frame summary
            det_summary = [
                f"{d.type}({d.confidence:.2f})" for d in output.detections
            ]
            warp_status = (
                f"warp={output.transformed_coords.output_width}x"
                f"{output.transformed_coords.output_height}"
                if output.transformed_coords is not None else "warp=skipped"
            )
            print(
                f"[OK] frame_id={frame_id:04d}  {img_path}\n"
                f"       detections={det_summary if det_summary else '[]'}  "
                f"{warp_status}"
            )

            total_ok  += 1
            frame_id  += 1

    print(f"\nDone. {total_ok} processed, {total_fail} failed.")