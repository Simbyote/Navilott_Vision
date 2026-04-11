"""
phase2_linker_s1s5.py
=====================
Phase 2 Pipeline — Stages 1 through 5 only.

Runs preprocessing, ROI crop, color branch, geometry branch, and feature
fusion on real frame data. Stages 6 (perspective transform) and 7 (output
packaging) are stubbed — they will be connected when implemented.

Intended use: dataset validation. Run against trackT3/T4/T5 to verify that
the fusion output is correct before Phase 3 integration begins.

Usage
-----
  python phase2_linker_s1s5.py

  Processes all images found under SAMPLE_DIRS.
  Writes per-stage debug images to sN/results/.
  Prints per-frame detection summary and timing breakdown every 30 frames.

To use as a module (live pipeline integration later):
  from phase2_linker_s1s5 import load_config, run_s1_to_s5
  cfg    = load_config()
  result = run_s1_to_s5(frame, frame_id, timestamp_ms, cfg)
  # result.detections → list[DetectionObject], ready for Stage 6 input
"""

import os
import sys
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

# ---------------------------------------------------------------------------
# Stage imports
# ---------------------------------------------------------------------------

from preprocess import preprocess_frame, load_calibration, build_undistort_maps
from roi_crop import crop_rois, draw_roi_overlay, ROICropResult
from color_branch import (
    extract_traffic_light_candidates,
    load_hsv_ranges,
    HSVRanges,
    BlobFilter,
    draw_candidates as draw_color_candidates,
)
from geometry import (
    run_geometry_branch,
    CannyParams,
    LaneContourFilter,
    SignContourFilter,
)
from feature_fusion import (
    fuse_detections,
    SourceROIInfo,
    DetectionObject,
)


# ---------------------------------------------------------------------------
# Calibration file paths
# ---------------------------------------------------------------------------

CALIBRATION_CAMERA    = "vision_stack/calibration/camera_matrix.npz"
CALIBRATION_HSV       = "vision_stack/calibration/hsv_ranges.json"

CALIBRATION_CAMERA_DUMMY = "vision_stack/dummy/dummy_camera_matrix.npz"
CALIBRATION_HSV_DUMMY    = "vision_stack/dummy/dummy_hsv_ranges.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class S1S5Config:
    """
    Calibration data and tunable parameters for Stages 1–5.

    Stages 6 and 7 have no config here — they are not yet connected.
    When they are added, extend this dataclass rather than creating a new one.
    """
    camera_matrix:        np.ndarray
    dist_coeffs:          np.ndarray
    undistort_map1:       np.ndarray
    undistort_map2:       np.ndarray
    hsv_ranges:           HSVRanges
    blob_filter:          BlobFilter
    canny_params:         CannyParams
    lane_filter:          LaneContourFilter
    sign_filter:          SignContourFilter
    gaussian_kernel_size: tuple = (5, 5)
    gaussian_sigma:       float = 0.0


def load_config() -> S1S5Config:
    """
    Load calibration files and assemble S1S5Config.

    Falls back to dummy calibration files if real ones are not present.
    Homography is intentionally not loaded here — Stage 6 is not connected.
    """
    if os.path.exists(CALIBRATION_CAMERA):
        camera_matrix, dist_coeffs = load_calibration(CALIBRATION_CAMERA)
        print(f"[CONFIG] Camera calibration loaded from {CALIBRATION_CAMERA}")
    else:
        print(
            f"[CONFIG] WARNING: {CALIBRATION_CAMERA} not found. "
            "Using dummy calibration — undistortion is a no-op."
        )
        camera_matrix, dist_coeffs = load_calibration(CALIBRATION_CAMERA_DUMMY)

    map1, map2 = build_undistort_maps(camera_matrix, dist_coeffs, image_size=(640, 480))

    if os.path.exists(CALIBRATION_HSV):
        hsv_ranges = load_hsv_ranges(CALIBRATION_HSV)
        print(f"[CONFIG] HSV ranges loaded from {CALIBRATION_HSV}")
    else:
        print(
            f"[CONFIG] WARNING: {CALIBRATION_HSV} not found. "
            "Using dummy HSV ranges — color branch detections will be unreliable."
        )
        hsv_ranges = load_hsv_ranges(CALIBRATION_HSV_DUMMY)

    return S1S5Config(
        camera_matrix        = camera_matrix,
        dist_coeffs          = dist_coeffs,
        undistort_map1       = map1,
        undistort_map2       = map2,
        hsv_ranges           = hsv_ranges,
        blob_filter          = BlobFilter(),
        canny_params         = CannyParams(),
        lane_filter          = LaneContourFilter(),
        sign_filter          = SignContourFilter(),
        gaussian_kernel_size = (5, 5),
        gaussian_sigma       = 0.0,
    )


# ---------------------------------------------------------------------------
# Stage 5 result container
# ---------------------------------------------------------------------------

@dataclass
class S1S5Result:
    """
    Output of the S1–S5 pipeline.

    detections and roi_result are the inputs Stage 6 will need.
    Everything else is timing and debug data.

    Stage 6 (perspective transform) receives:
        roi_result.lane_roi  — the lane ROI to warp
        detections           — carried forward unchanged into Stage 7

    Stage 7 (output packaging) receives:
        detections           — from here
        transformed_coords   — from Stage 6 (None until then)
        frame_id, timestamp_ms
    """
    detections:   List[DetectionObject]   # → Stage 6/7 input
    roi_result:   ROICropResult           # → Stage 6 needs lane_roi
    frame_id:     int
    timestamp_ms: int
    times:        dict                    # per-stage ms timings
    fusion_log:   List[str]              # suppression/discard log from fusion


# ---------------------------------------------------------------------------
# Per-frame executor
# ---------------------------------------------------------------------------

def run_s1_to_s5(
    frame:        np.ndarray,
    frame_id:     int,
    timestamp_ms: int,
    cfg:          S1S5Config,
    debug_dir:    Optional[str] = None,
    stem:         str           = "frame",
) -> S1S5Result:
    """
    Run Stages 1–5 on a single BGR frame.

    Parameters
    ----------
    frame        : uint8 BGR from Phase 1 (or loaded from disk for dataset runs)
    frame_id     : monotonic frame counter
    timestamp_ms : capture timestamp in milliseconds
    cfg          : S1S5Config from load_config()
    debug_dir    : if set, write debug images here; None disables all disk writes
    stem         : filename stem for debug outputs (e.g. "frame_00042")

    Returns
    -------
    S1S5Result
    """
    times = {}

    # -------------------------------------------------------------------
    # Stage 1 — Preprocessing
    # -------------------------------------------------------------------
    t = time.time()
    conditioned = preprocess_frame(
        frame                = frame,
        camera_matrix        = cfg.camera_matrix,
        dist_coeffs          = cfg.dist_coeffs,
        gaussian_kernel_size = cfg.gaussian_kernel_size,
        gaussian_sigma       = cfg.gaussian_sigma,
        undistort_maps       = (cfg.undistort_map1, cfg.undistort_map2),
    )
    times["s1_preprocess"] = (time.time() - t) * 1000

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s1_preprocessed.png"), conditioned)

    # -------------------------------------------------------------------
    # Stage 2 — ROI Crop
    # -------------------------------------------------------------------
    t = time.time()
    roi_result: ROICropResult = crop_rois(conditioned, frame_id=frame_id)
    times["s2_roi_crop"] = (time.time() - t) * 1000

    if debug_dir:
        cv2.imwrite(
            os.path.join(debug_dir, f"{stem}_s2_roi_overlay.png"),
            draw_roi_overlay(conditioned, roi_result),
        )
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_lane.png"),
                    roi_result.lane_roi.copy())
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_traffic.png"),
                    roi_result.traffic_roi.copy())
        cv2.imwrite(os.path.join(debug_dir, f"{stem}_s2_roi_sign.png"),
                    roi_result.sign_roi.copy())

    # -------------------------------------------------------------------
    # Stage 3 — Color Branch
    # -------------------------------------------------------------------
    t = time.time()
    traffic_candidates, color_debug = extract_traffic_light_candidates(
        roi          = roi_result.traffic_roi,
        hsv_ranges   = cfg.hsv_ranges,
        blob_filter  = cfg.blob_filter,
        frame_id     = frame_id,
        timestamp_ms = timestamp_ms,
    )
    times["s3_color_branch"] = (time.time() - t) * 1000

    if debug_dir:
        h_channel = color_debug["hsv"][:, :, 0]
        cv2.imwrite(
            os.path.join(debug_dir, f"{stem}_s3_cb_hsv.png"),
            cv2.applyColorMap(h_channel, cv2.COLORMAP_HSV),
        )
        for mask_name in ("red", "yellow", "green"):
            cv2.imwrite(
                os.path.join(debug_dir, f"{stem}_s3_cb_mask_{mask_name}.png"),
                color_debug[mask_name],
            )
        cv2.imwrite(
            os.path.join(debug_dir, f"{stem}_s3_cb_candidates.png"),
            draw_color_candidates(roi_result.traffic_roi, traffic_candidates),
        )

    # -------------------------------------------------------------------
    # Stage 4 — Geometry Branch
    # -------------------------------------------------------------------
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
    times["s4_geometry_branch"] = (time.time() - t) * 1000

    if debug_dir:
        for key, img in lane_debug.items():
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_s4_gb_lane_{key}.png"), img)
        for key, img in sign_debug.items():
            cv2.imwrite(os.path.join(debug_dir, f"{stem}_s4_gb_sign_{key}.png"), img)

    # -------------------------------------------------------------------
    # Stage 5 — Feature Fusion
    # -------------------------------------------------------------------
    t = time.time()
    detections, fusion_summary = fuse_detections(
        traffic_candidates = traffic_candidates,
        lane_candidates    = geo_result.lane_candidates,
        sign_candidates    = geo_result.sign_candidates,
        frame_id           = frame_id,
        timestamp_ms       = timestamp_ms,
        source_rois        = SourceROIInfo(
            lane_shape    = roi_result.lane_roi.shape[:2],
            traffic_shape = roi_result.traffic_roi.shape[:2],
            sign_shape    = roi_result.sign_roi.shape[:2],
        ),
    )
    times["s5_feature_fusion"] = (time.time() - t) * 1000

    if debug_dir and fusion_summary.get("log"):
        with open(os.path.join(debug_dir, f"{stem}_s5_fusion_log.txt"), "w") as f:
            f.write("\n".join(fusion_summary["log"]))

    # -------------------------------------------------------------------
    # Stages 6 and 7 — NOT YET CONNECTED
    #
    # Stage 6 stub: perspective_transform.warp_lane_roi(roi_result.lane_roi, homography)
    #   → WarpResult → TransformedCoords
    #   Connect when calibration/homography_matrix.npz is available.
    #
    # Stage 7 stub: phase2_out.package_phase2_output(detections, transformed_coords, ...)
    #   → Phase2Output
    #   Connect when Stage 6 is live. Until then, detections in S1S5Result
    #   is the equivalent handoff point.
    # -------------------------------------------------------------------

    if frame_id % 30 == 0:
        breakdown = "  ".join(f"{k}={v:.1f}ms" for k, v in times.items())
        total     = sum(times.values())
        budget    = 33.3
        flag      = " *** OVER BUDGET ***" if total > budget else ""
        print(f"[timing] frame={frame_id:04d}  {breakdown}  total={total:.1f}ms{flag}")

    return S1S5Result(
        detections   = detections,
        roi_result   = roi_result,
        frame_id     = frame_id,
        timestamp_ms = timestamp_ms,
        times        = times,
        fusion_log   = fusion_summary.get("log", []),
    )


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

SAMPLE_DIRS = [
    "vision_stack/frames/trackT3",
    "vision_stack/frames/trackT4",
    "vision_stack/frames/trackT5",
]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


if __name__ == "__main__":
    try:
        cfg = load_config()
    except FileNotFoundError as e:
        print(f"[FATAL] Config load failed: {e}")
        sys.exit(1)

    total_ok   = 0
    total_fail = 0
    frame_id   = 0

    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        image_files = sorted(
            f for f in os.listdir(sample_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            print(f"[SKIP] No images in {sample_dir}")
            continue

        for filename in image_files:
            img_path = os.path.join(sample_dir, filename)
            stem     = os.path.splitext(filename)[0]

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[FAIL] Could not read: {img_path}")
                total_fail += 1
                continue

            try:
                result = run_s1_to_s5(
                    frame        = frame,
                    frame_id     = frame_id,
                    timestamp_ms = int(time.time() * 1000),
                    cfg          = cfg,
                    debug_dir    = results_dir,
                    stem         = stem,
                )
            except Exception as e:
                print(f"[FAIL] frame_id={frame_id:04d}  {img_path}: {type(e).__name__}: {e}")
                total_fail += 1
                frame_id   += 1
                continue

            # Per-frame summary
            by_type = {}
            for d in result.detections:
                by_type.setdefault(d.type, []).append(f"{d.confidence:.2f}")
            det_str = "  ".join(
                f"{t}=[{', '.join(cs)}]" for t, cs in by_type.items()
            ) or "none"

            print(f"[OK] frame={frame_id:04d}  {filename}  {det_str}")

            if result.fusion_log:
                for entry in result.fusion_log:
                    print(f"       {entry}")

            total_ok += 1
            frame_id += 1

    print(f"\nDone. {total_ok} processed, {total_fail} failed.")