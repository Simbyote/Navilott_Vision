"""
phase2_linker_s1s6.py
=====================
Phase 2 Pipeline — Stages 1 through 6.

Adds Stage 6 (lane offset estimation) to the S1-S5 chain.
Stage 7 (output packaging) remains stubbed.

Stage 6 replaces the perspective transform with a direct pixel-based
lateral offset computation. No homography or calibration file required.

Usage
-----
  python phase2_linker_s1s6.py

To use as a module:
  from phase2_linker_s1s6 import load_config, run_s1_to_s6
  cfg    = load_config()
  result = run_s1_to_s6(frame, frame_id, timestamp_ms, cfg)
  # result.lane_offset  → LaneOffsetResult, ready for Phase 3
  # result.detections   → list[DetectionObject], carried forward to Stage 7
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

from preprocess import preprocess_frame
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
from lane_offset import compute_lane_offset, LaneOffsetResult   # Stage 6


# ---------------------------------------------------------------------------
# Calibration file paths
# ---------------------------------------------------------------------------

CALIBRATION_HSV          = "vision_stack/calibration/hsv_ranges.json"
CALIBRATION_HSV_DUMMY    = "vision_stack/dummy/dummy_hsv_ranges.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class S1S6Config:
    """
    Calibration data and tunable parameters for Stages 1-6.

    lane_offset_conf_threshold : minimum candidate confidence for Stage 6.
        Candidates below this are excluded from offset anchor selection.
        Default 0.30 gates out the low-confidence noise tail. Lower if the
        robot loses lane center in frames where best candidates sit ~0.25.
    """
    hsv_ranges:                HSVRanges
    blob_filter:               BlobFilter
    canny_params:              CannyParams
    lane_filter:               LaneContourFilter
    sign_filter:               SignContourFilter
    gaussian_kernel_size:      tuple = (5, 5)
    gaussian_sigma:            float = 0.0
    lane_offset_conf_threshold: float = 0.30


def load_config() -> S1S6Config:
    if os.path.exists(CALIBRATION_HSV):
        hsv_ranges = load_hsv_ranges(CALIBRATION_HSV)
        print(f"[CONFIG] HSV ranges loaded from {CALIBRATION_HSV}")
    else:
        print(
            f"[CONFIG] WARNING: {CALIBRATION_HSV} not found. "
            "Using dummy HSV ranges — color branch detections will be unreliable."
        )
        hsv_ranges = load_hsv_ranges(CALIBRATION_HSV_DUMMY)

    return S1S6Config(
        hsv_ranges           = hsv_ranges,
        blob_filter          = BlobFilter(),
        canny_params         = CannyParams(),
        lane_filter          = LaneContourFilter(),
        sign_filter          = SignContourFilter(),
        gaussian_kernel_size = (5, 5),
        gaussian_sigma       = 0.0,
        lane_offset_conf_threshold = 0.30,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class S1S6Result:
    """
    Output of the S1-S6 pipeline.

    Phase 3 consumes:
        lane_offset   — primary steering error signal
        detections    — full detection list (traffic light, stop sign pass-through)
        frame_id, timestamp_ms

    Stage 7 (output packaging) receives:
        detections + lane_offset + frame_id + timestamp_ms
        → Phase2Output (not yet connected)
    """
    detections:   List[DetectionObject]
    lane_offset:  LaneOffsetResult        # Stage 6 output — Phase 3 entry point
    roi_result:   ROICropResult           # retained for debug overlay use
    frame_id:     int
    timestamp_ms: int
    times:        dict
    fusion_log:   List[str]


# ---------------------------------------------------------------------------
# Per-frame executor
# ---------------------------------------------------------------------------

def run_s1_to_s6(
    frame:        np.ndarray,
    frame_id:     int,
    timestamp_ms: int,
    cfg:          S1S6Config,
    debug_dir:    Optional[str] = None,
    stem:         str           = "frame",
) -> S1S6Result:
    """
    Run Stages 1–6 on a single BGR frame.

    Parameters
    ----------
    frame        : uint8 BGR from Phase 1 (or loaded from disk for dataset runs)
    frame_id     : monotonic frame counter
    timestamp_ms : capture timestamp in milliseconds
    cfg          : S1S6Config from load_config()
    debug_dir    : if set, write debug images here; None disables all disk writes
    stem         : filename stem for debug outputs

    Returns
    -------
    S1S6Result
    """
    times = {}

    # -------------------------------------------------------------------
    # Stage 1 — Preprocessing
    # -------------------------------------------------------------------
    t = time.time()
    conditioned = preprocess_frame(
        frame                = frame,
        gaussian_kernel_size = cfg.gaussian_kernel_size,
        gaussian_sigma       = cfg.gaussian_sigma,
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
    # Stage 6 — Lane Offset Estimation
    # frame_width is the lane ROI width, not the full frame width,
    # because position.x values from fusion are in lane ROI coordinates.
    # -------------------------------------------------------------------
    t = time.time()
    lane_roi_width = roi_result.lane_roi.shape[1]
    lane_offset: LaneOffsetResult = compute_lane_offset(
        detections      = detections,
        frame_width     = lane_roi_width,
        frame_id        = frame_id,
        timestamp        = timestamp_ms,
        conf_threshold  = cfg.lane_offset_conf_threshold,
    )
    times["s6_lane_offset"] = (time.time() - t) * 1000

    # -------------------------------------------------------------------
    # Stage 7 — NOT YET CONNECTED
    #
    # Stage 7 stub: phase2_out.package_phase2_output(
    #     detections, lane_offset, frame_id, timestamp_ms
    # ) → Phase2Output
    # -------------------------------------------------------------------

    if frame_id % 30 == 0:
        breakdown = "  ".join(f"{k}={v:.1f}ms" for k, v in times.items())
        total     = sum(times.values())
        flag      = " *** OVER BUDGET ***" if total > 33.3 else ""
        print(f"[timing] frame={frame_id:04d}  {breakdown}  total={total:.1f}ms{flag}")

    return S1S6Result(
        detections   = detections,
        lane_offset  = lane_offset,
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
                result = run_s1_to_s6(
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

            lo = result.lane_offset
            offset_str = (
                f"offset={lo.offset:+.4f}  mode={lo.mode}  "
                f"conf={lo.confidence:.2f}  boundaries={lo.boundary_count}"
            )
            if lo.lane_width_px is not None:
                offset_str += f"  width_px={lo.lane_width_px:.1f}"

            print(f"[OK] frame={frame_id:04d}  {filename}  {offset_str}")

            if result.fusion_log:
                for entry in result.fusion_log:
                    print(f"       {entry}")

            total_ok += 1
            frame_id += 1

    print(f"\nDone. {total_ok} processed, {total_fail} failed.")