"""
main_pipeline.py
================
AutoBot Unified Pipeline — Navilott Senior Design Project

Links Phase 1 (Camera Acquisition), Phase 2 (Vision Perception), and
Phase 3 (Navigation Signal Processing) into a single sequential loop.

Concurrency model: Time-slicing (Option 2). Execution is strictly sequential
so that per-frame timing is deterministic and tied directly to the camera
frame rate. Navigation update rate == camera frame rate.

Architecture:
    Phase 1  →  Phase 2  →  Phase 3  →  EstimationPacket (Navigation)
    capture     preprocess              EMA temporal filter
                roi_crop                motion consistency
                color_branch            confidence threshold
                geometry_branch         dead-reckoning fallback
                feature_fusion          IMU heading integration
                lane_offset
                phase2_out

Usage:
    python main_pipeline.py

    Ctrl-C to stop. Logs per-frame timing to stdout.
    Set SAVE_VIDEO = True to write a debug overlay to output.avi.

Directory layout expected:
    vision_stack/
        src/
            capture.py
            preprocess.py
            roi_crop.py
            color_branch.py
            geometry.py
            feature_fusion.py
            lane_offset.py
            phase2_out.py
            estimation.py
        calibration/
            hsv_ranges.json          (required for color branch)
            homography.npz           (optional — absent disables warp)
        dummy/
            dummy_hsv_ranges.json    (fallback for color branch)
"""

# =============================================================================
# Standard library
# =============================================================================
import sys
import time
import logging

# =============================================================================
# Third-party
# =============================================================================
import cv2
import numpy as np

# =============================================================================
# Pipeline modules — adjust sys.path if running from project root
# =============================================================================
sys.path.insert(0, "vision_stack/src")

from preprocess   import preprocess_frame
from roi_crop     import crop_rois
from color_branch import (
    extract_traffic_light_candidates,
    HSVRanges, BlobFilter,
    load_hsv_ranges,
)
from geometry     import (
    run_geometry_branch,
    CannyParams, LaneContourFilter, SignContourFilter,
)
from feature_fusion import fuse_detections, SourceROIInfo
from lane_offset    import compute_lane_offset
from phase2_out     import package_phase2_output
from estimation     import (
    Phase3Processor, Phase3Config,
    SensorSample,
    DetectionObject as P3DetectionObject,
    Phase2Output    as P3Phase2Output,
)

# =============================================================================
# ── PIPELINE CONFIGURATION  (edit here) ──────────────────────────────────────
# =============================================================================

# Capture resolution
FRAME_WIDTH  = 480
FRAME_HEIGHT = 360

# Frame rate — loop budget is derived from this
FPS          = 15
LOOP_BUDGET_MS = 1000.0 / FPS          # e.g. 66.6 ms at 15 FPS

# Color space emitted by GStreamer: "YUV" or "BGR"
# Default: YUV  (matches IMX219 capture pipeline)
COLOR_SPACE  = "YUV"

# Set True to write a debug overlay video to output.avi
SAVE_VIDEO   = False

# Calibration file paths
HSV_RANGES_PATH  = "vision_stack/calibration/hsv_ranges.json"
HSV_DUMMY_PATH   = "vision_stack/dummy/dummy_hsv_ranges.json"

# Demo-mode confidence gates (traffic/sign disabled until calibrated)
TRAFFIC_CONF_THRESHOLD = 1.1   # effectively disabled; max is 1.0
SIGN_CONF_THRESHOLD    = 1.1   # effectively disabled
LANE_CONF_THRESHOLD    = 0.30  # operational

# Minimum lane width in pixels for two-boundary mode
MIN_LANE_WIDTH_PX = 150.0

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# =============================================================================
# GStreamer pipeline string
# =============================================================================

def _build_gst_pipeline(width: int, height: int, fps: int, color_space: str) -> str:
    """
    Build a libcamera → GStreamer → OpenCV pipeline string.

    color_space "YUV" : outputs I420 → videoconvert will hand BGR to OpenCV
                        but the capture.py convention keeps the raw YUV.
                        We let videoconvert handle the final format so that
                        cv2.VideoCapture always gives us BGR. Downstream
                        stages that expect YUV must be updated if this is
                        changed — see NOTE below.

    NOTE: capture.py documents YUV as the emitted color space but OpenCV's
    CAP_GSTREAMER sink always returns BGR from videoconvert. The preprocess
    and geometry branches convert from YUV internally (cv2.COLOR_YUV2BGR).
    If you intend to pass raw YUV frames you need a custom appsink format
    filter; the pipeline below matches what capture.py does in practice.
    """
    fmt_map = {
        "YUV": f"video/x-raw,colorimetry=bt709,width={width},height={height},framerate={fps}/1",
        "BGR": f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1",
    }
    fmt_string = fmt_map.get(color_space.upper(), fmt_map["YUV"])

    return (
        "libcamerasrc ! "
        f"{fmt_string} ! "
        "videoconvert ! "
        "videoflip method=rotate-180 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

# =============================================================================
# Calibration loaders
# =============================================================================

def _load_hsv(path: str, dummy_path: str) -> HSVRanges:
    """Load HSV calibration with hard-error on missing calibrated file."""
    try:
        ranges = load_hsv_ranges(path)
        log.info("HSV ranges loaded from %s", path)
        return ranges
    except FileNotFoundError:
        log.warning(
            "HSV calibration not found at %s — "
            "falling back to dummy ranges. Color branch unreliable.", path
        )
    try:
        ranges = load_hsv_ranges(dummy_path)
        log.warning("Using dummy HSV ranges from %s", dummy_path)
        return ranges
    except Exception as exc:
        log.error("Failed to load dummy HSV ranges: %s — using uncalibrated defaults.", exc)
        return HSVRanges()   # structural scaffold only


# =============================================================================
# Phase 2 → Phase 3 adapter
# =============================================================================

def _adapt_detections_for_p3(p2_detections) -> list:
    """
    Convert feature_fusion.DetectionObject list into estimation.DetectionObject
    list expected by Phase3Processor.

    Field mapping:
      feature_fusion.DetectionObject  →  estimation.DetectionObject
        .type          → .type
        .label_detail  → .label
        .position["x"] → .position_x
        .position["y"] → .position_y
        .confidence    → .confidence
        .timestamp     → .timestamp
    """
    adapted = []
    for d in p2_detections:
        adapted.append(P3DetectionObject(
            type       = d.type,
            label      = d.label_detail,
            position_x = d.position["x"],
            position_y = d.position["y"],
            confidence = d.confidence,
            timestamp  = d.timestamp,
        ))
    return adapted


def _build_p3_input(p2_out, detections_p3) -> P3Phase2Output:
    """Wrap adapted detections in the Phase 3 Phase2Output container."""
    return P3Phase2Output(
        detections   = detections_p3,
        frame_id     = p2_out.frame_id,
        timestamp_ms = p2_out.timestamp_ms,
    )


# =============================================================================
# Sensor stub  (TODO: replace with real pigpio/IMU reads)
# =============================================================================

def _read_sensors() -> SensorSample:
    """
    Returns a SensorSample for the current frame window.

    All fields are None until the hardware interface is wired up.
    Replace each None with the corresponding pigpio/MPU-6050 read.

    TODO (Ana / Mike):
        wheel_speed       — encoder tick delta / dt
        distance_traveled — cumulative encoder ticks → metres
        yaw_rate          — MPU-6050 gyro Z (deg/s)
        lateral_accel     — MPU-6050 accel Y (m/s²)
    """
    return SensorSample(
        wheel_speed       = None,   # TODO: encoder
        distance_traveled = None,   # TODO: encoder
        yaw_rate          = None,   # TODO: MPU-6050
        lateral_accel     = None,   # TODO: MPU-6050
    )


# =============================================================================
# Main loop
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Startup: calibration, stage configs, Phase 3 processor
    # -------------------------------------------------------------------------
    hsv_ranges  = _load_hsv(HSV_RANGES_PATH, HSV_DUMMY_PATH)
    blob_filter = BlobFilter()
    canny_params = CannyParams()
    lane_filter  = LaneContourFilter()
    sign_filter  = SignContourFilter()

    p3_config = Phase3Config(
        ema_alpha              = 0.35,
        vote_window            = 3,
        min_confidence_lane    = LANE_CONF_THRESHOLD,
        min_confidence_traffic = TRAFFIC_CONF_THRESHOLD,
        min_confidence_sign    = SIGN_CONF_THRESHOLD,
        px_per_meter           = (FRAME_WIDTH / 2) / 0.35,  # ≈ 686 px/m at 480px
        deadreck_max_frames    = 10,
    )
    p3_processor = Phase3Processor(p3_config)

    # -------------------------------------------------------------------------
    # Phase 1: open camera
    # -------------------------------------------------------------------------
    gst_pipeline = _build_gst_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS, COLOR_SPACE)
    log.info("Opening camera: %s", gst_pipeline)

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        log.error("Failed to open camera pipeline — is libcamera available?")
        sys.exit(1)

    log.info(
        "Camera open. Resolution=%dx%d  FPS=%d  Budget=%.1f ms/frame",
        FRAME_WIDTH, FRAME_HEIGHT, FPS, LOOP_BUDGET_MS,
    )

    # Optional debug video writer
    out_writer = None
    if SAVE_VIDEO:
        fourcc     = cv2.VideoWriter_fourcc(*"XVID")
        out_writer = cv2.VideoWriter("output.avi", fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        log.info("Debug video writer opened → output.avi")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    frame_id = 0
    try:
        while True:
            t_frame_start = time.perf_counter()

            # -----------------------------------------------------------------
            # Phase 1 — Capture
            # -----------------------------------------------------------------
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                log.warning("Frame %d: read failed — skipping", frame_id)
                frame_id += 1
                continue

            timestamp_ms = int(time.time() * 1000)

            # -----------------------------------------------------------------
            # Phase 2 — Vision Perception
            # -----------------------------------------------------------------

            # Step 1: Preprocessing (histogram equalization + Gaussian blur)
            # preprocess_frame() expects YUV in, returns YUV out.
            # capture.py / GStreamer hands us BGR from videoconvert, so we
            # convert before and after to satisfy the stage contract.
            frame_yuv      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
            preprocessed   = preprocess_frame(frame_yuv)               # → YUV
            preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_YUV2BGR)

            # Step 2: ROI crop — returns NumPy views (no copy)
            roi_result = crop_rois(preprocessed_bgr, frame_id=frame_id)

            # Step 3a: Color branch — traffic light candidates
            # roi_crop produces BGR views; color_branch expects BGR
            tl_candidates, _tl_debug = extract_traffic_light_candidates(
                roi          = roi_result.traffic_roi.copy(),   # copy: branch modifies internally
                hsv_ranges   = hsv_ranges,
                blob_filter  = blob_filter,
                frame_id     = frame_id,
                timestamp_ms = timestamp_ms,
            )

            # Step 3b: Geometry branch — lane + stop sign candidates
            # geometry.py _to_grayscale() internally handles BGR → gray
            geo_result, _lane_debug, _sign_debug = run_geometry_branch(
                lane_roi     = roi_result.lane_roi.copy(),
                sign_roi     = roi_result.sign_roi.copy(),
                canny_params = canny_params,
                lane_filter  = lane_filter,
                sign_filter  = sign_filter,
                frame_id     = frame_id,
                timestamp_ms = timestamp_ms,
            )

            # Step 4: Feature fusion — normalize and resolve conflicts
            source_rois = SourceROIInfo(
                lane_shape    = roi_result.lane_roi.shape[:2],
                traffic_shape = roi_result.traffic_roi.shape[:2],
                sign_shape    = roi_result.sign_roi.shape[:2],
            )
            detections, _fusion_summary = fuse_detections(
                traffic_candidates = tl_candidates,
                lane_candidates    = geo_result.lane_candidates,
                sign_candidates    = geo_result.sign_candidates,
                frame_id           = frame_id,
                timestamp_ms       = timestamp_ms,
                source_rois        = source_rois,
            )

            # Step 5: Lane offset estimation
            lane_boundary_dets = [d for d in detections if d.type == "lane_boundary"]
            lane_offset_result = compute_lane_offset(
                detections        = lane_boundary_dets,
                frame_width       = roi_result.lane_roi.shape[1],
                frame_id          = frame_id,
                timestamp         = timestamp_ms,
                conf_threshold    = LANE_CONF_THRESHOLD,
                min_lane_width_px = MIN_LANE_WIDTH_PX,
            )

            # Step 6: Package Phase 2 output
            # transformed_coords = None (homography removed from demo scope)
            p2_out = package_phase2_output(
                detections         = detections,
                frame_id           = frame_id,
                timestamp_ms       = timestamp_ms,
            )

            # -----------------------------------------------------------------
            # Phase 3 — Navigation Signal Processing
            # -----------------------------------------------------------------

            # Read sensors (stubs until hardware wired)
            sensor_sample = _read_sensors()

            # Adapt Phase 2 detections to Phase 3 schema and run processor
            p3_detections = _adapt_detections_for_p3(p2_out.detections)
            p3_input      = _build_p3_input(p2_out, p3_detections)
            nav_packet    = p3_processor.process(p3_input, sensor_sample)

            # -----------------------------------------------------------------
            # Timing check
            # -----------------------------------------------------------------
            t_frame_end    = time.perf_counter()
            frame_time_ms  = (t_frame_end - t_frame_start) * 1000.0

            if frame_time_ms > LOOP_BUDGET_MS:
                log.warning(
                    "Frame %d: budget exceeded  %.1f ms  (budget %.1f ms)",
                    frame_id, frame_time_ms, LOOP_BUDGET_MS,
                )

            # -----------------------------------------------------------------
            # Navigation packet log (every frame)
            # -----------------------------------------------------------------
            log.info(
                "f=%04d  t=%.1fms  offset=%+.4f  head=%+.2f°  drive=%-7s  "
                "stop_sign=%s  lane_mode=%s",
                frame_id,
                frame_time_ms,
                nav_packet.lane_offset,
                nav_packet.heading_error,
                nav_packet.drive_state,
                "T" if nav_packet.stop_sign_detected else "F",
                lane_offset_result.mode,
            )

            # -----------------------------------------------------------------
            # Optional debug video
            # -----------------------------------------------------------------
            if out_writer is not None:
                _overlay = _draw_debug_overlay(
                    frame_bgr, nav_packet, lane_offset_result, frame_time_ms
                )
                out_writer.write(_overlay)

            frame_id += 1

    except KeyboardInterrupt:
        log.info("Stopped by user after %d frames.", frame_id)

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
            log.info("Debug video saved → output.avi")
        log.info("Pipeline shutdown complete.")


# =============================================================================
# Debug overlay helper  (used only when SAVE_VIDEO = True)
# =============================================================================

def _draw_debug_overlay(frame_bgr, nav_packet, lane_result, frame_time_ms) -> np.ndarray:
    """
    Render a lightweight HUD on a copy of the frame for saved video debugging.
    Does not add meaningful CPU cost when SAVE_VIDEO = False.
    """
    vis = frame_bgr.copy()
    H, W = vis.shape[:2]

    # Semi-transparent background strip at top
    overlay_strip = vis.copy()
    cv2.rectangle(overlay_strip, (0, 0), (W, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay_strip, 0.55, vis, 0.45, 0, vis)

    # Drive state color
    state_color = {
        "go":      (0, 200, 0),
        "caution": (0, 180, 220),
        "stop":    (0, 0, 220),
    }.get(nav_packet.drive_state, (200, 200, 200))

    hud_text = (
        f"off={nav_packet.lane_offset:+.3f}  "
        f"head={nav_packet.heading_error:+.1f}deg  "
        f"[{nav_packet.drive_state.upper()}]  "
        f"{frame_time_ms:.1f}ms  mode={lane_result.mode}"
    )
    cv2.putText(vis, hud_text, (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1, cv2.LINE_AA)

    # Offset bar at bottom of frame
    bar_y   = H - 10
    bar_mid = W // 2
    bar_len = int(nav_packet.lane_offset * bar_mid)
    cv2.line(vis, (bar_mid, bar_y - 4), (bar_mid, bar_y + 4), (180, 180, 180), 1)
    cv2.line(vis, (bar_mid, bar_y), (bar_mid + bar_len, bar_y), state_color, 3)

    return vis


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()