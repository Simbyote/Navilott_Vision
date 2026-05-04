"""
main_pipeline.py

Complete Navilott Pipeline: Senior Design Project

Purpose:
    Links Phase 1 (Camera Acquisition), Phase 2 (Vision Perception), and
    Phase 3 (Navigation Signal Processing) into a single sequential loop

Concurrency model: Time-slicing (single-threaded)
    Execution is strictly sequential so that per-frame timing is deterministic 
    and tied directly to the camera frame rate 
    Navigation update rate is tied to the camera frame rate

Architecture:
    Phase 1 ->  Phase 2  ->  Phase 3  ->  EstimationPacket (Navigation)
    capture     preprocess                EMA temporal filter
                roi_crop                  motion consistency
                color_branch              confidence threshold
                geometry_branch           dead-reckoning fallback
                feature_fusion            IMU heading integration
                lane_offset
                phase2_out
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
import pigpio

# =============================================================================
# Pipeline modules
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
# PIPELINE CONFIGURATION
# =============================================================================

# Capture resolution
FRAME_WIDTH = 480
FRAME_HEIGHT = 360

# Frame rate: loop budget is derived from this
FPS = 15
LOOP_BUDGET_MS = 1000.0 / FPS          # e.g. 66.6 ms at 15 FPS

# Color space emitted by GStreamer: "YUV" or "BGR"
# Default: YUV
COLOR_SPACE = "YUV"

# Set True to write a debug overlay video to output.avi
SAVE_VIDEO = False

# Calibration file paths
HSV_RANGES_PATH = "vision_stack/calibration/hsv_ranges.json"
HSV_DUMMY_PATH = "vision_stack/dummy/dummy_hsv_ranges.json"

# Demo-mode confidence gates (traffic/sign disabled until calibrated)
TRAFFIC_CONF_THRESHOLD = 1.1   # effectively disabled; max is 1.0
SIGN_CONF_THRESHOLD = 1.1   # effectively disabled
LANE_CONF_THRESHOLD = 0.30  # operational

# Minimum lane width in pixels for two-boundary mode
MIN_LANE_WIDTH_PX = 150.0

# Offset of the camera
OFFSET_TRIM   = -0.12   # meters

# =============================================================================
# Motor Control Parameters
# =============================================================================

BASE_SPEED = 0.45   # Constant forward speed (0.0 to 1.0)
KP         = 0.40   # Proportional gain
KD         = 0.05   # Derivative gain — smooths correction jitter

_last_error: float = 0.0

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
# Hardware Setup
# =============================================================================

pi = pigpio.pi()

# Motor A (Left) — TB6612 AIN side
_ain1 = 24
_ain2 = 25
_pwma = 13
# Motor B (Right) — TB6612 BIN side
_bin1 = 27
_bin2 = 22
_pwmb = 12
_stby = 23

pi.set_mode(_ain1, pigpio.OUTPUT)
pi.set_mode(_ain2, pigpio.OUTPUT)
pi.set_mode(_bin1, pigpio.OUTPUT)
pi.set_mode(_bin2, pigpio.OUTPUT)
pi.set_mode(_stby, pigpio.OUTPUT)


def _drive(
        left_speed: float, 
        right_speed: float
    ) -> None:
    """
    Purpose:
        Drive the robot with specified left and right motor speeds 
        using the TB6612 motor driver

    Inputs:
        left_speed: [-1.0, 1.0]
        right_speed: [-1.0, 1.0]
    """
    right_speed = -right_speed  # invert if right motor is wired backward
    pi.write(_stby, 1)
    # Left
    spd_l = int(max(0.0, min(1.0, abs(left_speed))) * 1000000)
    pi.hardware_PWM(_pwma, 1000, spd_l)
    pi.write(_ain1, 1 if left_speed > 0 else 0)
    pi.write(_ain2, 1 if left_speed < 0 else 0)
    # Right
    spd_r = int(max(0.0, min(1.0, abs(right_speed))) * 1000000)
    pi.hardware_PWM(_pwmb, 1000, spd_r)
    pi.write(_bin1, 1 if right_speed > 0 else 0)
    pi.write(_bin2, 1 if right_speed < 0 else 0)

def _motor_selftest() -> None:
    """
    Purpose:
        Runs a short motor validation sequence before the main loop.
        Confirms both motors spin in the expected direction and that
        differential drive correction works as intended.

    Sequence:
        1. Forward       — both motors forward, should move straight
        2. Left turn     — left slow, right fast, should arc left
        3. Right turn    — left fast, right slow, should arc right
        4. Stop          — all stop, verify no coast

    Notes:
        Bot should be on a flat surface with room to move ~30 cm.
        Watch each phase and kill with Ctrl+C if behavior is wrong.
    """
    log.info("=== MOTOR SELF-TEST START ===")
    input("Place bot on ground with 30cm clearance. Press Enter to begin...")

    log.info("Phase 1: Forward (both motors equal)")
    _drive(0.3, 0.3)
    time.sleep(1.5)
    _drive(0.0, 0.0)
    time.sleep(0.5)

    result = input("Did it go straight? [y/n]: ").strip().lower()
    if result != "y":
        log.warning("Forward test failed — check motor polarity inversion in _drive()")
        log.warning("Aborting self-test. Fix _drive() before running pipeline.")
        _drive(0.0, 0.0)
        pi.write(_stby, 0)
        sys.exit(1)

    log.info("Phase 2: Left turn (left slow, right fast)")
    _drive(0.15, 0.35)
    time.sleep(1.0)
    _drive(0.0, 0.0)
    time.sleep(0.5)

    result = input("Did it arc LEFT? [y/n]: ").strip().lower()
    if result != "y":
        log.warning("Left turn failed — correction sign may be inverted in main loop")

    log.info("Phase 3: Right turn (left fast, right slow)")
    _drive(0.35, 0.15)
    time.sleep(1.0)
    _drive(0.0, 0.0)
    time.sleep(0.5)

    result = input("Did it arc RIGHT? [y/n]: ").strip().lower()
    if result != "y":
        log.warning("Right turn failed — correction sign may be inverted in main loop")

    log.info("Phase 4: Stop")
    _drive(0.0, 0.0)
    pi.write(_stby, 0)
    time.sleep(0.5)
    pi.write(_stby, 1)

    log.info("=== MOTOR SELF-TEST COMPLETE ===")
    input("Press Enter to continue to pipeline...")

# =============================================================================
# GStreamer pipeline string
# =============================================================================

def _build_gst_pipeline(
        width: int, 
        height: int, 
        fps: int, 
        color_space: str
    ) -> str:
    """
    Purpose:
        Builds a libcamera -> GStreamer -> OpenCV pipeline string
    
    Inputs:
        width: frame width in pixels
        height: frame height in pixels
        fps: frames per second
        color_space: "YUV" or "BGR"
    
    Outputs:
        GStreamer pipeline string to pass to cv2.VideoCapture() for Phase 1 camera acquisition
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

def _load_hsv(
        path: str, 
        dummy_path: str
    ) -> HSVRanges:
    """
    Purpose:
        Load HSV calibration with hard-error on missing calibrated file

    Inputs:
        path: path to calibration file
        dummy_path: path to dummy calibration file    
    Outputs:
        HSVRanges dataclass instance with loaded or dummy values
    """
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
# Phase 2 -> Phase 3 adapter
# =============================================================================

def _adapt_detections_for_p3(p2_detections) -> list:
    """
    Purpose:
        Convert feature_fusion.DetectionObject list into estimation.DetectionObject
        list expected by Phase3Processor
    
    Inputs:
        p2_detections: list of feature_fusion.DetectionObject from Phase 2

    Outputs:
        list of estimation.DetectionObject:
        feature_fusion.DetectionObject ->  estimation.DetectionObject
            .type          -> .type
            .label_detail  -> .label
            .position["x"] -> .position_x
            .position["y"] -> .position_y
            .confidence    -> .confidence
            .timestamp     -> .timestamp
    """
    adapted = []
    for d in p2_detections:
        adapted.append(P3DetectionObject(
            type = d.type,
            label = d.label_detail,
            position_x = d.position["x"],
            position_y = d.position["y"],
            confidence = d.confidence,
            timestamp = d.timestamp,
        ))
    return adapted


def _build_p3_input(
        p2_out, 
        detections_p3
    ) -> P3Phase2Output:
    """
    Purpose:
        Wrap adapted detections in the Phase 3 Phase2Output container
    Inputs:
        p2_out: Phase2Output from Phase 2, used for frame_id and timestamp
        detections_p3: list of estimation.DetectionObject adapted from Phase 2 detections
    Outputs:     
        P3Phase2Output with detections and metadata for Phase 3 processing
    """
    return P3Phase2Output(
        detections = detections_p3,
        frame_id = p2_out.frame_id,
        timestamp_ms = p2_out.timestamp_ms,
    )

# =============================================================================
# Sensor stub  TODO: replace with real pigpio/IMU reads
# =============================================================================

def _read_sensors() -> SensorSample:
    """
    Purpose:
        Returns a SensorSample for the current frame window.

    All fields are None until the hardware interface is wired up.
    Replace each None with the corresponding pigpio/MPU-6050 read.

    TODO:
        wheel_speed: encoder tick delta / dt
        distance_traveled: cumulative encoder ticks -> metres
        yaw_rate: MPU-6050 gyro Z (deg/s)
        lateral_accel: MPU-6050 accel Y (m/s²)
    """
    return SensorSample(
        wheel_speed = None,         # TODO: encoder
        distance_traveled = None,   # TODO: encoder
        yaw_rate = None,            # TODO: MPU-6050
        lateral_accel = None,       # TODO: MPU-6050
    )


# =============================================================================
# Main loop
# =============================================================================

def main() -> None:
    # Motor Test:
    _motor_selftest()
    time.sleep(5)

    # ==========================================================================
    # Startup: calibration, stage configs, Phase 3 processor
    # ==========================================================================
    hsv_ranges = _load_hsv(HSV_RANGES_PATH, HSV_DUMMY_PATH)
    blob_filter = BlobFilter()
    canny_params = CannyParams()
    lane_filter = LaneContourFilter()
    sign_filter = SignContourFilter()

    p3_config = Phase3Config(
        ema_alpha = 0.35,
        vote_window = 3,
        min_confidence_lane = LANE_CONF_THRESHOLD,
        min_confidence_traffic = TRAFFIC_CONF_THRESHOLD,
        min_confidence_sign = SIGN_CONF_THRESHOLD,
        px_per_meter = (FRAME_WIDTH / 2) / 0.35,
        deadreck_max_frames = 10,
    )
    p3_processor = Phase3Processor(p3_config)

    # ==========================================================================
    # Phase 1: open camera
    # ==========================================================================
    gst_pipeline = _build_gst_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS, COLOR_SPACE)
    log.info("Opening camera: %s", gst_pipeline)

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        log.error("Failed to open camera pipeline; is libcamera available?")
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
        log.info("Debug video writer opened -> output.avi")

    # ==========================================================================
    # Main loop
    # ==========================================================================
    frame_id = 0
    try:
        while True:
            t_frame_start = time.perf_counter()

            # =================================================================
            # Phase 1 — Capture
            # =================================================================
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                log.warning("Frame %d: read failed — skipping", frame_id)
                frame_id += 1
                continue

            timestamp_ms = int(time.time() * 1000)

            # =================================================================
            # Phase 2: Vision Perception
            # =================================================================
            # Step 1: Preprocessing (histogram equalization + Gaussian blur)
            # preprocess_frame() expects YUV in, returns YUV out.
            # capture.py / GStreamer hands us BGR from videoconvert, so we
            # convert before and after to satisfy the stage contract.
            frame_yuv      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
            preprocessed   = preprocess_frame(frame_yuv)               # -> YUV
            preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_YUV2BGR)

            # Step 2: ROI crop returns NumPy views (no copy)
            roi_result = crop_rois(preprocessed_bgr, frame_id=frame_id)

            # Step 3a: Color branch traffic light candidates
            # roi_crop produces BGR views; color_branch expects BGR
            tl_candidates, _tl_debug = extract_traffic_light_candidates(
                roi          = roi_result.traffic_roi.copy(),   # copy: branch modifies internally
                hsv_ranges   = hsv_ranges,
                blob_filter  = blob_filter,
                frame_id     = frame_id,
                timestamp_ms = timestamp_ms,
            )

            # Step 3b: Geometry branch lane + stop sign candidates
            # geometry.py _to_grayscale() internally handles BGR -> gray
            geo_result, _lane_debug, _sign_debug = run_geometry_branch(
                lane_roi     = roi_result.lane_roi.copy(),
                sign_roi     = roi_result.sign_roi.copy(),
                canny_params = canny_params,
                lane_filter  = lane_filter,
                sign_filter  = sign_filter,
                frame_id     = frame_id,
                timestamp_ms = timestamp_ms,
            )

            # Step 4: Feature fusion normalize and resolve conflicts
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
            p2_out = package_phase2_output(
                detections         = detections,
                frame_id           = frame_id,
                timestamp_ms       = timestamp_ms,
            )

            # =================================================================
            # Phase 3: Navigation Signal Processing
            # =================================================================

            # Read sensors (NOTE: currently returns dummy sample with all None fields)
            sensor_sample = _read_sensors()

            # Adapt Phase 2 detections to Phase 3 schema and run processor
            p3_detections = _adapt_detections_for_p3(p2_out.detections)
            p3_input      = _build_p3_input(p2_out, p3_detections)
            nav_packet    = p3_processor.process(p3_input, sensor_sample)

            # =================================================================
            # Motor Control
            # =================================================================
            global _last_error

            if nav_packet.drive_state == "stop":
                _drive(0.0, 0.0)
            else:
                error      = nav_packet.lane_offset + OFFSET_TRIM
                derivative = error - _last_error
                correction = (error * KP) + (derivative * KD)
                _drive(BASE_SPEED - correction, BASE_SPEED + correction)
                _last_error = error

            # =================================================================
            # Timing check
            # =================================================================
            t_frame_end    = time.perf_counter()
            frame_time_ms  = (t_frame_end - t_frame_start) * 1000.0

            if frame_time_ms > LOOP_BUDGET_MS:
                log.warning(
                    "Frame %d: budget exceeded  %.1f ms  (budget %.1f ms)",
                    frame_id, frame_time_ms, LOOP_BUDGET_MS,
                )

            # =================================================================
            # Navigation packet log
            # =================================================================
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

            # =================================================================
            # Optional debug video
            # =================================================================
            if out_writer is not None:
                _overlay = _draw_debug_overlay(
                    frame_bgr, nav_packet, lane_offset_result, frame_time_ms
                )
                out_writer.write(_overlay)

            frame_id += 1

    except KeyboardInterrupt:
        log.info("Stopped by user after %d frames.", frame_id)

    finally:
        _drive(0.0, 0.0)
        pi.write(_stby, 0)
        pi.stop()
        cap.release()
        if out_writer is not None:
            out_writer.release()
            log.info("Debug video saved -> output.avi")
        log.info("Pipeline shutdown complete.")


# =============================================================================
# Debug overlay helper  (used to SAVE VIDEO)
# =============================================================================

def _draw_debug_overlay(
        frame_bgr, 
        nav_packet, 
        lane_result, 
        frame_time_ms
    ) -> np.ndarray:
    """
    Purpose:
        Render a lightweight HUD on a copy of the frame for saved video debugging

    Inputs:
        frame_bgr: The input frame in BGR format
        nav_packet: The navigation packet containing state information
        lane_result: The result of lane detection
        frame_time_ms: The time taken to process the frame in milliseconds

    Output:
        vis: A copy of the input frame with a debug overlay showing navigation info and lane mode
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