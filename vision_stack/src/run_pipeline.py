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
    Phase 1 (Capture) -> Phase 2 (Processing) -> Phase 3 (Estimation)
    capture -> preprocess EMA temporal filter
                roi_crop motion consistency
                color_branch: confidence threshold
                geometry_branch: dead-reckoning fallback
                feature_fusion: IMU heading integration
                lane_offset
                phase2_out
"""

# =============================================================================
# Standard library
# =============================================================================
import os
import sys
import time
import logging
import itertools

# =============================================================================
# Testing Mode
# =============================================================================
BENCH_MODE = False      # Non-raspberry pi testing
DEBUG_CONTOURS = True   # Logs contours for debugging

# =============================================================================
# Third-Party
# =============================================================================
import cv2
import numpy as np
if not BENCH_MODE:
    import pigpio   # DMA PWM and requires pigpiod

# =============================================================================
# Pipeline Modules
# =============================================================================
sys.path.insert(0, "vision_stack/src")

if not BENCH_MODE:
    from system import System               # Button and display integration
    from imu import IMUReader, IMUFrame     # Board imports and bus I/O
else:
    # =============================================================================
    # TEST ENV ONLY: System and IMU placeholders
    # =============================================================================
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class IMUFrame:
        """
        # Mirror of imu.IMUFrame: imports board and bus I/O; cannot be imported off pi
        """
        mean_yaw_rate_dps: float | None = None
        peak_lateral_accel: float | None = None
        sample_count: int = 0
        valid: bool = False

    class IMUReader:
        """
        snapshot() always returns IMUFrame(valid=False). Heading behavior is not testable
        on bench
        """
        def __init__(self, *args, **kwargs): pass
        def start(self) -> None: pass
        def stop(self, timeout: float = 0.5) -> None: pass
        def snapshot(self) -> IMUFrame: return IMUFrame()

    class System:
        """
        wait_for_start() begins immediately with no button input; display updates are ignored
        """
        def wait_for_start(self) -> None: pass
        def run_countdown(self) -> None: pass
        def update_display(self, elapsed_s: float) -> None: pass
        def show_final_time(self, elapsed_s: float) -> None:
            print(f"[BENCH] final time: {elapsed_s:.2f} s")
        def cleanup(self) -> None: pass

from config import(
    Config, FrameTags, 
    intersection_edge_ratio,
    INTERSECTION_EDGE_RATIO_THRESH,
)
from csv_logger import RunLogger


from preprocess import preprocess_frame
from roi_crop import crop_rois
from color_branch import (
    extract_traffic_light_candidates,
    HSVRanges, BlobFilter,
    load_hsv_ranges,
)
from geometry import (
    run_geometry_branch,
    CannyParams, LaneContourFilter, SignContourFilter,
)
from feature_fusion import fuse_detections, SourceROIInfo
from lane_offset import compute_lane_offset
from phase2_out import package_phase2_output
from estimation import (
    Phase3Processor, Phase3Config,
    SensorSample,
    DetectionObject as P3DetectionObject,
    Phase2Output as P3Phase2Output,
)

# =============================================================================
# Configuration Parameters
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
SAVE_VIDEO = True

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
OFFSET_TRIM = 0.0   # meters

# =============================================================================
# Motor Control Parameters
# =============================================================================
BASE_SPEED = 0.45   # Constant forward speed (0.0 to 1.0)
KP = 0.40   # Proportional gain
KD = 0.05   # Derivative gain;smooths correction jitter

# Soft-start: seconds to linearly ramp from 0 -> BASE_SPEED any time driving
# resumes from a full stop (segment start, or a "stop" drive_state ending).
# Targets motor inrush current at the moment of the speed jump, not just
# steady-state draw -- lowering BASE_SPEED alone doesn't address this.
RAMP_SECONDS = 0.75

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

# Motor A (Left) — TB6612 AIN side
_ain1 = 25
_ain2 = 24
_pwma = 13
# Motor B (Right) — TB6612 BIN side
_bin1 = 27
_bin2 = 22
_pwmb = 12
_stby = 23

if not BENCH_MODE:
    pi = pigpio.pi()
    pi.set_mode(_ain1, pigpio.OUTPUT)
    pi.set_mode(_ain2, pigpio.OUTPUT)
    pi.set_mode(_bin1, pigpio.OUTPUT)
    pi.set_mode(_bin2, pigpio.OUTPUT)
    pi.set_mode(_stby, pigpio.OUTPUT)
else:
    # ============================================================================
    # TEST ENV ONLY: pigpio placeholder
    # All GPIO writes / PWM calls are ignored
    # _drive executes on normal PD path; motor controls are observable
    # ============================================================================
    class _DummyPi:
        connected = True
        def set_mode(self, *args) -> None: pass
        def write(self, *args) -> None: pass
        def hardware_PWM(self, *args) -> None: pass
        def stop(self) -> None: pass

    pi = _DummyPi()

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
    # =============================================================================
    # TEST ENV ONLY: pi is a dummy; all writes are ignored
    # =============================================================================
    if BENCH_MODE:
        log.debug("[BENCH] _drive L=%.3f R=%.3f", left_speed, right_speed)

    right_speed = -right_speed  # invert if right motor
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
# Calibration Loaders
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
# Phase 2 -> Phase 3 Adapter
# =============================================================================
def _adapt_detections_for_p3(
        p2_detections,
        lane_roi_width: float,
    ) -> list:
    """
    Purpose:
        Convert feature_fusion.DetectionObject list into estimation.DetectionObject
        list expected by Phase3Processor

    Inputs:
        p2_detections: list of feature_fusion.DetectionObject from Phase 2
        lane_roi_width: width in px of the lane ROI this frame's detections
            were measured against (roi_result.lane_roi.shape[1]) — the
            center reference for the lane_boundary conversion below

    Outputs:
        list of estimation.DetectionObject:
        feature_fusion.DetectionObject -> estimation.DetectionObject
            .type -> .type
            .label_detail -> .label
            .position["x"] -> .position_x
                lane_boundary: estimation.py's contract requires position_x
                to already be the signed pixel offset from the ROI center
                column. feature_fusion.DetectionObject.position["x"] is
                ROI-local (per feature_fusion.py's own docstring: "position
                is the centroid ... in ROI pixel coordinates", always >= 0),
                so it's centered here: x - lane_roi_width/2. Matches the
                same center reference compute_lane_offset() already uses
                (frame_width=lane_roi_width there too).
                traffic_light / stop_sign: left uncentered — nothing in
                Phase 3 documents them as requiring a signed offset, and
                _motion_consistency()/_classify_traffic()/_classify_stop_sign()
                only use position_x/position_y for jump-distance comparisons,
                which don't care about the coordinate origin.
            .position["y"] -> .position_y
            .confidence -> .confidence
            .timestamp -> .timestamp
    """
    lane_center_px = lane_roi_width / 2.0
    adapted = []
    for d in p2_detections:
        x = d.position["x"]
        if d.type == "lane_boundary":
            x = x - lane_center_px
        adapted.append(P3DetectionObject(
            type = d.type,
            label = d.label_detail,
            position_x = x,
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
# IMU Reader @TODO Integrate encoder data
# =============================================================================

def _read_sensors(
        imu: IMUReader
    ) -> tuple[SensorSample, IMUFrame]:
    frame = imu.snapshot()
    sample = SensorSample(
        wheel_speed = None,
        distance_traveled = None,
        yaw_rate = frame.mean_yaw_rate_dps if frame.valid else None,
        lateral_accel = frame.peak_lateral_accel if frame.valid else None,
    )
    return sample, frame

# ============================================================================
# Bench Frame Source
# ============================================================================
def _bench_frame_iter(
        frames_dir: str
    ):
    """
    TEST ENV ONLY: Iterate BGR frames from a directory of images (sorted by filename)
    as a stand-in for the camera. The Gstreamer/libcamera pipeline is unaffected.
    Alternate source is selected only when --frames is given on bench

    Yields (ret, frame_bgr) matching cv2.VideoCapture.read() semantics, then
    (False, None) once exhausted so the main loop terminates
    """
    import os
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            f"_bench_frame_iter: {frames_dir!r} not found "
            f"(cwd={os.getcwd()}) — paths are relative to the repo root"
        )
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    log.info("[BENCH] frame source: %d images from %s", len(files), frames_dir)
    def _gen():                       # ← the yields live in here now
        for path in files:
            frame = cv2.imread(path)
            yield (frame is not None), frame
        yield False, None
    return _gen()

def _parse_args():
    """
    CLI: independently toggle fixes, optionally select a bench image dir
    """
    import argparse
    ap = argparse.ArgumentParser(description="Navilott pipeline (fix-toggle bench build)")
    ap.add_argument(
        "--fix", action="append", default=[], metavar="NAME",
        choices=list(Config.FIX_NAMES),
        help=f"enable a fix (repeatable): {', '.join(Config.FIX_NAMES)}",
    )
    ap.add_argument(
        "--frames", default=None, metavar="DIR",
        help="BENCH: read frames from an image directory instead of the camera",
    )
    ap.add_argument(
        "--csv", default=None, metavar="PREFIX",
        help=(
            "write structured per-frame/per-contour CSV logs to "
            "PREFIX_frames.csv and PREFIX_contours.csv (extension-less "
            "prefix). When omitted, per-frame detail is only ever printed "
            "to the terminal/text log as before"
        ),
    )
    _session_group = ap.add_mutually_exclusive_group()
    _session_group.add_argument(
        "--session", action="store_true",
        help=(
            "run the segmented single-fix hardware session instead of a "
            "single fixed-config run: baseline + each fix alone, one "
            "segment at a time, pausing on the start button between "
            "segments so the robot can be repositioned"
        ),
    )
    _session_group.add_argument(
        "--combo-session", action="store_true",
        help=(
            "run the segmented fix-COMBINATION hardware session instead "
            "of a single fixed-config run: baseline + every combination "
            "of --fix (or all Config.FIX_NAMES if --fix is omitted) at "
            "each --combo-size, one segment at a time, pausing on the "
            "start button between segments"
        ),
    )
    ap.add_argument(
        "--combo-size", action="append", type=int, default=[], metavar="N",
        help=(
            "with --combo-session: combination size to test (repeatable, "
            "e.g. --combo-size 2 --combo-size 3 for pairs and triples "
            "only). Defaults to every size from 2 up to the number of "
            "fixes being combined (singles are already covered by "
            "--session)"
        ),
    )
    ap.add_argument(
        "--segment-seconds", type=float, default=10.0, metavar="SECONDS",
        help="wall-clock duration of each --session/--combo-session segment (default: 10.0)",
    )
    ap.add_argument(
        "--max-segments", type=int, default=20, metavar="N",
        help=(
            "with --combo-session: safety cap on total segment count "
            "(including baseline), since each segment is a manual "
            "reposition-and-button-press on hardware. Pass a higher "
            "value explicitly if that many segments is intended "
            "(default: 20)"
        ),
    )
    ap.add_argument(
        "--list-segments", action="store_true",
        help=(
            "with --session/--combo-session: print the planned segment "
            "order and letters, then exit without touching hardware or "
            "waiting on the start button — use this to sanity-check a "
            "combo sweep's segment count before committing to it"
        ),
    )
    return ap.parse_args()

# =============================================================================
# Main loop
# =============================================================================
def main(
        fix_cfg: Config = Config(), 
        frames_dir: str | None = None,
        csv_prefix: str | None = None,
    ) -> None:
    # ==========================================================================
    # Initial Startup
    # ==========================================================================
    log.info("Starting Navilott Pipeline")

    # Fix flag states
    flags_str = fix_cfg.flags_str()
    log.info("fix flags: [%s] (I=roi_inset T=trapezoid O=orientat D=dilate A=anchor)", 
            flags_str)
    
    if BENCH_MODE:
        log.info("BENCH_MODE active: hardware peripherals are stubbed")

    # =========================================================================
    # CSV Logging
    # run_logger is a no-op (log_frame/log_contours/close all become
    # harmless) when csv_prefix is None, so no branching is needed at the
    # per-frame call sites below
    # =========================================================================
    run_logger = RunLogger(csv_prefix, fix_cfg)
    if csv_prefix is not None:
        log.info("CSV logging enabled -> %s_frames.csv / %s_contours.csv",
                 csv_prefix, csv_prefix)

    # Button & countdown
    s = System()
    s.wait_for_start()
    s.run_countdown()

    # Capture run-start wall time — everything below is timed from here
    t_run_start = time.perf_counter()

    # ==========================================================================
    # Startup: IMU
    # ==========================================================================
    imu = IMUReader(address=0x68, rate_hz=100.0)
    imu.start()

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
    cap = None
    bench_iter = None

    if BENCH_MODE and frames_dir is not None:
        # TEST ENV ONLY: override camera with a directory of images
        bench_iter = _bench_frame_iter(frames_dir)
    else:
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
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_writer = cv2.VideoWriter("output.avi", fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        log.info("Debug video writer opened -> output.avi")

    # ==========================================================================
    # Main loop
    # ==========================================================================
    frame_id = 0
    ramp_start_ts: float | None = None   # None -> ramp restarts on next drive
    try:
        while True:
            t_frame_start = time.perf_counter()

            # =================================================================
            # Phase 1: Capture
            # =================================================================
            if bench_iter is not None:
                # TEST ENV ONLY: read from bench image directory
                ret, frame_bgr = next(bench_iter, (False, None))
                if not ret:
                    log.info("[BENCH] frame source exhausted after %d frames", frame_id)
                    break
            else:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    log.warning("Frame %d: read failed — skipping", frame_id)
                    frame_id += 1
                    continue

            timestamp_ms = int(time.time() * 1000)

            # Per-frame failure mode instrumentation
            tags = FrameTags()

            # =================================================================
            # Phase 2: Vision Perception
            # =================================================================
            # Step 1: Preprocessing (histogram equalization + Gaussian blur)
            # preprocess_frame() expects YUV in, returns YUV out.
            # capture.py / GStreamer hands us BGR from videoconvert, so we
            # convert before and after to satisfy the stage contract.
            frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
            preprocessed = preprocess_frame(frame_yuv)               # -> YUV
            preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_YUV2BGR)

            # Step 2: ROI crop returns NumPy views (no copy)
            # RESOLUTION 1: roi_inset applies when enabled
            roi_result = crop_rois(preprocessed_bgr, frame_id=frame_id, fix_cfg=fix_cfg)

            # Step 3a: Color branch traffic light candidates
            # roi_crop produces BGR views; color_branch expects BGR
            tl_candidates, _tl_debug = extract_traffic_light_candidates(
                roi = roi_result.traffic_roi.copy(),   # copy: branch modifies internally
                hsv_ranges = hsv_ranges,
                blob_filter = blob_filter,
                frame_id = frame_id,
                timestamp_ms = timestamp_ms,
            )

            # Step 3b: Geometry branch lane + stop sign candidates
            # geometry.py _to_grayscale() internally handles BGR -> gray
            geo_result, _lane_debug, _sign_debug = run_geometry_branch(
                lane_roi = roi_result.lane_roi.copy(),
                sign_roi = roi_result.sign_roi.copy(),
                canny_params = canny_params,
                lane_filter = lane_filter,
                sign_filter = sign_filter,
                frame_id = frame_id,
                timestamp_ms = timestamp_ms,
                fix_cfg = fix_cfg,
                tags = tags,
            )

            # Step 4: Feature fusion normalize and resolve conflicts
            source_rois = SourceROIInfo(
                lane_shape = roi_result.lane_roi.shape[:2],
                traffic_shape = roi_result.traffic_roi.shape[:2],
                sign_shape = roi_result.sign_roi.shape[:2],
            )
            detections, _fusion_summary = fuse_detections(
                traffic_candidates = tl_candidates,
                lane_candidates = geo_result.lane_candidates,
                sign_candidates = geo_result.sign_candidates,
                frame_id = frame_id,
                timestamp_ms = timestamp_ms,
                source_rois = source_rois,
            )

            # Step 5: Lane offset estimation
            lane_boundary_dets = [d for d in detections if d.type == "lane_boundary"]
            lane_offset_result = compute_lane_offset(
                detections = lane_boundary_dets,
                frame_width = roi_result.lane_roi.shape[1],
                frame_id = frame_id,
                timestamp = timestamp_ms,
                conf_threshold = LANE_CONF_THRESHOLD,
                min_lane_width_px = MIN_LANE_WIDTH_PX,
                fix_cfg = fix_cfg,
                tags = tags,
            )

            # =================================================================
            # Intersection edge-ratio detection
            # Reads the edge image the contour extractor consumed. Two
            # count_nonzero calls — negligible against the frame budget
            # =================================================================
            edge_ratio = intersection_edge_ratio(_lane_debug["edges_processed"])
            intersection_trigger = edge_ratio > INTERSECTION_EDGE_RATIO_THRESH
            if intersection_trigger and csv_prefix is None:
                # Only echoed to the terminal when CSV logging is off;
                # otherwise this is just the intersection_trigger column
                log.info(
                    "[%s] f=%04d [INTERSECTION] edge_ratio=%.2f > %.2f — would trigger",
                    flags_str, frame_id, edge_ratio, INTERSECTION_EDGE_RATIO_THRESH,
                )
                # TEST ENV ONLY: 
                # TURN_EXECUTING is IMU-driven and the IMU is stubbed on bench 
                # TODO: Toggle the PipelineMode during turns; include a state machine?
                pass

            # Step 6: Package Phase 2 output
            p2_out = package_phase2_output(
                detections = detections,
                frame_id = frame_id,
                timestamp_ms = timestamp_ms,
            )

            # =================================================================
            # Phase 3: Navigation Signal Processing
            # =================================================================
            # Read sensors
            sensor_sample, imu_frame = _read_sensors(imu)

            # Adapt Phase 2 detections to Phase 3 schema and run processor
            p3_detections = _adapt_detections_for_p3(p2_out.detections, roi_result.lane_roi.shape[1])
            p3_input = _build_p3_input(p2_out, p3_detections)
            nav_packet = p3_processor.process(p3_input, sensor_sample)

            # =================================================================
            # Motor Control
            # =================================================================
            global _last_error

            if nav_packet.drive_state == "stop":
                _drive(0.0, 0.0)
                ramp_start_ts = None
            else:
                if ramp_start_ts is None:
                    ramp_start_ts = t_frame_start
                ramp_frac  = min(1.0, (t_frame_start - ramp_start_ts) / RAMP_SECONDS)
                ramped_base = BASE_SPEED * ramp_frac

                error = nav_packet.lane_offset + OFFSET_TRIM
                derivative = error - _last_error
                correction = (error * KP) + (derivative * KD)
                _drive(ramped_base - correction, ramped_base + correction)
                _last_error = error

            # =================================================================
            # Timing check
            # =================================================================
            t_frame_end = time.perf_counter()
            frame_time_ms = (t_frame_end - t_frame_start) * 1000.0

            s.update_display(t_frame_end - t_run_start)

            if frame_time_ms > LOOP_BUDGET_MS:
                log.warning(
                    "Frame %d: budget exceeded %.1f ms (budget %.1f ms)",
                    frame_id, frame_time_ms, LOOP_BUDGET_MS,
                )

            # =================================================================
            # Navigation Packet Log
            # =================================================================
            # INSTRUMENTATION: Failure mode tagging
            # summary() returns "" for clean frames.
            budget_exceeded = frame_time_ms > LOOP_BUDGET_MS
            tag_str = tags.summary(lane_offset_result.mode)

            if csv_prefix is not None:
                # Structured path: one frames.csv row + one contours.csv
                # row per accepted/rejected contour (no-op if DEBUG_CONTOURS
                # left tags.contour_debug empty)
                run_logger.log_frame(
                    frame_id = frame_id,
                    timestamp_ms = timestamp_ms,
                    frame_time_ms = frame_time_ms,
                    budget_exceeded = budget_exceeded,
                    offset = nav_packet.lane_offset,
                    raw_offset = lane_offset_result.offset,
                    heading_error = nav_packet.heading_error,
                    drive_state = nav_packet.drive_state,
                    stop_sign_detected = nav_packet.stop_sign_detected,
                    lane_mode = lane_offset_result.mode,
                    lane_confidence = lane_offset_result.confidence,
                    edge_ratio = edge_ratio,
                    intersection_trigger = intersection_trigger,
                    tags = tags,
                    imu_sample_count = imu_frame.sample_count,
                    imu_yaw_rate_dps = imu_frame.mean_yaw_rate_dps if imu_frame.valid else 0.0,
                )
                if DEBUG_CONTOURS:
                    run_logger.log_contours(frame_id, timestamp_ms, tags.contour_debug)
            else:
                # Legacy path: same info as unstructured terminal/text-log lines
                if DEBUG_CONTOURS:
                    for cd in tags.contour_debug:
                        log.info(f"[{flags_str}] f={frame_id:04d} CONTOUR "
                                f"accepted={cd.accepted} area={cd.area:.1f} aspect={cd.aspect:.2f} "
                                f"intensity={cd.intensity:.1f} roi_span={cd.roi_span:.2f} "
                                f"reject_reason={cd.reject_reason}")
                if tag_str:
                    log.info(
                        "[%s] f=%04d TAGS=%s offset=%+.4f mode=%s conf=%.2f",
                        flags_str, frame_id, tag_str,
                        lane_offset_result.offset, lane_offset_result.mode,
                        lane_offset_result.confidence,
                    )
                log.info(
                    "f=%04d t=%.1fms offset=%+.4f head=%+.2f° drive=%-7s "
                    "stop_sign=%s lane_mode=%s imu_n=%d yaw=%+.1f°/s",
                    frame_id,
                    frame_time_ms,
                    nav_packet.lane_offset,
                    nav_packet.heading_error,
                    nav_packet.drive_state,
                    "T" if nav_packet.stop_sign_detected else "F",
                    lane_offset_result.mode,
                    imu_frame.sample_count,
                    imu_frame.mean_yaw_rate_dps if imu_frame.valid else 0.0,
                )

            # =================================================================
            # Optional Debug Video
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
        imu.stop()
        if cap is not None:
            cap.release()
        if out_writer is not None:
            out_writer.release()
        run_logger.close()

        elapsed_s = time.perf_counter() - t_run_start
        s.show_final_time(elapsed_s)   # freeze final time on display
        time.sleep(5.0)
        s.cleanup()
        log.info("Pipeline shutdown complete.")

# =============================================================================
# Debug Overlay Helper (SAVE VIDEO)
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
        "go": (0, 200, 0),
        "caution": (0, 180, 220),
        "stop": (0, 0, 220),
    }.get(nav_packet.drive_state, (200, 200, 200))

    hud_text = (
        f"off={nav_packet.lane_offset:+.3f}"
        f"head={nav_packet.heading_error:+.1f}deg"
        f"[{nav_packet.drive_state.upper()}]"
        f"{frame_time_ms:.1f}ms mode={lane_result.mode}"
    )
    cv2.putText(vis, hud_text, (6, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1, cv2.LINE_AA)

    # Offset bar at bottom of frame
    bar_y = H - 10
    bar_mid = W // 2
    bar_len = int(nav_packet.lane_offset * bar_mid)
    cv2.line(vis, (bar_mid, bar_y - 4), (bar_mid, bar_y + 4), (180, 180, 180), 1)
    cv2.line(vis, (bar_mid, bar_y), (bar_mid + bar_len, bar_y), state_color, 3)

    return vis

# =============================================================================
# Combo-Letter Tag (matches FilenameLegend.md's convention)
# =============================================================================
_FIX_LETTERS = {
    "roi_inset": "I",
    "trapezoid_mask": "T",
    "orientation_filt": "O",
    "dashed_dilate": "D",
    "anchor_halves": "A",
}

def _combo_letters(
        fix_cfg: Config
    ) -> str:
    """
    Purpose:
        Concatenate the letters for whichever fixes are active, in
        Config.FIX_NAMES order, matching the <LETTERS> directory
        convention combine_runs.py/plot_runs.py already expect.
        "NONE" when zero fixes are active.
    """
    active = "".join(_FIX_LETTERS[n] for n in Config.FIX_NAMES if getattr(fix_cfg, n))
    return active or "NONE"

# =============================================================================
# Segmented Hardware Session — shared driver
# =============================================================================
def _build_combo_segments(
        fix_names: list[str],
        combo_sizes: list[int],
    ) -> list[tuple[str, "Config"]]:
    """
    Purpose:
        Expand fix_names into every combination at each requested size,
        as ("name1+name2+...", Config) pairs. Combinations at a given
        size are generated in Config.FIX_NAMES order (itertools.combinations
        preserves input order), which is also the order _combo_letters()
        re-derives its letters in, so segment labels and <LETTERS>
        directory names stay consistent with each other and with
        run_single_fix_session's single-fix segments.

    Inputs:
        fix_names: pool of fix names to draw combinations from
        combo_sizes: which combination sizes to generate, e.g. [2, 3]
            for pairs and triples. A size >= len(fix_names) collapses to
            the single all-fixes-active combination.

    Outputs:
        list of (label, Config) tuples, one per combination, NOT including
        the "none" baseline — callers that want a baseline segment prepend
        it themselves, same as run_single_fix_session does.
    """
    out: list[tuple[str, Config]] = []
    for k in combo_sizes:
        if k < 1 or k > len(fix_names):
            raise ValueError(
                f"_build_combo_segments: combo size {k} invalid for "
                f"{len(fix_names)} fix name(s) {fix_names}"
            )
        for combo in itertools.combinations(fix_names, k):
            out.append(("+".join(combo), Config.from_names(list(combo))))
    return out


def _describe_segments(
        segments: list[tuple[str, "Config"]],
        segment_seconds: float,
    ) -> None:
    """
    Purpose:
        Log the planned segment order/letters/duration without touching
        hardware or the start button — a dry-run preview so a combo sweep's
        segment count (which grows fast) can be sanity-checked before
        committing to a physical session.
    """
    total_s = len(segments) * segment_seconds
    log.info(
        "Segment plan: %d segment(s), %.1fs each, ~%.1f min total "
        "(button presses only — no camera/motor activity in preview)",
        len(segments), segment_seconds, total_s / 60.0,
    )
    for i, (seg_label, fix_cfg) in enumerate(segments):
        log.info("  [%02d] %-24s letters=%s", i, seg_label, _combo_letters(fix_cfg))


def _run_fix_sweep_session(
        segments: list[tuple[str, "Config"]],
        segment_seconds: float = 10.0,
        frames_dir: str | None = None,
        log_dir: str = "vision_stack/logs/HWLiveLogs",
        sweep_label: str = "fix",
    ) -> None:
    """
    Purpose:
        Step through a pre-built list of (label, Config) segments, one
        segment at a time, pausing on the start button between segments
        so the robot can be repositioned by hand. Each segment gets its
        own fresh Phase3Processor (no EMA/dead-reckoning state carried
        over from the previous segment's config) and its own RunLogger, so
        the resulting CSVs drop into <log_dir>/<LETTERS>/ exactly like an
        offline sweep run and combine_runs.py/plot_runs.py can ingest
        them unmodified. This is the shared driver behind both
        run_single_fix_session (segments = baseline + each fix alone) and
        run_fix_combo_session (segments = baseline + fix combinations) —
        neither builds its own copy of the per-frame loop.

    Inputs:
        segments: ordered (label, Config) pairs to run, in session order.
            Callers own the "none" baseline decision — both wrapper
            functions below prepend one.
        segment_seconds: wall-clock duration of each segment
        frames_dir: BENCH-only — read segments from an image directory
            instead of the camera, to dry-run the session logic before
            hardware. Same semantics as main()'s frames_dir.
        log_dir: base directory for per-segment CSVs (and, if SAVE_VIDEO,
            per-segment debug videos)
        sweep_label: only used in the startup log line ("single-fix",
            "combo", etc.) to distinguish which sweep is running
    """
    global _last_error

    session_timestamp = time.strftime("%Y%m%d-%H%M")

    log.info("Starting Navilott %s hardware session: %d segments, %.1fs each",
             sweep_label, len(segments), segment_seconds)
    if BENCH_MODE:
        log.info("BENCH_MODE active: hardware peripherals are stubbed")

    # ==========================================================================
    # One-time setup — shared across every segment
    # ==========================================================================
    s = System()
    t_run_start = time.perf_counter()

    imu = IMUReader(address=0x68, rate_hz=100.0)
    imu.start()

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

    cap = None
    bench_iter = None
    if BENCH_MODE and frames_dir is not None:
        bench_iter = _bench_frame_iter(frames_dir)
    else:
        gst_pipeline = _build_gst_pipeline(FRAME_WIDTH, FRAME_HEIGHT, FPS, COLOR_SPACE)
        log.info("Opening camera: %s", gst_pipeline)
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            log.error("Failed to open camera pipeline; is libcamera available?")
            sys.exit(1)

    try:
        for seg_label, fix_cfg in segments:
            letters = _combo_letters(fix_cfg)
            flags_str = fix_cfg.flags_str()
            log.info(
                "=== Segment '%s' [%s] — press start to run %.1fs ===",
                seg_label, flags_str, segment_seconds,
            )

            # Button-gated pause: reposition the robot, then press start
            # to begin exactly this segment
            s.wait_for_start()
            s.run_countdown()

            # Fresh per-segment state: EMA/dead-reckoning history, the PD
            # derivative term, and the soft-start ramp must not leak from
            # one fix's run into the next's
            p3_processor = Phase3Processor(p3_config)
            _last_error = 0.0
            ramp_start_ts: float | None = None   # None -> ramp restarts on next drive

            seg_dir = f"{log_dir}/{letters}"
            os.makedirs(seg_dir, exist_ok=True)
            csv_prefix = f"{seg_dir}/live__{session_timestamp}"
            run_logger = RunLogger(csv_prefix, fix_cfg)
            log.info("CSV logging -> %s_frames.csv / %s_contours.csv", csv_prefix, csv_prefix)

            out_writer = None
            if SAVE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_path = f"{seg_dir}/live__{session_timestamp}.avi"
                out_writer = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                log.info("Debug video writer opened -> %s", video_path)

            frame_id = 0
            seg_start = time.perf_counter()
            try:
                while (time.perf_counter() - seg_start) < segment_seconds:
                    t_frame_start = time.perf_counter()

                    # -----------------------------------------------------------
                    # Phase 1: Capture
                    # -----------------------------------------------------------
                    if bench_iter is not None:
                        ret, frame_bgr = next(bench_iter, (False, None))
                        if not ret:
                            log.info("[BENCH] frame source exhausted after %d frames", frame_id)
                            break
                    else:
                        ret, frame_bgr = cap.read()
                        if not ret or frame_bgr is None:
                            log.warning("Frame %d: read failed — skipping", frame_id)
                            frame_id += 1
                            continue

                    timestamp_ms = int(time.time() * 1000)
                    tags = FrameTags()

                    # -----------------------------------------------------------
                    # Phase 2: Vision Perception
                    # -----------------------------------------------------------
                    frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
                    preprocessed = preprocess_frame(frame_yuv)
                    preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_YUV2BGR)

                    roi_result = crop_rois(preprocessed_bgr, frame_id=frame_id, fix_cfg=fix_cfg)

                    tl_candidates, _tl_debug = extract_traffic_light_candidates(
                        roi = roi_result.traffic_roi.copy(),
                        hsv_ranges = hsv_ranges,
                        blob_filter = blob_filter,
                        frame_id = frame_id,
                        timestamp_ms = timestamp_ms,
                    )

                    geo_result, _lane_debug, _sign_debug = run_geometry_branch(
                        lane_roi = roi_result.lane_roi.copy(),
                        sign_roi = roi_result.sign_roi.copy(),
                        canny_params = canny_params,
                        lane_filter = lane_filter,
                        sign_filter = sign_filter,
                        frame_id = frame_id,
                        timestamp_ms = timestamp_ms,
                        fix_cfg = fix_cfg,
                        tags = tags,
                    )

                    source_rois = SourceROIInfo(
                        lane_shape = roi_result.lane_roi.shape[:2],
                        traffic_shape = roi_result.traffic_roi.shape[:2],
                        sign_shape = roi_result.sign_roi.shape[:2],
                    )
                    detections, _fusion_summary = fuse_detections(
                        traffic_candidates = tl_candidates,
                        lane_candidates = geo_result.lane_candidates,
                        sign_candidates = geo_result.sign_candidates,
                        frame_id = frame_id,
                        timestamp_ms = timestamp_ms,
                        source_rois = source_rois,
                    )

                    lane_boundary_dets = [d for d in detections if d.type == "lane_boundary"]
                    lane_offset_result = compute_lane_offset(
                        detections = lane_boundary_dets,
                        frame_width = roi_result.lane_roi.shape[1],
                        frame_id = frame_id,
                        timestamp = timestamp_ms,
                        conf_threshold = LANE_CONF_THRESHOLD,
                        min_lane_width_px = MIN_LANE_WIDTH_PX,
                        fix_cfg = fix_cfg,
                        tags = tags,
                    )

                    edge_ratio = intersection_edge_ratio(_lane_debug["edges_processed"])
                    intersection_trigger = edge_ratio > INTERSECTION_EDGE_RATIO_THRESH

                    p2_out = package_phase2_output(
                        detections = detections,
                        frame_id = frame_id,
                        timestamp_ms = timestamp_ms,
                    )

                    # -----------------------------------------------------------
                    # Phase 3: Navigation Signal Processing
                    # -----------------------------------------------------------
                    sensor_sample, imu_frame = _read_sensors(imu)
                    p3_detections = _adapt_detections_for_p3(p2_out.detections, roi_result.lane_roi.shape[1])
                    p3_input = _build_p3_input(p2_out, p3_detections)
                    nav_packet = p3_processor.process(p3_input, sensor_sample)

                    # -----------------------------------------------------------
                    # Motor Control
                    # -----------------------------------------------------------
                    if nav_packet.drive_state == "stop":
                        _drive(0.0, 0.0)
                        ramp_start_ts = None
                    else:
                        if ramp_start_ts is None:
                            ramp_start_ts = t_frame_start
                        ramp_frac  = min(1.0, (t_frame_start - ramp_start_ts) / RAMP_SECONDS)
                        ramped_base = BASE_SPEED * ramp_frac

                        error = nav_packet.lane_offset + OFFSET_TRIM
                        derivative = error - _last_error
                        correction = (error * KP) + (derivative * KD)
                        _drive(ramped_base - correction, ramped_base + correction)
                        _last_error = error

                    # -----------------------------------------------------------
                    # Timing + per-frame CSV log
                    # -----------------------------------------------------------
                    t_frame_end = time.perf_counter()
                    frame_time_ms = (t_frame_end - t_frame_start) * 1000.0
                    s.update_display(t_frame_end - t_run_start)

                    if frame_time_ms > LOOP_BUDGET_MS:
                        log.warning(
                            "[%s] Frame %d: budget exceeded %.1f ms (budget %.1f ms)",
                            letters, frame_id, frame_time_ms, LOOP_BUDGET_MS,
                        )

                    budget_exceeded = frame_time_ms > LOOP_BUDGET_MS
                    run_logger.log_frame(
                        frame_id = frame_id,
                        timestamp_ms = timestamp_ms,
                        frame_time_ms = frame_time_ms,
                        budget_exceeded = budget_exceeded,
                        offset = nav_packet.lane_offset,
                        raw_offset = lane_offset_result.offset,
                        heading_error = nav_packet.heading_error,
                        drive_state = nav_packet.drive_state,
                        stop_sign_detected = nav_packet.stop_sign_detected,
                        lane_mode = lane_offset_result.mode,
                        lane_confidence = lane_offset_result.confidence,
                        edge_ratio = edge_ratio,
                        intersection_trigger = intersection_trigger,
                        tags = tags,
                        imu_sample_count = imu_frame.sample_count,
                        imu_yaw_rate_dps = imu_frame.mean_yaw_rate_dps if imu_frame.valid else 0.0,
                    )
                    if DEBUG_CONTOURS:
                        run_logger.log_contours(frame_id, timestamp_ms, tags.contour_debug)

                    if out_writer is not None:
                        _overlay = _draw_debug_overlay(
                            frame_bgr, nav_packet, lane_offset_result, frame_time_ms
                        )
                        out_writer.write(_overlay)

                    frame_id += 1

            finally:
                _drive(0.0, 0.0)
                if out_writer is not None:
                    out_writer.release()
                run_logger.close()
                log.info("Segment '%s' [%s] done: %d frames -> %s_frames.csv",
                          seg_label, letters, frame_id, csv_prefix)

    except KeyboardInterrupt:
        log.info("Session stopped by user.")

    finally:
        _drive(0.0, 0.0)
        pi.write(_stby, 0)
        pi.stop()
        imu.stop()
        if cap is not None:
            cap.release()

        elapsed_s = time.perf_counter() - t_run_start
        s.show_final_time(elapsed_s)
        time.sleep(5.0)
        s.cleanup()
        log.info("Session complete.")


# =============================================================================
# Segmented Single-Fix Hardware Session
# =============================================================================
def run_single_fix_session(
        fix_names: list[str] | None = None,
        segment_seconds: float = 10.0,
        frames_dir: str | None = None,
        log_dir: str = "vision_stack/logs/HWLiveLogs",
    ) -> None:
    """
    Purpose:
        Step through baseline + each fix ALONE (never combined). Thin
        wrapper around _run_fix_sweep_session — see that function for the
        actual per-segment driver.

    Inputs:
        fix_names: which fixes to test in isolation, in this order,
            after an automatic "none" baseline segment. Defaults to
            Config.FIX_NAMES (all five, in their canonical order) — pass
            a subset to test fewer.
        segment_seconds: wall-clock duration of each segment
        frames_dir: BENCH-only — read segments from an image directory
            instead of the camera, to dry-run the session logic before
            hardware. Same semantics as main()'s frames_dir.
        log_dir: base directory for per-segment CSVs (and, if SAVE_VIDEO,
            per-segment debug videos)
    """
    segments: list[tuple[str, Config]] = [("none", Config())] + [
        (name, Config.from_names([name])) for name in (fix_names or Config.FIX_NAMES)
    ]
    _run_fix_sweep_session(
        segments = segments,
        segment_seconds = segment_seconds,
        frames_dir = frames_dir,
        log_dir = log_dir,
        sweep_label = "single-fix",
    )


# =============================================================================
# Segmented Fix-Combination Hardware Session
# =============================================================================
def run_fix_combo_session(
        fix_names: list[str] | None = None,
        combo_sizes: list[int] | None = None,
        segment_seconds: float = 10.0,
        frames_dir: str | None = None,
        log_dir: str = "vision_stack/logs/HWLiveLogs",
        max_segments: int | None = 20,
    ) -> None:
    """
    Purpose:
        Step through baseline + every combination of fixes at the
        requested size(s), the same way run_single_fix_session steps
        through fixes alone. Reuses _run_fix_sweep_session for the actual
        per-segment loop, so segment behavior (fresh Phase3Processor,
        fresh RunLogger, <log_dir>/<LETTERS>/ layout) is identical between
        the two — only how the segment list is built differs.

    Inputs:
        fix_names: pool of fixes to combine. Defaults to Config.FIX_NAMES
            (all five, in their canonical order) — pass a subset to
            combine fewer.
        combo_sizes: which combination sizes to test, e.g. [2] for pairs
            only, or [2, 3] for pairs and triples. Defaults to every size
            from 2 up to len(fix_names) — i.e. the full powerset above
            singles, since singles are already covered by
            run_single_fix_session. Combination count grows fast
            (C(5,2..5) = 26 for all five default fixes), so this is
            usually worth narrowing on a real hardware session.
        segment_seconds: wall-clock duration of each segment
        frames_dir: BENCH-only — read segments from an image directory
            instead of the camera, to dry-run the session logic before
            hardware. Same semantics as main()'s frames_dir.
        log_dir: base directory for per-segment CSVs (and, if SAVE_VIDEO,
            per-segment debug videos)
        max_segments: safety cap on total segment count (including the
            "none" baseline). Each segment needs a physical
            reposition-and-button-press on hardware, so an unnarrowed
            combo_sizes on all five fixes can silently ask for far more
            manual segments than intended. Raises ValueError instead of
            running if the built segment list exceeds this. Pass None to
            disable the check.
    """
    names = fix_names or list(Config.FIX_NAMES)
    sizes = combo_sizes or list(range(2, len(names) + 1))

    segments: list[tuple[str, Config]] = [("none", Config())] + _build_combo_segments(names, sizes)

    if max_segments is not None and len(segments) > max_segments:
        raise ValueError(
            f"run_fix_combo_session: {len(segments)} segments requested "
            f"(fix_names={names}, combo_sizes={sizes}) exceeds max_segments="
            f"{max_segments}. Narrow combo_sizes/fix_names, or pass a "
            f"higher max_segments explicitly if this many segments is "
            f"intended."
        )

    _run_fix_sweep_session(
        segments = segments,
        segment_seconds = segment_seconds,
        frames_dir = frames_dir,
        log_dir = log_dir,
        sweep_label = "combo",
    )


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    _args = _parse_args()
    if _args.session:
        if _args.list_segments:
            _segments = [("none", Config())] + [
                (name, Config.from_names([name]))
                for name in (_args.fix or Config.FIX_NAMES)
            ]
            _describe_segments(_segments, _args.segment_seconds)
        else:
            run_single_fix_session(
                fix_names = _args.fix or None,
                segment_seconds = _args.segment_seconds,
                frames_dir = _args.frames,
            )
    elif _args.combo_session:
        _combo_names = _args.fix or list(Config.FIX_NAMES)
        _combo_sizes = _args.combo_size or list(range(2, len(_combo_names) + 1))
        if _args.list_segments:
            _segments = [("none", Config())] + _build_combo_segments(_combo_names, _combo_sizes)
            _describe_segments(_segments, _args.segment_seconds)
        else:
            run_fix_combo_session(
                fix_names = _combo_names,
                combo_sizes = _combo_sizes,
                segment_seconds = _args.segment_seconds,
                frames_dir = _args.frames,
                max_segments = _args.max_segments,
            )
    else:
        main(
            fix_cfg = Config.from_names(_args.fix),
            frames_dir = _args.frames,
            csv_prefix = _args.csv,
        )