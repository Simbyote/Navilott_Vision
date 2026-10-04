"""Constants that more than one module has to agree on.

Purpose:
    Hardware facts, file locations, and the names that cross stage boundaries
    as plain strings. A camera change or a rename happens here once instead of
    in every file that repeated it, and a typo in a shared name fails as an
    import error instead of a string that silently never matches.

    Only shared values live here. Tuning stays in each stage's config
    dataclass (PipelineConfig bundles them), constants used by one module
    stay in that module next to their reasoning, and anything derivable
    (ROI rects, lane ROI width) is computed from these, never stored.

Main package:
    None; module-level constants only.
"""
from pathlib import Path

# --- Camera ---
# IMX290 native 1920x1080, 10-bit. Pinned so the field of view doesn't change
# with output size. A lens calibration is only valid for this mode, the output
# size, the flip and the lens focus it was captured with.
SENSOR_CONFIG = "sensor/config,width=1920,height=1080,depth=10"
CAMERA_ROTATE_180 = True        # camera is mounted upside down
FRAME_W, FRAME_H = 480, 270     # 16:9, matching the sensor mode
FPS = 20
# libcamerasrc controls set when the camera opens (capture/camera.py), as
# {name: value}: e.g. {"ae-constraint-mode": "highlight", "exposure-value":
# -1.0, "awb-mode": "daylight"}. Empty: the camera's own auto exposure and
# white balance. The linkers' --camera-control KEY=VALUE adds to these for
# one run, to try values before setting them here. Under a dim room the
# auto exposure blew the traffic lamps out to white (2026-10-04,
# calibrate_lamps: lamp saturation 3 of 255); setting exposure changes how
# the lanes look too, so re-check lane detection with any change
CAMERA_CONTROLS: dict = {}
# Supported frame-rate band for the Pi Zero 2 W. Below MIN_FPS, per-frame
# control updates are too sparse for lane following; above MAX_FPS the
# quad-core A53 can't keep up with capture plus processing.
MIN_FPS = 5
MAX_FPS = 30
# OpenCV's worker threads (src/perception/__init__.py sets it). This Pi's
# OpenCV runs its parallel operations on TBB, one thread per core by
# default, the extra three spinning while they wait for work. 60 s runs on
# the lane, 2026-10-03, threads 4 / 2 / 1: median frame 32.7 / 33.7 /
# 42.7 ms (preprocess 11.2 / 12.4 / 22.0), process CPU 189 / 165 / 153%
# of a core, the workers' preemptions ~3,800 / 6 / 0 a second. Two keep
# the parallel speedup and give back a quarter of a core
OPENCV_THREADS = 2

# --- Paths ---
# Resolved from this file (<root>/src/params.py) so the working directory never matters
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PIPELINE_ROOT / "calibration"     # camera, HSV and IMU calibrations
CAMERA_CALIB_PATH = CALIBRATION_DIR / "camera_calibration.json"
HSV_RANGES_PATH = CALIBRATION_DIR / "hsv_ranges.json"
GROUND_HOMOGRAPHY_PATH = CALIBRATION_DIR / "ground_homography.json"   # frame px -> floor cm; scripts/calibrate_ground.py
STOP_LINE_TABLE_PATH = CALIBRATION_DIR / "stop_line_table.json"       # stop-line rows -> cm; scripts/calibrate_stop_line.py
RUNS_DIR = PIPELINE_ROOT / "runs"                   # live_view output, one timestamped folder per run

# --- GPIO (BCM numbers; Product Spec GPIO Table 7) ---
# One list, so no two modules can claim the same pin. Add the motor driver's here too.
GPIO_DISPLAY_CLK = 5        # TM1637 clock, header pin 29
GPIO_DISPLAY_DIO = 6        # TM1637 data, header pin 31
GPIO_START_BUTTON = 17      # active-high, pull-down, header pin 11

# --- IMU ---
IMU_I2C_ADDRESS = 0x68      # MPU-6050 with AD0 low
IMU_RATE_HZ = 100.0         # background sampling; the on-chip filter is set to 44 Hz to match
# Multiplies raw gyro Z so yaw reads + = turning right, Estimation's convention.
# The driver reads + = left for a Z-up IMU; this robot's is mounted upside
# down, so its raw gyro Z already reads + for a RIGHT turn (checked on the
# robot 2026-10-01, after the motor sides were fixed: the 2026-09-30 -1 was
# measured with them swapped). Per robot: measure a new mount.
IMU_YAW_SIGN = 1

# --- Sensor collection (src/peripherals/sensing.py) ---
# The hub reads the IMU and both encoders together at this rate. It samples
# the IMU, so it runs at the rate the on-chip filter is set for
SENSOR_RATE_HZ = IMU_RATE_HZ
# Readings kept between drains. Once per frame (~50 ms) empties it; 2 s only
# fills if the frame loop stalls, and then the oldest readings are dropped
SENSOR_HISTORY_S = 2.0

# --- Motor watchdog (src/peripherals/drive.py) ---
# The run loops command the motors once per frame (~50 ms at FPS; p95 54 ms
# over a 60 s run, 2026-10-03). If no command arrives for this long, the loop
# is stuck (the camera stopped delivering frames mid-run that day, and
# cap.read() blocked with the last command still driving), so the driver
# brakes the motors on its own. 0.5 s = 10 missed frames: clear of any
# frame the loop has taken in a run, short enough to stop within a few cm.
MOTOR_WATCHDOG_S = 0.5

# --- ROI names (DetectionObject.source_roi) ---
ROI_LANE, ROI_TRAFFIC, ROI_SIGN = "lane", "traffic", "sign"

# --- Detection types (candidate labels, DetectionObject.type) ---
LANE_BOUNDARY = "lane_boundary"
TRAFFIC_LIGHT = "traffic_light"
STOP_SIGN = "stop_sign"
STOP_LINE = "stop_line"          # StopLineCandidate.label; not a DetectionObject type

# --- Traffic light colors (TrafficLightCandidate.label, DetectionObject.label_detail) ---
RED, YELLOW, GREEN = "red", "yellow", "green"

# --- Lane offset modes (LaneOffsetResult.mode) ---
MODE_TWO_BOUNDARY = "two_boundary"
MODE_LEFT_ONLY = "left_only"
MODE_RIGHT_ONLY = "right_only"
MODE_SINGLE_UNCALIBRATED = "single_uncalibrated"
MODE_NONE = "none"

# Rows above a contour's lowest point averaged into foot_x. Geometry computes
# foot_x with it, and lane_offset recomputes with it when a candidate has none,
# so both must use the same value.
FOOT_BAND_PX = 6

# --- Debug artifact filename suffixes ---
# One list, so no two stages can write the same file name.
UNDISTORT_SUFFIX = "_0_undistorted.png"     # preprocess; the number orders the stages
GRAY_SUFFIX = "_1_gray.png"
EQUALIZED_SUFFIX = "_2_equalized.png"
GRAY_BLUR_SUFFIX = "_3_blurred_gray.png"    # _4 belonged to the removed histogram stage
COLOR_BLUR_SUFFIX = "_5_blurred.png"
ROI_OVERLAY_SUFFIX = "_roi_overlay.png"     # roi_crop
LANE_ROI_SUFFIX = "_roi_lane.png"
TRAFFIC_ROI_SUFFIX = "_roi_traffic.png"
SIGN_ROI_SUFFIX = "_roi_sign.png"
FUSION_OVERLAY_SUFFIX = "_overlay.png"      # feature_fusion
FUSION_SUMMARY_SUFFIX = "_summary.txt"