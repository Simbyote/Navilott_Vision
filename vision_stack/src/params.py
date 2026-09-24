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
# Supported frame-rate band for the Pi Zero 2 W. Below MIN_FPS, per-frame
# control updates are too sparse for lane following; above MAX_FPS the
# quad-core A53 can't keep up with capture plus processing.
MIN_FPS = 5
MAX_FPS = 30

# --- Paths ---
# Resolved from this file (<root>/src/params.py) so the working directory never matters
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PIPELINE_ROOT / "calibration"     # camera, HSV and IMU calibrations
CAMERA_CALIB_PATH = CALIBRATION_DIR / "camera_calib.json"
HSV_RANGES_PATH = CALIBRATION_DIR / "hsv_ranges.json"
RUNS_DIR = PIPELINE_ROOT / "runs"                   # live_view output, one timestamped folder per run

# --- GPIO (BCM numbers; Product Spec GPIO Table 7) ---
# One list, so no two modules can claim the same pin. Add the motor driver's here too.
GPIO_DISPLAY_CLK = 5        # TM1637 clock, header pin 29
GPIO_DISPLAY_DIO = 6        # TM1637 data, header pin 31
GPIO_START_BUTTON = 17      # active-high, pull-down, header pin 11

# --- IMU ---
IMU_I2C_ADDRESS = 0x68      # MPU-6050 with AD0 low
IMU_RATE_HZ = 100.0         # background sampling; the on-chip filter is set to 44 Hz to match

# --- ROI names (DetectionObject.source_roi) ---
ROI_LANE, ROI_TRAFFIC, ROI_SIGN = "lane", "traffic", "sign"

# --- Detection types (candidate labels, DetectionObject.type) ---
LANE_BOUNDARY = "lane_boundary"
TRAFFIC_LIGHT = "traffic_light"
STOP_SIGN = "stop_sign"

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