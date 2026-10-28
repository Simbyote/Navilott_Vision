"""Constants that more than one module has to agree on.

Hardware facts, file locations and the names that cross stage boundaries as
strings, so a change happens once and a typo fails at import. Tuning stays
in each stage's config, bundled in config.py.
"""
import math
from pathlib import Path

# Pinned, so the field of view doesn't change with output size; a lens calibration holds only for this mode
SENSOR_CONFIG = "sensor/config,width=1920,height=1080,depth=10"
CAMERA_ROTATE_180 = False        # if camera is mounted upside down
FRAME_W, FRAME_H = 480, 270     # 16:9, as the sensor
FPS = 20
# libcamerasrc controls, e.g. {"awb-mode": "daylight", "exposure-value": 1.0}; empty is the
# camera's auto. make camera-controls lists them; make sweep-camera tries them. They move the lanes too
CAMERA_CONTROLS: dict = {}
# Outside this band CameraSource warns: too sparse to steer, or too fast for the Pi
MIN_FPS = 5
MAX_FPS = 30
# OpenCV's TBB workers spin while idle. Threads 4 / 2 / 1 on the lane (2026-10-03):
# median frame 32.7 / 33.7 / 42.7 ms, CPU 189 / 165 / 153% of a core
OPENCV_THREADS = 2
# Color and obstacle beside geometry (perception/parallel.py). Off until measured
# on the Pi; --parallel / --serial override it for one run
PARALLEL_PERCEPTION = False

# A collection holds the GIL and stops every thread: one took 122 ms in a figure
# eight. Off during a run, the loop collects the youngest generation itself
GC_DISABLE_IN_RUN = True
GC_COLLECT_EVERY = 200          # frames, ~10 s at 20 FPS

# From this file, so the working directory never matters
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PIPELINE_ROOT / "calibration"
CAMERA_CALIB_PATH = CALIBRATION_DIR / "camera_calibration.json"
HSV_RANGES_PATH = CALIBRATION_DIR / "hsv_ranges.json"
GROUND_HOMOGRAPHY_PATH = CALIBRATION_DIR / "ground_homography.json"
STOP_LINE_TABLE_PATH = CALIBRATION_DIR / "stop_line_table.json"
ODOMETRY_PATH = CALIBRATION_DIR / "odometry.json"
RUNS_DIR = PIPELINE_ROOT / "runs"

# One list, so no two modules claim the same pin
GPIO_DISPLAY_CLK = 5        # TM1637 clock, header pin 29
GPIO_DISPLAY_DIO = 6        # TM1637 data, header pin 31
GPIO_START_BUTTON = 17      # active-high, pull-down, header pin 11

IMU_I2C_ADDRESS = 0x68      # MPU-6050 with AD0 low

ADS1115_I2C_ADDRESS = 0x48  # ADDR to GND; A0 reads the 4:1 divider
# A battery read is two transfers; single shots were 11-30 over ~9 ms on the
# bus the IMU shares (make i2c-trace, 2026-10-20)
ADS1115_CONTINUOUS = True
IMU_RATE_HZ = 100.0         # the on-chip filter is 44 Hz to match
# The IMU samples on its own clock and the hub reads its FIFO once per frame;
# a stall under the FIFO's 1.28 s loses nothing. False: the hub polls
IMU_FIFO = True
# Maps raw gyro Z to + = turning right. This IMU is upside down, so raw already
# reads + for a right turn (2026-10-01; the earlier -1 was measured with the
# motor sides swapped). Measure again for a new mount
IMU_YAW_SIGN = 1

SENSOR_RATE_HZ = IMU_RATE_HZ
# Only fills if the frame loop stalls; then the oldest readings drop
SENSOR_HISTORY_S = 2.0
# SCHED_FIFO for the hub's 10 ms tick: a normal thread woke up to 5.4 ms late,
# 0.22 ms at this (make sched-latency, 2026-10-20). Under the IRQ threads' 50.
# Needs root or an rtprio limit; None never asks
SENSOR_HUB_RT_PRIORITY = 20

# 10 missed frames: clear of any frame a run has taken, short enough to stop
# within a few cm. The camera once stalled mid-run with the wheels driving (2026-10-03)
MOTOR_WATCHDOG_S = 0.5
# A copy inside pigpiod brakes even when Python stalls: a 1,558 ms frame on
# 2026-10-20 kept the turn's command on the wheels
MOTOR_DAEMON_WATCHDOG = True
# The TB6612's PWM, on the Pi's hardware PWM. At 1 kHz (until 2026-10-28) each
# period outlasted the motor's current rise many times over, so the current came
# in pulses: more heat for the same speed, and a whine. Small brushed motors are
# usually driven at 10-25 kHz; the TB6612 takes up to 100 kHz. Pulled after the
# right N20 died on 2026-10-28; the duty-to-speed gain and the stall duty were
# measured at 1 kHz, so measure them again
MOTOR_PWM_HZ = 20000

# The nominal scale until calibrate_odometry runs: a 7-pulse encoder behind a
# 200:1 gearbox. Measured beats it: tyres squash
WHEEL_DIAMETER_CM = 3.3
ENCODER_COUNTS_PER_REV = 1400
NOMINAL_CM_PER_COUNT = math.pi * WHEEL_DIAMETER_CM / ENCODER_COUNTS_PER_REV     # ~0.0074

ROI_LANE, ROI_TRAFFIC, ROI_SIGN = "lane", "traffic", "sign"
ROI_OBSTACLE = "obstacle"

LANE_BOUNDARY = "lane_boundary"
TRAFFIC_LIGHT = "traffic_light"
STOP_SIGN = "stop_sign"
STOP_LINE = "stop_line"          # StopLineCandidate.label; not a DetectionObject type

RED, YELLOW, GREEN = "red", "yellow", "green"

MODE_TWO_BOUNDARY = "two_boundary"
MODE_LEFT_ONLY = "left_only"
MODE_RIGHT_ONLY = "right_only"
MODE_SINGLE_UNCALIBRATED = "single_uncalibrated"
MODE_NONE = "none"

# Geometry and lane_offset must compute foot_x over the same band
FOOT_BAND_PX = 6

# One list, so no two stages write the same file name
UNDISTORT_SUFFIX = "_0_undistorted.png"
GRAY_SUFFIX = "_1_gray.png"
EQUALIZED_SUFFIX = "_2_equalized.png"
GRAY_BLUR_SUFFIX = "_3_blurred_gray.png"
COLOR_BLUR_SUFFIX = "_5_blurred.png"
ROI_OVERLAY_SUFFIX = "_roi_overlay.png"
LANE_ROI_SUFFIX = "_roi_lane.png"
TRAFFIC_ROI_SUFFIX = "_roi_traffic.png"
SIGN_ROI_SUFFIX = "_roi_sign.png"
FUSION_OVERLAY_SUFFIX = "_overlay.png"
FUSION_SUMMARY_SUFFIX = "_summary.txt"
