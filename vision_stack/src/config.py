"""Pipeline tuning: every Phase 2 stage's config as one unit, and the robot's measured values.

Purpose:
    The one home for tuning, so the robot and every tool that checks it run
    the same numbers; re-tuning after a camera change is an edit here, not a
    code change. It holds no pipeline code, so importing the tuning never
    pulls in a stage. params.py keeps what isn't tuning: hardware facts,
    paths and shared names.

Main package:
    MEASURED: the robot's Phase 2 tuning (a PipelineConfig: each stage's
    config, the ground homography and the stop-line table), and
    MEASURED_ESTIMATION, its Phase 3 tuning (a Phase3Config).

Flow, at import only:
    1. The HSV ranges load, and fail now if missing, not on the first frame.
    2. The ground homography and the stop-line table load if they match how
       preprocess runs; otherwise a warning and no cm.
    3. The odometry scale loads, else the nominal wheel scale.
"""
import warnings
from dataclasses import dataclass, field

from src.estimation.estimation import Phase3Config
from src.estimation.odometry import load_odometry
from src.maneuver import ManeuverConfig
from src.params import (
    CAMERA_CALIB_PATH, FRAME_H, FRAME_W, GROUND_HOMOGRAPHY_PATH, HSV_RANGES_PATH, PIPELINE_ROOT,
    NOMINAL_CM_PER_COUNT, ODOMETRY_PATH, STOP_LINE_TABLE_PATH,
)
from src.perception.color_branch import BlobFilter, ColorConfig, GlowFilter, load_color_config
from src.perception.geometry import GeometryConfig, SignContourFilter, StopLineFilter
from src.perception.ground import GroundHomography, load_ground_homography
from src.perception.lane_offset import LaneOffsetConfig
from src.perception.obstacle import ObstacleConfig
from src.perception.preprocess import PreprocessParams
from src.perception.roi_crop import ROIConfig
from src.perception.stop_line_distance import StopLineDistanceConfig
from src.perception.stop_line_table import StopLineTable, load_stop_line_table


@dataclass(frozen=True)
class PipelineConfig:
    """Every stage's tuning as one unit."""
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    roi: ROIConfig = field(default_factory=ROIConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    color: ColorConfig = field(default_factory=ColorConfig)                 # off until HSV ranges are given
    lane_offset: LaneOffsetConfig = field(default_factory=LaneOffsetConfig)
    stop_line: StopLineDistanceConfig = field(default_factory=StopLineDistanceConfig)
    ground: GroundHomography | None = None      # frame px -> floor cm; None turns the cm outputs off
    # Stop-line rows -> cm from tape marks; used for the stop line's and the
    # obstacle's cm when ground is None
    stop_line_table: StopLineTable | None = None
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)


# Lane gates revised from the 4827-candidate CSV sweep rather than from course
# dimensions. min_proximity sat above the observed median of 0.17 and killed
# 1752 candidates the geometry branch had already scored at or above 0.30;
# max_width_px at 25 clipped detections whose p90 is 27 and max is 41.7.
#
# Undistortion and the color branch are on because the robot runs with both.
# The lane gates were swept on distorted frames; undistortion moves lane marks
# by up to ~20 px at the ROI edges (k1 = -0.29), so re-sweep them on
# undistorted captures before trusting the px gates. The HSV ranges load
# whether or not they have been tuned under course lighting.
#
# Synthetic frames are drawn already undistorted, so tests that feed them use
# src/tests/scenes.SCENE_CONFIG: this, with undistortion off.
#
# The ground homography was fit on frames from exactly this preprocess, so it
# is loaded against it; any other lens calibration, alpha or size refuses it.
# The stop-line table (tape marks, scripts/calibrate_stop_line.py) is the
# same, and gives the stop line's cm when there's no homography. The
# homography is optional: the stop line's cm are all that reads it. Only
# having neither warns (below), since then the stop line has no cm.
_MEASURED_PREPROCESS = PreprocessParams(calibration_path = str(CAMERA_CALIB_PATH))
# Traffic-light blob gates for the course's light: bare 5 mm LEDs on a teal
# board, seen from the stop line (nav run frames at 11.4 cm, 2026-10-06). A
# lit LED is a white center with a thin colored ring, 10-35 px^2 of color in
# the traffic ROI, so min_area is 4 and confidence saturates at 30 (a lit red
# ring scores over Phase 3's 0.40). The green LED's glow reaches ~700 px^2,
# hence max_area. The white center and the red ring trace as separate blobs,
# so the core gate is off (on, it dropped 80 of 93 red frames); the tight
# traffic ROI (roi_crop.TRAFFIC) keeps out what it was there for. With these
# and calibration/hsv_ranges.json: 201 of 201 frames right (87 green, 21
# yellow, 93 red), and still right with the light 10 px off sideways or 6 px
# up or down. ref_area 45 or the core gate back on lose yellow and red. The
# lowest red and yellow scores were 0.46 and 0.44, close to the 0.40 gate;
# ref_area 20 lifts them to 0.75 but lets 73 frames carry a wrong color over
# 0.40 too (the unlit yellow lens while green is lit), against 4 at 30
_TRAFFIC_LIGHT_BLOB = BlobFilter(min_area = 4.0, max_area = 3000.0, ref_area = 30.0, min_roundness = 0.2,
                                 min_core_px = 0)
# Glow mode (color_branch.GlowFilter): the frame's clipped-white spot with the
# most white pixels, named by its ring's band. The blob gates above passed
# unlit lenses at the stop light: their colored ring is no brighter than the
# lens unlit (lit V 133 / 103 at the 10th percentile against 178 / 174 off,
# red / yellow, calibrate_lamps with off=, 2026-10-06), and only a lit LED
# clips white. On the 2026-10-06 frames at the stop: 196 of 196 labelled
# right, every one at confidence 1.0; the blob gates stay for glow=None
_TRAFFIC_LIGHT_GLOW = GlowFilter()
MEASURED = PipelineConfig(
    preprocess = _MEASURED_PREPROCESS,
    color = load_color_config(str(HSV_RANGES_PATH), blob = _TRAFFIC_LIGHT_BLOB, glow = _TRAFFIC_LIGHT_GLOW),
    ground = load_ground_homography(GROUND_HOMOGRAPHY_PATH, _MEASURED_PREPROCESS, (FRAME_H, FRAME_W)),
    stop_line_table = load_stop_line_table(STOP_LINE_TABLE_PATH, _MEASURED_PREPROCESS, (FRAME_H, FRAME_W)),
    # Stop lines within 5 deg of horizontal (default 20): the near end of a
    # thick diagonal lane line passed for a stop line past an intersection
    # (2026-10-01 run; 15 then); 5 cut the false lines further on the mat
    # (2026-10-05), but still took parts of the intersection itself for stop
    # lines, each counted as another intersection; 2 stopped that (2026-10-07)
    # but missed a real line approached a little off square, so 4 (2026-10-19),
    # then 5 again (2026-10-23). The costs: a real line approached more than
    # 5 deg off square is missed (detect-range --set target=line --set
    # angles=... maps where), and parts of the intersection can pass for a
    # line (nav.csv's step column counts one intersection too many)
    # Ghost signs (seen on course runs, Oct 2026): a leg or parts of a chair in
    # the background passed as a stop sign now and then, and the robot stopped
    # at a plain stop line. Any solid, roundish red blob reaches 8-9 hull
    # vertices, so shape and content are gated too. Synthetic octagons read
    # aspect 1.0 and extent 0.75-0.81 (0.83 for a perfect one); their
    # letters 11-16% white from radius 28 px. The course's own sign, on a
    # run's frames (sign_check, 2026-10-20): aspect 0.55-0.72 and extent
    # 0.63-0.75, narrower than a square-on octagon, so the first guesses
    # (0.6-1.6, 0.68-0.92) cut through it; lowered to 0.5 and 0.6. A leg is
    # still ~0.3 wide for its height, a chair back still boxier than 0.92
    geometry = GeometryConfig(
        stop_line = StopLineFilter(max_tilt_deg = 5.0),
        sign = SignContourFilter(aspect_range = (0.5, 1.6), extent_range = (0.6, 0.92), min_letter_share = 0.03),
    ),
    lane_offset = LaneOffsetConfig(
        conf_threshold = 0.25,
        min_proximity = 0.05,
        max_width_px = 45.0,
        # 130 from the candidate sweep; 90 since one darker stretch of the
        # course dimmed the tape enough that lane_offset refused it (2026-10-05)
        min_intensity = 90.0,
        # Half the lane, for the single-line projection. 228 was hand-set; the
        # 2026-10-12 runs measured the lane at 350 / 357 px (two-boundary
        # median, 515 frames), so half is ~178. At 228 every left_only /
        # right_only frame was off by ~0.23-0.26 against the two-boundary
        # frames around it (one dash seen: a hard swerve, then the jump gate
        # dropped the lane to hold and stale); at 178, ~0.04-0.08
        expected_half_lane_px = 178.0,
        # 60 let pairs 61-65 px apart through as a lane (2026-10-20: one line
        # and something beside it, as the robot neared the right line); one
        # was accepted, flipped the estimate's sign and the robot veered off.
        # The same run's real pairs were 325-400 px; its 250-300 px pairs put
        # the robot twice as far left as the right line alone did. ~0.84 of
        # the measured 357 px lane: a narrower pair falls to one line
        min_lane_width_px = 300.0,
    ),
)

if MEASURED.ground is None and MEASURED.stop_line_table is None:
    warnings.warn("stop line: no ground homography or stop-line table loaded, so its distance has no cm "
                  "this run. Measure the table: python3 -m src.scripts.calibrate_stop_line", stacklevel=2)

# Gyro Z at rest, deg/s, in the sensor hub's frame (+ = turning right, after
# IMU_YAW_SIGN): the one bias every yaw reading is corrected by before it's
# integrated, in Phase 3's heading, the intersection's heading hold and the
# drive trial. -0.90 raw at rest (2026-09-29 phase3_linker run; with
# IMU_YAW_SIGN = 1 the hub's frame is the raw one). Re-measure, robot still:
# maneuver_linker prints it ("gyro bias at rest ... measured"), as does
# test_imu --hardware (bias_dps)
GYRO_BIAS_DPS = -0.90

# Phase 3: estimation.Phase3Config's defaults until course runs tune them,
# with this robot's gyro bias and odometry scale (calibration/odometry.json;
# without it params' NOMINAL_CM_PER_COUNT, from the wheel and the gearbox).
# lane_roi_width_px stays None: Pipeline derives it from MEASURED's lane ROI
# when cm_per_px is set.
MEASURED_ESTIMATION = Phase3Config(gyro_bias_dps=GYRO_BIAS_DPS,
                                   cm_per_count=load_odometry(ODOMETRY_PATH, NOMINAL_CM_PER_COUNT))

# maneuver_linker's drive trial: ManeuverConfig's placeholders until course
# runs set them. leg_counts in particular is a guess until counts per meter
# are measured; maneuver_linker's flags override any field for one run.
MANEUVER = ManeuverConfig(gyro_bias_dps=GYRO_BIAS_DPS)

# The course plan: one maneuver per intersection and how the run finishes
# (src/navigation/route.py). A file, not a constant, so it changes between
# runs without touching code; the run reads and checks it at startup, before
# the start button. Edit vision_stack/route.json, e.g.
#     {"maneuvers": ["left", "straight", "right"], "finish": "edge"}
# An empty list crosses every intersection straight and finishes where the lane ends
ROUTE_PATH = PIPELINE_ROOT / "route.json"
