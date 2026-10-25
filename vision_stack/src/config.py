"""Pipeline tuning: every Phase 2 stage's config as one unit, and the robot's measured values.

Purpose:
    The one home for tuning, so the robot and every tool run the same
    numbers; re-tuning after a camera change is an edit here, not a code
    change. params.py keeps what isn't tuning.

Main package:
    MEASURED (Phase 2) and MEASURED_ESTIMATION (Phase 3): the robot's tuning,
    loaded with its calibration files at import.
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
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    roi: ROIConfig = field(default_factory=ROIConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    color: ColorConfig = field(default_factory=ColorConfig)                 # off without HSV ranges
    lane_offset: LaneOffsetConfig = field(default_factory=LaneOffsetConfig)
    stop_line: StopLineDistanceConfig = field(default_factory=StopLineDistanceConfig)
    ground: GroundHomography | None = None
    # The stop line's and the obstacle's cm when ground is None
    stop_line_table: StopLineTable | None = None
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)


# The lane gates come from a 4827-candidate sweep on distorted frames;
# undistortion moves marks up to ~20 px at the ROI edges, so they are due a
# re-sweep. The ground homography and stop-line table load only against this
# exact preprocess. Tests on synthetic frames use scenes.SCENE_CONFIG instead
_MEASURED_PREPROCESS = PreprocessParams(calibration_path = str(CAMERA_CALIB_PATH))
# Blob gates for the course's bare LEDs (2026-10-06): 10-35 px^2 of ring each,
# the green's glow ~700. The core gate is off: the white center and the ring
# trace as separate blobs. 201 of 201 frames right. Glow mode replaces these
_TRAFFIC_LIGHT_BLOB = BlobFilter(min_area = 4.0, max_area = 3000.0, ref_area = 30.0, min_roundness = 0.2,
                                 min_core_px = 0)
# Glow mode: the blob gates passed unlit lenses, whose ring is no dimmer than a
# lit one's, but only a lit LED clips white. 196 of 196 frames right (2026-10-06)
_TRAFFIC_LIGHT_GLOW = GlowFilter()
MEASURED = PipelineConfig(
    preprocess = _MEASURED_PREPROCESS,
    color = load_color_config(str(HSV_RANGES_PATH), blob = _TRAFFIC_LIGHT_BLOB, glow = _TRAFFIC_LIGHT_GLOW),
    ground = load_ground_homography(GROUND_HOMOGRAPHY_PATH, _MEASURED_PREPROCESS, (FRAME_H, FRAME_W)),
    stop_line_table = load_stop_line_table(STOP_LINE_TABLE_PATH, _MEASURED_PREPROCESS, (FRAME_H, FRAME_W)),
    # Stop lines within 5 deg (default 20): wider let a diagonal lane line pass;
    # 2 missed lines approached slightly off square. At 5, parts of the
    # intersection have passed for lines (2026-10-05): watch nav.csv's step column
    # Ghost signs (a leg, a chair) stopped the robot at plain lines. The course's
    # sign reads aspect 0.55-0.72, extent 0.63-0.75 (sign_check, 2026-10-20); a
    # leg is ~0.3 wide, a chair back boxier than 0.92
    geometry = GeometryConfig(
        stop_line = StopLineFilter(max_tilt_deg = 5.0),
        sign = SignContourFilter(aspect_range = (0.5, 1.6), extent_range = (0.6, 0.92), min_letter_share = 0.03),
    ),
    lane_offset = LaneOffsetConfig(
        conf_threshold = 0.25,
        min_proximity = 0.05,
        max_width_px = 45.0,
        # 130 from the sweep; a darker stretch of the course needed 90 (2026-10-05)
        min_intensity = 90.0,
        # Half the 357 px lane measured on 2026-10-12. The hand-set 228 put one-line
        # frames ~0.25 off their neighbours; 178 puts them ~0.06 off
        expected_half_lane_px = 178.0,
        # ~0.84 of the lane. At 60 a line and something beside it passed as a
        # lane, flipped the estimate's sign and the robot veered off (2026-10-20)
        min_lane_width_px = 300.0,
    ),
)

if MEASURED.ground is None and MEASURED.stop_line_table is None:
    warnings.warn("stop line: no ground homography or stop-line table loaded, so its distance has no cm "
                  "this run. Measure the table: python3 -m src.scripts.calibrate_stop_line", stacklevel=2)

# Gyro Z at rest in the hub's frame, subtracted everywhere before integrating
# (2026-09-29 phase3_linker run). maneuver_linker and test_imu --hardware re-measure it
GYRO_BIAS_DPS = -0.90

# Phase3Config's defaults, with this robot's bias and odometry scale (the
# nominal scale without calibration/odometry.json)
MEASURED_ESTIMATION = Phase3Config(gyro_bias_dps=GYRO_BIAS_DPS,
                                   cm_per_count=load_odometry(ODOMETRY_PATH, NOMINAL_CM_PER_COUNT))

# The drive trial's placeholders; leg_counts is a guess. Its flags override any field
MANEUVER = ManeuverConfig(gyro_bias_dps=GYRO_BIAS_DPS)

# The course plan, read and checked at startup, e.g.
#     {"maneuvers": ["left", "straight", "right"], "finish": "edge"}
ROUTE_PATH = PIPELINE_ROOT / "route.json"
