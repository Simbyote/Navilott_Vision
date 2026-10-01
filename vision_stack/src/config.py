"""Pipeline tuning: every Phase 2 stage's config as one unit, and the robot's measured values.

Purpose:
    The one home for tuning. The main pipeline, both linkers, the debuggers
    and the tests import it from here, so the robot and every tool that
    checks it run the same numbers. Re-tuning after a camera change is an
    edit here and a re-run, not a code change. It holds no pipeline code, so
    importing the tuning never pulls in a stage or a debugger.

    params.py keeps what isn't tuning: hardware facts, paths and shared
    names. Each stage keeps its own config dataclass next to the code that
    reads it; PipelineConfig only bundles them.

Main package:
    PipelineConfig: preprocess, ROI, geometry, color, lane-offset and
        stop-line distance tuning, and the ground homography (frame px ->
        floor cm) the cm outputs project through.
        Its defaults are each stage's own defaults: no undistortion, color
        branch off.
    MEASURED: the robot's Phase 2 tuning. Undistorts with the lens
        calibration, runs the color branch with the calibrated HSV ranges,
        and uses the lane gates from the candidate sweep.
    MEASURED_ESTIMATION: the robot's Phase 3 tuning (Phase3Config), kept
        apart from PipelineConfig because run_chain and the Phase 2 parity
        tests take Phase 2 tuning only.

Flow:
    Import-time only. MEASURED reads calibration/hsv_ranges.json once, when
    this module is first imported, and fails then if the file is missing or
    malformed rather than on the first frame. It also loads
    calibration/ground_homography.json once, checked against MEASURED's own
    preprocess settings; missing or mismatched, it warns and leaves ground
    None, which only turns the cm outputs off. The lens calibration is stored
    as a path and read by preprocess on first use.
"""
from dataclasses import dataclass, field

from src.estimation.estimation import Phase3Config
from src.maneuver import ManeuverConfig
from src.params import (
    CAMERA_CALIB_PATH, FRAME_H, FRAME_W, GROUND_HOMOGRAPHY_PATH, HSV_RANGES_PATH, PIPELINE_ROOT,
    STOP_LINE_TABLE_PATH,
)
from src.perception.color_branch import ColorConfig, load_color_config
from src.perception.geometry import GeometryConfig, StopLineFilter
from src.perception.ground import GroundHomography, load_ground_homography
from src.perception.lane_offset import LaneOffsetConfig
from src.perception.preprocess import PreprocessParams
from src.perception.roi_crop import ROIConfig
from src.perception.stop_line_distance import StopLineDistanceConfig
from src.perception.stop_line_table import StopLineTable, load_stop_line_table


# =============================================================================
# Config bundle
# =============================================================================

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
    # Stop-line rows -> cm from tape marks; used for the stop line's cm when ground is None
    stop_line_table: StopLineTable | None = None


# =============================================================================
# The robot's tuning
# =============================================================================

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
# same, and gives the stop line's cm when there's no homography.
_MEASURED_PREPROCESS = PreprocessParams(calibration_path = str(CAMERA_CALIB_PATH))
MEASURED = PipelineConfig(
    preprocess = _MEASURED_PREPROCESS,
    color = load_color_config(str(HSV_RANGES_PATH)),
    ground = load_ground_homography(GROUND_HOMOGRAPHY_PATH, _MEASURED_PREPROCESS, (FRAME_H, FRAME_W)),
    stop_line_table = load_stop_line_table(STOP_LINE_TABLE_PATH, _MEASURED_PREPROCESS, (FRAME_H, FRAME_W)),
    # Stop lines within 15 deg of horizontal (default 20): the near end of a
    # thick diagonal lane line passed for a stop line past an intersection
    # (2026-10-01 run) and restarted the crossing
    geometry = GeometryConfig(stop_line = StopLineFilter(max_tilt_deg = 15.0)),
    lane_offset = LaneOffsetConfig(
        conf_threshold = 0.25,
        min_proximity = 0.05,
        max_width_px = 45.0,
        min_intensity = 130.0,
    ),
)

# Phase 3: estimation.Phase3Config's defaults until course runs tune them.
# lane_roi_width_px stays None: Pipeline derives it from MEASURED's lane ROI
# when cm_per_px is set. gyro_bias_dps stays 0 until a bench measurement
# (or phase3_linker's --gyro-bias) sets it.
MEASURED_ESTIMATION = Phase3Config()

# maneuver_linker's drive trial: ManeuverConfig's placeholders until course
# runs set them. leg_counts in particular is a guess until counts per meter
# are measured; maneuver_linker's flags override any field for one run.
MANEUVER = ManeuverConfig()

# The course plan: one maneuver per intersection and how the run finishes
# (src/navigation/route.py). A file, not a constant, so it changes between
# runs without touching code; the run reads and checks it at startup, before
# the start button. Edit vision_stack/route.json, e.g.
#     {"maneuvers": ["left", "straight", "right"], "finish": "edge"}
# An empty list crosses every intersection straight and finishes where the lane ends
ROUTE_PATH = PIPELINE_ROOT / "route.json"
