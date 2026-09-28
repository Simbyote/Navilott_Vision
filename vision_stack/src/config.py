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
        stop-line distance tuning.
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
    malformed rather than on the first frame. The lens calibration is stored
    as a path and read by preprocess on first use.
"""
from dataclasses import dataclass, field

from src.estimation import Phase3Config
from src.params import CAMERA_CALIB_PATH, HSV_RANGES_PATH
from src.perception.color_branch import ColorConfig, load_color_config
from src.perception.geometry import GeometryConfig
from src.perception.lane_offset import LaneOffsetConfig
from src.perception.preprocess import PreprocessParams
from src.perception.roi_crop import ROIConfig
from src.perception.stop_line_distance import StopLineDistanceConfig


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
MEASURED = PipelineConfig(
    preprocess = PreprocessParams(calibration_path = str(CAMERA_CALIB_PATH)),
    color = load_color_config(str(HSV_RANGES_PATH)),
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
