"""
contracts.py

Shared Phase 2 <-> Phase 3 Schema

Purpose:
    

    Single definition of every object that crosses the Phase 2 / Phase 3 seam.

    Before this module, estimation.py re-declared its own private mirrors of
    DetectionObject and Phase2Output. No import edge crossed the seam, so no
    import error, type checker, or linter could ever detect the two schemas
    drifting apart. They drifted: Phase 3's docstring promised a pre-reduced
    lane-center offset and Phase 2 handed it one detection per boundary.

    Both phases now import from here. Drift becomes an import-time failure
    instead of a silent runtime bias.

=============================================================================
CANONICAL SIGN CONVENTION 
=============================================================================

    Image x increases to the robot's RIGHT after an 180 deg videoflip

    offset > 0  ->  lane center appears RIGHT of image center
                ->  robot is LEFT of lane center
                ->  robot must steer RIGHT

    offset < 0  ->  robot is RIGHT of lane center, must steer LEFT
    offset = 0  ->  centerd

    This is the ONLY convention in the codebase. Every producer normalises to
    it before emitting. estimation.py's old EstimationPacket docstring said the
    opposite ("Positive = robot is right of center") and was wrong; the code
    always implemented the convention above.

=============================================================================
UNITS
=============================================================================

    LaneEstimate.offset_norm   normalised, [-1.0, +1.0], 1.0 = one lane
                               half-width. This is the physically meaningful
                               quantity; it is what the vision stack measures.

    EstimationPacket.lane_offset
                               metres. Retained at its existing scale so the
                               currently-tuned PD gains stay in range. NOTE:
                               px_per_meter is a SCALING constant, not a metric
                               one -- there is no homography, so it does not
                               survive a change in camera pitch or mount height.
                               Do not treat it as true metres at range.

    EstimationPacket.lane_offset_norm
                               the same signal, normalised. Prefer this for new
                               consumers. Migrating the PD loop to it requires
                               scaling KP/KD by px_per_meter/(FRAME_WIDTH/2)
                               (~0.35x) to preserve loop gain.

    architecture.md line 94 documents lane_offset as metres; pipeline.md line
    190 documents it as normalised. Both fields are now emitted, so both docs
    are satisfiable -- but pipeline.md's units row should be corrected to point
    at lane_offset_norm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# =============================================================================
# Lane detection modes
# =============================================================================
# Kept as plain strings (not an Enum) so existing log lines and the
# lane_offset.py return values need no transcription changes.

LANE_MODE_TWO_BOUNDARY = "two_boundary"
LANE_MODE_LEFT_ONLY = "left_only"
LANE_MODE_RIGHT_ONLY = "right_only"
LANE_MODE_WIDTH_REJECTED = "width_rejected"
LANE_MODE_NONE = "none"

# Modes whose offset is derived from two independent anchors. Single-anchor
# modes are inferred, not measured, and carry a confidence penalty downstream.
LANE_MODES_TWO_ANCHOR = frozenset({LANE_MODE_TWO_BOUNDARY, LANE_MODE_WIDTH_REJECTED})

LANE_MODES_ALL = frozenset({
    LANE_MODE_TWO_BOUNDARY, LANE_MODE_LEFT_ONLY, LANE_MODE_RIGHT_ONLY,
    LANE_MODE_WIDTH_REJECTED, LANE_MODE_NONE,
})


# =============================================================================
# Phase 2 -> Phase 3: pre-reduced lane estimate
# =============================================================================

@dataclass
class LaneEstimate:
    """
    The lane-center estimate for one frame, ALREADY REDUCED to a single value.

    This is the fix for the original contract violation. Phase 3 used to
    receive N lane_boundary detections and average their x-positions, which
    only equals the lane center when the surviving detections are laterally
    symmetric -- they never are, because a dashed center line fragments into
    several contours while a solid edge line yields one.

    Reduction is now done exactly once, in lane_offset.compute_lane_offset(),
    which is the function that was always documented to do it.

    offset_norm: signed normalised offset, [-1.0, +1.0], canonical convention
    mode: one of LANE_MODES_ALL
    confidence: [0.0, 1.0]; mean confidence of the contributing anchors
    valid: False when mode == "none" (no usable anchors this frame)
    boundary_count: how many lane_boundary detections passed the conf gate
    lane_width_px: anchor separation in px; None unless two anchors were used
    frame_id: frame identifier from the capture loop
    timestamp_ms: monotonic timestamp from the capture loop
    """
    offset_norm:    float
    mode:           str
    confidence:     float
    valid:          bool
    boundary_count: int
    lane_width_px:  Optional[float]
    frame_id:       int
    timestamp_ms:   int

    @staticmethod
    def empty(frame_id: int = 0, timestamp_ms: int = 0) -> "LaneEstimate":
        """
        Purpose:
            Construct the no-lane-this-frame estimate.
        """
        return LaneEstimate(
            offset_norm=0.0, mode=LANE_MODE_NONE, confidence=0.0, valid=False,
            boundary_count=0, lane_width_px=None,
            frame_id=frame_id, timestamp_ms=timestamp_ms,
        )


# =============================================================================
# Phase 2 -> Phase 3: discrete detections
# =============================================================================

@dataclass
class NavDetection:
    """
    A single non-lane detection crossing the seam.

    Lane boundaries no longer travel as NavDetections -- they are reduced to a
    LaneEstimate first. They may still be forwarded for diagnostics, but Phase 3
    must not derive lane_offset from them.

    type: "traffic_light" | "stop_sign"
    label: for traffic_light: "red" | "yellow" | "green"
           for stop_sign: "stop_sign"
    position_x: bbox centroid x in SOURCE FRAME pixels (re-projected out of
                ROI-local coordinates by feature_fusion; see R7)
    position_y: bbox centroid y in SOURCE FRAME pixels
    confidence: [0.0, 1.0]
    timestamp_ms: monotonic timestamp of the frame this detection came from
    """
    type:         str
    label:        str
    position_x:   float
    position_y:   float
    confidence:   float
    timestamp_ms: int


@dataclass
class Phase2Snapshot:
    """
    Everything Phase 2 hands Phase 3 for one frame.

    detections: traffic_light and stop_sign detections only
    lane: the pre-reduced lane estimate (never None; use LaneEstimate.empty())
    frame_id: frame identifier from the capture loop
    timestamp_ms: monotonic timestamp from the capture loop
    """
    detections:   List[NavDetection]
    lane:         LaneEstimate
    frame_id:     int
    timestamp_ms: int


# =============================================================================
# Sensor input
# =============================================================================

@dataclass
class SensorSample:
    """
    Odometry and IMU readings for one frame window.
    All fields may be None if the sensor is not fitted.

    wheel_speed: m/s
    distance_traveled: m, cumulative since reset
    yaw_rate: deg/s, mean over the frame window
    lateral_accel: m/s^2, peak over the frame window
    """
    wheel_speed:       Optional[float] = None
    distance_traveled: Optional[float] = None
    yaw_rate:          Optional[float] = None
    lateral_accel:     Optional[float] = None


# =============================================================================
# Phase 3 -> Navigation
# =============================================================================

@dataclass
class EstimationPacket:
    """
    Validated navigation signals produced by Phase 3.
    This is the handoff contract to the Navigation subsystem.

    lane_offset: lateral offset from lane center, metres (see UNITS above)
                 POSITIVE = robot is LEFT of lane center, must steer RIGHT
    lane_offset_norm: the same signal normalised to [-1.0, +1.0]

    lane_offset_valid: False when this frame's value is dead-reckoned or stale.
                 Navigation MUST check this before acting on lane_offset.
                 Its absence was the reason the post-intersection failure was
                 silent: a 39-frame-old estimate was indistinguishable from a
                 fresh one.
    lane_offset_age: frames elapsed since the last confident vision fix.
                 0 = measured this frame.
    lane_offset_stale: True once age exceeds deadreck_max_frames. The value is
                 still populated for continuity but carries no information.
    lane_mode: the LaneEstimate mode that produced this value, or "none"

    heading_error: angular deviation from target heading, degrees
    heading_source: "vision" | "imu" | "hold" -- which estimator produced
                 heading_error this frame. heading_error and lane_offset are
                 NOT independent when this is "vision"; do not fuse them as
                 two measurements.

    drive_state: "go" | "caution" | "stop"
    stop_sign_detected: stop sign takes precedence over traffic light state
    yaw_rate: pass-through from SensorSample (0.0 if None)
    lateral_accel: pass-through from SensorSample (0.0 if None)
    wheel_speed: pass-through from SensorSample (0.0 if None)
    frame_id: frame identifier from the capture loop
    timestamp_ms: monotonic timestamp from the capture loop
    """
    lane_offset:        float
    lane_offset_norm:   float
    lane_offset_valid:  bool
    lane_offset_age:    int
    lane_offset_stale:  bool
    lane_mode:          str

    heading_error:      float
    heading_source:     str

    drive_state:        str
    stop_sign_detected: bool

    yaw_rate:           float
    lateral_accel:      float
    wheel_speed:        float

    frame_id:           int
    timestamp_ms:       int