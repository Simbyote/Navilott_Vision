"""Intersection rule: cross an intersection straight, on the gyro, until both lane boundaries are back.

Purpose:
    Lane keeping goes wrong entering an intersection: a boundary on the
    right that crosses the robot's path pulls the offset to the left
    (2026-10-01 runs). So from the moment the stop line leaves the bottom of
    the view (the shared StopLineTracker's CROSSING), this rule drives
    straight instead: base duty, steering only against the heading turned
    since then (yaw_rate integrated over packet time). Once the robot has
    reached the line, the crossing ends when Phase 2 sees both lane
    boundaries (lane_mode MODE_TWO_BOUNDARY) for TWO_BOUNDARY_FRAMES frames
    in a row, or after MAX_CROSS_MS of driving, and lane keeping takes over.
    Which way to go comes from the route (route.py): the RouteProgress
    navigation.Navigation advances as each intersection is entered. Only
    straight is built: left and right (TURNS_TBD, Ignacio's turn logic) are
    driven straight and recorded as TBD until they are.

Main package:
    IntersectionRule: update(packet, held) -> a straight command while
        crossing, None otherwise.

Flow:
    1. CROSSING starts the crossing; heading zeroed there.
    2. Each frame: integrate yaw; steer against it through
       LaneKeepingNavigator.steer (same base duty, clamp and stall floor).
    3. After reached, on frames no higher-priority rule holds the robot
       (stop sign, red light): count frames with both boundaries, and time
       driving.
    4. End on TWO_BOUNDARY_FRAMES in a row or MAX_CROSS_MS.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.lane_keeping import KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import Command
from src.params import MODE_TWO_BOUNDARY
from src.navigation.route import TURNS_TBD, RouteProgress
from src.navigation.stop_line import CROSSING, StopLineTracker

# Both boundaries this many frames in a row ends the crossing (~0.15 s at
# 20 FPS); one frame could be a stray mark
TWO_BOUNDARY_FRAMES = 3
# Driving time after reaching the line before lane keeping takes over
# anyway; a guess at crossing one intersection, to tune on the mat
MAX_CROSS_MS = 2000
# Longest packet gap integrated as one step, as Phase 3's max_dt_s
MAX_DT_MS = 500
REASON_CROSSING = "crossing"
SOURCE_HEADING_HOLD = "heading_hold"


class IntersectionRule:
    """
    Straight across an intersection, holding the heading it entered with.

    Inputs:
        tracker: The StopLineTracker navigation.Navigation updates each frame.
        lane: The LaneKeepingNavigator whose steer() it drives through.
        kp_heading: Duty per degree turned; lane keeping's own by default.
        gyro_bias_dps: Subtracted from the packet's yaw_rate, which is raw
            (Phase 3 subtracts its bias only inside heading_error).
        progress: The run's RouteProgress, for this intersection's maneuver;
            RouteProgress() (no plan: straight) by default.

    record: {"reason": REASON_CROSSING, "source": SOURCE_HEADING_HOLD,
        "steer", "heading_deg", "step" (progress.label()), "maneuver",
        "tbd" (a turn driven straight)} while crossing; {} otherwise.
    """
    def __init__(self, tracker: StopLineTracker, lane: LaneKeepingNavigator,
                 kp_heading: float = KP_HEADING, gyro_bias_dps: float = 0.0,
                 progress: RouteProgress | None = None) -> None:
        self.tracker = tracker
        self.progress = progress or RouteProgress()
        self.lane = lane
        self.kp_heading = kp_heading
        self.gyro_bias_dps = gyro_bias_dps
        self.reset()

    def reset(self) -> None:
        """Not crossing."""
        self._active = self._at_line = False
        self._heading = 0.0
        self._driving_ms = 0
        self._two_boundary = 0
        self._last_ms: int | None = None
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """
        This frame's say.

        Inputs:
            packet: This frame's packet; the tracker has already seen it.
            held: A higher-priority rule (stop sign, red light) decided this
                frame: the robot is braked at the line, so neither the time
                nor what Phase 2 sees counts towards ending the crossing.
        Outputs:
            A straight command while crossing; None otherwise.
        """
        now = packet.timestamp_ms
        if not self._active and self.tracker.phase == CROSSING:
            self.reset()
            self._active, self._last_ms = True, now
        if not self._active:
            return None

        dt_ms = min(max(now - self._last_ms, 0), MAX_DT_MS)
        self._last_ms = now
        self._heading += (packet.yaw_rate - self.gyro_bias_dps) * dt_ms / 1000.0
        if self.tracker.reached:
            self._at_line = True
        if self._at_line and not held:          # held: braked at the line, not crossing yet
            self._driving_ms += dt_ms
            self._two_boundary = self._two_boundary + 1 if packet.lane_mode == MODE_TWO_BOUNDARY else 0
            if self._two_boundary >= TWO_BOUNDARY_FRAMES or self._driving_ms >= MAX_CROSS_MS:
                self.reset()
                return None

        # + heading = turned right, so steer left: + steering
        cmd = self.lane.steer(self.kp_heading * self._heading, SOURCE_HEADING_HOLD)
        maneuver = self.progress.current[2] if self.progress.current else None
        self.record = {"reason": REASON_CROSSING, "source": SOURCE_HEADING_HOLD,
                       "steer": self.lane.record.get("steer", 0.0), "heading_deg": round(self._heading, 2),
                       "step": self.progress.label(), "maneuver": maneuver,
                       "tbd": maneuver in TURNS_TBD}
        return cmd
