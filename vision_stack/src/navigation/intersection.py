"""Intersection rule: cross straight, left or right as the route says, then hand back to lane keeping.

Purpose:
    The crossing street's lines pull lane keeping sideways, so from the
    moment the stop line leaves the view this rule drives on a gyro heading,
    counted from the street rather than the robot, until the new lane is back.

Main package:
    IntersectionRule.update(packet) -> a Command while crossing, else None.

Flow:
    1. to_line: straight to the line on the heading hold.
    2. advance (turns only): on into the intersection.
    3. turn: until the gyro reads TURN_TARGET_DEG, or the time limit.
    4. exit: straight on the new heading until the lane is back.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.lane_keeping import KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import Command
from src.params import MODE_LEFT_ONLY, MODE_RIGHT_ONLY, MODE_TWO_BOUNDARY
from src.navigation.route import LEFT, RIGHT, STRAIGHT, RouteProgress
from src.navigation.stop_line import CROSSING, StopLineTracker
from src.navigation.wheel_speed import CPS_PER_DUTY, WheelSpeed

# Two boundaries this many frames (~0.5 s) ends the crossing. 3 ended it as
# soon as the far side's lines showed (2026-10-05)
TWO_BOUNDARY_FRAMES = 10
# ...or at least one for this many (~1 s): past some intersections only one
# line shows, and the crossing never ended (2026-10-01; 40 until 2026-10-26)
ONE_BOUNDARY_FRAMES = 20
BOUNDARY_MODES = (MODE_TWO_BOUNDARY, MODE_LEFT_ONLY, MODE_RIGHT_ONLY)
# ...or this much exit driving (tuned on the mat, 2026-10-19)
MAX_CROSS_MS = 5500
# The same by odometry, from stage_cm where timed exits ended (2026-10-20)
MAX_CROSS_CM: float | None = 49.7
MAX_DT_MS = 500
# Drive into the intersection before turning: the stop lines sit back from
# it and the robot stops early for the hanging light, so turning from the
# line would cut the corner. Straight crossings don't advance
ADVANCE_MS = 2500
# The same by odometry, which doesn't drift with the battery; from stage_cm
# where timed advances ended (2026-10-20)
ADVANCE_CM: float | None = 22.3
# Tuned on the mat at ~11.7 V (2026-10-19; left 0.48 -> 0.50, 2026-10-26). With encoders, WheelSpeed holds
# these duties' wheel speeds at any voltage
LEFT_TURN = Command(0.50, 0.73)
RIGHT_TURN = Command(0.78, 0.25)
# 90 less the coast after the turn stops
TURN_TARGET_DEG = 85.0
# Heading deg per deg of stop-line tilt, measured with detect-range at
# -6..6 deg (2026-10-20; how: navigation_contract.md). Negative and over 1: a
# low camera flattens the line. Check its sign on the robot: wrong, it doubles the error
TILT_TO_HEADING: float | None = -4.3
# The tilt a square approach reads, from the same fit
TILT_ZERO_DEG = 0.19
# Caps a wrong scale; the stop-line tilt gate refuses steeper lines anyway
MAX_APPROACH_DEG = 10.0
# Turn time limits when the gyro never gets there: open-loop times plus half
LEFT_TURN_MAX_MS = 4100
RIGHT_TURN_MAX_MS = 2400
TURNS = {LEFT: (LEFT_TURN, -1, LEFT_TURN_MAX_MS),        # (duties, heading sign, time limit); + heading = right
         RIGHT: (RIGHT_TURN, +1, RIGHT_TURN_MAX_MS)}

STAGE_TO_LINE, STAGE_ADVANCE, STAGE_TURN, STAGE_EXIT = "to_line", "advance", "turn", "exit"
TURN_END_GYRO, TURN_END_TIME = "gyro target", "time limit"
REASON_CROSSING, REASON_TURNING = "crossing", "turning"
SOURCE_HEADING_HOLD, SOURCE_TURN = "heading_hold", "turn"


def turned_deg(packet, dt_ms: float, gyro_bias_dps: float) -> float:
    """Degrees turned this frame net of the bias, + = right; yaw_deg counts a long frame gap in full."""
    if getattr(packet, "yaw_deg", None) is not None:
        return packet.yaw_deg - gyro_bias_dps * (packet.yaw_span_s or 0.0)
    return (packet.yaw_rate - gyro_bias_dps) * dt_ms / 1000.0


def turn_cps(duties: Command) -> tuple[float, float]:
    """The turn's duties as counts/s at the voltage they were tuned at."""
    return duties.left * CPS_PER_DUTY, duties.right * CPS_PER_DUTY


class IntersectionRule:
    """
    gyro_bias_dps is subtracted here because the packet's yaw_rate is raw.
    wheels=None drives the turn's duties as they are. record carries the
    stage, heading, how the turn ended and stage_cm for nav.csv.
    """
    def __init__(self, tracker: StopLineTracker, lane: LaneKeepingNavigator,
                 kp_heading: float = KP_HEADING, gyro_bias_dps: float = 0.0,
                 progress: RouteProgress | None = None, wheels: WheelSpeed | None = None,
                 advance_ms: int = ADVANCE_MS, advance_cm: float | None = ADVANCE_CM,
                 max_cross_cm: float | None = MAX_CROSS_CM,
                 tilt_to_heading: float | None = TILT_TO_HEADING) -> None:
        self.tracker = tracker
        self.wheels = wheels
        self.advance_ms = advance_ms
        self.advance_cm = advance_cm
        self.max_cross_cm = max_cross_cm
        self.tilt_to_heading = tilt_to_heading
        self.progress = progress or RouteProgress()
        self.lane = lane
        self.kp_heading = kp_heading
        self.gyro_bias_dps = gyro_bias_dps
        self.reset()

    def reset(self) -> None:
        self._active = False
        self.stage: str | None = None
        self._maneuver = STRAIGHT
        self._heading = self._approach = 0.0
        self._turn_ms = self._driving_ms = self._advance_ms = 0
        self._turn_end: str | None = None
        self._turn_deg: float | None = None
        self._two_boundary = self._any_boundary = 0
        self._last_ms: int | None = None
        self._stage_from_cm: float | None = None      # odometry where the current stage began
        self._stage_seen: str | None = None
        self.record: dict = {}

    def _stage_cm(self, packet: EstimationPacket) -> float | None:
        if self.stage != self._stage_seen:
            self._stage_seen, self._stage_from_cm = self.stage, packet.distance_cm
        if packet.distance_cm is None:
            return None
        if self._stage_from_cm is None:              # odometry came up mid-stage: measure from here
            self._stage_from_cm = packet.distance_cm
        return packet.distance_cm - self._stage_from_cm

    def approach_deg(self, tilt_deg: float | None) -> float:
        """+ = turned right of square; 0.0 without a scale or a tilt."""
        if self.tilt_to_heading is None or tilt_deg is None:
            return 0.0
        return max(-MAX_APPROACH_DEG, min(MAX_APPROACH_DEG, self.tilt_to_heading * (tilt_deg - TILT_ZERO_DEG)))

    @staticmethod
    def _done(cm: float | None, limit_cm: float | None, ms: int, limit_ms: int) -> bool:
        if limit_cm is not None and cm is not None:
            return cm >= limit_cm
        return ms >= limit_ms

    @property
    def active(self) -> bool:
        return self._active

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """While held (braked at the line), no timer runs and the lane doesn't count toward the exit."""
        now = packet.timestamp_ms
        if not self._active and self.tracker.phase == CROSSING:
            self.reset()
            self._active, self._last_ms, self.stage = True, now, STAGE_TO_LINE
            self._heading = self._approach = self.approach_deg(self.tracker.approach_tilt_deg)
            current = self.progress.current
            self._maneuver = current[2] if current and current[2] in TURNS else STRAIGHT
        if not self._active:
            return None

        dt_ms = min(max(now - self._last_ms, 0), MAX_DT_MS)
        self._last_ms = now
        self._heading += turned_deg(packet, dt_ms, self.gyro_bias_dps)

        if self.stage == STAGE_TO_LINE and self.tracker.reached:
            self.stage = STAGE_ADVANCE if self._maneuver in TURNS else STAGE_EXIT

        if self.stage == STAGE_ADVANCE:
            if not held:
                self._advance_ms += dt_ms
            if self._done(self._stage_cm(packet), self.advance_cm, self._advance_ms, self.advance_ms):
                self.stage = STAGE_TURN
                if self.wheels is not None:
                    self.wheels.start()
        if self.stage == STAGE_TURN:
            duties, sign, max_ms = TURNS[self._maneuver]
            if not held:
                self._turn_ms += dt_ms
            if sign * self._heading >= TURN_TARGET_DEG or self._turn_ms >= max_ms:
                self._turn_end = TURN_END_GYRO if sign * self._heading >= TURN_TARGET_DEG else TURN_END_TIME
                self._turn_deg = round(self._heading, 1)
                self.stage, self._heading = STAGE_EXIT, 0.0     # hold the heading the turn ended on
        if self.stage == STAGE_EXIT and not held:
            self._driving_ms += dt_ms
            self._two_boundary = self._two_boundary + 1 if packet.lane_mode == MODE_TWO_BOUNDARY else 0
            self._any_boundary = self._any_boundary + 1 if packet.lane_mode in BOUNDARY_MODES else 0
            if (self._two_boundary >= TWO_BOUNDARY_FRAMES or self._any_boundary >= ONE_BOUNDARY_FRAMES
                    or self._done(self._stage_cm(packet), self.max_cross_cm, self._driving_ms, MAX_CROSS_MS)):
                self.reset()
                return None

        wheels = None
        if self.stage == STAGE_TURN:
            if self.wheels is None:
                cmd = duties
            else:
                cmd = self.wheels.drive(*turn_cps(duties), packet)
                wheels = dict(self.wheels.record)
            reason, source, steer = REASON_TURNING, SOURCE_TURN, round((cmd.right - cmd.left) / 2.0, 3)
        else:
            # + heading = turned right, so steer left: + steering
            cmd = self.lane.steer(self.kp_heading * self._heading, SOURCE_HEADING_HOLD)
            reason, source, steer = REASON_CROSSING, SOURCE_HEADING_HOLD, self.lane.record.get("steer", 0.0)
        self.record = {"reason": reason, "source": source, "steer": steer,
                       "heading_deg": round(self._heading, 2), "stage": self.stage,
                       "step": self.progress.label(), "maneuver": self._maneuver, "turn_end": self._turn_end,
                       "turn_deg": self._turn_deg, "approach_deg": round(self._approach, 1),
                       "stage_cm": None if (cm := self._stage_cm(packet)) is None else round(cm, 1)}
        if wheels is not None:
            self.record["wheels"] = wheels
        return cmd
