"""Lane keeping: the first Navigator, proportional differential steering on the lane.

Purpose:
    Turns each frame's EstimationPacket into a motor Command under the
    Navigation contract (src/navigation.py): brake for a red light, a stop
    sign or a reached stop line; otherwise drive at a base duty and steer
    against the lane offset (on vision) or the heading turned since vision
    was lost (on hold / stale). Pure logic: it never imports pigpio or the
    motor driver, so it runs and tests anywhere; whoever runs the loop drives
    the Command it returns (scripts/lane_keeping_demo.py on the robot).

Main package:
    LaneKeepingNavigator: the navigator. update(packet) -> Command, reset().
    check_stop_line_trigger(): whether a packet's stop line has been reached.

Flow:
    1. Stop triggers first: drive_state stop, stop sign, stop line reached -> BRAKE.
    2. Steering: lane_offset_cm (or lane_offset without a scale) on vision,
       heading_error on hold / stale, clamped to max_steering_adj.
    3. Left = base - steering, right = base + steering, each kept at or above
       STALL_DUTY; anything the contract rejects becomes BRAKE.
"""
from src.estimation import LANE_HOLD, LANE_STALE, LANE_VISION, EstimationPacket
from src.navigation import BRAKE, STALL_DUTY, Command, command_problems

# First-cut gains and limits (Ignacio, 2026-09-30); not yet tuned on the robot
BASE_SPEED = 0.40               # forward duty: maneuver_linker's leg duty, ~1166 counts/s per wheel
KP_CM = 0.035                   # duty per cm of lane_offset_cm
KP_NORM = 0.60                  # duty per unit of lane_offset, used until cm_per_px is set
KP_HEADING = 0.020              # duty per degree of heading_error while vision is lost
MAX_STEERING_ADJ = 0.40         # largest steering duty either way
STOP_LINE_THRESHOLD_CM = 3.0    # brake once the stop line is this close (floor cm)
# Duties this close to zero are treated as zero, not bumped up to the stall duty
ZERO_DUTY_EPS = 1e-4


def check_stop_line_trigger(packet: EstimationPacket, threshold_cm: float = STOP_LINE_THRESHOLD_CM) -> bool:
    """
    True once the robot has reached or crossed the stop line.

    Inputs:
        packet: Uses stop_line_detected and stop_line_distance_cm; a line
            with no cm distance (no ground homography) never triggers.
        threshold_cm: Distance at or under which it counts as reached.
    """
    if packet.stop_line_detected and packet.stop_line_distance_cm is not None:
        return packet.stop_line_distance_cm <= threshold_cm
    return False


class LaneKeepingNavigator:
    """
    Proportional differential steering navigator for lane keeping.

    Inputs:
        base_speed: Forward duty on both wheels before steering; clamped to
            [STALL_DUTY, 1].
        kp_cm, kp_norm, kp_heading: Steering gains, see the constants above.
        max_steering_adj: Steering clamp.
        stop_line_threshold_cm: See check_stop_line_trigger().
    """
    def __init__(
        self,
        base_speed: float = BASE_SPEED,
        kp_cm: float = KP_CM,
        kp_norm: float = KP_NORM,
        kp_heading: float = KP_HEADING,
        max_steering_adj: float = MAX_STEERING_ADJ,
        stop_line_threshold_cm: float = STOP_LINE_THRESHOLD_CM,
    ) -> None:
        self.base_speed = max(STALL_DUTY, min(1.0, base_speed))
        self.kp_cm = kp_cm
        self.kp_norm = kp_norm
        self.kp_heading = kp_heading
        self.max_steering_adj = max_steering_adj
        self.stop_line_threshold_cm = stop_line_threshold_cm

    def reset(self) -> None:
        """Nothing to forget: every command comes from the current packet alone."""

    def update(self, packet: EstimationPacket) -> Command:
        """
        This frame's command.

        Outputs:
            BRAKE on a stop trigger or a command the contract rejects;
            otherwise forward duty with the steering split across the wheels
            (+ steering = left slower = turn left, against a + offset).
        """
        if packet.drive_state == "stop" or packet.stop_sign_detected:
            return BRAKE
        if check_stop_line_trigger(packet, self.stop_line_threshold_cm):
            return BRAKE

        steering_adj = 0.0
        if packet.lane_status == LANE_VISION:
            if packet.lane_offset_cm is not None:
                steering_adj = packet.lane_offset_cm * self.kp_cm
            else:
                steering_adj = packet.lane_offset * self.kp_norm
        elif packet.lane_status in (LANE_HOLD, LANE_STALE):
            # @TODO (open): the contract says don't drive forward on a stale
            # lane; this keeps driving on heading. Decide which is wanted.
            steering_adj = packet.heading_error * self.kp_heading
        steering_adj = max(-self.max_steering_adj, min(self.max_steering_adj, steering_adj))

        cmd = Command(left=self._sanitize_duty(self.base_speed - steering_adj),
                      right=self._sanitize_duty(self.base_speed + steering_adj))
        if command_problems(cmd):
            return BRAKE
        return cmd

    def _sanitize_duty(self, duty: float) -> float:
        """Clamp to [-1, 1] and lift a nonzero duty to at least STALL_DUTY."""
        if abs(duty) < ZERO_DUTY_EPS:
            return 0.0
        clamped = max(-1.0, min(1.0, duty))
        if 0.0 < clamped < STALL_DUTY:
            return STALL_DUTY
        if -STALL_DUTY < clamped < 0.0:
            return -STALL_DUTY
        return clamped
