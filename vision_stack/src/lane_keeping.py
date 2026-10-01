"""Lane keeping: proportional differential steering on the lane.

Purpose:
    Turns each frame's EstimationPacket into a motor Command under the
    Navigation contract (src/navigation.py): drives at a base duty and steers
    against the lane offset (on vision) or the heading turned since vision
    was lost (on hold / stale). Pure logic: it never imports pigpio or the
    motor driver, so it runs and tests anywhere; whoever runs the loop drives
    the Command it returns (scripts/lane_keeping_demo.py on the robot).

Main package:
    LaneKeepingNavigator: the navigator. update(packet) -> Command, reset().
        Steering gain grows with the offset (variable gain scaling), so
        small offsets get gentle corrections and large ones firm ones.
    REASON_*: why a frame's command is what it is, in LaneKeepingNavigator.record.

Flow:
    1. Steering: lane_offset_cm (or lane_offset without a scale) on vision,
       times a gain that grows with |offset|; heading_error on hold / stale;
       clamped to max_steering_adj.
    2. Left = base - steering, right = base + steering, each kept at or above
       STALL_DUTY; anything the contract rejects becomes BRAKE.
"""
from src.estimation import LANE_HOLD, LANE_STALE, LANE_VISION, EstimationPacket
from src.navigation_contract import BRAKE, STALL_DUTY, Command, command_problems

# Gains and limits (Ignacio, 2026-09-30, commit ae34566: variable gain
# scaling, set on the bench demo); not yet tuned on the course
BASE_SPEED = 0.40               # forward duty: maneuver_linker's leg duty, ~1166 counts/s per wheel
KP_CM = 0.015                   # duty per cm of lane_offset_cm, at zero offset
KP_NORM = 0.30                  # duty per unit of lane_offset at zero offset, used until cm_per_px is set
# The gain grows by this fraction per cm of offset: kp x (1 + GAIN_SCALE x |offset cm|)
GAIN_SCALE = 0.05
# cm per unit of lane_offset, so the normalized path scales its gain like the
# cm path. Carried from ae34566's mock packets; NOT measured. lane_offset is in
# half lane-ROI widths and the lane is ~14 cm wide (docs/course.md), so this is
# likely too large; it stops mattering once cm_per_px is set
NORM_TO_CM = 30.0
KP_HEADING = 0.010              # duty per degree of heading_error while vision is lost
MAX_STEERING_ADJ = 0.40         # largest steering duty either way
# Duties this close to zero are treated as zero, not bumped up to the stall duty
ZERO_DUTY_EPS = 1e-4

# LaneKeepingNavigator.record["reason"]: what decided this frame's command
REASON_REJECTED = "rejected"            # the contract refused the steered command
REASON_STEER = "steer"
# record["source"]: what the steering came from
SOURCE_CM, SOURCE_NORM, SOURCE_HEADING, SOURCE_NONE = "offset_cm", "offset", "heading", "none"


class LaneKeepingNavigator:
    """
    Proportional differential steering navigator for lane keeping.

    Inputs:
        base_speed: Forward duty on both wheels before steering; clamped to
            [STALL_DUTY, 1].
        kp_cm, kp_norm, kp_heading: Steering gains, see the constants above.
        gain_scale: How fast the offset gains grow with |offset|; 0 is plain
            proportional steering.
        max_steering_adj: Steering clamp.

    record: Why the last update() returned what it did, for the linkers'
        logs and video: {"reason": REASON_*, "source": SOURCE_*,
        "steer": clamped steering duty (0.0 when braking)}. Debug output
        only; not part of the Navigator contract.
    """
    def __init__(
        self,
        base_speed: float = BASE_SPEED,
        kp_cm: float = KP_CM,
        gain_scale: float = GAIN_SCALE,
        kp_norm: float = KP_NORM,
        kp_heading: float = KP_HEADING,
        max_steering_adj: float = MAX_STEERING_ADJ,
    ) -> None:
        self.base_speed = max(STALL_DUTY, min(1.0, base_speed))
        self.kp_cm = kp_cm
        self.gain_scale = gain_scale
        self.kp_norm = kp_norm
        self.kp_heading = kp_heading
        self.max_steering_adj = max_steering_adj
        self.record: dict = {}

    def reset(self) -> None:
        """Nothing to forget: every command comes from the current packet alone."""
        self.record = {}

    def _brake(self, reason: str) -> Command:
        self.record = {"reason": reason, "source": SOURCE_NONE, "steer": 0.0}
        return BRAKE

    def update(self, packet: EstimationPacket) -> Command:
        """
        This frame's command.

        Outputs:
            BRAKE only if a command is rejected by the contract;
            otherwise forward duty with the steering split across the wheels
            (+ steering = left slower = turn left, against a + offset).
        """
        steering_adj, source = 0.0, SOURCE_NONE
        if packet.lane_status == LANE_VISION:
            source = SOURCE_CM if packet.lane_offset_cm is not None else SOURCE_NORM
            if packet.lane_offset_cm is not None:
                offset_cm = packet.lane_offset_cm
                steering_adj = offset_cm * self.kp_cm * (1.0 + self.gain_scale * abs(offset_cm))
            else:
                offset = packet.lane_offset
                steering_adj = offset * self.kp_norm * (1.0 + self.gain_scale * abs(offset) * NORM_TO_CM)
        elif packet.lane_status in (LANE_HOLD, LANE_STALE):
            # @TODO (open): the contract says don't drive forward on a stale
            # lane; this keeps driving on heading. Decide which is wanted.
            steering_adj = packet.heading_error * self.kp_heading
            source = SOURCE_HEADING
        return self.steer(steering_adj, source)

    def steer(self, steering_adj: float, source: str = SOURCE_NONE) -> Command:
        """
        Forward at base_speed with this steering split across the wheels.

        Shared with the intersection rule, so both steer with the same base
        duty, clamp and stall floor.

        Inputs:
            steering_adj: + turns left (left wheel slower); clamped to
                max_steering_adj.
            source: What the steering came from, for record.
        Outputs:
            The command; BRAKE if the contract rejects it.
        """
        steering_adj = max(-self.max_steering_adj, min(self.max_steering_adj, steering_adj))
        cmd = Command(left=self._sanitize_duty(self.base_speed - steering_adj),
                      right=self._sanitize_duty(self.base_speed + steering_adj))
        if command_problems(cmd):
            return self._brake(REASON_REJECTED)
        self.record = {"reason": REASON_STEER, "source": source, "steer": steering_adj}
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
