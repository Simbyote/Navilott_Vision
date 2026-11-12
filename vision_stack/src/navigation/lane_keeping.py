"""Lane keeping: proportional differential steering on the lane.

Purpose:
    The command for every frame no rule speaks for: a base duty, steered
    against the lane offset, or against the heading turned once vision is
    lost. Its steer() is shared by the intersection and end-of-course rules.

Main package:
    LaneKeepingNavigator.update(packet) -> Command.
"""
from src.estimation.estimation import LANE_HOLD, LANE_STALE, LANE_VISION, EstimationPacket
from src.navigation.navigation_contract import BRAKE, STALL_DUTY, Command, command_problems

BASE_SPEED = 0.40               # ~1166 counts/s per wheel
# Steering on cm since 2026-11-12, ported from what the robot drove with until then: no
# cm_per_px was set, so every run steered by the normalized offset, offset x 0.30 x (1 + 0.05
# x |offset| x 30), with 216 px a unit (the 5-95% lane ROI's half) and 0.0355 cm/px, 7.67 cm
# a unit. The same steering in cm is 0.30 / 7.67 a cm, growing 0.05 x 30 / 7.67 a cm. The
# bench demo's own cm gains (0.015, 0.05; 2026-09-30) never drove
KP_CM = 0.039                   # duty per cm of offset
GAIN_SCALE = 0.195              # gain grows by this per cm: kp x (1 + GAIN_SCALE x |cm|)
# cm per unit of normalized offset, for a packet without lane_offset_cm: half the lane ROI
# (240 px since 2026-11-12) x config.CM_PER_PX. test_settings holds it to them. 30 until
# 2026-11-12, never measured: ~4x the real scale, so the gain grew ~4x too fast
NORM_TO_CM = 8.52
KP_HEADING = 0.010              # duty per degree, while vision is lost
MAX_STEERING_ADJ = 0.40
ZERO_DUTY_EPS = 1e-4            # below this a duty is zero, not lifted to the stall duty

REASON_REJECTED = "rejected"            # the contract refused the steered command
REASON_STEER = "steer"
SOURCE_CM, SOURCE_NORM, SOURCE_HEADING, SOURCE_NONE = "offset_cm", "offset", "heading", "none"


class LaneKeepingNavigator:
    """base_speed is clamped to [STALL_DUTY, 1]. record (reason, source, steer) is for the linkers' logs only."""
    def __init__(
        self,
        base_speed: float = BASE_SPEED,
        kp_cm: float = KP_CM,
        gain_scale: float = GAIN_SCALE,
        norm_to_cm: float = NORM_TO_CM,
        kp_heading: float = KP_HEADING,
        max_steering_adj: float = MAX_STEERING_ADJ,
    ) -> None:
        self.base_speed = max(STALL_DUTY, min(1.0, base_speed))
        self.kp_cm = kp_cm
        self.gain_scale = gain_scale
        self.norm_to_cm = norm_to_cm
        self.kp_heading = kp_heading
        self.max_steering_adj = max_steering_adj
        self.record: dict = {}

    def reset(self) -> None:
        self.record = {}

    def _brake(self, reason: str) -> Command:
        self.record = {"reason": reason, "source": SOURCE_NONE, "steer": 0.0}
        return BRAKE

    def update(self, packet: EstimationPacket) -> Command:
        steering_adj, source = 0.0, SOURCE_NONE
        if packet.lane_status == LANE_VISION:
            source = SOURCE_CM if packet.lane_offset_cm is not None else SOURCE_NORM
            # One formula on cm; a packet without them is converted, so the two can't drift apart
            offset_cm = (packet.lane_offset_cm if packet.lane_offset_cm is not None
                         else packet.lane_offset * self.norm_to_cm)
            steering_adj = offset_cm * self.kp_cm * (1.0 + self.gain_scale * abs(offset_cm))
        elif packet.lane_status in (LANE_HOLD, LANE_STALE):
            # On a stale lane EndOfCourseRule creeps and stops ahead of this rule
            steering_adj = packet.heading_error * self.kp_heading
            source = SOURCE_HEADING
        return self.steer(steering_adj, source)

    def steer(self, steering_adj: float, source: str = SOURCE_NONE, base_speed: float | None = None,
              max_steering_adj: float | None = None) -> Command:
        """
        Forward with steering_adj split across the wheels (+ turns left), or
        BRAKE if the contract rejects it. base_speed and max_steering_adj
        default to the navigator's own.
        """
        base = self.base_speed if base_speed is None else base_speed
        limit = self.max_steering_adj if max_steering_adj is None else max_steering_adj
        steering_adj = max(-limit, min(limit, steering_adj))
        cmd = Command(left=self._sanitize_duty(base - steering_adj),
                      right=self._sanitize_duty(base + steering_adj))
        if command_problems(cmd):
            return self._brake(REASON_REJECTED)
        self.record = {"reason": REASON_STEER, "source": source, "steer": steering_adj}
        return cmd

    def _sanitize_duty(self, duty: float) -> float:
        if abs(duty) < ZERO_DUTY_EPS:
            return 0.0
        clamped = max(-1.0, min(1.0, duty))
        if 0.0 < clamped < STALL_DUTY:
            return STALL_DUTY
        if -STALL_DUTY < clamped < 0.0:
            return -STALL_DUTY
        return clamped
