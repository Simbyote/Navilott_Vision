"""Navigation: one Command per frame, from the rules in priority order, else lane keeping.

Purpose:
    Each decision lives in its own rule; this is the one place that sets the
    order they're asked in. Every rule sees every frame, so its state stays
    current even when a higher rule decided.

Main package:
    Navigation.update(packet) -> Command, with record saying which rule decided.

Flow, every frame:
    1. The stop-line tracker updates.
    2. Obstacle, stop sign, traffic light, intersection, end of course: the first Command wins.
    3. Lane keeping when none spoke; BRAKE on every frame once finished.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.end_of_course import EndOfCourseRule
from src.navigation.intersection import IntersectionRule
from src.navigation.lane_keeping import LaneKeepingNavigator
from src.navigation.navigation_contract import BRAKE, STALL_DUTY, Command, Navigator, command_problems, enforce
from src.navigation.obstacle import ObstacleRule
from src.navigation.route import Route, RouteProgress
from src.navigation.stop_line import StopLineTracker
from src.navigation.stop_sign import StopSignRule
from src.navigation.traffic_light import TrafficLightRule
from src.navigation.wheel_speed import WheelSpeed

# Re-exported: the rules import the contract directly, everyone else from here
__all__ = ["BRAKE", "STALL_DUTY", "Command", "Navigator", "command_problems", "enforce", "Navigation",
           "RULE_OBSTACLE", "RULE_STOP_SIGN", "RULE_TRAFFIC_LIGHT", "RULE_INTERSECTION", "RULE_END_OF_COURSE",
           "RULE_LANE_KEEPING"]

RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, RULE_INTERSECTION, RULE_END_OF_COURSE, RULE_LANE_KEEPING = (
    "stop_sign", "traffic_light", "intersection", "end_of_course", "lane_keeping")
RULE_OBSTACLE = "obstacle"


class Navigation:
    """
    gyro_bias_dps is for the intersection's heading hold, since the packet's
    yaw_rate is raw. route defaults to Route(): every intersection straight,
    the lost lane the finish. wheels learns the duty-to-speed gain each frame
    so the turns' arcs hold as the battery drains.
    """
    def __init__(self, lane: LaneKeepingNavigator | None = None, tracker: StopLineTracker | None = None,
                 gyro_bias_dps: float = 0.0, route: Route | None = None,
                 obstacle: ObstacleRule | None = None) -> None:
        self.lane = lane or LaneKeepingNavigator()
        self.tracker = tracker or StopLineTracker()
        self.progress = RouteProgress(route)
        self.wheels = WheelSpeed()
        self._last_cmd: Command | None = None
        self.rules = [
            (RULE_OBSTACLE, obstacle or ObstacleRule()),
            (RULE_STOP_SIGN, StopSignRule(self.tracker)),
            (RULE_TRAFFIC_LIGHT, TrafficLightRule(self.tracker)),
            (RULE_INTERSECTION, IntersectionRule(self.tracker, self.lane, gyro_bias_dps=gyro_bias_dps,
                                                 progress=self.progress, wheels=self.wheels)),
            (RULE_END_OF_COURSE, EndOfCourseRule(self.lane, self.progress, self.tracker)),
        ]
        self._crossing = self.rules[3][1]
        self._end = self.rules[-1][1]
        self.record: dict = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.progress.reset()
        self.wheels.reset()
        self._last_cmd = None
        for _, rule in self.rules:
            rule.reset()
        self.lane.reset()
        self.record = {}

    @property
    def finished(self) -> bool:
        return self._end.finished

    @property
    def outcome(self) -> str | None:
        return self._end.outcome

    @property
    def end_step(self) -> int | None:
        return self._end.end_step

    def update(self, packet: EstimationPacket) -> Command:
        # This packet's wheel speeds answer the last command
        self.wheels.observe(self._last_cmd, packet)
        if self.finished:
            self.record = {"rule": RULE_END_OF_COURSE, "phase": self.tracker.phase, "line_left_cm": self.tracker.left_cm, **self._end.record}
            self._last_cmd = BRAKE
            return BRAKE
        # No new stop line while crossing: the far side's lines were counted as intersections
        self.tracker.update(packet, accept_new=not self._crossing.active)
        if self.tracker.entered:
            self.progress.enter()
        decided = None
        for name, rule in self.rules:
            cmd = rule.update(packet, held=decided is not None)
            if cmd is not None and decided is None:
                decided = (name, rule, cmd)
        if decided is None:
            decided = (RULE_LANE_KEEPING, self.lane, self.lane.update(packet))
        if self.finished:                        # the finish line, reached this frame, beats any rule
            decided = (RULE_END_OF_COURSE, self._end, BRAKE)
        name, part, cmd = decided
        self.record = {"rule": name, "phase": self.tracker.phase, "line_left_cm": self.tracker.left_cm, "step": self.progress.label(), **part.record}
        self._last_cmd = cmd
        return cmd
