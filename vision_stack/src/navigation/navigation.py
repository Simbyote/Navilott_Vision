"""Navigation: the subsystem's orchestrator, one Command per frame from its rules and lane keeping.

Purpose:
    Each navigation decision lives in its own file, as a rule with one job.
    This is the one place that says in which order they're asked: every
    frame the shared stop-line tracker advances, then each rule in priority
    order says either a Command or None ("nothing to say"), and the first
    Command wins. With no rule speaking, lane keeping steers. Navigation is
    the Navigator the pipeline and navigation_linker drive with.

    The contract types (Command, BRAKE, STALL_DUTY, Navigator,
    command_problems) live in navigation_contract.py, so the rules can use
    them without importing this module; they're re-exported here.

Main package:
    Navigation: update(packet) -> Command, reset(), record, finished,
        outcome, end_step; progress (the route).
    RULE_*: which part decided a frame, in record["rule"].

Flow, every frame:
    1. StopLineTracker.update(packet): has the robot reached a stop line?
       A line leaving the view is the next intersection: RouteProgress.enter().
    2. StopSignRule      stop sign at the line: stop, hold, go
    3. TrafficLightRule  red light at the line: wait for it
    4. IntersectionRule  past the line: straight on the gyro until both lane
                         boundaries are back (turns: TBD, Ignacio's)
    5. EndOfCourseRule   lane stale: creep; stale too long: brake, finished
                         (or ended early, before the route is done); the
                         route's finish line: brake, finished
    6. LaneKeepingNavigator, when none of them spoke.
    Once finished, every frame is BRAKE and the run's loop ends on it.
    Every rule sees every frame (held = a higher rule already decided it),
    so each keeps its own state current whether or not it's the one heard.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.end_of_course import EndOfCourseRule
from src.navigation.intersection import IntersectionRule
from src.navigation.lane_keeping import LaneKeepingNavigator
from src.navigation.navigation_contract import BRAKE, STALL_DUTY, Command, Navigator, command_problems, enforce
from src.navigation.route import Route, RouteProgress
from src.navigation.stop_line import StopLineTracker
from src.navigation.stop_sign import StopSignRule
from src.navigation.traffic_light import TrafficLightRule

__all__ = ["BRAKE", "STALL_DUTY", "Command", "Navigator", "command_problems", "enforce", "Navigation",
           "RULE_STOP_SIGN", "RULE_TRAFFIC_LIGHT", "RULE_INTERSECTION", "RULE_END_OF_COURSE",
           "RULE_LANE_KEEPING"]

RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, RULE_INTERSECTION, RULE_END_OF_COURSE, RULE_LANE_KEEPING = (
    "stop_sign", "traffic_light", "intersection", "end_of_course", "lane_keeping")


class Navigation:
    """
    The robot's Navigator: the rules in priority order over lane keeping.

    Inputs:
        lane: The lane keeper; LaneKeepingNavigator() by default.
        tracker: The shared stop-line tracker; StopLineTracker() by default.
        gyro_bias_dps: For the intersection's heading hold (the packet's
            yaw_rate is raw); the run's Phase 3 gyro bias.
        route: The course plan (route.py); Route() (no maneuvers, finish at
            the edge: every intersection straight, a lost lane the finish) by default.

    record: Why the last update() returned what it did:
        {"rule": RULE_*, "phase": the tracker's phase, plus the deciding
        part's own record ("reason", and "source" / "steer" when steering)}.
    """
    def __init__(self, lane: LaneKeepingNavigator | None = None, tracker: StopLineTracker | None = None,
                 gyro_bias_dps: float = 0.0, route: Route | None = None) -> None:
        self.lane = lane or LaneKeepingNavigator()
        self.tracker = tracker or StopLineTracker()
        self.progress = RouteProgress(route)
        self.rules = [
            (RULE_STOP_SIGN, StopSignRule(self.tracker)),
            (RULE_TRAFFIC_LIGHT, TrafficLightRule(self.tracker)),
            (RULE_INTERSECTION, IntersectionRule(self.tracker, self.lane, gyro_bias_dps=gyro_bias_dps,
                                                 progress=self.progress)),
            (RULE_END_OF_COURSE, EndOfCourseRule(self.lane, self.progress, self.tracker)),
        ]
        self._end = self.rules[-1][1]
        self.record: dict = {}

    def reset(self) -> None:
        """Forget everything: tracker, route progress, every rule, lane keeping."""
        self.tracker.reset()
        self.progress.reset()
        for _, rule in self.rules:
            rule.reset()
        self.lane.reset()
        self.record = {}

    @property
    def finished(self) -> bool:
        """The run is over (finished, or ended early); every command is BRAKE from here."""
        return self._end.finished

    @property
    def outcome(self) -> str | None:
        """end_of_course.OUTCOME_FINISHED or OUTCOME_EARLY once finished; None before."""
        return self._end.outcome

    @property
    def end_step(self) -> int | None:
        """The route step the run ended on; None before."""
        return self._end.end_step

    def update(self, packet: EstimationPacket) -> Command:
        """This frame's command: BRAKE once finished, else the first rule that speaks, else lane keeping."""
        if self.finished:
            self.record = {"rule": RULE_END_OF_COURSE, "phase": self.tracker.phase, **self._end.record}
            return BRAKE
        self.tracker.update(packet)
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
        self.record = {"rule": name, "phase": self.tracker.phase, "step": self.progress.label(), **part.record}
        return cmd
