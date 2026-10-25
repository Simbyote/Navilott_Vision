"""End of course: finish the run at the route's finish, or end it early when the lane is lost.

Purpose:
    A lane that stays lost after the last maneuver is the course's end when
    the route finishes at the edge; with a finish line, the first stop line
    after it is. A lane lost before then ends the run early, as a safety
    stop. The count pauses while a higher rule holds the frame.

Main package:
    EndOfCourseRule.update(packet) -> a creep on a stale lane, BRAKE once
    finished, else None; outcome and end_step say how the run ended.
"""
from src.estimation.estimation import LANE_STALE, EstimationPacket
from src.navigation.lane_keeping import SOURCE_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.route import FINISH_EDGE, RouteProgress
from src.navigation.stop_line import StopLineTracker

# Above the 0.25 stall duty; the maneuver trial's slow band moved the robot (2026-09-30)
SLOW_DUTY = 0.30
# The inner wheel may stop but never reverse
SLOW_MAX_STEERING_ADJ = SLOW_DUTY
# A guess between ending on glare and rolling off the mat; counts after
# Phase 3's ~0.35 s hold
END_STALE_MS = 1000
# Longest packet gap counted as one step, as Phase 3's max_dt_s
MAX_DT_MS = 500

REASON_SLOW, REASON_FINISHED = "lane_stale_slow", "end_of_course"
OUTCOME_FINISHED, OUTCOME_EARLY = "finished", "ended_early"


class EndOfCourseRule:
    """
    progress defaults to RouteProgress() (finish at the edge), which makes any
    lost lane a normal finish. tracker is needed only for a finish line.
    Once finished, it stays finished until reset().
    """
    def __init__(self, lane: LaneKeepingNavigator, progress: RouteProgress | None = None,
                 tracker: StopLineTracker | None = None, end_stale_ms: int = END_STALE_MS) -> None:
        self.lane = lane
        self.progress = progress or RouteProgress()
        self.tracker = tracker
        self.end_stale_ms = end_stale_ms
        self.reset()

    @property
    def finished(self) -> bool:
        return self.outcome is not None

    def _end(self, outcome: str) -> Command:
        self.outcome, self.end_step = outcome, self.progress.step
        self.record = {"reason": REASON_FINISHED, "outcome": outcome}
        return BRAKE

    def reset(self) -> None:
        self.outcome: str | None = None
        self.end_step: int | None = None
        self._stale_ms = 0
        self._last_ms: int | None = None
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """A creep while stale, BRAKE once finished, else None. Held frames don't count toward the end."""
        now = packet.timestamp_ms
        dt_ms = 0 if self._last_ms is None else min(max(now - self._last_ms, 0), MAX_DT_MS)
        self._last_ms = now
        if self.finished:
            self.record = {"reason": REASON_FINISHED, "outcome": self.outcome}
            return BRAKE
        if self.tracker is not None and self.tracker.reached and self.progress.at_finish_line:
            return self._end(OUTCOME_FINISHED)
        if packet.lane_status != LANE_STALE:
            self._stale_ms = 0
            self.record = {}
            return None
        if not held:
            self._stale_ms += dt_ms
        if self._stale_ms >= self.end_stale_ms:
            at_edge = self.progress.done and self.progress.route.finish == FINISH_EDGE
            return self._end(OUTCOME_FINISHED if at_edge else OUTCOME_EARLY)
        cmd = self.lane.steer(packet.heading_error * self.lane.kp_heading, SOURCE_HEADING,
                              base_speed=SLOW_DUTY, max_steering_adj=SLOW_MAX_STEERING_ADJ)
        self.record = {**self.lane.record, "reason": REASON_SLOW, "stale_ms": self._stale_ms}
        return cmd
