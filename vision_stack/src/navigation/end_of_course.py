"""End of course: finish the run at the route's finish, or end it early on a lane lost off course.

Purpose:
    How a run ends depends on the route (route.py, decided 2026-10-01):
        - finish "edge": after the last maneuver, a lane that stays lost is
          the mat's edge, the course's end: OUTCOME_FINISHED.
        - finish "stop_line": the robot stops at the first stop line after
          the last maneuver (the tracker's reached): OUTCOME_FINISHED.
        - A lane that stays lost before then is off course: a safety stop,
          OUTCOME_EARLY, with the step it got to.
    Phase 3 marks the lane stale after a short hold (hold_max_frames, about
    0.35 s). From then on this rule slows to SLOW_DUTY, steering by the
    heading turned since vision was lost, as lane keeping does; if the lane
    is still stale after END_STALE_MS of that, it brakes and ends the run.
    The lane coming back (vision or hold) cancels it all.

    The count pauses while a higher-priority rule holds the frame: crossing
    an intersection (no lane boundaries in the middle of one) and stopped at
    a stop sign or red light aren't the end of the course.

Main package:
    EndOfCourseRule: update(packet, held) -> a slow command, BRAKE once
        finished, None while the lane is fine. finished: the run is over;
        outcome: OUTCOME_FINISHED or OUTCOME_EARLY; end_step: the step it
        ended on.

Flow:
    lane vision / hold -> nothing to say, count reset
    lane stale, not held -> creep at SLOW_DUTY; count the time
    count >= END_STALE_MS -> BRAKE, finished: OUTCOME_FINISHED if the route
                             is done and finishes at the edge, else OUTCOME_EARLY
    the finish line reached -> BRAKE, finished: OUTCOME_FINISHED
    (BRAKE from then on)
"""
from src.estimation.estimation import LANE_STALE, EstimationPacket
from src.navigation.lane_keeping import SOURCE_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.route import FINISH_EDGE, RouteProgress
from src.navigation.stop_line import StopLineTracker

# Duty while the lane is stale. Half of lane keeping's 0.40 would be 0.20,
# under the 0.25 stall duty; 0.30 is the maneuver trial's slow turn band,
# which moved the robot (2026-09-30)
SLOW_DUTY = 0.30
# Steering clamp at SLOW_DUTY: as lane keeping's 0.40 at 0.40, the inner
# wheel may stop but never reverse
SLOW_MAX_STEERING_ADJ = SLOW_DUTY
# Stale this long at SLOW_DUTY (on top of Phase 3's ~0.35 s hold) ends the
# run. A guess between ending on glare and rolling off the mat: measure the
# roll-out past the lane's end on the mat and tune
END_STALE_MS = 1000
# Longest packet gap counted as one step, as Phase 3's max_dt_s
MAX_DT_MS = 500

REASON_SLOW, REASON_FINISHED = "lane_stale_slow", "end_of_course"
OUTCOME_FINISHED, OUTCOME_EARLY = "finished", "ended_early"


class EndOfCourseRule:
    """
    Creep on a stale lane; finish the run when it stays stale.

    Inputs:
        lane: The LaneKeepingNavigator whose steer() and heading gain it uses.
        progress: The run's RouteProgress, which navigation.Navigation
            advances; RouteProgress() (no maneuvers, finish at the edge)
            makes any lost lane a normal finish.
        tracker: The shared StopLineTracker, for the finish line; None
            without one.
        end_stale_ms: See END_STALE_MS.

    finished: True once the run is over; stays True until reset().
    outcome: OUTCOME_FINISHED or OUTCOME_EARLY once finished; None before.
    end_step: progress.step when it ended.
    record: {"reason": REASON_SLOW, "stale_ms", "source", "steer"} while
        creeping, {"reason": REASON_FINISHED, "outcome"} once finished, {} otherwise.
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
        """Not finished, no stale time."""
        self.outcome: str | None = None
        self.end_step: int | None = None
        self._stale_ms = 0
        self._last_ms: int | None = None
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """
        This frame's say.

        Inputs:
            packet: This frame's packet.
            held: A higher-priority rule (an intersection crossing, a stop
                sign or red light) decided this frame: the time doesn't count.
        Outputs:
            BRAKE once finished; a slow command while the lane is stale;
            None otherwise.
        """
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
