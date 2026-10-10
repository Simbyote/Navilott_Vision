"""Intersection rule: cross an intersection straight, left or right, as the route says, then hand back to lane keeping.

Purpose:
    Everything the robot does inside an intersection lives here, so one
    record (nav.csv's intersection rows) shows the whole crossing.
    Lane keeping goes wrong entering an intersection: a boundary on the
    right that crosses the robot's path pulls the offset to the left
    (2026-10-01 runs). So from the moment the stop line leaves the bottom of
    the view (the shared StopLineTracker's CROSSING) this rule drives, in
    four stages:

        STAGE_TO_LINE  straight to the line on a gyro heading hold (base
                       duty, steering only against the heading turned).
        STAGE_ADVANCE  left or right only, from the line (the tracker's
                       reached, STOP_DELAY_MS later): on into the
                       intersection on the heading hold for ADVANCE_MS of
                       driving, since the stop lines sit back in the street.
        STAGE_TURN     left or right only, after the advance: fixed wheel
                       duties (LEFT_TURN, RIGHT_TURN, with their history)
                       until the gyro has turned TURN_TARGET_DEG that way,
                       or the turn's time limit if it never does.
        STAGE_EXIT     straight on the new heading until the lane is back:
                       both boundaries for TWO_BOUNDARY_FRAMES frames in a
                       row, at least one for ONE_BOUNDARY_FRAMES, or
                       MAX_CROSS_MS of driving. Then lane keeping takes over.

    Straight goes from STAGE_TO_LINE to STAGE_EXIT at the line, with no
    advance. Which way to go comes from the route (route.py) through the
    RouteProgress that
    navigation.Navigation advances as each intersection is entered; past
    the plan, or at the finish line, it's straight. A stop sign or red light
    at the line (a higher-priority rule) holds the robot first: held frames
    advance no timer, so the turn starts once the robot is let go.

    While a crossing is active, Navigation keeps the tracker from taking up
    a new stop line, so a line-like mark inside the intersection can't
    restart it.

Main package:
    IntersectionRule: update(packet, held) -> the crossing's command,
        None when not crossing. active, stage.

Flow (each frame while active):
    1. Integrate yaw (net of the gyro bias) into the heading.
    2. At the line: STAGE_ADVANCE for left / right, else STAGE_EXIT.
    3. STAGE_ADVANCE: the heading hold for ADVANCE_MS; then STAGE_TURN.
    4. STAGE_TURN: the turn's duties until the heading reaches the target
       or the time limit; then STAGE_EXIT, holding the heading it ended on.
    5. STAGE_EXIT: count boundary frames and driving time; end on any limit.
    Every stage before the end returns a command and a record.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.lane_keeping import KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import Command
from src.params import MODE_LEFT_ONLY, MODE_RIGHT_ONLY, MODE_TWO_BOUNDARY
from src.navigation.route import LEFT, RIGHT, STRAIGHT, RouteProgress
from src.navigation.stop_line import CROSSING, StopLineTracker

# Both boundaries this many frames in a row ends the crossing (~0.5 s at
# 20 FPS). 3 ended it as soon as the far side's lane lines showed, and the
# next lane's stop line was taken up right after a straight crossing; the
# crossing keeps new stop lines out while it lasts (2026-10-05)
TWO_BOUNDARY_FRAMES = 10
# At least one boundary (two, or left_only / right_only) this many frames in
# a row also ends it: past some intersections the camera sees only one lane
# line, and the crossing never ended (2026-10-01 run: right_only for 5 s,
# the robot pressed against the right line). Longer than TWO_BOUNDARY_FRAMES,
# since one line is weaker evidence than two (~1 s at 20 FPS; 6 until
# 2026-10-05, raised with it)
ONE_BOUNDARY_FRAMES = 20
BOUNDARY_MODES = (MODE_TWO_BOUNDARY, MODE_LEFT_ONLY, MODE_RIGHT_ONLY)
# Driving time in STAGE_EXIT before lane keeping takes over anyway; a guess
# at crossing one intersection, tuned on the mat (3 s 2026-10-01, 4 s
# 2026-10-05 with the longer boundary counts)
MAX_CROSS_MS = 4000
# Longest packet gap integrated as one step, as Phase 3's max_dt_s
MAX_DT_MS = 500
# Driving from the line into the intersection before a turn starts, on the
# heading hold. The course's stop lines are moving back into the street, away
# from the intersection, and the robot now stops as the line leaves the view
# (STOP_DELAY_MS = 0), so a turn from there would cut the corner short
# (2026-10-07). Straight crossings don't advance: they drive on to the exit.
# Counts only while not held, like the turn's time. Tune on the mat
ADVANCE_MS = 1000

# Turn duties (left, right): left is a wide arc into the far lane, right a
# tight arc around the right wheel. Measured on the mat 2026-10-01 as
# (0.36, 0.63) and (0.45, 0.0); raised 2026-10-06 (motor recalibration)
LEFT_TURN = Command(0.46, 0.73)
RIGHT_TURN = Command(0.55, 0.25)
# A turn ends once the gyro reads this many degrees turned its way: 90 less
# the drive's coast after the turn stops (maneuver_linker's 180 turn aims
# at 171). Tune on the mat: overshoot -> lower, short -> raise
TURN_TARGET_DEG = 85.0
# ... or after this long if the gyro never gets there (no IMU, a wrong
# sign): open-loop times (left 2.75 s, right 1.62 s) plus half
LEFT_TURN_MAX_MS = 4100
RIGHT_TURN_MAX_MS = 2400
TURNS = {LEFT: (LEFT_TURN, -1, LEFT_TURN_MAX_MS),        # (duties, heading sign, time limit); + heading = right
         RIGHT: (RIGHT_TURN, +1, RIGHT_TURN_MAX_MS)}

STAGE_TO_LINE, STAGE_ADVANCE, STAGE_TURN, STAGE_EXIT = "to_line", "advance", "turn", "exit"
TURN_END_GYRO, TURN_END_TIME = "gyro target", "time limit"
REASON_CROSSING, REASON_TURNING = "crossing", "turning"
SOURCE_HEADING_HOLD, SOURCE_TURN = "heading_hold", "turn"


class IntersectionRule:
    """
    Across an intersection the way the route says: to the line, the turn if
    any, out on the new heading.

    Inputs:
        tracker: The StopLineTracker navigation.Navigation updates each frame.
        lane: The LaneKeepingNavigator whose steer() the heading hold drives through.
        kp_heading: Duty per degree turned; lane keeping's own by default.
        gyro_bias_dps: Subtracted from the packet's yaw_rate, which is raw
            (Phase 3 subtracts its bias only inside heading_error).
        progress: The run's RouteProgress, for this intersection's maneuver;
            RouteProgress() (no plan: straight) by default.

    active: True while crossing. stage: STAGE_* while active, None otherwise.
    record: {"reason": REASON_CROSSING or REASON_TURNING, "source", "steer",
        "heading_deg", "stage", "step" (progress.label()), "maneuver",
        "turn_end" (how this crossing's turn ended: TURN_END_GYRO,
        TURN_END_TIME; None before it does, or with no turn)} while crossing;
        {} otherwise.
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
        self._active = False
        self.stage: str | None = None
        self._maneuver = STRAIGHT
        self._heading = 0.0
        self._turn_ms = self._driving_ms = self._advance_ms = 0
        self._turn_end: str | None = None
        self._two_boundary = self._any_boundary = 0
        self._last_ms: int | None = None
        self.record: dict = {}

    @property
    def active(self) -> bool:
        """Crossing: from the line leaving the view until the crossing ends."""
        return self._active

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """
        This frame's say.

        Inputs:
            packet: This frame's packet; the tracker has already seen it.
            held: A higher-priority rule (stop sign, red light) decided this
                frame: the robot is braked at the line, so no timer runs and
                what Phase 2 sees doesn't count towards ending the crossing.
        Outputs:
            The crossing's command (heading hold or turn) while crossing;
            None otherwise.
        """
        now = packet.timestamp_ms
        if not self._active and self.tracker.phase == CROSSING:
            self.reset()
            self._active, self._last_ms, self.stage = True, now, STAGE_TO_LINE
            current = self.progress.current
            self._maneuver = current[2] if current and current[2] in TURNS else STRAIGHT
        if not self._active:
            return None

        dt_ms = min(max(now - self._last_ms, 0), MAX_DT_MS)
        self._last_ms = now
        self._heading += (packet.yaw_rate - self.gyro_bias_dps) * dt_ms / 1000.0

        if self.stage == STAGE_TO_LINE and self.tracker.reached:
            # Route through STAGE_ADVANCE if turning, otherwise go straight to EXIT
            self.stage = STAGE_ADVANCE if self._maneuver in TURNS else STAGE_EXIT

        if self.stage == STAGE_ADVANCE:
            if not held:
                self._advance_ms += dt_ms
            if self._advance_ms >= ADVANCE_MS:
                self.stage = STAGE_TURN
        if self.stage == STAGE_TURN:
            duties, sign, max_ms = TURNS[self._maneuver]
            if not held:
                self._turn_ms += dt_ms
            if sign * self._heading >= TURN_TARGET_DEG or self._turn_ms >= max_ms:
                self._turn_end = TURN_END_GYRO if sign * self._heading >= TURN_TARGET_DEG else TURN_END_TIME
                self.stage, self._heading = STAGE_EXIT, 0.0     # hold the heading the turn ended on
        if self.stage == STAGE_EXIT and not held:
            self._driving_ms += dt_ms
            self._two_boundary = self._two_boundary + 1 if packet.lane_mode == MODE_TWO_BOUNDARY else 0
            self._any_boundary = self._any_boundary + 1 if packet.lane_mode in BOUNDARY_MODES else 0
            if (self._two_boundary >= TWO_BOUNDARY_FRAMES or self._any_boundary >= ONE_BOUNDARY_FRAMES
                    or self._driving_ms >= MAX_CROSS_MS):
                self.reset()
                return None

        if self.stage == STAGE_TURN:
            cmd = duties
            reason, source, steer = REASON_TURNING, SOURCE_TURN, round((cmd.right - cmd.left) / 2.0, 3)
        else:
            # + heading = turned right, so steer left: + steering
            cmd = self.lane.steer(self.kp_heading * self._heading, SOURCE_HEADING_HOLD)
            reason, source, steer = REASON_CROSSING, SOURCE_HEADING_HOLD, self.lane.record.get("steer", 0.0)
        self.record = {"reason": reason, "source": source, "steer": steer,
                       "heading_deg": round(self._heading, 2), "stage": self.stage,
                       "step": self.progress.label(), "maneuver": self._maneuver, "turn_end": self._turn_end}
        return cmd
