"""Intersection rule: cross an intersection straight, left or right, as the route says, then hand back to lane keeping.

Purpose:
    Lane keeping goes wrong entering an intersection: the crossing street's
    lines pull the offset sideways. So from the moment the stop line leaves
    the view, this rule drives, on a gyro heading counted from the street
    (the approach angle) rather than from the robot, until the new lane is
    back. Everything the robot does inside an intersection lives here, so
    one record shows the whole crossing (docs/vision_pipeline/navigation_contract.md, item 5).

Main package:
    IntersectionRule: update(packet, held) -> a Command for each frame of the
    crossing, None outside one; record has the stage, heading and turn_end.

Flow:
    1. to_line: straight to the line on the heading hold.
    2. advance (left / right): on into the intersection, by distance or time.
    3. turn (left / right): wheel speeds until the gyro reads TURN_TARGET_DEG, or the time limit.
    4. exit: straight on the new heading until the lane is back.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.lane_keeping import KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import Command
from src.params import MODE_LEFT_ONLY, MODE_RIGHT_ONLY, MODE_TWO_BOUNDARY
from src.navigation.route import LEFT, RIGHT, STRAIGHT, RouteProgress
from src.navigation.stop_line import CROSSING, StopLineTracker
from src.navigation.wheel_speed import CPS_PER_DUTY, WheelSpeed

# Both boundaries this many frames in a row ends the crossing (~0.5 s at 20 FPS).
# Fewer (3) ended it as soon as the far side's lane lines showed, and the next
# lane's stop line was taken up right after; the crossing keeps new stop lines
# out while it lasts (2026-10-05)
TWO_BOUNDARY_FRAMES = 10
# At least one boundary (two, or left_only / right_only) this many frames in a
# row also ends it: past some intersections the camera sees only one lane line,
# and the crossing never ended (2026-10-01 run: right_only for 5 s, the robot
# pressed against the right line). Longer than TWO_BOUNDARY_FRAMES, since one
# line is weaker evidence than two (~2 s at 20 FPS; tuned on the mat 2026-10-19)
ONE_BOUNDARY_FRAMES = 40
BOUNDARY_MODES = (MODE_TWO_BOUNDARY, MODE_LEFT_ONLY, MODE_RIGHT_ONLY)
# Driving time in STAGE_EXIT before lane keeping takes over anyway: crossing
# one intersection, tuned on the mat with the boundary counts (2026-10-19)
MAX_CROSS_MS = 5500
# ... or this far, by odometry; None: by MAX_CROSS_MS. From a run's stage_cm at
# the end of timed exits (nav.csv, 2026-10-20)
MAX_CROSS_CM: float | None = 49.7
# Longest packet gap integrated as one step, as Phase 3's max_dt_s
MAX_DT_MS = 500
# Driving from the line into the intersection before a turn starts, on the
# heading hold. The course's stop lines sit back in the street, away from the
# intersection, and the robot stops as the line leaves the view
# (STOP_DELAY_MS = 0, for the hanging traffic light), so a turn from there
# would cut the corner short. 2500, tuned on the course. Straight crossings
# don't advance: they drive on to the exit. Counts only while not held, like
# the turn's time
ADVANCE_MS = 2500
# The advance by odometry instead: floor cm from the line. A time moves with
# the speed, and the speed with the battery (the heading hold drives a duty);
# a distance doesn't. None: by ADVANCE_MS. From a run's stage_cm at the end of
# timed advances (nav.csv, 2026-10-20)
ADVANCE_CM: float | None = 22.3

# Turn duties (left, right): left is a wide arc into the far lane, right a
# tight arc around the right wheel, tuned on the mat (2026-10-19).
# They're the arcs as tuned, at ~11.7 V: with the wheel-speed control
# (wheel_speed.py) the turn asks for these duties' wheel speeds at that
# voltage (x CPS_PER_DUTY), and finds whatever duty gets them on the day's
# battery; without encoders it drives these duties as they are
LEFT_TURN = Command(0.48, 0.73)
RIGHT_TURN = Command(0.78, 0.25)
# A turn ends once the gyro reads this many degrees turned its way: 90 less
# the drive's coast after the turn stops (maneuver_linker's 180 turn aims
# at 171). Tune on the mat: overshoot -> lower, short -> raise
TURN_TARGET_DEG = 85.0
# The approach angle from the stop line's tilt: heading deg (+ = turned
# right of square) per deg of the line's tilt in the image (+ = its right end
# nearer). A line on the floor looks flatter than it is from a low camera, so
# this is larger than 1, and negative: a robot turned right sees the line's
# left end nearer. Measured, not computed (guides/navigation_linker.md,
# "Measuring TILT_TO_HEADING": park at a known angle to a line, read
# stop_line_tilt_deg). None: no correction, the heading starts at 0.
# -4.3: detect-range target=line, 10 cm,
# -6..6 deg (2026-10-20): tilts +1.39, +1.26, +0.03, -0.49, -1.22, redone
# looks within 0.1-0.3 deg. Check the sign on the robot before trusting it
# (an angled start's approach_deg reads + when turned right): wrong, it
# doubles the error instead of removing it
TILT_TO_HEADING: float | None = -4.3
# The tilt a square approach reads: a camera mounted a little rotated sees
# every line tilted. Measured with TILT_TO_HEADING, subtracted before scaling
TILT_ZERO_DEG = 0.19
# The largest approach angle believed. The geometry's tilt gate (max_tilt_deg)
# already refuses steeper lines; this caps a wrong scale
MAX_APPROACH_DEG = 10.0
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


def turned_deg(packet, dt_ms: float, gyro_bias_dps: float) -> float:
    """
    Degrees the robot turned this frame, net of the gyro bias, + = right.

    The packet's yaw_deg when it has one: integrated reading by reading over
    the sensor window, so a long frame gap is counted in full. Otherwise
    yaw_rate x dt_ms (the frame step, already capped at MAX_DT_MS).
    """
    if getattr(packet, "yaw_deg", None) is not None:
        return packet.yaw_deg - gyro_bias_dps * (packet.yaw_span_s or 0.0)
    return (packet.yaw_rate - gyro_bias_dps) * dt_ms / 1000.0


def turn_cps(duties: Command) -> tuple[float, float]:
    """A turn's wheel speeds (counts/s): its duties at the voltage they were tuned at."""
    return duties.left * CPS_PER_DUTY, duties.right * CPS_PER_DUTY


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
        wheels: The WheelSpeed the turn drives through (navigation.Navigation
            shares its own); None drives the turn's duties as they are.
        advance_ms: See ADVANCE_MS.
        advance_cm, max_cross_cm: See ADVANCE_CM, MAX_CROSS_CM; None goes by time.
        tilt_to_heading: See TILT_TO_HEADING; None leaves the approach uncorrected.

    active: True while crossing. stage: STAGE_* while active, None otherwise.
    record: {"reason": REASON_CROSSING or REASON_TURNING, "source", "steer",
        "heading_deg", "stage", "step" (progress.label()), "maneuver",
        "turn_end" (how this crossing's turn ended: TURN_END_GYRO,
        TURN_END_TIME; None before it does, or with no turn), "turn_deg" (the
        heading the turn ended on, before the exit holds from 0; None before
        it does, or with no turn), "approach_deg"
        (the approach angle the heading started at; 0.0 uncorrected), "stage_cm"
        (odometry cm driven in this stage; None without odometry), and while
        turning with wheels, "wheels" (WheelSpeed.record)} while crossing;
        {} otherwise.
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
        """Not crossing."""
        self._active = False
        self.stage: str | None = None
        self._maneuver = STRAIGHT
        self._heading = self._approach = 0.0
        self._turn_ms = self._driving_ms = self._advance_ms = 0
        self._turn_end: str | None = None
        self._turn_deg: float | None = None           # the heading the turn ended on, for the logs
        self._two_boundary = self._any_boundary = 0
        self._last_ms: int | None = None
        self._stage_from_cm: float | None = None      # odometry where the current stage began
        self._stage_seen: str | None = None
        self.record: dict = {}

    def _stage_cm(self, packet: EstimationPacket) -> float | None:
        """Odometry cm driven since the current stage began; None without odometry."""
        if self.stage != self._stage_seen:
            self._stage_seen, self._stage_from_cm = self.stage, packet.distance_cm
        if packet.distance_cm is None:
            return None
        if self._stage_from_cm is None:              # odometry came up mid-stage: measure from here
            self._stage_from_cm = packet.distance_cm
        return packet.distance_cm - self._stage_from_cm

    def approach_deg(self, tilt_deg: float | None) -> float:
        """The approach angle a stop line's tilt says, + = turned right of square; 0.0 without a scale or a tilt."""
        if self.tilt_to_heading is None or tilt_deg is None:
            return 0.0
        return max(-MAX_APPROACH_DEG, min(MAX_APPROACH_DEG, self.tilt_to_heading * (tilt_deg - TILT_ZERO_DEG)))

    @staticmethod
    def _done(cm: float | None, limit_cm: float | None, ms: int, limit_ms: int) -> bool:
        """A stage's distance reached, or its time without odometry or a distance set."""
        if limit_cm is not None and cm is not None:
            return cm >= limit_cm
        return ms >= limit_ms

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
            self._heading = self._approach = self.approach_deg(self.tracker.approach_tilt_deg)
            current = self.progress.current
            self._maneuver = current[2] if current and current[2] in TURNS else STRAIGHT
        if not self._active:
            return None

        dt_ms = min(max(now - self._last_ms, 0), MAX_DT_MS)
        self._last_ms = now
        self._heading += turned_deg(packet, dt_ms, self.gyro_bias_dps)

        if self.stage == STAGE_TO_LINE and self.tracker.reached:
            # Route through STAGE_ADVANCE if turning, otherwise go straight to EXIT
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
