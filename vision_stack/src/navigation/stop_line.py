"""Stop line tracking: when the robot has reached a stop line it can no longer see.

Purpose:
    The view of the floor ends about 10 cm ahead, so a line is out of sight
    before the robot gets to it. This follows one line per intersection,
    from first sight until the robot is on it, by odometry or by time. The
    sign, light and intersection rules all read this one tracker.

Main package:
    StopLineTracker: phase, and the one-frame flags entered (a new
    intersection) and reached (at the line).
"""
from collections import deque
from statistics import median

from src.estimation.estimation import EstimationPacket

IDLE, APPROACH, CROSSING = "idle", "approach", "crossing"

# Without odometry: at the line as soon as it leaves the view. The course's
# light hangs over the intersection, and any drive past the line loses it
# from view (tuned on the course; 500 stopped at the line, 2026-10-05)
STOP_DELAY_MS = 0
# With odometry: at the line when this many cm of it are left ahead of the
# front edge. A guess (2026-10-20) from where a line leaves the view (~10 cm),
# so the robot stops while the hanging light is still in view
REACH_CM: float | None = 9.0
# The approach angle is the median of the last tilts before the line leaves the view (~0.25 s)
TILT_FRAMES = 5
# A line lost this near the bottom passed under the view; higher up it was
# lost. 22.6 rows is ~11.9 cm on the 2026-09-30 table
NEAR_BOTTOM_ROWS = 25.0
# The line leaves the view when its near edge reaches the bottom (clipped),
# not once the whole tape has passed; ADVANCE_MS and REACH_CM assume this
CROSS_ON_CLIP = True


class StopLineTracker:
    """
    Read by the rules after update(). seen_ms, approach_tilt_deg and left_cm
    stay set on the reached frame so the rules know which line it was.
    approach_tilt_deg is + when the line's right end is nearer the robot.
    """
    def __init__(self, delay_ms: int = STOP_DELAY_MS, near_bottom_rows: float = NEAR_BOTTOM_ROWS,
                 reach_cm: float | None = REACH_CM, cross_on_clip: bool = CROSS_ON_CLIP) -> None:
        self.delay_ms = delay_ms
        self.cross_on_clip = cross_on_clip
        self.near_bottom_rows = near_bottom_rows
        self.reach_cm = reach_cm
        self._odometry: float | None = None       # the previous frame's distance_cm
        self._step_cm = 0.0                       # odometry over the previous frame
        self.reset()

    def reset(self) -> None:
        self.phase = IDLE
        self.entered = self.reached = False
        self.last_rows: float | None = None
        self.lost_ms: int | None = None
        self.seen_ms: int | None = None
        self.left_cm: float | None = None
        self.approach_tilt_deg: float | None = None
        self._tilts: deque = deque(maxlen=TILT_FRAMES)
        self._line_at: float | None = None        # the line's odometry position
        self._sighting: float | None = None       # the cm it was last measured at

    def _locate(self, packet: EstimationPacket) -> None:
        """
        A fresh sighting puts the line at an odometry position. Held and
        clipped readings aren't fresh: a clipped line reads 0 cm though its
        edge is ~9-10 cm ahead.
        """
        if packet.stop_line_tilt_deg is not None:
            self._tilts.append(packet.stop_line_tilt_deg)
        cm, odo = packet.stop_line_distance_cm, packet.distance_cm
        if packet.stop_line_distance_px == 0:
            return
        if cm is not None and odo is not None and cm != self._sighting:
            self._line_at, self._sighting = odo + cm, cm

    def _left(self, packet: EstimationPacket) -> float | None:
        return None if self._line_at is None or packet.distance_cm is None else self._line_at - packet.distance_cm

    def _at_line(self, packet: EstimationPacket) -> bool:
        if self.reach_cm is not None and self.left_cm is not None:
            return self.left_cm - self._step_cm / 2.0 <= self.reach_cm
        return packet.timestamp_ms - self.lost_ms >= self.delay_ms

    def update(self, packet: EstimationPacket, accept_new: bool = True) -> None:
        """accept_new=False keeps a new line from starting while an intersection is crossed."""
        self.entered = self.reached = False
        if self.phase == IDLE:                  # what the reached frame kept is over
            self.left_cm = self.approach_tilt_deg = None
        odo = packet.distance_cm
        self._step_cm = 0.0 if odo is None or self._odometry is None else max(odo - self._odometry, 0.0)
        self._odometry = odo
        seen = bool(packet.stop_line_detected)
        clipped = seen and self.cross_on_clip and packet.stop_line_distance_px == 0
        if self.phase == IDLE:
            # First seen already clipped: the line just reached, or one the robot started on
            if seen and accept_new and not clipped:
                self.phase, self.last_rows = APPROACH, packet.stop_line_distance_px
                self.seen_ms = packet.timestamp_ms
                self._locate(packet)
        elif self.phase == APPROACH:
            if seen and not clipped:
                self.last_rows = packet.stop_line_distance_px
                self._locate(packet)
            elif clipped or (self.last_rows is not None and self.last_rows <= self.near_bottom_rows):
                self.phase, self.lost_ms = CROSSING, packet.timestamp_ms
                self.entered = True
                self.approach_tilt_deg = median(self._tilts) if self._tilts else None
            else:
                self.reset()                    # lost far away: not this robot driving over it
        else:                                   # CROSSING; a re-sighting is the same line flickering
            self.left_cm = self._left(packet)
            if self._at_line(packet):
                seen_ms, left, tilt = self.seen_ms, self.left_cm, self.approach_tilt_deg
                self.reset()
                self.reached, self.seen_ms, self.left_cm = True, seen_ms, left    # the rules read which line it was
                self.approach_tilt_deg = tilt
                return
        if self.phase != IDLE:
            self.left_cm = self._left(packet)
