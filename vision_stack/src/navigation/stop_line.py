"""Stop line tracking: when the robot has reached a stop line, from the line leaving the camera's view.

Purpose:
    The camera's view of the floor ends about 10 cm ahead of the robot, so a
    stop line is out of sight before the robot gets to it. This tracker
    follows one line per intersection from first sight to the robot being on
    it, by time or odometry, and the angle it was approached at. The sign, light and
    intersection rules all read this one tracker (navigation_contract.md, item 1).

Main package:
    StopLineTracker: its phase (IDLE, APPROACH, CROSSING), entered (a new
    intersection, one frame), reached (at the line, one frame) and
    approach_tilt_deg.

Flow:
    1. IDLE -> APPROACH when a line is voted in.
    2. APPROACH -> CROSSING when its near edge reaches the view's bottom: entered.
    3. Reached STOP_DELAY_MS later, or REACH_CM short of it by odometry -> IDLE.
    4. A line voted out far up the view is flicker: back to IDLE.
"""
from collections import deque
from statistics import median

from src.estimation.estimation import EstimationPacket

IDLE, APPROACH, CROSSING = "idle", "approach", "crossing"

# Time from the line leaving the bottom of the view to the robot being "at"
# it, where a stop sign or red light brakes it. 1500 stopped it a few cm past
# the line, 500 at it (2026-10-05). 0, tuned on the course: its traffic
# light hangs over the center of the intersection, and any drive past the
# line carries the robot under it and out of view; braking as the line leaves
# the view keeps the whole light in one frame. Turns get their distance from
# the intersection rule's ADVANCE_MS instead. It's a time, so it moves with
# speed (the battery)
STOP_DELAY_MS = 0
# By odometry instead: the robot is at the line when this many cm of it are
# left ahead of the stop-line table's reference point (the front edge), so
# 0 stops on it. The line leaves the view ~10 cm ahead, and a crossing needs
# one frame past that, so anything near 10 or more is STOP_DELAY_MS 0 again.
# None: by STOP_DELAY_MS. Set from left_cm on reached frames (nav.csv's
# line_left_cm, 2026-10-20)
REACH_CM: float | None = 9.0
# Fresh tilts (the packet's stop_line_tilt_deg) the approach angle is the
# median of: the last ones before the line leaves the view, the nearest and
# so the largest in the image. 5 is a quarter second at 20 FPS
TILT_FRAMES = 5
# A line last seen within this many lane-ROI rows of the bottom left by
# passing under the view; higher up, it was lost. The 2026-09-30 table puts
# 22.6 rows at ~11.9 cm, 2 cm past the view bottom (~10 cm)
NEAR_BOTTOM_ROWS = 25.0
# The line has left the view when its near edge reaches the view's bottom
# (the voted line reads clipped: 0 px), not when the whole tape has passed
# under it. False: when it's voted out, a tape's width (plus a frame of the
# vote) later; ADVANCE_MS and REACH_CM were first tuned that way
CROSS_ON_CLIP = True


class StopLineTracker:
    """
    One stop line at a time, from first sight to the robot reaching it.

    Attributes, read by the rules after update():
        phase: IDLE, APPROACH or CROSSING.
        entered: True only on the frame the line leaves the view near the
            bottom (CROSSING starts): one intersection, for the route's count.
        reached: True only on the frame the robot reaches the line.
        last_rows: The line's last seen stop_line_distance_px; None in IDLE.
        lost_ms: timestamp_ms the line left the view; None unless CROSSING.
        seen_ms: timestamp_ms the current line first came into view; on the
            reached frame, the line just reached's. None in IDLE before any line.
        approach_tilt_deg: the median of the line's last TILT_FRAMES fresh
            tilts in the image, + = its right end nearer the robot; set while
            CROSSING (and on the reached frame), None otherwise or with none.
        left_cm: cm of the tracked line still ahead, by odometry from its
            last fresh sighting; on the reached frame, what was left when it
            was reached. None without odometry or a cm for the line, and in IDLE.
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
        """Back to IDLE, no line."""
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
        A fresh sighting's cm puts the line at an odometry position. A held
        cm (Phase 3 repeats it through a missed frame) isn't fresh, and
        neither is a clipped one (distance_px 0: the line runs off the view
        bottom, so its near edge is somewhere under the view's edge, ~9-10 cm
        ahead, which the stop-line table reports as 0); the last whole
        sighting stays the measurement.
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
        """CROSSING: reached by distance when measured and set, else by STOP_DELAY_MS."""
        if self.reach_cm is not None and self.left_cm is not None:
            return self.left_cm - self._step_cm / 2.0 <= self.reach_cm
        return packet.timestamp_ms - self.lost_ms >= self.delay_ms

    def update(self, packet: EstimationPacket, accept_new: bool = True) -> None:
        """
        Advance on this frame's packet (stop_line_detected, stop_line_distance_px, timestamp_ms).

        Inputs:
            accept_new: False keeps a line seen while IDLE from starting a
                new APPROACH (Navigation, while an intersection is being
                crossed); a line already tracked carries on.
        """
        self.entered = self.reached = False
        if self.phase == IDLE:                  # what the reached frame kept is over
            self.left_cm = self.approach_tilt_deg = None
        odo = packet.distance_cm
        self._step_cm = 0.0 if odo is None or self._odometry is None else max(odo - self._odometry, 0.0)
        self._odometry = odo
        seen = bool(packet.stop_line_detected)
        clipped = seen and self.cross_on_clip and packet.stop_line_distance_px == 0
        if self.phase == IDLE:
            # A line first seen already clipped is the tape just reached, still under the view's
            # edge (or one the robot started on): not a line coming up
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
