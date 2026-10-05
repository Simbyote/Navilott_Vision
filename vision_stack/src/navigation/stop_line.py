"""Stop line tracking: when the robot has reached a stop line, from the line leaving the camera's view.

Purpose:
    The camera's view of the floor ends about 10 cm ahead of the robot
    (2026-09-30 stop-line table), so a stop line is out of sight before the
    robot gets to it. Braking the moment it disappears stopped the robot
    short. This tracker follows one stop line per intersection through the
    packets: seen (APPROACH), gone from the bottom of the view (CROSSING),
    and STOP_DELAY_MS later the robot is at it (reached, one frame). The
    navigation rules (stop sign, traffic light, intersection) all read this
    one tracker, which navigation.Navigation updates once per frame.

    A line that drops out while still far up the image is flicker or a lost
    detection, not the robot driving over it: only a line last seen within
    NEAR_BOTTOM_ROWS of the lane ROI's bottom counts as passing under the view.

    A stop line on its own decides nothing; it says an intersection is
    coming. What happens there is the rules' call.

Main package:
    StopLineTracker: update(packet) once per frame; phase, reached, lost_ms.
    IDLE, APPROACH, CROSSING: its phases.

Flow:
    IDLE -> APPROACH (line voted in, when accept_new) -> CROSSING (voted out near the bottom:
    entered, one intersection) -> reached on the frame STOP_DELAY_MS after it
    left -> IDLE.
    APPROACH -> IDLE when it's voted out far away.
"""
from src.estimation.estimation import EstimationPacket

IDLE, APPROACH, CROSSING = "idle", "approach", "crossing"

# Time from the line leaving the bottom of the view to the robot being at it.
# Braking at the moment it left stopped the robot early (2026-10-01 runs);
# "a second or two after it leaves the frame". Tune on the mat
STOP_DELAY_MS = 500
# A line last seen within this many lane-ROI rows of the bottom left by
# passing under the view; higher up, it was lost. The 2026-09-30 table puts
# 22.6 rows at ~11.9 cm, 2 cm past the view bottom (~10 cm)
NEAR_BOTTOM_ROWS = 25.0


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
    """
    def __init__(self, delay_ms: int = STOP_DELAY_MS, near_bottom_rows: float = NEAR_BOTTOM_ROWS) -> None:
        self.delay_ms = delay_ms
        self.near_bottom_rows = near_bottom_rows
        self.reset()

    def reset(self) -> None:
        """Back to IDLE, no line."""
        self.phase = IDLE
        self.entered = self.reached = False
        self.last_rows: float | None = None
        self.lost_ms: int | None = None

    def update(self, packet: EstimationPacket, accept_new: bool = True) -> None:
        """
        Advance on this frame's packet (stop_line_detected, stop_line_distance_px, timestamp_ms).

        Inputs:
            accept_new: False keeps a line seen while IDLE from starting a
                new APPROACH (Navigation, while an intersection is being
                crossed); a line already tracked carries on.
        """
        self.entered = self.reached = False
        seen = bool(packet.stop_line_detected)
        if self.phase == IDLE:
            if seen and accept_new:
                self.phase, self.last_rows = APPROACH, packet.stop_line_distance_px
        elif self.phase == APPROACH:
            if seen:
                self.last_rows = packet.stop_line_distance_px
            elif self.last_rows is not None and self.last_rows <= self.near_bottom_rows:
                self.phase, self.lost_ms = CROSSING, packet.timestamp_ms
                self.entered = True
            else:
                self.reset()                    # lost far away: not this robot driving over it
        elif packet.timestamp_ms - self.lost_ms >= self.delay_ms:     # CROSSING; a re-sighting is the same line flickering
            self.reset()
            self.reached = True
