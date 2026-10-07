"""Traffic light rule: at a stop line, wait while the light is red or yellow.

Purpose:
    A stop line stops the robot only with a stop sign (stop_sign.py) or a
    red light. When the shared StopLineTracker says the robot has reached
    the line and Phase 3's voted drive_state was "stop" (red) or "caution"
    (yellow) within RED_MEMORY_MS, this rule brakes, and keeps braking until
    the light has read green or nothing for RELEASE_MS. Only green drives on.

    Yellow stops too (2026-10-07; until then only red did): from some angles
    the overexposed red LED's ring reads orange, inside yellow's band, and
    red read as yellow drove through. Stopping on both, a red misread as
    yellow still stops the robot, and a real yellow turns red anyway.

    The light is a few small LEDs, and the detector can miss one for a
    frame. The rule used to need red on the exact frame the line was
    reached, and let go on the first frame that wasn't red, so one dropped
    frame ran the light either way. The memory and the release time take
    the red from the frames around it instead (2026-10-06).

Main package:
    TrafficLightRule: update(packet, held) -> BRAKE while waiting at a red
        or yellow light, None otherwise (navigation.Navigation falls through).

Flow:
    1. Remember when the light last read red or yellow.
    2. On the tracker's reached frame: one within RED_MEMORY_MS starts the wait.
    3. Each frame while waiting: red or yellow -> BRAKE; neither for RELEASE_MS -> release.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.stop_line import StopLineTracker

RED_STATE = "stop"              # Phase 3's drive_state for a red light
CAUTION_STATE = "caution"       # ... and for a yellow one
STOP_STATES = (RED_STATE, CAUTION_STATE)
REASON_RED = "red_light"

# Red (or yellow) seen this long before the robot reaches the line still stops it: a
# light dropped on the reached frame alone ran the red. ~10 frames at 20
# FPS; a light that turned green longer ago than this drives on. Tune on the mat
RED_MEMORY_MS = 500
# Waiting ends once the light has read neither red nor yellow this long:
# one dropped frame let the robot go on a red. ~5 frames at 20 FPS
RELEASE_MS = 250


class TrafficLightRule:
    """
    Wait at a stop line while the light is red or yellow.

    Inputs:
        tracker: The StopLineTracker navigation.Navigation updates each frame.
        red_memory_ms, release_ms: See the constants above.

    record: {"reason": REASON_RED} while it brakes; {} otherwise.
    """
    def __init__(self, tracker: StopLineTracker, red_memory_ms: int = RED_MEMORY_MS,
                 release_ms: int = RELEASE_MS) -> None:
        self.tracker = tracker
        self.red_memory_ms = red_memory_ms
        self.release_ms = release_ms
        self.reset()

    def reset(self) -> None:
        """Not waiting, no red remembered."""
        self._waiting = False
        self._red_ms: int | None = None         # when the light last read red
        self._clear_ms: int | None = None       # while waiting: when it stopped reading red
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """
        This frame's say.

        Inputs:
            packet: This frame's packet; the tracker has already seen it.
            held: A higher-priority rule (the stop sign) decided this frame.
                The light is still watched, so a red seen at the line holds
                the robot after the stop sign's hold ends.
        Outputs:
            BRAKE while waiting at a red or yellow light; None otherwise.
        """
        now = packet.timestamp_ms
        red = packet.drive_state in STOP_STATES
        if red:
            self._red_ms = now
        if (self.tracker.reached and self._red_ms is not None
                and now - self._red_ms <= self.red_memory_ms):
            self._waiting = True
        if self._waiting:
            if red:
                self._clear_ms = None
            elif self._clear_ms is None:
                self._clear_ms = now
            if self._clear_ms is not None and now - self._clear_ms >= self.release_ms:
                self._waiting, self._clear_ms = False, None
        self.record = {"reason": REASON_RED} if self._waiting else {}
        return BRAKE if self._waiting else None
