"""Traffic light rule: at a stop line, wait while the light is red or yellow.

Purpose:
    A light read red or yellow just before the line is reached brakes the
    robot until the light has read green or nothing for a short while.
    Yellow stops too, since a red LED's ring can read yellow.

Main package:
    TrafficLightRule.update(packet) -> BRAKE while waiting, else None.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.stop_line import StopLineTracker

RED_STATE = "stop"              # Phase 3's drive_state for a red light
CAUTION_STATE = "caution"       # ... and for a yellow one
STOP_STATES = (RED_STATE, CAUTION_STATE)
REASON_RED = "red_light"

# Red seen this long before the line still counts (~10 frames): a light
# dropped on the reached frame alone once ran the red
RED_MEMORY_MS = 500
# Release after this long without red (~5 frames): one dropped frame once
# let the robot go on a red
RELEASE_MS = 250


class TrafficLightRule:
    """tracker is the one Navigation updates each frame. record holds the reason while braking."""
    def __init__(self, tracker: StopLineTracker, red_memory_ms: int = RED_MEMORY_MS,
                 release_ms: int = RELEASE_MS) -> None:
        self.tracker = tracker
        self.red_memory_ms = red_memory_ms
        self.release_ms = release_ms
        self.reset()

    def reset(self) -> None:
        self._waiting = False
        self._red_ms: int | None = None         # when the light last read red
        self._clear_ms: int | None = None       # while waiting: when it stopped reading red
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """BRAKE while waiting, else None. The light is watched even when held, so a red holds past a stop sign."""
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
