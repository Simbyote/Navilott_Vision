"""Traffic light rule: at a stop line, wait while the light is red.

Purpose:
    A stop line stops the robot only with a stop sign (stop_sign.py) or a
    red light. When the shared StopLineTracker says the robot has reached
    the line and Phase 3's voted drive_state is "stop" (red), this rule
    brakes, and keeps braking until the light is no longer red. Green and
    caution (yellow) drive on: only red stops (decided 2026-10-01).

Main package:
    TrafficLightRule: update(packet, held) -> BRAKE while waiting at a red
        light, None otherwise (navigation.Navigation falls through).

Flow:
    1. On the tracker's reached frame: red starts the wait.
    2. Each frame while waiting: still red -> BRAKE; anything else -> release.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.stop_line import StopLineTracker

RED_STATE = "stop"              # Phase 3's drive_state for a red light
REASON_RED = "red_light"


class TrafficLightRule:
    """
    Wait at a stop line while the light is red.

    Inputs:
        tracker: The StopLineTracker navigation.Navigation updates each frame.

    record: {"reason": REASON_RED} while it brakes; {} otherwise.
    """
    def __init__(self, tracker: StopLineTracker) -> None:
        self.tracker = tracker
        self.reset()

    def reset(self) -> None:
        """Not waiting."""
        self._waiting = False
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
            BRAKE while waiting at a red light; None otherwise.
        """
        red = packet.drive_state == RED_STATE
        if self.tracker.reached and red:
            self._waiting = True
        if self._waiting and not red:
            self._waiting = False
        self.record = {"reason": REASON_RED} if self._waiting else {}
        return BRAKE if self._waiting else None
