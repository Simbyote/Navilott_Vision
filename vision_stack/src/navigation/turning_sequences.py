"""Custom maneuvers for lane keeping and stop line navigation."""

import time
from src.navigation.navigation import Command, DriveState
from src.navigation.stop_line import StopLineTracker


class CustomSequenceNavigator:
    """Navigator providing modular maneuver functions for stop line navigation."""

    def __init__(self, base_navigator):
        self.base_nav = base_navigator
        self.finished = False
        self.outcome = None
        self.end_step = None
        self.record = {}

    def reset(self) -> None:
        """Reset the base navigator and telemetry."""
        self.base_nav.reset()
        self.finished = False
        self.outcome = None
        self.end_step = None
        self.record = {}

    # -------------------------------------------------------------------------
    # Maneuver Methods
    # -------------------------------------------------------------------------

    def drive_forward_after_stopline(self, speed: float = 0.4) -> Command:
        """Standard forward drive command executed after detecting a stop line crossing."""
        return Command(left=speed, right=speed, brake=False)

    def drive_left_turn(self, left_speed: float = 0.36, right_speed: float = 0.63) -> Command:
        """Left turn maneuver command with asymmetric wheel speeds."""
        return Command(left=left_speed, right=right_speed, brake=False)

    def drive_right_turn(self, left_speed: float = 0.45, right_speed: float = 0.0) -> Command:
        """Right turn maneuver command with asymmetric wheel speeds."""
        return Command(left=left_speed, right=right_speed, brake=False)

    def is_stop_line_crossing(self, pkt) -> bool:
        """Check if StopLineTracker indicates a CROSSING state."""
        return (
            getattr(pkt, "drive_state", None) == DriveState.CROSSING
            or getattr(pkt, "stop_line_tracker", None) == "CROSSING"
            or getattr(pkt, "stop_line_tracker", None) == StopLineTracker.CROSSING
            or getattr(pkt, "stop_line_detected", False)
        )

    # -------------------------------------------------------------------------
    # Base Update Loop
    # -------------------------------------------------------------------------

    def update(self, pkt) -> Command:
        """Delegate frame updates to the base navigator."""
        cmd = self.base_nav.update(pkt)

        # Sync telemetry metadata for reporting and nav.csv logging
        base_record = getattr(self.base_nav, "record", {}) or {}
        self.record = dict(base_record)

        self.finished = getattr(self.base_nav, "finished", False)
        self.outcome = getattr(self.base_nav, "outcome", None)
        self.end_step = getattr(self.base_nav, "end_step", None)

        return cmd