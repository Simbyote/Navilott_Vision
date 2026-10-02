"""Custom maneuvers for lane keeping and stop line navigation with timed sequence handling."""

import time
from src.navigation.navigation import Command, DriveState
from src.navigation.stop_line import StopLineTracker, CROSSING
from src.params import MODE_TWO_BOUNDARY

MAX_CROSS_RIGHT = 1.62
MAX_CROSS_LEFT = 2.75

# Defaults matching intersection timing rules
TWO_BOUNDARY_FRAMES = 3
MAX_CROSS_MS = 3000
MAX_DT_MS = 500


class CustomSequenceNavigator:
    """Navigator providing stateful, timed modular maneuver functions for stop line navigation."""

    def __init__(self, base_navigator):
        self.base_nav = base_navigator
        self.finished = False
        self.outcome = None
        self.end_step = None
        self.record = {}

        # Timing and state tracking attributes
        self._active = False
        self._at_line = False
        self._driving_ms = 0.0
        self._two_boundary = 0
        self._last_ms: int | None = None
        self._current_maneuver = None

    def reset(self) -> None:
        """Reset the base navigator, internal maneuver state, and telemetry."""
        self.base_nav.reset()
        self.finished = False
        self.outcome = None
        self.end_step = None
        self.record = {}

        self._active = False
        self._at_line = False
        self._driving_ms = 0.0
        self._two_boundary = 0
        self._last_ms = None
        self._current_maneuver = None

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
        """Check if packet or StopLineTracker indicates a CROSSING state."""
        tracker = getattr(pkt, "stop_line_tracker", None)
        return (
            getattr(pkt, "drive_state", None) == DriveState.CROSSING
            or tracker == "CROSSING"
            or tracker == StopLineTracker.CROSSING
            or tracker == CROSSING
            or getattr(pkt, "stop_line_detected", False)
        )

    # -------------------------------------------------------------------------
    # Timed Sequence Update Loop
    # -------------------------------------------------------------------------

    def update_maneuver(
        self,
        pkt,
        maneuver_type: str = "straight",
        max_duration_s: float = 3.0,
        held: bool = False,
    ) -> Command | None:
        """Execute timed maneuver until two boundaries are seen or duration limit is reached.

        Args:
            pkt: Current estimation packet with `timestamp_ms` and `lane_mode`.
            maneuver_type: Maneuver type ('straight', 'left', or 'right').
            max_duration_s: Maximum duration (in seconds) allowed for the maneuver.
            held: If True, higher-priority rules (e.g., stop sign) pause maneuver timers.

        Returns:
            Command during active maneuver execution, or None when completed/inactive.
        """
        now = getattr(pkt, "timestamp_ms", int(time.time() * 1000))

        # 1. Trigger maneuver onset when crossing is detected
        if not self._active and self.is_stop_line_crossing(pkt):
            self.reset()
            self._active = True
            self._last_ms = now
            self._current_maneuver = maneuver_type

        if not self._active:
            return None

        # 2. Calculate time delta (bounded by MAX_DT_MS)
        dt_ms = min(max(now - (self._last_ms or now), 0), MAX_DT_MS)
        self._last_ms = now

        # Check line status via StopLineTracker or packet flag
        tracker = getattr(pkt, "stop_line_tracker", None)
        tracker_reached = getattr(tracker, "reached", False) if tracker else True
        if tracker_reached or getattr(pkt, "reached_line", True):
            self._at_line = True

        # 3. Update active maneuver progress and exit checks
        max_duration_ms = max_duration_s * 1000.0
        if self._at_line and not held:
            self._driving_ms += dt_ms
            lane_mode = getattr(pkt, "lane_mode", None)
            self._two_boundary = (
                self._two_boundary + 1 if lane_mode == MODE_TWO_BOUNDARY else 0
            )

            # Exit criteria: consecutive boundaries recognized or timer exceeded
            if (
                self._two_boundary >= TWO_BOUNDARY_FRAMES
                or self._driving_ms >= max_duration_ms
            ):
                self._active = False
                return None

        # 4. Return action command for the selected maneuver
        if self._current_maneuver == "left":
            cmd = self.drive_left_turn()
        elif self._current_maneuver == "right":
            cmd = self.drive_right_turn()
        else:
            cmd = self.drive_forward_after_stopline()

        self.record = {
            "reason": "custom_sequence",
            "maneuver": self._current_maneuver,
            "driving_ms": self._driving_ms,
            "two_boundary_count": self._two_boundary,
        }

        return cmd

    def update(self, pkt) -> Command:
        """Delegate updates to custom maneuvers if active, otherwise base navigator."""
        # Check active custom maneuver sequence first
        custom_cmd = self.update_maneuver(pkt)
        if custom_cmd is not None:
            return custom_cmd

        # Fall back to base navigator update loop
        cmd = self.base_nav.update(pkt)

        # Sync telemetry metadata
        base_record = getattr(self.base_nav, "record", {}) or {}
        self.record = dict(base_record)

        self.finished = getattr(self.base_nav, "finished", False)
        self.outcome = getattr(self.base_nav, "outcome", None)
        self.end_step = getattr(self.base_nav, "end_step", None)

        return cmd