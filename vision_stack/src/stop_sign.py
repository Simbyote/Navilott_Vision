"""Stop sign navigation module.

Purpose:
    Navigates stop signs and stop lines. When a stop condition is met (either a 
    stop sign or a stop line detected in EstimationPacket), it engages BRAKE and 
    holds it for a required wait duration (2 seconds) before allowing motion to resume.
"""

from src.estimation import EstimationPacket
from src.navigation import BRAKE, Command
from stop_line import StopLineNavigator, check_stop_line_trigger

STOP_SIGN_HOLD_TIME_MS = 2000  # 2 seconds stop duration


class StopSignNavigator:
    """
    Navigator for stop sign and stop line handling.

    Inputs:
        hold_time_ms: Required duration in milliseconds to remain stopped (default: 2000ms).
        stop_line_navigator: Optional custom instance of StopLineNavigator.
    """

    def __init__(
        self,
        hold_time_ms: int = STOP_SIGN_HOLD_TIME_MS,
        stop_line_navigator: StopLineNavigator | None = None,
    ) -> None:
        self.hold_time_ms = hold_time_ms
        self.stop_line_nav = stop_line_navigator or StopLineNavigator()

        # State tracking for the 2-second timed stop
        self._stop_start_ms: int | None = None
        self._has_stopped: bool = False

    def reset(self) -> None:
        """Resets the internal stop timer and state memory."""
        self._stop_start_ms = None
        self._has_stopped = False

    def update(self, packet: EstimationPacket) -> Command:
        """
        Processes this frame's EstimationPacket.

        Outputs:
            BRAKE while a stop condition is active or during the 2-second wait window;
            otherwise returns zero duty command.
        """
        # 1. Evaluate stop conditions using stop_line_detected and stop_sign_detected
        stop_line_triggered = packet.stop_line_detected or check_stop_line_trigger(
            packet, threshold_px=self.stop_line_nav.stop_line_threshold_px
        )
        stop_sign_triggered = bool(packet.stop_sign_detected)

        is_trigger_active = stop_line_triggered or stop_sign_triggered

        # 2. Trigger active -> Start or maintain the timed stop
        if is_trigger_active and not self._has_stopped:
            if self._stop_start_ms is None:
                self._stop_start_ms = packet.timestamp_ms
            return BRAKE

        # 3. Check if we are currently holding an active timed stop
        if self._stop_start_ms is not None:
            elapsed_ms = packet.timestamp_ms - self._stop_start_ms
            if elapsed_ms < self.hold_time_ms:
                return BRAKE

            # Timer complete: mark as stopped so it doesn't re-trigger while standing still
            self._has_stopped = True
            self._stop_start_ms = None

        # 4. Reset trigger lock once all detection flags clear completely
        if not is_trigger_active:
            self._has_stopped = False

        return Command(left=0.0, right=0.0)