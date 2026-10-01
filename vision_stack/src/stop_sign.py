"""Stop sign navigation module.

Purpose:
    Navigates stop signs and stop lines. Latches stop detection until the robot
    comes to a complete halt (wheel CPS = 0), then holds BRAKE for 2 seconds
    before allowing motion to resume.
"""

from src.estimation import EstimationPacket
from src.navigation import BRAKE, Command
from stop_line import StopLineNavigator, check_stop_line_trigger

STOP_SIGN_HOLD_TIME_MS = 2000  # 2 seconds stop duration
ZERO_SPEED_EPS = 1e-3         # Counts per second threshold for zero motion


class StopSignNavigator:
    """
    Navigator for stop sign and stop line handling with latched stop-line detection.

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

        # State tracking
        self._latched_stop_trigger: bool = False
        self._stop_start_ms: int | None = None
        self._has_stopped: bool = False

    def reset(self) -> None:
        """Resets the internal state and timer memory."""
        self._latched_stop_trigger = False
        self._stop_start_ms = None
        self._has_stopped = False
        self.stop_line_nav.reset()

    def update(self, packet: EstimationPacket) -> Command:
        """
        Processes this frame's EstimationPacket.

        Outputs:
            BRAKE while slowing down, latched, or during the 2-second stop duration;
            otherwise returns zero duty command.
        """
        # 1. Check current frame detection triggers (no threshold argument needed)
        stop_line_detected = check_stop_line_trigger(packet)
        stop_sign_detected = bool(packet.stop_sign_detected)

        # 2. Latch the detection once true until we come to a full stop
        if (stop_line_detected or stop_sign_detected) and not self._has_stopped:
            self._latched_stop_trigger = True

        # 3. Handle active latched stopping sequence
        if self._latched_stop_trigger:
            # Check if wheels have completely stopped
            wheels_stopped = (
                abs(packet.left_wheel_cps) < ZERO_SPEED_EPS
                and abs(packet.right_wheel_cps) < ZERO_SPEED_EPS
            )

            # Start timer only after the motors have reached zero speed
            if wheels_stopped and self._stop_start_ms is None:
                self._stop_start_ms = packet.timestamp_ms

            # Evaluate timer progress once counting has started
            if self._stop_start_ms is not None:
                elapsed_ms = packet.timestamp_ms - self._stop_start_ms
                if elapsed_ms >= self.hold_time_ms:
                    # Timer finished: complete stop cycle and release latch
                    self._has_stopped = True
                    self._latched_stop_trigger = False
                    self._stop_start_ms = None

            return BRAKE

        # 4. Reset lock once all visual triggers vanish from vision
        if not stop_line_detected and not stop_sign_detected:
            self._has_stopped = False

        return Command(left=0.0, right=0.0)