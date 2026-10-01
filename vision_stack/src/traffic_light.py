"""Traffic light navigation module.

Purpose:
    Handles traffic light decisions upon stop line detection transitions.
    Once a stop line goes out of sight (transitioning True -> False), it reads
    packet.drive_state:
    - "stop": Stops (outputs BRAKE).
    - "caution": Stops (outputs BRAKE).
    - "go": Keeps going (outputs neutral 0.0 duty to let base driving take over).
"""

from src.estimation import EstimationPacket
from src.navigation import BRAKE, Command
from stop_line import StopLineNavigator, check_stop_line_trigger


class TrafficLightNavigator:
    """
    Navigator to evaluate traffic light drive states after crossing a stop line.
    """

    def __init__(
        self,
        stop_line_navigator: StopLineNavigator | None = None,
    ) -> None:
        self.stop_line_nav = stop_line_navigator or StopLineNavigator()
        self._saw_stop_line: bool = False
        self._line_lost: bool = False

    def reset(self) -> None:
        """Resets stop line vision state tracking."""
        self._saw_stop_line = False
        self._line_lost = False
        self.stop_line_nav.reset()

    def update(self, packet: EstimationPacket) -> Command:
        """
        Processes this frame's EstimationPacket.

        Outputs:
            BRAKE if the stop line transitions from True to False and drive_state is "stop" or "caution";
            otherwise returns neutral Command(0.0, 0.0).
        """
        currently_detected = check_stop_line_trigger(packet)

        # Track transition: True -> False
        if currently_detected:
            self._saw_stop_line = True
        elif self._saw_stop_line and not currently_detected:
            self._line_lost = True

        # Evaluate drive state after stop line goes out of sight
        if self._line_lost:
            drive_state = packet.drive_state.lower() if packet.drive_state else "stop"

            if drive_state in ("stop", "caution"):
                return BRAKE
            elif drive_state == "go":
                # Clear flags so the car can proceed and detect the next stop line
                self._saw_stop_line = False
                self._line_lost = False

        return Command(left=0.0, right=0.0)