"""Stop line navigator using vision state transitions.

Purpose:
    Monitors EstimationPacket for stop_line_detected. Latches when a stop line is 
    detected and issues BRAKE once it goes out of sight (transitioning from True to False).
"""

from src.estimation import EstimationPacket
from src.navigation import BRAKE, Command


def check_stop_line_trigger(packet: EstimationPacket) -> bool:
    """True if stop_line_detected is True in the packet."""
    return bool(packet.stop_line_detected)


class StopLineNavigator:
    """
    Navigator to issue stop commands after a stop line passes out of sight.
    """

    def __init__(self) -> None:
        self._saw_stop_line: bool = False
        self._line_lost: bool = False

    def reset(self) -> None:
        """Resets stop line vision state tracking."""
        self._saw_stop_line = False
        self._line_lost = False

    def update(self, packet: EstimationPacket) -> Command:
        """
        Evaluates the frame packet for stop line visibility transitions.

        Outputs:
            BRAKE once stop_line_detected transitions from True to False;
            otherwise neutral zero duty command.
        """
        currently_detected = check_stop_line_trigger(packet)

        # Track transition: True -> False
        if currently_detected:
            self._saw_stop_line = True
        elif self._saw_stop_line and not currently_detected:
            # Transition occurred: stop line was seen and is now out of sight
            self._line_lost = True

        # Output BRAKE after line goes out of sight
        if self._line_lost:
            return BRAKE

        return Command(left=0.0, right=0.0)