"""Stop line navigator using pixel distance threshold.

Purpose:
    Monitors EstimationPacket for detected stop lines. Returns BRAKE when a
    stop line is within 20 pixels of the bottom ROI; otherwise outputs a
    neutral command (0.0 duty).
"""

from src.estimation import EstimationPacket
from src.navigation import BRAKE, Command

STOP_LINE_THRESHOLD_PX = 20.0


def check_stop_line_trigger(
    packet: EstimationPacket, threshold_px: float = STOP_LINE_THRESHOLD_PX
) -> bool:
    """
    True once the robot's stop line distance in pixels reaches or falls below threshold.

    Inputs:
        packet: Uses stop_line_detected and stop_line_distance_px.
        threshold_px: Distance in pixels at or under which the robot must stop.
    """
    if packet.stop_line_detected and packet.stop_line_distance_px is not None:
        return packet.stop_line_distance_px <= threshold_px
    return False


class StopLineNavigator:
    """
    Navigator to issue stop commands upon reaching a pixel distance threshold for a stop line.

    Inputs:
        stop_line_threshold_px: Pixel distance threshold to trigger BRAKE.
    """

    def __init__(
        self,
        stop_line_threshold_px: float = STOP_LINE_THRESHOLD_PX,
    ) -> None:
        self.stop_line_threshold_px = stop_line_threshold_px

    def reset(self) -> None:
        """Stateless: every command is determined purely by the current packet."""
        pass

    def update(self, packet: EstimationPacket) -> Command:
        """
        Evaluates the frame packet for stop line pixel proximity.

        Outputs:
            BRAKE if the stop line pixel threshold is reached; otherwise zero duty.
        """
        if check_stop_line_trigger(packet, self.stop_line_threshold_px):
            return BRAKE

        return Command(left=0.0, right=0.0)