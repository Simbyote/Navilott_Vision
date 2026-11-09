"""Obstacle rule: brake for something in the path, wait until it's gone, then go on.

Purpose:
    The first rule, ahead of the stop sign. It releases only after no
    obstacle has been seen at any distance for a while, so an object the
    robot stopped just short of can't creep it closer one stop at a time.

Main package:
    ObstacleRule.update(packet) -> BRAKE while blocked, else None.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command

# cm from the front edge. The view's nearest floor is ~10.2 cm (2026-09-30
# table), so 12 leaves room for the coast and the vote's frame
OBSTACLE_STOP_CM = 12.0
# The same ~12 cm in rows above the view bottom, for runs without cm
OBSTACLE_STOP_PX = 24.0
# No obstacle at any distance for this long releases the brake (~20 frames)
OBSTACLE_CLEAR_MS = 1000

REASON_OBSTACLE, REASON_CLEARING = "obstacle", "obstacle_clearing"


def obstacle_near(packet: EstimationPacket, stop_cm: float = OBSTACLE_STOP_CM,
                  stop_px: float = OBSTACLE_STOP_PX) -> bool:
    """Clipped, or within stop_cm (stop_px when there are no cm)."""
    if not packet.obstacle_detected:
        return False
    if packet.obstacle_clipped:
        return True
    if packet.obstacle_distance_cm is not None:
        return packet.obstacle_distance_cm <= stop_cm
    return packet.obstacle_distance_px is not None and packet.obstacle_distance_px <= stop_px


class ObstacleRule:
    """blocked is True from the first brake until release. record holds the reason and distance, or the clear time."""
    def __init__(self, stop_cm: float = OBSTACLE_STOP_CM, stop_px: float = OBSTACLE_STOP_PX,
                 clear_ms: int = OBSTACLE_CLEAR_MS) -> None:
        self.stop_cm = stop_cm
        self.stop_px = stop_px
        self.clear_ms = clear_ms
        self.reset()

    def reset(self) -> None:
        self.blocked = False
        self._clear_since: int | None = None    # while blocked: when the obstacle was last gone from
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """BRAKE while blocked, else None. held is unused: this is the first rule."""
        now = packet.timestamp_ms
        if obstacle_near(packet, self.stop_cm, self.stop_px):
            self.blocked = True
        if not self.blocked:
            self.record = {}
            return None
        if packet.obstacle_detected:
            self._clear_since = None
            self.record = {"reason": REASON_OBSTACLE, "distance_cm": packet.obstacle_distance_cm,
                           "distance_px": packet.obstacle_distance_px}
            return BRAKE
        if self._clear_since is None:
            self._clear_since = now
        clear = now - self._clear_since
        if clear >= self.clear_ms:
            self.blocked, self._clear_since = False, None
            self.record = {}
            return None
        self.record = {"reason": REASON_CLEARING, "clear_ms": clear}
        return BRAKE
