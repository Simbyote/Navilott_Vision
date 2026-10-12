"""Obstacle rule: something in the robot's path within reach: brake, wait until it's gone, then go on.

Purpose:
    The robot's first rule, above the stop sign: nothing else it does is
    worth hitting something for. When Phase 3's obstacle vote says there is
    something in the path and its base is within OBSTACLE_STOP_CM (or, with
    no cm, OBSTACLE_STOP_PX rows of the view bottom; or its base is already
    out of view), this rule brakes. It keeps braking until no obstacle has
    been seen at all, at any distance, for OBSTACLE_CLEAR_MS: an object the
    robot stopped just short of reads a little beyond the stop distance, and
    releasing on "not near" would creep the robot into it a stop at a time.
    Then it says nothing, and lane keeping (or the rule below) drives on.

    The distance comes from the stop-line table (perception/obstacle.py),
    measured from the robot's front edge. The camera's view starts ~10.2 cm
    ahead of it, so "within 10 cm" can't be seen as a distance: the stop
    distance sits just beyond the view's edge instead, so the brake starts
    while the base is still in view, and the coast past a zero duty
    (~0.15-0.18 s, 2026-09-30) plus a frame or two of vote ends the robot
    near 10 cm. A clipped obstacle (base out of view) is always within it.

    While it brakes, the rules below see held=True: the end of course
    doesn't count a lane hidden behind the obstacle as the course ending,
    and an intersection crossing's timers wait. All timing is on packet
    timestamps, so replays behave the same.

Main package:
    ObstacleRule: update(packet, held) -> BRAKE while an obstacle is near or
        hasn't been clear long enough, None otherwise. blocked: it is braking.

Flow:
    1. near = an obstacle voted, and clipped or within the stop distance.
    2. near -> blocked; the clear timer resets.
    3. blocked and no obstacle at all this frame -> the clear timer runs;
       OBSTACLE_CLEAR_MS of it releases. An obstacle seen at any distance
       resets it.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command

# Brake for an obstacle whose base is this near, in the stop-line table's cm
# from the robot's front edge. The view's nearest floor is ~10.2 cm (the
# table's row 0, 2026-09-30); 12 cm leaves the coast and the vote's frame
# to end near 10. Measure the stop on the mat and tune
OBSTACLE_STOP_CM = 12.0
# The same distance in rows above the view bottom, for when there are no cm
# (no ground homography or stop-line table): ~12 cm on the 2026-09-30 table
OBSTACLE_STOP_PX = 24.0
# No obstacle seen, at any distance, for this long releases the brake
# (~20 frames at 20 FPS): a hand pulled away, a box lifted off the mat
OBSTACLE_CLEAR_MS = 1000

REASON_OBSTACLE, REASON_CLEARING = "obstacle", "obstacle_clearing"


def obstacle_near(packet: EstimationPacket, stop_cm: float = OBSTACLE_STOP_CM,
                  stop_px: float = OBSTACLE_STOP_PX) -> bool:
    """The packet's obstacle is close enough to stop for: clipped, or within stop_cm (stop_px with no cm)."""
    if not packet.obstacle_detected:
        return False
    if packet.obstacle_clipped:
        return True
    if packet.obstacle_distance_cm is not None:
        return packet.obstacle_distance_cm <= stop_cm
    return packet.obstacle_distance_px is not None and packet.obstacle_distance_px <= stop_px


class ObstacleRule:
    """
    Brake for an obstacle within reach; go on once it has been gone a while.

    Inputs:
        stop_cm, stop_px, clear_ms: See the constants above.

    blocked: True from the frame it brakes until it releases.
    record: {"reason": REASON_OBSTACLE, "distance_cm", "distance_px"} while
        an obstacle is in view, {"reason": REASON_CLEARING, "clear_ms"} while
        it waits out the clear time; {} otherwise.
    """
    def __init__(self, stop_cm: float = OBSTACLE_STOP_CM, stop_px: float = OBSTACLE_STOP_PX,
                 clear_ms: int = OBSTACLE_CLEAR_MS) -> None:
        self.stop_cm = stop_cm
        self.stop_px = stop_px
        self.clear_ms = clear_ms
        self.reset()

    def reset(self) -> None:
        """Not blocked."""
        self.blocked = False
        self._clear_since: int | None = None    # while blocked: when the obstacle was last gone from
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """
        This frame's say.

        Inputs:
            packet: This frame's packet.
            held: A higher-priority rule decided this frame (unused: this is
                the first rule).
        Outputs:
            BRAKE while blocked; None otherwise.
        """
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
