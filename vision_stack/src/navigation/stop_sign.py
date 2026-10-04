"""Stop sign rule: at a stop line with a stop sign, come to a full stop, hold, then go.

Purpose:
    A stop line alone only says an intersection is coming; it stops the
    robot only with a stop sign (or a red light, traffic_light.py). When the
    shared StopLineTracker says the robot has reached the line and a stop
    sign was seen within SIGN_MEMORY_MS before that, this rule brakes until
    the wheels have stopped, holds the brake STOP_SIGN_HOLD_TIME_MS, and
    lets go. Signs are often seen before the line comes into view, hence the
    memory. All timing is on packet timestamps, so replays behave the same.

Main package:
    StopSignRule: update(packet, held) -> BRAKE while stopping or holding,
        None otherwise (navigation.Navigation falls through to the next rule).

Flow:
    1. Remember when a stop sign was last seen.
    2. On the tracker's reached frame: a sign seen recently starts the stop.
    3. Brake; once the wheels read stopped (or STOP_SETTLE_MAX_MS has passed),
       hold STOP_SIGN_HOLD_TIME_MS more, then release.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.stop_line import StopLineTracker

# How long to stay stopped at a stop sign (2026-09-30)
STOP_SIGN_HOLD_TIME_MS = 2000
# Wheels at or under this many counts/s each are stopped. Cruising at 0.40
# duty is ~1166 counts/s (2026-09-30 trials); encoder jitter at rest must
# not keep the hold from starting
STOPPED_CPS = 20.0
# Start the hold anyway after braking this long: the wheels never read
# stopped without encoders (they read 0.0 then, which starts it at once) or
# if they jitter above STOPPED_CPS. The brake stops the robot in ~0.15 s
STOP_SETTLE_MAX_MS = 1000
# A stop sign seen this long before the robot reaches the line still counts.
# The sign comes into view before the line does; not yet measured on the course
SIGN_MEMORY_MS = 5000

REASON_STOPPING, REASON_HOLD = "stop_sign_stopping", "stop_sign_hold"


class StopSignRule:
    """
    Full stop at a stop line that has a stop sign.

    Inputs:
        tracker: The StopLineTracker navigation.Navigation updates each frame.
        hold_ms, sign_memory_ms: See the constants above.

    record: {"reason": REASON_STOPPING or REASON_HOLD} while it brakes; {} otherwise.
    """
    def __init__(self, tracker: StopLineTracker, hold_ms: int = STOP_SIGN_HOLD_TIME_MS,
                 sign_memory_ms: int = SIGN_MEMORY_MS) -> None:
        self.tracker = tracker
        self.hold_ms = hold_ms
        self.sign_memory_ms = sign_memory_ms
        self.reset()

    def reset(self) -> None:
        """Forget any sign seen and any stop in progress."""
        self._sign_ms: int | None = None
        self._braking_ms: int | None = None     # when this stop started braking
        self._hold_ms: int | None = None        # when the hold started
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """
        This frame's say.

        Inputs:
            packet: This frame's packet; the tracker has already seen it.
            held: A higher-priority rule decided this frame (unused: this is
                the first rule).
        Outputs:
            BRAKE while stopping or holding; None otherwise.
        """
        now = packet.timestamp_ms
        if packet.stop_sign_detected:
            self._sign_ms = now
        if (self.tracker.reached and self._sign_ms is not None
                and now - self._sign_ms <= self.sign_memory_ms):
            self._braking_ms, self._sign_ms = now, None     # this sign is used up
        if self._braking_ms is None:
            self.record = {}
            return None

        if self._hold_ms is None:
            stopped = (abs(packet.left_wheel_cps) <= STOPPED_CPS and abs(packet.right_wheel_cps) <= STOPPED_CPS)
            if stopped or now - self._braking_ms >= STOP_SETTLE_MAX_MS:
                self._hold_ms = now
        if self._hold_ms is not None and now - self._hold_ms >= self.hold_ms:
            self._braking_ms = self._hold_ms = None
            self.record = {}
            return None
        self.record = {"reason": REASON_STOPPING if self._hold_ms is None else REASON_HOLD}
        return BRAKE
