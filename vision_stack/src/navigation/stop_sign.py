"""Stop sign rule: at a stop line with a stop sign, come to a full stop, hold, then go.

Purpose:
    A stop line alone only says an intersection is coming. With a stop sign
    seen shortly before the line came into view (signs are often seen first,
    hence the memory), this rule brakes at the line until the wheels stop,
    holds, and lets go. All timing is on packet timestamps, so replays behave
    the same.

Main package:
    StopSignRule: update(packet, held) -> BRAKE while stopping or holding,
    None otherwise.

Flow:
    1. Remember when a stop sign was last seen.
    2. On the tracker's reached frame: a sign seen within SIGN_MEMORY_MS, and
       no more than SIGN_LEAD_MS before the line came into view, starts the stop.
    3. Brake until the wheels read stopped (or STOP_SETTLE_MAX_MS), hold
       STOP_SIGN_HOLD_TIME_MS, release.
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
# ... and only if it was seen no more than this long before the line itself
# came into view: a ghost sign (a leg or a chair in the background read as a
# sign) seen on the way to another stretch can't stop the robot at an
# unrelated line. A first guess: the sign shows up before the line does, by
# how much isn't measured
SIGN_LEAD_MS = 2000

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
                 sign_memory_ms: int = SIGN_MEMORY_MS, sign_lead_ms: int | None = SIGN_LEAD_MS) -> None:
        self.tracker = tracker
        self.hold_ms = hold_ms
        self.sign_memory_ms = sign_memory_ms
        self.sign_lead_ms = sign_lead_ms
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
            held: A higher-priority rule (the obstacle) decided this frame.
                Unused: a stop in progress keeps its timing, and it brakes
                either way.
        Outputs:
            BRAKE while stopping or holding; None otherwise.
        """
        now = packet.timestamp_ms
        if packet.stop_sign_detected:
            self._sign_ms = now
        if (self.tracker.reached and self._sign_ms is not None
                and now - self._sign_ms <= self.sign_memory_ms and self._for_this_line()):
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

    def _for_this_line(self) -> bool:
        """The remembered sign was seen no more than sign_lead_ms before the line reached came into view."""
        line_ms = self.tracker.seen_ms
        return self.sign_lead_ms is None or line_ms is None or self._sign_ms >= line_ms - self.sign_lead_ms
