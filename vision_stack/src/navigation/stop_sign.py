"""Stop sign rule: stop at a stop line that has a sign, hold, then go.

Purpose:
    A stop line alone doesn't stop the robot. A sign voted present for a few
    frames in a row, seen shortly before the line came into view, does: brake
    until the wheels stop, hold, release.

Main package:
    StopSignRule.update(packet) -> BRAKE while stopping or holding, else None.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.stop_line import StopLineTracker

STOP_SIGN_HOLD_TIME_MS = 2000
# Both wheels at or under this read stopped; cruising is ~1166 (2026-09-30)
STOPPED_CPS = 20.0
# Hold anyway after braking this long, in case the encoders never read stopped
STOP_SETTLE_MAX_MS = 1000
# A sign counts if seen this long before the line is reached (a guess)
SIGN_MEMORY_MS = 5000
# ...and no more than this before the line came into view, so a ghost sign
# on another stretch can't stop the robot (a guess)
SIGN_LEAD_MS = 2000
# ...and only once Phase 3 has voted it present this many frames in a row. The three
# false stops of 5 Nov (D2) were each a sign voted for 1-3 frames beside a red lamp;
# a real sign stays voted while it is in view, from ~20-60 cm out, tens of frames.
# 4 is the least that refuses all three; approach-log's sign_frames per approach
# says how far it can rise (requirements.md D2; resolution_plan.md F6)
SIGN_MIN_FRAMES = 4

REASON_STOPPING, REASON_HOLD = "stop_sign_stopping", "stop_sign_hold"


class StopSignRule:
    """tracker is the one Navigation updates each frame. record holds the reason while braking."""
    def __init__(self, tracker: StopLineTracker, hold_ms: int = STOP_SIGN_HOLD_TIME_MS,
                 sign_memory_ms: int = SIGN_MEMORY_MS, sign_lead_ms: int | None = SIGN_LEAD_MS,
                 sign_min_frames: int = SIGN_MIN_FRAMES) -> None:
        self.tracker = tracker
        self.hold_ms = hold_ms
        self.sign_memory_ms = sign_memory_ms
        self.sign_lead_ms = sign_lead_ms
        self.sign_min_frames = max(1, sign_min_frames)
        self.reset()

    def reset(self) -> None:
        self._sign_ms: int | None = None        # when a sign that counts was last seen
        self._sign_run = 0                      # frames in a row the sign has been voted present
        self._braking_ms: int | None = None     # when this stop started braking
        self._hold_ms: int | None = None        # when the hold started
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """BRAKE while stopping or holding, else None. held is unused: a stop keeps its timing."""
        now = packet.timestamp_ms
        self._sign_run = self._sign_run + 1 if packet.stop_sign_detected else 0
        if self._sign_run >= self.sign_min_frames:         # a brief ghost never counts
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
        line_ms = self.tracker.seen_ms
        return self.sign_lead_ms is None or line_ms is None or self._sign_ms >= line_ms - self.sign_lead_ms
