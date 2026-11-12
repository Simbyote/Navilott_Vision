"""Stop sign rule: stop at a stop line that has a sign, hold, then go.

Purpose:
    A stop line alone doesn't stop the robot. A sign voted present for a few
    frames in one sighting, seen shortly before the line came into view, does:
    brake until the wheels stop, hold, release.

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
# ...and only once Phase 3 has voted it present this many frames in one sighting, a
# sighting ending after SIGN_GAP_MS without a vote. The three false stops of 5 Nov (D2)
# were each a sign voted 1-3 frames beside a red lamp. The 8 real sign stops of the
# 28 Oct figure eight were voted 9-82 frames a sighting, but the 2-of-3 vote broke them
# into runs as short as 4 frames, so the frames are counted, not required in a row.
# 6 leaves 3 either side (resolution_plan.md F6)
SIGN_MIN_FRAMES = 6
# A gap this long without a vote ends a sighting (the analysis merged gaps up to 20 frames)
SIGN_GAP_MS = 1000

REASON_STOPPING, REASON_HOLD = "stop_sign_stopping", "stop_sign_hold"


class StopSignRule:
    """tracker is the one Navigation updates each frame. record holds the reason while braking."""
    def __init__(self, tracker: StopLineTracker, hold_ms: int = STOP_SIGN_HOLD_TIME_MS,
                 sign_memory_ms: int = SIGN_MEMORY_MS, sign_lead_ms: int | None = SIGN_LEAD_MS,
                 sign_min_frames: int = SIGN_MIN_FRAMES, sign_gap_ms: int = SIGN_GAP_MS) -> None:
        self.tracker = tracker
        self.hold_ms = hold_ms
        self.sign_memory_ms = sign_memory_ms
        self.sign_lead_ms = sign_lead_ms
        self.sign_min_frames = max(1, sign_min_frames)
        self.sign_gap_ms = sign_gap_ms
        self.reset()

    def reset(self) -> None:
        self._sign_ms: int | None = None        # when a sign that counts was last seen
        self._sighting = 0                      # frames voted in the current sighting
        self._voted_ms: int | None = None       # when the sign was last voted present, counting or not
        self._braking_ms: int | None = None     # when this stop started braking
        self._hold_ms: int | None = None        # when the hold started
        self.record: dict = {}

    def update(self, packet: EstimationPacket, held: bool = False) -> Command | None:
        """BRAKE while stopping or holding, else None. held is unused: a stop keeps its timing."""
        now = packet.timestamp_ms
        if packet.stop_sign_detected:
            if self._voted_ms is None or now - self._voted_ms > self.sign_gap_ms:
                self._sighting = 0                          # a new sighting
            self._sighting += 1
            self._voted_ms = now
            if self._sighting >= self.sign_min_frames:      # a brief ghost never counts
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
