"""Wheel stall guard: end the run when a wheel is driven hard and doesn't turn.

Purpose:
    The wheel-speed PI hides a failing motor by raising its duty, and lane
    keeping steers on into a wheel that has stopped: on 2026-10-28 the right
    N20 held full duty at 0 counts for 4 s, then 0.8 for 10 s more, as it
    died. A stalled motor has no back-EMF, so the whole supply heats its
    winding. Past STALL_MS of that, this brakes for good.

Main package:
    WheelStallGuard.observe(cmd, packet, wheels) -> tripped; record names the wheel.
"""
from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import Command
from src.navigation.wheel_speed import WheelSpeed

# Just above STALL_DUTY (0.25), where a loaded wheel may honestly not turn; under lane
# keeping's 0.40. Replayed over the 2026-10-28 figure eight, healthy wheels
# read stalled for 52 ms at most (a start from rest) at any floor from 0.26
MIN_DUTY = 0.30
# Turning slower than this share of what its duty gives is stalled. The dying
# turn read 0-15%; a healthy one 85-110%, wheel slips included (they spin faster)
MOVING_SHARE = 0.15
# The motor's ~0.12 s spin-up from rest, several times over
STALL_MS = 500
MAX_DT_MS = 500                 # longest packet gap counted, as Phase 3's max_dt_s

REASON_STALLED = "wheel_stalled"
OUTCOME_STALLED = "wheel_stalled"


class WheelStallGuard:
    """Armed once the encoders read (wheels.encoders), so a run without them, or without motors, never trips."""
    def __init__(self, min_duty: float = MIN_DUTY, moving_share: float = MOVING_SHARE,
                 stall_ms: int = STALL_MS) -> None:
        self.min_duty, self.moving_share, self.stall_ms = min_duty, moving_share, stall_ms
        self.reset()

    def reset(self) -> None:
        self.tripped = False
        self._stalled_ms = [0, 0]
        self._last_ms: int | None = None
        self.record: dict = {}

    def observe(self, cmd: Command | None, packet: EstimationPacket, wheels: WheelSpeed) -> bool:
        """packet's speeds answer cmd, the previous frame's command. True once tripped, until reset()."""
        if self.tripped:
            return True
        now = packet.timestamp_ms
        dt_ms = 0 if self._last_ms is None else min(max(now - self._last_ms, 0), MAX_DT_MS)
        self._last_ms = now
        if cmd is None or cmd.brake or not wheels.encoders:
            self._stalled_ms = [0, 0]
            return False
        measured = (packet.left_wheel_cps, packet.right_wheel_cps)
        for i, (side, duty, cps, gain) in enumerate(zip(("left", "right"), (cmd.left, cmd.right), measured,
                                                        wheels.gains)):
            expected = abs(duty) * gain
            forward = cps if duty > 0 else -cps        # turning the wrong way is not turning
            if abs(duty) >= self.min_duty and forward < self.moving_share * expected:
                self._stalled_ms[i] += dt_ms
            else:
                self._stalled_ms[i] = 0
            if self._stalled_ms[i] >= self.stall_ms:
                self.tripped = True
                self.record = {"reason": REASON_STALLED, "wheel": side, "duty": duty, "cps": cps,
                               "expected_cps": round(expected), "stalled_ms": self._stalled_ms[i]}
                return True
        return False
