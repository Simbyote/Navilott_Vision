"""Wheel speed: drive each wheel at counts per second instead of a duty, whatever the battery reads.

Purpose:
    A duty is a share of the battery's voltage, so a turn tuned on a fresh
    pack draws a slower, smaller arc on a tired one. This asks for a speed
    and finds the duty: a learned feedforward plus a PI trim from the
    encoders, or the feedforward alone without them.

Main package:
    WheelSpeed.drive(left_cps, right_cps, packet) -> Command.
"""
from dataclasses import dataclass

from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import STALL_DUTY, Command
from src.utils import clamp

# Starting gain, counts/s per duty: fit over the 2026-10-12 runs at 11.6-11.8 V.
# observe() learns it per run
CPS_PER_DUTY = 3000.0
# Simulated on a 0.12 s motor at 16-20 FPS: a 15% weaker motor is back on
# target in ~0.5 s without overshoot. Not yet tuned on the mat
KP = 0.3 / CPS_PER_DUTY
KI = 3.0 / CPS_PER_DUTY
INTEGRAL_LIMIT = 0.25           # duty
INTEGRATE_BAND = 0.3            # integrate only when |error| is under this share of the target
# The gain learns only on steady frames (two commands within LEARN_STEADY,
# both at least LEARN_MIN_DUTY), over ~1 s, and stays within GAIN_LIMITS of
# CPS_PER_DUTY so a lifted robot can't run it away
LEARN_MIN_DUTY = 0.30
LEARN_STEADY = 0.02
LEARN_ALPHA = 0.05
GAIN_LIMITS = (0.6, 1.4)
MAX_DT_MS = 500


@dataclass
class _Wheel:
    gain: float
    integral: float = 0.0


class WheelSpeed:
    """encoders turns True once a driven wheel reads moving; until then drive() is feedforward only."""
    def __init__(self, cps_per_duty: float = CPS_PER_DUTY, kp: float = KP, ki: float = KI) -> None:
        self.cps_per_duty, self.kp, self.ki = cps_per_duty, kp, ki
        self.reset()

    def reset(self) -> None:
        self._wheels = (_Wheel(self.cps_per_duty), _Wheel(self.cps_per_duty))
        self.encoders = False
        self._before: Command | None = None     # the command before the one observe() was last given
        self._last_ms: int | None = None
        self.record: dict = {}

    @property
    def gains(self) -> tuple[float, float]:
        return self._wheels[0].gain, self._wheels[1].gain

    def observe(self, cmd: Command | None, packet: EstimationPacket) -> None:
        """Learn the gain from this packet's speeds, which answer cmd, the previous frame's command."""
        before, self._before = self._before, cmd
        if cmd is None or cmd.brake:
            return
        lo, hi = (f * self.cps_per_duty for f in GAIN_LIMITS)
        measured = (packet.left_wheel_cps, packet.right_wheel_cps)
        for i, (wheel, duty, cps) in enumerate(zip(self._wheels, (cmd.left, cmd.right), measured)):
            if abs(duty) >= STALL_DUTY and cps != 0.0:
                self.encoders = True
            prev = None if before is None or before.brake else (before.left, before.right)[i]
            steady = prev is not None and abs(duty - prev) <= LEARN_STEADY
            if steady and abs(duty) >= LEARN_MIN_DUTY and cps * duty > 0.0:
                wheel.gain += LEARN_ALPHA * (clamp(cps / duty, lo, hi) - wheel.gain)

    def start(self) -> None:
        """Clear the PI state at the start of a turn."""
        for w in self._wheels:
            w.integral = 0.0
        self._last_ms = None

    def drive(self, left_cps: float, right_cps: float, packet: EstimationPacket) -> Command:
        """Duties for the target counts/s. A nonzero target gets at least STALL_DUTY; zero drives nothing."""
        now = packet.timestamp_ms
        dt_s = 0.0 if self._last_ms is None else min(max(now - self._last_ms, 0), MAX_DT_MS) / 1000.0
        self._last_ms = now
        measured = (packet.left_wheel_cps, packet.right_wheel_cps)
        duties = []
        for wheel, target, cps in zip(self._wheels, (left_cps, right_cps), measured):
            if target == 0.0:
                wheel.integral = 0.0
                duties.append(0.0)
                continue
            duty = target / wheel.gain
            if self.encoders:
                err = target - cps
                if abs(err) < INTEGRATE_BAND * abs(target):
                    wheel.integral = clamp(wheel.integral + self.ki * err * dt_s, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                duty += self.kp * err + wheel.integral
            duty = clamp(duty, -1.0, 1.0)
            if abs(duty) < STALL_DUTY:
                duty = STALL_DUTY if target > 0 else -STALL_DUTY
            duties.append(round(duty, 4))
        self.record = {"target": (left_cps, right_cps), "measured": measured, "duty": tuple(duties),
                       "gain": tuple(round(g, 1) for g in self.gains), "closed": self.encoders}
        return Command(duties[0], duties[1])
