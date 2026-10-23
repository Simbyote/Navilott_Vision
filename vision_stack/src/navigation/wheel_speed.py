"""Wheel speed: drive each wheel at counts per second instead of a duty, whatever the battery reads.

Purpose:
    A duty is a share of the battery's voltage, so a turn tuned as duties on
    a fresh pack draws a smaller, slower arc on a tired one. The encoders
    measure what the wheels actually do, so this asks for a speed and finds
    the duty: a learned feedforward plus a PI trim; without encoders, the
    feedforward alone (docs/vision_pipeline/navigation_contract.md, "Wheel speed").

Main package:
    WheelSpeed: drive(left_cps, right_cps, packet) -> Command, from each
    wheel's learned gain (counts/s per unit duty) and the PI trim.

Flow, each frame:
    1. observe(): on a steady frame, each wheel's gain moves toward measured / duty.
    2. drive(): per wheel, the feedforward plus the PI trim when there are
       encoders; a driven wheel gets at least STALL_DUTY.
"""
from dataclasses import dataclass

from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import STALL_DUTY, Command
from src.utils import clamp

# Counts/s per unit duty: the three 2026-10-12 nav runs, 1511 steady frames at
# 11.6-11.8 V, fit cps = 3083 * duty + 59 (0.25 duty ~815 counts/s, 0.40
# ~1300, 0.73 ~2300). Only the starting point: observe() learns it per run
CPS_PER_DUTY = 3000.0
# PI trim, as duty per (counts/s) and per (counts/s x s), written per gain so
# they read as "duty per full-scale error". Simulated against a first-order
# motor (0.12 s) at 16-20 FPS with a frame's delay: a 15% weaker motor is
# back on target in ~0.5 s, with no overshoot to speak of. Tune on the mat
KP = 0.3 / CPS_PER_DUTY
KI = 3.0 / CPS_PER_DUTY
INTEGRAL_LIMIT = 0.25           # the most duty the integral can add or take
INTEGRATE_BAND = 0.3            # integrate only when |error| < this share of the target
# Learning the gain: a frame counts when both its command and the one before
# drove the wheel at least LEARN_MIN_DUTY, within LEARN_STEADY of each other.
# 0.05 per frame is ~1 s at 20 FPS. The gain stays within these limits of
# CPS_PER_DUTY: a spinning-up wheel or a lifted robot can't run it away
LEARN_MIN_DUTY = 0.30
LEARN_STEADY = 0.02
LEARN_ALPHA = 0.05
GAIN_LIMITS = (0.6, 1.4)
MAX_DT_MS = 500                 # longest frame gap integrated as one step, as Phase 3's max_dt_s


@dataclass
class _Wheel:
    gain: float
    integral: float = 0.0


class WheelSpeed:
    """
    Both wheels' speed control; see the module docstring.

    Inputs:
        cps_per_duty, kp, ki: See the constants above.

    gains: (left, right) learned counts/s per duty.
    encoders: True once a wheel has read moving while driven; until then
        drive() is feedforward only.
    record: drive()'s last: {"target", "measured", "duty", "gain"} as
        (left, right) pairs, and "closed" (the PI ran).
    """
    def __init__(self, cps_per_duty: float = CPS_PER_DUTY, kp: float = KP, ki: float = KI) -> None:
        self.cps_per_duty, self.kp, self.ki = cps_per_duty, kp, ki
        self.reset()

    def reset(self) -> None:
        """Forget the learned gains, the encoders and the PI state, as at the start of a run."""
        self._wheels = (_Wheel(self.cps_per_duty), _Wheel(self.cps_per_duty))
        self.encoders = False
        self._before: Command | None = None     # the command before the one observe() was last given
        self._last_ms: int | None = None
        self.record: dict = {}

    @property
    def gains(self) -> tuple[float, float]:
        return self._wheels[0].gain, self._wheels[1].gain

    def observe(self, cmd: Command | None, packet: EstimationPacket) -> None:
        """
        Learn from this packet's wheel speeds, which answer cmd (the command
        sent the frame before). None or a brake teaches nothing.
        """
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
        """Clear the PI's integral and clock: a new controlled stretch (a turn) starts."""
        for w in self._wheels:
            w.integral = 0.0
        self._last_ms = None

    def drive(self, left_cps: float, right_cps: float, packet: EstimationPacket) -> Command:
        """
        This frame's duties for the target speeds (counts/s, + = forward).

        Outputs:
            A Command within [-1, 1]; a wheel with a nonzero target gets at
            least STALL_DUTY, and a zero target drives nothing.
        """
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
