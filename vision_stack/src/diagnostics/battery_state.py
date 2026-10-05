"""
Battery state machine (pure logic, no hardware imports)
========================================================
E1.08 Magnetronics

Turns a stream of battery voltage readings into OK / WARNING / CRITICAL,
with debouncing and hysteresis so motor-load sag doesn't cause false
alarms or flicker.

Rules:
    - Escalation needs `confirm` consecutive readings at/below a threshold.
    - WARNING clears only after `confirm` consecutive readings at/above
      (warning + hysteresis).
    - CRITICAL is latched for the rest of the run (never clears).

Kept separate from battery.py so it can be unit-tested off the Pi.
"""

from enum import IntEnum
from typing import Optional


class BatteryState(IntEnum):
    OK = 0
    WARNING = 1
    CRITICAL = 2


class BatteryStateMachine:
    def __init__(
        self,
        warning_v: float = 10.5,
        critical_v: float = 9.9,
        hysteresis_v: float = 0.3,
        confirm: int = 3,
    ):
        self.warning_v = warning_v
        self.critical_v = critical_v
        self.hysteresis_v = hysteresis_v
        self.confirm = confirm

        self.state = BatteryState.OK
        self._below_warn = 0
        self._below_crit = 0
        self._above_clear = 0

    def update(self, volts: float) -> Optional[BatteryState]:
        """Feed one reading. Returns the new state if it changed, else None."""
        if self.state == BatteryState.CRITICAL:
            return None  # latched

        self._below_crit = self._below_crit + 1 if volts <= self.critical_v else 0
        self._below_warn = self._below_warn + 1 if volts <= self.warning_v else 0
        clear_v = self.warning_v + self.hysteresis_v
        self._above_clear = self._above_clear + 1 if volts >= clear_v else 0

        new = self.state
        if self._below_crit >= self.confirm:
            new = BatteryState.CRITICAL
        elif self.state == BatteryState.OK and self._below_warn >= self.confirm:
            new = BatteryState.WARNING
        elif self.state == BatteryState.WARNING and self._above_clear >= self.confirm:
            new = BatteryState.OK

        if new != self.state:
            self.state = new
            return new
        return None
