"""Battery for a run: open the monitor if it's fitted, judge the pack before the start, watch it during.

Purpose:
    The course run (src.main) and the navigation linker check the pack the
    same way, so the rule lives here once. At rest, a pack under CRITICAL
    refuses a run that would drive the motors; under WARNING it only warns,
    so a sagging or misread pack can't lock the robot out on the day. A
    missing or unreadable ADC never stops a run: it's said once.

Main package:
    The verdict: preflight() -> GO, LOW or REFUSE, and why.

Flow:
    1. open_battery(): a Power, or None without the ADC.
    2. preflight() before the start screen; start() at GO.
    3. The caller ends the run once should_stop(); cleanup() at the end.
"""
from src.peripherals.battery import Power

GO, LOW, REFUSE = "ok", "low", "refuse"
CODE_BATTERY = "Lo-b"           # as diagnostics.battery_check shows it
# What a missing or unreadable ADC raises: no Blinka off the Pi, no device
# on the bus, a bad read
_OPEN_ERRORS = (ImportError, OSError, ValueError, RuntimeError)


def open_battery(say=print, factory=Power):
    """The battery monitor, or None (said once) when the ADC can't be opened."""
    try:
        return factory()
    except _OPEN_ERRORS as exc:
        say(f"battery  not monitored ({exc!r})")
        return None


def preflight(power, motors_on: bool, say=print) -> tuple[str, float | None]:
    """
    The resting pack, motors idle: (verdict, volts).

    Outputs:
        REFUSE: at or under VOLTAGE_CRITICAL with the motors on. LOW: under
        VOLTAGE_WARNING, or critical on a dry run (nothing moves). GO
        otherwise, and when the ADC can't be read (volts None), which is
        said: the run goes on unmonitored.
    """
    ok, volts = power.preflight()
    if not ok and volts == 0.0:                     # no reading at all: the ADC, not the pack
        say("battery  unreadable: running without the battery check")
        return GO, None
    if volts <= power.VOLTAGE_CRITICAL:
        if motors_on:
            say(f"battery  {volts:.2f} V: at or under {power.VOLTAGE_CRITICAL} V (critical). "
                "Charge or swap the pack; not starting")
            return REFUSE, volts
        say(f"battery  {volts:.2f} V: critical, but the motors are off, so carrying on")
        return LOW, volts
    if not ok:
        say(f"battery  {volts:.2f} V: under {power.VOLTAGE_WARNING} V (low). Starting; "
            "charge it after this run")
        return LOW, volts
    say(f"battery  {volts:.2f} V ok")
    return GO, volts


def start(power, say=print) -> bool:
    """Start monitoring; False (said) if the first read fails, so the caller runs without it."""
    try:
        power.start_monitoring()
        return True
    except _OPEN_ERRORS as exc:
        say(f"battery  monitoring didn't start ({exc!r}): running without it")
        return False
