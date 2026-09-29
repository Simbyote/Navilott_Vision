"""
presence.py  --  shared "is it fitted?" checks for the hardware tests.

One chassis has no motors, and not every chassis carries every peripheral, so
each hardware test runs the same ladder before touching the device:

    1. Can the host reach it at all?   pigpio daemon, I2C bus.   No  -> skip "unavailable"
    2. Is the device there?            a probe that gets an answer
                                       from the part itself.       No  -> skip "not found"
    3. Run the test.                   From here a failure is a fault on a
                                       chassis that has the part, never absence.

A probe must tell "absent" apart from "present but broken" where it can, so a
wiring fault still fails instead of hiding as a skip. Where the hardware gives
no way to tell (a pulled-down button reads the same with nothing attached),
the test says so in its docstring.

Skip reasons all start with the peripheral's name, so `pytest -rs` lists what
this chassis has and lacks.
"""
import time

import pytest


def pigpio_or_skip(name):
    """A connected pigpio.pi, or skip. The caller owns it and must stop() it."""
    try:
        import pigpio
    except ImportError as e:
        pytest.skip(f"{name} unavailable: pigpio not installed ({e})")
    pi = pigpio.pi()
    if not pi.connected:
        pytest.skip(f"{name} unavailable: pigpio daemon not reachable (sudo pigpiod)")
    return pi


def i2c_device_or_skip(name, addresses):
    """Skip unless one of addresses answers on the default I2C bus; returns the address found."""
    try:
        import board
        i2c = board.I2C()                       # the same singleton the Adafruit drivers use
    except Exception as e:
        pytest.skip(f"{name} unavailable: no I2C bus ({e})")
    deadline = time.monotonic() + 1.0
    while not i2c.try_lock():
        if time.monotonic() > deadline:
            pytest.skip(f"{name} unavailable: I2C bus stayed locked")
        time.sleep(0.01)
    try:
        found = i2c.scan()
    finally:
        i2c.unlock()
    hits = [a for a in addresses if a in found]
    if not hits:
        want = ", ".join(f"0x{a:02x}" for a in addresses)
        have = ", ".join(f"0x{a:02x}" for a in found) or "nothing"
        pytest.skip(f"{name} not found on this chassis: no reply at {want} (bus has {have})")
    return hits[0]


def tm1637_acks(pi, clk, dio):
    """
    True if a TM1637 on clk/dio acknowledges a byte.

    Bit-bangs one data-command byte (0x40, harmless: it only sets the address
    mode) and reads DIO on the 9th clock, where the chip pulls it low. With no
    chip, the pull-up leaves DIO high. pigpio's per-call latency keeps the
    clock far below the part's limit, so no delays are needed.
    """
    import pigpio
    for pin in (clk, dio):
        pi.set_mode(pin, pigpio.OUTPUT)
        pi.write(pin, 1)
    pi.write(dio, 0)                            # start: DIO falls while CLK is high
    for bit in range(8):                        # LSB first, DIO changes while CLK is low
        pi.write(clk, 0)
        pi.write(dio, (0x40 >> bit) & 1)
        pi.write(clk, 1)
    pi.write(clk, 0)
    pi.set_mode(dio, pigpio.INPUT)
    pi.set_pull_up_down(dio, pigpio.PUD_UP)
    pi.write(clk, 1)
    ack = pi.read(dio) == 0
    pi.write(clk, 0)
    pi.set_mode(dio, pigpio.OUTPUT)             # stop: DIO rises while CLK is high
    pi.write(dio, 0)
    pi.write(clk, 1)
    pi.write(dio, 1)
    return ack