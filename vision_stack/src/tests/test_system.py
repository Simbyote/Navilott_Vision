"""
test_system.py  --  src/peripherals/system.py

system.py imports pigpio and tm1637 at module load, so software tests swap in
fakes before importing it and patch time, so the countdown and debounce run
instantly and the display throttle can be stepped by hand.

--software  Button debounce, countdown, MM:SS formatting, throttling and
            cleanup, against fake pigpio / tm1637. No GPIO.
--hardware  Opens the real pigpio daemon and display: shows "rdy", runs the
            countdown, ticks the clock for a few seconds, reads the button
            once and cleans up. Watch the display; the button is not pressed
            (wait_for_start() would block for a person).
"""
import importlib
import sys
import types

import pytest

from src.params import GPIO_DISPLAY_CLK, GPIO_DISPLAY_DIO, GPIO_START_BUTTON

SYSTEM_MODULE = "src.peripherals.system"


class FakePi:
    """pigpio.pi stand-in: records pin setup and plays back a script of button reads."""
    def __init__(self, connected=True, reads=()):
        self.connected = connected
        self.reads = list(reads)
        self.modes, self.pulls = {}, {}
        self.stopped = False

    def set_mode(self, pin, mode):
        self.modes[pin] = mode

    def set_pull_up_down(self, pin, pud):
        self.pulls[pin] = pud

    def read(self, pin):
        return self.reads.pop(0) if self.reads else 1     # held once the script runs out

    def stop(self):
        self.stopped = True


class FakeDisplay:
    """tm1637.TM1637 stand-in: records every write in order."""
    def __init__(self, clk, dio):
        self.pins = (clk, dio)
        self.writes = []
        self.fail_show = False

    def brightness(self, level):
        self.level = level

    def show(self, text):
        if self.fail_show:
            raise OSError("simulated display error")
        self.writes.append(text)

    def numbers(self, minutes, seconds):
        self.writes.append((minutes, seconds))


class FakeClock:
    """time stand-in: sleep() advances monotonic() and is recorded, so nothing waits."""
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


@pytest.fixture
def env(monkeypatch):
    """Imports system.py against fake pigpio / tm1637 / time; returns (module, pi, clock)."""
    pi = FakePi()
    pigpio = types.ModuleType("pigpio")
    pigpio.INPUT, pigpio.PUD_DOWN = "INPUT", "PUD_DOWN"
    pigpio.pi = lambda: pi
    tm1637 = types.ModuleType("tm1637")
    tm1637.TM1637 = FakeDisplay
    monkeypatch.setitem(sys.modules, "pigpio", pigpio)
    monkeypatch.setitem(sys.modules, "tm1637", tm1637)
    monkeypatch.delitem(sys.modules, SYSTEM_MODULE, raising=False)
    mod = importlib.import_module(SYSTEM_MODULE)
    clock = FakeClock()
    monkeypatch.setattr(mod, "time", clock)
    yield mod, pi, clock
    sys.modules.pop(SYSTEM_MODULE, None)        # the next import gets the real (or fresh fake) modules


@pytest.mark.software
def test_missing_pigpio_daemon_raises_with_the_fix(env):
    mod, pi, _ = env
    pi.connected = False
    with pytest.raises(RuntimeError, match="sudo pigpiod"):
        mod.System()


@pytest.mark.software
def test_pins_come_from_params(env):
    mod, pi, _ = env
    s = mod.System()
    assert pi.modes[GPIO_START_BUTTON] == "INPUT" and pi.pulls[GPIO_START_BUTTON] == "PUD_DOWN"
    assert s._display.pins == (GPIO_DISPLAY_CLK, GPIO_DISPLAY_DIO)


@pytest.mark.software
def test_wait_for_start_shows_ready_and_returns_on_a_held_press(env):
    mod, pi, clock = env
    s = mod.System()
    pi.reads = [0, 0, 1, 1]
    s.wait_for_start()
    assert s._display.writes == ["rdy "]
    assert pi.reads == []                       # two idle polls, then press + confirm


@pytest.mark.software
def test_wait_for_start_ignores_a_bounce_shorter_than_the_debounce(env):
    mod, pi, clock = env
    s = mod.System()
    pi.reads = [1, 0, 0, 1, 1]                  # high, gone 50 ms later, then a real press
    s.wait_for_start()
    assert pi.reads == []
    assert clock.sleeps.count(0.05) == 2        # both highs were re-checked after the debounce hold


@pytest.mark.software
def test_countdown_shows_five_to_one_then_blanks(env):
    mod, _, clock = env
    s = mod.System()
    s.run_countdown()
    assert s._display.writes == ["  5 ", "  4 ", "  3 ", "  2 ", "  1 ", "    "]
    assert clock.sleeps == [1.0] * 5


@pytest.mark.software
@pytest.mark.parametrize("elapsed, shown", [(0, (0, 0)), (59.9, (0, 59)), (125.4, (2, 5)),
                                            (3599, (59, 59)), (6000, (99, 0))])
def test_update_display_formats_mm_ss_and_caps_minutes(env, elapsed, shown):
    mod, _, _ = env
    s = mod.System()
    s.update_display(elapsed)
    assert s._display.writes == [shown]


@pytest.mark.software
def test_update_display_writes_at_most_once_per_interval(env):
    mod, _, clock = env
    s = mod.System()
    s.update_display(10)
    clock.now += 0.5
    s.update_display(10.5)                      # inside the interval: skipped
    clock.now += mod._DISPLAY_UPDATE_INTERVAL_S
    s.update_display(11.5)
    assert s._display.writes == [(0, 10), (0, 11)]


@pytest.mark.software
def test_final_time_is_written_even_inside_the_throttle_window(env):
    mod, _, _ = env
    s = mod.System()
    s.update_display(10)
    s.show_final_time(10.9)                     # same second, no clock advance
    assert s._display.writes == [(0, 10), (0, 10)]


@pytest.mark.software
def test_final_time_caps_minutes(env):
    mod, _, _ = env
    s = mod.System()
    s.show_final_time(100 * 60 + 7)
    assert s._display.writes == [(99, 7)]


@pytest.mark.software
def test_cleanup_blanks_the_display_and_releases_pigpio(env):
    mod, pi, _ = env
    s = mod.System()
    s.cleanup()
    assert s._display.writes == ["    "] and pi.stopped


@pytest.mark.software
def test_cleanup_still_releases_pigpio_if_the_display_fails(env):
    # cleanup runs in the finally block; a dead display must not leak the daemon handle
    mod, pi, _ = env
    s = mod.System()
    s._display.fail_show = True
    s.cleanup()
    assert pi.stopped


@pytest.mark.hardware
def test_system_characterization(artifacts):
    import time
    try:
        mod = importlib.import_module(SYSTEM_MODULE)
        s = mod.System()
    except Exception as e:                      # no pigpio/tm1637, or the daemon isn't running
        pytest.skip(f"system peripherals unavailable: {e}")

    try:
        s._display.show("rdy ")
        time.sleep(1.0)
        s.run_countdown()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            s.update_display(time.perf_counter() - t0)
            time.sleep(0.05)                    # roughly frame-paced, so the throttle is exercised
        s.show_final_time(time.perf_counter() - t0)
        button = s._pi.read(GPIO_START_BUTTON)
        time.sleep(2.0)                         # leave the final time up long enough to read
    finally:
        s.cleanup()

    artifacts.json("summary.json", {
        "pins": {"display_clk": GPIO_DISPLAY_CLK, "display_dio": GPIO_DISPLAY_DIO,
                 "start_button": GPIO_START_BUTTON},
        "button_level_at_rest": button,         # expect 0: pull-down, not pressed
    })
    assert button == 0, "start button reads high at rest: check the pull-down and wiring"