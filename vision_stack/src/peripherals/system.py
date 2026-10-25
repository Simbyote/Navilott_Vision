"""Human interface: the start button and the TM1637 run-time display.

Purpose:
    Holds the robot until the button is pressed, counts down, shows the
    elapsed time during the run and the final time after. GPIO goes through
    the pigpio daemon; the pins are in params.py.

Main package:
    System: the display and the button for one run.
"""

import time
import logging
import pigpio
import tm1637

from src.os_threads import name_pigpio_threads
from src.params import GPIO_DISPLAY_CLK, GPIO_DISPLAY_DIO, GPIO_START_BUTTON

log = logging.getLogger("system")

_DISPLAY_UPDATE_INTERVAL_S = 1.0   # each write is bit-banged, ~1-2 ms


class System:
    """RuntimeError if the pigpio daemon isn't reachable. The TM1637 library bit-bangs its own timing."""

    def __init__(self) -> None:
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError(
                "pigpio daemon not reachable. Run: sudo pigpiod"
            )
        # The linkers open this after drive.py has named its threads
        name_pigpio_threads()

        self._pi.set_mode(GPIO_START_BUTTON, pigpio.INPUT)
        self._pi.set_pull_up_down(GPIO_START_BUTTON, pigpio.PUD_DOWN)

        self._display = tm1637.TM1637(clk=GPIO_DISPLAY_CLK, dio=GPIO_DISPLAY_DIO)
        self._display.brightness(2)   # of 0-7; readable indoors

        self._last_display_update: float = 0.0
        self._press_since: float | None = None      # button_pressed(): when the pin went high
        self._press_latched = True                  # a press held from wait_for_start() doesn't count

        log.info(
            "System: button GPIO %d, display CLK %d / DIO %d",
            GPIO_START_BUTTON, GPIO_DISPLAY_CLK, GPIO_DISPLAY_DIO,
        )

    def wait_for_start(self, text: str = "rdy ") -> None:
        """
        Blocks until a press, showing text (4 characters). A press counts if the pin is still high 50 ms later.
        """
        self.show_text(text)
        log.info("System: waiting for start button (GPIO %d)...", GPIO_START_BUTTON)

        while True:
            if self._pi.read(GPIO_START_BUTTON):
                time.sleep(0.05)                   # debounce hold
                if self._pi.read(GPIO_START_BUTTON):
                    log.info("System: start button pressed.")
                    break
            time.sleep(0.01)

    def button_pressed(self) -> bool:
        """
        Non-blocking: True once per press, debounced like wait_for_start(). A button still
        held from wait_for_start() counts only after it's released and pressed again.
        """
        now = time.monotonic()
        if not self._pi.read(GPIO_START_BUTTON):
            self._press_since = None
            self._press_latched = False
            return False
        if self._press_latched:
            return False
        if self._press_since is None:
            self._press_since = now
            return False
        if now - self._press_since >= 0.05:
            self._press_latched = True
            return True
        return False

    def show_text(self, text: str) -> None:
        """Padded or cut to 4 digits; letters as the 7-segment font allows."""
        self._display.show(f"{text:<4}"[:4])

    def run_countdown(self) -> None:
        """5-4-3-2-1, one per second, then blank. Blocks 5 s."""
        log.info("System: starting countdown...")
        for count in range(5, 0, -1):
            self._display.show(f"  {count} ")
            log.info("System: countdown %d", count)
            time.sleep(1.0)

        self._display.show("    ")
        log.info("System: GO")

    def update_display(self, elapsed_s: float) -> None:
        """MM:SS, minutes capped at 99. Call every frame; it writes once a second."""
        now = time.monotonic()
        if now - self._last_display_update < _DISPLAY_UPDATE_INTERVAL_S:
            return
        self._last_display_update = now

        minutes = int(elapsed_s) // 60
        seconds = int(elapsed_s) % 60

        minutes = min(minutes, 99)

        self._display.numbers(minutes, seconds)
        log.debug("System: display update %02d:%02d", minutes, seconds)

    def show_final_time(self, elapsed_s: float) -> None:
        minutes = min(int(elapsed_s) // 60, 99)
        seconds = int(elapsed_s) % 60
        self._display.numbers(minutes, seconds)
        log.info("System: final time %02d:%02d", minutes, seconds)

    def cleanup(self, blank: bool = True) -> None:
        """
        blank=False leaves the last screen up; the TM1637 holds it while powered. Safe if the display fails.
        """
        if blank:
            try:
                self._display.show("    ")
            except Exception:
                pass
        self._pi.stop()
        log.info("System: cleanup complete.")