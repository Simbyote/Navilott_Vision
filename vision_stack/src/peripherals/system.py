"""Human interface: the start button and the TM1637 run-time display.

Purpose:
    Holds the robot until the start button is pressed, counts down, shows
    elapsed MM:SS during the run and the final time after it, then releases
    the GPIO. All GPIO goes through the pigpio daemon, so the two share pins 
    without conflict. Requires the tm1637 package and a running daemon 
    (sudo pigpiod). Pins come from params.py, which follows Product Spec GPIO 
    Table 7.

Main package:
    System: owns the display and the button for one run.

Flow:
    1. wait_for_start(): show "rdy" and block until a debounced press.
    2. run_countdown(): 5-4-3-2-1, one per second, then blank.
    3. update_display(elapsed_s), every frame; throttled internally.
    4. show_final_time(elapsed_s) once the loop exits.
    5. cleanup(), in the finally block.
"""

import time
import logging
import pigpio
import tm1637

from src.params import GPIO_DISPLAY_CLK, GPIO_DISPLAY_DIO, GPIO_START_BUTTON

log = logging.getLogger("system")

_DISPLAY_UPDATE_INTERVAL_S = 1.0   # MM:SS only changes once a second, and each write is bit-banged (~1-2 ms)


class System:
    """
    Owns the TM1637 display and start button for the Navilott robot.

    The TM1637 library bit-bangs its own timing, independent of pigpio.

    Raises:
        RuntimeError: On construction, if the pigpio daemon isn't reachable.
    """

    def __init__(self) -> None:
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError(
                "pigpio daemon not reachable. Run: sudo pigpiod"
            )

        self._pi.set_mode(GPIO_START_BUTTON, pigpio.INPUT)
        self._pi.set_pull_up_down(GPIO_START_BUTTON, pigpio.PUD_DOWN)

        self._display = tm1637.TM1637(clk=GPIO_DISPLAY_CLK, dio=GPIO_DISPLAY_DIO)
        self._display.brightness(2)   # 0 (dim) - 7 (max); 2 is readable indoors

        self._last_display_update: float = 0.0

        log.info(
            "System: button GPIO %d, display CLK %d / DIO %d",
            GPIO_START_BUTTON, GPIO_DISPLAY_CLK, GPIO_DISPLAY_DIO,
        )

    def wait_for_start(self) -> None:
        """
        Block until the start button is pressed, showing "rdy" meanwhile.

        A press counts only if the pin is still high 50 ms later, which
        debounces contact bounce.
        """
        self._display.show("rdy ")         # show() takes 4 chars, one per digit
        log.info("System: waiting for start button (GPIO %d)...", GPIO_START_BUTTON)

        while True:
            if self._pi.read(GPIO_START_BUTTON):
                time.sleep(0.05)                   # debounce hold
                if self._pi.read(GPIO_START_BUTTON):
                    log.info("System: start button pressed.")
                    break
            time.sleep(0.01)   # 10 ms poll: negligible CPU, imperceptible latency

    def run_countdown(self) -> None:
        """Show 5-4-3-2-1, one digit per second, then blank the display for the run. Blocks 5 s."""
        log.info("System: starting countdown...")
        for count in range(5, 0, -1):
            self._display.show(f"  {count} ")
            log.info("System: countdown %d", count)
            time.sleep(1.0)

        self._display.show("    ")
        log.info("System: GO")

    def update_display(self, elapsed_s: float) -> None:
        """
        Show elapsed time as MM:SS. Call every frame; writes are throttled to one per second.

        Inputs:
            elapsed_s: Seconds since the run started. Minutes cap at 99.
        """
        now = time.monotonic()
        if now - self._last_display_update < _DISPLAY_UPDATE_INTERVAL_S:
            return
        self._last_display_update = now

        minutes = int(elapsed_s) // 60
        seconds = int(elapsed_s) % 60

        # 4 digits: runs won't reach 99:59, but a longer one must not overflow the display
        minutes = min(minutes, 99)

        self._display.numbers(minutes, seconds)
        log.debug("System: display update %02d:%02d", minutes, seconds)

    def show_final_time(self, elapsed_s: float) -> None:
        """
        Freeze the final elapsed time on the display. Call once after the pipeline loop exits.

        Inputs:
            elapsed_s: Total run time in seconds. Minutes cap at 99.
        """
        minutes = min(int(elapsed_s) // 60, 99)
        seconds = int(elapsed_s) % 60
        self._display.numbers(minutes, seconds)
        log.info("System: final time %02d:%02d", minutes, seconds)

    def cleanup(self) -> None:
        """Blank the display and release pigpio. Safe to call even if the display write fails."""
        try:
            self._display.show("    ")
        except Exception:
            pass
        self._pi.stop()
        log.info("System: cleanup complete.")