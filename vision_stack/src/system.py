"""
system.py

Human Interface Module — Navilott Senior Design

Responsibilities:
    - Start button: wait for press (GPIO 17, active-high)
    - TM1637 4-digit display: countdown, MM:SS elapsed time, final time
    - Cleanup of all GPIO resources on shutdown

Hardware (from GPIO pinout table, Product Spec Table 7):
    TM1637 CLK  -> GPIO 5  (pin 29)
    TM1637 DIO  -> GPIO 6  (pin 31)
    Start button -> GPIO 17 (pin 11), active-high, pull-down

Dependencies:
    pip install tm1637 --break-system-packages
    pigpio daemon must be running: sudo pigpiod

Usage in run_pipeline.py:
    from system import System

    s = System()
    s.wait_for_start()
    s.run_countdown()
    t_run_start = time.perf_counter()

    # inside frame loop:
    s.update_display(elapsed_s)

    # after loop exits:
    s.show_final_time(elapsed_s)
    time.sleep(5.0)
    s.cleanup()
"""

import time
import logging
import pigpio
import tm1637

log = logging.getLogger("system")

# =============================================================================
# Pin assignments — match Product Spec GPIO Table 7
# =============================================================================
_CLK_PIN    = 5    # TM1637 clock
_DIO_PIN    = 6    # TM1637 data
_BUTTON_PIN = 17   # Start button (active-high, pull-down)

# =============================================================================
# Display update throttle
# =============================================================================
_DISPLAY_UPDATE_INTERVAL_S = 1.0   # TM1637 is bit-banged; no need to write > 1 Hz


class System:
    """
    Owns the TM1637 display and start button for the Navilott robot.

    All GPIO is managed through pigpio so the same daemon instance used
    by the motor driver in run_pipeline.py handles both without conflict.
    The TM1637 library handles its own bit-bang timing independently.
    """

    def __init__(self) -> None:
        # pigpio connection — daemon must be running (sudo pigpiod)
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError(
                "pigpio daemon not reachable. Run: sudo pigpiod"
            )

        # Button — input with pull-down
        self._pi.set_mode(_BUTTON_PIN, pigpio.INPUT)
        self._pi.set_pull_up_down(_BUTTON_PIN, pigpio.PUD_DOWN)

        # TM1637 display
        self._display = tm1637.TM1637(clk=_CLK_PIN, dio=_DIO_PIN)
        self._display.brightness(2)   # 0 (dim) – 7 (max); 2 is readable indoors

        # Throttle state for update_display()
        self._last_display_update: float = 0.0

        log.info(
            "System: button GPIO %d, display CLK %d / DIO %d",
            _BUTTON_PIN, _CLK_PIN, _DIO_PIN,
        )

    # -------------------------------------------------------------------------
    # Pre-run sequence
    # -------------------------------------------------------------------------

    def wait_for_start(self) -> None:
        """
        Block until the start button is pressed (GPIO 17 goes high).

        Shows 'rdy' on the display while waiting.
        Debounces by requiring the pin to stay high for 50 ms.
        """
        # TM1637 show_str renders up to 4 chars; 'rdy ' fills all digits
        self._display.show("rdy ")
        log.info("System: waiting for start button (GPIO %d)...", _BUTTON_PIN)

        while True:
            if self._pi.read(_BUTTON_PIN):
                time.sleep(0.05)                   # debounce hold
                if self._pi.read(_BUTTON_PIN):     # still high after 50 ms
                    log.info("System: start button pressed.")
                    break
            time.sleep(0.01)   # 10 ms poll — negligible CPU

    def run_countdown(self) -> None:
        """
        Display 5-4-3-2-1 countdown, one digit per second.
        Leaves the display blank at the end, ready for elapsed time.
        """
        log.info("System: starting countdown...")
        for count in range(5, 0, -1):
            # show() accepts a 4-char string; right-justify the digit
            self._display.show(f"  {count} ")
            log.info("System: countdown %d", count)
            time.sleep(1.0)

        self._display.show("    ")   # blank — pipeline is starting
        log.info("System: GO")

    # -------------------------------------------------------------------------
    # In-loop display update
    # -------------------------------------------------------------------------

    def update_display(self, elapsed_s: float) -> None:
        """
        Write elapsed time to the display in MM:SS format.

        Throttled internally to _DISPLAY_UPDATE_INTERVAL_S (1 Hz) because
        TM1637 is bit-banged and a full 4-digit write costs ~1-2 ms.
        Call this every frame; the throttle absorbs the rate mismatch.

        Inputs:
            elapsed_s: seconds since t_run_start
        """
        now = time.monotonic()
        if now - self._last_display_update < 1.0:   # update at 1 Hz
            return
        self._last_display_update = now

        minutes = int(elapsed_s) // 60
        seconds = int(elapsed_s) % 60

        # Clamp to 99:59 — competition runs won't exceed this
        minutes = min(minutes, 99)

        # numbers() writes 4 raw digits with optional colon between digit 1/2
        self._display.numbers(minutes, seconds)
        log.debug("System: display update %02d:%02d", minutes, seconds)

    # -------------------------------------------------------------------------
    # Post-run
    # -------------------------------------------------------------------------

    def show_final_time(self, elapsed_s: float) -> None:
        """
        Freeze the final elapsed time on the display.
        Called once after the pipeline loop exits.

        Inputs:
            elapsed_s: total run time in seconds
        """
        minutes = min(int(elapsed_s) // 60, 99)
        seconds = int(elapsed_s) % 60
        self._display.numbers(minutes, seconds)
        log.info("System: final time %02d:%02d", minutes, seconds)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Release display and pigpio resources.
        Call in the finally block of run_pipeline.py.
        """
        try:
            self._display.show("    ")   # blank display on shutdown
        except Exception:
            pass
        self._pi.stop()
        log.info("System: cleanup complete.")