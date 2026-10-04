"""
TB6612 motor driver and N20 quadrature encoders.

Purpose:
    The robot's drive hardware, and nothing above it: MotorController turns
    per-wheel duty into the TB6612's PWM and direction pins (drive, a short
    brake, standby), and EncoderReader counts each wheel's quadrature edges
    on pigpio interrupt callbacks, so counting never blocks the main loop.
    Deciding how to drive belongs to navigation (src/navigation/) and the
    drive trial (src/maneuver.py); reading the encoders once per frame
    belongs to the sensor hub (src/peripherals/sensing.py), which turns
    counts() into counts per second per frame.

Main package:
    EncoderReader: non-blocking quadrature decoder registering state transitions
        on left (GPIO 21/20) and right (GPIO 16/19) channel interrupts via pigpio;
        counts() -> (left, right) live totals, + = forward.
    MotorController: drive(left, right) in [-1, 1] per wheel, brake(), stop().
        A watchdog thread brakes the motors when drive() goes quiet for
        MOTOR_WATCHDOG_S while they're driving: a stuck run loop (a camera
        that stops delivering frames) can't leave the robot driving blind.

Flow:
    1. Instantiate pigpio.pi connection and check daemon status.
    2. Initialize EncoderReader to start background interrupt callbacks.
    3. Initialize MotorController with configured TB6612 control pins.
    4. drive() / brake() each frame; stop() and cancel() on exit to disengage
       hardware PWM and callbacks safely. Between drive()s, the watchdog
       brakes if the next one is MOTOR_WATCHDOG_S late.
"""

import logging
import threading
import time

import pigpio

from src.diagnostics.threads import name_os_thread, name_pigpio_threads
from src.params import MOTOR_WATCHDOG_S

log = logging.getLogger(__name__)

# The watchdog looks this many times per MOTOR_WATCHDOG_S, so it brakes at
# most 1/WATCHDOG_CHECKS of the timeout late (0.6 s for 0.5 s)
WATCHDOG_CHECKS = 5


# =============================================================================
# N20 Encoder Reader Class (pigpio Hardware Interrupts)
# =============================================================================
class EncoderReader:
    """
    Reads quadrature encoders on Left (GPIO 19/16) and Right (GPIO 20/21) motors.
    """
    # Pairing: each encoder goes with the wheel the same-named motor command
    # turns. 2026-09-29 the encoder on 21/20 turned with the left command; the
    # 2026-10-01 motor pin fix moved the left command to the other driver
    # channel, so 21/20 is now the right wheel's and 16/19 the left's
    # (test_drive_characterization, 2026-10-03: "left motor moved the right
    # encoder"; maneuver_linker's straight legs curved, the count correction
    # steering the wrong wheel).
    # Signs: + = forward was measured per encoder (2026-09-29, turning each
    # wheel by hand) and stays with it. Each pair also swaps which pin is C1,
    # so the decode below keeps giving + forward: forward, 19 leads 16 and 21
    # leads 20.
    LEFT_C1  = 19
    LEFT_C2  = 16
    RIGHT_C1 = 20
    RIGHT_C2 = 21

    def __init__(self, pi: pigpio.pi) -> None:
        self._pi = pi
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

        self._left_pos = 0
        self._right_pos = 0

        self._left_c1_state = 0
        self._left_c2_state = 0
        self._right_c1_state = 0
        self._right_c2_state = 0

        # Configure GPIO modes and pull-ups
        for pin in [self.LEFT_C1, self.LEFT_C2, self.RIGHT_C1, self.RIGHT_C2]:
            self._pi.set_mode(pin, pigpio.INPUT)
            self._pi.set_pull_up_down(pin, pigpio.PUD_UP)

        # Set up callbacks for quadrature decoding
        self._cb_l1 = self._pi.callback(self.LEFT_C1, pigpio.EITHER_EDGE, self._left_cb)
        self._cb_l2 = self._pi.callback(self.LEFT_C2, pigpio.EITHER_EDGE, self._left_cb)
        self._cb_r1 = self._pi.callback(self.RIGHT_C1, pigpio.EITHER_EDGE, self._right_cb)
        self._cb_r2 = self._pi.callback(self.RIGHT_C2, pigpio.EITHER_EDGE, self._right_cb)
        name_pigpio_threads()           # the counting runs in pigpio's callback thread: show it by name

    def _left_cb(self, gpio: int, level: int, tick: int) -> None:
        if gpio == self.LEFT_C1:
            self._left_c1_state = level
        elif gpio == self.LEFT_C2:
            self._left_c2_state = level

        # + = forward: measured 2026-09-29 by turning each wheel forward by
        # hand, which read negative under the previous decode
        if gpio == self.LEFT_C1 and level == 1:
            if self._left_c2_state == 0:
                self._left_pos += 1
            else:
                self._left_pos -= 1

    def _right_cb(self, gpio: int, level: int, tick: int) -> None:
        if gpio == self.RIGHT_C1:
            self._right_c1_state = level
        elif gpio == self.RIGHT_C2:
            self._right_c2_state = level

        if gpio == self.RIGHT_C1 and level == 1:
            if self._right_c2_state == 0:
                self._right_pos -= 1
            else:
                self._right_pos += 1

    def counts(self) -> tuple[int, int]:
        """(left, right) counts since reset(), + = forward. Changes no state, so any thread may call it."""
        return self._left_pos, self._right_pos

    def reset(self) -> None:
        """Zero both counts."""
        self._left_pos = 0
        self._right_pos = 0

    def cancel(self) -> None:
        """Clean up pigpio callbacks."""
        self._cb_l1.cancel()
        self._cb_l2.cancel()
        self._cb_r1.cancel()
        self._cb_r2.cancel()


# =============================================================================
# Motor Controller Class (TB6612 Driver & Closed-Loop Control)
# =============================================================================
class MotorController:
    """
    Controls a dual DC motor setup via a TB6612 motor driver and pigpio:
    per-wheel duty, a short brake, and standby.

    Watchdog: drive() arms it; brake() and stop() disarm it, since both
    leave the motors safe however long the loop is gone. Armed and with no
    drive() for watchdog_s, its thread short-brakes the motors and counts a
    trip in watchdog_trips. The next drive() drives again: a loop that was
    only slow carries on, a dead one stays braked. Every pin sequence runs
    under one lock, so a brake from the watchdog never interleaves with a
    drive() from the loop.
    """

    def __init__(
        self,
        pi: pigpio.pi,
        pwma: int = 12,   # left motor: PWM
        ain1: int = 22,   # left motor: high = reverse
        ain2: int = 27,   # left motor: high = forward
        pwmb: int = 13,   # right motor: PWM
        bin1: int = 25,   # right motor: high = forward
        bin2: int = 24,   # right motor: high = reverse
        stby: int = 23,
        pwm_freq: int = 1000,
        watchdog_s: float | None = MOTOR_WATCHDOG_S,
        clock=time.monotonic,
    ) -> None:
        """
        Inputs:
            pi: A connected pigpio.pi.
            pwma ... stby: TB6612 pins (BCM).
            pwm_freq: Hardware PWM frequency in Hz.
            watchdog_s: Brake after this long without a drive(); None or 0
                turns the watchdog off.
            clock: Monotonic seconds; tests pass a fake.
        """
        self._pi = pi
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

        self.pwma = pwma
        self.ain1 = ain1
        self.ain2 = ain2
        self.pwmb = pwmb
        self.bin1 = bin1
        self.bin2 = bin2
        self.stby = stby
        self.pwm_freq = pwm_freq

        self.watchdog_s = watchdog_s or None
        self.watchdog_trips = 0
        self._clock = clock
        self._lock = threading.Lock()
        self._driven_at: float | None = None      # last drive() while armed; None when disarmed
        self._wd_stop = threading.Event()
        self._wd_thread: threading.Thread | None = None

        self._init_gpio()
        name_pigpio_threads()

    def _init_gpio(self) -> None:
        """Configure TB6612 pin modes on startup."""
        for pin in [self.ain1, self.ain2, self.bin1, self.bin2, self.stby]:
            self._pi.set_mode(pin, pigpio.OUTPUT)

    def drive(self, left_speed: float, right_speed: float) -> None:
        """
        Set raw speeds for left and right motors (-1.0 to 1.0), and arm
        the watchdog: the next drive() is due within watchdog_s.
        """
        with self._lock:
            self._pi.write(self.stby, 1)

            # Left Motor Direction & Duty Cycle (0 to 1,000,000 for hardware_PWM)
            spd_l = int(max(0.0, min(1.0, abs(left_speed))) * 1000000)
            self._pi.hardware_PWM(self.pwma, self.pwm_freq, spd_l)
            self._pi.write(self.ain1, 1 if left_speed < 0 else 0)
            self._pi.write(self.ain2, 1 if left_speed > 0 else 0)

            # Right Motor Direction & Duty Cycle
            spd_r = int(max(0.0, min(1.0, abs(right_speed))) * 1000000)
            self._pi.hardware_PWM(self.pwmb, self.pwm_freq, spd_r)
            self._pi.write(self.bin1, 1 if right_speed > 0 else 0)
            self._pi.write(self.bin2, 1 if right_speed < 0 else 0)
            self._driven_at = self._clock()
        self._start_watchdog()

    def stop(self) -> None:
        """Stop both motors immediately, set standby LOW, and end the watchdog."""
        with self._lock:
            self._pi.hardware_PWM(self.pwma, self.pwm_freq, 0)
            self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 0)
            self._pi.write(self.ain1, 0)
            self._pi.write(self.ain2, 0)
            self._pi.write(self.bin1, 0)
            self._pi.write(self.bin2, 0)
            self._pi.write(self.stby, 0)
            self._driven_at = None
        self._stop_watchdog()

    def brake(self) -> None:
        """
        Short-brake both motors: they stop in a fraction of the time stop() lets them coast.

        TB6612 with IN1 = IN2 = HIGH ties both motor terminals to the same
        rail, so a spinning motor's back-EMF drives current through its own
        winding and brakes it; stop() leaves the terminals open and the
        wheels freewheel. STBY stays HIGH (standby would float the outputs,
        which is coasting) and PWM is full on, as the datasheet specifies
        for short brake. At standstill it draws no current. Call stop()
        afterwards to put the driver in standby. Disarms the watchdog.
        """
        with self._lock:
            self._short_brake()
            self._driven_at = None

    def check(self) -> bool:
        """
        One watchdog look: brake if armed and drive() is watchdog_s overdue.
        The watchdog thread calls it every watchdog_s / WATCHDOG_CHECKS;
        tests call it directly.

        Outputs:
            True if it braked the motors just now.

        Side effects:
            On a trip: short brake, disarm, watchdog_trips += 1, a warning logged.
        """
        with self._lock:
            if self.watchdog_s is None or self._driven_at is None:
                return False
            quiet = self._clock() - self._driven_at
            if quiet < self.watchdog_s:
                return False
            self._short_brake()
            self._driven_at = None
            self.watchdog_trips += 1
        log.warning("motor watchdog: no drive() for %.2f s (limit %.2f s), motors braked: "
                    "the run loop is stuck (a camera that stopped delivering frames?)",
                    quiet, self.watchdog_s)
        return True

    def _short_brake(self) -> None:
        """TB6612 short brake (see brake()). The caller holds the lock."""
        self._pi.write(self.stby, 1)
        for pin in (self.ain1, self.ain2, self.bin1, self.bin2):
            self._pi.write(pin, 1)
        self._pi.hardware_PWM(self.pwma, self.pwm_freq, 1000000)
        self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 1000000)

    def _start_watchdog(self) -> None:
        """Start the watchdog thread if it's on and not already running."""
        if self.watchdog_s is None or (self._wd_thread is not None and self._wd_thread.is_alive()):
            return
        self._wd_stop = threading.Event()        # its own, so a slow old thread can't be revived
        self._wd_thread = threading.Thread(target=self._watch, args=(self._wd_stop,),
                                           name="motor-watchdog", daemon=True)
        self._wd_thread.start()

    def _stop_watchdog(self) -> None:
        """End the watchdog thread; the next drive() starts a new one."""
        self._wd_stop.set()
        t = self._wd_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._wd_thread = None

    def _watch(self, stop_evt: threading.Event) -> None:
        """check() every watchdog_s / WATCHDOG_CHECKS until stop()."""
        name_os_thread("motor-watchdog")        # top, ps and diagnostics.monitor show it by name
        period = self.watchdog_s / WATCHDOG_CHECKS
        while not stop_evt.wait(period):
            try:
                self.check()
            except Exception:                   # a failed brake must not end the watching
                log.exception("motor watchdog: check failed")
