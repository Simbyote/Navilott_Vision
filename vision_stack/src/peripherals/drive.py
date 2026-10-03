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

Flow:
    1. Instantiate pigpio.pi connection and check daemon status.
    2. Initialize EncoderReader to start background interrupt callbacks.
    3. Initialize MotorController with configured TB6612 control pins.
    4. drive() / brake() each frame; stop() and cancel() on exit to disengage
       hardware PWM and callbacks safely.
"""

import pigpio

from src.diagnostics.threads import name_pigpio_threads


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
    ) -> None:
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

        self._init_gpio()
        name_pigpio_threads()

    def _init_gpio(self) -> None:
        """Configure TB6612 pin modes on startup."""
        for pin in [self.ain1, self.ain2, self.bin1, self.bin2, self.stby]:
            self._pi.set_mode(pin, pigpio.OUTPUT)

    def drive(self, left_speed: float, right_speed: float) -> None:
        """
        Set raw speeds for left and right motors (-1.0 to 1.0).
        """
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

    def stop(self) -> None:
        """Stop both motors immediately and set standby LOW."""
        self._pi.hardware_PWM(self.pwma, self.pwm_freq, 0)
        self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 0)
        self._pi.write(self.ain1, 0)
        self._pi.write(self.ain2, 0)
        self._pi.write(self.bin1, 0)
        self._pi.write(self.bin2, 0)
        self._pi.write(self.stby, 0)

    def brake(self) -> None:
        """
        Short-brake both motors: they stop in a fraction of the time stop() lets them coast.

        TB6612 with IN1 = IN2 = HIGH ties both motor terminals to the same
        rail, so a spinning motor's back-EMF drives current through its own
        winding and brakes it; stop() leaves the terminals open and the
        wheels freewheel. STBY stays HIGH (standby would float the outputs,
        which is coasting) and PWM is full on, as the datasheet specifies
        for short brake. At standstill it draws no current. Call stop()
        afterwards to put the driver in standby.
        """
        self._pi.write(self.stby, 1)
        for pin in (self.ain1, self.ain2, self.bin1, self.bin2):
            self._pi.write(pin, 1)
        self._pi.hardware_PWM(self.pwma, self.pwm_freq, 1000000)
        self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 1000000)
