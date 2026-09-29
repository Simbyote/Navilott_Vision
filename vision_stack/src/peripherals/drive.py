"""
TB6612 motor driver and N20 quadrature encoder closed-loop control interface.

Purpose:
    Provides hardware-level motor command generation for a TB6612 dual-channel 
    driver alongside background encoder feedback. Direct pigpio hardware PWM and 
    interrupt-driven callbacks monitor wheel rotation continuously without blocking 
    the main control thread. Higher-level routines (such as closed-loop straight-line 
    driving and differential turns) ingest these encoder counts to correct heading 
    drift and execute duration-based maneuvers.

Main package:
    EncoderFrame: one frame window's cumulative encoder counts and calculated 
        instantaneous wheel speeds (counts per second).
    EncoderReader: non-blocking quadrature decoder registering state transitions 
        on left (GPIO 16/19) and right (GPIO 21/20) channel interrupts via pigpio.
    MotorController: abstraction layer translating normalized speed vectors (-1.0 
        to 1.0) into TB6612 direction control pins and hardware PWM duty cycles, 
        and encapsulating closed-loop execution routines.

Flow:
    1. Instantiate pigpio.pi connection and check daemon status.
    2. Initialize EncoderReader to start background interrupt callbacks.
    3. Initialize MotorController with configured TB6612 control pins.
    4. Call drive_straight_closed_loop() or drive_differential_for_duration() to 
       execute routines with real-time feedback logging.
    5. Call stop() and cancel() on exit to disengage hardware PWM and callbacks safely.
"""

import math
import time
import logging
from dataclasses import dataclass
import pigpio

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# =============================================================================
# Telemetry Data Containers
# =============================================================================
@dataclass
class EncoderFrame:
    """Encoder pulse counts and calculated speeds for one frame window."""
    left_count: int = 0
    right_count: int = 0
    left_cps: float = 0.0   # Counts per second
    right_cps: float = 0.0  # Counts per second


# =============================================================================
# N20 Encoder Reader Class (pigpio Hardware Interrupts)
# =============================================================================
class EncoderReader:
    """
    Reads quadrature encoders on Left (GPIO 16/19) and Right (GPIO 21/20) motors.
    """
    LEFT_C1  = 16
    LEFT_C2  = 19
    RIGHT_C1 = 21
    RIGHT_C2 = 20

    def __init__(self, pi: pigpio.pi) -> None:
        self._pi = pi
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

        self._left_pos = 0
        self._right_pos = 0
        self._last_time = time.perf_counter()

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

    def _left_cb(self, gpio: int, level: int, tick: int) -> None:
        if gpio == self.LEFT_C1:
            self._left_c1_state = level
        elif gpio == self.LEFT_C2:
            self._left_c2_state = level

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

    def snapshot(self) -> EncoderFrame:
        """Atomically read current counts and compute counts per second."""
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        l_count = self._left_pos
        r_count = self._right_pos

        l_cps = (l_count / dt) if dt > 0 else 0.0
        r_cps = (r_count / dt) if dt > 0 else 0.0

        return EncoderFrame(
            left_count=l_count,
            right_count=r_count,
            left_cps=l_cps,
            right_cps=r_cps,
        )

    def reset(self) -> None:
        """Reset internal encoder count offsets to zero."""
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
    Controls a dual DC motor setup via a TB6612 motor driver and pigpio.
    Handles raw motor command generation and higher-level closed-loop routines.
    """

    def __init__(
        self,
        pi: pigpio.pi,
        pwma: int = 13,
        ain1: int = 24,
        ain2: int = 25,
        pwmb: int = 12,
        bin1: int = 27,
        bin2: int = 22,
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
        self._pi.write(self.ain1, 1 if left_speed > 0 else 0)
        self._pi.write(self.ain2, 1 if left_speed < 0 else 0)

        # Right Motor Direction & Duty Cycle
        spd_r = int(max(0.0, min(1.0, abs(right_speed))) * 1000000)
        self._pi.hardware_PWM(self.pwmb, self.pwm_freq, spd_r)
        self._pi.write(self.bin1, 1 if right_speed < 0 else 0)
        self._pi.write(self.bin2, 1 if right_speed > 0 else 0)

    def stop(self) -> None:
        """Stop both motors immediately and set standby LOW."""
        self._pi.hardware_PWM(self.pwma, self.pwm_freq, 0)
        self._pi.hardware_PWM(self.pwmb, self.pwm_freq, 0)
        self._pi.write(self.ain1, 0)
        self._pi.write(self.ain2, 0)
        self._pi.write(self.bin1, 0)
        self._pi.write(self.bin2, 0)
        self._pi.write(self.stby, 0)

    def drive_straight_closed_loop(
        self,
        encoders: EncoderReader,
        base_speed: float,
        duration: float,
        kp: float = 0.0015,
        min_speed: float = 0.25,
        max_corr: float = 0.15,
    ) -> None:
        """
        Drives forward using proportional encoder feedback to maintain a straight line.
        """
        encoders.reset()
        start_time = time.perf_counter()

        try:
            while (time.perf_counter() - start_time) < duration:
                frame = encoders.snapshot()

                # Calculate difference (error = Left - Right)
                error = frame.left_count - frame.right_count

                # Compute and clamp proportional speed correction
                correction = error * kp
                correction = max(-max_corr, min(max_corr, correction))

                # Adjust speeds
                left_cmd = max(min_speed, min(1.0, base_speed + correction))
                right_cmd = max(min_speed, min(1.0, base_speed - correction))

                self.drive(left_cmd, right_cmd)

                log.info(
                    f"Closed-Loop | Time: {time.perf_counter() - start_time:.2f}s | "
                    f"Counts (L/R): {frame.left_count}/{frame.right_count} | "
                    f"Error: {error:+d} | Speeds (L/R): {left_cmd:.3f}/{right_cmd:.3f}"
                )
                time.sleep(0.02)
        finally:
            self.stop()

    def drive_differential_for_duration(
        self,
        encoders: EncoderReader,
        left_speed: float,
        right_speed: float,
        duration: float,
    ) -> None:
        """
        Open-loop maneuver for turns using fixed differential speeds while logging encoders.
        """
        encoders.reset()
        start_time = time.perf_counter()

        self.drive(left_speed, right_speed)

        try:
            while (time.perf_counter() - start_time) < duration:
                frame = encoders.snapshot()
                log.info(
                    f"Turning... | Time: {time.perf_counter() - start_time:.2f}s | "
                    f"L: {frame.left_count} ({frame.left_cps:.1f} cps) | "
                    f"R: {frame.right_count} ({frame.right_cps:.1f} cps)"
                )
                time.sleep(0.05)
        finally:
            self.stop()