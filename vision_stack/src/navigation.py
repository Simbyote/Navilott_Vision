"""
Navigation contract: what Navigation receives each frame and what it must give back.

Purpose:
    The handoff between the vision stack and Navigation, fixed before
    Navigation exists so both sides build against the same thing. Everything
    stated here was proven on the robot by maneuver_linker (2026-09-30): the
    packet fields, their signs and units, the frame rate, and how the drive
    responds to a command. This module holds only the interface and the
    command type; the navigation logic itself belongs to the Navigation
    subsystem. docs/vision_pipeline/navigation_contract.md is the full
    per-field contract.

Main package:
    Command: one frame's motor instruction, per-wheel duty or a short brake.
    Navigator: the interface a Navigation implementation provides:
    update(EstimationPacket) -> Command once per frame, and reset().

Flow:
    Once per camera frame (~5 FPS at 0.2s intervals): the pipeline hands the frame's
    EstimationPacket to Navigator.update(), and drives the Command it returns.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable
import pigpio

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("navigation")

# Minimum motor duty to prevent N20 motor stalling under load
STALL_DUTY = 0.25

# =============================================================================
# Hardware Pin Definitions (TB6612 Motor Driver)
# =============================================================================
_ain1 = 24
_ain2 = 25
_pwma = 13

_bin1 = 27
_bin2 = 22
_pwmb = 12
_stby = 23


# =============================================================================
# Contracts & Data Structures
# =============================================================================
@dataclass(frozen=True)
class EstimationPacket:
    """Telemetry packet supplied by the vision/estimation stack every frame."""
    drive_state: str                    # "drive", "stop"
    stop_sign_detected: bool
    stop_line_detected: bool
    stop_line_distance_cm: Optional[float]
    stop_line_distance_px: Optional[float]
    lane_status: str                    # "vision", "hold", "stale", or "none"
    lane_offset_cm: Optional[float]     # + = right of center, - = left of center
    lane_offset: float                  # Normalized offset [-1.0, 1.0]
    heading_error: float                # Yaw error relative to lane path
    yaw_rate: float
    lateral_accel: float
    wheel_speed: float
    left_wheel_cps: float
    right_wheel_cps: float
    frame_id: int
    timestamp_ms: int


@dataclass(frozen=True)
class Command:
    """Motor duty instruction returned to vehicle drive controller."""
    left: float = 0.0                   # [-1.0, 1.0]
    right: float = 0.0                  # [-1.0, 1.0]
    brake: bool = False                 # Hard short-brake activation


BRAKE = Command(brake=True)


@runtime_checkable
class Navigator(Protocol):
    def update(self, packet: EstimationPacket) -> Command:
        ...

    def reset(self) -> None:
        ...


def command_problems(cmd: Command) -> list[str]:
    """Validates command parameters against physical safety rules."""
    problems = []
    if not isinstance(cmd, Command):
        return [f"not a Command: {cmd!r}"]
    for side, duty in (("left", cmd.left), ("right", cmd.right)):
        if not -1.0 <= duty <= 1.0:
            problems.append(f"{side} duty {duty} outside [-1, 1]")
        elif duty != 0.0 and abs(duty) < STALL_DUTY:
            problems.append(f"{side} duty {duty} below stall duty {STALL_DUTY}")
    if cmd.brake and (cmd.left, cmd.right) != (0.0, 0.0):
        problems.append(f"brake with duty ({cmd.left}, {cmd.right}); brake carries none")
    return problems


def check_stop_line_trigger(packet: EstimationPacket, threshold_cm: float = 3.0) -> bool:
    """Returns True if the vehicle has reached or crossed the stop line threshold."""
    if packet.stop_line_detected and packet.stop_line_distance_cm is not None:
        return packet.stop_line_distance_cm <= threshold_cm
    return False


# =============================================================================
# Lane Keeping Controller Implementation
# =============================================================================
class LaneKeepingNavigator(Navigator):
    """Proportional differential steering navigator for lane keeping."""

    def __init__(
        self,
        base_speed: float = 0.40,
        kp_cm: float = 0.015,             # Reduced proportional gain for milder steering
        kp_norm: float = 0.30,            # Reduced normalized gain
        kp_heading: float = 0.010,        # Reduced heading gain
        max_steering_adj: float = 0.20,   # Capped maximum steering correction
        stop_line_threshold_cm: float = 3.0,
    ) -> None:
        self.base_speed = max(STALL_DUTY, min(1.0, base_speed))
        self.kp_cm = kp_cm
        self.kp_norm = kp_norm
        self.kp_heading = kp_heading
        self.max_steering_adj = max_steering_adj
        self.stop_line_threshold_cm = stop_line_threshold_cm

    def reset(self) -> None:
        pass

    def update(self, packet: EstimationPacket) -> Command:
        # High priority stop triggers
        if packet.drive_state == "stop" or packet.stop_sign_detected:
            return BRAKE

        if check_stop_line_trigger(packet, self.stop_line_threshold_cm):
            return BRAKE

        # Proportional steering adjustment calculation
        steering_adj = 0.0

        if packet.lane_status == "vision":
            if packet.lane_offset_cm is not None:
                steering_adj = packet.lane_offset_cm * self.kp_cm
            else:
                steering_adj = packet.lane_offset * self.kp_norm

        elif packet.lane_status in ("hold", "stale"):
            steering_adj = packet.heading_error * self.kp_heading

        steering_adj = max(-self.max_steering_adj, min(self.max_steering_adj, steering_adj))

        # Differential speed calculation
        left_duty = self.base_speed - steering_adj
        right_duty = self.base_speed + steering_adj

        left_duty = self._sanitize_duty(left_duty)
        right_duty = self._sanitize_duty(right_duty)

        cmd = Command(left=left_duty, right=right_duty, brake=False)

        if command_problems(cmd):
            return BRAKE

        return cmd

    def _sanitize_duty(self, duty: float) -> float:
        if abs(duty) < 1e-4:
            return 0.0
        clamped = max(-1.0, min(1.0, duty))
        if 0.0 < clamped < STALL_DUTY:
            return STALL_DUTY
        elif -STALL_DUTY < clamped < 0.0:
            return -STALL_DUTY
        return clamped


# =============================================================================
# Motor Driver Interface
# =============================================================================
def init_motors(pi: pigpio.pi) -> None:
    """Initialize GPIO pins for the TB6612 motor driver."""
    for pin in [_ain1, _ain2, _bin1, _bin2, _stby]:
        pi.set_mode(pin, pigpio.OUTPUT)


def brake(pi: pigpio.pi) -> None:
    """Activates TB6612 short brake mode on both motor channels."""
    pi.write(_stby, 1)
    pi.write(_ain1, 1)
    pi.write(_ain2, 1)
    pi.hardware_PWM(_pwma, 1000, 1000000)

    pi.write(_bin1, 1)
    pi.write(_bin2, 1)
    pi.hardware_PWM(_pwmb, 1000, 1000000)


def drive(pi: pigpio.pi, left_speed: float, right_speed: float) -> None:
    """Drives left and right motors with normalized speeds (-1.0 to 1.0)."""
    pi.write(_stby, 1)

    # Left Motor
    spd_l = int(max(0.0, min(1.0, abs(left_speed))) * 1000000)
    pi.hardware_PWM(_pwma, 1000, spd_l)
    pi.write(_ain1, 1 if left_speed < 0 else 0)
    pi.write(_ain2, 1 if left_speed > 0 else 0)

    # Right Motor
    spd_r = int(max(0.0, min(1.0, abs(right_speed))) * 1000000)
    pi.hardware_PWM(_pwmb, 1000, spd_r)
    pi.write(_bin1, 1 if right_speed > 0 else 0)
    pi.write(_bin2, 1 if right_speed < 0 else 0)


def execute_command(pi: pigpio.pi, cmd: Command) -> None:
    """Dispatches Command object instructions directly to motor driver functions."""
    if cmd.brake:
        brake(pi)
    else:
        drive(pi, cmd.left, cmd.right)


# =============================================================================
# Main Execution Loop
# =============================================================================
def create_mock_packet(frame_id: int, offset_cm: Optional[float] = 0.0) -> EstimationPacket:
    """Helper to assemble a test frame packet."""
    return EstimationPacket(
        drive_state="drive",
        stop_sign_detected=False,
        stop_line_detected=False,
        stop_line_distance_cm=None,
        stop_line_distance_px=None,
        lane_status="vision",
        lane_offset_cm=offset_cm,
        lane_offset=0.0 if offset_cm is None else offset_cm / 30.0,
        heading_error=0.0,
        yaw_rate=0.0,
        lateral_accel=0.0,
        wheel_speed=0.0,
        left_wheel_cps=0.0,
        right_wheel_cps=0.0,
        frame_id=frame_id,
        timestamp_ms=int(time.perf_counter() * 1000),
    )


def main() -> None:
    log.info("Starting Lane Keeping Navigation Motor Loop (5-Second Run @ 0.2s interval)...")

    pi = pigpio.pi()
    if not pi.connected:
        log.error("Failed to connect to pigpio daemon. Run 'sudo pigpiod' first.")
        return

    init_motors(pi)
    navigator = LaneKeepingNavigator(
        base_speed=0.40,
        kp_cm=0.015,             # Reduced steering response
        max_steering_adj=0.20,   # Lower max steering adjustment clamp
    )

    run_duration_sec = 6.0
    frame_interval = 0.2  # 0.2s correction interval (5 Hz)

    # Sequence of test lane offsets in cm
    offset_pattern = [0.0, -2.0, -5.0, -3.0, 0.0, 3.0, 6.0, 4.0, 1.0, -1.0]
    pattern_length = len(offset_pattern)

    start_time = time.time()
    frame_id = 1

    try:
        while (time.time() - start_time) < run_duration_sec:
            loop_start = time.time()
            elapsed = loop_start - start_time

            simulated_offset_cm = offset_pattern[(frame_id - 1) % pattern_length]

            packet = create_mock_packet(frame_id, offset_cm=simulated_offset_cm)
            cmd = navigator.update(packet)
            execute_command(pi, cmd)

            log.info(
                f"Frame {packet.frame_id:03d} | Elapsed: {elapsed:.2f}s | "
                f"Offset: {simulated_offset_cm:+5.2f} cm | "
                f"Cmd -> L: {cmd.left:.3f}, R: {cmd.right:.3f}, Brake: {cmd.brake}"
            )

            frame_id += 1

            # Precision loop timing to maintain 0.2s intervals
            computation_time = time.time() - loop_start
            sleep_time = frame_interval - computation_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        brake(pi)
        pi.write(_stby, 0)
        pi.stop()
        log.info("Finished 5-second run. Motor driver cleaned up safely.")


if __name__ == "__main__":
    main()