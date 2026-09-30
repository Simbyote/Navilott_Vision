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
    Once per camera frame (~20 FPS on the Pi): the pipeline hands the frame's
    EstimationPacket to Navigator.update(), and drives the Command it returns.
"""

"""
Lane Keeping Navigation System

Provides:
- EstimationPacket data structure
- Command & Navigator protocol contract
- LaneKeepingNavigator proportional controller
- Interactive test runner for all operating modes
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

# Minimum motor duty to turn N20 motors under load without stalling
STALL_DUTY = 0.25


# =============================================================================
# Pipeline Contract & Data Structures
# =============================================================================

@dataclass(frozen=True)
class EstimationPacket:
    """Telemetry packet supplied by the vision/estimation stack every frame."""
    drive_state: str                    # "drive", "stop", etc.
    stop_sign_detected: bool
    stop_line_detected: bool
    stop_line_distance_cm: Optional[float]
    stop_line_distance_px: Optional[float]
    lane_status: str                    # "vision", "hold", "stale", or "none"
    lane_offset_cm: Optional[float]     # + = right of center, - = left of center
    lane_offset: float                  # Normalized offset in [-1.0, 1.0]
    heading_error: float                # Radians or degrees error relative to lane path
    yaw_rate: float
    lateral_accel: float
    wheel_speed: float
    left_wheel_cps: float
    right_wheel_cps: float
    frame_id: int
    timestamp_ms: int


@dataclass(frozen=True)
class Command:
    """Motor duty instruction returned to the vehicle drive controller."""
    left: float = 0.0                   # [-1.0, 1.0]
    right: float = 0.0                  # [-1.0, 1.0]
    brake: bool = False                 # Hard short-brake activation


BRAKE = Command(brake=True)


@runtime_checkable
class Navigator(Protocol):
    """Interface required for all Navigation implementations."""
    def update(self, packet: EstimationPacket) -> Command:
        ...

    def reset(self) -> None:
        ...


def command_problems(cmd: Command) -> list[str]:
    """Validates command against motor and safety boundary rules."""
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


# =============================================================================
# Lane Keeping Navigator Implementation
# =============================================================================

class LaneKeepingNavigator(Navigator):
    """
    Proportional differential steering navigator.
    
    Computes left/right wheel duties to steer the robot toward lane center.
    Automatically handles stop signs, close stop lines, and vision fallback.
    """

    def __init__(
        self,
        base_speed: float = 0.40,
        kp_cm: float = 0.015,
        kp_norm: float = 0.35,
        kp_heading: float = 0.008,
        max_steering_adj: float = 0.25,
    ) -> None:
        self.base_speed = max(STALL_DUTY, min(1.0, base_speed))
        self.kp_cm = kp_cm
        self.kp_norm = kp_norm
        self.kp_heading = kp_heading
        self.max_steering_adj = max_steering_adj

    def reset(self) -> None:
        """Resets internal controller state across test/run resets."""
        pass

    def update(self, packet: EstimationPacket) -> Command:
        """Processes a single frame EstimationPacket and returns a motor Command."""
        # Rule 1: High priority stop triggers (State override / Stop sign / Nearby stop line)
        if packet.drive_state == "stop" or packet.stop_sign_detected:
            return BRAKE

        if packet.stop_line_detected and packet.stop_line_distance_cm is not None:
            if packet.stop_line_distance_cm <= 10.0:  # Brake within 10 cm
                return BRAKE

        # Rule 2: Determine steering error correction
        steering_adj = 0.0

        if packet.lane_status == "vision":
            if packet.lane_offset_cm is not None:
                # Primary steering: physical offset in cm
                steering_adj = packet.lane_offset_cm * self.kp_cm
            else:
                # Fallback: normalized offset [-1.0, 1.0]
                steering_adj = packet.lane_offset * self.kp_norm

        elif packet.lane_status in ("hold", "stale"):
            # Vision loss fallback: IMU/gyro heading correction
            steering_adj = packet.heading_error * self.kp_heading

        # Clamp maximum differential steering adjustment
        steering_adj = max(-self.max_steering_adj, min(self.max_steering_adj, steering_adj))

        # Rule 3: Differential duty calculation
        # (+ offset -> vehicle is right of center -> lower right duty / increase left duty -> steer left)
        left_duty = self.base_speed - steering_adj
        right_duty = self.base_speed + steering_adj

        # Rule 4: Apply stall thresholding and clamping bounds
        left_duty = self._sanitize_duty(left_duty)
        right_duty = self._sanitize_duty(right_duty)

        cmd = Command(left=left_duty, right=right_duty, brake=False)

        # Final safety contract verification
        problems = command_problems(cmd)
        if problems:
            return BRAKE

        return cmd

    def _sanitize_duty(self, duty: float) -> float:
        """Enforces minimum STALL_DUTY limits and [-1.0, 1.0] bounds."""
        if abs(duty) < 1e-4:
            return 0.0
        
        clamped = max(-1.0, min(1.0, duty))
        
        if 0.0 < clamped < STALL_DUTY:
            return STALL_DUTY
        elif -STALL_DUTY < clamped < 0.0:
            return -STALL_DUTY
            
        return clamped


# =============================================================================
# Automated Self-Test Harness
# =============================================================================

def create_mock_packet(**kwargs) -> EstimationPacket:
    """Generates an EstimationPacket populated with safe default values."""
    defaults = {
        "drive_state": "drive",
        "stop_sign_detected": False,
        "stop_line_detected": False,
        "stop_line_distance_cm": None,
        "stop_line_distance_px": None,
        "lane_status": "vision",
        "lane_offset_cm": 0.0,
        "lane_offset": 0.0,
        "heading_error": 0.0,
        "yaw_rate": 0.0,
        "lateral_accel": 0.0,
        "wheel_speed": 0.0,
        "left_wheel_cps": 0.0,
        "right_wheel_cps": 0.0,
        "frame_id": 100,
        "timestamp_ms": 5000,
    }
    defaults.update(kwargs)
    return EstimationPacket(**defaults)


if __name__ == "__main__":
    navigator = LaneKeepingNavigator(base_speed=0.40)

    test_scenarios = [
        ("Centered on Lane", create_mock_packet(lane_offset_cm=0.0)),
        ("Offset Left (-6.0 cm -> Steer Right)", create_mock_packet(lane_offset_cm=-6.0)),
        ("Offset Right (+6.0 cm -> Steer Left)", create_mock_packet(lane_offset_cm=6.0)),
        ("Stop Sign Trigger", create_mock_packet(stop_sign_detected=True)),
        ("Stop Line Near (8 cm)", create_mock_packet(stop_line_detected=True, stop_line_distance_cm=8.0)),
        ("Vision Lost (Heading Correction)", create_mock_packet(lane_status="stale", heading_error=15.0)),
    ]

    print(f"{'Scenario':<38} | {'Left Duty':<10} | {'Right Duty':<10} | {'Brake':<6}")
    print("-" * 75)

    for name, packet in test_scenarios:
        cmd = navigator.update(packet)
        print(f"{name:<38} | {cmd.left:<10.3f} | {cmd.right:<10.3f} | {str(cmd.brake):<6}")