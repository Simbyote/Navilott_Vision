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

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.estimation import EstimationPacket

# The lowest duty that turns the N20s under load; below it they stall
STALL_DUTY = 0.25


@dataclass(frozen=True)
class Command:
    left: float = 0.0
    right: float = 0.0
    brake: bool = False


BRAKE = Command(brake=True)


@runtime_checkable
class Navigator(Protocol):
    def update(self, packet: EstimationPacket) -> Command:
        ...

    def reset(self) -> None:
        ...


def command_problems(cmd: Command) -> list[str]:
    problems = []
    if not isinstance(cmd, Command):
        return [f"not a Command: {cmd!r}"]
    for side, duty in (("left", cmd.left), ("right", cmd.right)):
        if not -1.0 <= duty <= 1.0:
            problems.append(f"{side} duty {duty} outside [-1, 1]")
        elif duty != 0.0 and abs(duty) < STALL_DUTY:
            problems.append(f"{side} duty {duty} below the stall duty {STALL_DUTY}")
    if cmd.brake and (cmd.left, cmd.right) != (0.0, 0.0):
        problems.append(f"brake with duty ({cmd.left}, {cmd.right}); a brake carries none")
    return problems


# =============================================================================
# Lane Keeping Navigator Implementation
# =============================================================================

class LaneKeepingNavigator(Navigator):
    """
    Proportional differential steering navigator for lane keeping.
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
        pass

    def update(self, packet: EstimationPacket) -> Command:
        # Rule 1: High priority stop signals
        if packet.drive_state == "stop" or packet.stop_sign_detected:
            return BRAKE

        if packet.stop_line_detected and packet.stop_line_distance_cm is not None:
            if packet.stop_line_distance_cm <= 10.0:
                return BRAKE

        # Rule 2: Determine steering error correction
        steering_adj = 0.0

        if packet.lane_status == "vision":
            if packet.lane_offset_cm is not None:
                steering_adj = packet.lane_offset_cm * self.kp_cm
            else:
                steering_adj = packet.lane_offset * self.kp_norm

        elif packet.lane_status in ("hold", "stale"):
            steering_adj = packet.heading_error * self.kp_heading

        steering_adj = max(-self.max_steering_adj, min(self.max_steering_adj, steering_adj))

        # Rule 3: Compute raw wheel duties
        left_duty = self.base_speed - steering_adj
        right_duty = self.base_speed + steering_adj

        # Rule 4: Apply STALL_DUTY limits
        left_duty = self._sanitize_duty(left_duty)
        right_duty = self._sanitize_duty(right_duty)

        cmd = Command(left=left_duty, right=right_duty, brake=False)

        problems = command_problems(cmd)
        if problems:
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


# Execution block MUST be placed at the very bottom, unindented
if __name__ == "__main__":
    try:
        test_packet = EstimationPacket(
            drive_state="drive",
            stop_sign_detected=False,
            stop_line_detected=False,
            stop_line_distance_cm=None,
            stop_line_distance_px=None,
            lane_status="vision",
            lane_offset_cm=-5.0,
            lane_offset=-0.2,
            heading_error=0.0,
            yaw_rate=0.0,
            lateral_accel=0.0,
            wheel_speed=0.0,
            left_wheel_cps=0.0,
            right_wheel_cps=0.0,
            frame_id=1,
            timestamp_ms=1000,
        )

        navigator = LaneKeepingNavigator(base_speed=0.40)
        command = navigator.update(test_packet)

        print("--- Test Run Output ---")
        print(f"Input Lane Offset : {test_packet.lane_offset_cm} cm")
        print(f"Resulting Command : Left={command.left:.3f}, Right={command.right:.3f}, Brake={command.brake}")

    except Exception as e:
        print(f"Error executing test: {e}")