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
# (drive.py's min_speed, held through the 2026-09-30 trials). A command that
# should move a wheel asks for at least this; 0 means that wheel doesn't drive
STALL_DUTY = 0.25


@dataclass(frozen=True)
class Command:
    """
    Motor duty for MotorController.drive(), each wheel in [-1, 1], + = forward.

    brake: Short-brake both motors instead (MotorController.brake()); left and
        right are then 0. Stops the robot sharply; a zero duty without it coasts.
    Spinning in place left is (-s, +s); right is (+s, -s).
    """
    left: float = 0.0
    right: float = 0.0
    brake: bool = False


# Stop and hold still. The drive coasts ~0.15-0.18 s past a zero duty; the
# brake stops it sharply (2026-09-30 trials)
BRAKE = Command(brake=True)


@runtime_checkable
class Navigator(Protocol):
    """
    What a Navigation implementation provides to the pipeline.

    update() is called once per frame, in frame order, with that frame's
    packet, and must return a Command every time: no blocking, no sleeping,
    no exceptions for ordinary input (a stale lane, no detections, None
    distances). The pipeline drives whatever comes back.
    """
    def update(self, packet: EstimationPacket) -> Command:
        """This frame's motor command, from this frame's packet and whatever state the navigator keeps."""
        ...

    def reset(self) -> None:
        """Forget all state, as at the start of a run."""
        ...


def command_problems(cmd: Command) -> list[str]:
    """
    Every way cmd breaks the contract's command rules; [] when it's valid.

    The rules: each duty within [-1, 1]; a brake carries zero duty; a wheel
    that is driven gets at least STALL_DUTY in magnitude.
    """
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
        """
        Initialize navigator gains and parameters.

        Args:
            base_speed: Nominal forward motor duty (default: 0.40).
            kp_cm: Proportional gain when using physical lane_offset_cm.
            kp_norm: Proportional gain when using normalized lane_offset [-1, 1].
            kp_heading: Proportional gain for heading correction when vision is lost.
            max_steering_adj: Maximum differential duty correction allowed.
        """
        self.base_speed = max(STALL_DUTY, min(1.0, base_speed))
        self.kp_cm = kp_cm
        self.kp_norm = kp_norm
        self.kp_heading = kp_heading
        self.max_steering_adj = max_steering_adj

    def reset(self) -> None:
        """Reset internal navigator state."""
        pass

    def update(self, packet: EstimationPacket) -> Command:
        """
        Processes an EstimationPacket and returns a valid Command.
        """
        # Rule 1: High priority stop signals (Red lights or detected stop line/sign)
        if packet.drive_state == "stop" or packet.stop_sign_detected:
            return BRAKE

        # Stop line safety check: apply brake if stop line is near
        if packet.stop_line_detected and packet.stop_line_distance_cm is not None:
            if packet.stop_line_distance_cm <= 10.0:  # Within 10 cm of stop line
                return BRAKE

        # Rule 2: Determine steering error correction
        steering_adj = 0.0

        if packet.lane_status == "vision":
            # Primary vision tracking: positive lane_offset means robot is to the right of center
            if packet.lane_offset_cm is not None:
                # Steering adjustment based on cm offset
                steering_adj = packet.lane_offset_cm * self.kp_cm
            else:
                # Fallback to normalized offset [-1.0, 1.0]
                steering_adj = packet.lane_offset * self.kp_norm

        elif packet.lane_status in ("hold", "stale"):
            # Vision dropped: perform heading correction using IMU heading error
            steering_adj = packet.heading_error * self.kp_heading

        # Clamp maximum steering correction adjustment
        steering_adj = max(-self.max_steering_adj, min(self.max_steering_adj, steering_adj))

        # Rule 3: Compute raw wheel duties (positive offset -> steer left)
        left_duty = self.base_speed - steering_adj
        right_duty = self.base_speed + steering_adj

        # Rule 4: Apply STALL_DUTY limits and bounds clamping
        left_duty = self._sanitize_duty(left_duty)
        right_duty = self._sanitize_duty(right_duty)

        cmd = Command(left=left_duty, right=right_duty, brake=False)

        # Final Contract Safety Check
        problems = command_problems(cmd)
        if problems:
            # Fallback to safe brake if an invalid command condition occurs
            return BRAKE

        return cmd

    def _sanitize_duty(self, duty: float) -> float:
        """
        Enforces stall limits and clamps motor duty within valid ranges.
        """
        if abs(duty) < 1e-4:
            return 0.0
        
        # Clamp duty to [-1.0, 1.0]
        clamped = max(-1.0, min(1.0, duty))
        
        # Enforce minimum STALL_DUTY magnitude for moving wheels
        if 0.0 < clamped < STALL_DUTY:
            return STALL_DUTY
        elif -STALL_DUTY < clamped < 0.0:
            return -STALL_DUTY
            
        return clamped

    if __name__ == "__main__":
    try:
        from src.estimation import EstimationPacket

        # Instantiate a mock packet with all required fields initialized
        test_packet = EstimationPacket(
            # Drive & vision status
            drive_state="drive",
            stop_sign_detected=False,
            stop_line_detected=False,
            stop_line_distance_cm=None,
            stop_line_distance_px=None,
            lane_status="vision",
            lane_offset_cm=-5.0,  # Robot is 5 cm to the left of center
            lane_offset=-0.2,
            heading_error=0.0,
            # Telemetry & sensor fields
            yaw_rate=0.0,
            lateral_accel=0.0,
            wheel_speed=0.0,
            left_wheel_cps=0.0,
            right_wheel_cps=0.0,
            frame_id=1,
            timestamp_ms=1000,
        )

        # Instantiate the navigator
        navigator = LaneKeepingNavigator(base_speed=0.40)

        # Process a frame
        command = navigator.update(test_packet)

        # Print output
        print("--- Test Run Output ---")
        print(f"Input Lane Offset : {test_packet.lane_offset_cm} cm")
        print(f"Resulting Command : Left={command.left:.3f}, Right={command.right:.3f}, Brake={command.brake}")

    except Exception as e:
        print(f"Error instantiating packet: {e}")