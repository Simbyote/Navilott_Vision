"""Navigation contract: what Navigation receives each frame and what it must give back.

Purpose:
    The handoff between the vision stack and Navigation, fixed so both sides
    build against the same thing; every field's sign, unit and rate was
    proven on the robot (maneuver_linker, 2026-09-30). It holds only the
    interface and the command type, so every rule can import it without
    importing the orchestrator (docs/vision_pipeline/navigation_contract.md).

Main package:
    Command: one frame's motor instruction, a duty per wheel or a short
    brake. Navigator is the interface that produces it; enforce() turns any
    Command that breaks the rules into BRAKE.

Flow:
    1. Once per camera frame, the loop hands Navigator.update() the frame's EstimationPacket.
    2. enforce() checks the Command it returns (command_problems()).
    3. The loop drives the Command, or BRAKE.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.estimation.estimation import EstimationPacket

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


def enforce(cmd: Command) -> tuple[Command, list[str]]:
    """
    The command to drive, whatever the navigator returned.

    Inputs:
        cmd: The navigator's answer for this frame.
    Outputs:
        (cmd, []) when it's valid; (BRAKE, its problems) when it breaks the
        contract, so a navigator bug stops the robot instead of driving it.
    """
    problems = command_problems(cmd)
    return (BRAKE, problems) if problems else (cmd, [])
