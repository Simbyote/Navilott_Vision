"""Navigation contract: what Navigation receives each frame and what it must give back.

Purpose:
    The interface between the vision stack and Navigation, kept apart from
    the orchestrator so every rule can import it. Signs, units and rates were
    measured on the robot (docs/vision_pipeline/navigation_contract.md).

Main package:
    Command: one frame's duty per wheel, or a brake. enforce() turns a
    Command that breaks the rules into BRAKE.
"""
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.estimation.estimation import EstimationPacket

# Lowest duty that turns the N20s under load (2026-09-30 trials); 0 means the wheel doesn't drive
STALL_DUTY = 0.25


@dataclass(frozen=True)
class Command:
    """Duty per wheel in [-1, 1], + = forward. brake short-brakes both motors; a zero duty alone coasts."""
    left: float = 0.0
    right: float = 0.0
    brake: bool = False


# A zero duty coasts ~0.15-0.18 s; this stops sharply (2026-09-30 trials)
BRAKE = Command(brake=True)


@runtime_checkable
class Navigator(Protocol):
    """update() runs once per frame, in order, and always returns a Command: it never blocks or raises on ordinary input."""
    def update(self, packet: EstimationPacket) -> Command:
        ...

    def reset(self) -> None:
        ...


def command_problems(cmd: Command) -> list[str]:
    """Every rule cmd breaks, [] when valid: duties within [-1, 1], at least STALL_DUTY when driven, none with a brake."""
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
    """(cmd, []) when valid, else (BRAKE, its problems), so a navigator bug stops the robot."""
    problems = command_problems(cmd)
    return (BRAKE, problems) if problems else (cmd, [])
