"""
navigation_checks.py  --  the Navigation contract as checks any Navigator can be run against

Purpose:
    docs/vision_pipeline/navigation_contract.md in code: each check feeds a
    Navigator a packet sequence and returns every way its commands break the
    contract ([] when it passes), so a navigator's test prints all of them at
    once. No robot, camera or sensors: packets are built by packet().

Main package:
    packet(): an EstimationPacket with contract-neutral defaults.
    frames(): numbered, 50 ms-spaced packets from a list of overrides.
    check_commands, check_no_forward_on_stale, check_no_forward_on_stop,
    check_steers_toward_center: the checks.

Flow:
    Each check resets the navigator, warms it on centered vision frames so
    it has seen a good lane, then feeds the case under test.
"""
from dataclasses import replace

from src.estimation import EstimationPacket
from src.navigation import Command, command_problems

# Frame spacing for built packets: the ~19.8 FPS measured by maneuver_linker (2026-09-30)
FRAME_MS = 50
# Centered vision frames fed before each case, so the navigator has a lane
WARMUP_FRAMES = 10
# Frames of the case under test: long enough to pass any vote or smoothing
CASE_FRAMES = 20

_BASE = EstimationPacket(
    lane_offset=0.0, lane_offset_cm=None, lane_status="vision", heading_error=0.0,
    drive_state="go", stop_sign_detected=False, stop_line_detected=False,
    stop_line_distance_px=None, stop_line_distance_cm=None,
    yaw_rate=0.0, lateral_accel=0.0, wheel_speed=0.0,
    frame_id=0, timestamp_ms=0, left_wheel_cps=0.0, right_wheel_cps=0.0,
)


def packet(**fields) -> EstimationPacket:
    """A centered, go, vision packet at frame 0, with fields overridden."""
    return replace(_BASE, **fields)


def frames(overrides: list[dict], start: int = 0) -> list[EstimationPacket]:
    """
    One packet per override dict, numbered from start and FRAME_MS apart.

    Inputs:
        overrides: Fields for each packet, in order.
        start: The first frame_id.
    Outputs:
        The packets, frame_id and timestamp_ms set.
    """
    return [packet(**o, frame_id=start + i, timestamp_ms=(start + i) * FRAME_MS)
            for i, o in enumerate(overrides)]


def forward(cmd: Command) -> bool:
    """True when cmd moves the robot forward: mean duty > 0. A spin in place isn't forward"""
    return not cmd.brake and cmd.left + cmd.right > 0.0


def _run(nav, case: list[dict]) -> list[tuple[EstimationPacket, Command]]:
    """Reset nav, warm it on centered vision frames, then feed case; the case's (packet, command) pairs."""
    nav.reset()
    warm = frames([{}] * WARMUP_FRAMES)
    for p in warm:
        nav.update(p)
    return [(p, nav.update(p)) for p in frames(case, start=WARMUP_FRAMES)]


def check_commands(nav, packets: list[EstimationPacket]) -> list[str]:
    """
    Every packet gets a valid Command.

    Inputs:
        nav: The Navigator, reset first.
        packets: Fed in order.
    Outputs:
        One line per bad command: not a Command, or failing command_problems().
    """
    nav.reset()
    problems = []
    for p in packets:
        cmd = nav.update(p)
        problems += [f"frame {p.frame_id}: {msg}" for msg in command_problems(cmd)]
    return problems


def check_no_forward_on_stale(nav) -> list[str]:
    """A stale lane never gets forward drive, whatever its (old) offset says."""
    case = [{"lane_status": "stale", "lane_offset": off} for off in (0.0, 0.3, -0.3) for _ in range(CASE_FRAMES // 3)]
    return [f"frame {p.frame_id}: drives forward {cmd} on a stale lane"
            for p, cmd in _run(nav, case) if forward(cmd)]


def check_no_forward_on_stop(nav) -> list[str]:
    """drive_state stop never gets forward drive, even on a good centered lane."""
    case = [{"drive_state": "stop"}] * CASE_FRAMES
    return [f"frame {p.frame_id}: drives forward {cmd} on drive_state stop"
            for p, cmd in _run(nav, case) if forward(cmd)]


def check_steers_toward_center(nav, offset: float = 0.5) -> list[str]:
    """
    On vision, driving forward off center steers back: robot right of center
    (+ offset) gives left duty < right duty, and the mirror case the reverse.

    Inputs:
        nav: The Navigator.
        offset: |lane_offset| of the case.
    Outputs:
        One line per wrong-way frame; one line if a case never drove forward,
        since then the steering couldn't be checked.
    """
    problems = []
    for off in (offset, -offset):
        driven = [(p, cmd) for p, cmd in _run(nav, [{"lane_offset": off}] * CASE_FRAMES) if forward(cmd)]
        if not driven:
            problems.append(f"offset {off:+}: never drove forward, so steering couldn't be checked")
        for p, cmd in driven:
            toward = cmd.left < cmd.right if off > 0 else cmd.left > cmd.right
            if not toward:
                problems.append(f"frame {p.frame_id}: offset {off:+} but {cmd} doesn't steer toward center")
    return problems


def contract_problems(nav) -> list[str]:
    """Every check above on nav, with a mixed packet sequence for check_commands."""
    mixed = frames([{"lane_status": s, "lane_offset": o, "drive_state": d}
                    for s in ("vision", "hold", "stale") for o in (-1.0, -0.2, 0.0, 0.2, 1.0)
                    for d in ("go", "caution", "stop")])
    return (check_commands(nav, mixed) + check_no_forward_on_stale(nav)
            + check_no_forward_on_stop(nav) + check_steers_toward_center(nav))
