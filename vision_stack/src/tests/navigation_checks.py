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
    intersection(): frames approaching a stop line, the line passing under
    the view, then the time after.
    check_commands, check_stale_lane_slows_then_stops, check_steers_toward_center,
    and the intersection checks (decided 2026-10-01: a stop line stops the
    robot only with a stop sign or a red light, once the robot reaches it):
    check_stops_at_a_red_line, check_crosses_a_green_line,
    check_ignores_a_red_light_without_a_line,
    check_stops_at_a_stop_sign_line_then_goes, check_goes_when_the_light_turns_green.

Flow:
    Each check resets the navigator, warms it on centered vision frames so
    it has seen a good lane, then feeds the case under test.
"""
from dataclasses import replace

from src.estimation.estimation import EstimationPacket
from src.navigation.navigation_contract import Command, command_problems
from src.navigation.stop_line import STOP_DELAY_MS

# Frame spacing for built packets: the ~19.8 FPS measured by maneuver_linker (2026-09-30)
FRAME_MS = 50


def reach_frames(delay_ms: int = STOP_DELAY_MS, frame_ms: int = FRAME_MS) -> int:
    """
    Frames from a stop line leaving the view to the robot reaching it: the
    delay in frames, but at least one. The tracker spends a frame in
    CROSSING, which the intersection rule starts on, even with no delay.
    """
    return max(1, -(-delay_ms // frame_ms))


# Centered vision frames fed before each case, so the navigator has a lane
WARMUP_FRAMES = 10
# Frames of the case under test: long enough to pass any vote or smoothing
CASE_FRAMES = 20

_BASE = EstimationPacket(
    lane_offset=0.0, lane_offset_cm=None, lane_status="vision", heading_error=0.0,
    drive_state="go", stop_sign_detected=False, stop_line_detected=False,
    stop_line_distance_px=None, stop_line_distance_cm=None,
    yaw_rate=0.0, lateral_accel=0.0,
    frame_id=0, timestamp_ms=0, left_wheel_cps=0.0, right_wheel_cps=0.0, lane_mode="two_boundary",
)

# A stale lane: mean wheel duty no more than this while it stays stale (plus
# room for a slow wheel lifted to the stall duty), and stopped within
# STALE_STOP_WITHIN_FRAMES (~2 s; the end-of-course rule stops in ~1 s)
STALE_MAX_DUTY = 0.30
STALE_LIFT_ROOM = 0.05
STALE_STOP_WITHIN_FRAMES = 40
STALE_FRAMES = 60

# A stop line coming down the image to the view bottom (lane-ROI rows above it)
APPROACH_ROWS = (60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0)
# Frames after the line leaves the view: 6 s, room for any stop delay, a
# stop sign's hold and the robot driving on
AFTER_FRAMES = 120


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


def check_stale_lane_slows_then_stops(nav, stale_frames: int = STALE_FRAMES) -> list[str]:
    """
    A stale lane (decided 2026-10-01): mean duty at most STALE_MAX_DUTY while it stays
    stale, then a stop within STALE_STOP_WITHIN_FRAMES that holds: the lane
    staying lost means the course has ended.

    Inputs:
        nav: The Navigator.
        stale_frames: How long the case keeps the lane stale.
    Outputs:
        One line per too-fast frame; one if it never stopped, or drove again after.
    """
    case = [{"lane_status": "stale", "lane_offset": off, "heading_error": hd}
            for off, hd in ((0.0, 0.0), (0.4, 5.0), (-0.4, -5.0)) for _ in range(stale_frames // 3)]
    results = _run(nav, case)
    problems = [f"frame {p.frame_id}: {cmd} faster than {STALE_MAX_DUTY} on a stale lane"
                for p, cmd in results if forward(cmd) and (cmd.left + cmd.right) / 2 > STALE_MAX_DUTY + STALE_LIFT_ROOM]
    stops = [i for i, (_, cmd) in enumerate(results) if not forward(cmd)]
    if not stops or stops[0] >= STALE_STOP_WITHIN_FRAMES:
        return problems + [f"never stopped within {STALE_STOP_WITHIN_FRAMES} frames of a stale lane"]
    return problems + [f"frame {p.frame_id}: drives again on a lane that stayed stale"
                       for p, cmd in results[stops[0]:] if forward(cmd)]


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


def intersection(approach: dict | None = None, after: dict | None = None,
                 after_frames: int = AFTER_FRAMES) -> list[dict]:
    """
    Overrides for a stop line coming into view, passing under it, and the time after.

    Inputs:
        approach: Fields for the frames the line is in view (e.g. a stop sign).
        after: Fields for every frame after it left (e.g. the light).
    Outputs:
        len(APPROACH_ROWS) + after_frames overrides, for _run() or frames().
    """
    approach, after = approach or {}, after or {}
    return ([{"stop_line_detected": True, "stop_line_distance_px": rows, **approach} for rows in APPROACH_ROWS]
            + [dict(after) for _ in range(after_frames)])


def _split(results):
    """(the frames with the line in view, the frames after it left)."""
    n = len(APPROACH_ROWS)
    return results[:n], results[n:]


def _braked_on_approach(seen) -> list[str]:
    return [f"frame {p.frame_id}: brakes with the stop line still in view, before reaching it"
            for p, cmd in seen if not forward(cmd)]


def check_stops_at_a_red_line(nav) -> list[str]:
    """A red light at a stop line: drive to the line, then brake and keep braking while it's red."""
    seen, after = _split(_run(nav, intersection({"drive_state": "stop"}, {"drive_state": "stop"})))
    problems = _braked_on_approach(seen)
    braking = [i for i, (_, cmd) in enumerate(after) if cmd.brake]
    if not braking:
        return problems + ["never braked at a red light after reaching the stop line"]
    return problems + [f"frame {p.frame_id}: drives again while the light is still red"
                       for p, cmd in after[braking[0]:] if not cmd.brake]


def check_crosses_a_green_line(nav) -> list[str]:
    """A stop line with no sign and a green light is only an intersection: never brake, keep driving."""
    return [f"frame {p.frame_id}: {'brakes' if cmd.brake else 'stops driving'} at a green stop line"
            for p, cmd in _run(nav, intersection()) if not forward(cmd)]


def check_ignores_a_red_light_without_a_line(nav) -> list[str]:
    """A red light with no stop line in sight doesn't stop the robot: it stops at the line."""
    return [f"frame {p.frame_id}: stops for a red light with no stop line"
            for p, cmd in _run(nav, [{"drive_state": "stop"}] * CASE_FRAMES) if not forward(cmd)]


def check_stops_at_a_stop_sign_line_then_goes(nav) -> list[str]:
    """A stop sign at a stop line: drive to the line, brake to a stop there, then drive on."""
    seen, after = _split(_run(nav, intersection({"stop_sign_detected": True})))
    problems = _braked_on_approach(seen)
    braking = [i for i, (_, cmd) in enumerate(after) if cmd.brake]
    if not braking:
        return problems + ["never stopped at a stop sign's line"]
    if not any(forward(cmd) for _, cmd in after[braking[-1] + 1:]):
        problems.append("never drove on after stopping at a stop sign")
    return problems


def check_goes_when_the_light_turns_green(nav, red_frames: int = 70) -> list[str]:
    """Waiting at a red line, a green light lets the robot drive on."""
    case = intersection({"drive_state": "stop"},
                        after_frames=0) + [{"drive_state": "stop"}] * red_frames + [{}] * (AFTER_FRAMES - red_frames)
    seen, after = _split(_run(nav, case))
    problems = _braked_on_approach(seen)
    if not any(cmd.brake for _, cmd in after[:red_frames]):
        problems.append("never stopped at the red light")
    if not any(forward(cmd) for _, cmd in after[red_frames:]):
        problems.append("never drove on after the light turned green")
    return problems


INTERSECTION_CHECKS = (check_stops_at_a_red_line, check_crosses_a_green_line,
                       check_ignores_a_red_light_without_a_line, check_stops_at_a_stop_sign_line_then_goes,
                       check_goes_when_the_light_turns_green)


def contract_problems(nav) -> list[str]:
    """Every check above on nav, with a mixed packet sequence for check_commands (ending on a stale lane long enough to stop)."""
    mixed = frames([{"lane_status": s, "lane_offset": o, "drive_state": d}
                    for s in ("vision", "hold", "stale") for o in (-1.0, -0.2, 0.0, 0.2, 1.0)
                    for d in ("go", "caution", "stop")] + [{"lane_status": "stale"}] * STALE_FRAMES)
    return (check_commands(nav, mixed) + check_stale_lane_slows_then_stops(nav) + check_steers_toward_center(nav)
            + [p for check in INTERSECTION_CHECKS for p in check(nav)])
