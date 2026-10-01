"""Navigation demo: scripted packets through navigation.Navigation onto the real motors.

Purpose:
    A bench check of the navigation subsystem on the robot without the
    camera: hand-built packets (lane keeping centered and off to either
    side, then a stop-sign intersection: the line coming closer, passing
    under the view, the stop and hold, the straight crossing, both
    boundaries back) are fed to Navigation, and each Command drives the
    motors through MotorController for STEP_S. Wheels off the ground: it
    moves them. (The file keeps its name from when it drove lane keeping only.)

Main package:
    mock_packet(): an EstimationPacket for one scripted frame.
    DEMO_FRAMES: the scripted frames, (description, packet, seconds).
    run(): feed frames to a navigator and drive a motor; testable with fakes.
    main(): the same on the robot.

Flow:
    python3 -m src.scripts.lane_keeping_demo     (needs sudo pigpiod)
"""
import logging
import time

from src.estimation.estimation import EstimationPacket
from src.navigation.navigation import Command, Navigation

log = logging.getLogger("lane_keeping_demo")

# Scale the demo uses to turn a cm offset into the normalized one, carried
# from the original demo; not a measured lane width. It doesn't change the
# demo's commands, since the navigator steers by lane_offset_cm when it's set
DEMO_HALF_LANE_CM = 30.0
STEP_S = 0.1                    # seconds each scripted frame drives for


def mock_packet(frame_id: int, offset_cm: float | None = 0.0, stop_sign: bool = False,
                stop_line_rows: float | None = None, lane_mode: str = "two_boundary") -> EstimationPacket:
    """
    A go, vision packet frame_id x STEP_S into the demo.

    Inputs:
        offset_cm: The lane offset; + = robot right of center.
        stop_sign: A voted stop sign this frame.
        stop_line_rows: A voted stop line this many lane-ROI rows above the
            view bottom; None for no line.
        lane_mode: Phase 2's lane mode; "two_boundary" ends an intersection crossing.
    """
    return EstimationPacket(
        lane_offset=0.0 if offset_cm is None else offset_cm / DEMO_HALF_LANE_CM,
        lane_offset_cm=offset_cm,
        lane_status="vision",
        heading_error=0.0,
        drive_state="go",
        stop_sign_detected=stop_sign,
        stop_line_detected=stop_line_rows is not None,
        stop_line_distance_px=stop_line_rows,
        stop_line_distance_cm=None,
        yaw_rate=0.0,
        lateral_accel=0.0,
        wheel_speed=0.0,
        frame_id=frame_id,
        timestamp_ms=int(frame_id * STEP_S * 1000),
        left_wheel_cps=0.0,
        right_wheel_cps=0.0,
        lane_mode=lane_mode,
    )


def _demo_frames() -> list[tuple[str, EstimationPacket, float]]:
    """Lane keeping, then a stop-sign intersection: approach, the line passing under the view, the stop, the crossing."""
    script = [("centered", {}), ("offset left -6.0 cm", {"offset_cm": -6.0}),
              ("offset right +6.0 cm", {"offset_cm": 6.0})]
    script += [(f"stop line at {rows} rows, sign", {"stop_line_rows": rows, "stop_sign": True})
               for rows in (60.0, 40.0, 20.0, 5.0)]
    # Line out of view: crossing straight, the stop after the delay, the hold, crossing on
    script += [("past the line", {"lane_mode": "right_only"})] * 45
    script += [("both boundaries again", {})] * 4
    return [(desc, mock_packet(i + 1, **kw), STEP_S) for i, (desc, kw) in enumerate(script)]


DEMO_FRAMES = _demo_frames()


def run(frames, navigator, motor, sleep=time.sleep) -> list[Command]:
    """
    Feed each frame's packet to the navigator and drive its command.

    Inputs:
        frames: (description, packet, seconds) tuples, in order.
        navigator: A Navigator.
        motor: Anything with drive(left, right), brake() and stop(), like MotorController.
        sleep: Waits each frame's seconds; tests pass a fake.
    Outputs:
        The commands, in order.
    Side effects:
        Drives the motor; always stops it at the end, even on an error.
    """
    commands = []
    try:
        for description, packet, seconds in frames:
            cmd = navigator.update(packet)
            motor.brake() if cmd.brake else motor.drive(cmd.left, cmd.right)
            commands.append(cmd)
            rule = getattr(navigator, "record", {}).get("rule", "")
            log.info("frame %02d | %-26s | %-13s | left %.3f right %.3f brake %s | %.1fs",
                     packet.frame_id, description, rule, cmd.left, cmd.right, cmd.brake, seconds)
            sleep(seconds)
    finally:
        motor.stop()
    return commands


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    import pigpio
    from src.peripherals.drive import MotorController

    pi = pigpio.pi()
    if not pi.connected:
        log.error("pigpio daemon not reachable. Run: sudo pigpiod")
        return 1
    try:
        run(DEMO_FRAMES, Navigation(), MotorController(pi))
    finally:
        pi.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
