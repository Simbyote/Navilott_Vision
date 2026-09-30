"""Lane keeping demo: scripted packets through LaneKeepingNavigator onto the real motors.

Purpose:
    A bench check of the navigator on the robot without the camera: a short
    list of hand-built packets (centered, offset left / right, closing on a
    stop line, a stop sign) is fed to LaneKeepingNavigator, and each Command
    drives the motors through MotorController for a moment. Wheels off the
    ground: it moves them.

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

from src.estimation import EstimationPacket
from src.lane_keeping import LaneKeepingNavigator
from src.navigation import Command

log = logging.getLogger("lane_keeping_demo")

# Scale the demo uses to turn a cm offset into the normalized one, carried
# from the original demo; not a measured lane width. It doesn't change the
# demo's commands, since the navigator steers by lane_offset_cm when it's set
DEMO_HALF_LANE_CM = 30.0
STEP_S = 0.1                    # seconds each scripted frame drives for


def mock_packet(frame_id: int, offset_cm: float | None = 0.0, stop_sign: bool = False,
                stop_line_dist: float | None = None) -> EstimationPacket:
    """A go, vision packet with the given lane offset, stop sign and stop-line distance."""
    return EstimationPacket(
        lane_offset=0.0 if offset_cm is None else offset_cm / DEMO_HALF_LANE_CM,
        lane_offset_cm=offset_cm,
        lane_status="vision",
        heading_error=0.0,
        drive_state="go",
        stop_sign_detected=stop_sign,
        stop_line_detected=stop_line_dist is not None,
        stop_line_distance_px=None,
        stop_line_distance_cm=stop_line_dist,
        yaw_rate=0.0,
        lateral_accel=0.0,
        wheel_speed=0.0,
        frame_id=frame_id,
        timestamp_ms=time.monotonic_ns() // 1_000_000,
        left_wheel_cps=0.0,
        right_wheel_cps=0.0,
    )


DEMO_FRAMES = [
    ("centered", mock_packet(1, offset_cm=0.0), STEP_S),
    ("offset left -6.0 cm", mock_packet(2, offset_cm=-6.0), STEP_S),
    ("offset right +6.0 cm", mock_packet(3, offset_cm=6.0), STEP_S),
    ("stop line 5.0 cm", mock_packet(4, offset_cm=0.0, stop_line_dist=5.0), STEP_S),
    ("stop line 2.0 cm", mock_packet(5, offset_cm=0.0, stop_line_dist=2.0), STEP_S),
    ("stop line 1.0 cm", mock_packet(6, offset_cm=0.0, stop_line_dist=1.0), STEP_S),
    ("stop line 0.5 cm", mock_packet(7, offset_cm=0.0, stop_line_dist=0.5), STEP_S),
    ("stop line 0.2 cm", mock_packet(8, offset_cm=0.0, stop_line_dist=0.2), STEP_S),
    ("stop sign", mock_packet(9, offset_cm=0.0, stop_sign=True), STEP_S),
]


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
            log.info("frame %02d | %-22s | left %.3f right %.3f brake %s | %.1fs",
                     packet.frame_id, description, cmd.left, cmd.right, cmd.brake, seconds)
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
        run(DEMO_FRAMES, LaneKeepingNavigator(), MotorController(pi))
    finally:
        pi.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
