"""Production run: the start button, the course, the time on the display. Nothing recorded.

Purpose:
    What the robot runs on the course: python3 -m src.main. It reads and
    checks the route, prints the plan and shows its step count on the
    display ("St 3"), waits for the start button and the countdown, then
    drives: every camera frame goes through Pipeline.step() (Phases 1-3 and
    Navigation) and the Command it returns goes to the motors, the last
    stage. The display shows the elapsed time while it drives. Nothing is
    recorded or written: the linkers are the instrumented twins
    (navigation_linker for this whole chain).

    How a run ends, motors first (brake, then standby) every time:
        finished          the route's finish: the final time stays up.
        ended early       the lane lost before the route was done: the
                          display alternates "E  N" (N = the step reached)
                          and the time until Ctrl-C.
        run time cap      MAX_RUN_S, a backstop in case the end is never
                          seen: alternates "E  t" and the time, as above.
        Ctrl-C            the time so far stays up.
        error             "Err " stays up; the traceback is printed.

Main package:
    run(): one run, from the button to the halt; returns a RunResult.
    end_screen(): what the display shows after it, and for how long.
    cli(): opens the hardware and runs it.

Flow:
    1. load_route(), print the plan; open the camera, sensors, motors and
       display; build the Pipeline. A bad route or missing hardware exits 2
       before anything moves.
    2. "St N" until the button, then the countdown.
    3. Per frame: camera -> Pipeline.step(frame, sensors) -> motors; time on
       the display; until Navigation finishes, the cap, Ctrl-C or an error.
    4. Halt the motors, release the camera and sensors.
    5. The end screen; release the display without blanking it.
"""
import argparse
import sys
import time
import traceback
from dataclasses import dataclass, replace

from src.capture.camera import CameraSource, CaptureError
from src.config import MANEUVER, MEASURED, MEASURED_ESTIMATION, ROUTE_PATH
from src.navigation.end_of_course import OUTCOME_EARLY
from src.navigation.route import Route, RouteError, load_route
from src.params import FPS, FRAME_H, FRAME_W
from src.pipeline import Pipeline

# The run's backstop: brake and end after this long, whatever Navigation
# does (decided 2026-10-01: a generous cap, in case the end of the course is
# never seen). Course runs are well under a minute; raise it for a longer course
MAX_RUN_S = 300.0
# How long the halt holds the short brake before standby. The drive coasts
# ~0.15-0.18 s past a zero duty (2026-09-30 trials); braking a little longer
# than that stops it before the outputs float
HALT_BRAKE_S = 0.3
# The ended-early screen alternates its code and the time this often (decided 2026-10-01)
ALTERNATE_S = 2.0

END_FINISHED, END_EARLY, END_CAP, END_INTERRUPT, END_ERROR = (
    "finished", "ended early", "run time cap", "interrupted (Ctrl-C)", "error")
CODE_CAP, CODE_ERROR = "E  t", "Err "


@dataclass(frozen=True)
class RunResult:
    """
    How a run went.

    ended_by: One of the END_* names.
    elapsed_s: From GO to the end; 0.0 if it never started.
    frames: Frames driven.
    end_step: The route step it ended on (Navigation's end_step, else its progress).
    error: The exception, for END_ERROR.
    """
    ended_by: str
    elapsed_s: float
    frames: int
    end_step: int | None = None
    error: BaseException | None = None


def step_text(route: Route) -> str:
    """The start screen: the route's maneuver count, "St 3" (capped at 99)."""
    return f"St{min(len(route.maneuvers), 99):>2}"


def early_code(step: int | None) -> str:
    """The ended-early code: "E  2" for step 2 (capped at 99)."""
    return f"E{min(step or 0, 99):>3}"


def mm_ss(seconds: float) -> str:
    """Seconds as the display shows them, "01:05" (minutes capped at 99)."""
    return f"{min(int(seconds) // 60, 99):02d}:{int(seconds) % 60:02d}"


def halt(motor, sleep=time.sleep) -> None:
    """
    Stop the robot: short brake for HALT_BRAKE_S, then standby.

    Side effects:
        Always ends with motor.stop(), even if the brake fails.
    """
    try:
        motor.brake()
        sleep(HALT_BRAKE_S)
    finally:
        motor.stop()


def run(camera, sensors, motor, pipeline: Pipeline, system, route: Route,
        clock=time.monotonic, sleep=time.sleep, max_run_s: float = MAX_RUN_S) -> RunResult:
    """
    One run, from the start button to the halt.

    Inputs:
        camera: capture.CameraSource, opened: read() -> FrameData, or None
            for a dropped frame; release().
        sensors: sensing.Sensors (sample() -> SensorSample | None, stop()).
        motor: drive.MotorController: drive(left, right), brake(), stop().
        pipeline: A fresh Pipeline built with route.
        system: peripherals.system.System: the button and display.
        route: For the start screen.
        clock, sleep: Seconds, monotonic; injected by tests.
        max_run_s: Brake and end once this long has passed since GO.

    Outputs:
        The RunResult. The display is left for end_screen().

    Side effects:
        Drives the motors. Always halts them first, then releases the
        camera and stops the sensors, whatever ended the run.
    """
    ended_by, error, frames, t0 = END_INTERRUPT, None, 0, None
    try:
        system.wait_for_start(step_text(route))
        system.run_countdown()
        sensors.sample()                                # start the first window at GO
        t0 = clock()
        while True:
            if clock() - t0 >= max_run_s:
                ended_by = END_CAP
                break
            fd = camera.read()
            if fd is None:                              # a dropped frame: no id spent, carry on
                continue
            cmd = pipeline.step(fd.frame, fd.frame_id, fd.timestamp_ms, sensors.sample())
            if cmd.brake:
                motor.brake()
            else:
                motor.drive(cmd.left, cmd.right)
            frames += 1
            system.update_display(clock() - t0)
            if pipeline.finished:
                ended_by = END_EARLY if pipeline.navigation.outcome == OUTCOME_EARLY else END_FINISHED
                break
    except KeyboardInterrupt:
        ended_by = END_INTERRUPT
    except Exception as exc:
        ended_by, error = END_ERROR, exc
    finally:
        try:
            halt(motor, sleep)                          # first, whatever happened
        finally:
            sensors.stop()
            camera.release()
    elapsed = 0.0 if t0 is None else clock() - t0
    end_step = pipeline.navigation.end_step
    return RunResult(ended_by, elapsed, frames,
                     pipeline.navigation.progress.step if end_step is None else end_step, error)


def end_screen(result: RunResult) -> list:
    """
    What the display shows after a run.

    Outputs:
        The screens in order: a str is a code for show_text(), a float the
        time for show_final_time(). One screen stays up; two alternate every
        ALTERNATE_S until Ctrl-C.
    """
    if result.ended_by == END_ERROR:
        return [CODE_ERROR]
    if result.ended_by == END_EARLY:
        return [early_code(result.end_step), result.elapsed_s]
    if result.ended_by == END_CAP:
        return [CODE_CAP, result.elapsed_s]
    return [result.elapsed_s]                           # finished, Ctrl-C


def show_end(system, screens: list, sleep=time.sleep) -> None:
    """
    Put the end screen up: one screen once; two alternating until Ctrl-C.

    Side effects:
        Writes the display; blocks while alternating.
    """
    def show(screen):
        if isinstance(screen, str):
            system.show_text(screen)
        else:
            system.show_final_time(screen)

    if len(screens) == 1:
        show(screens[0])
        return
    try:
        while True:
            for screen in screens:
                show(screen)
                sleep(ALTERNATE_S)
    except KeyboardInterrupt:
        pass


def summary(result: RunResult) -> str:
    """The terminal's last line."""
    line = f"{result.ended_by} at step {result.end_step} after {mm_ss(result.elapsed_s)} ({result.frames} frames)"
    return line if result.error is None else f"{line}: {result.error!r}"


def cli(argv: list[str] | None = None) -> int:
    """
    Command line: python3 -m src.main [--route PATH] [--max-run-s S]

    Outputs:
        0 after a run (however it ended); 2 for a bad route or hardware
        that wouldn't open, before anything moves.
    """
    ap = argparse.ArgumentParser(description="Run the course: start button, drive, time on the display.")
    ap.add_argument("--route", default=str(ROUTE_PATH), metavar="PATH",
                    help="the course plan (JSON: maneuvers, finish); default config.ROUTE_PATH")
    ap.add_argument("--max-run-s", type=float, default=MAX_RUN_S,
                    help=f"brake and end after this long (default {MAX_RUN_S:.0f})")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        route = load_route(args.route)
    except RouteError as exc:
        print(f"route error: {exc}")
        return 2
    print("\n".join(route.describe()))

    # The gyro bias navigation_linker defaults to, so the two drive alike
    estimation = replace(MEASURED_ESTIMATION, gyro_bias_dps=MANEUVER.gyro_bias_dps)
    pipeline = Pipeline(MEASURED, estimation, route=route)
    camera = sensors = motor = system = None
    try:
        import pigpio
        from src.peripherals.drive import MotorController
        from src.peripherals.sensing import Sensors
        from src.peripherals.system import System
        motor = MotorController(pigpio.pi())
        motor.stop()
        system = System()
        sensors = Sensors(imu=True, encoders=True)
        camera = CameraSource(FRAME_W, FRAME_H, FPS).open()
    except (CaptureError, OSError, RuntimeError, ImportError) as exc:
        print(f"hardware error: {exc!r}")
        for close in (getattr(camera, "release", None), getattr(sensors, "stop", None),
                      getattr(motor, "stop", None), getattr(system, "cleanup", None)):
            if close is not None:
                close()
        return 2

    print("press the start button")
    result = run(camera, sensors, motor, pipeline, system, route, max_run_s=args.max_run_s)
    print(summary(result))
    if result.error is not None:
        traceback.print_exception(result.error)
    screens = end_screen(result)
    if len(screens) > 1:
        print("Ctrl-C to exit")
    try:
        show_end(system, screens)
    finally:
        system.cleanup(blank=False)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
