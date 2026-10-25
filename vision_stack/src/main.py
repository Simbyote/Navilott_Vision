"""Production run: the start button, the course, the time on the display. Nothing recorded.

Purpose:
    What the robot runs on the course (python3 -m src.main): every frame
    through Pipeline.step(), its Command to the motors. The linkers are the
    instrumented twins. Every way a run ends stops the motors first, and the
    display says which it was.

Main package:
    RunResult: how the run ended, its time and the route step it reached.

Flow:
    1. Load the route; open the hardware; check the battery at rest.
    2. Warm Phase 2 up; "St N" until the button; the countdown.
    3. Per frame: camera -> Pipeline.step() -> motors; time on the display.
    4. Halt the motors, release the hardware, leave the end screen up.
"""
import argparse
import sys
import time
import traceback
from dataclasses import dataclass

from src.capture.camera import CameraSource, CaptureError
from src.config import MEASURED, MEASURED_ESTIMATION, ROUTE_PATH
from src.peripherals import battery_run
from src.peripherals.battery_run import CODE_BATTERY
from src.navigation.end_of_course import OUTCOME_EARLY
from src.navigation.route import Route, RouteError, load_route
from src.params import FPS, FRAME_H, FRAME_W, GC_COLLECT_EVERY, GC_DISABLE_IN_RUN
from src.pipeline import Pipeline
from src.utils import GcWatch
from src.perception import parallel

# A backstop in case the course's end is never seen; runs are well under a minute
MAX_RUN_S = 300.0
# A little longer than the drive's ~0.15-0.18 s coast, before the outputs float
HALT_BRAKE_S = 0.3
ALTERNATE_S = 2.0

END_FINISHED, END_EARLY, END_CAP, END_INTERRUPT, END_ERROR = (
    "finished", "ended early", "run time cap", "interrupted (Ctrl-C)", "error")
END_BATTERY = "battery critical"
CODE_CAP, CODE_ERROR = "E  t", "Err "


@dataclass(frozen=True)
class RunResult:
    ended_by: str
    elapsed_s: float
    frames: int
    end_step: int | None = None
    error: BaseException | None = None


def step_text(route: Route) -> str:
    return f"St{min(len(route.maneuvers), 99):>2}"


def early_code(step: int | None) -> str:
    return f"E{min(step or 0, 99):>3}"


def mm_ss(seconds: float) -> str:
    return f"{min(int(seconds) // 60, 99):02d}:{int(seconds) % 60:02d}"


def halt(motor, sleep=time.sleep) -> None:
    """Short brake for HALT_BRAKE_S, then standby; always ends with motor.stop(), even if the brake fails."""
    try:
        motor.brake()
        sleep(HALT_BRAKE_S)
    finally:
        motor.stop()


def run(camera, sensors, motor, pipeline: Pipeline, system, route: Route,
        clock=time.monotonic, sleep=time.sleep, max_run_s: float = MAX_RUN_S, battery=None,
        warmup: bool = True) -> RunResult:
    """
    One run, from the start button to the halt; the display is left for end_screen().
    Always halts the motors first, then releases the camera and sensors, whatever ended it.
    clock and sleep are injected by tests.
    """
    ended_by, error, frames, t0, gc_watch = END_INTERRUPT, None, 0, None, None
    try:
        fd = camera.read() if warmup else None
        if fd is not None:
            pipeline.warm_up(fd.frame)
        system.wait_for_start(step_text(route))
        system.run_countdown()
        sensors.sample()                                # start the first window at GO
        if battery is not None and not battery_run.start(battery):
            battery = None
        gc_watch = GcWatch(GC_DISABLE_IN_RUN, GC_COLLECT_EVERY).start()
        t0 = clock()
        while True:
            if clock() - t0 >= max_run_s:
                ended_by = END_CAP
                break
            if battery is not None and battery.should_stop():
                ended_by = END_BATTERY
                break
            fd = camera.read()
            if fd is None:                              # dropped: no id spent
                continue
            cmd = pipeline.step(fd.frame, fd.frame_id, fd.timestamp_ms, sensors.sample())
            if cmd.brake:
                motor.brake()
            else:
                motor.drive(cmd.left, cmd.right)
            frames += 1
            gc_watch.tick(cmd.brake)
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
            if gc_watch is not None:
                gc_watch.stop()
    elapsed = 0.0 if t0 is None else clock() - t0
    end_step = pipeline.navigation.end_step
    return RunResult(ended_by, elapsed, frames,
                     pipeline.navigation.progress.step if end_step is None else end_step, error)


def end_screen(result: RunResult) -> list:
    """The screens: a str is a code, a float the time. One stays up; two alternate every ALTERNATE_S."""
    if result.ended_by == END_ERROR:
        return [CODE_ERROR]
    if result.ended_by == END_EARLY:
        return [early_code(result.end_step), result.elapsed_s]
    if result.ended_by == END_CAP:
        return [CODE_CAP, result.elapsed_s]
    if result.ended_by == END_BATTERY:
        return [CODE_BATTERY, result.elapsed_s]
    return [result.elapsed_s]                           # finished, Ctrl-C


def show_end(system, screens: list, sleep=time.sleep) -> None:
    """Blocks while two screens alternate, until Ctrl-C."""
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
    line = f"{result.ended_by} at step {result.end_step} after {mm_ss(result.elapsed_s)} ({result.frames} frames)"
    return line if result.error is None else f"{line}: {result.error!r}"


def cli(argv: list[str] | None = None) -> int:
    """
    0 after any run; 2 for a bad route, hardware that won't open or a critical pack, before anything moves.
    """
    ap = argparse.ArgumentParser(description="Run the course: start button, drive, time on the display.")
    ap.add_argument("--route", default=str(ROUTE_PATH), metavar="PATH",
                    help="the course plan (JSON: maneuvers, finish); default config.ROUTE_PATH")
    ap.add_argument("--max-run-s", type=float, default=MAX_RUN_S,
                    help=f"brake and end after this long (default {MAX_RUN_S:.0f})")
    parallel.add_arguments(ap)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    parallel.apply(args)
    if parallel.enabled():
        print("perception: color and obstacle on a side thread (--serial to turn off)")

    try:
        route = load_route(args.route)
    except RouteError as exc:
        print(f"route error: {exc}")
        return 2
    print("\n".join(route.describe()))

    pipeline = Pipeline(MEASURED, MEASURED_ESTIMATION, route=route)
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

    battery = battery_run.open_battery()
    if battery is not None and battery_run.preflight(battery, motors_on=True)[0] == battery_run.REFUSE:
        system.show_text(CODE_BATTERY)
        for close in (camera.release, sensors.stop, motor.stop, battery.cleanup):
            close()
        system.cleanup(blank=False)                     # "Lo-b" stays up
        return 2

    print("press the start button")
    try:
        result = run(camera, sensors, motor, pipeline, system, route, max_run_s=args.max_run_s, battery=battery)
    finally:
        if battery is not None:
            battery.cleanup()
    print(summary(result))
    if battery is not None:
        print(f"battery  {battery.voltage():.2f} V at the end ({battery.state().name})")
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
