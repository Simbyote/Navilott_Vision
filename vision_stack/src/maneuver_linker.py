"""Phase 1-3 plus the motors: a scripted drive trial that records everything and renders it afterwards.

Purpose:
    The integration test between driving, the IMU, the encoders and the
    vision pipeline, ahead of the Navigation state machine. The robot
    settles, checks the IMU's yaw sign with two short spins, drives a leg
    kept straight by the encoders and the gyro, turns 180 degrees on the
    gyro, and drives back (src/maneuver.py holds that logic, hardware-free).
    Every frame still runs the whole vision chain, reused from the upstream
    linkers (run_phase3_chain: phase2_linker's run_chain, then the traced
    Phase 3), but vision only records: it never steers. Nothing is drawn
    while the robot moves; frames and per-frame records go to disk on a
    background thread, and the video is rendered once the trial ends or is
    interrupted.

Main package:
    run(): one trial; returns its findings (Maneuver.report() plus timing,
    lane re-acquisition after the turn, and recorder drops). The run folder
    holds maneuver.csv, p3.csv, frames/, records.pkl, config.json,
    report.json, summary.txt and, after rendering, maneuver.avi.

Flow:
    1. Open the camera, sensors, motors and start button; wait for the press.
    2. Per frame: read the camera and sensors, step the maneuver, drive the
       motors, run the vision chain, log, queue the frame for the recorder.
    3. On the end, an abort or Ctrl-C: stop the motors first, flush the
       recorder, write the summary.
    4. Render maneuver.avi from the recording, and play it if there's a display.
"""
import argparse
import csv
import json
import os
import pickle
import queue
import select
import sys
import threading
import time
import traceback
from dataclasses import asdict, fields, replace

import cv2

from src.config import MANEUVER, MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.debugger.debug_maneuver import FRAMES_DIR, RECORDS_FILE, frame_path, render_run
from src.debugger.live_view import CameraFrameSource, Display
from src.estimation.estimation import LANE_VISION, Phase3Config
from src.debugger.estimation_debug import TracedPhase3Processor
from src.maneuver import FORWARD_2, Maneuver, ManeuverConfig, Tick
from src.params import FPS, FRAME_H, FRAME_W, MODE_TWO_BOUNDARY, RUNS_DIR
from src.perception.color_branch import ColorConfig, load_hsv_ranges
from src.peripherals.sensing import Sensors
from src.phase3_linker import CsvLog, Phase3Stats, run_phase3_chain

JPEG_QUALITY = 90       # frames are only a video background; decisions come from the records
QUEUE_FRAMES = 64       # ~2.5 s at 25 FPS of slack before the recorder drops frames

# --help text. Kept apart from the module docstring, which documents the code.
_CLI_HELP = """\
Drive trial: settle, spin-check the IMU's yaw sign, drive a leg, turn 180,
drive back, while the vision pipeline records. Straightness and the turn use
the encoders and the IMU only. The video is rendered after the run.

SAFETY: the robot moves. Place it at the end of the mat furthest from the
intersection, pointing down the lane, with room for the leg length ahead.
Ctrl-C stops the motors. A run also stops itself on a stall, a late frame,
a turn past 270 degrees or --max-run-s.

Examples (from vision_stack/, with sudo pigpiod running):
    python3 -m src.maneuver_linker                       # full trial, start button
    python3 -m src.maneuver_linker --leg-counts 2000 --speed 0.35
    python3 -m src.maneuver_linker --set turn_slow_band_deg=30 --set kp_heading=0.02
    python3 -m src.maneuver_linker --no-motors --no-button   # bench: nothing moves
    python3 -m src.maneuver_linker --hold                 # stop after each step to measure it
    python3 -m src.maneuver_linker --render runs/maneuver_20261001_101500

Output (--out DIR, default <root>/runs/maneuver_<timestamp>):
    summary.txt     the findings: gyro bias at rest, yaw sign, each leg, the
                    turn's PASS / FAIL, timings, the Phase 3 summary
    report.json     the same findings, machine-readable
    maneuver.csv    every frame: step, motor commands, corrections, encoder
                    counts and counts per second, yaw, heading, turn angle
    p3.csv          every frame's Phase 2 input and Phase 3 packet (as phase3_linker)
    maneuver.avi    rendered after the run: the Phase 3 video with a
                    maneuver strip under it
    config.json     the ManeuverConfig this run used
    frames/, records.pkl   what the video is rendered from
"""


# =============================================================================
# Recording
# =============================================================================

class FrameRecorder:
    """
    Writes each frame's JPEG and pickled record on a background thread, so the
    control loop never waits on the disk.

    A full queue drops the frame (counted in dropped) rather than blocking:
    the video loses a frame, the robot doesn't lose a control step. The CSVs
    are written in the loop and are always complete.
    """
    def __init__(self, out_dir: str, maxsize: int = QUEUE_FRAMES) -> None:
        self.out_dir = out_dir
        os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
        self.dropped = self.written = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._records = open(os.path.join(out_dir, RECORDS_FILE), "wb")
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._work, name="frame-recorder", daemon=True)
        self._thread.start()

    def put(self, frame_id: int, frame, record: dict) -> None:
        """Queue one frame and its record; copies the frame, since the camera may reuse its buffer."""
        try:
            self._q.put_nowait((frame_id, frame.copy(), record))
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        try:
            while (item := self._q.get()) is not None:
                fid, frame, record = item
                cv2.imwrite(frame_path(self.out_dir, fid), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                pickle.dump(record, self._records, protocol=pickle.HIGHEST_PROTOCOL)
                self.written += 1
        except BaseException as exc:            # reported by close(); the loop must not die with it
            self._error = exc

    def close(self) -> None:
        """Flush everything queued, then close. Raises what the thread hit, if anything."""
        self._q.put(None)
        self._thread.join()
        self._records.close()
        if self._error is not None:
            raise RuntimeError(f"frame recorder failed: {self._error!r}") from self._error


def chain_record(res) -> dict:
    """
    What debug_maneuver.as_result() needs to redraw a frame's Phase 3 view,
    without the images the chain carries. Shared with navigation_linker.
    """
    chain = res.chain
    return {"frame_id": res.packet.frame_id, "timestamp_ms": res.packet.timestamp_ms,
            "geometry": chain.geometry, "offset": chain.offset, "offset_debug": chain.offset_debug,
            "lane_rect": chain.roi.lane_rect, "packet": res.packet, "p3_debug": res.p3_debug,
            "timings": dict(res.timings_ms)}


def _record(res, machine: Maneuver) -> dict:
    """What render_run() needs to redraw this frame: the chain plus the maneuver's record."""
    return {**chain_record(res), "maneuver": dict(machine.record)}


# =============================================================================
# The trial
# =============================================================================

class Resume:
    """
    --hold's continue signal: the start button (System.button_pressed(),
    non-blocking) or Enter in the terminal, whichever comes first. Polled
    once per frame while holding, so the loop never waits on it.
    """
    def __init__(self, system=None, stdin=None) -> None:
        self._system = system
        self._stdin = sys.stdin if stdin is None else stdin
        self._tty = hasattr(self._stdin, "isatty") and self._stdin.isatty()

    def __call__(self) -> bool:
        if self._system is not None and self._system.button_pressed():
            return True
        if self._tty and select.select([self._stdin], [], [], 0)[0]:
            self._stdin.readline()
            return True
        return False


class _NoMotors:
    """--no-motors: commands are logged in maneuver.csv but never sent."""
    def drive(self, left: float, right: float) -> None:
        pass

    def brake(self) -> None:
        pass

    def stop(self) -> None:
        pass


def run(
        source,
        sensors,
        motor,
        cfg: ManeuverConfig = MANEUVER,
        config: PipelineConfig = MEASURED,
        p3_config: Phase3Config = MEASURED_ESTIMATION,
        out_dir: str = str(RUNS_DIR / "maneuver"),
        system=None,
        clock=time.perf_counter,
        render: bool = True,
        display: bool = False,
        scale: int = 1,
        hold: bool = False,
        resume=None,
    ) -> dict:
    """
    Run one drive trial.

    Inputs:
        source: A live_view FrameSource; the camera on the robot.
        sensors: sensing.Sensors with the IMU and encoders started
            (anything with read() -> (SensorSample, imu_frame, encoder_frame)
            and stop()).
        motor: drive.MotorController, or anything with drive(left, right),
            brake() and stop(); _NoMotors for a bench run.
        cfg: The trial's tuning; MANEUVER from config.py plus any flags.
        config, p3_config: Phase 2 and Phase 3 tuning, as phase3_linker.
        system: peripherals.system.System for the start button, countdown
            and run-time display; None starts at once.
        clock: Seconds, monotonic; injected by tests.
        render: Render maneuver.avi after the run.
        display: Play the rendered video in a window afterwards.
        hold: Brake after the pulses, each leg and the turn until resumed,
            so each step can be measured by hand. Hold time doesn't count
            against max_run_s.
        resume: Called once per frame while holding; True continues.
            Defaults to Resume(system): the start button or Enter.

    Outputs:
        The findings (see the module docstring), also written to report.json.

    Side effects:
        Drives the motors. Writes the run folder. Always stops the motors
        before returning or raising, whatever ended the run.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    machine = Maneuver(cfg, hold=hold)
    resume = resume if resume is not None else Resume(system)
    processor = TracedPhase3Processor(p3_config)
    stats = Phase3Stats(budget_ms=1000.0 / max(source.fps, 1))
    p3_log = CsvLog(os.path.join(out_dir, "p3.csv"))
    m_file = open(os.path.join(out_dir, "maneuver.csv"), "w", newline="")
    m_log = csv.writer(m_file)
    m_log.writerow(("frame_id", *Maneuver.RECORD_FIELDS, "capture_ms", "phase2_ms", "phase3_ms",
                    "lane_status", "lane_offset", "p2_mode"))
    recorder = FrameRecorder(out_dir)
    extra = {"camera_drops": 0}
    reacquire = {"after_turn_t": None, "lane_reacquired_s": None}
    error = None
    t0 = prev = None

    try:
        if system is not None:
            system.wait_for_start()
            system.run_countdown()
        sensors.read()                              # start every sensor window at the go
        t0 = prev = clock()
        while not machine.done:
            c0 = clock()
            item = source.read()
            capture_ms = (clock() - c0) * 1000.0
            if item is None:
                machine.abort("the camera stopped delivering frames")
                break
            frame, fid, ts = item
            if frame is None:
                extra["camera_drops"] += 1
                continue

            now = clock()
            dt, prev = (now - prev if stats.frames else 0.0), now
            sample, batch = sensors.read()
            if machine.holding and resume():
                machine.resume()
            tick = Tick(t=now - t0, dt=dt,
                        yaw_dps=None if sample is None else sample.yaw_rate_dps,
                        lateral_accel=None if sample is None else sample.lateral_accel_mps2,
                        left_count=0 if batch is None or batch.left_count is None else batch.left_count,
                        right_count=0 if batch is None or batch.right_count is None else batch.right_count,
                        left_cps=0.0 if sample is None or sample.left_wheel_cps is None else sample.left_wheel_cps,
                        right_cps=0.0 if sample is None or sample.right_wheel_cps is None else sample.right_wheel_cps)
            cmd = machine.step(tick)
            # Before vision: control never waits on it
            if cmd.brake:
                motor.brake()
            else:
                motor.drive(cmd.left, cmd.right)
            if machine.record.get("event"):
                print(f"  t={tick.t:6.2f}s  {machine.record['event']}")

            res = run_phase3_chain(frame, fid, ts, processor, sample, config, capture_ms)
            stats.update(res)
            p3_log.write(res)
            m = machine.record
            m_log.writerow((fid, *(m.get(k, "") for k in Maneuver.RECORD_FIELDS),
                            round(capture_ms, 2), round(res.timings_ms["phase2"], 2),
                            round(res.timings_ms["phase3"], 3), res.packet.lane_status,
                            res.packet.lane_offset, res.chain.offset.mode))
            recorder.put(fid, frame, _record(res, machine))

            # Vision after the turn: how long until the lane is found again
            if m.get("event", "").startswith(f"-> {FORWARD_2}"):
                reacquire["after_turn_t"] = tick.t
            if (reacquire["after_turn_t"] is not None and reacquire["lane_reacquired_s"] is None
                    and res.packet.lane_status == LANE_VISION and res.chain.offset.mode == MODE_TWO_BOUNDARY):
                reacquire["lane_reacquired_s"] = round(tick.t - reacquire["after_turn_t"], 3)
            if system is not None:
                system.update_display(tick.t)
    except KeyboardInterrupt:
        machine.abort("interrupted (Ctrl-C)")
    except Exception as exc:
        error = exc
        machine.abort(f"error: {exc!r}")
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            traceback.print_exc(file=f)
    finally:
        motor.stop()                                # first, whatever happened
        try:
            recorder.close()
        finally:
            p3_log.close()
            m_file.close()
            sensors.stop()
            source.close()
            if system is not None:
                system.show_final_time(0.0 if t0 is None else clock() - t0)
                system.cleanup(blank=False)          # the final time stays up

    wall = 0.0 if t0 is None else clock() - t0
    report = machine.report()
    if "turn" in report:
        report["turn"]["lane_reacquired_s"] = reacquire["lane_reacquired_s"]
    report["run"] = {"frames": stats.frames, "wall_s": round(wall, 2),
                     "fps": round(stats.frames / wall, 2) if wall > 0 else 0.0,
                     "camera_drops": extra["camera_drops"], "recorder_dropped": recorder.dropped,
                     "recorder_written": recorder.written}
    lines = summary_lines(report, cfg) + [""] + stats.report()

    if render and recorder.written:
        window = Display(display, out_dir, ["maneuver"])
        period = 1.0 / max(report["run"]["fps"], 1.0)

        def show(img):
            window.show({"maneuver": img})
            if window.enabled:
                time.sleep(period)              # real speed; waitKey(1) alone would race
        try:
            rendered = render_run(out_dir, config.lane_offset, cfg, max(report["run"]["fps"], 1.0),
                                  scale, show)
        finally:
            window.close()
        report["run"]["video_frames"] = rendered["rendered"]
        lines += [""] + rendered["report"]

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    if error is not None:
        raise error
    return report


def summary_lines(report: dict, cfg: ManeuverConfig) -> list[str]:
    """summary.txt's [MANEUVER] section, from run()'s findings."""
    out = ["[MANEUVER] " + ("completed" if report["completed"] else f"STOPPED: {report['abort_reason']}")]
    if "settle" in report:
        s = report["settle"]
        out.append(f" gyro bias at rest     {s['gyro_bias_measured_dps']:+.3f} deg/s measured "
                   f"(configured {s['gyro_bias_configured_dps']:+.3f}), noise sd {s['gyro_noise_sd_dps']:.3f}")
        if "lateral_accel_baseline" in s:
            out.append(f" lateral accel at rest {s['lateral_accel_baseline']:+.3f} m/s^2 (the mount's tilt shows here)")
    if "yaw_sign" in report:
        y = report["yaw_sign"]
        pulse = lambda k: "--" if k not in y else f"{y[k]:+.1f} deg"
        out.append(f" yaw sign              + yaw = {y['plus_yaw_is'] or 'undecided'}  "
                   f"(left pulse {pulse('left_deg')}, right pulse {pulse('right_deg')})")
    for name in ("forward_1", "forward_2"):
        if name in report:
            g = report[name]
            out.append(f" {name:<21} counts L {g['left_counts']} R {g['right_counts']} "
                       f"(imbalance {g['imbalance_pct']:+.2f}%), {g['duration_s']:.2f} s, ended by {g['ended_by']}, "
                       f"heading at end {g['heading_end_deg']:+.1f} deg, max correction counts "
                       f"{g['max_c_counts']:.3f} heading {g['max_c_heading']:.3f}")
    if "turn" in report:
        t = report["turn"]
        verdict = "PASS" if t.get("success") else "FAIL"
        line = f" turn 180              {verdict}"
        if "final_deg" in t:
            line += f"  final {t['final_deg']:.1f} deg (target {cfg.turn_target_deg:.0f} +/- {cfg.turn_tolerance_deg:.0f})"
        if t.get("time_to_target_s") is not None:
            line += f", reached in {t['time_to_target_s']:.2f} s (limit {cfg.turn_timeout_s:.0f})"
        if t.get("overshoot_deg") is not None:
            line += f", overshoot {t['overshoot_deg']:+.1f}"
        out.append(line)
        if t.get("reason"):
            out.append(f"   why: {t['reason']}")
        if t.get("counts_per_deg") is not None:
            out.append(f"   encoders during the turn: L {t['left_counts']} R {t['right_counts']}, "
                       f"{t['counts_per_deg']:.2f} counts per degree")
        if "lane_reacquired_s" in t:
            v = t["lane_reacquired_s"]
            out.append("   lane found again     " + ("never" if v is None else f"{v:.2f} s after the turn"))
    for h in report.get("holds", ()):
        held = f"held {h['held_s']:.1f} s" if "held_s" in h else "still holding when the run ended"
        out.append(f" hold {h['point']:<16} at t={h['t']:.1f} s, counts L {h['left_count']} R {h['right_count']}, "
                   f"{held}: {h['measure']}")
    r = report.get("run")
    if r:
        out.append(f" run                   {r['frames']} frames in {r['wall_s']:.1f} s ({r['fps']:.1f} FPS), "
                   f"camera drops {r['camera_drops']}, recorder dropped {r['recorder_dropped']}")
    return out


# =============================================================================
# Command line
# =============================================================================

# Flags for the fields people will change most; --set reaches any field
_FLAGS = {"leg_counts": "--leg-counts", "leg_max_s": "--leg-max-s", "speed": "--speed",
          "turn_speed": "--turn-speed", "gyro_bias_dps": "--gyro-bias", "kp_counts": "--kp-counts",
          "kp_heading": "--kp-heading", "turn_tolerance_deg": "--turn-tolerance",
          "turn_timeout_s": "--turn-timeout", "max_run_s": "--max-run-s"}


def trial_config(base: ManeuverConfig, args) -> ManeuverConfig:
    """base with the named flags and every --set field=value applied; raises ValueError on an unknown field."""
    types_ = {f.name: f.type for f in fields(ManeuverConfig)}
    changes = {name: getattr(args, flag[2:].replace("-", "_"))
               for name, flag in _FLAGS.items() if getattr(args, flag[2:].replace("-", "_")) is not None}
    for item in args.set or ():
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in types_ or not value:
            raise ValueError(f"--set {item!r}: fields are {', '.join(sorted(types_))}")
        changes[key] = int(value) if types_[key] in (int, "int") else float(value)
    return replace(base, **changes)


def cli(argv: list[str] | None = None) -> int:
    """
    Parse arguments, open the hardware, and run one trial (or only render one).

    Outputs:
        Process exit code: 0 when the trial completed, 1 when it stopped
        early, 2 on a bad argument or hardware that won't open.
    """
    ap = argparse.ArgumentParser(prog="maneuver_linker", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, flag in _FLAGS.items():
        ap.add_argument(flag, type=type(getattr(MANEUVER, name)), default=None,
                        help=f"default {getattr(MANEUVER, name)} (config.MANEUVER.{name})")
    ap.add_argument("--set", action="append", metavar="FIELD=VALUE",
                    help="override any ManeuverConfig field for this run; repeatable")
    ap.add_argument("--hold", action="store_true",
                    help="brake after the spin pulses, each leg and the turn until the start "
                         "button (or Enter) is pressed, to measure each step by hand")
    ap.add_argument("--no-motors", action="store_true",
                    help="bench run: everything but the motors; stops at the yaw-sign check")
    ap.add_argument("--no-button", action="store_true", help="start after a 3 s console countdown")
    ap.add_argument("--no-render", action="store_true", help="skip the video; render later with --render")
    ap.add_argument("--no-display", action="store_true", help="don't play the video afterwards")
    ap.add_argument("--render", metavar="RUN_DIR", help="only render an earlier run's video")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--width", type=int, default=FRAME_W)
    ap.add_argument("--height", type=int, default=FRAME_H)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--hsv", default=None, metavar="PATH", help="HSV ranges instead of MEASURED's")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        cfg = trial_config(MANEUVER, args)
    except ValueError as exc:
        print(exc)
        return 2
    config = MEASURED
    if args.hsv:
        config = replace(config, color=ColorConfig(load_hsv_ranges(args.hsv), config.color.blob))

    if args.render:
        saved = os.path.join(args.render, "config.json")
        if os.path.exists(saved):
            with open(saved) as f:
                cfg = ManeuverConfig(**json.load(f))
        fps = args.fps
        rep = os.path.join(args.render, "report.json")
        if os.path.exists(rep):
            with open(rep) as f:
                fps = json.load(f).get("run", {}).get("fps") or fps
        out = render_run(args.render, config.lane_offset, cfg, fps, args.scale)
        print(f"rendered {out['rendered']} frames ({out['missing']} missing) to "
              f"{os.path.join(args.render, 'maneuver.avi')}")
        return 0

    out_dir = args.out or str(RUNS_DIR / ("maneuver_" + time.strftime("%Y%m%d_%H%M%S")))
    p3_config = replace(MEASURED_ESTIMATION, gyro_bias_dps=cfg.gyro_bias_dps)
    source = sensors = motor = system = None
    try:
        source = CameraFrameSource(args.width, args.height, args.fps)
        sensors = Sensors(imu=True, encoders=True)
        if args.no_motors:
            motor = _NoMotors()
        else:
            import pigpio
            from src.peripherals.drive import MotorController
            motor = MotorController(pigpio.pi())
        if not args.no_button:
            from src.peripherals.system import System
            system = System()
    except Exception as exc:
        print(f"hardware error: {exc!r}")
        for thing in (motor, sensors, source):
            if thing is not None:
                (thing.stop if hasattr(thing, "stop") else thing.close)()
        return 2

    print(f"output   {out_dir}")
    print(f"trial    leg {cfg.leg_counts} counts (cap {cfg.leg_max_s:.0f} s), speed {cfg.speed}, "
          f"turn {cfg.turn_target_deg:.0f} +/- {cfg.turn_tolerance_deg:.0f} deg in {cfg.turn_timeout_s:.0f} s"
          + ("   MOTORS OFF" if args.no_motors else ""))
    if system is None:
        for n in (3, 2, 1):
            print(f"  starting in {n}")
            time.sleep(1.0)
    else:
        print("press the start button")
    report = run(source, sensors, motor, cfg, config, p3_config, out_dir, system,
                 render=not args.no_render, display=not args.no_display, scale=args.scale,
                 hold=args.hold)
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.exit(cli())
