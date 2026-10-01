"""Phases 1-3 plus Navigation: the whole chain driving the robot, recorded and rendered afterwards.

Purpose:
    The integration test of the Navigation step, ahead of adding it to
    pipeline.py. Every frame runs capture, Phase 2 and the traced Phase 3
    (run_phase3_chain, reused from phase3_linker), hands the packet to the
    Navigation subsystem (navigation.Navigation: the stop sign, traffic light
    and intersection rules over lane keeping, which is everything behind it),
    checks the Command against the contract, and drives the motors with it. Nothing is drawn while the robot
    drives: frames and per-frame records go to disk on maneuver_linker's
    background recorder, and nav.avi is rendered once the run ends, so the
    run measures the real control loop.

    The motors only run with the camera. --no-motors is a bench dry run, and
    --video / --frames replays are always dry: they show what the navigator
    would have commanded on recorded footage.

Main package:
    run(): one run; returns its findings. The run folder holds nav.csv,
    p3.csv, frames/, records.pkl, report.json, summary.txt and, after
    rendering, nav.avi.
    NavStats: the [NAVIGATION] summary: how the run ended, which rule decided
    how many frames, time driving and braked (and why), steering, command
    latency.

Flow:
    1. Open the source, sensors, motors and start button; wait for the press.
    2. Per frame: read the camera and sensors, run the vision chain, ask the
       navigator, check the command, drive the motors, log, queue the frame.
    3. On the navigator finishing (end of course), the run-time cap, the
       source ending, Ctrl-C or an error: stop the motors first, flush the
       recorder, write the summary.
    4. Render nav.avi from the recording, and play it if there's a display.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import replace

import numpy as np

from src.capture.camera import CaptureError
from src.config import MANEUVER, MEASURED, MEASURED_ESTIMATION, ROUTE_PATH, PipelineConfig
from src.debugger.debug_navigation import VIDEO_FILE, render_run
from src.debugger.live_view import CameraFrameSource, DirectoryFrameSource, Display, VideoFrameSource
from src.estimation.estimation import Phase3Config
from src.maneuver_linker import FrameRecorder, _NoMotors, chain_record
from src.navigation.end_of_course import OUTCOME_EARLY
from src.navigation.navigation import BRAKE, Navigation, command_problems
from src.navigation.route import RouteError, load_route
from src.params import FPS, FRAME_H, FRAME_W, RUNS_DIR
from src.perception.color_branch import ColorConfig, load_hsv_ranges
from src.phase3_linker import CsvLog, Phase3Stats, Sensors, make_processor, run_phase3_chain

# The run's backstop: brake and end after this long, whatever the navigator
# does. The only linker-level safety stop (decided 2026-09-30); a run that
# reaches a stop line stays braked there until it
MAX_RUN_S = 30.0

END_CAP, END_SOURCE, END_INTERRUPT, END_ERROR, END_LIMIT = (
    "run time cap", "the source ended", "interrupted (Ctrl-C)", "error", "frame limit")
END_COURSE = "end of course"                    # the navigator finished the route
END_EARLY = "ended early (lane lost before the route was done)"
REASON_CONTRACT = "contract"        # the linker braked: the navigator's command broke the contract

NAV_FIELDS = ("frame_id", "t", "capture_ms", "phase2_ms", "phase3_ms", "nav_ms", "latency_ms",
              "rule", "phase", "step", "maneuver", "lane_mode", "lane_status", "lane_offset", "lane_offset_cm", "heading_error", "drive_state",
              "stop_sign", "stop_line_cm", "reason", "source", "steer",
              "cmd_left", "cmd_right", "brake", "left_cps", "right_cps", "event")

# --help text. Kept apart from the module docstring, which documents the code.
_CLI_HELP = """\
Drive the robot with the whole chain: camera -> perception -> estimation ->
navigation (stop sign, red light, intersection and end-of-course rules over
lane keeping) -> motors. The video is rendered after the run, so recording
never slows the steering.

SAFETY: with --camera the robot moves. Ctrl-C stops the motors. A run ends
when the lane stays lost (the end of the course: it creeps, then brakes),
or after --max-run-s, whichever comes first.

Examples (from vision_stack/, with sudo pigpiod running):
    python3 -m src.navigation_linker --camera                 # drive, start button
    python3 -m src.navigation_linker --camera --no-motors     # bench: everything but the motors
    python3 -m src.navigation_linker --camera --max-run-s 10 --cm-per-px 0.05
    python3 -m src.navigation_linker --video runs/run.avi     # what it would have commanded
    python3 -m src.navigation_linker --render runs/nav_20261001_101500

Output (--out DIR, default <root>/runs/nav_<timestamp>):
    summary.txt   how the run ended, time driving vs braked and why, steering,
                  command latency (frame in -> motors), then the Phase 3 summary
    report.json   the same, machine-readable
    nav.csv       every frame: timings, the packet fields the navigator used,
                  why it drove or braked, the steering and the command sent
    p3.csv        every frame's Phase 2 input and Phase 3 packet (as phase3_linker)
    nav.avi       rendered after the run: the Phase 3 video with a navigation strip
    frames/, records.pkl   what the video is rendered from
"""


# =============================================================================
# Findings
# =============================================================================

class NavStats:
    """Per-frame navigation outcomes, summarized for summary.txt and report.json."""
    def __init__(self) -> None:
        self.frames = self.driving = self.rejected = 0
        self.brake_reasons: Counter = Counter()
        self.sources: Counter = Counter()
        self.rules: Counter = Counter()
        self.steer_abs: list[float] = []
        self.latency_ms: list[float] = []

    def update(self, n: dict) -> None:
        """Count one frame's record["nav"]."""
        self.frames += 1
        self.latency_ms.append(n["latency_ms"])
        self.rules[n.get("rule") or "-"] += 1
        if n["reason"] == REASON_CONTRACT:
            self.rejected += 1
        if n["brake"]:
            self.brake_reasons[n["reason"]] += 1
        else:
            self.driving += 1
            self.sources[n["source"]] += 1
            self.steer_abs.append(abs(n["steer"]))

    def report(self) -> dict:
        lat = np.array(self.latency_ms) if self.latency_ms else np.zeros(1)
        steer = np.array(self.steer_abs) if self.steer_abs else np.zeros(1)
        return {"frames": self.frames, "driving": self.driving, "braked": self.frames - self.driving,
                "brake_reasons": dict(self.brake_reasons), "steer_sources": dict(self.sources),
                "decided_by": dict(self.rules),
                "steer_abs_mean": round(float(steer.mean()), 4), "steer_abs_max": round(float(steer.max()), 4),
                "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                               "p95": round(float(np.percentile(lat, 95)), 2),
                               "max": round(float(lat.max()), 2)},
                "rejected": self.rejected}


def summary_lines(report: dict) -> list[str]:
    """summary.txt's [NAVIGATION] section, from run()'s findings."""
    n, r = report["nav"], report["run"]
    pct = lambda k: 100.0 * k / n["frames"] if n["frames"] else 0.0
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(n["brake_reasons"].items())) or "none"
    sources = ", ".join(f"{k} {v}" for k, v in sorted(n["steer_sources"].items())) or "none"
    rules = ", ".join(f"{k} {v}" for k, v in sorted(n["decided_by"].items())) or "none"
    lat = n["latency_ms"]
    return [
        f"[NAVIGATION] ended by {report['ended_by']}"
        + ("" if report.get("end_step") is None else f" at step {report['end_step']}")
        + f"   motors {'ON' if report['motors'] else 'OFF (dry run)'}",
        f" run                   {r['frames']} frames in {r['wall_s']:.1f} s ({r['fps']:.1f} FPS), "
        f"camera drops {r['camera_drops']}, recorder dropped {r['recorder_dropped']}",
        f" decided by            {rules}",
        f" driving               {n['driving']} frames ({pct(n['driving']):.0f}%), steering by: {sources}",
        f" braked                {n['braked']} frames ({pct(n['braked']):.0f}%): {reasons}",
        f" steering |duty|       mean {n['steer_abs_mean']:.3f}, max {n['steer_abs_max']:.3f}",
        f" command latency       p50 {lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms, max {lat['max']:.1f} ms "
        f"(frame in -> motors)",
        f" contract              {n['rejected']} commands rejected and braked",
    ]


# =============================================================================
# The run
# =============================================================================

def run(
        source,
        sensors,
        motor,
        navigator,
        config: PipelineConfig = MEASURED,
        p3_config: Phase3Config = MEASURED_ESTIMATION,
        out_dir: str = str(RUNS_DIR / "nav"),
        system=None,
        clock=time.perf_counter,
        max_run_s: float = MAX_RUN_S,
        limit: int | None = None,
        motors_on: bool = True,
        render: bool = True,
        display: bool = False,
        scale: int = 1,
    ) -> dict:
    """
    Run the navigator on every frame until the cap, the source's end or Ctrl-C.

    Inputs:
        source: A live_view FrameSource: the camera, or a replay.
        sensors: phase3_linker.Sensors (anything with read() -> (SensorSample,
            batch) and stop()); None runs Phase 3 without sensors.
        motor: drive.MotorController, or anything with drive(left, right),
            brake() and stop(); _NoMotors for a dry run.
        navigator: A Navigator; navigation.Navigation. Its record, if it has
            one, says which rule decided each command and why.
        config, p3_config: Phase 2 and Phase 3 tuning, as phase3_linker.
        system: peripherals.system.System for the start button, countdown
            and run-time display; None starts at once.
        clock: Seconds, monotonic; injected by tests.
        max_run_s: Brake and end once this long has passed since the start.
        limit: Stop after this many frames (replays); None runs to the end.
        motors_on: Reported in the summary; False for a dry run.
        render: Render nav.avi after the run.
        display: Play the rendered video in a window afterwards.

    Outputs:
        The findings, also written to report.json.

    Side effects:
        Drives the motors. Writes the run folder. Always stops the motors
        before returning or raising, whatever ended the run.
    """
    os.makedirs(out_dir, exist_ok=True)
    navigator.reset()
    processor = None
    stats = Phase3Stats(budget_ms=1000.0 / max(source.fps, 1))
    nav_stats = NavStats()
    p3_log = CsvLog(os.path.join(out_dir, "p3.csv"))
    n_file = open(os.path.join(out_dir, "nav.csv"), "w", newline="")
    n_log = csv.DictWriter(n_file, NAV_FIELDS, extrasaction="ignore")
    n_log.writeheader()
    recorder = FrameRecorder(out_dir)
    camera_drops = 0
    ended_by, error = END_SOURCE, None
    last_reason = None
    t0 = None

    try:
        if system is not None:
            system.wait_for_start()
            system.run_countdown()
        if sensors is not None:
            sensors.read()                          # start every sensor window at the go
        t0 = clock()
        while True:
            if limit is not None and nav_stats.frames >= limit:
                ended_by = END_LIMIT
                break
            c0 = clock()
            if c0 - t0 >= max_run_s:
                ended_by = END_CAP
                break
            item = source.read()
            arrived = clock()
            capture_ms = (arrived - c0) * 1000.0
            if item is None:
                ended_by = END_SOURCE
                break
            frame, fid, ts = item
            if frame is None:
                camera_drops += 1
                continue

            sample = None if sensors is None else sensors.read()[0]
            if processor is None:
                processor = make_processor(frame, fid, ts, config, p3_config)
            res = run_phase3_chain(frame, fid, ts, processor, sample, config, capture_ms)
            pkt = res.packet

            n0 = clock()
            cmd = navigator.update(pkt)
            rec = dict(getattr(navigator, "record", {}) or {})
            problems = command_problems(cmd)
            if problems:                            # the harness enforces the contract, whatever the navigator
                cmd = BRAKE
                rec = {**rec, "reason": REASON_CONTRACT, "source": "none", "steer": 0.0}
            nav_ms = (clock() - n0) * 1000.0
            if cmd.brake:
                motor.brake()
            else:
                motor.drive(cmd.left, cmd.right)
            latency_ms = (clock() - arrived) * 1000.0

            reason = rec.get("reason", "brake" if cmd.brake else "drive")
            event = "; ".join(problems) if problems else ("" if reason == last_reason else f"-> {reason}")
            last_reason = reason
            n = {"frame_id": fid, "t": round(arrived - t0, 3), "capture_ms": round(capture_ms, 2),
                 "phase2_ms": round(res.timings_ms["phase2"], 2), "phase3_ms": round(res.timings_ms["phase3"], 3),
                 "nav_ms": round(nav_ms, 3), "latency_ms": round(latency_ms, 2),
                 "rule": rec.get("rule", ""), "phase": rec.get("phase", ""), "step": rec.get("step", ""),
                 "maneuver": rec.get("maneuver") or "", "lane_mode": pkt.lane_mode,
                 "lane_status": pkt.lane_status, "lane_offset": pkt.lane_offset,
                 "lane_offset_cm": pkt.lane_offset_cm, "heading_error": pkt.heading_error,
                 "drive_state": pkt.drive_state, "stop_sign": int(pkt.stop_sign_detected),
                 "stop_line_cm": pkt.stop_line_distance_cm if pkt.stop_line_detected else None,
                 "reason": reason, "source": rec.get("source", ""), "steer": rec.get("steer", 0.0),
                 "cmd_left": cmd.left, "cmd_right": cmd.right, "brake": int(cmd.brake),
                 "left_cps": pkt.left_wheel_cps, "right_cps": pkt.right_wheel_cps, "event": event}
            if event:
                print(f"  t={n['t']:6.2f}s  frame {fid}  {event}")
            stats.update(res)
            nav_stats.update(n)
            p3_log.write(res)
            n_log.writerow(n)
            recorder.put(fid, frame, {**chain_record(res), "nav": n})
            if system is not None:
                system.update_display(arrived - t0)
            if getattr(navigator, "finished", False):     # the navigator ended the run, braked, this frame
                ended_by = END_EARLY if getattr(navigator, "outcome", None) == OUTCOME_EARLY else END_COURSE
                break
    except KeyboardInterrupt:
        ended_by = END_INTERRUPT
    except Exception as exc:
        ended_by, error = END_ERROR, exc
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            traceback.print_exc(file=f)
    finally:
        motor.stop()                                # first, whatever happened
        try:
            recorder.close()
        finally:
            p3_log.close()
            n_file.close()
            if sensors is not None:
                sensors.stop()
            source.close()
            if system is not None:
                system.show_final_time(0.0 if t0 is None else clock() - t0)
                system.cleanup()

    wall = 0.0 if t0 is None else clock() - t0
    report = {"ended_by": ended_by if error is None else f"{END_ERROR}: {error!r}", "motors": motors_on,
              "outcome": getattr(navigator, "outcome", None), "end_step": getattr(navigator, "end_step", None),
              "nav": nav_stats.report(),
              "run": {"frames": nav_stats.frames, "wall_s": round(wall, 2),
                      "fps": round(nav_stats.frames / wall, 2) if wall > 0 else 0.0,
                      "camera_drops": camera_drops, "recorder_dropped": recorder.dropped,
                      "recorder_written": recorder.written}}
    lines = summary_lines(report) + [""] + stats.report()

    if render and recorder.written:
        window = Display(display, out_dir, ["navigation"])
        period = 1.0 / max(report["run"]["fps"], 1.0)

        def show(img):
            window.show({"navigation": img})
            if window.enabled:
                time.sleep(period)              # real speed; waitKey(1) alone would race
        try:
            rendered = render_run(out_dir, config.lane_offset, max(report["run"]["fps"], 1.0), scale, show)
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


# =============================================================================
# Command line
# =============================================================================

def cli(argv: list[str] | None = None) -> int:
    """
    Parse arguments, open the source and hardware, and run (or only render a run).

    Outputs:
        Process exit code: 0 after a run, 2 on a bad argument or a source or
        hardware that won't open.
    """
    ap = argparse.ArgumentParser(prog="navigation_linker", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true", help="live: the motors drive")
    src.add_argument("--video", metavar="PATH", help="replay a clip; motors always off")
    src.add_argument("--frames", metavar="DIR", help="replay an image sequence; motors always off")
    src.add_argument("--render", metavar="RUN_DIR", help="only render an earlier run's video")
    ap.add_argument("--no-motors", action="store_true", help="with --camera: everything but the motors")
    ap.add_argument("--no-button", action="store_true", help="with --camera: start after a 3 s countdown")
    ap.add_argument("--max-run-s", type=float, default=MAX_RUN_S,
                    help=f"brake and end after this long (default {MAX_RUN_S:.0f})")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--route", default=str(ROUTE_PATH), metavar="PATH",
                    help="the course plan (JSON: maneuvers, finish); default config.ROUTE_PATH")
    ap.add_argument("--gyro-bias", type=float, default=MANEUVER.gyro_bias_dps, metavar="DPS",
                    help=f"gyro Z at rest, + = right frame (default {MANEUVER.gyro_bias_dps}, config.MANEUVER)")
    ap.add_argument("--cm-per-px", type=float, default=None, metavar="S",
                    help="hand-measured ground scale; fills lane_offset_cm, which the navigator prefers")
    ap.add_argument("--hsv", default=None, metavar="PATH", help="HSV ranges instead of MEASURED's")
    ap.add_argument("--no-render", action="store_true", help="skip the video; render later with --render")
    ap.add_argument("--no-display", action="store_true", help="don't play the video afterwards")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--width", type=int, default=FRAME_W)
    ap.add_argument("--height", type=int, default=FRAME_H)
    ap.add_argument("--fps", type=int, default=None, help="capture/replay rate (video files default to their own)")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    config = MEASURED
    if args.hsv:
        config = replace(config, color=ColorConfig(load_hsv_ranges(args.hsv), config.color.blob))

    if args.render:
        fps = args.fps or FPS
        rep = os.path.join(args.render, "report.json")
        if os.path.exists(rep):
            with open(rep) as f:
                fps = json.load(f).get("run", {}).get("fps") or fps
        out = render_run(args.render, config.lane_offset, fps, args.scale)
        print(f"rendered {out['rendered']} frames ({out['missing']} missing) to "
              f"{os.path.join(args.render, VIDEO_FILE)}")
        return 0

    try:
        route = load_route(args.route)
    except RouteError as exc:
        print(f"route error: {exc}")
        return 2
    out_dir = args.out or str(RUNS_DIR / ("nav_" + time.strftime("%Y%m%d_%H%M%S")))
    p3_config = replace(MEASURED_ESTIMATION, gyro_bias_dps=args.gyro_bias, cm_per_px=args.cm_per_px)
    motors_on = bool(args.camera and not args.no_motors)
    source = sensors = motor = system = None
    try:
        if args.camera:
            source = CameraFrameSource(args.width, args.height, args.fps or FPS)
            sensors = Sensors(imu=True, encoders=True)
        elif args.video:
            source = VideoFrameSource(args.video, args.fps)
        else:
            source = DirectoryFrameSource(args.frames, args.fps or FPS)
        if motors_on:
            import pigpio
            from src.peripherals.drive import MotorController
            motor = MotorController(pigpio.pi())
        else:
            motor = _NoMotors()
        if args.camera and not args.no_button:
            from src.peripherals.system import System
            system = System()
    except (CaptureError, OSError, RuntimeError, ImportError) as exc:
        print(f"source / hardware error: {exc!r}")
        for thing in (motor, sensors, source):
            if thing is not None:
                (thing.stop if hasattr(thing, "stop") else thing.close)()
        return 2

    print(f"source   {source.label} @ {source.fps:.0f} FPS")
    print(f"output   {out_dir}")
    print(f"motors   {'ON' if motors_on else 'OFF (dry run)'}   cap {args.max_run_s:.0f} s   "
          f"cm/px {args.cm_per_px if args.cm_per_px else 'uncalibrated'}")
    print("\n".join(route.describe()))
    if args.camera and system is None:
        for n in (3, 2, 1):
            print(f"  starting in {n}")
            time.sleep(1.0)
    elif system is not None:
        print("press the start button")
    run(source, sensors, motor, Navigation(gyro_bias_dps=args.gyro_bias, route=route), config, p3_config,
        out_dir, system,
        max_run_s=args.max_run_s, limit=args.limit, motors_on=motors_on,
        render=not args.no_render, display=not args.no_display, scale=args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
