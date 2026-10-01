"""Stop line crossing driver.

Purpose:
    Drives the robot using the vision + navigation chain until a stop line is 
    detected and subsequently disappears (crossing the line). Does not record video 
    or save frame records to disk.
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
from src.debugger.live_view import CameraFrameSource, DirectoryFrameSource, VideoFrameSource
from src.estimation.estimation import Phase3Config
from src.maneuver_linker import _NoMotors
from src.navigation.navigation import Navigation, enforce
from src.navigation.route import RouteError, load_route
from src.params import FPS, FRAME_H, FRAME_W, RUNS_DIR
from src.perception.color_branch import ColorConfig, load_hsv_ranges
from src.phase3_linker import CsvLog, Phase3Stats, Sensors, make_processor, run_phase3_chain

MAX_RUN_S = 30.0

END_CAP, END_SOURCE, END_INTERRUPT, END_ERROR, END_LIMIT = (
    "run time cap", "the source ended", "interrupted (Ctrl-C)", "error", "frame limit"
)
END_STOP_LINE_CROSSED = "stop line crossed"
REASON_CONTRACT = "contract"

NAV_FIELDS = (
    "frame_id", "t", "capture_ms", "phase2_ms", "phase3_ms", "nav_ms", "latency_ms",
    "rule", "phase", "step", "maneuver", "lane_mode", "lane_status", "lane_offset", 
    "lane_offset_cm", "heading_error", "drive_state", "stop_sign", "stop_line_cm", 
    "reason", "source", "steer", "cmd_left", "cmd_right", "brake", "left_cps", 
    "right_cps", "event"
)

# =============================================================================
# Findings
# =============================================================================

class NavStats:
    """Per-frame navigation outcomes summarized for logging."""
    def __init__(self) -> None:
        self.frames = self.driving = self.rejected = 0
        self.brake_reasons: Counter = Counter()
        self.sources: Counter = Counter()
        self.rules: Counter = Counter()
        self.steer_abs: list[float] = []
        self.latency_ms: list[float] = []

    def update(self, n: dict) -> None:
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
        return {
            "frames": self.frames, "driving": self.driving, "braked": self.frames - self.driving,
            "brake_reasons": dict(self.brake_reasons), "steer_sources": dict(self.sources),
            "decided_by": dict(self.rules),
            "steer_abs_mean": round(float(steer.mean()), 4), "steer_abs_max": round(float(steer.max()), 4),
            "latency_ms": {
                "p50": round(float(np.percentile(lat, 50)), 2),
                "p95": round(float(np.percentile(lat, 95)), 2),
                "max": round(float(lat.max()), 2)
            },
            "rejected": self.rejected
        }


def summary_lines(report: dict) -> list[str]:
    n, r = report["nav"], report["run"]
    pct = lambda k: 100.0 * k / n["frames"] if n["frames"] else 0.0
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(n["brake_reasons"].items())) or "none"
    sources = ", ".join(f"{k} {v}" for k, v in sorted(n["steer_sources"].items())) or "none"
    rules = ", ".join(f"{k} {v}" for k, v in sorted(n["decided_by"].items())) or "none"
    lat = n["latency_ms"]
    return [
        f"[NAVIGATION] ended by {report['ended_by']} "
        + f" motors {'ON' if report['motors'] else 'OFF (dry run)'}",
        f" run                   {r['frames']} frames in {r['wall_s']:.1f} s ({r['fps']:.1f} FPS), "
        f"camera drops {r['camera_drops']}",
        f" decided by            {rules}",
        f" driving               {n['driving']} frames ({pct(n['driving']):.0f}%), steering by: {sources}",
        f" braked                {n['braked']} frames ({pct(n['braked']):.0f}%): {reasons}",
        f" steering |duty|       mean {n['steer_abs_mean']:.3f}, max {n['steer_abs_max']:.3f}",
        f" command latency       p50 {lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms, max {lat['max']:.1f} ms",
        f" contract              {n['rejected']} commands rejected and braked",
    ]


# =============================================================================
# The Run Loop
# =============================================================================

def run(
        source,
        sensors,
        motor,
        navigator,
        config: PipelineConfig = MEASURED,
        p3_config: Phase3Config = MEASURED_ESTIMATION,
        out_dir: str = str(RUNS_DIR / "nav_stop_line"),
        system=None,
        clock=time.perf_counter,
        max_run_s: float = MAX_RUN_S,
        limit: int | None = None,
        motors_on: bool = True,
    ) -> dict:

    os.makedirs(out_dir, exist_ok=True)
    navigator.reset()
    processor = None
    stats = Phase3Stats(budget_ms=1000.0 / max(source.fps, 1))
    nav_stats = NavStats()
    p3_log = CsvLog(os.path.join(out_dir, "p3.csv"))
    n_file = open(os.path.join(out_dir, "nav.csv"), "w", newline="")
    n_log = csv.DictWriter(n_file, NAV_FIELDS, extrasaction="ignore")
    n_log.writeheader()

    camera_drops = 0
    ended_by, error = END_SOURCE, None
    last_reason = None
    t0 = None

    # Track line crossing state
    stop_line_seen = False

    try:
        if system is not None:
            system.wait_for_start()
            system.run_countdown()
        if sensors is not None:
            sensors.read()

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

            # Check stop line visibility transition
            if pkt.stop_line_detected:
                stop_line_seen = True
            elif stop_line_seen and not pkt.stop_line_detected:
                # Line was previously visible and is now out of sight (crossed)
                ended_by = END_STOP_LINE_CROSSED
                motor.brake()
                break

            n0 = clock()
            cmd, problems = enforce(navigator.update(pkt))
            rec = dict(getattr(navigator, "record", {}) or {})
            if problems:
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
            n = {
                "frame_id": fid, "t": round(arrived - t0, 3), "capture_ms": round(capture_ms, 2),
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
                "left_cps": pkt.left_wheel_cps, "right_cps": pkt.right_wheel_cps, "event": event
            }
            if event:
                print(f"  t={n['t']:6.2f}s  frame {fid}  {event}")

            stats.update(res)
            nav_stats.update(n)
            p3_log.write(res)
            n_log.writerow(n)

            if system is not None:
                system.update_display(arrived - t0)

    except KeyboardInterrupt:
        ended_by = END_INTERRUPT
    except Exception as exc:
        ended_by, error = END_ERROR, exc
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            traceback.print_exc(file=f)
    finally:
        motor.stop()
        p3_log.close()
        n_file.close()
        if sensors is not None:
            sensors.stop()
        source.close()
        if system is not None:
            system.show_final_time(0.0 if t0 is None else clock() - t0)
            system.cleanup(blank=False)

    wall = 0.0 if t0 is None else clock() - t0
    report = {
        "ended_by": ended_by if error is None else f"{END_ERROR}: {error!r}", 
        "motors": motors_on,
        "nav": nav_stats.report(),
        "run": {
            "frames": nav_stats.frames, "wall_s": round(wall, 2),
            "fps": round(nav_stats.frames / wall, 2) if wall > 0 else 0.0,
            "camera_drops": camera_drops
        }
    }
    lines = summary_lines(report) + [""] + stats.report()

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    
    print("\n" + "\n".join(lines))
    if error is not None:
        raise error
    return report


# =============================================================================
# Command Line Interface
# =============================================================================

def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Drive until stop line crossing without recording.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true", help="live mode with camera")
    src.add_argument("--video", metavar="PATH", help="replay a video clip")
    src.add_argument("--frames", metavar="DIR", help="replay image frames directory")
    ap.add_argument("--no-motors", action="store_true", help="bench dry run")
    ap.add_argument("--no-button", action="store_true", help="start after 3s countdown")
    ap.add_argument("--max-run-s", type=float, default=MAX_RUN_S, help="maximum run duration")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--route", default=str(ROUTE_PATH), help="path to route plan")
    ap.add_argument("--gyro-bias", type=float, default=MANEUVER.gyro_bias_dps)
    ap.add_argument("--cm-per-px", type=float, default=None)
    ap.add_argument("--hsv", default=None, help="custom HSV ranges file")
    ap.add_argument("--width", type=int, default=FRAME_W)
    ap.add_argument("--height", type=int, default=FRAME_H)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    config = MEASURED
    if args.hsv:
        config = replace(config, color=ColorConfig(load_hsv_ranges(args.hsv), config.color.blob))

    try:
        route = load_route(args.route)
    except RouteError as exc:
        print(f"route error: {exc}")
        return 2

    out_dir = args.out or str(RUNS_DIR / ("nav_stop_line_" + time.strftime("%Y%m%d_%H%M%S")))
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
        print(f"hardware setup error: {exc!r}")
        for thing in (motor, sensors, source):
            if thing is not None:
                (thing.stop if hasattr(thing, "stop") else thing.close)()
        return 2

    if args.camera and system is None:
        for n in (3, 2, 1):
            print(f"  starting in {n}")
            time.sleep(1.0)
    elif system is not None:
        print("press the start button")

    run(
        source, sensors, motor, 
        Navigation(gyro_bias_dps=args.gyro_bias, route=route), 
        config, p3_config, out_dir, system,
        max_run_s=args.max_run_s, limit=args.limit, motors_on=motors_on
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli())