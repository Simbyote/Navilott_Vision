#!/usr/bin/env python3
"""Lane calibration: where a centered robot sees its lane, and how wide the lane is in px.

Purpose:
    Lane offset reads the robot as sitting at robot_x_px (the lane ROI's middle
    when unset), and single-line frames project the lane's middle
    expected_half_lane_px from the one line they see. Both are facts about
    the camera mount: a camera shifted or turned off the centerline makes the
    robot hold a lane that isn't centered, and a new height or tilt changes
    the lane's width in px. Parked centered in a straight lane, this measures
    both from the two lane lines Phase 2 finds.

Main package:
    The robot_x_px, expected_half_lane_px and min_lane_width_px for config.py
    (MEASURED's LaneOffsetConfig), with what the current ones read here.
"""
import argparse
import sys
from dataclasses import replace

import numpy as np

from src.capture.camera import CaptureError
from src.config import MEASURED
from src.params import CAMERA_CONTROLS, FPS, FRAME_H, FRAME_W

FRAMES = 100
# The pair search is loosened so a lane that changed width with the mount is still found
SEARCH_MIN_PX, SEARCH_MAX_PX = 60.0, 480.0
# MEASURED's min_lane_width_px as a share of the lane, as set on 2026-10-20 (300 of 357)
MIN_WIDTH_SHARE = 0.84
# Fewer two-line frames than this, or a wider spread, and the numbers aren't to be trusted
MIN_PAIRED_SHARE = 0.5
MAX_CENTER_SPREAD_PX = 8.0

_CLI_HELP = """\
Measure where a centered robot sees its lane and how wide it is in px, for
MEASURED's lane offset (config.py). Park the robot in the middle of a straight
lane, square to it (a ruler from each line to the wheels), then:

    make calib-lane                                  100 frames from the camera
    make calib-lane CAM="awb-mode=daylight"          with the camera set as it will run
    python3 -m src.scripts.calibrate_lane --frames runs/nav_<time>/frames --range 0-80   a recording made centered
    python3 -m src.scripts.calibrate_lane --offset-cm 2     parked 2 cm right of center on purpose

It prints the three lines for config.py, and what the current values read on
these frames: a centered robot should read offset 0.
"""


def frames_from(args):
    """(frame, frame_id, timestamp_ms) for each frame of the source, up to --count."""
    from src.debugger.live_view import CameraFrameSource, DirectoryFrameSource, VideoFrameSource
    from src.capture.camera import parse_controls
    if args.frames:
        src = DirectoryFrameSource(args.frames)
    elif args.video:
        src = VideoFrameSource(args.video)
    else:
        src = CameraFrameSource(FRAME_W, FRAME_H, FPS, {**CAMERA_CONTROLS, **parse_controls(args.camera_control)})
    lo, hi = (int(x) for x in args.range.split("-")) if args.range else (0, None)
    out, k = [], 0
    try:
        while len(out) < args.count:
            item = src.read()
            if item is None:
                break
            frame, fid, ts = item
            if frame is None:
                continue
            if k >= lo and (hi is None or k <= hi):
                out.append((frame, fid, ts))
            k += 1
            if hi is not None and k > hi:
                break
    finally:
        src.close()
    return out


def measure(frames, config=MEASURED, chain=None) -> dict:
    """Each frame through Phase 2 with the pair search loosened: the two-line frames' lane middle and width."""
    if chain is None:
        from src.phase2_linker import run_chain
        chain = run_chain
    loose = replace(config, lane_offset=replace(config.lane_offset, min_lane_width_px=SEARCH_MIN_PX,
                                                max_lane_width_px=SEARCH_MAX_PX, robot_x_px=None))
    centers, widths, now, modes = [], [], [], {}
    for frame, fid, ts in frames:
        r = chain(frame, fid, ts, loose).offset
        modes[r.mode] = modes.get(r.mode, 0) + 1
        if r.mode == "two_boundary":
            centers.append((r.left_x + r.right_x) / 2.0)
            widths.append(r.right_x - r.left_x)
        now.append(chain(frame, fid, ts, config).offset)
    return {"frames": len(frames), "modes": modes, "centers": centers, "widths": widths, "now": now}


def recommend(m: dict, roi_width: float, offset_cm: float = 0.0, lane_cm: float = 14.0) -> dict:
    """The three settings from the measurement; offset_cm moves robot_x to where the robot really was."""
    if not m["centers"]:
        raise ValueError("no frame had both lane lines: park in a straight lane with both in view")
    center, width = float(np.median(m["centers"])), float(np.median(m["widths"]))
    px_per_cm = width / lane_cm
    robot_x = center + offset_cm * px_per_cm        # the robot sat offset_cm right: its column is right of the lane's middle
    now = [r.offset for r in m["now"] if r.mode != "none"]
    return {"robot_x_px": round(robot_x, 1), "expected_half_lane_px": round(width / 2.0, 1),
            "min_lane_width_px": round(MIN_WIDTH_SHARE * width, 0), "lane_width_px": round(width, 1),
            "roi_middle_px": roi_width / 2.0, "shift_px": round(robot_x - roi_width / 2.0, 1),
            "center_spread_px": round(float(np.percentile(m["centers"], 90) - np.percentile(m["centers"], 10)), 1),
            "paired_share": len(m["centers"]) / max(m["frames"], 1),
            "offset_now": round(float(np.median(now)), 3) if now else None,
            "offset_now_cm": round(float(np.median(now)) * (roi_width / 2.0) / px_per_cm, 1) if now else None}


def report(rec: dict, m: dict, config=MEASURED) -> list[str]:
    cur = config.lane_offset
    lines = [f"{m['frames']} frames: " + ", ".join(f"{k} {v}" for k, v in sorted(m["modes"].items(), key=lambda kv: -kv[1])),
             f"lane middle at {rec['robot_x_px']:.1f} px of the ROI (its middle is {rec['roi_middle_px']:.0f}: "
             f"{rec['shift_px']:+.1f} px), lane {rec['lane_width_px']:.1f} px wide",
             f"with the current settings this centered robot reads offset {rec['offset_now']:+.3f} "
             f"(~{rec['offset_now_cm']:+.1f} cm, + = right of center): lane keeping holds it that far the other way"
             if rec["offset_now"] is not None else "the current settings read no lane on these frames",
             "", "for config.py, MEASURED's LaneOffsetConfig:",
             f"        robot_x_px = {rec['robot_x_px']},",
             f"        expected_half_lane_px = {rec['expected_half_lane_px']},      # now {cur.expected_half_lane_px}",
             f"        min_lane_width_px = {rec['min_lane_width_px']:.0f},          # now {cur.min_lane_width_px:.0f}"]
    warn = []
    if rec["paired_share"] < MIN_PAIRED_SHARE:
        warn.append(f"only {100 * rec['paired_share']:.0f}% of frames saw both lines: check the lines are in view")
    if rec["center_spread_px"] > MAX_CENTER_SPREAD_PX:
        warn.append(f"the lane middle moved {rec['center_spread_px']:.0f} px across frames: was the robot still?")
    if cur.min_lane_width_px > rec["lane_width_px"]:
        warn.append(f"the current min_lane_width_px ({cur.min_lane_width_px:.0f}) is wider than the lane "
                    f"({rec['lane_width_px']:.0f}): every two-line frame falls back to one line, "
                    "and expected_half_lane_px steers instead")
    if cur.max_lane_width_px < rec["lane_width_px"]:
        warn.append(f"max_lane_width_px ({cur.max_lane_width_px:.0f}) is under the lane's width: raise it")
    return lines + ([""] + [f"! {w}" for w in warn] if warn else [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m src.scripts.calibrate_lane", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--frames", metavar="DIR", help="a folder of frames instead of the camera")
    src.add_argument("--video", metavar="PATH", help="a video instead of the camera")
    ap.add_argument("--range", metavar="A-B", help="only these frames of a recording (by position)")
    ap.add_argument("--count", type=int, default=FRAMES, help="frames to measure")
    ap.add_argument("--offset-cm", type=float, default=0.0, help="the robot was parked this far right of center")
    ap.add_argument("--lane-cm", type=float, default=14.0, help="lane width, line center to line center")
    ap.add_argument("--camera-control", action="append", default=None, metavar="KEY=VALUE",
                    help="a libcamerasrc control, over params.CAMERA_CONTROLS; repeatable")
    args = ap.parse_args(argv)
    try:
        frames = frames_from(args)
        if not frames:
            raise ValueError("no frames")
        m = measure(frames)
        from src.perception.roi_crop import resolve
        roi_width = resolve(MEASURED.roi.lane, frames[0][0].shape[:2])[2]
        rec = recommend(m, roi_width, args.offset_cm, args.lane_cm)
    except (ValueError, OSError, CaptureError) as exc:
        print(f"lane calibration: {exc}")
        return 2
    print("\n".join(report(rec, m)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
