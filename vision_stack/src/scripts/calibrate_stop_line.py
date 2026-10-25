#!/usr/bin/env python3
"""Stop-line calibration: tape marks at measured distances -> the curve from image rows to cm.

Purpose:
    The board-free way to give the stop line cm. Tape lies across the lane
    at a measured distance, and the robot's own Phase 2 says how many rows up
    it sees it, so whatever the detector does is calibrated in
    (calibrate_stop_line.md).

Main package:
    calibration/stop_line_table.json: the curve's A, B and C, each mark and
    its error, and the image setup it's valid for.
"""
import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import MEASURED
from src.params import FRAME_H, FRAME_W, STOP_LINE_TABLE_PATH
from src.perception.ground import lens_id
from src.perception.roi_crop import resolve
from src.perception.stop_line_table import MIN_MARKS, fit_stop_line_table

FRAMES_PER_MARK = 15
# Of its frames
MIN_SEEN = 0.5
# p90 - p10 over this: the detector wavered; kept, with a warning
MAX_SPREAD_ROWS = 3.0
# Verdicts on the worst mark's error
GOOD_CM, OK_CM = 0.5, 1.0
DEFAULT_REFERENCE = "the front edge of the robot"

# --help text, apart from the module docstring
_CLI_HELP = """\
Calibrate the stop line's distance in cm from tape marks, without the
checkerboard. Writes calibration/stop_line_table.json; the pipeline then
reports the stop line's cm (distance_cm) from it whenever there's no ground
homography.

For each mark: lay a strip of tape (or the real stop-line tape) straight
across the lane, square to the robot, at a distance you measure, then type
that distance. Measure every mark from the same point on the robot
(--reference, default the front edge). Use 4-6 marks from where the camera's
view begins to as far as you need, e.g. 3, 6, 10, 15, 20 cm.

Run from vision_stack/, lens calibration done first:
    python3 -m src.scripts.calibrate_stop_line
    python3 -m src.scripts.calibrate_stop_line --marks 3:4.5,6:21,10:38,15:54,20:66   # refit, no camera

Redo it after changing the camera mount or angle, undistort_alpha, the output
size or the lens calibration.
"""


def measure_mark(frames, config) -> dict:
    """The median rows over frames that saw a line, with seen, spread (p90 - p10), clipped and candidates."""
    from src.phase2_linker import run_chain     # heavy

    rows, clipped, most = [], 0, 0
    for i, frame in enumerate(frames):
        s = run_chain(frame, i, 0, config).stop_line
        most = max(most, s.candidate_count)
        if s.detected:
            rows.append(s.distance_px)
            clipped += int(s.clipped)
    r = np.array(rows, np.float64)
    return {"rows": round(float(np.median(r)), 2) if len(r) else None, "seen": len(r),
            "frames": len(frames), "clipped": clipped, "candidates": most,
            "spread": round(float(np.percentile(r, 90) - np.percentile(r, 10)), 2) if len(r) else 0.0}


def mark_problem(m: dict) -> str | None:
    if m["seen"] < MIN_SEEN * m["frames"]:
        return (f"the stop line was seen in only {m['seen']} of {m['frames']} frames "
                f"(up to {m['candidates']} candidates, below the confidence gate or none): the whole strip "
                "must cross the lane in view, straight and lit, not touching a lane mark")
    if m["clipped"] > m["seen"] / 2:
        return "the line runs off the bottom of the view: move the mark further out"
    return None


def parse_marks(text: str) -> list[dict]:
    """"cm:rows,cm:rows,..." for a refit without the camera. ValueError if malformed."""
    marks = []
    for item in text.split(","):
        cm, sep, rows = item.partition(":")
        if not sep:
            raise ValueError(f"--marks item {item!r}: write it as cm:rows")
        marks.append({"cm": float(cm), "rows": float(rows)})
    return marks


def build_record(marks: list[dict], preprocess, frame_size, reference: str) -> dict:
    """The JSON load_stop_line_table reads, plus what's needed to audit or refit it."""
    cm = [m["cm"] for m in marks]
    rows = [m["rows"] for m in marks]
    (a, b, c), errors = fit_stop_line_table(rows, cm)
    lens_path = Path(preprocess.calibration_path)
    return {
        "model": "stop_line_rows_to_cm",
        "curve": [a, b, c],
        "formula": "cm = curve[0] / (curve[1] - rows) + curve[2]; rows = distance_px above the lane ROI bottom",
        "reference": reference,
        "marks": [{**m, "fit_cm": round(float(a / (b - m["rows"]) + c), 3), "error_cm": round(float(e), 3)}
                  for m, e in zip(marks, errors)],
        "error_cm": {"mean": float(np.mean(errors)), "max": float(np.max(errors))},
        "image_size": list(frame_size),
        "undistort_alpha": float(preprocess.undistort_alpha),
        "lens_calibration": {"file": lens_path.name, "sha256": lens_id(lens_path)},
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def report_lines(rec: dict, roi_height: int) -> list[str]:
    a, b, c = rec["curve"]
    err = rec["error_cm"]["max"]
    verdict = "GOOD" if err < GOOD_CM else "OK" if err < OK_CM else "POOR - see calibrate_stop_line.md"
    out = [f"  {'mark cm':>8} {'rows':>7} {'fit cm':>7} {'error':>6}"]
    out += [f"  {m['cm']:>8.2f} {m['rows']:>7.1f} {m['fit_cm']:>7.2f} {m['error_cm']:>6.2f}" for m in rec["marks"]]
    out.append(f"  worst error {err:.2f} cm, mean {rec['error_cm']['mean']:.2f} cm -> {verdict}")
    if len(rec["marks"]) == MIN_MARKS:
        out.append(f"  WARNING: {MIN_MARKS} marks fit the curve exactly, so nothing checks them; add a fourth")
    far = max(m["rows"] for m in rec["marks"])
    out.append(f"  view bottom reads {max(a / b + c, 0.0):.1f} cm; the farthest mark ({far:.0f} rows) "
               f"{max(a / (b - far) + c, 0.0):.1f} cm")
    if far < 0.6 * roi_height:
        out.append(f"  NOTE: marks reach {far:.0f} of the lane ROI's {roi_height} rows; farther lines are extrapolated")
    return out


def collect_marks(grab, ask, say, config, frames_per_mark: int) -> list[dict]:
    """A distance typed per mark; grab(n) gives the frames."""
    marks = []
    while True:
        text = ask(f"\nMark {len(marks) + 1}: tape's distance in cm (Enter to fit, 'u' to undo): ").strip()
        if not text:
            if len(marks) >= MIN_MARKS:
                return marks
            say(f"  need at least {MIN_MARKS} marks, have {len(marks)}")
            continue
        if text.lower() == "u":
            if marks:
                say(f"  removed the {marks.pop()['cm']} cm mark")
            continue
        try:
            cm = float(text)
        except ValueError:
            say(f"  {text!r} isn't a distance in cm")
            continue
        m = measure_mark(grab(frames_per_mark), config)
        problem = mark_problem(m)
        if problem:
            say(f"  not recorded: {problem}")
            continue
        marks.append({"cm": cm, **m})
        spread = "" if m["spread"] <= MAX_SPREAD_ROWS else f"  WARNING: rows spread {m['spread']:.1f}; check for glare"
        say(f"  {cm} cm -> {m['rows']:.1f} rows (seen in {m['seen']}/{m['frames']} frames){spread}")


def main(argv: list[str] | None = None, grab=None, ask=input, say=print) -> int:
    p = argparse.ArgumentParser(prog="calibrate_stop_line", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--marks", default=None, metavar="CM:ROWS,...",
                   help="fit these marks instead of measuring with the camera")
    p.add_argument("--frames", type=int, default=FRAMES_PER_MARK, help="frames measured per mark")
    p.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help="what every mark's distance is measured from, recorded in the file")
    p.add_argument("--out", default=str(STOP_LINE_TABLE_PATH))
    args = p.parse_args(argv)

    preprocess = MEASURED.preprocess
    if preprocess.calibration_path is None or lens_id(preprocess.calibration_path) is None:
        say("ERROR: no lens calibration; the pipeline measures on undistorted frames. "
            "Run scripts/calibrate_camera.py first.")
        return 2
    # Rows only, so an existing cm calibration can't feed back in
    config = replace(MEASURED, ground=None, stop_line_table=None)

    if args.marks:
        try:
            marks = parse_marks(args.marks)
        except ValueError as exc:
            say(f"ERROR: {exc}")
            return 2
    else:
        cap = None
        if grab is None:
            from src.scripts.calibrate_camera import grab_upright, open_camera
            cap = open_camera(FRAME_W, FRAME_H)

            def grab(n):
                return grab_upright(cap, n)
        say(f"Measure every mark from {args.reference}.")
        try:
            marks = collect_marks(grab, ask, say, config, args.frames)
        finally:
            if cap is not None:
                cap.release()

    try:
        rec = build_record(marks, preprocess, (FRAME_W, FRAME_H), args.reference)
    except ValueError as exc:
        say(f"ERROR: {exc}")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    roi_h = resolve(MEASURED.roi.lane, (FRAME_H, FRAME_W))[3]
    say(f"\nStop-line table written to {out}")
    for line in report_lines(rec, roi_h):
        say(line)
    say(f"\nCheck it: a strip of tape at a new measured distance should read that distance in "
        f"phase3_linker's status line (line=<px>/<cm>cm). Distances are from {args.reference}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
