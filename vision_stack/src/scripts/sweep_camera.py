#!/usr/bin/env python3
"""Camera sweep: try camera control settings on the live view and score what Phase 2 reads under each.

Purpose:
    libcamera's auto exposure and white balance follow the room, so the same
    sign can read red one run and purple the next (2026-10-26, backlit by
    windows). Each setting opens the camera, lets the auto loops settle, and
    runs Phase 2 with MEASURED on the frames: the stop sign, the light and the
    lane, side by side, with a picture of each.

Main package:
    runs/camera_<time>/: sweep.csv, summary.txt (best first) and sheet.png.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import CaptureError, check_control, describe_controls, parse_controls
from src.config import MEASURED, MEASURED_ESTIMATION
from src.params import CAMERA_CONTROLS, FPS, FRAME_H, FRAME_W, RUNS_DIR

# Settings for a sign or light washed out by a bright background. "current" is params.CAMERA_CONTROLS
BACKLIGHT = (
    "",
    "awb-mode=daylight",
    "awb-mode=indoor",
    "awb-mode=fluorescent",
    "ae-constraint-mode=shadows",
    "exposure-value=1",
    "exposure-value=1 ae-constraint-mode=shadows",
    "saturation=1.5",
    "awb-mode=daylight saturation=1.5",
    "awb-mode=daylight exposure-value=1 saturation=1.5",
)
# The auto loops take ~1-2 s to land after the camera opens
SETTLE_S = 3.0
FRAMES = 40
CLIPPED_V = 250
SHEET_COLS = 3
THUMB = (320, 180)

FIELDS = ("round", "setting", "frames", "sign_seen_pct", "sign_pass_pct", "sign_conf_max", "sign_redness_p99",
          "light", "light_pass_pct", "lane_two_pct", "lane_any_pct", "mean_v", "clipped_pct", "picture")

_CLI_HELP = """\
Try camera control settings on the live view, one after another, and score
what Phase 2 reads under each: the stop sign, the traffic light, the lane.
Park the robot where it fails (the sign in view, the bright windows behind),
stop any other run (one process can hold the camera), then:

    make sweep-camera                                   the backlight set below
    make sweep-camera CAM_TRY="awb-mode=daylight|exposure-value=1 saturation=1.5"
    python3 -m src.scripts.sweep_camera --try "awb-mode=daylight saturation=1.5" --rounds 2
    python3 -m src.scripts.sweep_camera --list          every control and its values

Each --try is one setting: KEY=VALUE pairs separated by spaces, over
params.CAMERA_CONTROLS. "current" (params alone) always runs first, as the
baseline. --rounds 2 runs the whole list twice: light through a window changes
by the minute, so a setting that only wins once may have won on the weather.

Output (--out DIR, default runs/camera_<time>/):
    summary.txt   one line per setting, best for the stop sign first
    sweep.csv     the same, every column
    sheet.png     one picture per setting, labelled
    frames/       those pictures full size

Keep the winner by putting it in params.CAMERA_CONTROLS, then check the lanes
and the traffic light on a run (make nav-dry), since exposure moves them too.
"""


def parse_setting(text: str) -> dict:
    """One --try: KEY=VALUE pairs split on spaces or commas. ValueError for a bad pair."""
    return parse_controls([p for p in text.replace(",", " ").split() if p])


def label(setting: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in setting.items()) or "current"


def score(chains: list, frames: list) -> dict:
    """Phase 2's reading over one setting's frames: how often each target was seen, and how bright the view was."""
    n = len(chains)
    pct = lambda k: round(100.0 * k / n, 1) if n else 0.0          # noqa: E731
    sign_conf = [max((c.confidence for c in ch.geometry.sign_candidates), default=0.0) for ch in chains]
    red = [float(np.percentile(ch.sign_debug["redness"], 99)) for ch in chains if ch.sign_debug.get("redness") is not None]
    lights = [max(ch.traffic, key=lambda t: t.confidence) if ch.traffic else None for ch in chains]
    seen = [t.label for t in lights if t is not None and t.confidence >= MEASURED_ESTIMATION.min_confidence_traffic]
    modes = [ch.offset.mode for ch in chains]
    v = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2] for f in frames]
    return {
        "frames": n,
        "sign_seen_pct": pct(sum(c > 0 for c in sign_conf)),
        "sign_pass_pct": pct(sum(c >= MEASURED_ESTIMATION.min_confidence_sign for c in sign_conf)),
        "sign_conf_max": round(max(sign_conf, default=0.0), 2),
        "sign_redness_p99": round(float(np.median(red)), 1) if red else None,
        "light": max(set(seen), key=seen.count) if seen else "",
        "light_pass_pct": pct(len(seen)),
        "lane_two_pct": pct(modes.count("two_boundary")),
        "lane_any_pct": pct(sum(m in ("two_boundary", "left_only", "right_only") for m in modes)),
        "mean_v": round(float(np.mean([x.mean() for x in v])), 1) if v else None,
        "clipped_pct": round(100.0 * float(np.mean([(x >= CLIPPED_V).mean() for x in v])), 2) if v else None,
    }


def ranked(rows: list) -> list:
    """Best for the stop sign first: passing share, then seen share, then redness."""
    return sorted(rows, key=lambda r: (-r["sign_pass_pct"], -r["sign_seen_pct"], -(r["sign_redness_p99"] or 0.0)))


def open_camera(controls: dict):
    from src.capture.camera import CameraSource
    return CameraSource(FRAME_W, FRAME_H, FPS, controls=controls).open()


def run_chain_measured(frame, frame_id: int, ts_ms: int):
    from src.phase2_linker import run_chain
    return run_chain(frame, frame_id, ts_ms, MEASURED, parallel=False)


def grab(source, settle_s: float, n: int, clock=time.monotonic) -> list:
    """Reads for settle_s and drops them, then n frames (FrameData). Drops don't count."""
    t_end = clock() + settle_s
    while clock() < t_end:
        source.read()
    got = []
    while len(got) < n:
        fd = source.read()
        if fd is not None:
            got.append(fd)
    return got


def sweep(settings: list, out: Path, rounds: int = 1, settle_s: float = SETTLE_S, frames: int = FRAMES,
          opener=open_camera, chain=run_chain_measured, say=print) -> list:
    """Every setting over params.CAMERA_CONTROLS, rounds times; writes the folder and returns the rows."""
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(1, rounds + 1):
        for k, setting in enumerate(settings):
            name = label(setting)
            say(f"round {r}  {name} ...")
            try:
                src = opener({**CAMERA_CONTROLS, **setting})
            except (CaptureError, ValueError) as exc:
                say(f"  {name}: camera didn't open ({exc}); skipped")
                continue
            try:
                fds = grab(src, settle_s, frames)
            finally:
                src.release()
            chains = [chain(fd.frame, fd.frame_id, fd.timestamp_ms) for fd in fds]
            pic = f"frames/r{r}_{k:02d}.png"
            cv2.imwrite(str(out / pic), fds[len(fds) // 2].frame)
            row = {"round": r, "setting": name, **score(chains, [fd.frame for fd in fds]), "picture": pic}
            rows.append(row)
            say(f"  sign {row['sign_pass_pct']}% (seen {row['sign_seen_pct']}%, redness {row['sign_redness_p99']})"
                f"  light {row['light'] or '-'} {row['light_pass_pct']}%  lane {row['lane_two_pct']}%"
                f"  V {row['mean_v']}")
    write(rows, out)
    return rows


def summary_lines(rows: list) -> list:
    lines = [f"Camera sweep over params.CAMERA_CONTROLS = {CAMERA_CONTROLS or '{}'}", "",
             f"{'sign pass':>9} {'seen':>5} {'red':>5}  {'light':>12}  {'lane 2':>6} {'V':>5} {'clip%':>5}  setting"]
    for r in ranked(rows):
        light = f"{r['light'] or '-'} {r['light_pass_pct']:.0f}%"
        red = "-" if r["sign_redness_p99"] is None else f"{r['sign_redness_p99']:.0f}"
        lines.append(f"{r['sign_pass_pct']:>8.0f}% {r['sign_seen_pct']:>4.0f}% {red:>5}  {light:>12}  "
                     f"{r['lane_two_pct']:>5.0f}% {r['mean_v']:>5.0f} {r['clipped_pct']:>5.1f}  "
                     f"{r['setting']}  (round {r['round']})")
    lines += ["", f"sign pass: frames with a sign at or over Phase 3's gate ({MEASURED_ESTIMATION.min_confidence_sign});"
              " red: the sign ROI's 99th-percentile redness (under ~20, min_redness, the sign can't show);",
              "lane 2: frames with both lane lines. Exposure moves the lanes and the light too: check them before keeping a winner."]
    return lines


def contact_sheet(rows: list, out: Path) -> np.ndarray | None:
    tiles = []
    for r in rows:
        img = cv2.imread(str(out / r["picture"]))
        if img is None:
            continue
        img = cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA)
        bar = np.zeros((40, THUMB[0], 3), np.uint8)
        cv2.putText(bar, r["setting"][:44], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"sign {r['sign_pass_pct']:.0f}%  light {r['light'] or '-'}  lane {r['lane_two_pct']:.0f}%",
                    (4, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 255, 180), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    if not tiles:
        return None
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % SHEET_COLS)
    return np.vstack([np.hstack(tiles[i:i + SHEET_COLS]) for i in range(0, len(tiles), SHEET_COLS)])


def write(rows: list, out: Path) -> None:
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.txt").write_text("\n".join(summary_lines(rows)) + "\n")
    sheet = contact_sheet(rows, out)
    if sheet is not None:
        cv2.imwrite(str(out / "sheet.png"), sheet)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m src.scripts.sweep_camera", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--try", dest="tries", action="append", default=None, metavar="'K=V K=V'",
                    help="one setting to try; repeatable, or several split by |. Without any, the backlight set")
    ap.add_argument("--rounds", type=int, default=1, help="run the whole list this many times")
    ap.add_argument("--settle", type=float, default=SETTLE_S, metavar="S", help="seconds dropped after opening")
    ap.add_argument("--frames", type=int, default=FRAMES, help="frames scored per setting")
    ap.add_argument("--out", default=None, metavar="DIR")
    ap.add_argument("--list", action="store_true", help="print every camera control and its values")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(describe_controls()))
        return 0
    try:
        texts = [t for arg in args.tries for t in arg.split("|")] if args.tries is not None else BACKLIGHT
        tries = [parse_setting(t) for t in texts]
    except ValueError as exc:
        print(f"camera control error: {exc}")
        return 2
    settings = [{}] + [t for t in tries if t]           # the baseline first, once
    bad = [p for p in (check_control(k, v) for k, v in CAMERA_CONTROLS.items()) if p]
    if bad:
        print(f"params.CAMERA_CONTROLS: {'; '.join(bad)}")
        return 2
    out = Path(args.out) if args.out else RUNS_DIR / time.strftime("camera_%Y%m%d_%H%M%S")
    rows = sweep(settings, out, args.rounds, args.settle, args.frames)
    print("\n" + "\n".join(summary_lines(rows)) + f"\n\n{out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
