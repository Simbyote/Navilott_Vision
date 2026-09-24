#!/usr/bin/env python3
"""Lens calibration for the IMX290 + M12 lens: measure the distortion once, write it for preprocess.

Purpose:
    preprocess.py undistorts every frame before the gray/color split so
    straight tape stays straight across the whole frame; this script produces
    the calibration it loads. The camera is opened through capture's own
    build_gst_pipeline(), and verify undistorts through preprocess's own
    undistort(), so the script can't drift from what the pipeline does. A
    calibration is only valid for the sensor mode, output size, flip and lens
    focus it was captured with; the JSON records the first three, and turning
    the lens means recalibrating.

Main package:
    calibration/camera_calib.json: image_size, camera_matrix and dist_coeffs
    (what preprocess reads), plus the RMS reprojection error, the frames used
    and dropped, the board pattern, and the sensor mode and flip it's valid for.

Flow (mode "all"):
    1. capture: auto-save checkerboard frames as the board moves around the view.
    2. solve: calibrate from them, drop outlier frames once, write the JSON.
    3. verify: save raw vs undistorted pairs and score line straightness.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import build_gst_pipeline
from src.params import (
    CAMERA_CALIB_PATH, CAMERA_ROTATE_180, FPS, FRAME_H, FRAME_W, PIPELINE_ROOT, SENSOR_CONFIG,
)
from src.perception.preprocess import PreprocessParams, undistort

DEFAULT_FRAMES = PIPELINE_ROOT / "calib_frames"
DEFAULT_VERIFY = PIPELINE_ROOT / "calib_verify"

FIND_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
              | cv2.CALIB_CB_NORMALIZE_IMAGE
              | cv2.CALIB_CB_FAST_CHECK)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

# --help text. Kept apart from the module docstring, which documents the code.
_CLI_HELP = """\
Lens calibration for the vision pipeline camera. Writes
calibration/camera_calib.json, which preprocess.py uses to undistort frames.

Modes:
    capture   Grab checkerboard frames from the camera. Headless: a frame is
              saved whenever the whole board is in view and has moved since
              the last save, and coverage of the view is printed.
    solve     Compute the camera matrix and distortion from the saved frames
              and write the JSON.
    verify    Save raw vs undistorted comparison images and print a
              straightness score for each (lower is straighter).
    all       capture, then solve, then verify (default). Starts fresh:
              existing frames are moved to calib_frames_prev/.

Run from the project root (the folder containing src/), venv active:
    python3 -m src.scripts.calibrate_camera

Recalibrate after changing the sensor mode, output size or flip in
params.py, or after anyone turns the lens.
"""


def open_camera(width: int, height: int) -> cv2.VideoCapture:
    """
    Open the camera through the live pipeline's own GStreamer string, then let auto-exposure settle.

    Side effects:
        Claims the camera; exits the script with a hint if it can't.
    """
    cap = cv2.VideoCapture(build_gst_pipeline(width, height, FPS), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(
            "ERROR: could not open the camera.\n"
            "  - Is phase2_linker (or rpicam-hello) still running? Stop it first.\n"
            "  - Does `rpicam-hello --list-cameras` show imx290?"
        )
    for _ in range(30):                 # let auto-exposure settle
        cap.read()
    return cap


def parse_pattern(text: str) -> tuple[int, int]:
    """"9x6" -> (9, 6): inner corners, cols x rows. Exits on anything else."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError:
        sys.exit(f"ERROR: --pattern must look like 9x6, got {text!r}")
    return cols, rows


def find_corners(gray: np.ndarray, pattern: tuple[int, int], fast: bool = True) -> np.ndarray | None:
    """
    Sub-pixel checkerboard corners, or None if the whole board isn't visible.

    Inputs:
        fast: Keep OpenCV's fast rejection check. Right for the live capture
            loop, where most frames have no board; off for solve and verify,
            where every saved frame should have one.
    """
    flags = FIND_FLAGS if fast else FIND_FLAGS & ~cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)


def straightness(corners: np.ndarray, pattern: tuple[int, int]) -> float:
    """
    Mean, over every row and column of corners, of the worst deviation (px)
    from a best-fit straight line. Barrel distortion bows these lines, so
    this number should drop after undistortion.
    """
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    worst = []
    for line in list(grid) + list(grid.transpose(1, 0, 2)):
        vx, vy, x0, y0 = cv2.fitLine(
            line.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = np.abs((line[:, 0] - x0) * vy - (line[:, 1] - y0) * vx)
        worst.append(float(d.max()))
    return float(np.mean(worst))


def capture(args) -> None:
    """
    Save checkerboard frames until --count, --timeout or Ctrl-C, then print coverage.

    A frame is saved only when the whole board is visible, at least
    --interval seconds have passed, and the corners moved at least --min-move
    px on average since the last save, so the set spreads across poses.
    Coverage counts saves per third of the frame (3x3); an empty region
    means the corners of the lens, where distortion is largest, went unmeasured.

    Side effects:
        Writes calib_NNN.png into --frames. With --fresh, first moves existing
        frames to <frames>_prev/ (replacing an older backup). Refuses to mix
        with existing frames unless --fresh or --append.
    """
    pattern = parse_pattern(args.pattern)
    frames_dir = Path(args.frames)
    existing = sorted(frames_dir.glob("calib_*.png"))

    if existing and not (args.fresh or args.append):
        sys.exit(
            f"ERROR: {frames_dir} already has {len(existing)} frames.\n"
            "  Run `all` to start over, or `capture --append` to add to them."
        )
    if existing and args.fresh:
        backup = frames_dir.with_name(frames_dir.name + "_prev")
        if backup.exists():
            shutil.rmtree(backup)
        frames_dir.rename(backup)
        print(f"Moved {len(existing)} old frames to {backup}")
        existing = []
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.width, args.height)
    cells = np.zeros((3, 3), int)
    saved = len(existing)
    index = saved
    last_pts = None
    last_save = 0.0
    last_hint = 0.0
    start = time.time()

    print(f"\nCapturing up to {args.count} frames into {frames_dir}")
    print("Hold the board so ALL of it is in view, then hold still ~1 s.")
    print("Move it between saves: left/center/right, high/low, near/far, tilted.\n")

    try:
        while saved < args.count:
            if time.time() - start > args.timeout:
                print(f"\nTimed out after {args.timeout:.0f} s.")
                break
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern)
            now = time.time()

            if corners is None:
                if now - last_hint > 3.0:
                    print("  ... no full board in view")
                    last_hint = now
                continue

            pts = corners.reshape(-1, 2)
            if now - last_save < args.interval:
                continue
            if last_pts is not None:
                moved = float(np.mean(np.linalg.norm(pts - last_pts, axis=1)))
                if moved < args.min_move:
                    if now - last_hint > 3.0:
                        print("  ... board found, move it to a new spot")
                        last_hint = now
                    continue

            cv2.imwrite(str(frames_dir / f"calib_{index:03d}.png"), frame)
            index += 1
            saved += 1
            last_pts, last_save, last_hint = pts, now, now

            cx, cy = pts.mean(axis=0)
            col = min(int(3 * cx / frame.shape[1]), 2)
            row = min(int(3 * cy / frame.shape[0]), 2)
            cells[row, col] += 1
            print(f"  [{saved:2d}/{args.count}] saved   "
                  f"region {['top', 'middle', 'bottom'][row]}-"
                  f"{['left', 'center', 'right'][col]}   "
                  f"coverage {int((cells > 0).sum())}/9")
    finally:
        cap.release()

    print("\nCoverage (frames per region, as seen by the camera):")
    for r, name in enumerate(["top   ", "middle", "bottom"]):
        print(f"  {name}  " + "  ".join(f"{n:2d}" for n in cells[r]))
    missing = int((cells == 0).sum())
    if missing:
        print(f"\nWARNING: {missing} region(s) have no frames. Run "
              "`capture --append` and hold the board in those regions.")


def solve(args) -> None:
    """
    Calibrate from the saved frames and write the JSON.

    Frames whose reprojection error is over 3x the median (at least 1 px) are
    dropped once and the solve repeated, as long as 10 frames remain; those
    are usually blurred or mis-detected boards.

    Side effects:
        Writes --out and prints the fit. Exits if the frames mix resolutions
        or fewer than 10 have a detectable board.
    """
    pattern = parse_pattern(args.pattern)
    cols, rows = pattern
    files = sorted(Path(args.frames).glob("calib_*.png"))
    if not files:
        sys.exit(f"ERROR: no calib_*.png frames in {args.frames}. Run capture first.")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_mm

    obj_pts, img_pts, used, size = [], [], [], None
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        this_size = (img.shape[1], img.shape[0])
        if size is None:
            size = this_size
        elif this_size != size:
            sys.exit(f"ERROR: {f.name} is {this_size}, others are {size}. "
                     "Frames from different resolutions can't be mixed.")
        corners = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                               pattern, fast=False)
        if corners is not None:
            obj_pts.append(objp)
            img_pts.append(corners)
            used.append(f.name)

    if len(used) < 10:
        sys.exit(f"ERROR: only {len(used)} usable frames (need 10+, 20 is better).")

    def run(o, i):
        """Calibrate; returns (rms, K, dist, per-frame reprojection RMS)."""
        rms, K, dist, rv, tv = cv2.calibrateCamera(o, i, size, None, None)
        errs = []
        for op, ip, r, t in zip(o, i, rv, tv):
            proj, _ = cv2.projectPoints(op, r, t, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj - ip) ** 2, axis=2)))))
        return rms, K, dist, np.array(errs)

    rms, K, dist, errs = run(obj_pts, img_pts)

    limit = max(1.0, 3.0 * float(np.median(errs)))
    keep = errs <= limit
    dropped = [n for n, k in zip(used, keep) if not k]
    if dropped and keep.sum() >= 10:
        obj_pts = [o for o, k in zip(obj_pts, keep) if k]
        img_pts = [i for i, k in zip(img_pts, keep) if k]
        used = [n for n, k in zip(used, keep) if k]
        rms, K, dist, errs = run(obj_pts, img_pts)

    calib = {
        "model": "pinhole",
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_px": float(rms),
        "n_frames": len(used),
        "frames": used,
        "dropped_frames": dropped,
        "pattern": [cols, rows],
        "square_mm": args.square_mm,
        "sensor_config": SENSOR_CONFIG,
        "rotate_180": CAMERA_ROTATE_180,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2))

    # Common rule of thumb for OpenCV checkerboard calibration
    if rms < 0.5:
        verdict = "GOOD"
    elif rms < 1.0:
        verdict = "OK"
    else:
        verdict = "POOR - recapture (see troubleshooting)"
    print(f"\nCalibration written to {out}")
    print(f"  frames used      {len(used)}"
          + (f"   (dropped {len(dropped)}: {', '.join(dropped)})" if dropped else ""))
    print(f"  reprojection     {rms:.3f} px   -> {verdict}")
    print(f"  focal (fx, fy)   {K[0, 0]:.1f}, {K[1, 1]:.1f} px")
    print(f"  center (cx, cy)  {K[0, 2]:.1f}, {K[1, 2]:.1f} px   "
          f"(image center {size[0] / 2:.0f}, {size[1] / 2:.0f})")
    print(f"  distortion       {np.array2string(dist.ravel(), precision=4)}")


def verify(args) -> None:
    """
    Undistort saved frames (and one live frame) through preprocess, and score how much straighter the board got.

    Uses the --verify-count frames where the board sat farthest from center,
    since that's where distortion shows and so where the fix should show.
    PASS means the average bow dropped by 40%, or below 0.5 px.

    Side effects:
        Writes verify_*.png side-by-side pairs into --verify-dir. Opens the
        camera for verify_live.png unless --no-live. Exits if --out is missing.
    """
    pattern = parse_pattern(args.pattern)
    calib_path = Path(args.out)
    if not calib_path.is_file():
        sys.exit(f"ERROR: {calib_path} not found. Run solve first.")
    calib = json.loads(calib_path.read_text())
    size = tuple(calib["image_size"])
    params = PreprocessParams(calibration_path=str(calib_path), undistort_alpha=args.alpha)

    scored = []
    for f in sorted(Path(args.frames).glob("calib_*.png")):
        img = cv2.imread(str(f))
        if img is None or (img.shape[1], img.shape[0]) != size:
            continue
        c = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern, fast=False)
        if c is None:
            continue
        off = np.linalg.norm(c.reshape(-1, 2).mean(0) - np.array(size) / 2)
        scored.append((off, f.name, img, c))
    scored.sort(key=lambda s: -s[0])

    out_dir = Path(args.verify_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStraightness (mean worst bow per board line, px; lower is straighter)")
    raw_scores, und_scores = [], []
    for _, name, img, c_raw in scored[:args.verify_count]:
        und = undistort(img, params)
        c_und = find_corners(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), pattern, fast=False)
        s_raw = straightness(c_raw, pattern)
        s_und = straightness(c_und, pattern) if c_und is not None else float("nan")
        raw_scores.append(s_raw)
        und_scores.append(s_und)
        print(f"  {name}   raw {s_raw:5.2f}   undistorted {s_und:5.2f}")
        _save_pair(out_dir / f"verify_{Path(name).stem}.png", img, und)

    # One live frame of whatever the camera sees now (e.g. the course)
    if not args.no_live:
        cap = open_camera(size[0], size[1])
        ok, frame = cap.read()
        cap.release()
        if ok:
            und = undistort(frame, params)
            _save_pair(out_dir / "verify_live.png", frame, und)
            print("  verify_live.png  saved (live view: check that tape looks straight)")

    if raw_scores:
        r, u = float(np.nanmean(raw_scores)), float(np.nanmean(und_scores))
        print(f"\n  average          raw {r:5.2f}   undistorted {u:5.2f}")
        passed = u < 0.6 * r or u < 0.5      # 0.5 px is about the noise floor
        print("  RESULT: " + ("PASS - lines straightened" if passed
                            else "CHECK - little improvement, see troubleshooting"))
    print(f"\nComparison images in {out_dir}")


def _save_pair(path: Path, raw: np.ndarray, und: np.ndarray) -> None:
    """Write raw and undistorted side by side, labeled, with a white gap between."""
    def label(img, text):
        img = img.copy()
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return img
    gap = np.full((raw.shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([label(raw, "RAW"), gap,
                                      label(und, "UNDISTORTED")]))


def main() -> None:
    """Parse arguments and run the chosen mode, or all three in order."""
    p = argparse.ArgumentParser(
        prog="calibrate_camera", description=_CLI_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="all",
                   choices=["capture", "solve", "verify", "all"])
    p.add_argument("--square-mm", type=float, default=25.0,
                   help="side of one printed square in mm; only scales the "
                        "reported board poses, does not affect undistortion")
    p.add_argument("--pattern", default="9x6",
                   help="INNER corners, cols x rows (OpenCV pattern.png = 9x6)")
    p.add_argument("--count", type=int, default=20, help="frames to capture")
    p.add_argument("--interval", type=float, default=1.0,
                   help="min seconds between saves")
    p.add_argument("--min-move", type=float, default=20.0,
                   help="min mean corner movement (px) between saves")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="give up capturing after this many seconds")
    p.add_argument("--fresh", action="store_true",
                   help="move old frames to calib_frames_prev/ before capturing")
    p.add_argument("--append", action="store_true",
                   help="add to existing frames")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="0 = crop to valid pixels (matches preprocess default)")
    p.add_argument("--verify-count", type=int, default=4)
    p.add_argument("--no-live", action="store_true",
                   help="verify from saved frames only, don't open the camera")
    p.add_argument("--width", type=int, default=FRAME_W)
    p.add_argument("--height", type=int, default=FRAME_H)
    p.add_argument("--frames", default=str(DEFAULT_FRAMES))
    p.add_argument("--out", default=str(CAMERA_CALIB_PATH))
    p.add_argument("--verify-dir", default=str(DEFAULT_VERIFY))
    args = p.parse_args()
    if args.mode == "all" and not args.append:
        args.fresh = True

    if args.mode in ("capture", "solve", "all") and (args.width, args.height) != (FRAME_W, FRAME_H):
        print(f"note: {args.width}x{args.height} differs from params.py ({FRAME_W}x{FRAME_H}); "
              "preprocess will reject this calibration at the pipeline's size")

    if args.mode in ("capture", "all"):
        capture(args)
    if args.mode in ("solve", "all"):
        solve(args)
    if args.mode in ("verify", "all"):
        verify(args)


if __name__ == "__main__":
    main()