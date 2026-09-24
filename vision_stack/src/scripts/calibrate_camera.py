#!/usr/bin/env python3
"""
calibrate_camera.py

Lens calibration for the IMX290 + M12 lens

Purpose:
    Measures the lens distortion once and writes it to
    calibration/camera_calib.json. preprocess.py loads that file and undistorts
    every frame before the gray/color split, so straight tape stays straight
    across the whole frame.

Modes:
    capture   Grab checkerboard frames from the camera. Headless: the script
              auto-saves a frame whenever it sees the whole board and the
              board has moved since the last save, and prints coverage.
    solve     Compute the camera matrix and distortion from the saved frames
              and write the JSON.
    verify    Save raw vs undistorted comparison images and print a
              straightness score for each (lower is straighter).
    all       capture, then solve, then verify (default). Starts fresh:
              existing frames are moved to calib_frames_prev/.

Run from the project root (the folder containing src/), venv active:
    python3 -m src.scripts.calibrate_camera

Validity:
    A calibration is only valid for the exact sensor mode, output size,
    flip, and lens focus it was captured with. If any of SENSOR_CONFIG,
    WIDTH/HEIGHT, or ROTATE_180 below stop matching src/capture/camera.py,
    or someone turns the lens, recalibrate.
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

# =============================================================================
# Settings (must match src/capture/camera.py)
# =============================================================================

SENSOR_CONFIG = "sensor/config,width=1920,height=1080,depth=10"
WIDTH, HEIGHT, FPS = 480, 270, 30
ROTATE_180 = True

ROOT = Path(__file__).resolve().parents[2]          # src/scripts/ -> project root
DEFAULT_OUT = ROOT / "calibration" / "camera_calib.json"
DEFAULT_FRAMES = ROOT / "calib_frames"
DEFAULT_VERIFY = ROOT / "calib_verify"

FIND_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
              | cv2.CALIB_CB_NORMALIZE_IMAGE
              | cv2.CALIB_CB_FAST_CHECK)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


# =============================================================================
# Helpers
# =============================================================================

def gst_pipeline(width: int, height: int, fps: int) -> str:
    """Same pipeline shape as the live capture, forced to BGR for OpenCV."""
    flip = "videoflip method=rotate-180 ! " if ROTATE_180 else ""
    return (
        f'libcamerasrc sensor-config="{SENSOR_CONFIG}" ! '
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        f"{flip}"
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_camera(width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(gst_pipeline(width, height, FPS), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(
            "ERROR: could not open the camera.\n"
            "  - Is phase2_linker (or rpicam-hello) still running? Stop it first.\n"
            "  - Does `rpicam-hello --list-cameras` show imx290?"
        )
    for _ in range(30):                 # let auto-exposure settle
        cap.read()
    return cap


def parse_pattern(text: str) -> tuple:
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError:
        sys.exit(f"ERROR: --pattern must look like 9x6, got {text!r}")
    return cols, rows


def find_corners(gray: np.ndarray, pattern: tuple, fast: bool = True):
    flags = FIND_FLAGS if fast else FIND_FLAGS & ~cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)


def straightness(corners: np.ndarray, pattern: tuple) -> float:
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


def undistort_maps(calib: dict, alpha: float):
    """Same map construction preprocess.py uses."""
    size = tuple(calib["image_size"])
    K = np.array(calib["camera_matrix"], np.float64)
    dist = np.array(calib["dist_coeffs"], np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    return cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)


# =============================================================================
# Capture
# =============================================================================

def capture(args) -> None:
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


# =============================================================================
# Solve
# =============================================================================

def solve(args) -> None:
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
        rms, K, dist, rv, tv = cv2.calibrateCamera(o, i, size, None, None)
        errs = []
        for op, ip, r, t in zip(o, i, rv, tv):
            proj, _ = cv2.projectPoints(op, r, t, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj - ip) ** 2, axis=2)))))
        return rms, K, dist, np.array(errs)

    rms, K, dist, errs = run(obj_pts, img_pts)

    # One pass of outlier rejection: blurred or mis-detected frames
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
        "rotate_180": ROTATE_180,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calib, indent=2))

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


# =============================================================================
# Verify
# =============================================================================

def verify(args) -> None:
    pattern = parse_pattern(args.pattern)
    calib_path = Path(args.out)
    if not calib_path.is_file():
        sys.exit(f"ERROR: {calib_path} not found. Run solve first.")
    calib = json.loads(calib_path.read_text())
    map1, map2 = undistort_maps(calib, args.alpha)
    size = tuple(calib["image_size"])

    # Prefer the frames where the board sat farthest from center:
    # that is where distortion shows, so that is where the fix should show
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
        und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
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
            und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
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
    def label(img, text):
        img = img.copy()
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return img
    gap = np.full((raw.shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.hstack([label(raw, "RAW"), gap,
                                      label(und, "UNDISTORTED")]))


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Lens calibration for the vision pipeline camera")
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
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
    p.add_argument("--frames", default=str(DEFAULT_FRAMES))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--verify-dir", default=str(DEFAULT_VERIFY))
    args = p.parse_args()
    if args.mode == "all" and not args.append:
        args.fresh = True

    if args.mode in ("capture", "all"):
        capture(args)
    if args.mode in ("solve", "all"):
        solve(args)
    if args.mode in ("verify", "all"):
        verify(args)


if __name__ == "__main__":
    main()