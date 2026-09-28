#!/usr/bin/env python3
"""Ground-plane calibration: fit the homography from undistorted frame pixels to floor centimeters.

Purpose:
    The camera is rigidly mounted and the floor is flat, so one homography
    turns any floor point in the frame into centimeters on the floor. A
    checkerboard lying flat across the lane ROI gives 54 points whose floor
    positions are known from the square size, and findHomography fits all of
    them. Frames go through preprocess_frame with MEASURED's own settings, so
    the fit is on exactly the undistorted frames the pipeline sees; the JSON
    records the lens calibration, undistort_alpha and size it's valid for,
    and ground.load_ground_homography refuses it under anything else.

Main package:
    calibration/ground_homography.json: H (frame px -> floor cm), the
    conditions it was fit under, the board and origin used, the per-corner
    reprojection error in cm and the averaged corners (so a refit needs no
    camera). calibration/ground_debug.png: the board with corner 0 marked,
    the floor axes and a projected 5 cm grid, to check before trusting it.

Flow:
    1. Grab frames (camera, or --image), undistort each through preprocess_frame.
    2. Find the 9x6 inner corners, refined in a window sized to the squares
       as perspective shrinks them; average them over the frames.
    3. Order them so corner 0 is the top-left in the image (far-left on the floor).
    4. Floor position of each corner from --square-cm and corner 0's position.
    5. findHomography over all corners; report the error in cm per corner.
    6. With no --origin-y-cm, move the origin to the floor at the bottom-center
       of the frame: the robot reference point.
    7. Write the JSON and the debug image.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import FrameData
from src.config import MEASURED
from src.params import CALIBRATION_DIR, FRAME_H, FRAME_W, GROUND_HOMOGRAPHY_PATH
from src.perception.ground import GroundHomography, fit_ground_homography, lens_id
from src.perception.preprocess import PreprocessParams, preprocess_frame
from src.perception.roi_crop import resolve
from src.scripts.calibrate_camera import FIND_FLAGS, SUBPIX_CRITERIA, open_camera, parse_pattern

DEFAULT_DEBUG = CALIBRATION_DIR / "ground_debug.png"
DEFAULT_RAW = CALIBRATION_DIR / "ground_raw.png"

# --help text. Kept apart from the module docstring, which documents the code.
_CLI_HELP = """\
Fit the ground-plane homography (undistorted frame px -> floor cm) from a
checkerboard lying flat on the floor across the lane ROI. Writes
calibration/ground_homography.json and a debug image to check it with.

Board: OpenCV's 10x7-square pattern (9x6 inner corners), printed flat and
taped down, its rows square to the robot, covering the bottom 30% of the view.

Floor axes: X right+, Y forward+, in cm. The origin is the robot reference
point: by default the floor at the bottom-center of the camera's view, found
from the fit itself. --origin-x-cm is corner 0's distance right (+) or left
(-) of the robot's centerline, measured. Corner 0 is the board's top-left
inner corner as the camera sees it (its far-left corner on the floor); the
debug image marks it. --origin-y-cm, if given, is corner 0's distance ahead
of a reference point you measured from instead.

Run from vision_stack/, with the lens calibration done first:
    python3 -m src.scripts.calibrate_ground --square-cm=2.46 --origin-x-cm=-9.8
    python3 -m src.scripts.calibrate_ground --image=calibration/ground_raw.png --square-cm=2.46 --origin-x-cm=-9.8

Redo it after changing the camera mount or angle, undistort_alpha, the output
size, or the lens calibration.
"""


# =============================================================================
# Corners
# =============================================================================

def undistorted_gray(frame_bgr: np.ndarray, preprocess: PreprocessParams) -> tuple[np.ndarray, np.ndarray]:
    """(undistorted BGR, its gray) through preprocess_frame, the pipeline's own undistortion."""
    und = preprocess_frame(FrameData(frame_bgr, 0, 0), preprocess).undistorted
    return und, cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

def order_corners(corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """
    (N, 2) corners in row-major order with corner 0 at the top-left of the image.

    Purpose:
        findChessboardCorners may start from either end of a 9x6 board. With
        the board square to the robot, top-left in the image is the far-left
        inner corner on the floor, so fixing it here fixes the floor axes.

    Raises:
        ValueError: If the board's 9-corner side runs up the image instead of
            across it; the board has to be turned a quarter turn.
    """
    cols, rows = pattern
    pts = np.asarray(corners, np.float64).reshape(-1, 2)
    if pts[0, 1] + pts[0, 0] > pts[-1, 1] + pts[-1, 0]:
        pts = pts[::-1].copy()
    grid = pts.reshape(rows, cols, 2)
    across = grid[0, -1] - grid[0, 0]           # along a row: should run left to right
    down = grid[-1, 0] - grid[0, 0]             # along a column: should run down the image
    if abs(across[0]) < abs(across[1]) or across[0] <= 0 or down[1] <= 0:
        raise ValueError(f"board is turned: its {cols}-corner side must run across the image, "
                         "left to right; turn it a quarter turn")
    return pts

def find_ground_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """
    Ordered sub-pixel corners of a board lying on the floor, or None if the whole board isn't visible.

    Purpose:
        calibrate_camera.find_corners refines in a fixed 11x11 window, fine
        for a board held up to the lens. A board on the floor is squeezed by
        perspective: a 2.5 cm square can be 7 px tall at the far edge of the
        lane ROI, so that window spans neighboring squares and pulls corners
        toward the wrong edges (1.8 px, 0.6 cm mean error on a synthetic
        floor). The window here is half the smallest spacing between
        neighboring corners, at most calibrate_camera's.

    Outputs:
        (N, 2), ordered by order_corners().
    """
    ok, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS & ~cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return None
    cols, rows = pattern
    grid = order_corners(corners, pattern).reshape(rows, cols, 2)
    spacing = min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
                  np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
    half = int(np.clip(spacing / 2.0 - 1.0, 1, 5))
    refined = cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), SUBPIX_CRITERIA)
    return order_corners(refined, pattern)

def average_corners(frames, pattern, preprocess) -> tuple[np.ndarray | None, int, np.ndarray | None]:
    """
    Corners found in each undistorted frame, ordered and averaged.

    Outputs:
        (corners (N, 2) or None, frames used, the last undistorted frame with a board).
    """
    found, last = [], None
    for frame in frames:
        und, gray = undistorted_gray(frame, preprocess)
        c = find_ground_corners(gray, pattern)
        if c is not None:
            found.append(c)
            last = und
    if not found:
        return None, 0, None
    return np.mean(found, axis=0), len(found), last


# =============================================================================
# Fit
# =============================================================================

def board_points_cm(pattern, square_cm: float, origin_x_cm: float, origin_y_cm: float) -> np.ndarray:
    """
    Floor position of each inner corner, row-major like order_corners().

    Corner (c, r) is c squares right of corner 0 and r squares nearer the
    robot: (x0 + c*s, y0 - r*s), with corner 0 at (origin_x_cm, origin_y_cm).
    """
    cols, rows = pattern
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([origin_x_cm + c.ravel() * square_cm,
                     origin_y_cm - r.ravel() * square_cm], axis=1).astype(np.float64)

def reference_at_view_bottom(H: np.ndarray, frame_w: int, frame_h: int) -> tuple[np.ndarray, float]:
    """
    Move the floor origin forward/back so the bottom-center of the frame is at Y = 0.

    Purpose:
        The robot reference point is the floor at the bottom of the camera's
        view. It's hard to find with a tape measure and exact from the fit:
        project the frame's bottom-center and translate Y by it. X is kept,
        since the robot's centerline comes from the measured --origin-x-cm.

    Outputs:
        (H with the new origin, the Y shift applied in cm).
    """
    y_bottom = float(cv2.perspectiveTransform(np.array([[[frame_w / 2.0, float(frame_h)]]]), H)[0, 0, 1])
    T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -y_bottom], [0.0, 0.0, 1.0]])
    H2 = T @ H
    return H2 / H2[2, 2], -y_bottom

def fit(corners_px, pattern, square_cm, origin_x_cm, origin_y_cm, frame_size) -> dict:
    """
    The homography and its quality from ordered corners.

    Inputs:
        frame_size: (width, height).
        origin_y_cm: Corner 0's distance ahead of the reference; None puts
            the reference at the bottom-center of the view.

    Outputs:
        dict with H, errors_cm (per corner), corner0_cm (its floor position
        under the final origin) and y_shift_cm.
    """
    board = board_points_cm(pattern, square_cm, origin_x_cm, 0.0 if origin_y_cm is None else origin_y_cm)
    H, errors = fit_ground_homography(corners_px, board)
    shift = 0.0
    if origin_y_cm is None:
        H, shift = reference_at_view_bottom(H, *frame_size)
    corner0 = cv2.perspectiveTransform(np.asarray(corners_px[:1], np.float64).reshape(1, 1, 2), H)[0, 0]
    return {"H": H, "errors_cm": errors, "corner0_cm": (float(corner0[0]), float(corner0[1])),
            "y_shift_cm": shift}


# =============================================================================
# Output
# =============================================================================

def debug_image(und: np.ndarray, corners_px: np.ndarray, pattern, H: np.ndarray, lane_rect,
                corner0_cm, scale: int = 3) -> np.ndarray:
    """
    The undistorted frame, enlarged, with the corners, corner 0, the floor axes, the lane ROI and a projected 5 cm grid.

    If corner 0 isn't the far-left corner of the board, or the grid doesn't
    sit square on the floor, the fit isn't to be trusted. The yellow line is
    the robot's centerline (X = 0) and the triangle the reference point.
    """
    k = float(scale)
    img = cv2.resize(und, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
    at = lambda p: tuple(int(round(v * k)) for v in p)
    g = GroundHomography.from_matrix(H, (und.shape[1], und.shape[0]), 0.0, None)

    def text(s, org, color):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    # 5 cm grid over the floor the lane ROI sees; X = 0 is the robot's centerline
    x, y, w, h = lane_rect
    roi_floor = g.to_floor([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
    xs = np.arange(np.floor(roi_floor[:, 0].min() / 5) * 5, roi_floor[:, 0].max() + 5, 5.0)
    ys = np.arange(max(np.floor(roi_floor[:, 1].min() / 5) * 5, 0.0), roi_floor[:, 1].max() + 5, 5.0)
    for gx in xs:
        pts = g.to_image(np.stack([np.full(30, gx), np.linspace(ys[0], ys[-1], 30)], 1)) * k
        on_axis = abs(gx) < 1e-6
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False,
                      (0, 255, 255) if on_axis else (255, 200, 0), 2 if on_axis else 1, cv2.LINE_AA)
    for gy in ys:
        pts = g.to_image(np.stack([np.linspace(xs[0], xs[-1], 30), np.full(30, gy)], 1)) * k
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (255, 200, 0), 1, cv2.LINE_AA)
        end = pts[np.argmax(pts[:, 0])]
        if 0 <= end[1] < img.shape[0]:
            text(f"{gy:.0f} cm", (min(int(end[0]) + 4, img.shape[1] - 80), int(end[1]) + 5), (255, 200, 0))
    cv2.rectangle(img, at((x, y)), at((x + w, y + h)), (0, 200, 255), 1)
    text("lane ROI", (at((x, y))[0] + 4, at((x, y))[1] + 18), (0, 200, 255))

    cv2.drawChessboardCorners(img, pattern, (corners_px * k).reshape(-1, 1, 2).astype(np.float32), True)

    c0 = at(corners_px[0])
    cv2.circle(img, c0, 14, (0, 0, 255), 3, cv2.LINE_AA)
    cols = pattern[0]
    right = corners_px[1] - corners_px[0]
    up = corners_px[0] - corners_px[cols]                   # toward corner 0 from the next row: +Y
    for vec, name, color in ((right, "+X", (0, 0, 255)), (up, "+Y", (0, 200, 0))):
        tip = at(corners_px[0] + 3.0 * vec)
        cv2.arrowedLine(img, c0, tip, color, 3, cv2.LINE_AA, tipLength=0.25)
        text(name, (tip[0] + 6, tip[1] + 6), color)
    # Label to the left of corner 0 and above it, clear of both arrows
    text(f"corner 0 ({corner0_cm[0]:+.1f}, {corner0_cm[1]:+.1f}) cm",
         (max(c0[0] - 250, 4), max(c0[1] - 26, 20)), (0, 0, 255))

    ref = at(g.to_image([(0.0, 0.0)])[0])
    ref = (min(max(ref[0], 0), img.shape[1] - 1), min(max(ref[1], 0), img.shape[0] - 1))
    cv2.drawMarker(img, ref, (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 22, 3)
    text("reference (0, 0)", (ref[0] + 14, ref[1] - 8), (0, 255, 255))
    return img

def record(result: dict, pattern, square_cm, origin_x_cm, origin_y_cm, frame_size,
           preprocess: PreprocessParams, corners_px, frames_used: int) -> dict:
    """The JSON load_ground_homography reads, plus what's needed to audit or refit it."""
    lens_path = Path(preprocess.calibration_path)
    lens = json.loads(lens_path.read_text())
    errors = result["errors_cm"]
    return {
        "model": "floor_homography",
        "H": result["H"].tolist(),
        "axes": "frame px -> floor cm; X right+, Y forward+, origin at the robot reference point",
        "reference": ("floor at the bottom-center of the view" if origin_y_cm is None
                      else "measured: corner 0 is origin_cm from it"),
        "image_size": list(frame_size),
        "undistort_alpha": float(preprocess.undistort_alpha),
        "lens_calibration": {"file": lens_path.name, "created": lens.get("created"),
                             "sha256": lens_id(lens_path)},
        "pattern": list(pattern),
        "square_cm": square_cm,
        "origin_cm": [origin_x_cm, origin_y_cm],
        "corner0_cm": list(result["corner0_cm"]),
        "reprojection_cm": {"mean": float(np.mean(errors)), "max": float(np.max(errors)),
                            "worst_corner": int(np.argmax(errors))},
        "frames_used": frames_used,
        "corners_px": np.asarray(corners_px).tolist(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


# =============================================================================
# Command line
# =============================================================================

def grab_frames(args) -> list[np.ndarray]:
    """--image files, or --frames frames from the camera."""
    if args.image:
        frames = [cv2.imread(p) for p in args.image]
        missing = [p for p, f in zip(args.image, frames) if f is None]
        if missing:
            sys.exit(f"ERROR: could not read {', '.join(missing)}")
        return frames
    cap = open_camera(FRAME_W, FRAME_H)
    frames = []
    try:
        while len(frames) < args.frames:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    return frames

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, fit, and write the JSON and the debug image."""
    p = argparse.ArgumentParser(prog="calibrate_ground", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--square-cm", type=float, required=True,
                   help="side of one printed square in cm, MEASURED across several squares")
    p.add_argument("--origin-x-cm", type=float, required=True,
                   help="corner 0's distance right (+) or left (-) of the robot's centerline")
    p.add_argument("--origin-y-cm", type=float, default=None,
                   help="corner 0's distance ahead of a measured reference point; omit to use "
                        "the floor at the bottom-center of the view")
    p.add_argument("--pattern", default="9x6", help="INNER corners, cols x rows (10x7 squares = 9x6)")
    p.add_argument("--frames", type=int, default=10, help="camera frames to average the corners over")
    p.add_argument("--image", action="append", default=None, metavar="PATH",
                   help="fit from saved raw frames instead of the camera (repeatable)")
    p.add_argument("--out", default=str(GROUND_HOMOGRAPHY_PATH))
    p.add_argument("--debug-image", default=str(DEFAULT_DEBUG))
    p.add_argument("--save-raw", default=str(DEFAULT_RAW),
                   help="where to keep one raw frame, for a refit with --image")
    args = p.parse_args(argv)

    pattern = parse_pattern(args.pattern)
    preprocess = MEASURED.preprocess
    if preprocess.calibration_path is None or lens_id(preprocess.calibration_path) is None:
        sys.exit("ERROR: no lens calibration; the homography must be fit on undistorted frames. "
                 "Run scripts/calibrate_camera.py first.")

    frames = grab_frames(args)
    if not frames:
        sys.exit("ERROR: no frames")
    size = (frames[0].shape[1], frames[0].shape[0])
    if size != (FRAME_W, FRAME_H):
        sys.exit(f"ERROR: frames are {size[0]}x{size[1]}, the pipeline runs at {FRAME_W}x{FRAME_H}")

    try:
        corners, used, und = average_corners(frames, pattern, preprocess)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if corners is None:
        sys.exit(f"ERROR: no {pattern[0]}x{pattern[1]} board found in {len(frames)} frame(s). "
                 "The whole board must be in view, flat and well lit.")

    result = fit(corners, pattern, args.square_cm, args.origin_x_cm, args.origin_y_cm, size)
    lane_rect = resolve(MEASURED.roi.lane, (FRAME_H, FRAME_W))
    rec = record(result, pattern, args.square_cm, args.origin_x_cm, args.origin_y_cm, size,
                 preprocess, corners, used)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    cv2.imwrite(args.debug_image, debug_image(und, corners, pattern, result["H"], lane_rect,
                                              result["corner0_cm"]))
    if args.save_raw and not args.image:
        cv2.imwrite(args.save_raw, frames[0])

    # Coverage: distances outside the board are extrapolated
    x, y, w, h = lane_rect
    span_w = (corners[:, 0].max() - corners[:, 0].min()) / w
    span_h = (corners[:, 1].max() - corners[:, 1].min()) / h
    err = rec["reprojection_cm"]
    verdict = "GOOD" if err["max"] < 0.5 else "OK" if err["max"] < 1.0 else "POOR - see calibrate_ground.md"
    print(f"\nGround homography written to {out}")
    print(f"  frames used       {used} of {len(frames)}")
    print(f"  reprojection      mean {err['mean']:.3f} cm   max {err['max']:.3f} cm "
          f"(corner {err['worst_corner']})   -> {verdict}")
    print(f"  corner 0          ({rec['corner0_cm'][0]:+.2f}, {rec['corner0_cm'][1]:+.2f}) cm from the reference")
    g = GroundHomography.from_matrix(result["H"], size, 0.0, None)
    bl, br, top = g.to_floor([(0, FRAME_H), (FRAME_W, FRAME_H), (FRAME_W / 2, y)])
    print(f"  view bottom       {br[0] - bl[0]:.1f} cm wide;  lane ROI top {top[1]:.1f} cm ahead")
    print(f"  board covers      {100 * span_w:.0f}% of the lane ROI's width, {100 * span_h:.0f}% of its height")
    if span_w < 0.6 or span_h < 0.6:
        print("  WARNING: the board covers little of the lane ROI; distances outside it are extrapolated")
    print(f"\nCheck {args.debug_image}: corner 0 must be the board's far-left inner corner, "
          "+X to the right, +Y away from the robot, and the grid square on the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
