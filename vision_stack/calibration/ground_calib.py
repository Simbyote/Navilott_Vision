"""
ground_calib.py

Ground Calibration Capture and Solver

Purpose:
    Produce vision_stack/calibration/ground_calib.json from measurements you
    can take with a tape measure. Until this file exists, every centimetre the
    scene pipeline prints is invented, and the ROI self-check will say so.

WHAT THE MODEL NEEDS

    Three numbers, all recoverable from photographs of one reference strip at
    two or more measured distances:

        v0       image row of the horizon
        a_cm_px  A = f * camera_height, so that  y_cm = A / (v - v0)
        f_px     focal length in pixels, so that x_cm = (u - u0) * y / f

    v0 and A come from two (row, distance) pairs. f comes from the strip's
    LATERAL SPAN in pixels against its true length.

PHYSICAL SETUP

    1. Cut a strip of tape of known length and lay it ACROSS the lane,
       perpendicular to the direction the robot faces. 30 cm is a good length:
       long enough to measure accurately, short enough to stay in frame at
       close range. Measure it and write the number down.

    2. Park the robot square to the strip. Square matters -- a yawed robot
       makes the strip's ends land at different distances, which biases both
       the row and the span.

    3. Tape-measure from the CAMERA LENS to the near edge of the strip. Use
       the lens, not the front bumper: the model's origin is the camera's
       ground projection.

    4. Capture at three or four distances spread across the working range.
       25, 40, 60 and 80 cm works well. Two is the minimum, but three or more
       lets the solver check itself -- see VALIDATION below.

MEASURE THE SPAN, NOT THE THICKNESS

    f_px is solved from how many pixels the strip's LENGTH occupies -- its
    lateral extent across the image. Do NOT use its thickness. A strip's
    thickness is a depth extent, and depth is compressed hard by perspective:
    a 1 cm strip at 40 cm occupies about one pixel row. Measuring f from that
    would put the resolution limit straight into your focal length.

VALIDATION

    With three or more positions the solver fits v0 and A from the two extreme
    distances and then predicts the others. The residual on the middle
    positions is the honest error estimate. A large one means the optical axis
    is not level -- the assumption the whole projection model rests on -- and
    no amount of re-measuring will fix it. Re-level the mount instead.

USAGE

    python calibrate_ground.py capture --distance 25      # on the robot
    python calibrate_ground.py capture --distance 40
    python calibrate_ground.py capture --distance 60

    python calibrate_ground.py measure --image vision_stack/calibration/pos_25cm.png
        Inspect one detection before committing. Writes a _detect.png overlay.

    python calibrate_ground.py solve --span-cm 30
        Measures every captured position, solves, validates, writes the JSON.

    python calibrate_ground.py check
        Re-runs the ROI mapping self-check against the saved calibration.

    Detection can be overridden per image if the strip is hard to segment:
        python calibrate_ground.py solve --span-cm 30 --manual
"""

import os
import re
import sys
import json
import glob
import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, "vision_stack/src")

from run_state_pipeline import (
    GroundCalibration, GROUND_CALIB_PATH,
    FRAME_WIDTH, FRAME_HEIGHT, FPS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("calibrate")

CALIB_DIR = os.path.dirname(GROUND_CALIB_PATH) or "vision_stack/calibration"
CAMERA_HEIGHT_CM = 4.0
LANE_Y_TOP_FRAC = 0.70          # must match roi_crop.py


# =============================================================================
# Capture
# =============================================================================
def capture(distance_cm: float) -> str:
    """
    Purpose:
        Grab one frame and save it tagged with the measured distance.

    Notes:
        Uses the SAME GStreamer string as the pipeline, videoflip included. A
        calibration frame captured without the 180 degree flip would solve a
        mirrored model that then gets applied to flipped frames, and every
        lateral sign in the system would be inverted.
    """
    os.makedirs(CALIB_DIR, exist_ok=True)
    gst = (
        "libcamerasrc ! "
        f"video/x-raw,format=BGR,width={FRAME_WIDTH},height={FRAME_HEIGHT},"
        f"framerate={FPS}/1 ! "
        "videoconvert ! videoflip method=rotate-180 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera pipeline")

    frame = None
    for _ in range(10):                 # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()

    if frame is None:
        raise RuntimeError("No frame captured")

    path = os.path.join(CALIB_DIR, f"pos_{distance_cm:g}cm.png")
    cv2.imwrite(path, frame)
    log.info("Captured %s  (%dx%d)", path, frame.shape[1], frame.shape[0])
    return path


# =============================================================================
# Detection
# =============================================================================
@dataclass
class Measurement:
    """
    distance_cm: tape-measured range to the strip
    row: image row of the strip's centroid, SOURCE FRAME coordinates
    span_px: the strip's lateral extent in pixels
    clipped: the strip touched a frame edge, so span_px is a lower bound and
        must NOT be used to solve f_px
    path: source image
    """
    distance_cm: float
    row:         float
    span_px:     float
    clipped:     bool
    path:        str


def detect_reference(
        img: np.ndarray,
        bright_percentile: float = 99.0,
        min_aspect: float = 3.0,
    ) -> Optional[Tuple[float, float, Tuple[int, int, int, int]]]:
    """
    Purpose:
        Find the reference strip: the widest bright, wide-and-short blob.

    Outputs:
        (centroid_row, span_px, bbox) or None

    Notes:
        Segments on a high intensity percentile rather than a fixed threshold,
        so it survives the exposure differences between near and far captures.
        The aspect requirement is what stops it locking onto a glare patch or
        the robot's own chassis.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = float(np.percentile(gray, bright_percentile))
    mask = (gray >= thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_w = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 1 or w / max(h, 1) < min_aspect:
            continue
        if w > best_w:
            best, best_w = (c, (x, y, w, h)), w

    if best is None:
        return None

    contour, bbox = best
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        return None
    row = m["m01"] / m["m00"]
    x, _, w, _ = bbox
    clipped = x <= 1 or (x + w) >= img.shape[1] - 1
    return row, float(bbox[2]), bbox, clipped


def measure_image(path: str, write_overlay: bool = True) -> Optional[Measurement]:
    """
    Purpose:
        Detect the strip in one captured position and report its row and span.
    """
    img = cv2.imread(path)
    if img is None:
        log.error("Could not read %s", path)
        return None

    m = re.search(r"pos_([\d.]+)cm", os.path.basename(path))
    if not m:
        log.error("%s: filename must be pos_<distance>cm.png", path)
        return None
    distance = float(m.group(1))

    found = detect_reference(img)
    if found is None:
        log.error("%s: no strip found. Check lighting, or use --manual.", path)
        return None
    row, span, bbox, clipped = found

    if write_overlay:
        vis = img.copy()
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.line(vis, (0, int(row)), (vis.shape[1], int(row)), (0, 160, 255), 1)
        cv2.putText(vis, f"{distance:g}cm  row={row:.1f}  span={span:.0f}px",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out = path.replace(".png", "_detect.png")
        cv2.imwrite(out, vis)
        log.info("  overlay -> %s", out)

    log.info("%-28s %6.1f cm   row=%6.1f   span=%5.0f px%s",
             os.path.basename(path), distance, row, span,
             "  CLIPPED" if clipped else "")
    if clipped:
        log.warning("    Strip runs off the frame edge. Its row is still usable "
                    "but its span is a\n    lower bound, so this position "
                    "cannot contribute to f_px.")
    return Measurement(distance, row, span, clipped, path)


def measure_manual(path: str) -> Optional[Measurement]:
    """
    Purpose:
        Prompt for row and span when automatic detection is unreliable.

    Notes:
        Read the numbers off the _detect.png overlay, or off the image in any
        viewer that shows pixel coordinates. Manual is not a fallback of last
        resort -- if the strip is genuinely hard to segment, a careful manual
        reading beats a confident wrong detection.
    """
    m = re.search(r"pos_([\d.]+)cm", os.path.basename(path))
    if not m:
        return None
    distance = float(m.group(1))
    print(f"\n{os.path.basename(path)}  ({distance:g} cm)")
    try:
        row = float(input("  strip centre row (px, 0 = top): "))
        span = float(input("  strip span across image (px): "))
    except (ValueError, EOFError):
        return None
    return Measurement(distance, row, span, False, path)


# =============================================================================
# Solve
# =============================================================================
def solve(span_cm: float, manual: bool = False) -> None:
    """
    Purpose:
        Measure every captured position, fit the model, validate, save.
    """
    paths = sorted(glob.glob(os.path.join(CALIB_DIR, "pos_*cm.png")))
    paths = [p for p in paths if not p.endswith("_detect.png")]
    if len(paths) < 2:
        log.error("Need at least two pos_<distance>cm.png captures in %s; "
                  "found %d.", CALIB_DIR, len(paths))
        return

    log.info("Measuring %d positions (strip is %.1f cm long)\n", len(paths), span_cm)
    ms: List[Measurement] = []
    for p in paths:
        m = measure_manual(p) if manual else measure_image(p)
        if m:
            ms.append(m)

    if len(ms) < 2:
        log.error("Fewer than two usable measurements.")
        return

    ms.sort(key=lambda m: m.distance_cm)
    near, far = ms[0], ms[-1]

    if near.row <= far.row:
        log.error(
            "The near strip (%.0f cm) is at row %.1f and the far strip (%.0f cm) "
            "at row %.1f. Nearer ground must appear LOWER in the image, so the "
            "near row must be the larger number. Check the distance labels, or "
            "whether the capture went through the 180 degree flip.",
            near.distance_cm, near.row, far.distance_cm, far.row)
        return

    # v0 and A come from the two extreme (row, distance) pairs. f_px comes
    # only from spans that are fully inside the frame: a strip running off the
    # edge reports a clipped span, which silently deflates f. The residual
    # check below cannot catch that, because it validates the depth model and
    # f plays no part in it.
    unclipped = [m for m in ms if not m.clipped]
    if not unclipped:
        log.error(
            "Every strip ran off the frame edge, so f_px cannot be solved.\n"
            "The frame is %d px wide and covers roughly W*y/f cm at range y, so\n"
            "a %.0f cm strip needs to be viewed from far enough that it fits.\n"
            "Either use a shorter strip or add a capture at a greater distance.",
            FRAME_WIDTH, span_cm)
        return

    # Nearest unclipped position gives the most pixels across the strip and so
    # the best-conditioned f estimate.
    f_src = min(unclipped, key=lambda m: m.distance_cm)

    calib = GroundCalibration.solve_from_two_positions(
        v_near=near.row, y_near_cm=near.distance_cm, width_px_near=near.span_px,
        v_far=far.row, y_far_cm=far.distance_cm, width_px_far=far.span_px,
        feature_width_cm=span_cm,
        u0=FRAME_WIDTH / 2.0,
        f_span_px=f_src.span_px,
        f_span_range_cm=f_src.distance_cm,
    )
    log.info("f_px solved from %s (%.0f cm, %.0f px unclipped)",
             os.path.basename(f_src.path), f_src.distance_cm, f_src.span_px)
    if len(unclipped) < len(ms):
        log.info("%d of %d positions were clipped and used for depth only.",
                 len(ms) - len(unclipped), len(ms))

    print("\n" + "=" * 62)
    print("SOLVED")
    print("=" * 62)
    print(f"  v0        {calib.v0:8.2f}   (horizon row)")
    print(f"  a_cm_px   {calib.a_cm_px:8.1f}   (= f * camera height)")
    print(f"  f_px      {calib.f_px:8.1f}")
    implied_h = calib.a_cm_px / max(calib.f_px, 1e-6)
    flag = "OK" if abs(implied_h - CAMERA_HEIGHT_CM) < 1.0 else "CHECK MOUNT"
    print(f"  implied camera height {implied_h:.2f} cm "
          f"(measured {CAMERA_HEIGHT_CM:.1f})   [{flag}]")
    if flag != "OK":
        print("    A and f are solved independently, so this is a real cross-check.")
        print("    A large gap means the optical axis is not level, or a tape")
        print("    measurement is off. Re-level before trusting the model.")

    # ---- residuals on the positions not used in the fit --------------------
    interior = ms[1:-1]
    if interior:
        print(f"\n  residuals on {len(interior)} held-out position(s):")
        worst = 0.0
        for m in interior:
            pred = calib.a_cm_px / (m.row - calib.v0) if m.row > calib.v0 else float("inf")
            err = pred - m.distance_cm
            worst = max(worst, abs(err))
            print(f"    {m.distance_cm:5.1f} cm measured -> {pred:6.1f} cm "
                  f"predicted   ({err:+.1f} cm)")
        verdict = "OK" if worst < 2.0 else "AXIS LIKELY NOT LEVEL"
        print(f"  worst residual {worst:.1f} cm   [{verdict}]")
        if worst >= 2.0:
            print("    The model fits the two extremes exactly by construction, so")
            print("    error here is the honest one. Re-level the mount rather than")
            print("    re-measuring; the level-axis assumption is what failed.")
    else:
        print("\n  Only two positions captured, so nothing is held out and the fit")
        print("  is exact by construction. Capture a third to get a real error bar.")

    calib.save(GROUND_CALIB_PATH)
    print()
    check()


# =============================================================================
# Check
# =============================================================================
def check() -> None:
    """
    Purpose:
        Report what the saved calibration does to the lane ROI. This is the
        check that catches a calibration which is self-consistent but wrong for
        the ROI it gets applied to.
    """
    calib = GroundCalibration.load(GROUND_CALIB_PATH)
    roi_top = int(FRAME_HEIGHT * LANE_Y_TOP_FRAC)
    lane_rect = (0, roi_top, FRAME_WIDTH, FRAME_HEIGHT - roi_top)
    info = calib.describe_rows(lane_rect)
    if info["usable_frac"] >= 0.90:
        log.info("Calibration looks usable. Run: python stage_check.py 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build ground_calib.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="grab one frame at a measured distance")
    c.add_argument("--distance", type=float, required=True,
                   help="tape-measured cm from the LENS to the strip")

    m = sub.add_parser("measure", help="inspect detection on one image")
    m.add_argument("--image", required=True)

    s = sub.add_parser("solve", help="fit the model and write the JSON")
    s.add_argument("--span-cm", type=float, required=True,
                   help="true LENGTH of the strip across the lane, in cm")
    s.add_argument("--manual", action="store_true",
                   help="type in row and span instead of detecting them")

    sub.add_parser("check", help="re-run the ROI mapping self-check")

    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a.distance)
    elif a.cmd == "measure":
        measure_image(a.image)
    elif a.cmd == "solve":
        solve(a.span_cm, a.manual)
    else:
        check()