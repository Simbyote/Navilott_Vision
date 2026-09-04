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

    python ground_calib.py capture --distance 25      # on the robot
    python ground_calib.py capture --distance 40
    python ground_calib.py capture --distance 60

    python ground_calib.py measure --image vision_stack/calibration/pos_25cm.png
        Inspect one detection before committing. Writes a _detect.png overlay.

    python ground_calib.py solve --span-cm 30
        Measures every captured position, solves, validates, writes the JSON.

    python ground_calib.py check
        Re-runs the ROI mapping self-check against the saved calibration.

    Detection can be overridden per image if the strip is hard to segment:
        python ground_calib.py solve --span-cm 30 --manual
"""

import os
import re
import math
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

from state_pipeline import (
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
        search_top_frac: float = 0.50,
        dark: bool = False,
        max_area_frac: float = 0.20,
    ) -> Optional[Tuple[float, float, Tuple[int, int, int, int], bool]]:
    """
    Purpose:
        Find the reference strip: the widest bright, wide-and-short blob.

    Inputs:
        search_top_frac: ignore everything above this fraction of the frame
            height. The strip is on the FLOOR a few tens of centimetres ahead
            of a camera mounted 4 cm up, so it cannot appear in the upper half
            of the image. Without this the brightest wide-and-short thing in
            the room wins -- a light fixture, a window, or the wall/floor seam
            -- and it wins consistently enough to look like a real detection.
        dark: the strip is DARKER than its background -- black tape on a light
            floor or table. The default assumes the opposite, which is right
            for white lane markings on the course mat and wrong for anything
            you improvise indoors. Getting this backwards does not fail
            loudly: the detector happily returns patches of background, which
            is why the consistency checks exist.
        max_area_frac: reject a blob covering more than this share of the
            search region. A tape strip is thin. Anything filling a fifth of
            the frame is the surface it is stuck to.

    Outputs:
        (centroid_row, span_px, bbox, clipped) or None

    Notes:
        Segments on a high intensity percentile rather than a fixed threshold,
        so it survives the exposure differences between near and far captures.
        The aspect requirement is what stops it locking onto a glare patch or
        the robot's own chassis.
    """
    y_cut = int(img.shape[0] * search_top_frac)
    img = img[y_cut:, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if dark:
        thresh = float(np.percentile(gray, 100.0 - bright_percentile))
        mask = (gray <= thresh).astype(np.uint8) * 255
    else:
        thresh = float(np.percentile(gray, bright_percentile))
        mask = (gray >= thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_cap = max_area_frac * img.shape[0] * img.shape[1]
    best, best_w = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 1 or w / max(h, 1) < min_aspect:
            continue
        if w * h > area_cap:
            continue                       # that is the surface, not the strip
        if w > best_w:
            best, best_w = (c, (x, y, w, h)), w

    if best is None:
        log.debug("no candidate passed aspect >= %.1f and area <= %.0f%% "
                  "(polarity=%s)", min_aspect, 100 * max_area_frac,
                  "dark" if dark else "bright")
        return None

    contour, bbox = best
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        return None
    row = m["m01"] / m["m00"] + y_cut
    x, by, w, bh = bbox
    clipped = x <= 1 or (x + w) >= img.shape[1] - 1
    return row, float(bbox[2]), (x, by + y_cut, w, bh), clipped


def measure_image(path: str, write_overlay: bool = True,
                  search_top_frac: float = 0.50,
                  dark: bool = False) -> Optional[Measurement]:
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

    found = detect_reference(img, search_top_frac=search_top_frac, dark=dark)
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
# Candidate listing
# =============================================================================
def candidates(path: str, dark: bool = False, search_top_frac: float = 0.50,
               top: int = 6) -> None:
    """
    Purpose:
        List the blobs competing to be "the strip", so a wrong detection can be
        diagnosed from a terminal instead of by opening the overlay.

    Notes:
        measure() takes the WIDEST candidate. When that is wrong, this shows
        what it beat and why. The row column is the one to read: the strip lies
        on the floor a known distance ahead, so its row must change a lot
        between captures at different distances. A candidate whose row barely
        moves between your 45 cm and 60 cm shots is not on the ground plane --
        it is on a wall, and it will fit a model that is confidently wrong.
    """
    img = cv2.imread(path)
    if img is None:
        log.error("Could not read %s", path)
        return

    y_cut = int(img.shape[0] * search_top_frac)
    crop = img[y_cut:, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if dark:
        mask = (gray <= float(np.percentile(gray, 1.0))).astype(np.uint8) * 255
    else:
        mask = (gray >= float(np.percentile(gray, 99.0))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 5), np.uint8))

    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    area_cap = 0.20 * crop.shape[0] * crop.shape[1]
    for c in conts:
        x, y, w, h = cv2.boundingRect(c)
        m = cv2.moments(c)
        if abs(m["m00"]) < 1e-6:
            continue
        aspect = w / max(h, 1)
        why = []
        if aspect < 3.0:
            why.append("aspect")
        if w * h > area_cap:
            why.append("area")
        if x <= 1 or (x + w) >= crop.shape[1] - 1:
            why.append("clipped")
        rows.append((w, m["m01"] / m["m00"] + y_cut, h, aspect,
                     int(gray[y:y + h, x:x + w].mean()), ",".join(why) or "-"))

    rows.sort(reverse=True)
    print(f"\n{os.path.basename(path)}   polarity={'dark' if dark else 'bright'}"
          f"   searching below row {y_cut}")
    print(f"  {'rank':>4} {'span':>6} {'row':>8} {'height':>7} {'aspect':>7} "
          f"{'mean':>6}  flags")
    for i, (w, r, h, a, mean, flags) in enumerate(rows[:top], 1):
        mark = "  <- measure() picks this" if i == 1 and flags in ("-", "clipped") else ""
        print(f"  {i:>4} {w:6.0f} {r:8.1f} {h:7d} {a:7.1f} {mean:6d}  "
              f"{flags}{mark}")
    if not rows:
        print("  no blobs at all -- try the other polarity, or a lower "
              "--search-top-frac")


# =============================================================================
# Consistency
# =============================================================================
def consistency(ms: List[Measurement], span_cm: float) -> bool:
    """
    Purpose:
        Two checks that catch a detector locked onto the wrong object, before
        any of it reaches the solver.

    Outputs:
        True when both pass.

    Notes:
        Check 1, monotonic rows: nearer ground images LOWER, so row must fall
        as distance rises. This is geometry, not tuning -- a violation means
        the detections are not all of the same ground-plane object.

        Check 2, constant span*distance: span_px = f * L / y, so span * y = f * L
        is the same number at every position. It is the strongest single
        diagnostic available here because it needs no reference value -- the
        spread across positions IS the error.
    """
    ok = True
    print(f"\n  {'dist':>6} {'row':>8} {'span':>7} {'span*dist':>11}   "
          f"{'implied f':>9}")
    for m in sorted(ms, key=lambda m: m.distance_cm):
        prod = m.span_px * m.distance_cm
        print(f"  {m.distance_cm:6.1f} {m.row:8.1f} {m.span_px:7.0f} "
              f"{prod:11.0f}   {prod / max(span_cm, 1e-6):9.0f}"
              f"{'  CLIPPED' if m.clipped else ''}")

    ordered = sorted(ms, key=lambda m: m.distance_cm)
    rows = [m.row for m in ordered]
    if any(a <= b for a, b in zip(rows, rows[1:])):
        print("\n  [FAIL] rows are not strictly decreasing with distance.")
        print("         Nearer ground must image lower (larger row). Open the")
        print("         _detect.png overlays -- the detector is not finding the")
        print("         same object in every frame.")
        ok = False

    prods = [m.span_px * m.distance_cm for m in ordered if not m.clipped]
    if len(prods) < 2:
        print(f"\n  [FAIL] only {len(prods)} position(s) had an unclipped span, so")
        print("         span*distance cannot be cross-checked and f_px would rest")
        print("         on a single uncorroborated measurement.")
        print("         The frame covers roughly W*y/f cm at range y, so a strip")
        print("         short enough to fit at your NEAREST capture is what fixes")
        print("         this -- around 12 cm for this camera.")
        ok = False
    else:
        spread = (max(prods) - min(prods)) / max(max(prods), 1e-6)
        if spread > 0.15:
            print(f"\n  [FAIL] span*distance varies by {100 * spread:.0f}%. "
                  f"It should be constant")
            print("         (= f * strip length). The spans are not all of the "
                  "same physical")
            print("         object, or the strip length differs from --span-cm.")
            ok = False
        else:
            print(f"\n  [OK]   span*distance consistent to {100 * spread:.0f}% "
                  f"over {len(prods)} unclipped positions")
    return ok


def inspect(span_cm: float, search_top_frac: float = 0.50,
            dark: bool = False) -> None:
    """
    Purpose:
        Measure every capture and run the consistency checks WITHOUT solving.
        Run this before solve whenever a detection looks doubtful.
    """
    paths = [p for p in sorted(glob.glob(os.path.join(CALIB_DIR, "pos_*cm.png")))
             if not p.endswith("_detect.png")]
    ms = [m for m in (measure_image(p, search_top_frac=search_top_frac,
                                    dark=dark)
                      for p in paths) if m]
    if len(ms) < 2:
        log.error("Need at least two usable measurements.")
        return
    if consistency(ms, span_cm):
        print("\n  Both checks pass. Run: solve --span-cm %g" % span_cm)


# =============================================================================
# Solve
# =============================================================================
def solve(span_cm: float, manual: bool = False, dark: bool = False,
          force: bool = False) -> None:
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

    log.info("Measuring %d positions (strip %.1f cm, polarity=%s)\n",
             len(paths), span_cm, "DARK" if dark else "bright")
    if not dark:
        log.info("  (pass --dark if the strip is darker than the surface; "
                 "inspect and solve\n   must use the SAME polarity or they "
                 "measure different objects)")
    ms: List[Measurement] = []
    for p in paths:
        m = measure_manual(p) if manual else measure_image(p, dark=dark)
        if m:
            ms.append(m)

    if len(ms) < 2:
        log.error("Fewer than two usable measurements.")
        return

    if not consistency(ms, span_cm):
        log.error("\nConsistency checks failed -- not solving. Fix the "
                  "detections first;\na model fitted to the wrong object is "
                  "worse than no model, because it\nfails plausibly.")
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
        # A comes from rows alone and does not involve span_cm; f is inversely
        # proportional to it. So the height error inverts directly into the
        # strip length that WOULD make the model consistent -- which separates
        # "the tape is not the length you told me" from "the mount is tilted".
        implied_span = f_src.span_px * f_src.distance_cm * CAMERA_HEIGHT_CM / \
            max(calib.a_cm_px, 1e-6)
        print(f"    For a {CAMERA_HEIGHT_CM:.1f} cm mount, the measured span "
              f"implies a strip {implied_span:.1f} cm long,")
        print(f"    not the {span_cm:.1f} cm you passed. If your tape really is "
              f"{span_cm:.1f} cm, then the")
        print("    detection is measuring something else. Check the _detect.png "
              "overlay before")
        print("    touching the mount -- --span-cm is a measurement, not a knob, "
              "and changing")
        print("    it just rescales f and this number with it.")

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
        verdict = "OK" if worst < 2.0 else "FAILED"
        print(f"  worst residual {worst:.1f} cm   [{verdict}]")
        if worst >= 2.0:
            print("    The model fits the two extremes exactly by construction, so")
            print("    error here is the honest one. Either the detections are not")
            print("    all of the same object, or the optical axis is not level.")
            print("\n  NOT WRITING ground_calib.json. A calibration that fails its")
            print("  own residual check is worse than none: the pipeline would run")
            print("  and report centimetres that are quietly wrong. Use --force to")
            print("  override.")
            if not force:
                return
    else:
        print("\n  Only two positions captured, so nothing is held out and the fit")
        print("  is exact by construction. Capture a third to get a real error bar.")

    calib.save(GROUND_CALIB_PATH)
    print()
    check()


# =============================================================================
# Solve from course geometry (no tape measure)
# =============================================================================
def detect_lane_pairs(
        img: np.ndarray,
        bright_percentile: float = 98.0,
        min_run_px: int = 1,
        max_run_px: int = 40,
    ) -> List[Tuple[int, float, float]]:
    """
    Purpose:
        For each image row, find pairs of bright runs that could be the two
        lane lines, and return (row, u_left, u_right).

    Notes:
        Scans row by row rather than fitting contours, because the quantity
        wanted is the SEPARATION at a given row, and a contour fit would smear
        that across the depth the contour spans.

        Only rows with exactly two runs are kept. A row with three runs is
        ambiguous -- a dash edge, a seam, glare -- and there are enough clean
        rows that guessing is unnecessary.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = float(np.percentile(gray, bright_percentile))
    out = []
    for v in range(gray.shape[0]):
        mask = gray[v] >= thresh
        runs, start = [], None
        for u, on in enumerate(mask):
            if on and start is None:
                start = u
            elif not on and start is not None:
                if min_run_px <= u - start <= max_run_px:
                    runs.append(0.5 * (start + u - 1))
                start = None
        if start is not None and min_run_px <= len(mask) - start <= max_run_px:
            runs.append(0.5 * (start + len(mask) - 1))
        if len(runs) == 2:
            out.append((v, runs[0], runs[1]))
    return out


def solve_from_course(
        image_path: str,
        lane_cm: float,
        height_cm: float,
        long_cm: Optional[float] = None,
        long_rows: Optional[Tuple[float, float]] = None,
        fov_deg: Optional[float] = None,
    ) -> None:
    """
    Purpose:
        Calibrate from known course geometry instead of tape-measured ranges.

    Inputs:
        image_path: one frame, robot parked SQUARE in a lane, both lines visible
        lane_cm: true separation between the two lane-line centres
        height_cm: lens height above the ground
        long_cm: a known distance ALONG the direction of travel (dash length,
            dash pitch, intersection span). Needed for A.
        long_rows: (near_row, far_row) bounding that distance in the image
        fov_deg: horizontal field of view, as an alternative route to f when no
            longitudinal reference is available

    Notes:
        v0 falls out of the lateral constraint with f cancelling entirely:

            lane_cm = du * height_cm / (v - v0)   ->   v0 = v - du*h/lane_cm

        so every row with both lines visible is an independent estimate needing
        no distance measurement at all. The median over all such rows is robust
        to the odd row where a dash edge or a seam was mistaken for a line.

        A needs a LONGITUDINAL reference; lateral geometry cannot supply it.
        Lane width constrains the ratio h/(v-v0) and says nothing about scale
        along the ground.

        CIRCULARITY WARNING: whatever dimension you feed in here stops being an
        independent check. If you calibrate on lane_cm=14, stage_check 4's
        lane-width test will pass by construction. Validate against a dimension
        you did NOT use -- the 30 cm street or the 57.7 cm section.
    """
    img = cv2.imread(image_path)
    if img is None:
        log.error("Could not read %s", image_path)
        return

    pairs = detect_lane_pairs(img)
    if len(pairs) < 10:
        log.error("Only %d rows had exactly two bright runs. Need the robot "
                  "parked with both lane lines clearly visible.", len(pairs))
        return

    ests = [v - (ur - ul) * height_cm / lane_cm for v, ul, ur in pairs]
    ests_sorted = sorted(ests)
    v0 = ests_sorted[len(ests_sorted) // 2]
    spread = ests_sorted[int(0.84 * len(ests_sorted))] - ests_sorted[int(0.16 * len(ests_sorted))]

    print("\n" + "=" * 62)
    print("SOLVED FROM COURSE GEOMETRY")
    print("=" * 62)
    print(f"  rows used         {len(pairs)}")
    print(f"  v0                {v0:8.2f}   (+/- {spread / 2:.1f} across rows)")
    if spread > 12.0:
        print("    Spread is wide. Either the robot is not square to the lane,")
        print("    or some rows paired a dash edge with a line. Check the")
        print("    _pairs.png overlay.")

    vis = img.copy()
    for v, ul, ur in pairs[::3]:
        cv2.line(vis, (int(ul), v), (int(ur), v), (0, 200, 0), 1)
    cv2.line(vis, (0, int(v0)), (vis.shape[1], int(v0)), (0, 160, 255), 1)
    cv2.putText(vis, f"v0={v0:.1f}", (6, max(14, int(v0) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 255), 1, cv2.LINE_AA)
    out = image_path.replace(".png", "_pairs.png")
    cv2.imwrite(out, vis)
    print(f"  overlay           {out}")

    a_cm_px = f_px = None
    if long_cm and long_rows:
        v_near, v_far = max(long_rows), min(long_rows)
        if v_far <= v0:
            log.error("The far row %.1f is at or above the solved horizon "
                      "%.1f; it cannot be on the ground.", v_far, v0)
            return
        denom = 1.0 / (v_far - v0) - 1.0 / (v_near - v0)
        if denom <= 0:
            log.error("Longitudinal rows are inconsistent with v0.")
            return
        a_cm_px = long_cm / denom
        f_px = a_cm_px / height_cm
        print(f"\n  from {long_cm:.1f} cm between rows {v_near:.0f} and {v_far:.0f}:")
    elif fov_deg:
        f_px = (FRAME_WIDTH / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        a_cm_px = f_px * height_cm
        print(f"\n  from a {fov_deg:.1f} deg horizontal FOV:")
        print("    Check this against the capture mode. libcamera may crop")
        print("    rather than downscale to 480x360, in which case the sensor's")
        print("    datasheet FOV does not apply.")
    else:
        print("\n  No longitudinal reference and no FOV given, so A and f are")
        print("  unsolved. Lane width constrains only h/(v-v0) -- it says")
        print("  nothing about scale along the ground. Re-run with either")
        print("  --long-cm/--long-rows or --fov-deg.")
        return

    print(f"  a_cm_px           {a_cm_px:8.1f}")
    print(f"  f_px              {f_px:8.1f}")

    calib = GroundCalibration(
        v0=v0, a_cm_px=a_cm_px, f_px=f_px, u0=FRAME_WIDTH / 2.0, calibrated=True)
    calib.save(GROUND_CALIB_PATH)

    print("\n  VALIDATE ON A DIMENSION YOU DID NOT USE. lane_cm went in as an")
    print("  input, so stage_check 4's lane-width test is now circular. Check")
    print("  the 30 cm street width or the 57.7 cm section instead.")
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
    m.add_argument("--search-top-frac", type=float, default=0.50,
                   help="ignore everything above this fraction of frame height")
    m.add_argument("--dark", action="store_true",
                   help="strip is DARKER than the surface (e.g. black tape)")

    i = sub.add_parser("inspect", help="measure all captures and check consistency")
    i.add_argument("--span-cm", type=float, required=True)
    i.add_argument("--search-top-frac", type=float, default=0.50)
    i.add_argument("--dark", action="store_true",
                   help="strip is DARKER than the surface (e.g. black tape)")

    s = sub.add_parser("solve", help="fit the model and write the JSON")
    s.add_argument("--span-cm", type=float, required=True,
                   help="true LENGTH of the strip across the lane, in cm")
    s.add_argument("--manual", action="store_true",
                   help="type in row and span instead of detecting them")
    s.add_argument("--dark", action="store_true",
                   help="strip is DARKER than the surface (e.g. black tape)")

    sc = sub.add_parser("solve-course",
                        help="calibrate from known course dimensions, no tape")
    sc.add_argument("--image", required=True,
                    help="frame with the robot parked square, both lines visible")
    sc.add_argument("--lane-cm", type=float, default=14.0,
                    help="separation between the two lane-line centres")
    sc.add_argument("--height-cm", type=float, default=CAMERA_HEIGHT_CM)
    sc.add_argument("--long-cm", type=float,
                    help="a known distance ALONG travel (dash length, 29 cm "
                         "intersection, 57.7 cm section)")
    sc.add_argument("--long-rows",
                    help="near,far image rows bounding --long-cm, e.g. 320,268")
    sc.add_argument("--fov-deg", type=float,
                    help="horizontal FOV, as an alternative to --long-cm")

    k = sub.add_parser("candidates", help="list competing blobs in one image")
    k.add_argument("--image", required=True)
    k.add_argument("--dark", action="store_true")
    k.add_argument("--search-top-frac", type=float, default=0.50)
    k.add_argument("--top", type=int, default=6)

    sub.add_parser("check", help="re-run the ROI mapping self-check")

    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a.distance)
    elif a.cmd == "measure":
        measure_image(a.image, search_top_frac=a.search_top_frac,
                      dark=a.dark)
    elif a.cmd == "candidates":
        candidates(a.image, a.dark, a.search_top_frac, a.top)
    elif a.cmd == "inspect":
        inspect(a.span_cm, a.search_top_frac, a.dark)
    elif a.cmd == "solve":
        solve(a.span_cm, a.manual, a.dark, a.force)
    elif a.cmd == "solve-course":
        rows = None
        if a.long_rows:
            parts = [float(x) for x in a.long_rows.split(",")]
            rows = (parts[0], parts[1])
        solve_from_course(a.image, a.lane_cm, a.height_cm,
                          a.long_cm, rows, a.fov_deg)
    else:
        check()