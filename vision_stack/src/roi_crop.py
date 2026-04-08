"""
roi_crop.py
===========
ROI Cropping substage — Vision Perception Phase (Phase 2, Step 2 of 7)

Purpose
-------
Partitions the preprocessed BGR frame into three spatially distinct regions
before the pipeline splits into parallel branches. Cropping serves two goals:

  1. Reduce the pixel area each downstream operation must process. Canny, HSV
     thresholding, and contour extraction all scale with pixel count. Feeding
     only the relevant region of the frame to each branch cuts their cost
     proportionally.

  2. Eliminate irrelevant regions from each branch's input. Lane detection has
     no use for the sky. Traffic light detection has no use for the road
     surface. Sign detection has no use for the left lane. Restricting input
     reduces false positive rate before any threshold is applied.

No detection, conversion, or analysis of any kind is performed here.

Input contract
--------------
  frame : np.ndarray
      Shape  : (H, W, 3)
      Dtype  : uint8
      Color  : BGR
      Source : output of preprocess_frame() — undistorted, equalized, blurred

  frame_id : int
      Monotonically increasing frame counter from the capture loop.
      Carried into each ROIResult for downstream traceability.

Output contract
---------------
  Returns : ROICropResult (dataclass)
      .lane_roi         np.ndarray  (H//2, W, 3)         uint8 BGR
      .traffic_roi      np.ndarray  (H//2, W//2, 3)      uint8 BGR
      .sign_roi         np.ndarray  (H, W//2, 3)         uint8 BGR
      .lane_rect        (x, y, w, h) pixel coords in source frame
      .traffic_rect     (x, y, w, h) pixel coords in source frame
      .sign_rect        (x, y, w, h) pixel coords in source frame
      .frame_id         int         copied from input
      .source_shape     (H, W)      original frame dimensions

  Each ROI array is a NumPy view (slice) of the input frame — no copy is made.
  The caller must not modify the input frame while any ROI array is live.
  If a copy is needed (e.g. for writing to disk), call .copy() explicitly.

ROI coordinate definitions
--------------------------
All coordinates are computed deterministically from (H, W) = frame.shape[:2].

  Lane ROI — lower half of frame
  ┌──────────────────────────────┐  y=0
  │                              │
  │         (discarded)          │  y=H//2 - 1
  ├──────────────────────────────┤  y=H//2
  │                              │
  │          lane_roi            │
  │                              │  y=H-1
  └──────────────────────────────┘
    x=0                       x=W-1

    x  = 0
    y  = H // 2
    w  = W
    h  = H - H // 2      (handles odd H correctly)
    slice: frame[H//2 : H,  0 : W]

  Traffic Light ROI — top-center half of frame
  ┌────┬──────────────┬────┐  y=0
  │    │              │    │
  │disc│  traffic_roi │disc│
  │    │              │    │  y=H//2 - 1
  ├────┴──────────────┴────┤  y=H//2
  │       (discarded)      │  y=H-1
  └────────────────────────┘
    x=0  x=W//4        x=3*W//4  x=W-1

    x  = W // 4
    y  = 0
    w  = W - 2*(W//4)   (= W//2 for even W; correct for odd W)
    h  = H // 2
    slice: frame[0 : H//2,  W//4 : W//4 + w]

  Sign ROI — right half of full frame height
  ┌──────────┬──────────┐  y=0
  │          │          │
  │(discarded│ sign_roi │
  │          │          │
  │          │          │  y=H-1
  └──────────┴──────────┘
    x=0    x=W//2     x=W-1

    x  = W // 2
    y  = 0
    w  = W - W // 2    (handles odd W correctly)
    h  = H
    slice: frame[0 : H,  W//2 : W]

Failure cases
-------------
  F1 — frame is None:
       Raises ValueError before any slicing occurs.

  F2 — frame is not (H, W, 3) uint8 BGR:
       Wrong dtype or wrong number of channels produces silently wrong ROIs
       downstream. Asserted at function entry.

  F3 — frame dimensions too small:
       If W < 4 or H < 2, integer division produces zero-size or negative
       slices. Minimum usable input is (2, 4, 3). Asserted at entry.

  F4 — caller modifies frame while ROI views are live:
       ROIs are NumPy views, not copies. Writes to the source frame after
       crop_rois() returns will silently corrupt ROI data. The caller owns
       this contract — this function cannot enforce it.

  F5 — frame_id not provided or negative:
       Accepted as-is; no semantic validation. Convention: 0-indexed, monotonic.
"""

import cv2
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ROICropResult:
    lane_roi:     np.ndarray   # (H//2,         W,     3) uint8 BGR — view
    traffic_roi:  np.ndarray   # (H//2,         W//2,  3) uint8 BGR — view
    sign_roi:     np.ndarray   # (H,            W//2,  3) uint8 BGR — view
    lane_rect:    tuple        # (x, y, w, h) in source frame pixels
    traffic_rect: tuple        # (x, y, w, h) in source frame pixels
    sign_rect:    tuple        # (x, y, w, h) in source frame pixels
    frame_id:     int          # from caller
    source_shape: tuple        # (H, W) of input frame


# ---------------------------------------------------------------------------
# Core substage
# ---------------------------------------------------------------------------

def crop_rois(frame: np.ndarray, frame_id: int = 0) -> ROICropResult:
    """
    Partition one preprocessed BGR frame into three ROIs.

    Parameters
    ----------
    frame    : uint8 BGR ndarray, shape (H, W, 3)
    frame_id : integer identifier carried from the capture loop

    Returns
    -------
    ROICropResult  (see module docstring for full field specification)
    """
    # --- Guard: input validation -------------------------------------------
    if frame is None:
        raise ValueError("crop_rois: received None frame")
    if frame.dtype != np.uint8:
        raise TypeError(f"crop_rois: expected uint8, got {frame.dtype}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"crop_rois: expected (H,W,3) BGR, got shape {frame.shape}")

    H, W = frame.shape[:2]

    if H < 2 or W < 4:
        raise ValueError(
            f"crop_rois: frame too small for ROI subdivision, got ({H}, {W}). "
            "Minimum usable size is (2, 4)."
        )

    # --- Coordinate computation --------------------------------------------
    # Lane: lower half
    lane_x = 0
    lane_y = H // 2
    lane_w = W
    lane_h = H - H // 2

    # Traffic light: top-center half
    tl_x = W // 4
    tl_y = 0
    tl_w = W - 2 * (W // 4)
    tl_h = H // 2

    # Sign: right half, full height
    sign_x = W // 2
    sign_y = 0
    sign_w = W - W // 2
    sign_h = H

    # --- Slicing (views, no copy) ------------------------------------------
    # Adding a copy would cost memory but would prevent cross-stage mutations (considered)
    lane_roi    = frame[lane_y : lane_y + lane_h,  lane_x : lane_x + lane_w]
    traffic_roi = frame[tl_y   : tl_y   + tl_h,   tl_x   : tl_x   + tl_w ]
    sign_roi    = frame[sign_y : sign_y + sign_h,  sign_x : sign_x + sign_w]

    return ROICropResult(
        lane_roi     = lane_roi,
        traffic_roi  = traffic_roi,
        sign_roi     = sign_roi,
        lane_rect    = (lane_x,  lane_y,  lane_w,  lane_h),
        traffic_rect = (tl_x,    tl_y,    tl_w,    tl_h),
        sign_rect    = (sign_x,  sign_y,  sign_w,  sign_h),
        frame_id     = frame_id,
        source_shape = (H, W),
    )


# ---------------------------------------------------------------------------
# Debug overlay helper
# ---------------------------------------------------------------------------

# BGR colors for each ROI rectangle drawn on the overlay image
_LANE_COLOR    = (0,   255,   0)   # green
_TRAFFIC_COLOR = (255,   0,   0)   # blue
_SIGN_COLOR    = (0,     0, 255)   # red
_RECT_THICKNESS = 2


def draw_roi_overlay(frame: np.ndarray, result: ROICropResult) -> np.ndarray:
    """
    Return a copy of frame with the three ROI rectangles drawn on it.
    Does not modify the input frame.
    Labels: 'lane' (green), 'traffic' (blue), 'sign' (red).
    """
    overlay = frame.copy()

    def _draw(rect, color, label):
        x, y, w, h = rect
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, _RECT_THICKNESS)
        cv2.putText(overlay, label, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    _draw(result.lane_rect,    _LANE_COLOR,    "lane")
    _draw(result.traffic_rect, _TRAFFIC_COLOR, "traffic")
    _draw(result.sign_rect,    _SIGN_COLOR,    "sign")

    return overlay


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone test harness.

    Loads every .jpg / .png from s1-s5, runs crop_rois(), and writes four
    debug images per input into vision_stack/sample_img/duckietown/sN/results/

      stem_roi_overlay.png   — original frame with three ROI rectangles
      stem_roi_lane.png      — lane ROI crop
      stem_roi_traffic.png   — traffic light ROI crop
      stem_roi_sign.png      — sign ROI crop

    Coordinate definitions are printed once per unique image resolution found.
    """
    import os

    SAMPLE_DIRS = [
        "vision_stack/frames/trackT3",
        "vision_stack/frames/trackT4",
        "vision_stack/frames/trackT5"
    ]
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    seen_shapes = set()
    total_ok    = 0
    total_fail  = 0
    frame_id    = 0

    for sample_dir in SAMPLE_DIRS:
        if not os.path.isdir(sample_dir):
            print(f"[SKIP] Not found: {sample_dir}")
            continue

        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        image_files = sorted(
            f for f in os.listdir(sample_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            print(f"[SKIP] No images in {sample_dir}")
            continue

        for filename in image_files:
            img_path = os.path.join(sample_dir, filename)
            stem     = os.path.splitext(filename)[0]

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[FAIL] Could not read: {img_path}")
                total_fail += 1
                continue

            try:
                result = crop_rois(frame, frame_id=frame_id)
            except (ValueError, TypeError) as e:
                print(f"[FAIL] {img_path}: {e}")
                total_fail += 1
                continue

            # Print coordinate table once per unique resolution
            H, W = result.source_shape
            if (H, W) not in seen_shapes:
                seen_shapes.add((H, W))
                print(f"\n[COORDS] Resolution {W}x{H}:")
                print(f"  lane_rect    : {result.lane_rect}")
                print(f"  traffic_rect : {result.traffic_rect}")
                print(f"  sign_rect    : {result.sign_rect}")

            # Debug image 1: overlay
            overlay = draw_roi_overlay(frame, result)
            cv2.imwrite(
                os.path.join(results_dir, f"{stem}_roi_overlay.png"), overlay)

            # Debug image 2: lane ROI (copy for safe write)
            cv2.imwrite(
                os.path.join(results_dir, f"{stem}_roi_lane.png"),
                result.lane_roi.copy())

            # Debug image 3: traffic light ROI
            cv2.imwrite(
                os.path.join(results_dir, f"{stem}_roi_traffic.png"),
                result.traffic_roi.copy())

            # Debug image 4: sign ROI
            cv2.imwrite(
                os.path.join(results_dir, f"{stem}_roi_sign.png"),
                result.sign_roi.copy())

            print(f"[OK] frame_id={frame_id}  {img_path}")
            frame_id  += 1
            total_ok  += 1

    print(f"\nDone. {total_ok} processed, {total_fail} failed.")