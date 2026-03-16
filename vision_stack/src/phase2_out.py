"""
phase2_output.py
================
Processing Phase Output Packaging — Phase 2, Step 7 of 7

Assembles the outputs of all upstream Phase 2 substages into a single
Phase2Output object and hands it to Phase 3. This substage performs no
computation — it only collects, validates structure, and packages.

Purpose
-------
Each Phase 2 substage produces its own result type:
  - feature_fusion.py  → list[DetectionObject]
  - perspective_transform.py → WarpResult | PointTransformResult | None

Phase 3 (navigation signal processing) needs a single, predictable object
to consume. This packaging substage is the contract boundary between Phase 2
and Phase 3. It enforces:

  1. A fixed, documented schema for the Phase 2 → Phase 3 handoff.
  2. Explicit presence/absence of transformed coordinates so Phase 3 can
     branch on whether metric-space data is available this frame.
  3. A frame-level timestamp and frame_id on the container itself, separate
     from per-detection timestamps, so Phase 3 can always identify the frame
     even when the detection list is empty.

No computation, filtering, thresholding, or decision logic belongs here.

Documented Phase 2 output contract (from pipeline.md)
------------------------------------------------------
Each DetectionObject already satisfies:
  {
    "type":       "traffic_light | stop_sign | lane_boundary",
    "position":   { "x": float, "y": float },
    "confidence": float,
    "timestamp":  int
  }

Phase2Output wraps a list of these plus frame-level metadata and optional
transformed coordinates. It does not add fields to DetectionObject.

Input contract
--------------
  detections : list[DetectionObject]
      From feature_fusion.fuse_detections().
      May be empty. An empty list is a valid frame with no detections.
      Each object must carry: type, position, confidence, timestamp.

  transformed_coords : TransformedCoords | None
      From perspective_transform.py, if lane candidates were present and
      homography calibration is available.
      None if:
        - No lane candidates survived this frame.
        - Homography calibration file is absent.
        - Perspective transform substage was skipped for this frame.
      Phase 3 must handle None explicitly.

  frame_id : int
      Authoritative frame counter from the capture loop.

  timestamp_ms : int
      Authoritative millisecond timestamp from the capture loop.
      Used as the frame-level timestamp on the container.
      Individual DetectionObjects carry their own per-detection timestamps
      inherited from upstream candidates — those are not replaced here.

Output contract
---------------
  Phase2Output (dataclass)

  .detections          list[DetectionObject]
                       All fused detections for this frame.
                       Ordering: as received from feature_fusion (confidence
                       descending within each class, traffic_light first,
                       then lane_boundary entries, then stop_sign — whatever
                       order fuse_detections() produced; not re-sorted here).

  .transformed_coords  TransformedCoords | None
                       Warped lane coordinate data. None if unavailable.

  .frame_id            int
                       Frame-level identifier.

  .timestamp_ms        int
                       Frame-level timestamp in milliseconds.

  .detection_count     int
                       len(detections). Convenience field; Phase 3 need not
                       call len() to check for empty frames.

TransformedCoords (dataclass)
  .warped_image        np.ndarray | None  — bird's-eye BGR image (Mode A)
                       None if only point transform was performed.
  .transformed_points  np.ndarray | None  — (N, 2) float32 bird's-eye coords
                       None if only image warp was performed.
  .source_points       np.ndarray | None  — (N, 2) float32 original points
                       Preserved from PointTransformResult; None if Mode A only.
  .output_width        int   — warped canvas width in pixels
  .output_height       int   — warped canvas height in pixels

  At least one of warped_image or transformed_points must be non-None.
  Both may be non-None if both transform modes were run.

Packaging rules
---------------
  PR-1  detections is taken as-is. No re-ordering, no filtering.
  PR-2  If transformed_coords is None, it is packaged as None.
        Phase 3 is responsible for handling the no-homography case.
  PR-3  detection_count is always len(detections) at package time.
  PR-4  frame_id and timestamp_ms are copied from the caller's arguments,
        not derived from any candidate or detection field.
  PR-5  TransformedCoords must have at least one non-None image or points
        field. If both are None, package_phase2_output raises ValueError.

Failure cases
-------------
  F1 — detections is None:
       Raises TypeError. An empty list must be passed explicitly for
       no-detection frames — None is not the same as empty.

  F2 — Any DetectionObject is missing a required field:
       Raises AttributeError at packaging time. Upstream substages are
       responsible for field completeness; this stage only checks presence.

  F3 — transformed_coords provided but both warped_image and
       transformed_points are None (PR-5):
       Raises ValueError. A TransformedCoords with no usable data is
       not a valid package input.

  F4 — frame_id or timestamp_ms not supplied:
       Both default to 0. The capture loop must supply correct values.
       Consistent zeros will cause Phase 3 temporal filtering to treat
       all frames as co-incident.

  F5 — transformed_coords.transformed_points shape is not (N, 2):
       No shape assertion is performed here — that validation is the
       responsibility of perspective_transform.py. Packaging does not
       re-validate upstream outputs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Stub DetectionObject — mirrors feature_fusion.DetectionObject exactly.
# In production, import DetectionObject from feature_fusion directly.
# ---------------------------------------------------------------------------

@dataclass
class DetectionObject:
    """
    Phase 2 per-detection schema (from pipeline.md).
    Fields forwarded to Phase 3: type, position, confidence, timestamp.
    Internal debug fields (bounding_box, label_detail) are carried through
    the package unchanged — Phase 3 ignores them.
    """
    type:         str    # "traffic_light" | "stop_sign" | "lane_boundary"
    position:     dict   # {"x": float, "y": float}
    confidence:   float  # [0.0, 1.0]
    timestamp:    int    # per-detection timestamp_ms

    # Internal debug fields — not navigation inputs
    bounding_box:  tuple = field(default=(0, 0, 0, 0))
    label_detail:  str   = field(default="")


# ---------------------------------------------------------------------------
# Transformed coordinate container
# ---------------------------------------------------------------------------

@dataclass
class TransformedCoords:
    """
    Holds perspective-transform outputs from perspective_transform.py.

    At least one of warped_image or transformed_points must be non-None.
    Both may be non-None if the caller ran both transform modes.

    warped_image        : (H_out, W_out, 3) uint8 BGR or None
    transformed_points  : (N, 2) float32 bird's-eye px coords or None
    source_points       : (N, 2) float32 original ROI coords or None
    output_width        : warped canvas width in pixels
    output_height       : warped canvas height in pixels
    """
    warped_image:       Optional[np.ndarray]   # Mode A result
    transformed_points: Optional[np.ndarray]   # Mode B result
    source_points:      Optional[np.ndarray]   # Mode B source, preserved
    output_width:       int
    output_height:      int


# ---------------------------------------------------------------------------
# Phase 2 output container
# ---------------------------------------------------------------------------

@dataclass
class Phase2Output:
    """
    Phase 2 → Phase 3 handoff object.

    Fields
    ------
    detections          : list[DetectionObject] — fused detections, as-is
    transformed_coords  : TransformedCoords | None — lane bird's-eye data
    frame_id            : int — frame-level identifier from capture loop
    timestamp_ms        : int — frame-level timestamp from capture loop
    detection_count     : int — len(detections), set at package time
    """
    detections:         List[DetectionObject]
    transformed_coords: Optional[TransformedCoords]
    frame_id:           int
    timestamp_ms:       int
    detection_count:    int


# ---------------------------------------------------------------------------
# Core substage
# ---------------------------------------------------------------------------

def package_phase2_output(
    detections:         List[DetectionObject],
    transformed_coords: Optional[TransformedCoords],
    frame_id:           int = 0,
    timestamp_ms:       int = 0,
) -> Phase2Output:
    """
    Package Phase 2 outputs into a Phase2Output for Phase 3 consumption.

    Parameters
    ----------
    detections          : list[DetectionObject] from feature_fusion.py
                          Empty list is valid. None is not.
    transformed_coords  : TransformedCoords from perspective_transform.py,
                          or None if homography data was unavailable.
    frame_id            : int from capture loop
    timestamp_ms        : int from capture loop

    Returns
    -------
    Phase2Output
    """
    # --- Guard: detections must be a list (F1) -----------------------------
    if detections is None:
        raise TypeError(
            "package_phase2_output: detections must be a list, not None. "
            "Pass an empty list for frames with no detections."
        )
    if not isinstance(detections, list):
        raise TypeError(
            f"package_phase2_output: detections must be list, got {type(detections).__name__}"
        )

    # --- Guard: verify required fields on each DetectionObject (F2) --------
    _REQUIRED = ("type", "position", "confidence", "timestamp")
    for i, d in enumerate(detections):
        for attr in _REQUIRED:
            if not hasattr(d, attr):
                raise AttributeError(
                    f"package_phase2_output: detections[{i}] missing field '{attr}'. "
                    "Upstream feature_fusion must supply complete DetectionObjects."
                )

    # --- Guard: TransformedCoords must have usable data (PR-5 / F3) --------
    if transformed_coords is not None:
        if (transformed_coords.warped_image is None and
                transformed_coords.transformed_points is None):
            raise ValueError(
                "package_phase2_output: transformed_coords has both "
                "warped_image and transformed_points as None. "
                "Pass None for transformed_coords if no transform data is available."
            )

    # --- Package (PR-1 through PR-4) ----------------------------------------
    return Phase2Output(
        detections         = detections,              # PR-1: as-is
        transformed_coords = transformed_coords,      # PR-2: None propagates
        frame_id           = frame_id,                # PR-4
        timestamp_ms       = timestamp_ms,            # PR-4
        detection_count    = len(detections),         # PR-3
    )


# ---------------------------------------------------------------------------
# Test example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal self-contained test exercising each packaging rule and failure case.
    No file I/O. No image loading. Prints pass/fail for each case.
    """
    import traceback

    def _pass(name):
        print(f"  [PASS] {name}")

    def _fail(name, exc):
        print(f"  [FAIL] {name}: {exc}")

    def _expect_raises(name, exc_type, fn):
        try:
            fn()
            print(f"  [FAIL] {name}: expected {exc_type.__name__}, got no exception")
        except exc_type as e:
            print(f"  [PASS] {name}: raised {exc_type.__name__}: {e}")
        except Exception as e:
            print(f"  [FAIL] {name}: wrong exception type {type(e).__name__}: {e}")

    print("\n--- package_phase2_output tests ---\n")

    # -----------------------------------------------------------------------
    # Case 1: Full package — all fields present, transformed coords available
    # -----------------------------------------------------------------------
    d1 = DetectionObject(
        type="traffic_light", position={"x": 80.0, "y": 45.0},
        confidence=0.91, timestamp=1000,
        bounding_box=(65, 30, 30, 30), label_detail="green")
    d2 = DetectionObject(
        type="lane_boundary", position={"x": 10.0, "y": 200.0},
        confidence=0.74, timestamp=1000,
        bounding_box=(3, 140, 14, 120), label_detail="lane_boundary")
    tc = TransformedCoords(
        warped_image       = np.zeros((240, 320, 3), dtype=np.uint8),
        transformed_points = np.array([[160.0, 220.0], [10.0, 200.0]], dtype=np.float32),
        source_points      = np.array([[80.0, 200.0],  [5.0, 180.0]], dtype=np.float32),
        output_width       = 320,
        output_height      = 240,
    )
    try:
        out = package_phase2_output([d1, d2], tc, frame_id=42, timestamp_ms=1712345678000)
        assert out.detection_count == 2
        assert out.frame_id == 42
        assert out.timestamp_ms == 1712345678000
        assert out.transformed_coords is tc
        assert len(out.detections) == 2
        _pass("Case 1: full package with transformed coords")
    except Exception as e:
        _fail("Case 1", e)

    # -----------------------------------------------------------------------
    # Case 2: Empty detection list, no transformed coords (PR-2)
    # -----------------------------------------------------------------------
    try:
        out = package_phase2_output([], None, frame_id=43, timestamp_ms=1712345678033)
        assert out.detection_count == 0
        assert out.transformed_coords is None
        _pass("Case 2: empty detections, no transformed coords")
    except Exception as e:
        _fail("Case 2", e)

    # -----------------------------------------------------------------------
    # Case 3: Transformed coords with points only (no warped image)
    # -----------------------------------------------------------------------
    tc_pts_only = TransformedCoords(
        warped_image       = None,
        transformed_points = np.array([[155.0, 218.0]], dtype=np.float32),
        source_points      = np.array([[77.0, 198.0]],  dtype=np.float32),
        output_width       = 320,
        output_height      = 240,
    )
    try:
        out = package_phase2_output([d2], tc_pts_only, frame_id=44, timestamp_ms=1712345678066)
        assert out.transformed_coords.warped_image is None
        assert out.transformed_coords.transformed_points is not None
        _pass("Case 3: points-only TransformedCoords")
    except Exception as e:
        _fail("Case 3", e)

    # -----------------------------------------------------------------------
    # Case 4: Warped image only (no transformed points)
    # -----------------------------------------------------------------------
    tc_img_only = TransformedCoords(
        warped_image       = np.zeros((240, 320, 3), dtype=np.uint8),
        transformed_points = None,
        source_points      = None,
        output_width       = 320,
        output_height      = 240,
    )
    try:
        out = package_phase2_output([d1], tc_img_only, frame_id=45, timestamp_ms=1712345678099)
        assert out.transformed_coords.transformed_points is None
        assert out.transformed_coords.warped_image is not None
        _pass("Case 4: image-only TransformedCoords")
    except Exception as e:
        _fail("Case 4", e)

    # -----------------------------------------------------------------------
    # F1 — detections is None
    # -----------------------------------------------------------------------
    _expect_raises(
        "F1: detections=None raises TypeError",
        TypeError,
        lambda: package_phase2_output(None, None)
    )

    # -----------------------------------------------------------------------
    # F1b — detections is not a list
    # -----------------------------------------------------------------------
    _expect_raises(
        "F1b: detections=dict raises TypeError",
        TypeError,
        lambda: package_phase2_output({"a": 1}, None)
    )

    # -----------------------------------------------------------------------
    # F2 — DetectionObject missing required field
    # -----------------------------------------------------------------------
    class _BadDetection:
        type       = "lane_boundary"
        position   = {"x": 0.0, "y": 0.0}
        # confidence is missing
        timestamp  = 0

    _expect_raises(
        "F2: missing 'confidence' field raises AttributeError",
        AttributeError,
        lambda: package_phase2_output([_BadDetection()], None)
    )

    # -----------------------------------------------------------------------
    # F3 — TransformedCoords with both fields None
    # -----------------------------------------------------------------------
    tc_empty = TransformedCoords(
        warped_image=None, transformed_points=None, source_points=None,
        output_width=320, output_height=240)
    _expect_raises(
        "F3: TransformedCoords both None raises ValueError",
        ValueError,
        lambda: package_phase2_output([d1], tc_empty)
    )

    # -----------------------------------------------------------------------
    # PR-1 verification — detections forwarded in original order
    # -----------------------------------------------------------------------
    d3 = DetectionObject(
        type="stop_sign", position={"x": 200.0, "y": 60.0},
        confidence=0.82, timestamp=1001)
    try:
        out = package_phase2_output([d1, d2, d3], None, frame_id=46)
        assert out.detections[0].type == "traffic_light"
        assert out.detections[1].type == "lane_boundary"
        assert out.detections[2].type == "stop_sign"
        assert out.detection_count == 3
        _pass("PR-1: detection order preserved")
    except Exception as e:
        _fail("PR-1", e)

    print("\n--- all tests complete ---\n")

    # -----------------------------------------------------------------------
    # Print a sample Phase2Output for visual inspection
    # -----------------------------------------------------------------------
    sample = package_phase2_output([d1, d2], tc, frame_id=99, timestamp_ms=1712345679000)
    print("Sample Phase2Output:")
    print(f"  frame_id         : {sample.frame_id}")
    print(f"  timestamp_ms     : {sample.timestamp_ms}")
    print(f"  detection_count  : {sample.detection_count}")
    print(f"  transformed_coords: warped={sample.transformed_coords.warped_image.shape}, "
          f"points={sample.transformed_coords.transformed_points}")
    for d in sample.detections:
        print(f"    {d.type:<16} pos={d.position}  conf={d.confidence:.2f}  ts={d.timestamp}")