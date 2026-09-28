"""Synthetic test frames, the configs they run under, and structural comparison.

Purpose:
    The one source of synthetic frames for the tests. Every parity test and
    every known-answer test draws from here, so a scene added once (a stop
    line, a new light) is exercised by all of them.

    Synthetic frames are drawn already undistorted: a mark drawn straight is
    straight. Running them through MEASURED would warp them with the real
    lens model and move every mark, so tests that feed synthetic frames use
    SCENE_CONFIG. Tests on real captures use MEASURED.

    Synthetic ground truth proves the arithmetic recovers what was drawn. It
    says nothing about whether the camera sees the world the way these
    frames assume, so it can't test accuracy in cm; that needs captures at
    measured offsets on the course.

Main package:
    synthetic_frame(): lane marks at known lane-ROI x on a plain road.
    scene(): synthetic_frame plus a stop line, a stop sign, lamps and noise.
    SCENES: named scenes covering every detection type and lane mode.
    SWEEP, sweep_frame(): mark brightness x width x noise across the lane gates.
    SCENE_CONFIG: MEASURED with undistortion off.
    ALT_CONFIG: SCENE_CONFIG with every stage's tuning changed, so a stage
        that ignores its config can't pass a parity test.
    same(): field-by-field equality across dataclasses, containers and arrays.
"""
from dataclasses import fields, is_dataclass, replace

import cv2
import numpy as np

from src.config import MEASURED
from src.params import FRAME_H, FRAME_W
from src.perception.color_branch import BlobFilter
from src.perception.geometry import CannyParams, LaneContourFilter, SignContourFilter
from src.perception.roi_crop import LANE, ROIBounds, resolve


# =============================================================================
# Configs
# =============================================================================

SCENE_CONFIG = replace(
    MEASURED,
    preprocess = replace(MEASURED.preprocess, calibration_path = None),
)

# Every stage's tuning moved off SCENE_CONFIG's. The values matter only in
# that each group changes the output on some scene (test_pipeline guards
# that); they are not a tuning.
ALT_CONFIG = replace(
    SCENE_CONFIG,
    preprocess = replace(SCENE_CONFIG.preprocess, gray_kernel = (7, 3), color_kernel = (3, 3)),
    roi = replace(
        SCENE_CONFIG.roi,
        lane = ROIBounds(x0 = 0.08, y0 = 0.62, x1 = 0.92, y1 = 1.00),
        traffic = ROIBounds(x0 = 0.30, y0 = 0.00, x1 = 0.70, y1 = 0.45),
        sign = ROIBounds(x0 = 0.55, y0 = 0.00, x1 = 1.00, y1 = 0.50),
    ),
    geometry = replace(
        SCENE_CONFIG.geometry,
        canny = CannyParams(threshold1 = 60.0, threshold2 = 180.0, close_kernel = (7, 3)),
        lane = LaneContourFilter(max_area = 1200.0, min_intensity = 100.0),
        sign = SignContourFilter(min_solidity = 0.75, epsilon_factor = 0.025),
    ),
    color = replace(SCENE_CONFIG.color, blob = BlobFilter(min_area = 80.0, ref_area = 400.0)),
    lane_offset = replace(
        SCENE_CONFIG.lane_offset,
        conf_threshold = 0.30, min_length_px = 20.0, max_width_px = 40.0,
        min_intensity = 140.0,          # the only one of these the scenes and sweep react to
    ),
)


# =============================================================================
# Lane marks
# =============================================================================

LANE_RECT = resolve(LANE, (FRAME_H, FRAME_W))   # what crop_rois cuts at this size with the default bounds
ROI_W, ROI_H = LANE_RECT[2], LANE_RECT[3]
ROI_CENTER = ROI_W / 2.0                        # where the robot sits in the lane ROI

def synthetic_frame(
        marks,
        mark_width: int = 6,
        road: int = 60,
        surround: int = 30,
        marking: int = 240,
    ) -> np.ndarray:
    """
    BGR frame whose lane ROI holds markings at known ROI-local x, so the correct offset is known exactly.

    Inputs:
        marks: ROI-local x positions, or (x, y_top, y_bottom) tuples to
            control vertical extent for dash and partial-visibility cases.
        mark_width: Marking width in px.
        road, surround, marking: Intensities for the road surface inside the
            lane ROI, everything outside it, and the markings.

    Outputs:
        (FRAME_H, FRAME_W, 3) uint8 BGR. A marking drawn at ROI x is recovered
        within half a pixel: marks at 150 and 290 come back as 149.5 and 289.5
        at 480x270 with SCENE_CONFIG.
    """
    x0, y0, w, h = LANE_RECT
    frame = np.full((FRAME_H, FRAME_W, 3), surround, np.uint8)
    frame[y0:y0 + h, x0:x0 + w] = road

    for mark in marks:
        if isinstance(mark, (int, float)):
            x, top, bottom = mark, 0, h
        else:
            x, top, bottom = mark
        fx = x0 + int(x)
        cv2.rectangle(
            frame,
            (fx - mark_width // 2, y0 + int(top)),
            (fx + mark_width // 2, y0 + int(bottom) - 1),
            (marking,) * 3, -1,
        )
    return frame

def expected_offset(left_x: float, right_x: float) -> float:
    """The offset the chain must recover for markings at these ROI x."""
    lane_center = (left_x + right_x) / 2.0
    return (ROI_CENTER - lane_center) / ROI_CENTER


# =============================================================================
# Scenes
# =============================================================================

def scene(
        marks=(150, 290),
        stop_line: tuple[int, int, int] | None = None,
        sign: bool = False,
        lights=(),
        noise_seed: int | None = None,
    ) -> np.ndarray:
    """
    A synthetic frame with any of the things the robot has to see.

    Inputs:
        marks: Lane marks, as synthetic_frame() takes them.
        stop_line: (x_left, x_right, y_top) in lane-ROI px: a 6 px white bar
            across the lane, as at an intersection.
        sign: A red octagon in the sign ROI (upper right).
        lights: BGR lamp colors, stacked downward in the traffic ROI (top center).
        noise_seed: Adds uniform noise in [0, 60) from this seed.

    Outputs:
        (FRAME_H, FRAME_W, 3) uint8 BGR.
    """
    frame = synthetic_frame(marks)
    if stop_line is not None:
        x_left, x_right, y_top = stop_line
        x0, y0 = LANE_RECT[0], LANE_RECT[1]
        cv2.rectangle(frame, (x0 + x_left, y0 + y_top), (x0 + x_right, y0 + y_top + 5),
                      (240, 240, 240), -1)
    if sign:
        cx, cy, r = int(FRAME_W * 0.78), int(FRAME_H * 0.25), 28
        pts = np.array([(cx + r * np.cos(np.pi / 8 + k * np.pi / 4),
                         cy + r * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)], np.int32)
        cv2.fillPoly(frame, [pts], (40, 40, 255))   # red, bright enough in gray for Canny
    for i, bgr in enumerate(lights):
        cv2.circle(frame, (FRAME_W // 2, 25 + i * 30), 11, bgr, -1)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        frame = cv2.add(frame, rng.integers(0, 60, frame.shape, dtype=np.uint8))
    return frame


RED_LAMP, YELLOW_LAMP, GREEN_LAMP = (0, 0, 255), (0, 220, 255), (0, 200, 0)

SCENES = {
    "two_boundary": scene(),
    "one_boundary": scene(marks=(150,)),
    "merge_close": scene(marks=(200, 230)),
    "span_wide": scene(marks=(5, 425)),
    "three_marks": scene(marks=(60, 200, 340)),
    "dashed": scene(marks=((150, 0, 30), (150, 50, 81), 290)),
    "blind": scene(marks=()),
    "sign_and_lights": scene(sign=True, lights=(RED_LAMP, YELLOW_LAMP, GREEN_LAMP)),
    "noise_a": scene(sign=True, lights=(RED_LAMP,), noise_seed=1),
    "noise_b": scene(marks=(100, 180, 300), noise_seed=7),
    # Intersections. Today a stop line passes the lane gates: the short one
    # becomes a lane candidate, the wide one merges with the left mark and
    # fails the area gate (see Section 6)
    "stop_line_short": scene(stop_line=(200, 240, 50)),
    "stop_line_wide": scene(stop_line=(170, 270, 50)),
    "stop_line_between_marks": scene(stop_line=(160, 280, 60)),
    "stop_line_far": scene(stop_line=(160, 280, 8)),
    "stop_line_no_marks": scene(marks=(), stop_line=(100, 330, 40)),
    "intersection": scene(stop_line=(170, 270, 50), sign=True, lights=(RED_LAMP,)),
}


# =============================================================================
# Gate sweep
# =============================================================================

# Marking brightness and width walked across the intensity, width and span
# gates in both geometry and lane offset, so a path whose gate is off by one
# flips a decision somewhere in the sweep. The fixed scenes sit far from
# every threshold and would not notice.
SWEEP = [(level, width, noise)
         for level in range(96, 160, 2)
         for width in (2, 6, 24, 44)
         for noise in (None, 3)]

def sweep_frame(level: int, width: int, noise: int | None) -> np.ndarray:
    """One SWEEP point: two marks at this brightness and width, with optional noise from this seed."""
    frame = synthetic_frame((150, 290), mark_width=width, marking=level)
    if noise is not None:
        rng = np.random.default_rng(noise)
        frame = cv2.add(frame, rng.integers(0, 12, frame.shape, dtype=np.uint8))
    return frame


# =============================================================================
# Comparison
# =============================================================================

def same(a, b, path: str = "root") -> None:
    """Structural equality across dataclasses, containers and numpy arrays; asserts with the path that differs."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{path}: arrays differ"
    elif is_dataclass(a) and is_dataclass(b):
        assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
        for f in fields(a):
            same(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for k in a:
            same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"

def differs(a, b) -> bool:
    """True if same() would fail."""
    try:
        same(a, b)
    except AssertionError:
        return True
    return False
