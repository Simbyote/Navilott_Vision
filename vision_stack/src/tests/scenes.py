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
    drive_sequence(): a stamped frame-and-sensor sequence long enough to move
        every Phase 3 vote, hold and integrator, for packet parity.
    SCENE_CONFIG: MEASURED with undistortion off and no ground plane or stop-line table.
    SYNTHETIC_GROUND: a known ground homography for 480x270 frames: floor
        cm from the bottom-center of the frame, 30 cm ahead at the lane ROI top.
    SYNTHETIC_STOP_LINE_TABLE: a known stop-line table for 480x270 frames.
    ALT_CONFIG: SCENE_CONFIG with every stage's tuning changed, so a stage
        that ignores its config can't pass a parity test.
    ALT_ESTIMATION: MEASURED_ESTIMATION with every Phase 3 field changed, for
        the same reason.
    floor_board(), lens_distort(): a checkerboard lying on a known floor, and
        the raw frame our lens would capture of it, for the ground calibration.
    same(): field-by-field equality across dataclasses, containers and arrays.
"""
from dataclasses import dataclass, fields, is_dataclass, replace

import cv2
import numpy as np

from src.config import MEASURED, MEASURED_ESTIMATION
from src.estimation.estimation import SensorSample
from src.params import FRAME_H, FRAME_W
from src.perception.color_branch import BlobFilter
from src.perception.geometry import CannyParams, LaneContourFilter, SignContourFilter, StopLineFilter
from src.perception.ground import GroundHomography
from src.perception.stop_line_table import StopLineTable
from src.perception.roi_crop import LANE, ROIBounds, resolve
from src.perception.stop_line_distance import StopLineDistanceConfig


# =============================================================================
# Configs
# =============================================================================

# A plausible floor plane for synthetic frames: the frame's bottom corners are
# 15 cm either side of the robot reference (bottom-center of the view), and
# 90 rows up, 30 cm ahead, the view is as wide as at the bottom minus the
# perspective squeeze. Only its being known matters to the tests.
SYNTHETIC_GROUND = GroundHomography.from_matrix(
    cv2.getPerspectiveTransform(
        np.float32([[0, FRAME_H], [FRAME_W, FRAME_H], [90, FRAME_H - 90], [FRAME_W - 90, FRAME_H - 90]]),
        np.float32([[-15, 0], [15, 0], [-15, 30], [15, 30]])),
    image_size=(FRAME_W, FRAME_H), undistort_alpha=0.0, lens_sha256=None)

# A known stop-line table for 480x270 frames: 3 cm at the view bottom, growing
# the way a flat floor does toward a horizon 160 rows up. Only its being known
# matters to the tests.
SYNTHETIC_STOP_LINE_TABLE = StopLineTable(a=800.0, b=160.0, c=-2.0, image_size=(FRAME_W, FRAME_H),
                                          undistort_alpha=0.0, lens_sha256=None, max_rows=70.0)

# Synthetic frames are drawn already undistorted, and a ground plane fit on
# real undistorted frames doesn't describe them, so both are off here
SCENE_CONFIG = replace(
    MEASURED,
    preprocess = replace(MEASURED.preprocess, calibration_path = None),
    ground = None,
    stop_line_table = None,
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
        lane = LaneContourFilter(max_area = 1200.0, min_intensity = 100.0, horizontal_edge_deg = 6.0,
                                 horizontal_min_run_px = 60.0, horizontal_band_px = 2.0),
        sign = SignContourFilter(min_solidity = 0.75, epsilon_factor = 0.025),
        stop_line = StopLineFilter(max_tilt_deg = 6.0, min_length_px = 80.0, min_thickness_px = 2.0,
                                   max_thickness_px = 30.0, min_intensity = 150.0,
                                   close_kernel = (11, 1), ref_length_px = 150.0),
    ),
    color = replace(SCENE_CONFIG.color, blob = BlobFilter(min_area = 80.0, ref_area = 400.0)),
    lane_offset = replace(
        SCENE_CONFIG.lane_offset,
        conf_threshold = 0.30, min_length_px = 20.0, max_width_px = 40.0,
        min_intensity = 140.0,          # the only one of these the scenes and sweep react to
        stop_line_overlap = 0.7,
    ),
    stop_line = StopLineDistanceConfig(min_confidence = 0.7),
    ground = SYNTHETIC_GROUND,
)

# Every Phase 3 field moved off MEASURED_ESTIMATION's, chosen so that each one
# alone changes the packets drive_sequence() produces (test_pipeline guards
# that). The sign gate sits above a small sign's 0.56, so the gated-sign path
# is reached; lane_roi_width_px is left for Pipeline to derive.
ALT_ESTIMATION = replace(
    MEASURED_ESTIMATION,
    ema_alpha = 0.6,
    max_offset_jump = 0.3,
    hold_max_frames = 4,
    cm_per_px = 0.05,
    vote_window = 5,
    min_confidence_traffic = 0.5,
    min_confidence_sign = 0.6,
    gyro_bias_dps = 1.5,
    heading_limit_deg = 20.0,
    max_dt_s = 0.3,
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
        lamp_radius: int = 11,
        sign_radius: int = 28,
        mark_width: int = 6,
        stop_line_thickness: int = 6,
        stop_line_tilt_deg: float = 0.0,
    ) -> np.ndarray:
    """
    A synthetic frame with any of the things the robot has to see.

    Inputs:
        marks: Lane marks, as synthetic_frame() takes them.
        stop_line: (x_left, x_right, y_top) in lane-ROI px: a white bar
            across the lane, as at an intersection.
        stop_line_thickness, stop_line_tilt_deg: The bar's height in px and
            its rotation about its center (+ = clockwise on screen, so the
            right end sits lower, nearer the robot).
        mark_width: Lane mark width in px; real tape is 20-40 px in the ROI.
        sign: A red octagon in the sign ROI (upper right).
        lights: BGR lamp colors, stacked downward in the traffic ROI (top center).
        noise_seed: Adds uniform noise in [0, 60) from this seed.
        lamp_radius, sign_radius: Size in px. Lamps score about 0.42 at 11,
            0.97 at 16 and 0.19 at 8; signs 0.71 at 28 and 0.56 at 16.

    Outputs:
        (FRAME_H, FRAME_W, 3) uint8 BGR.
    """
    frame = synthetic_frame(marks, mark_width=mark_width)
    if stop_line is not None:
        x_left, x_right, y_top = stop_line
        x0, y0 = LANE_RECT[0], LANE_RECT[1]
        if stop_line_tilt_deg == 0.0:
            cv2.rectangle(frame, (x0 + x_left, y0 + y_top),
                          (x0 + x_right, y0 + y_top + stop_line_thickness - 1), (240, 240, 240), -1)
        else:
            center = (x0 + (x_left + x_right) / 2.0, y0 + y_top + stop_line_thickness / 2.0)
            box = cv2.boxPoints((center, (x_right - x_left, stop_line_thickness), stop_line_tilt_deg))
            cv2.fillPoly(frame, [np.round(box).astype(np.int32)], (240, 240, 240))
    if sign:
        cx, cy, r = int(FRAME_W * 0.78), int(FRAME_H * 0.25), sign_radius
        pts = np.array([(cx + r * np.cos(np.pi / 8 + k * np.pi / 4),
                         cy + r * np.sin(np.pi / 8 + k * np.pi / 4)) for k in range(8)], np.int32)
        cv2.fillPoly(frame, [pts], (40, 40, 255))   # red, bright enough in gray for Canny
    for i, bgr in enumerate(lights):
        cv2.circle(frame, (FRAME_W // 2, 25 + i * 30), lamp_radius, bgr, -1)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        frame = cv2.add(frame, rng.integers(0, 60, frame.shape, dtype=np.uint8))
    return frame


RED_LAMP, YELLOW_LAMP, GREEN_LAMP = (0, 0, 255), (0, 220, 255), (0, 200, 0)

def _paint(frame, x_left, x_right, y_top, height, value):
    """A gray rectangle in lane-ROI coordinates, on a copy of frame."""
    out = frame.copy()
    x0, y0 = LANE_RECT[0], LANE_RECT[1]
    cv2.rectangle(out, (x0 + x_left, y0 + y_top), (x0 + x_right, y0 + y_top + height - 1), (value,) * 3, -1)
    return out

def _faint_stop_line():
    """
    A 165-gray line on a lighter patch of road (110): an edge weak enough
    that the Canny thresholds decide it. ALT_CONFIG finds it only with its
    own Canny settings, so a detector handed the wrong edges fails parity.
    """
    return _paint(_paint(scene(), 100, 340, 25, 41, 110), 120, 320, 40, 8, 165)

def _two_stop_lines():
    """
    The near stop line and the far side's line across the intersection.
    The far one starts further left, so detection order (by x) differs from
    nearest-first and a detector that doesn't sort fails parity.
    """
    return _paint(scene(stop_line=(170, 270, 60)), 110, 330, 6, 6, 240)

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
    # Stop lines. Geometry finds them from the lane ROI's edges without
    # changing the lane candidates. Short and wide stay apart from the lane
    # marks, pass the lane gates too, and lane_offset has to skip them. The
    # ones that touch a lane mark close into one contour with it, which
    # fails the lane area gate: those frames lose the lane (Phase 3 holds)
    "stop_line_short": scene(stop_line=(185, 255, 50)),
    "stop_line_wide": scene(stop_line=(170, 270, 50)),
    "stop_line_between_marks": scene(stop_line=(160, 280, 60)),
    "stop_line_far": scene(stop_line=(160, 280, 8)),
    "stop_line_no_marks": scene(marks=(), stop_line=(100, 330, 40)),
    "stop_line_touching": scene(stop_line=(120, 320, 40)),
    "stop_line_touching_left": scene(stop_line=(140, 260, 40)),
    "stop_line_thick_marks": scene(marks=(150, 330), mark_width=30, stop_line=(100, 380, 40),
                                   stop_line_thickness=12),
    "stop_line_clipped": scene(stop_line=(120, 320, 76)),
    "stop_line_tilted": scene(stop_line=(160, 280, 36), stop_line_tilt_deg=8.0),
    "stop_line_faint": _faint_stop_line(),
    "two_stop_lines": _two_stop_lines(),
    "intersection": scene(stop_line=(170, 270, 50), sign=True, lights=(RED_LAMP,)),
    # Not a stop line: shorter than any stop line, so no stop-line gate
    # accepts it, yet it passes the lane gates and still moves the offset
    "horizontal_blob": scene(stop_line=(200, 240, 50)),
}


# =============================================================================
# Frame sequences (Phase 3 works across frames)
# =============================================================================

@dataclass(frozen=True)
class SequenceFrame:
    """One stamped frame of a sequence, with the sensor readings for its window."""
    frame: np.ndarray
    frame_id: int
    timestamp_ms: int
    sensors: SensorSample | None
    segment: str                    # which part of the drive this frame belongs to

FRAME_MS = 50                       # 20 FPS

def drive_sequence() -> list[SequenceFrame]:
    """
    A drive that moves every Phase 3 stage: votes change both ways, a light
    and a sign below their gates, a lane dropout through hold into stale
    with the gyro turning, a frame gap inside the dropout, an offset jump,
    an approach to a stop line with one missed frame, noise, and frames with
    no sensors or no yaw.

    Outputs:
        SequenceFrames with frame ids from 100 and timestamps FRAME_MS apart
        (plus one 900 ms gap). Same content on every call.
    """
    steady = SensorSample(yaw_rate_dps=0.5, lateral_accel_mps2=0.1, wheel_speed_mps=0.3)
    turning = SensorSample(yaw_rate_dps=60.0, lateral_accel_mps2=-0.8, wheel_speed_mps=0.25)
    no_yaw = SensorSample(yaw_rate_dps=None, lateral_accel_mps2=0.2, wheel_speed_mps=0.3)
    strong, dim = dict(lamp_radius=16), dict(lamp_radius=8)
    segments = [
        # (segment, frame, count, sensors)
        ("cruise", scene(), 4, steady),
        ("red", scene(lights=(RED_LAMP,), **strong), 6, steady),
        ("yellow", scene(lights=(YELLOW_LAMP,), **strong), 6, steady),
        ("green", scene(lights=(GREEN_LAMP,), **strong), 6, None),
        ("red_flicker", scene(lights=(RED_LAMP,), **strong), 1, steady),
        ("green_after_flicker", scene(lights=(GREEN_LAMP,), **strong), 2, steady),
        ("dim_red", scene(lights=(RED_LAMP,), **dim), 4, steady),
        ("stop_sign", scene(sign=True), 6, steady),
        ("clear", scene(), 6, steady),
        ("small_sign", scene(sign=True, sign_radius=16), 6, steady),
        ("dropout", scene(marks=()), 5, turning),
        ("gap", scene(marks=()), 1, turning),              # arrives 900 ms late
        ("dropout_no_yaw", scene(marks=()), 2, no_yaw),
        ("dropout_late", scene(marks=()), 6, turning),
        ("recover", scene(), 4, steady),
        ("jump", scene(marks=(64, 204)), 3, steady),
        # Driving up to a stop line: nearer each frame, one frame that misses
        # it (glare), then on it, then past it
        ("stop_line_far", scene(stop_line=(170, 270, 12)), 2, steady),
        ("stop_line_nearer", scene(stop_line=(170, 270, 30)), 2, steady),
        ("stop_line_missed", scene(), 1, steady),
        ("stop_line_near", scene(stop_line=(170, 270, 50)), 2, steady),
        ("stop_line_on", scene(stop_line=(120, 320, 76)), 3, steady),
        ("stop_line_past", scene(), 4, steady),
        ("intersection", scene(stop_line=(170, 270, 50), sign=True, lights=(RED_LAMP,), **strong), 6, steady),
        ("noise", scene(sign=True, lights=(GREEN_LAMP,), noise_seed=3, **strong), 4, steady),
    ]
    out, fid, ts = [], 100, 10_000
    for segment, frame, count, sensors in segments:
        for _ in range(count):
            if segment == "gap":
                ts += 900 - FRAME_MS
            out.append(SequenceFrame(frame, fid, ts, sensors, segment))
            fid, ts = fid + 1, ts + FRAME_MS
    return out


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
# Ground plane
# =============================================================================

def floor_board(ground, origin_x_cm, origin_y_cm, square_cm, pattern=(9, 6)) -> np.ndarray:
    """
    An undistorted BGR frame of a checkerboard lying flat on the floor ground describes.

    Corner 0 (the far-left inner corner) is at (origin_x_cm, origin_y_cm) on the
    floor, squares run +X across and toward the robot down the image, with a
    white margin around them like a printed page. Each pixel is colored by
    where its center lands on the floor.
    """
    cols, rows = pattern
    ys, xs = np.mgrid[0:FRAME_H, 0:FRAME_W]
    floor = ground.to_floor(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)).reshape(FRAME_H, FRAME_W, 2)
    u = (floor[..., 0] - origin_x_cm) / square_cm + 1          # square column
    v = (origin_y_cm - floor[..., 1]) / square_cm + 1          # square row, toward the robot
    img = np.full((FRAME_H, FRAME_W), 90, np.uint8)
    img[(u >= -0.6) & (u < cols + 1.6) & (v >= -0.6) & (v < rows + 1.6)] = 255
    squares = (u >= 0) & (u < cols + 1) & (v >= 0) & (v < rows + 1)
    img[squares & ((np.floor(u) + np.floor(v)) % 2 == 1)] = 20
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def lens_distort(undistorted: np.ndarray, calibration_path: str, alpha: float = 0.0) -> np.ndarray:
    """
    The raw frame our lens would capture, given the undistorted view preprocess produces from it.

    The inverse of preprocess's undistortion: every raw pixel is sampled from
    where undistortion would send it. Resampling twice softens edges slightly.
    """
    import json
    calib = json.loads(open(calibration_path).read())
    K, dist = np.array(calib["camera_matrix"]), np.array(calib["dist_coeffs"])
    h, w = undistorted.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    und = cv2.undistortPoints(np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2),
                              K, dist, P=new_K).reshape(h, w, 2)
    return cv2.remap(undistorted, und[..., 0], und[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


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
