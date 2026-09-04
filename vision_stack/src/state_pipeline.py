"""
state_pipeline.py

Navilott Scene-State Pipeline (experimental alternative to run_pipeline.py)

Purpose:
    Replaces the midpoint lane-offset estimator with a layered scene estimator.
    run_pipeline.py is untouched and remains the working reference; this file
    is a parallel entry point so both can be run against the same modules.

    The midpoint estimator answers exactly one question -- "am I centred
    between two detections this frame" -- and has no answer at all when a
    boundary drops out, when an intersection replaces the boundaries with a
    transverse line, or when confidence falls below the anchor gate. This
    pipeline replaces that single question with five layers, each of which is
    independently testable and none of which needs to know what a "T" or an "L"
    looks like in the image.

LAYER STACK

    L1  Ground projection   contour  -> segment in centimetres on the ground
    L2  Classification      segment  -> longitudinal / transverse / rejected
    L3  Ground map          segments -> short-horizon map, odometry-propagated
    L4  Scene state         map      -> LANE_FOLLOW / APPROACHING_STOP / ...
    L5  Control             state    -> wheel command (reuses pd_control.py)

    Each layer consumes only the layer below it. L1-L3 have no notion of
    driving; L5 has no notion of pixels. This is the split that lets you
    retune perception without retuning control.

WHY SHAPES ARE NOT CLASSIFIED

    "T" and "L" are appearances, not properties of the course. The same
    physical intersection presents as _, T, L or a bare stroke depending on
    approach angle, lateral offset, and which arm happens to fall inside the
    ROI. Mapping appearance to meaning is many-to-many and needs a special case
    per viewpoint.

    L2 classifies primitives instead: a segment is longitudinal or transverse
    in GROUND coordinates. A "T" is then just a longitudinal segment plus a
    transverse segment in the same frame -- which falls out of L3 for free and
    still works when one arm is clipped or missed entirely.

WHY GROUND COORDINATES

    Image-space angle depends on the robot's lateral offset and heading as much
    as on the line itself, so the boundary between "/" and "_" moves around as
    the robot drives. After projection the thresholds are fixed, and width in
    centimetres becomes available -- which is the single most discriminative
    feature the pipeline has. A lane line is ~1 cm wide; a puzzle-mat seam, a
    curb, and the arena wall base are not. Most false positives die on width
    alone, before any angle logic runs.

THE 22 cm BLIND ZONE

    Nothing closer than ~22 cm is observable (camera at 4 cm, forward-facing,
    chassis occluding the near field). A stop line detected at 30 cm therefore
    leaves the field of view 22 cm BEFORE the robot reaches it. A purely
    reactive state machine loses the intersection during the approach, which is
    when it matters.

    L3 exists for this. A transverse segment is entered into the ground map on
    detection and then propagated by odometry, so the state machine keeps
    acting on it after it disappears. Every intersection trigger in L4 is a
    DISTANCE trigger, never a timeout and never a frame count.

CAMERA MOUNTING

    The camera is mounted upside down. The GStreamer pipeline corrects this
    with `videoflip method=rotate-180`, so every stage downstream -- including
    all of this file -- sees a right-way-up frame. That is also what makes
    contracts.py's "image x increases to the robot's RIGHT" true, and what lets
    roi_crop take the BOTTOM of the frame as the lane ROI.

    Two consequences worth holding on to:

    1. Saved frame captures must have gone through the same flip. If a dataset
       was dumped without it, roi_crop's lane ROI grabs the ceiling and nothing
       downstream is meaningful. Cheap check: the lane ROI fixtures should show
       floor.

    2. rotate-180 cancels a mount that is inverted EXACTLY. Physical roll in
       the bracket survives it, and shows up as every lane line reading with
       the same nonzero angle. GroundCalibration.roll_deg corrects that; see
       estimate_roll_from_transverse() for how to measure it.

SIGN CONVENTIONS (canonical, matching contracts.py)

    Ground frame is robot-fixed:  +x = right,  +y = forward,  origin at the
    camera's ground projection.

    offset_norm > 0  ->  lane centre is RIGHT of the robot
                     ->  robot is LEFT of lane centre
                     ->  robot must steer RIGHT

    heading_error > 0  ->  lane direction runs off to the right ahead
                       ->  robot must steer RIGHT

    yaw_rate > 0 (imu.py)  ->  robot is turning right.

DELIBERATELY NOT IMPLEMENTED

    The out-of-bounds recovery behaviour (stop, localise, re-plan) is scoped
    out per the design discussion. LOST is a terminal state here: it stops the
    robot and holds. Wire recovery in only after basic lane-keeping is solid.

    Traffic light and stop sign detection are not run. They are gated off in
    run_pipeline.py anyway, and adding them here before the geometry path is
    trusted only widens the surface being debugged.

STATUS OF THE NUMBERS IN THIS FILE

    GroundCalibration ships with PLACEHOLDER optics. They are geometrically
    self-consistent but they are not your camera. Run the two-position
    calibration (see GroundCalibration.solve_from_two_positions) and write the
    JSON before trusting any centimetre value this file prints.
"""

# =============================================================================
# Standard library
# =============================================================================
import os
import sys
import json
import math
import time
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Deque

# =============================================================================
# Third-party
# =============================================================================
import cv2
import numpy as np

# =============================================================================
# Pipeline modules
# =============================================================================
sys.path.insert(0, "vision_stack/src")

from roi_crop import crop_rois
from preprocess import preprocess_frame
from geometry import (
    run_geometry_branch,
    CannyParams, LaneContourFilter, SignContourFilter,
)
from pd_control import PDController, PDConfig

# Hardware-backed modules are imported lazily inside main(). imu.py pulls in
# board/busio/adafruit and system.py pulls in pigpio/tm1637; neither exists on
# a development machine, and replay mode must run there.

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("state_pipeline")

# =============================================================================
# Capture configuration (mirrors run_pipeline.py)
# =============================================================================
FRAME_WIDTH = 480
FRAME_HEIGHT = 360
FPS = 15
LOOP_BUDGET_MS = 1000.0 / FPS
COLOR_SPACE = "BGR"

GROUND_CALIB_PATH = "vision_stack/calibration/ground_calib.json"


# =============================================================================
# L1: Ground Projection
# =============================================================================
@dataclass
class GroundCalibration:
    """
    Pinhole ground-plane projection for a forward-facing camera whose optical
    axis is parallel to the floor.

    For a point on the ground plane at forward distance y and lateral offset x:

        v - v0 = f * h / y          ->      y = A / (v - v0),  A = f * h
        u - u0 = f * x / y          ->      x = (u - u0) * y / f

    so the whole projection is three numbers: the horizon row v0, the product
    A = f*h, and the focal length f in pixels. Both are recoverable from the
    two-position calibration capture; see solve_from_two_positions().

    v0: image row of the horizon, SOURCE FRAME coordinates. Rows at or above
        this project to infinity and are rejected.
    a_cm_px: A = f_px * camera_height_cm. Units are cm*px.
    f_px: focal length in pixels.
    u0: image column of the optical centre, source frame coordinates.
    min_range_cm: nearest observable ground distance. This is an OCCLUSION
        limit (chassis in frame), not an optical one, so it is a validity gate
        rather than a projection parameter. Segments nearer than this are
        discarded -- whatever produced them is not the floor ahead.
    max_range_cm: far cutoff. Beyond this one pixel of row error is worth
        several centimetres and the estimate is not usable.
    calibrated: False when these are the shipped placeholders. Logged loudly.
    roll_deg: residual image rotation left over AFTER the videoflip, degrees,
        positive = image rotated clockwise. An inverted mount is corrected by
        rotate-180 in the GStreamer pipeline, but that only cancels a mount
        that is inverted EXACTLY. Any physical roll in the bracket survives the
        flip and shows up here. See estimate_roll_deg().
    """
    v0:           float = 190.0
    a_cm_px:      float = 1600.0
    f_px:         float = 400.0
    u0:           float = FRAME_WIDTH / 2.0
    min_range_cm: float = 22.0
    max_range_cm: float = 90.0
    calibrated:   bool = False
    roll_deg:     float = 0.0

    def _derotate(self, u, v):
        """
        Purpose:
            Undo residual camera roll before projection. Works elementwise, so
            it takes scalars or numpy arrays.

        Notes:
            Rotation is taken about (u0, v0). The principal point and the
            horizon row coincide only when the optical axis is exactly level,
            which is the assumption the whole projection model already makes --
            this introduces no new one.

            Correcting roll by subtracting a constant from angle_deg would be
            wrong. Roll also displaces lateral position, and the displacement
            grows with distance from the principal point, so a segment far to
            one side is shifted more than one near the centre. Rotating the
            pixels first handles position and angle together.
        """
        if abs(self.roll_deg) < 1e-9:
            return u, v
        t = math.radians(-self.roll_deg)
        cos_t, sin_t = math.cos(t), math.sin(t)
        du, dv = u - self.u0, v - self.v0
        return (self.u0 + cos_t * du - sin_t * dv,
                self.v0 + sin_t * du + cos_t * dv)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def solve_from_two_positions(
            cls,
            v_near: float, y_near_cm: float, width_px_near: float,
            v_far: float, y_far_cm: float, width_px_far: float,
            feature_width_cm: float,
            u0: float = FRAME_WIDTH / 2.0,
            min_range_cm: float = 22.0,
            max_range_cm: float = 90.0,
        ) -> "GroundCalibration":
        """
        Purpose:
            Solve v0, A and f_px from the two-position capture.

        Inputs:
            v_near / v_far: image row of the feature at each position
            y_near_cm / y_far_cm: measured ground distance to the feature
            width_px_near / width_px_far: measured pixel width of the feature
            feature_width_cm: true width of that feature (e.g. 1.0 for a
                dividing line, 14.0 for a lane, 29.5 for a full street)

        Outputs:
            GroundCalibration with calibrated=True

        Notes:
            v0 falls out of the two range measurements:
                y1 = A/(v1-v0), y2 = A/(v2-v0)
                => v0 = (y2*v2 - y1*v1) / (y2 - y1)
            f_px falls out of either width measurement:
                width_px = f * feature_width_cm / y
            Both positions give an f estimate; they are averaged, and a large
            disagreement means the optical axis is not level or one of the
            distance measurements is wrong. It is checked, not assumed.
        """
        if abs(y_far_cm - y_near_cm) < 1e-6:
            raise ValueError("solve_from_two_positions: the two ranges are identical")

        v0 = (y_far_cm * v_far - y_near_cm * v_near) / (y_far_cm - y_near_cm)
        a_cm_px = y_near_cm * (v_near - v0)

        if a_cm_px <= 0.0:
            raise ValueError(
                f"solve_from_two_positions: non-physical A={a_cm_px:.1f}. "
                "Check that the NEAR row is below the FAR row in image "
                "coordinates (v increases downward)."
            )

        f_near = width_px_near * y_near_cm / max(feature_width_cm, 1e-6)
        f_far = width_px_far * y_far_cm / max(feature_width_cm, 1e-6)
        f_px = 0.5 * (f_near + f_far)

        disagreement = abs(f_near - f_far) / max(f_px, 1e-6)
        if disagreement > 0.15:
            log.warning(
                "Focal estimates disagree by %.0f%% (near=%.0f px, far=%.0f px). "
                "Optical axis may not be level, or a range measurement is off.",
                disagreement * 100.0, f_near, f_far,
            )

        # Consistency check: A should equal f * camera height. With the mount
        # at 4 cm, an implied height far from that means the axis is pitched.
        implied_h = a_cm_px / max(f_px, 1e-6)
        log.info(
            "Calibration solved: v0=%.1f  A=%.0f  f=%.0f px  implied mount "
            "height=%.1f cm", v0, a_cm_px, f_px, implied_h,
        )

        return cls(
            v0=v0, a_cm_px=a_cm_px, f_px=f_px, u0=u0,
            min_range_cm=min_range_cm, max_range_cm=max_range_cm,
            calibrated=True,
        )

    @classmethod
    def placeholder_for_roi(
            cls,
            roi_top_row: int,
            frame_height: int = FRAME_HEIGHT,
            min_range_cm: float = 22.0,
            max_range_cm: float = 90.0,
            camera_height_cm: float = 4.0,
        ) -> "GroundCalibration":
        """
        Purpose:
            Build placeholder optics that are at least CONSISTENT WITH THE ROI,
            by forcing the bottom ROI row to land at min_range and the top row
            at max_range.

        Notes:
            The earlier hard-coded placeholders (v0=190, A=1600) were internally
            sensible as optics and wrong for the ROI they get applied to: with a
            lane ROI starting at row 252, they mapped the bottom of the frame to
            9.5 cm, so 89% of ROI rows fell under the 22 cm near gate and were
            discarded as unprojectable. Everything that survived landed in a
            4 cm deep slice, which makes every contour wide-and-shallow in
            ground space and therefore TRANSVERSE. That is how an uncalibrated
            run produces a hundred phantom stop lines and almost no lane lines.

            These are still fake numbers. They are only shaped so an
            uncalibrated run fails visibly rather than plausibly.
        """
        bot = frame_height - 1
        top = roi_top_row
        if bot <= top:
            return cls()
        a_cm_px = (bot - top) / (1.0 / min_range_cm - 1.0 / max_range_cm)
        v0 = bot - a_cm_px / min_range_cm
        return cls(
            v0=v0, a_cm_px=a_cm_px, f_px=a_cm_px / max(camera_height_cm, 1e-6),
            min_range_cm=min_range_cm, max_range_cm=max_range_cm,
            calibrated=False,
        )

    def describe_rows(self, lane_rect: Tuple[int, int, int, int]) -> dict:
        """
        Purpose:
            Report what depth band this ROI actually maps to, and how much of
            it survives the range gate.

        Inputs:
            lane_rect: (x, y, w, h) of the lane ROI in the source frame

        Outputs:
            dict with the mapping and a 'usable_frac' in [0,1]

        Notes:
            Run this before reading any other number. If usable_frac is small,
            the pipeline is not looking at the road -- it is looking at a thin
            slice of it, and every downstream statistic is drawn from that
            slice rather than from the scene.
        """
        top = lane_rect[1]
        bot = lane_rect[1] + lane_rect[3] - 1
        rows = range(top, bot + 1)

        inside = [v for v in rows if v > self.v0
                  and self.min_range_cm <= self.a_cm_px / (v - self.v0) <= self.max_range_cm]

        def _depth(v):
            return self.a_cm_px / (v - self.v0) if v > self.v0 else float("inf")

        info = {
            "top_row": top, "bottom_row": bot, "rows": len(list(rows)),
            "depth_at_top_cm": _depth(top), "depth_at_bottom_cm": _depth(bot),
            "usable_rows": len(inside),
            "usable_frac": len(inside) / max(len(list(rows)), 1),
        }

        log.info(
            "ROI rows %d-%d map to %.1f-%.1f cm; %d of %d rows (%.0f%%) fall "
            "inside the [%.0f, %.0f] cm gate",
            info["top_row"], info["bottom_row"],
            info["depth_at_bottom_cm"], info["depth_at_top_cm"],
            info["usable_rows"], info["rows"], 100.0 * info["usable_frac"],
            self.min_range_cm, self.max_range_cm,
        )
        if info["usable_frac"] < 0.60:
            log.error(
                "Only %.0f%% of the lane ROI is projectable. Contours will be "
                "squeezed into a %.1f cm deep slice, which makes almost "
                "everything classify as TRANSVERSE. Fix the calibration before "
                "reading any result below.",
                100.0 * info["usable_frac"],
                abs(info["depth_at_top_cm"] - info["depth_at_bottom_cm"]),
            )
        return info

    @classmethod
    def load(cls, path: str) -> "GroundCalibration":
        """
        Purpose:
            Load calibration JSON, falling back to placeholders with a warning.
        """
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            calib = cls(**data)
            log.info("Ground calibration loaded from %s", path)
            return calib
        except FileNotFoundError:
            log.warning(
                "No ground calibration at %s -- using PLACEHOLDER optics. "
                "Every centimetre value below is a guess.", path,
            )
            return cls.placeholder_for_roi(
                roi_top_row=int(FRAME_HEIGHT * 0.70))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.error("Ground calibration at %s is malformed (%s) -- using "
                      "placeholders.", path, exc)
            return cls()

    def save(self, path: str) -> None:
        """
        Purpose:
            Write calibration JSON alongside the HSV range files.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        log.info("Ground calibration written to %s", path)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------
    def project(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Purpose:
            Project one SOURCE-FRAME pixel onto the ground plane.

        Inputs:
            u: column in source frame pixels
            v: row in source frame pixels

        Outputs:
            (x_cm, y_cm) in the robot frame, or None if the pixel is at or
            above the horizon, or outside the usable range band.
        """
        u, v = self._derotate(u, v)

        dv = v - self.v0
        if dv <= 1e-6:
            return None                      # at or above the horizon
        y_cm = self.a_cm_px / dv
        if y_cm < self.min_range_cm or y_cm > self.max_range_cm:
            return None
        x_cm = (u - self.u0) * y_cm / self.f_px
        return x_cm, y_cm

    def project_array(self, uv: np.ndarray) -> np.ndarray:
        """
        Purpose:
            Vectorised project() for a whole contour at once.

        Inputs:
            uv: (N, 2) float array of SOURCE-FRAME (u, v) pixel coordinates

        Outputs:
            (M, 2) float32 array of ground (x_cm, y_cm), M <= N. Points at or
            above the horizon and points outside the range band are dropped.

        Notes:
            Per-point Python projection costs roughly 1.5 ms/frame across a
            typical contour set, against a 66.7 ms budget that already gets
            exceeded. This keeps L1 off the critical path.
        """
        u = uv[:, 0].astype(np.float64)
        v = uv[:, 1].astype(np.float64)
        u, v = self._derotate(u, v)

        dv = v - self.v0
        ok = dv > 1e-6
        y = np.full_like(dv, np.inf)
        np.divide(self.a_cm_px, dv, out=y, where=ok)

        ok &= (y >= self.min_range_cm) & (y <= self.max_range_cm)
        if not np.any(ok):
            return np.empty((0, 2), dtype=np.float32)

        x = (u[ok] - self.u0) * y[ok] / self.f_px
        return np.stack([x, y[ok]], axis=1).astype(np.float32)

    def cm_per_px_lateral(self, v: float) -> float:
        """
        Purpose:
            Lateral scale at one image row. Useful for sanity printouts and for
            sizing pixel-domain thresholds from centimetre requirements.
        """
        dv = max(v - self.v0, 1e-6)
        return (self.a_cm_px / dv) / self.f_px


@dataclass
class GroundSegment:
    """
    One lane-marking candidate expressed on the ground plane.

    All lengths are centimetres in the robot frame (+x right, +y forward).

    p0: (x, y) of the near endpoint of the fitted centreline
    p1: (x, y) of the far endpoint
    length_cm: centreline length
    width_cm: perpendicular width. The physical discriminator: a dividing line
        is ~1 cm, a lane edge line ~1 cm, a mat seam or curb is not.
    angle_deg: signed angle from the forward axis. 0 = parallel to travel,
        positive = leaning right as it recedes. +/-90 = fully transverse.
    mid: (x, y) midpoint
    kind: "longitudinal" | "transverse" | "oblique" | "rejected"
    reject_reason: populated when kind == "rejected"; the field name that did it
    quality: [0,1] geometric plausibility, NOT the geometry.py contour score
    frame_id: frame this was observed in
    """
    p0:            Tuple[float, float]
    p1:            Tuple[float, float]
    length_cm:     float
    width_cm:      float
    angle_deg:     float
    mid:           Tuple[float, float]
    kind:          str = "oblique"
    reject_reason: str = ""
    quality:       float = 0.0
    frame_id:      int = 0

    def x_at(self, y_cm: float) -> Optional[float]:
        """
        Purpose:
            Lateral position of this segment at a given forward distance,
            extrapolating along the fitted direction.

        Notes:
            This is how the lookahead lateral estimate is taken. Evaluating the
            segment at a fixed y is what makes two segments of different
            lengths comparable -- the old estimator compared bbox centroids,
            which are a function of how much of each line survived the ROI.
        """
        (x0, y0), (x1, y1) = self.p0, self.p1
        dy = y1 - y0
        if abs(dy) < 1e-6:
            return None                      # transverse: no unique x at y
        t = (y_cm - y0) / dy
        return x0 + t * (x1 - x0)


def segment_from_contour(
        contour: np.ndarray,
        lane_rect: Tuple[int, int, int, int],
        calib: GroundCalibration,
        frame_id: int = 0,
    ) -> Optional[GroundSegment]:
    """
    Purpose:
        Fit a ground-plane segment to one geometry.py lane contour.

    Inputs:
        contour: ROI-local contour, shape (N,1,2), from LaneCandidate.contour
        lane_rect: (x, y, w, h) of the lane ROI within the source frame
        calib: GroundCalibration
        frame_id: carried through for tracing

    Outputs:
        GroundSegment, or None if too few points survive projection.

    Notes:
        Consumes geometry.py output DIRECTLY rather than going through
        feature_fusion. DetectionObject keeps only a centroid and a bbox, and
        neither is enough to fit a direction or measure a perpendicular width.
        Re-projection out of ROI-local coordinates is done here instead, which
        is the same correction feature_fusion applies (R7).

        THE RECTANGLE IS FITTED IN GROUND SPACE, NOT IMAGE SPACE. Fitting
        minAreaRect on the pixel contour and projecting its sides afterwards
        looks equivalent and is not: a line spanning depth images as a
        trapezoid converging toward the vanishing point, so the enclosing
        pixel rectangle is much wider than the line. On a synthetic 1.0 cm
        line spanning 25-45 cm that error measured 4.29 cm -- enough to fail
        the width gate that this whole design leans on. Projecting first makes
        the shape an actual rectangle before anything is measured.
    """
    if contour is None or len(contour) < 5:
        return None

    roi_x, roi_y = lane_rect[0], lane_rect[1]
    uv = contour.reshape(-1, 2).astype(np.float64)
    uv[:, 0] += roi_x
    uv[:, 1] += roi_y

    ground = calib.project_array(uv)
    if len(ground) < 5:
        return None                          # too little of it is on the floor

    box = cv2.boxPoints(cv2.minAreaRect(ground))   # 4x2, in centimetres

    def _mid(a, b):
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    side_a = float(np.linalg.norm(box[1] - box[0]))
    side_b = float(np.linalg.norm(box[2] - box[1]))

    if side_a >= side_b:
        # box[0]-box[1] is the long side; short sides are 1-2 and 3-0
        end_1, end_2 = _mid(box[1], box[2]), _mid(box[3], box[0])
        length_cm, width_cm = side_a, side_b
    else:
        end_1, end_2 = _mid(box[0], box[1]), _mid(box[2], box[3])
        length_cm, width_cm = side_b, side_a

    # Order endpoints near-to-far so the angle sign is well defined
    p0, p1 = (end_1, end_2) if end_1[1] <= end_2[1] else (end_2, end_1)

    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    angle_deg = math.degrees(math.atan2(dx, dy)) if length_cm > 1e-6 else 0.0

    return GroundSegment(
        p0=p0, p1=p1,
        length_cm=length_cm,
        width_cm=width_cm,
        angle_deg=angle_deg,
        mid=((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5),
        frame_id=frame_id,
    )


# =============================================================================
# L2: Segment Classification
# =============================================================================
def estimate_roll_from_transverse(contour: np.ndarray) -> float:
    """
    Purpose:
        Measure residual camera roll from a physically straight reference laid
        ACROSS the field of view, with the robot parked square to it.

    Inputs:
        contour: the reference's contour in pixel coordinates. ROI-local or
            source-frame both work; only the direction is used.

    Outputs:
        roll in degrees, positive = image rotated clockwise. Write it straight
        into GroundCalibration.roll_deg.

    Notes:
        MEASURE ROLL ON A TRANSVERSE REFERENCE, NOT A LANE LINE. Roll barely
        changes the ground-projected ANGLE of a longitudinal segment -- a
        synthetic 3 deg roll moved it by under 0.3 deg -- because roll mostly
        perturbs the row, and row maps to depth, and depth then rescales the
        lateral coordinate. So it surfaces as a LATERAL bias, asymmetric
        between the left and right boundaries, which reads as a lane-centre
        offset on a robot that is genuinely centred. Median lane-line angle is
        blind to it.

        A transverse reference has the leverage instead: it should lie along a
        constant image row, it spans most of the frame width, and its tilt in
        the raw image IS the roll. A metre rule or a taped straight edge laid
        across the lane works; so does a stop line, if you can park square to
        one.

        Direction comes from boxPoints rather than minAreaRect's angle field,
        because that field's convention changed in OpenCV 4.5 and this has to
        agree with segment_from_contour either way.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))

    side_a = box[1] - box[0]
    side_b = box[2] - box[1]
    d = side_a if np.linalg.norm(side_a) >= np.linalg.norm(side_b) else side_b

    ang = math.degrees(math.atan2(float(d[1]), float(d[0])))
    while ang > 90.0:
        ang -= 180.0
    while ang < -90.0:
        ang += 180.0
    return ang


def bias_report(
        longitudinal: List["GroundSegment"],
        offsets: List[float],
    ) -> None:
    """
    Purpose:
        Print the two bias checks to run on a capture where the robot is parked
        square and centred in a lane. Both numbers must be zero; each nonzero
        one implicates something different.

    Inputs:
        longitudinal: every longitudinal segment accumulated over the capture
        offsets: the per-frame offset_norm values from two_boundary frames only

    Notes:
        A robot that drifts the same way across repeated runs is showing a
        constant perception bias, not a gain problem. No amount of PD tuning
        removes a constant added to the measurement.
    """
    angles = sorted(s.angle_deg for s in longitudinal)
    if angles:
        mid = len(angles) // 2
        med_ang = angles[mid] if len(angles) % 2 else \
            0.5 * (angles[mid - 1] + angles[mid])
        print(f"\nMedian longitudinal angle: {med_ang:+.2f} deg "
              f"({len(angles)} segments)")
        print("  Nonzero on a square capture => heading bias: the robot is not "
              "parked as\n  square as assumed, or the mount is yawed. Roll "
              "barely moves this number.")

    if offsets:
        offs = sorted(offsets)
        mid = len(offs) // 2
        med_off = offs[mid] if len(offs) % 2 else \
            0.5 * (offs[mid - 1] + offs[mid])
        print(f"\nMedian two-boundary offset: {med_off:+.3f} "
              f"({len(offs)} frames)")
        print("  Nonzero on a centred capture => lateral bias: camera roll, or "
              "u0 not at the\n  true optical centre. Measure roll on a "
              "transverse reference with\n  estimate_roll_from_transverse(), "
              "set roll_deg, and re-run before touching u0.")


@dataclass
class CourseGeometry:
    """
    Physical constants of the arena, in centimetres. Measured, not tuned.

    lane_width_cm: centre of dividing line to lane edge
    line_width_cm: painted/taped line width
    street_width_cm: full two-lane street
    intersection_cm: side length of the intersection square
    """
    lane_width_cm:    float = 14.0
    line_width_cm:    float = 1.0
    street_width_cm:  float = 29.5
    intersection_cm:  float = 29.0


@dataclass
class SegmentFilter:
    """
    Acceptance band for a ground segment, all in physical units.

    Unlike LaneContourFilter these thresholds do not need re-tuning when the
    ROI, the resolution, or the mount changes -- they describe the course.

    ORIENTATION MATTERS TO WHICH GATE APPLIES. For a longitudinal segment the
    fitted rectangle's short side is the marking's lateral width, which is the
    best feature the pipeline has. For a TRANSVERSE segment the short side is
    its depth extent, and depth is compressed hard by perspective: a 1 cm stop
    line at 40 cm range occupies about one pixel row, so its measured width is
    at the resolution limit and reads near zero. Gating a stop line on width
    rejects every one of them and silently disables the intersection logic.
    Transverse segments are gated on LENGTH -- their lateral extent, which is
    measured in the well-resolved direction.

    width_min_cm / width_max_cm: plausible marking width, LONGITUDINAL ONLY.
        Nominal line is 1.0 cm; the band allows for blur bleed and projection
        error at range. Puzzle-mat seams and the curb fail here.
    length_min_cm: below this a longitudinal segment has no reliable direction
    transverse_min_length_cm / transverse_max_length_cm: a stop line spans
        roughly a lane to a street; shorter is debris, longer is the arena edge
    transverse_max_width_cm: loose upper bound only, to reject depth-extended
        blobs. There is deliberately no lower bound.
    longitudinal_max_deg: |angle| under this counts as along-lane
    transverse_min_deg: |angle| over this counts as a stop line
        Between the two is "oblique": real, tracked, but not used as either a
        lane reference or an intersection cue this frame.
    """
    width_min_cm:              float = 0.4
    width_max_cm:              float = 3.0
    length_min_cm:             float = 2.5
    transverse_min_length_cm:  float = 8.0
    transverse_max_length_cm:  float = 40.0
    transverse_max_width_cm:   float = 5.0
    longitudinal_max_deg:      float = 35.0
    transverse_min_deg:        float = 60.0


def classify_segment(seg: GroundSegment, f: SegmentFilter) -> GroundSegment:
    """
    Purpose:
        Assign kind and quality to a ground segment. Mutates and returns it.

    Notes:
        Orientation is decided FIRST, then the gate appropriate to that
        orientation is applied. Applying one width band to everything is what
        would have thrown away the stop lines; see the SegmentFilter docstring.

        Rejection is recorded by field name in reject_reason rather than by
        dropping the segment silently. geometry.py already learned this lesson
        with reject_counts: a frame that finds nothing must be distinguishable
        from a frame that found ten and threw them all away for one reason.
    """
    a = abs(seg.angle_deg)

    # ---- transverse: gate on lateral extent, not on depth -------------------
    if a >= f.transverse_min_deg:
        if seg.length_cm < f.transverse_min_length_cm:
            seg.kind, seg.reject_reason = "rejected", "length_cm"
            return seg
        if seg.length_cm > f.transverse_max_length_cm:
            seg.kind, seg.reject_reason = "rejected", "length_cm"
            return seg
        if seg.width_cm > f.transverse_max_width_cm:
            seg.kind, seg.reject_reason = "rejected", "width_cm"
            return seg
        seg.kind = "transverse"
        # Quality from how close the span is to a lane crossing. Depth extent
        # is not usable here, so it contributes nothing.
        span_term = min(1.0, seg.length_cm / 14.0)
        seg.quality = round(span_term, 4)
        return seg

    # ---- longitudinal and oblique: width is the discriminator ---------------
    if seg.width_cm < f.width_min_cm or seg.width_cm > f.width_max_cm:
        seg.kind, seg.reject_reason = "rejected", "width_cm"
        return seg
    if seg.length_cm < f.length_min_cm:
        seg.kind, seg.reject_reason = "rejected", "length_cm"
        return seg

    seg.kind = "longitudinal" if a <= f.longitudinal_max_deg else "oblique"

    # Quality: how close the width is to nominal, scaled by how much line we
    # actually got. Both terms are physical, so this number means the same
    # thing at 25 cm and at 70 cm -- which the pixel-area score never did.
    nominal = 0.5 * (f.width_min_cm + f.width_max_cm)
    width_term = max(0.0, 1.0 - abs(seg.width_cm - nominal) / max(nominal, 1e-6))
    length_term = min(1.0, seg.length_cm / 12.0)
    seg.quality = round(0.6 * width_term + 0.4 * length_term, 4)
    return seg


# =============================================================================
# L3: Odometry and the Ground Map
# =============================================================================
class Odometry:
    """
    Tracks incremental motion between frames so the ground map can be
    propagated after a feature leaves the field of view.

    Encoder distance is preferred. The N20 motors have encoders wired back to
    the Pi but run_pipeline._read_sensors() still passes wheel_speed=None, so
    the fallback integrates COMMANDED speed instead. That is open-loop and will
    drift under wheel slip -- source is reported on every update so a trace can
    tell the two apart rather than silently trusting the estimate.
    """

    def __init__(self, speed_cm_per_s_at_full: float = 40.0):
        self._speed_full = speed_cm_per_s_at_full
        self.total_cm = 0.0
        self.total_yaw_deg = 0.0
        self.last_ds_cm = 0.0
        self.last_dyaw_deg = 0.0
        self.source = "none"

    def update(
            self,
            dt_s: float,
            encoder_distance_m: Optional[float],
            yaw_rate_dps: Optional[float],
            commanded_speed: float,
        ) -> Tuple[float, float]:
        """
        Purpose:
            Advance odometry by one frame.

        Inputs:
            dt_s: seconds since the previous frame
            encoder_distance_m: cumulative distance from SensorSample, or None
            yaw_rate_dps: mean yaw rate this frame window, or None
            commanded_speed: the base speed actually sent to the wheels, [0,1]

        Outputs:
            (ds_cm, dyaw_deg) for this frame
        """
        if encoder_distance_m is not None:
            ds_cm = max(0.0, encoder_distance_m * 100.0 - self.total_cm)
            self.source = "encoder"
        else:
            ds_cm = max(0.0, commanded_speed) * self._speed_full * dt_s
            self.source = "commanded"

        dyaw = (yaw_rate_dps or 0.0) * dt_s

        self.total_cm += ds_cm
        self.total_yaw_deg += dyaw
        self.last_ds_cm, self.last_dyaw_deg = ds_cm, dyaw
        return ds_cm, dyaw


class GroundMap:
    """
    Short-horizon memory of ground segments in the CURRENT robot frame.

    Every stored segment is rigidly transformed on each update so it stays in
    the live robot frame. This is what lets the state machine keep acting on a
    stop line for the 22 cm during which it is invisible.

    Segments are dropped once they fall behind the robot or age out.
    """

    def __init__(self, max_age_frames: int = 30, behind_margin_cm: float = 10.0):
        self._segments: List[GroundSegment] = []
        self._ages: List[int] = []
        self._max_age = max_age_frames
        self._behind = behind_margin_cm

    def propagate(self, ds_cm: float, dyaw_deg: float) -> None:
        """
        Purpose:
            Move every stored segment into the new robot frame.

        Notes:
            The robot advanced ds along +y and turned dyaw to the right. A
            point fixed in the world therefore moves back by ds and rotates by
            +dyaw in robot coordinates:

                x' =  cos(t)*x - sin(t)*(y - ds)
                y' =  sin(t)*x + cos(t)*(y - ds)

            Check: turn right 90 deg with ds=0, a point 10 cm straight ahead
            becomes (-10, 0) -- 10 cm to the left. Correct.
        """
        t = math.radians(dyaw_deg)
        c, s = math.cos(t), math.sin(t)

        def _move(p):
            x, y = p[0], p[1] - ds_cm
            return (c * x - s * y, s * x + c * y)

        kept_segs, kept_ages = [], []
        for seg, age in zip(self._segments, self._ages):
            seg.p0 = _move(seg.p0)
            seg.p1 = _move(seg.p1)
            seg.mid = ((seg.p0[0] + seg.p1[0]) * 0.5,
                       (seg.p0[1] + seg.p1[1]) * 0.5)
            age += 1
            if age <= self._max_age and seg.mid[1] > -self._behind:
                kept_segs.append(seg)
                kept_ages.append(age)

        self._segments, self._ages = kept_segs, kept_ages

    def add(self, segments: List[GroundSegment]) -> None:
        """
        Purpose:
            Insert this frame's accepted segments at age 0.
        """
        for seg in segments:
            self._segments.append(seg)
            self._ages.append(0)

    def of_kind(self, kind: str, max_age: Optional[int] = None) -> List[GroundSegment]:
        """
        Purpose:
            All tracked segments of one kind, optionally age-limited.
        """
        out = []
        for seg, age in zip(self._segments, self._ages):
            if seg.kind == kind and (max_age is None or age <= max_age):
                out.append(seg)
        return out

    def nearest_transverse_distance(
            self, ahead_only: bool = True
        ) -> Optional[float]:
        """
        Purpose:
            Forward distance to the closest transverse segment AHEAD.

        Inputs:
            ahead_only: when True (the default) segments the robot has already
                driven past are excluded.

        Notes:
            ahead_only exists because of a real double-trigger. The map retains
            a crossed stop line until it is behind_margin_cm behind, so an
            unfiltered minimum returns something like -9 cm for several frames
            after an intersection. Both "is a stop line near" (-9 <= 55) and
            "have we reached it" (-9 <= 0) are then trivially true, so the
            machine re-armed and re-entered six frames after exiting, with no
            transverse segment detected in any of those frames.

            Pass ahead_only=False only for diagnostics.
        """
        cands = [s.mid[1] for s in self.of_kind("transverse")
                 if not ahead_only or s.mid[1] > 0.0]
        return min(cands) if cands else None

    def __len__(self) -> int:
        return len(self._segments)


# =============================================================================
# L4: Scene State Machine
# =============================================================================
STATE_LANE_FOLLOW = "lane_follow"
STATE_APPROACHING_STOP = "approaching_stop"
STATE_IN_INTERSECTION = "in_intersection"
STATE_EXITING = "exiting"
STATE_LOST = "lost"


@dataclass
class FSMConfig:
    """
    Transition thresholds. Every trigger is evidence-based or distance-based;
    none is a frame count or a timeout.

    vote_window / vote_needed: N-of-M hysteresis on scene evidence. A single
        frame never moves the machine, because a single frame is exactly what
        a dashed-line gap, a blur smear, or a reflection looks like.
    stop_line_arm_cm: a transverse segment further than this is noted but does
        not arm the approach. Arming early means braking for the far kerb.
    lost_after_frames: consecutive frames with no longitudinal reference before
        declaring LOST. Must exceed the dash gap, or the machine trips between
        every pair of dashes.
    exit_search_frac: fraction of the intersection to cross before looking for
        the outbound lane lines again.
    """
    vote_window:       int = 5
    vote_needed:       int = 3
    stop_line_arm_cm:  float = 55.0
    lost_after_frames: int = 12
    exit_search_frac:  float = 0.6


@dataclass
class SceneEstimate:
    """
    L4 output. This is the object a navigation consumer should read.

    state: one of the STATE_* constants
    offset_norm: lateral offset, canonical sign, normalised to one lane
        half-width. Valid only when offset_valid.
    offset_valid: False when this frame produced no lateral measurement.
        Consumers MUST check this; pd_control holds and decays on False rather
        than re-applying a proportional term against a frozen error.
    heading_error_deg: signed, canonical sign convention
    heading_valid: False when no longitudinal reference was available
    lane_mode: "two_boundary" | "left_only" | "right_only" | "none"
    stop_line_cm: distance to the tracked stop line, or None
    intersection_progress_cm: distance travelled since entering, or None
    longitudinal_count / transverse_count: accepted segments this frame
    drive_state: "go" | "caution" | "stop"
    frame_id / timestamp_ms
    """
    state:                    str
    offset_norm:              float
    offset_valid:             bool
    heading_error_deg:        float
    heading_valid:            bool
    lane_mode:                str
    stop_line_cm:             Optional[float]
    intersection_progress_cm: Optional[float]
    longitudinal_count:       int
    transverse_count:         int
    drive_state:              str
    frame_id:                 int
    timestamp_ms:             int


class SceneStateMachine:
    """
    Consumes the ground map, emits a SceneEstimate. Knows nothing about pixels,
    motors, or PWM.
    """

    def __init__(
            self,
            course: CourseGeometry,
            cfg: FSMConfig,
            lookahead_cm: float = 30.0,
        ):
        self._course = course
        self._cfg = cfg
        self._lookahead = lookahead_cm

        self.state = STATE_LANE_FOLLOW
        self._transverse_votes: Deque[bool] = deque(maxlen=cfg.vote_window)
        self._longitudinal_votes: Deque[bool] = deque(maxlen=cfg.vote_window)
        self._no_lane_frames = 0

        self._entry_odom_cm: Optional[float] = None    # odometer at intersection entry
        self._armed_stop_cm: Optional[float] = None   # range to the armed line
        self._armed_odom_cm: Optional[float] = None   # odometer when it was armed

    # ------------------------------------------------------------------
    # Lateral estimate
    # ------------------------------------------------------------------
    def _lateral_from(self, longitudinal: List[GroundSegment]) -> Tuple[float, bool, str]:
        """
        Purpose:
            Reduce longitudinal segments to one signed normalised offset.

        Outputs:
            (offset_norm, valid, mode)

        Notes:
            Each segment is evaluated at the SAME lookahead distance before
            comparison. The old estimator compared bbox centroids, so a dashed
            line fragmenting into three contours voted three times and pulled
            the "centre" toward itself. Here a fragmented line contributes one
            lateral reading per fragment but all at the same y, so they agree
            with each other instead of biasing the mean.

            Left and right are defined relative to the ROBOT, not the image,
            and a boundary is assigned by the sign of its lateral position.
        """
        half = self._course.lane_width_cm / 2.0

        readings = []
        for seg in longitudinal:
            x = seg.x_at(self._lookahead)
            if x is not None and abs(x) < self._course.street_width_cm:
                readings.append((x, seg.quality))

        if not readings:
            return 0.0, False, "none"

        left = [r for r in readings if r[0] < 0.0]
        right = [r for r in readings if r[0] >= 0.0]

        def _weighted(rs):
            w = sum(q for _, q in rs) or 1.0
            return sum(x * q for x, q in rs) / w

        if left and right:
            lane_centre = 0.5 * (_weighted(left) + _weighted(right))
            mode = "two_boundary"
        elif left:
            lane_centre = _weighted(left) + half
            mode = "left_only"
        else:
            lane_centre = _weighted(right) - half
            mode = "right_only"

        # Canonical sign: lane centre to the robot's right => robot is left of
        # centre => positive => steer right.
        offset = max(-1.0, min(1.0, lane_centre / max(half, 1e-6)))
        return offset, True, mode

    def lateral_estimate(
            self, longitudinal: List[GroundSegment]
        ) -> Tuple[float, bool, str]:
        """
        Purpose:
            Public entry to the lateral reduction, for harnesses that want the
            estimate without running the state machine. Same contract as
            _lateral_from: (offset_norm, valid, mode).
        """
        return self._lateral_from(longitudinal)

    @staticmethod
    def _heading_from(longitudinal: List[GroundSegment]) -> Tuple[float, bool]:
        """
        Purpose:
            Quality-weighted mean lane direction, in degrees, canonical sign.
        """
        usable = [s for s in longitudinal if s.length_cm > 4.0]
        if not usable:
            return 0.0, False
        w = sum(s.quality for s in usable) or 1.0
        return sum(s.angle_deg * s.quality for s in usable) / w, True

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def update(
            self,
            gmap: GroundMap,
            odom_cm: float,
            frame_id: int,
            timestamp_ms: int,
        ) -> SceneEstimate:
        """
        Purpose:
            Advance the machine one frame and emit the scene estimate.

        Inputs:
            gmap: the odometry-propagated ground map, already updated
            odom_cm: cumulative distance travelled
            frame_id / timestamp_ms: from the capture loop
        """
        cfg = self._cfg
        longitudinal = gmap.of_kind("longitudinal", max_age=2)
        transverse = gmap.of_kind("transverse", max_age=2)

        offset, offset_valid, mode = self._lateral_from(longitudinal)
        heading, heading_valid = self._heading_from(longitudinal)

        self._longitudinal_votes.append(offset_valid)
        stop_cm = gmap.nearest_transverse_distance()
        self._transverse_votes.append(
            stop_cm is not None and stop_cm <= cfg.stop_line_arm_cm
        )

        self._no_lane_frames = 0 if offset_valid else self._no_lane_frames + 1

        have_stop = sum(self._transverse_votes) >= cfg.vote_needed
        have_lane = sum(self._longitudinal_votes) >= cfg.vote_needed
        progress = None

        # ---- transitions -------------------------------------------------
        if self.state == STATE_LANE_FOLLOW:
            if have_stop:
                self._armed_stop_cm = stop_cm
                self._armed_odom_cm = odom_cm
                self.state = STATE_APPROACHING_STOP
                log.info("FSM: LANE_FOLLOW -> APPROACHING_STOP at %.1f cm", stop_cm or -1)
            elif self._no_lane_frames >= cfg.lost_after_frames:
                self.state = STATE_LOST
                log.warning("FSM: LANE_FOLLOW -> LOST (%d frames without a lane)",
                            self._no_lane_frames)

        elif self.state == STATE_APPROACHING_STOP:
            # Distance trigger on the ARMED line, propagated by odometry. The
            # line is invisible for the last 22 cm, so re-querying the map here
            # is wrong twice over: it loses the target inside the blind zone,
            # and it can latch onto a different segment than the one that armed
            # the approach.
            travelled = odom_cm - (self._armed_odom_cm or odom_cm)
            remaining = (self._armed_stop_cm or 0.0) - travelled

            # A fresh measurement of a line still ahead refines the estimate;
            # anything behind us is ignored by nearest_transverse_distance.
            if stop_cm is not None:
                remaining = stop_cm
                self._armed_stop_cm, self._armed_odom_cm = stop_cm, odom_cm

            if remaining <= 0.0:
                self._entry_odom_cm = odom_cm
                self._armed_stop_cm = self._armed_odom_cm = None
                self.state = STATE_IN_INTERSECTION
                log.info("FSM: APPROACHING_STOP -> IN_INTERSECTION")
            elif not have_stop and stop_cm is None and remaining > cfg.stop_line_arm_cm:
                self._armed_stop_cm = self._armed_odom_cm = None
                self.state = STATE_LANE_FOLLOW
                log.info("FSM: APPROACHING_STOP -> LANE_FOLLOW (evidence lapsed)")

        elif self.state == STATE_IN_INTERSECTION:
            progress = odom_cm - (self._entry_odom_cm or odom_cm)
            if progress >= self._course.intersection_cm * cfg.exit_search_frac:
                self.state = STATE_EXITING
                log.info("FSM: IN_INTERSECTION -> EXITING at %.1f cm", progress)

        elif self.state == STATE_EXITING:
            progress = odom_cm - (self._entry_odom_cm or odom_cm)
            if have_lane:
                self.state = STATE_LANE_FOLLOW
                self._entry_odom_cm = None
                self._transverse_votes.clear()
                log.info("FSM: EXITING -> LANE_FOLLOW (lane reacquired at %.1f cm)",
                         progress)
            elif progress > self._course.intersection_cm * 2.0:
                self.state = STATE_LOST
                log.warning("FSM: EXITING -> LOST (no lane %.1f cm past entry)", progress)

        elif self.state == STATE_LOST:
            if have_lane:
                self.state = STATE_LANE_FOLLOW
                log.info("FSM: LOST -> LANE_FOLLOW (lane reacquired)")

        # ---- per-state output policy -------------------------------------
        if self.state == STATE_LANE_FOLLOW:
            drive = "go"
        elif self.state == STATE_APPROACHING_STOP:
            drive = "caution"
        elif self.state in (STATE_IN_INTERSECTION, STATE_EXITING):
            # No lateral reference is EXPECTED here, so a missing measurement
            # is not an error. Marking it invalid puts pd_control on its hold
            # and decay path, which is the correct behaviour for crossing.
            drive = "caution"
            offset_valid = False
        else:
            drive = "stop"
            offset_valid = False

        return SceneEstimate(
            state=self.state,
            offset_norm=round(offset, 4),
            offset_valid=offset_valid,
            heading_error_deg=round(heading, 2),
            heading_valid=heading_valid,
            lane_mode=mode,
            stop_line_cm=round(stop_cm, 1) if stop_cm is not None else None,
            intersection_progress_cm=round(progress, 1) if progress is not None else None,
            longitudinal_count=len(longitudinal),
            transverse_count=len(transverse),
            drive_state=drive,
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
        )


# =============================================================================
# L5: Control
# =============================================================================
@dataclass
class ControlConfig:
    """
    heading_gain: how much of the heading error folds into the PD error term,
        in normalised-offset units per degree. Offset alone is a position
        controller and oscillates; the heading term is the damping that a
        pure-pursuit formulation gets from its geometry.
    heading_clip_deg: heading contribution saturates here
    caution_speed_scale: base speed multiplier while approaching or crossing
    kp / kd: see the note below on units
    """
    heading_gain:        float = 0.012
    heading_clip_deg:    float = 30.0
    caution_speed_scale: float = 0.6
    kp:                  float = 0.14     # 0.40 * 0.35, see note
    kd:                  float = 0.018    # 0.05 * 0.35
    base_speed:          float = 0.45
    ramp_seconds:        float = 0.75


# NOTE ON GAIN UNITS
#   run_pipeline.py feeds pd_control a value in EstimationPacket.lane_offset,
#   which is scaled metres. This file emits offset_norm directly. Per the units
#   block in contracts.py, preserving loop gain across that change means
#   scaling KP/KD by lane_half_width_px / px_per_meter = 240/685.7 = 0.35.
#   The defaults above are the current tuned gains times that factor. They are
#   a starting point, not a validated tune -- bench them before a full run.


def control_error(est: SceneEstimate, cfg: ControlConfig) -> float:
    """
    Purpose:
        Blend lateral offset and heading error into the single scalar
        pd_control consumes.
    """
    heading = max(-cfg.heading_clip_deg, min(cfg.heading_clip_deg,
                                             est.heading_error_deg))
    heading_term = cfg.heading_gain * heading if est.heading_valid else 0.0
    return est.offset_norm + heading_term


# =============================================================================
# Per-Frame Assembly
# =============================================================================
@dataclass
class StageConfigs:
    """
    Everything the per-frame function needs that does not change between
    frames. Bundled so replay and live share one construction path.
    """
    calib:        GroundCalibration
    course:       CourseGeometry
    seg_filter:   SegmentFilter
    canny:        CannyParams = field(default_factory=CannyParams)
    lane_filter:  LaneContourFilter = field(default_factory=LaneContourFilter)
    sign_filter:  SignContourFilter = field(default_factory=SignContourFilter)


def segments_for_frame(
        lane_roi: np.ndarray,
        sign_roi: np.ndarray,
        lane_rect: Tuple[int, int, int, int],
        cfgs: StageConfigs,
        frame_id: int,
        timestamp_ms: int,
    ) -> Tuple[List[GroundSegment], dict]:
    """
    Purpose:
        L1 + L2 for one frame: contours -> classified ground segments.

    Outputs:
        (accepted segments, trace dict)

    Notes:
        Every contour geometry.py produced is projected and classified. The
        pixel-domain filter is deliberately left wide here -- physical width is
        a far better discriminator than contour area, and running both means
        tuning two filters that disagree.
    """
    geo_result, _lane_dbg, _sign_dbg = run_geometry_branch(
        lane_roi=lane_roi,
        sign_roi=sign_roi,
        canny_params=cfgs.canny,
        lane_filter=cfgs.lane_filter,
        sign_filter=cfgs.sign_filter,
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        draw_overlays=False,
    )

    trace = {"contours": len(geo_result.lane_candidates),
             "unprojectable": 0, "width_cm": 0, "length_cm": 0,
             "longitudinal": 0, "transverse": 0, "oblique": 0}

    accepted: List[GroundSegment] = []
    for cand in geo_result.lane_candidates:
        seg = segment_from_contour(cand.contour, lane_rect, cfgs.calib, frame_id)
        if seg is None:
            trace["unprojectable"] += 1
            continue
        seg = classify_segment(seg, cfgs.seg_filter)
        if seg.kind == "rejected":
            trace[seg.reject_reason] += 1
            continue
        trace[seg.kind] += 1
        accepted.append(seg)

    return accepted, trace


# =============================================================================
# Bird's-Eye Debug Render
# =============================================================================
def render_ground_map(
        gmap: GroundMap,
        est: SceneEstimate,
        course: CourseGeometry,
        size_px: int = 420,
        span_cm: float = 100.0,
    ) -> np.ndarray:
    """
    Purpose:
        Top-down view of the ground map. This is the single most useful debug
        artifact for this design: a projection error that is invisible in the
        image overlay is obvious the moment lines stop being parallel here.

    Inputs:
        gmap: current ground map
        est: this frame's scene estimate
        course: for drawing the expected lane band
        size_px: output is size_px square
        span_cm: forward extent shown; lateral extent is the same

    Outputs:
        BGR image
    """
    img = np.full((size_px, size_px, 3), 24, dtype=np.uint8)
    scale = size_px / span_cm
    origin = (size_px // 2, size_px - 20)          # robot at bottom centre

    def _px(pt):
        return (int(origin[0] + pt[0] * scale),
                int(origin[1] - pt[1] * scale))

    # Expected lane band and lookahead ring
    half = course.lane_width_cm / 2.0
    for x in (-half, half):
        cv2.line(img, _px((x, 0)), _px((x, span_cm)), (55, 55, 55), 1)
    cv2.circle(img, origin, int(30.0 * scale), (45, 45, 45), 1)
    cv2.circle(img, origin, int(22.0 * scale), (40, 40, 70), 1)   # blind radius

    colors = {"longitudinal": (0, 220, 0), "transverse": (0, 160, 255),
              "oblique": (120, 120, 120)}
    for seg in gmap.of_kind("longitudinal") + gmap.of_kind("transverse") + \
               gmap.of_kind("oblique"):
        cv2.line(img, _px(seg.p0), _px(seg.p1),
                 colors.get(seg.kind, (90, 90, 90)), 2)

    cv2.drawMarker(img, origin, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 12, 2)

    hud = f"{est.state}  off={est.offset_norm:+.2f} {'OK' if est.offset_valid else 'HOLD'}"
    cv2.putText(img, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (230, 230, 230), 1, cv2.LINE_AA)
    if est.stop_line_cm is not None:
        cv2.putText(img, f"stop @ {est.stop_line_cm:+.0f} cm", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 255), 1, cv2.LINE_AA)
    return img


# =============================================================================
# Replay: run the stack over a captured frame directory, no hardware
# =============================================================================
def replay(sample_dirs: List[str], write_maps: bool = True) -> None:
    """
    Purpose:
        Drive the full L1-L4 stack from saved frames so the state machine can
        be validated before it ever touches the robot.

        Reads the same *_roi_lane.png / *_roi_sign.png fixtures geometry.py's
        harness uses, so run roi_crop.py first. Odometry is synthesised from a
        constant assumed speed, which is enough to exercise the distance
        triggers but is NOT a substitute for a hardware run.

    Outputs per frame -> <sample_dir>/results/:
        stem_map.png: bird's-eye ground map
    """
    calib = GroundCalibration.load(GROUND_CALIB_PATH)
    if not calib.calibrated:
        log.warning("Replaying with placeholder optics -- treat centimetre "
                    "values as relative, not absolute.")

    cfgs = StageConfigs(
        calib=calib,
        course=CourseGeometry(),
        seg_filter=SegmentFilter(),
    )
    fsm = SceneStateMachine(cfgs.course, FSMConfig())
    gmap = GroundMap()
    odom = Odometry()

    dt_s = 1.0 / FPS
    frame_id = 0
    totals = {}
    all_longitudinal: List[GroundSegment] = []
    two_boundary_offsets: List[float] = []

    for sample_dir in sample_dirs:
        results_dir = os.path.join(sample_dir, "results")
        if not os.path.isdir(results_dir):
            print(f"[SKIP] No results dir: {results_dir} -- run roi_crop.py first")
            continue

        lane_files, sign_files = {}, {}
        for f in os.listdir(results_dir):
            if f.endswith("_roi_lane.png"):
                lane_files[f[: -len("_roi_lane.png")]] = os.path.join(results_dir, f)
            elif f.endswith("_roi_sign.png"):
                sign_files[f[: -len("_roi_sign.png")]] = os.path.join(results_dir, f)

        stems = sorted(set(lane_files) & set(sign_files))
        if not stems:
            print(f"[SKIP] No ROI fixture pairs in {results_dir}")
            continue

        for stem in stems:
            lane_roi = cv2.imread(lane_files[stem])
            sign_roi = cv2.imread(sign_files[stem])
            if lane_roi is None or sign_roi is None:
                print(f"[FAIL] Unreadable ROI pair: {stem}")
                continue

            # roi_crop places the lane ROI at (0, LANE_Y_TOP_FRAC*H). Recover
            # the offset from the fixture height rather than re-importing the
            # constant, so a re-crop cannot silently desync the projection.
            lane_rect = (0, FRAME_HEIGHT - lane_roi.shape[0],
                         lane_roi.shape[1], lane_roi.shape[0])
            if frame_id == 0:
                calib.describe_rows(lane_rect)

            ts_ms = int(time.time() * 1000)
            segs, trace = segments_for_frame(
                lane_roi, sign_roi, lane_rect, cfgs, frame_id, ts_ms)

            ds_cm, dyaw = odom.update(dt_s, None, None, ControlConfig().base_speed)
            gmap.propagate(ds_cm, dyaw)
            gmap.add(segs)
            est = fsm.update(gmap, odom.total_cm, frame_id, ts_ms)

            for k, v in trace.items():
                totals[k] = totals.get(k, 0) + v
            all_longitudinal.extend(s for s in segs if s.kind == "longitudinal")
            if est.lane_mode == "two_boundary" and est.offset_valid:
                two_boundary_offsets.append(est.offset_norm)

            if write_maps:
                cv2.imwrite(os.path.join(results_dir, f"{stem}_map.png"),
                            render_ground_map(gmap, est, cfgs.course))

            print(
                f"[{frame_id:05d}] {est.state:<17} "
                f"off={est.offset_norm:+.3f}{'' if est.offset_valid else '*'} "
                f"head={est.heading_error_deg:+6.1f} "
                f"mode={est.lane_mode:<13} "
                f"lon={est.longitudinal_count} tra={est.transverse_count} "
                f"stop={est.stop_line_cm if est.stop_line_cm is not None else '--':>6} "
                f"map={len(gmap):02d} drive={est.drive_state}"
            )
            frame_id += 1

    bias_report(all_longitudinal, two_boundary_offsets)

    print("\nSegment trace totals:")
    for k in ("contours", "unprojectable", "width_cm", "length_cm",
              "longitudinal", "transverse", "oblique"):
        print(f"  {k:<15} {totals.get(k, 0)}")


# =============================================================================
# Live: hardware loop
# =============================================================================
def main() -> None:
    """
    Purpose:
        Live run. Mirrors run_pipeline.main() stage for stage, but replaces
        lane_offset + estimation with the L1-L5 stack above.
    """
    import pigpio
    from imu import IMUReader
    from system import System

    log.info("Starting Navilott scene-state pipeline")

    # ---- hardware ----------------------------------------------------------
    pi = pigpio.pi()
    _ain1, _ain2, _pwma = 24, 25, 13
    _bin1, _bin2, _pwmb = 22, 27, 12
    _stby = 23
    for pin in (_ain1, _ain2, _bin1, _bin2, _stby):
        pi.set_mode(pin, pigpio.OUTPUT)

    def _drive(left_speed: float, right_speed: float) -> None:
        pi.write(_stby, 1)
        pi.hardware_PWM(_pwma, 1000,
                        int(max(0.0, min(1.0, abs(left_speed))) * 1000000))
        pi.write(_ain1, 1 if left_speed < 0 else 0)
        pi.write(_ain2, 1 if left_speed > 0 else 0)
        pi.hardware_PWM(_pwmb, 1000,
                        int(max(0.0, min(1.0, abs(right_speed))) * 1000000))
        pi.write(_bin1, 1 if right_speed > 0 else 0)
        pi.write(_bin2, 1 if right_speed < 0 else 0)

    s = System()
    s.wait_for_start()
    s.run_countdown()
    t_run_start = time.perf_counter()

    imu = IMUReader(address=0x68, rate_hz=100.0)
    imu.start()

    # ---- stack -------------------------------------------------------------
    cfgs = StageConfigs(
        calib=GroundCalibration.load(GROUND_CALIB_PATH),
        course=CourseGeometry(),
        seg_filter=SegmentFilter(),
    )
    ctrl_cfg = ControlConfig()
    fsm = SceneStateMachine(cfgs.course, FSMConfig())
    gmap = GroundMap()
    odom = Odometry()
    pd = PDController(PDConfig(
        kp=ctrl_cfg.kp, kd=ctrl_cfg.kd,
        base_speed=ctrl_cfg.base_speed,
        ramp_seconds=ctrl_cfg.ramp_seconds,
    ))

    # ---- camera ------------------------------------------------------------
    gst = (
        "libcamerasrc ! "
        f"video/x-raw,format=BGR,width={FRAME_WIDTH},height={FRAME_HEIGHT},"
        f"framerate={FPS}/1 ! "
        "videoconvert ! videoflip method=rotate-180 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        log.error("Failed to open camera pipeline")
        sys.exit(1)

    frame_id = 0
    t_prev = time.perf_counter()
    commanded_speed = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            dt_s = max(t0 - t_prev, 1e-3)
            t_prev = t0

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                log.warning("Frame %d: read failed", frame_id)
                frame_id += 1
                continue

            ts_ms = int(t0 * 1000.0)

            # ---- Phase 1/2: capture, preprocess, crop, segments ------------
            frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
            pre_bgr = cv2.cvtColor(preprocess_frame(frame_yuv), cv2.COLOR_YUV2BGR)
            roi = crop_rois(pre_bgr, frame_id=frame_id)

            segs, trace = segments_for_frame(
                roi.lane_roi.copy(), roi.sign_roi.copy(), roi.lane_rect,
                cfgs, frame_id, ts_ms,
            )

            # ---- L3: odometry and map --------------------------------------
            imu_frame = imu.snapshot()
            ds_cm, dyaw = odom.update(
                dt_s=dt_s,
                encoder_distance_m=None,          # TODO: wire N20 encoders
                yaw_rate_dps=imu_frame.mean_yaw_rate_dps if imu_frame.valid else None,
                commanded_speed=commanded_speed,
            )
            gmap.propagate(ds_cm, dyaw)
            gmap.add(segs)

            # ---- L4: scene state -------------------------------------------
            est = fsm.update(gmap, odom.total_cm, frame_id, ts_ms)

            # ---- L5: control -----------------------------------------------
            if est.drive_state == "stop":
                left, right = pd.stop()
                commanded_speed = 0.0
            else:
                left, right = pd.update(
                    offset=control_error(est, ctrl_cfg),
                    offset_valid=est.offset_valid,
                    now_s=t0,
                )
                if est.drive_state == "caution":
                    left *= ctrl_cfg.caution_speed_scale
                    right *= ctrl_cfg.caution_speed_scale
                commanded_speed = 0.5 * (abs(left) + abs(right))
            _drive(left, right)

            # ---- trace ------------------------------------------------------
            frame_ms = (time.perf_counter() - t0) * 1000.0
            s.update_display(time.perf_counter() - t_run_start)
            if frame_ms > LOOP_BUDGET_MS:
                log.warning("Frame %d: budget exceeded %.1f ms", frame_id, frame_ms)

            log.info(
                "f=%04d %.1fms %-17s off=%+.3f %s head=%+6.1f mode=%-13s "
                "lon=%d tra=%d stop=%s map=%02d odom=%.1fcm(%s) drive=%s",
                frame_id, frame_ms, est.state, est.offset_norm,
                "OK  " if est.offset_valid else "HOLD",
                est.heading_error_deg, est.lane_mode,
                est.longitudinal_count, est.transverse_count,
                f"{est.stop_line_cm:+.0f}" if est.stop_line_cm is not None else "--",
                len(gmap), odom.total_cm, odom.source, est.drive_state,
            )
            frame_id += 1

    except KeyboardInterrupt:
        log.info("Stopped by user after %d frames.", frame_id)

    finally:
        _drive(0.0, 0.0)
        pi.write(_stby, 0)
        pi.stop()
        imu.stop()
        cap.release()
        elapsed = time.perf_counter() - t_run_start
        s.show_final_time(elapsed)
        time.sleep(5.0)
        s.cleanup()
        log.info("Scene-state pipeline shutdown complete.")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    """
    Standalone modes:

        python state_pipeline.py replay        # offline, no hardware
        python state_pipeline.py live          # on the robot

    Replay is the mode to start in. It needs roi_crop.py to have run first, and
    it writes a bird's-eye map per frame next to the existing debug images.
    """
    SAMPLE_DIRS = [
        "vision_stack/frames/Sample1",
        "vision_stack/frames/Sample2",
        "vision_stack/frames/Sample3",
    ]

    mode = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if mode == "live":
        main()
    else:
        replay(SAMPLE_DIRS)