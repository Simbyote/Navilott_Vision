"""
phase3.py

Navigation Signal Processing

Purpose:
    Receives Phase2Output from the vision pipeline and produces an EstimationPacket
    for the Navigation subsystem

Pipeline stages
  1. Motion consistency check  — suppress candidates whose bbox centroid jumped
                                  farther than a speed-scaled threshold between frames
  2. Temporal filtering        — EMA smoothing of lane_offset and heading_error
                                  over a configurable window; majority vote on
                                  discrete states (drive_state, stop_sign_detected)
  3. Confidence thresholding   — reject any detection whose smoothed confidence
                                  is below the configured floor
  4. State estimation          — fuse vision outputs with inertial sensor readings
                                  to produce final validated navigation signals

Inputs:
  phase2_out: Phase2Output
      .detections        list[DetectionObject]
          Each element has .type, .label, .position_x, .position_y,
          .confidence, .timestamp
          Types: "traffic_light" | "stop_sign" | "lane_boundary"
          For "traffic_light": .label carries "red" | "yellow" | "green"
          For "lane_boundary": .position_x is the signed pixel offset of the
                               lane center from the image center column.
                               Phase 3 converts to meters via px_per_meter.
      .frame_id          int
      .timestamp_ms      int

  sensor_sample: SensorSample
      Odometry and IMU readings captured during the same frame window.
      All fields are optional (None = sensor not yet available or not fitted).
      .wheel_speed        float | None   m/s
      .distance_traveled  float | None   m (cumulative since reset)
      .yaw_rate           float | None   deg/s  (mean over frame window)
      .lateral_accel      float | None   m/s²   (peak over frame window)

Outputs:
  EstimationPacket:
    .lane_offset          float   Lateral distance from lane center (m).
                                    Positive = robot is right of center.
                                    Vision-primary; encoder dead-reckoning fallback.
    .heading_error        float   Angular deviation from target heading (deg).
                                    Vision-primary; IMU yaw integration fallback.
    .drive_state          str     "go" | "caution" | "stop"
    .stop_sign_detected   bool
    .yaw_rate             float   Pass-through from SensorSample (0.0 if None).
    .lateral_accel        float   Pass-through from SensorSample (0.0 if None).
    .wheel_speed          float   Pass-through from SensorSample (0.0 if None).
    .frame_id             int
    .timestamp_ms         int
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


# ============================================================================
# Detection type from Phase 2 output
# ============================================================================

@dataclass
class DetectionObject:
    """
    Minimal mirror of the Phase 2 DetectionObject contract.

    .type: "traffic_light" | "stop_sign" | "lane_boundary"
    .label: (For traffic_light): "red" | "yellow" | "green"
            (For stop_sign and lane_boundary): "stop_sign" and "lane_boundary"
    .position_x: (For lane_boundary): signed pixel offset of lane center from image center column
                 (For traffic_light and stop_sign): bbox centroid x
    .position_y: bbox centroid y (pixels); used for motion consistency check
    .confidence: confidence level from [0.0, 1.0]
    .timestamp: frame timestamp

    For lane_boundary detections, position_x is the signed pixel offset of
    the lane center from the image center column
    For traffic_light detections, label carries the color string
    """
    type:       str
    label:      str
    position_x: float
    position_y: float
    confidence: float
    timestamp:  int

@dataclass
class Phase2Output:
    """
    Minimal Phase 2 output contract consumed by Phase 3.
    
    .detections: Detections list from Phase 2
    .frame_id: Frame identifier from capture loop
    .timestamp_ms: Timestamp from capture loop
    """
    detections:   List[DetectionObject]
    frame_id:     int
    timestamp_ms: int

# ============================================================================
# Sensor Input
# ============================================================================

@dataclass
class SensorSample:
    """
    Odometry and IMU readings for one frame window
    All fields may be None if the sensor is not fitted

    .wheel_speed: speed of the robot
    .distance_traveled: cumulative distance traveled since last reset
    .yaw_rate: rate of change of heading in degrees per second
    .lateral_accel: peak lateral acceleration in m/s² during the frame window
    """
    wheel_speed:       Optional[float] = None
    distance_traveled: Optional[float] = None
    yaw_rate:          Optional[float] = None
    lateral_accel:     Optional[float] = None

# ============================================================================
# Phase 3 Output
# ============================================================================

@dataclass
class EstimationPacket:
    """
    Validated navigation signals produced by Phase 3
    This is the handoff contract to the Navigation subsystem

    .lane_offset: Lateral distance from lane center
                Positive = robot is right of center
                Vision-primary; encoder dead-reckoning fallback
    .heading_error: Angular deviation from target heading (deg)
                    Vision-primary; IMU yaw integration fallback
    .drive_state: "go" | "caution" | "stop"
    .stop_sign_detected: detecting a stop sign takes precedence over traffic light state
    .intersection_active: True while Phase 3's debounced intersection
                    state machine considers the robot inside an
                    intersection opening — lane_offset/heading_error are
                    on dead-reckoning/IMU fallback rather than fresh
                    vision detections for the whole time this is True
    .yaw_rate: Pass-through from SensorSample (0.0 if None)
    .lateral_accel: Pass-through from SensorSample (0.0 if None)
    .wheel_speed: Pass-through from SensorSample (0.0 if None)
    .frame_id: frame identifier from capture loop
    .timestamp_ms: timestamp from capture loop
    """
    lane_offset:        float
    heading_error:      float
    drive_state:        str
    stop_sign_detected: bool
    intersection_active: bool
    yaw_rate:           float
    lateral_accel:      float
    wheel_speed:        float
    frame_id:           int
    timestamp_ms:       int


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Phase3Config:
    """
    All tuning parameters for Phase 3

    1) Temporal Filtering:
        ema_alpha: EMA smoothing factor for lane_offset and heading_error
                    Higher = faster response, less smoothing
                    Range [0.1, 0.9]; start at 0.35 for 18 FPS
        vote_window: Number of frames over which majority vote is taken
                    Used for discrete states

    2) Confidence Thresholding:
        min_confidence_lane: Minimum confidence to accept a lane boundary detection
        min_confidence_traffic: Minimum confidence to accept a traffic light detection
        min_confidence_sign: Minimum confidence to accept a stop sign detection

    3) Lane offset:
        px_per_meter: Pixel-to-meter conversion for lane_offset
                At 480x360 with ~35 cm visible lane half-width,
                a starting estimate is (480/2) / 0.35 ≈ 686 px/m

    4) Motion consistency:
        max_centroid_jump_px: Maximum allowed centroid displacement between
                consecutive frames for the same detection class

    5) Dead-reckoning:
        deadreck_max_frames: Maximum frames to hold last lane_offset before
                the estimate is considered stale

    6) Intersection handling:
        intersection_enabled: hard on/off switch for the whole
                intersection state machine, independent of
                INTERSECTION_EDGE_RATIO_THRESH's value. Unlike
                min_confidence_* (bounded [0.0, 1.0], so 1.1 is a
                provable disable), edge_ratio in config.py is
                open-ended — an arbitrarily high threshold is NOT a
                guaranteed off switch for it, only a less-likely one.
                This flag is the actual off switch. Defaults to False
                (disabled) until INTERSECTION_EDGE_RATIO_THRESH is
                calibrated against real intersection-crossing data —
                matches how traffic/sign detection is already gated
                off via their own confidence thresholds pending
                calibration
        intersection_enter_frames: consecutive raw intersection triggers
                (from run_pipeline.py's intersection_edge_ratio() check)
                required before Phase 3 commits to INTERSECTION mode —
                debounces a single noisy edge-ratio spike into a real
                crossing. TODO: REQUIRES CALIBRATION against actual
                course speed/FPS dwell time
        intersection_exit_frames: consecutive non-triggers required
                before Phase 3 leaves INTERSECTION mode.
                TODO: REQUIRES CALIBRATION
        intersection_max_frames: hard cap on how long INTERSECTION mode
                can stay active regardless of the raw trigger, so a
                stuck-on edge-ratio reading can't wedge the robot into
                dead-reckoning indefinitely. TODO: REQUIRES CALIBRATION
    """
    ema_alpha:              float = 0.35
    vote_window:            int   = 3

    min_confidence_lane:    float = 0.30
    min_confidence_traffic: float = 0.40
    min_confidence_sign:    float = 0.45

    px_per_meter:           float = 686.0

    max_centroid_jump_px:   float = 80.0

    deadreck_max_frames:    int   = 10

    intersection_enabled:      bool = False
    intersection_enter_frames: int  = 2
    intersection_exit_frames:  int  = 2
    intersection_max_frames:   int  = 45


# ============================================================================
# Internal Filter State
# ============================================================================

@dataclass
class _EMAState:
    """
    Exponential moving average for a single scalar

    value: current EMA value; None if uninitialized
    """
    value: Optional[float] = None

    def update(
            self, 
            sample: float, 
            alpha: float
        ) -> float:
        """
        Purpose:
            Update EMA with new sample

        Inputs:
            sample: new measurement to incorporate
            alpha: smoothing factor in [0.0, 1.0]

        Output:
            Updated EMA value
        """
        if self.value is None:
            self.value = sample
        else:
            self.value = alpha * sample + (1.0 - alpha) * self.value
        return self.value

@dataclass
class _CentroidTracker:
    """
    Tracks last known centroid per detection class for consistency check
    
    last_x: last known x position
    last_y: last known y position
    """
    last_x: Optional[float] = None
    last_y: Optional[float] = None

    def update(
            self, 
            x: float, 
            y: float, 
            threshold: float
        ) -> bool:
        """
        Purpose:
            Check if centroid is within threshold of last known position

        Inputs:
            x: current x position
            y: current y position
            threshold: maximum allowed jump in pixels

        Output:
            True if within threshold; False to break
        lock the tracker

        NOTE: unconditionally commits (x, y) as the new reference, even
        when returning False. Safe ONLY for a detection class that never
        has more than one candidate per frame (traffic_light, stop_sign
        — fuse_detections() already reduces those to a single best
        candidate upstream). For a class that can have several
        simultaneous candidates per frame (lane_boundary), calling this
        once per candidate chains each comparison off the previous
        candidate's position instead of the prior frame's, and commits
        the reference even on rejection — see select_best() below for
        the version that avoids both.
        """
        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return True
        if self.last_y is None:
            self.last_x, self.last_y = x, y
            return True
        dx = x - self.last_x
        dy = y - self.last_y
        dist = (dx * dx + dy * dy) ** 0.5
        self.last_x, self.last_y = x, y
        return dist <= threshold

    def select_best(
            self,
            candidates: list,
            threshold: float,
        ):
        """
        Purpose:
            Pick (at most) one DetectionObject out of several simultaneous
            same-frame candidates for this tracker's side, and commit to
            it as the new reference. Unlike update(), every candidate is
            checked against the SAME held-fixed prior-frame reference —
            never against another candidate from this frame — and the
            reference only moves when a candidate actually qualifies, so
            a rejected outlier can never smuggle itself in as next
            frame's reference point.

        Inputs:
            candidates: DetectionObject list for this frame, already
                restricted to this tracker's side (e.g. all left-of-center
                lane_boundary detections). May be empty.
            threshold: maximum allowed centroid displacement in px from
                the last accepted position

        Output:
            the accepted DetectionObject (highest-confidence one that's
            within threshold of the prior reference, or the
            highest-confidence candidate outright if this tracker has no
            history yet), or None if candidates is empty or none qualify
        """
        if not candidates:
            return None

        if self.last_x is None or self.last_y is None:
            best = max(candidates, key=lambda d: d.confidence)
            self.last_x, self.last_y = best.position_x, best.position_y
            return best

        qualifying = [
            d for d in candidates
            if ((d.position_x - self.last_x) ** 2
                + (d.position_y - self.last_y) ** 2) ** 0.5 <= threshold
        ]
        if not qualifying:
            return None

        best = max(qualifying, key=lambda d: d.confidence)
        self.last_x, self.last_y = best.position_x, best.position_y
        return best


# ============================================================================
# Phase3Processor
# ============================================================================

class Phase3Processor:
    """
    Instantiate once at pipeline startup; call process() on every frame
    @TODO: Threading support if processing time becomes an issue later on 
    """
    def __init__(
            self, 
            config: Optional[Phase3Config] = None
        ):
        self._cfg = config or Phase3Config()

        # EMA states
        self._lane_offset_ema   = _EMAState()
        self._heading_error_ema = _EMAState()

        # Majority vote buffers
        self._drive_state_buf: Deque[str]  = deque(maxlen=self._cfg.vote_window)
        self._stop_sign_buf:   Deque[bool] = deque(maxlen=self._cfg.vote_window)

        # Centroid trackers (one per detection class)
        # lane_boundary gets TWO trackers (left/right of image center),
        # since fuse_detections() can hand it several simultaneous
        # candidates per frame — unlike traffic_light/stop_sign, which
        # are already reduced to a single best candidate upstream and
        # so are safe with one shared tracker each
        self._lane_tracker_left:  _CentroidTracker = _CentroidTracker()
        self._lane_tracker_right: _CentroidTracker = _CentroidTracker()
        self._traffic_tracker:    _CentroidTracker = _CentroidTracker()
        self._sign_tracker:       _CentroidTracker = _CentroidTracker()

        # Dead-reckoning state
        self._last_timestamp_ms:  int   = 0
        self._deadreck_frames:    int   = 0
        self._last_lane_offset_m: float = 0.0

        # IMU heading integration state
        self._heading_from_imu: float = 0.0

        # Intersection state machine (debounced from the raw per-frame
        # intersection_edge_ratio() trigger — see _update_intersection_state)
        self._intersection_active:        bool = False
        self._intersection_enter_streak:  int  = 0
        self._intersection_exit_streak:   int  = 0
        self._intersection_frames_active: int  = 0

    # ===========================================================================
    # Entry Point
    # ===========================================================================
    def process(
        self,
        phase2_out:    Phase2Output,
        sensor_sample: SensorSample,
        intersection_trigger: bool = False,
    ) -> EstimationPacket:
        """
        Purpose:
            Run one Phase 3 cycle

        Input:
            phase2_out: Phase2Output from the vision pipeline
            sensor_sample: SensorSample for this frame window
            intersection_trigger: this frame's raw intersection_edge_ratio()
                > INTERSECTION_EDGE_RATIO_THRESH result from run_pipeline.py.
                Debounced internally into a persistent INTERSECTION mode —
                see _update_intersection_state(). Defaults to False so
                existing callers (and the standalone test block below)
                keep working unchanged if they don't pass it.

        Output:
            EstimationPacket ready for Navigation
        """
        cfg = self._cfg

        # dt for integration
        dt = 0.0
        if self._last_timestamp_ms > 0 and phase2_out.timestamp_ms > 0:
            dt = (phase2_out.timestamp_ms - self._last_timestamp_ms) / 1000.0
            dt = max(0.0, min(dt, 0.5))   # clamp: ignore stale/jumped timestamps
        self._last_timestamp_ms = phase2_out.timestamp_ms

        # Compute jump threshold
        jump_thresh = cfg.max_centroid_jump_px
        if sensor_sample.wheel_speed is not None and dt > 0.0:
            speed_scale = 1.0 + sensor_sample.wheel_speed * dt * cfg.px_per_meter
            jump_thresh = cfg.max_centroid_jump_px * min(speed_scale, 3.0)

        # Intersection state machine — update BEFORE filtering so this
        # frame's lane/heading estimate already reflects whichever mode
        # we're in
        intersection_active = self._update_intersection_state(intersection_trigger)

        # Stage 1: Motion consistency check
        consistent = self._motion_consistency(phase2_out.detections, jump_thresh)

        # Stages 2 & 3: Temporal filtering & confidence threshold
        lane_offset_m, lane_confident = self._filter_lane(consistent, dt, sensor_sample,
                                                            intersection_active)
        heading_error_deg             = self._filter_heading(consistent, dt,
                                                              sensor_sample, lane_confident)
        raw_drive_state               = self._classify_traffic(consistent)
        raw_stop_sign                 = self._classify_stop_sign(consistent)

        # Majority vote on discrete states
        self._drive_state_buf.append(raw_drive_state)
        self._stop_sign_buf.append(raw_stop_sign)

        drive_state        = _majority_vote_str(self._drive_state_buf, default="go")
        stop_sign_detected = _majority_vote_bool(self._stop_sign_buf)

        # Assemble Packet
        return EstimationPacket(
            lane_offset = round(lane_offset_m, 4),
            heading_error = round(heading_error_deg, 3),
            drive_state = drive_state,
            stop_sign_detected = stop_sign_detected,
            intersection_active = intersection_active,
            yaw_rate = sensor_sample.yaw_rate      or 0.0,
            lateral_accel = sensor_sample.lateral_accel or 0.0,
            wheel_speed = sensor_sample.wheel_speed   or 0.0,
            frame_id = phase2_out.frame_id,
            timestamp_ms = phase2_out.timestamp_ms,
        )

    # ===========================================================================
    # Intersection State Machine
    # ===========================================================================
    def _update_intersection_state(
            self,
            raw_trigger: bool,
        ) -> bool:
        """
        Purpose:
            Debounce the raw per-frame intersection_edge_ratio() trigger
            into a persistent INTERSECTION mode with hysteresis, so a
            single noisy frame can't flip behavior and a real crossing
            doesn't need to re-trigger every frame to stay active.

            On the exit transition (mode was active, now leaving), the
            lane_boundary side-trackers are reset to "no history" so the
            first fresh candidates seen after the intersection are
            accepted outright rather than gated against a now-stale
            pre-intersection position — otherwise reacquiring the real
            lane lines right as vision becomes trustworthy again would
            hit the exact motion-consistency rejection this was meant to
            avoid.

        Inputs:
            raw_trigger: this frame's intersection_edge_ratio() >
                INTERSECTION_EDGE_RATIO_THRESH result

        Output:
            True if INTERSECTION mode is active this frame (after
            applying this frame's update), False otherwise
        """
        cfg = self._cfg

        if not cfg.intersection_enabled:
            # Hard disable — independent of raw_trigger and of whatever
            # INTERSECTION_EDGE_RATIO_THRESH happens to be set to.
            # edge_ratio is open-ended (unlike the [0.0, 1.0]-bounded
            # confidence fields), so no threshold value can be trusted
            # as a guaranteed off switch for it; this flag is. Also
            # clears any state a prior enabled run left behind, so
            # toggling this off mid-development can't leave a stale
            # partial streak/active-frame count around.
            self._intersection_active = False
            self._intersection_enter_streak = 0
            self._intersection_exit_streak = 0
            self._intersection_frames_active = 0
            return False

        if not self._intersection_active:
            self._intersection_enter_streak = (
                self._intersection_enter_streak + 1 if raw_trigger else 0
            )
            if self._intersection_enter_streak >= cfg.intersection_enter_frames:
                self._intersection_active = True
                self._intersection_exit_streak = 0
                self._intersection_frames_active = 0
        else:
            self._intersection_frames_active += 1
            self._intersection_exit_streak = (
                self._intersection_exit_streak + 1 if not raw_trigger else 0
            )
            if (self._intersection_exit_streak >= cfg.intersection_exit_frames
                    or self._intersection_frames_active >= cfg.intersection_max_frames):
                self._intersection_active = False
                self._intersection_enter_streak = 0
                # Fresh start: don't let post-intersection candidates get
                # gated against a stale pre-intersection reference point
                self._lane_tracker_left  = _CentroidTracker()
                self._lane_tracker_right = _CentroidTracker()

        return self._intersection_active

    # ===========================================================================
    # Stage 1: Motion Consistency Check
    # ===========================================================================

    def _motion_consistency(
        self,
        detections:  List[DetectionObject],
        jump_thresh: float,
    ) -> List[DetectionObject]:
        """
        Purpose:
            Discard detections whose centroid jumped more than jump_thresh pixels
            from their last known position for the same detection class

        Inputs:
            detections: list of DetectionObject from Phase 2
            jump_thresh: maximum allowed centroid jump in pixels

        Output:
            filtered detections; also updates the internal centroid trackers
        """
        result = []

        # lane_boundary: routed through _filter_lane_boundaries() — see
        # that method for why this can't reuse the single-tracker
        # update() path traffic_light/stop_sign use below
        lane_dets = [d for d in detections if d.type == "lane_boundary"]
        result.extend(self._filter_lane_boundaries(lane_dets, jump_thresh))

        for det in detections:
            if det.type == "lane_boundary":
                continue
            elif det.type == "traffic_light":
                ok = self._traffic_tracker.update(
                    det.position_x, det.position_y, jump_thresh)
            elif det.type == "stop_sign":
                ok = self._sign_tracker.update(
                    det.position_x, det.position_y, jump_thresh)
            else:
                ok = True
            if ok:
                result.append(det)
        return result

    def _filter_lane_boundaries(
        self,
        lane_dets:   List[DetectionObject],
        jump_thresh: float,
    ) -> List[DetectionObject]:
        """
        Purpose:
            Gate lane_boundary detections against per-side reference
            trackers (left/right of the image center column, split on
            the sign of position_x — the same centered-offset convention
            _adapt_detections_for_p3 already establishes upstream).

            fuse_detections() does NOT reduce lane_boundary to a single
            best candidate the way it does for traffic_light/stop_sign
            — every contour that clears geometry filtering becomes its
            own detection, so a frame can carry several at once (more so
            right at an intersection, where extra edges/curbs/stop-line
            geometry also clear the filter). Routing all of them through
            one shared _CentroidTracker.update() would compare each
            candidate to whichever candidate was checked immediately
            before it IN THE SAME FRAME rather than to the prior frame's
            position, and — since update() commits its reference even on
            rejection — would let a single spurious high-confidence
            candidate silently become the new reference point that
            following frames then measure "consistency" against. Left/
            right trackers plus select_best() (which holds the reference
            fixed across all of this frame's candidates and only commits
            on an actual accept) avoid both problems.

        Inputs:
            lane_dets: this frame's lane_boundary detections, any order
            jump_thresh: maximum allowed centroid jump in pixels

        Output:
            accepted lane_boundary detections — at most one per side,
            each the highest-confidence qualifying candidate on that side
        """
        left_candidates  = [d for d in lane_dets if d.position_x <  0.0]
        right_candidates = [d for d in lane_dets if d.position_x >= 0.0]

        accepted = []
        for tracker, candidates in (
            (self._lane_tracker_left,  left_candidates),
            (self._lane_tracker_right, right_candidates),
        ):
            best = tracker.select_best(candidates, jump_thresh)
            if best is not None:
                accepted.append(best)
        return accepted

    # ===========================================================================
    # Stage 2 & 3: Lane Offset 
    # ===========================================================================
    def _filter_lane(
        self,
        detections:    List[DetectionObject],
        dt:            float,
        sensor_sample: SensorSample,
        intersection_active: bool = False,
    ) -> tuple:
        """
        Purpose:
            Compute lane offset

        Inputs:
            detections: list of DetectionObject from Phase 2
            dt: time delta in seconds between this frame and the previous one
            sensor_sample: SensorSample for this frame
            intersection_active: True when Phase 3's debounced
                intersection state machine considers the robot inside an
                intersection opening. When True, vision candidates for
                this frame are ignored entirely (not just filtered) and
                the estimate coasts on dead-reckoning — extra geometry
                inside an intersection (crossing-street edges, the stop
                line, missing lane markings across the gap) makes
                whatever candidates DO pass motion consistency this
                frame untrustworthy as "lane center", not just noisy

        Output:
            lane_offset_m: lateral offset from lane center in meters
            lane_confident: True if the lane offset is based on a confident vision detection;
                            False if it's a dead-reckoning fallback
        """
        cfg = self._cfg

        if intersection_active:
            # Same hold-last-estimate path as an empty-detections frame,
            # but doesn't bypass deadreck_max_frames — a long intersection
            # can still go stale exactly like an ordinary vision dropout
            if self._deadreck_frames < cfg.deadreck_max_frames:
                self._deadreck_frames += 1
            return self._last_lane_offset_m, False

        lane_dets = [
            d for d in detections
            if d.type == "lane_boundary" and d.confidence >= cfg.min_confidence_lane
        ]

        if lane_dets:
            avg_px   = sum(d.position_x for d in lane_dets) / len(lane_dets)
            offset_m = avg_px / cfg.px_per_meter
            smoothed = self._lane_offset_ema.update(offset_m, cfg.ema_alpha)
            self._last_lane_offset_m = smoothed
            self._deadreck_frames    = 0
            return smoothed, True

        # Dead-reckoning: hold last estimate within the allowed window
        if self._deadreck_frames < cfg.deadreck_max_frames:
            self._deadreck_frames += 1
            return self._last_lane_offset_m, False

        # Stale — hold silently; Navigation must not rely on this value
        return self._last_lane_offset_m, False

    # ===========================================================================
    # Stage 2 & 3: Heading Error
    # ===========================================================================
    def _filter_heading(
        self,
        detections:     List[DetectionObject],
        dt:             float,
        sensor_sample:  SensorSample,
        lane_confident: bool,
    ) -> float:
        """
        Purpose:
            Compute heading error from lane offset and/or IMU integration

        Inputs:
            detections: list of DetectionObject from Phase 2
            dt: time delta in seconds between this frame and the previous one
            sensor_sample: SensorSample for this frame
            lane_confident: True if the lane offset is based on a confident vision detection;
                            False if it's a dead-reckoning fallback

        Output:
            heading_error_deg: heading error in degrees
        """
        HEADING_SCALE = 30.0   # deg per meter of lateral offset; tune empirically

        if lane_confident and self._lane_offset_ema.value is not None:
            raw_heading = self._lane_offset_ema.value * HEADING_SCALE
            smoothed    = self._heading_error_ema.update(raw_heading, self._cfg.ema_alpha)
            self._heading_from_imu = smoothed   # sync accumulator to vision
            return smoothed

        if sensor_sample.yaw_rate is not None and dt > 0.0:
            self._heading_from_imu += sensor_sample.yaw_rate * dt
            self._heading_from_imu  = max(-90.0, min(90.0, self._heading_from_imu))
            return self._heading_from_imu

        return self._heading_error_ema.value or 0.0

    # ===========================================================================
    # Stage 2 & 3: Traffic Light State
    # ===========================================================================
    def _classify_traffic(
            self, 
            detections: List[DetectionObject]
        ) -> str:
        """
        Purpose:
            Classify traffic light state

        Inputs:
            detections: list of DetectionObject from Phase 2

        Output:
            traffic_state: "go", "caution", or "stop"
        """
        cfg = self._cfg
        tl_dets = [
            d for d in detections
            if d.type == "traffic_light" and d.confidence >= cfg.min_confidence_traffic
        ]
        if not tl_dets:
            return "go"

        best = max(tl_dets, key=lambda d: d.confidence)
        if best.label == "red":
            return "stop"
        elif best.label == "yellow":
            return "caution"
        return "go"

    # ===========================================================================
    # Stage 2 & 3: Stop Sign Detection
    # ===========================================================================
    def _classify_stop_sign(
            self, 
            detections: List[DetectionObject]
        ) -> bool:
        """
        Purpose:
            Detect stop sign presence

        Inputs:
            detections: list of DetectionObject from Phase 2

        Output:
            stop_sign_present: True if a stop sign is present with confidence above threshold; False otherwise
        """
        cfg = self._cfg
        return any(
            d for d in detections
            if d.type == "stop_sign" and d.confidence >= cfg.min_confidence_sign
        )

# ============================================================================
# Helper functions for majority voting
# ============================================================================

def _majority_vote_str(
        buf: Deque[str], 
        default: str
    ) -> str:
    """
    Purpose:
        Return the most common string in buf; if tie, return the one that appears first

    Inputs:
        buf: Deque of strings to vote on
        default: String to return if buf is empty

    Output:
        Most common string in buf, or default if buf is empty
    """
    if not buf:
        return default
    counts: dict = {}
    for v in buf:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], list(buf).index(k) if k in buf else 0))

def _majority_vote_bool(buf: Deque[bool]) -> bool:
    """
    Purpose:
        Return True if the majority of values in buf are True; False otherwise
    
    Inputs:
        buf: Deque of bools to vote on

    Output:
        True if the majority of values in buf are True; False otherwise
    """
    if not buf:
        return False
    return sum(1 for v in buf if v) > len(buf) / 2

# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Mock Test:
        Runs the Phase3Processor through a synthetic 15-frame sequence

    Test Cases:
        Frames  0-2 : steady green light + centered lane (normal operation)
        Frames  3-4 : red light appears
        Frame   5   : no detections (dead-reckoning + IMU fallback)
        Frames  6-8 : stop sign detected at course exit
        Frames  9-14: intersection crossing — raw edge-ratio trigger fires
            for 3 consecutive frames (enters INTERSECTION mode after
            intersection_enter_frames=2), spurious wide-jump lane
            candidates appear during the crossing (should be ignored
            entirely, not just motion-consistency-filtered, while
            intersection_active is True), then the trigger clears and a
            fresh, differently-positioned lane pair reappears (should be
            accepted immediately once INTERSECTION mode exits, thanks to
            the tracker reset in _update_intersection_state)
    """

    def _make_det(
        type_, 
        label, 
        px_x, 
        px_y, 
        conf
    ):
        """
        Purpose:
            Mock DetectionObject factory for testing

        Inputs:
            type_: "traffic_light" | "stop_sign" | "lane_boundary"
            label: For traffic_light: "red" | "yellow" | "green"
                   For stop_sign and lane_boundary: "stop_sign" and "lane_boundary"
            px_x: For lane_boundary: signed pixel offset of lane center from image center column
                  For traffic_light and stop_sign: bbox centroid x
            px_y: bbox centroid y (pixels)
            conf: confidence level from [0.0, 1.0]

        Output:
            DetectionObject instance with the specified properties and timestamp=0
        """
        return DetectionObject(
            type=type_, label=label,
            position_x=px_x, position_y=px_y,
            confidence=conf, timestamp=0,
        )

    cfg = Phase3Config(
        ema_alpha=0.35,
        vote_window=3,
        min_confidence_lane=0.30,
        min_confidence_traffic=0.40,
        min_confidence_sign=0.45,
        px_per_meter=686.0,
        deadreck_max_frames=5,
        intersection_enabled=True,   # demo only — real pipeline defaults to False
        intersection_enter_frames=2,
        intersection_exit_frames=2,
        intersection_max_frames=45,
    )
    proc = Phase3Processor(cfg)

    sequences = [
        # (frame_id, ts_ms, detections, sensor_sample, intersection_trigger)
        (0, 0,   [_make_det("traffic_light", "green", 240, 50, 0.85),
                  _make_det("lane_boundary", "lane_boundary", 15, 300, 0.70)],
         SensorSample(wheel_speed=0.3, yaw_rate=0.5), False),
        (1, 55,  [_make_det("traffic_light", "green", 241, 50, 0.88),
                  _make_det("lane_boundary", "lane_boundary", -10, 300, 0.72)],
         SensorSample(wheel_speed=0.3, yaw_rate=0.3), False),
        (2, 110, [_make_det("traffic_light", "green", 242, 50, 0.83),
                  _make_det("lane_boundary", "lane_boundary", 5, 300, 0.68)],
         SensorSample(wheel_speed=0.3, yaw_rate=0.2), False),
        (3, 165, [_make_det("traffic_light", "red", 242, 50, 0.92),
                  _make_det("lane_boundary", "lane_boundary", 5, 300, 0.65)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), False),
        (4, 220, [_make_det("traffic_light", "red", 243, 50, 0.90)],
         SensorSample(wheel_speed=0.1, yaw_rate=1.0), False),
        (5, 275, [],   # no detections — fallback frame
         SensorSample(wheel_speed=0.1, yaw_rate=2.0), False),
        (6, 330, [_make_det("stop_sign", "stop_sign", 380, 200, 0.75)],
         SensorSample(wheel_speed=0.0), False),
        (7, 385, [_make_det("stop_sign", "stop_sign", 381, 200, 0.78)],
         SensorSample(wheel_speed=0.0), False),
        (8, 440, [_make_det("stop_sign", "stop_sign", 382, 200, 0.81),
                  _make_det("traffic_light", "red", 243, 50, 0.88)],
         SensorSample(wheel_speed=0.0), False),
        # --- Intersection crossing ---
        (9, 495, [_make_det("lane_boundary", "lane_boundary", 8, 300, 0.65)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), True),   # trigger #1 — not yet debounced in
        (10, 550, [_make_det("lane_boundary", "lane_boundary", 260, 300, 0.95)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), True),   # trigger #2 — INTERSECTION now active
        (11, 605, [_make_det("lane_boundary", "lane_boundary", 270, 300, 0.97)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), True),   # spurious candidate ignored outright
        (12, 660, [_make_det("lane_boundary", "lane_boundary", 265, 300, 0.96)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), False),  # trigger clears — exit streak #1
        (13, 715, [_make_det("lane_boundary", "lane_boundary", 6, 300, 0.70),
                    _make_det("lane_boundary", "lane_boundary", -145, 300, 0.60)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), False),  # exit streak #2 — mode exits this frame
        (14, 770, [_make_det("lane_boundary", "lane_boundary", 10, 300, 0.72),
                    _make_det("lane_boundary", "lane_boundary", -140, 300, 0.62)],
         SensorSample(wheel_speed=0.2, yaw_rate=0.0), False),  # fresh pair accepted immediately
    ]

    print(f"{'f':>3} {'lane_off':>9} {'head_err':>9} {'drive':>7} {'stop':>5} {'isect':>5}")
    print("-" * 50)

    for frame_id, ts_ms, dets, sensors, isect_trig in sequences:
        p2  = Phase2Output(detections=dets, frame_id=frame_id, timestamp_ms=ts_ms)
        pkt = proc.process(p2, sensors, intersection_trigger=isect_trig)
        print(
            f"{pkt.frame_id:>3} "
            f"{pkt.lane_offset:>+9.4f} "
            f"{pkt.heading_error:>+9.3f} "
            f"{pkt.drive_state:>7} "
            f"{'T' if pkt.stop_sign_detected else 'F':>5} "
            f"{'T' if pkt.intersection_active else 'F':>5}"
        )