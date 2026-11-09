"""Phase 3 debug twin: the production stages, run with a record of each decision and timed per stage.

Purpose:
    phase3_linker's instrumented copy of estimation.py, as run_chain is
    Phase 2's. Each Traced* stage calls its production update(), so the math
    exists once, and test_estimation_debug holds the packets identical.
    pipeline.py never imports it (docs/vision_pipeline/phase3_estimation.md).

Main package:
    TracedPhase3Processor.process(phase2, sensors) -> (packet, debug), with
    each stage's .record and timings_ms per stage.
"""
import time

from src.estimation.estimation import (
    LANE_VISION, USABLE_LANE_MODES, EstimationPacket, HeadingTracker, LaneFilter, lane_mode_of,
    ObstacleClassifier, Phase3Config, Phase3Processor, SensorSample, StopLineClassifier,
    StopSignClassifier, TrafficClassifier,
)
from src.params import STOP_SIGN, TRAFFIC_LIGHT
from src.perception.phase2_out import Phase2Output

# Lane rejection reasons, in the order LaneFilter checks them
NO_RESULT, UNUSABLE_MODE, JUMP_GATE = "no_result", "unusable_mode", "jump_gate"


class TracedLaneFilter(LaneFilter):
    """
    record: raw_offset, mode, accepted, reason (no_result, unusable_mode, jump_gate), jump, EMA before and after, status.
    """
    record: dict | None = None         # set by every update()

    def update(self, results, log):
        cfg = self._cfg
        result = results[0] if results else None
        before = self._ema.value
        jump = None if result is None or before is None else result.offset - before
        # LaneFilter.update()'s checks in order; the parity test pins "accepted" to its status
        if result is None:
            reason = NO_RESULT
        elif result.mode not in USABLE_LANE_MODES:
            reason = UNUSABLE_MODE
        elif cfg.max_offset_jump is not None and jump is not None and abs(jump) > cfg.max_offset_jump:
            reason = JUMP_GATE
        else:
            reason = None

        est = super().update(results, log)
        self.record = {
            "raw_offset": None if result is None else result.offset,
            "mode": None if result is None else result.mode,
            "accepted": reason is None,
            "reason": reason,
            "jump": jump,
            "max_jump": cfg.max_offset_jump,
            "ema_before": before,
            "ema_after": self._ema.value,
            "missed": self._missed,
            "hold_max": cfg.hold_max_frames,
            "status": est.status,
            "offset": est.offset,
        }
        return est


class TracedHeadingTracker(HeadingTracker):
    """record: yaw_rate, dt, heading, reset (vision zeroes it), integrated."""
    record: dict | None = None         # set by every update()

    def update(self, lane_status, yaw_rate_dps, dt, log):
        heading = super().update(lane_status, yaw_rate_dps, dt, log)
        reset = lane_status == LANE_VISION
        self.record = {
            "yaw_rate": yaw_rate_dps,
            "dt": dt,
            "heading": heading,
            "reset": reset,
            "integrated": not reset and yaw_rate_dps is not None and dt > 0.0,
        }
        return heading


def _gated(detections, kind: str, gate: float) -> list[dict]:
    """Per detection of kind: label, confidence, gate, passed, bbox."""
    return [{
        "label": d.label_detail,
        "confidence": d.confidence,
        "gate": gate,
        "passed": not d.confidence < gate,        # the classifiers skip conf < gate
        "bbox": d.bounding_box,
        "source_rect": d.source_rect,
    } for d in detections if d.type == kind]


def _vote_record(vote, detections=None) -> dict:
    """raw (the sample just added), buffer, window, state."""
    rec = {"raw_vote": vote.buffer[-1], "buffer": vote.buffer,
           "window": vote.window, "state": vote.state}
    if detections is not None:
        rec["detections"] = detections
    return rec


class TracedTrafficClassifier(TrafficClassifier):
    record: dict | None = None         # set by every update()

    def update(self, detections, log):
        state = super().update(detections, log)
        self.record = _vote_record(self._vote, _gated(detections, TRAFFIC_LIGHT,
                                                      self._cfg.min_confidence_traffic))
        return state


class TracedStopSignClassifier(StopSignClassifier):
    record: dict | None = None         # set by every update()

    def update(self, detections, log):
        state = super().update(detections, log)
        self.record = _vote_record(self._vote, _gated(detections, STOP_SIGN,
                                                      self._cfg.min_confidence_sign))
        return state


class TracedStopLineClassifier(StopLineClassifier):
    """
    Adds seen, this frame's measured_*, held (on an earlier frame's distance), and the packet's reported_*.
    """
    record: dict | None = None         # set by every update()

    def update(self, results, log):
        seen = next((r for r in results if r.detected), None)
        flag, px, cm = super().update(results, log)
        self.record = {
            **_vote_record(self._vote),
            "seen": seen is not None,
            "measured_px": None if seen is None else seen.distance_px,
            "measured_cm": None if seen is None else seen.distance_cm,
            "measured_tilt": None if seen is None else seen.tilt_deg,
            "held": flag and seen is None,
            "reported_px": px,
            "reported_cm": cm,
        }
        return flag, px, cm


class TracedObstacleClassifier(ObstacleClassifier):
    """Adds seen, this frame's measured_*, held, and the packet's reported_* and clipped."""
    record: dict | None = None         # set by every update()

    def update(self, results, log):
        seen = next((r for r in results if r.detected), None)
        flag, px, cm, clipped = super().update(results, log)
        self.record = {
            **_vote_record(self._vote),
            "seen": seen is not None,
            "measured_px": None if seen is None else seen.distance_px,
            "measured_cm": None if seen is None else seen.distance_cm,
            "held": flag and seen is None,
            "reported_px": px,
            "reported_cm": cm,
            "clipped": clipped,
        }
        return flag, px, cm, clipped


class TracedPhase3Processor(Phase3Processor):
    """Phase3Processor with traced stages and per-stage timing; process() every frame in order."""
    STAGES = ("p3_dt", "p3_lane", "p3_heading", "p3_traffic", "p3_stop_sign",
              "p3_stop_line", "p3_obstacle", "p3_odometry", "p3_package")

    def __init__(self, config: Phase3Config | None = None) -> None:
        super().__init__(config)
        self.lane = TracedLaneFilter(self._cfg)
        self.heading = TracedHeadingTracker(self._cfg)
        self.traffic = TracedTrafficClassifier(self._cfg)
        self.stop_sign = TracedStopSignClassifier(self._cfg)
        self.stop_line = TracedStopLineClassifier(self._cfg)
        self.obstacle = TracedObstacleClassifier(self._cfg)

    def process(
            self,
            phase2: Phase2Output,
            sensors: SensorSample | None = None,
        ) -> tuple[EstimationPacket, dict]:
        """
        One Phase 3 cycle, recording every stage.

        packet is what Phase3Processor would return; debug holds its fields, each
        stage's record and timings_ms per STAGES name. ValueError if phase2 is None.
        """
        if phase2 is None:
            raise ValueError("TracedPhase3Processor.process: phase2 output is None")
        sensors = sensors or SensorSample()
        log, timings = [], {}
        mark = [time.perf_counter()]

        def lap(name):
            now = time.perf_counter()
            timings[name] = (now - mark[0]) * 1000.0
            mark[0] = now

        dt = self._dt(phase2.timestamp_ms, log)
        lap("p3_dt")
        lane = self.lane.update(phase2.lane_offset_results, log)
        lap("p3_lane")
        heading = self.heading.update(lane.status, sensors.yaw_rate_dps, dt, log)
        lap("p3_heading")
        drive_state = self.traffic.update(phase2.detections, log)
        lap("p3_traffic")
        stop_sign = self.stop_sign.update(phase2.detections, log)
        lap("p3_stop_sign")
        stop_line, stop_line_px, stop_line_cm = self.stop_line.update(phase2.stop_line_results, log)
        lap("p3_stop_line")
        obstacle, obstacle_px, obstacle_cm, obstacle_clipped = self.obstacle.update(phase2.obstacle_results, log)
        lap("p3_obstacle")
        odom = self.odometry.update(sensors.left_count, sensors.right_count, log)
        lap("p3_odometry")

        packet = EstimationPacket(
            lane_offset = lane.offset,
            lane_offset_cm = lane.offset_cm,
            lane_status = lane.status,
            heading_error = heading,
            drive_state = drive_state,
            stop_sign_detected = stop_sign,
            stop_line_detected = stop_line,
            stop_line_distance_px = stop_line_px,
            stop_line_distance_cm = stop_line_cm,
            yaw_rate = sensors.yaw_rate_dps or 0.0,
            lateral_accel = sensors.lateral_accel_mps2 or 0.0,
            frame_id = phase2.frame_id,
            timestamp_ms = phase2.timestamp_ms,
            left_wheel_cps = sensors.left_wheel_cps or 0.0,
            right_wheel_cps = sensors.right_wheel_cps or 0.0,
            lane_mode = lane_mode_of(phase2),
            obstacle_detected = obstacle,
            obstacle_distance_px = obstacle_px,
            obstacle_distance_cm = obstacle_cm,
            obstacle_clipped = obstacle_clipped,
            distance_cm = odom.distance_cm,
            left_distance_cm = odom.left_cm,
            right_distance_cm = odom.right_cm,
            stop_line_tilt_deg = self.stop_line.tilt,
        )
        lap("p3_package")
        return packet, {
            "frame_id": phase2.frame_id,
            "timestamp_ms": phase2.timestamp_ms,
            "dt": round(dt, 4),
            "log": log,
            "lane": self.lane.record,
            "heading": self.heading.record,
            "traffic": self.traffic.record,
            "stop_sign": self.stop_sign.record,
            "stop_line": self.stop_line.record,
            "obstacle": self.obstacle.record,
            "odometry": self.odometry.record,
            "timings_ms": timings,
        }
