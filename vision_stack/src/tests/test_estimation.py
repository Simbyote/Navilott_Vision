"""
test_estimation.py  --  Phase 3 estimation

Each stage is tested alone through its update(), then Phase3Processor is
tested for ordering, stamps and pass-through. Inputs are built from the real
Phase 2 dataclasses (DetectionObject, LaneOffsetResult, Phase2Output) so a
change to the Phase 2 contract breaks these tests rather than the robot.

--software  Stage behavior and packet contract. No camera, no IMU.
"""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.estimation import (
    CAUTION, GO, LANE_HOLD, LANE_STALE, LANE_VISION, STOP,
    HeadingTracker, LaneFilter, Phase3Config, Phase3Processor, SensorSample,
    StopSignClassifier, TrafficClassifier,
)
from src.perception.feature_fusion import DetectionObject
from src.perception.lane_offset import LaneOffsetResult
from src.perception.phase2_out import Phase2Output


# =============================================================================
# Helpers
# =============================================================================
FID, TS = 1, 100

def lane(offset=0.0, mode="two_boundary", frame_id=FID, ts=TS):
    return LaneOffsetResult(offset, 10.0, 20.0, 10.0, 0.6, 2, mode, frame_id, ts)

def det(det_type, label="", conf=0.9, frame_id=FID, ts=TS):
    roi = {"traffic_light": "traffic", "stop_sign": "sign"}.get(det_type, "lane")
    return DetectionObject(det_type, label, conf, {"x": 1.0, "y": 2.0}, (0, 0, 2, 4),
                           roi, (0, 0, 10, 10), frame_id, ts)

def p2(dets=(), lanes=None, frame_id=FID, ts=TS):
    lanes = [lane(frame_id=frame_id, ts=ts)] if lanes is None else lanes
    return Phase2Output(list(dets), list(lanes), frame_id, ts)

def feed(stage, values):
    log = []
    return [stage.update(v, log) for v in values]


# =============================================================================
# Software: LaneFilter
# =============================================================================
@pytest.mark.software
def test_lane_first_usable_frame_seeds_the_estimate():
    est = LaneFilter(Phase3Config()).update([lane(0.2)], [])
    assert (est.offset, est.status) == (0.2, LANE_VISION)


@pytest.mark.software
def test_lane_is_stale_before_any_measurement():
    est = LaneFilter(Phase3Config()).update([lane(mode="none")], [])
    assert (est.offset, est.status) == (0.0, LANE_STALE)


@pytest.mark.software
@pytest.mark.parametrize("mode", ["none", "single_uncalibrated"])
def test_lane_unusable_modes_are_dropouts_not_measurements(mode):
    f = LaneFilter(Phase3Config())
    f.update([lane(0.2)], [])
    est = f.update([lane(0.9, mode=mode)], [])
    assert (est.offset, est.status) == (0.2, LANE_HOLD)


@pytest.mark.software
def test_lane_ema_smooths_toward_new_measurements():
    f = LaneFilter(Phase3Config(ema_alpha=0.5, max_offset_jump=None))
    f.update([lane(0.0)], [])
    assert f.update([lane(0.4)], []).offset == pytest.approx(0.2)


@pytest.mark.software
def test_lane_hold_expires_to_stale_after_hold_max_frames():
    f = LaneFilter(Phase3Config(hold_max_frames=2))
    f.update([lane(0.1)], [])
    statuses = [f.update([], []).status for _ in range(3)]
    assert statuses == [LANE_HOLD, LANE_HOLD, LANE_STALE]


@pytest.mark.software
def test_lane_jump_beyond_gate_is_a_dropout():
    f = LaneFilter(Phase3Config(max_offset_jump=0.3))
    f.update([lane(0.0)], [])
    log = []
    est = f.update([lane(0.8)], log)
    assert (est.offset, est.status) == (0.0, LANE_HOLD)
    assert any("[LANE] jump" in e for e in log)


@pytest.mark.software
def test_lane_reseeds_after_stale_so_the_jump_gate_cannot_lock_out():
    f = LaneFilter(Phase3Config(max_offset_jump=0.3, hold_max_frames=1))
    f.update([lane(0.0)], [])
    f.update([lane(0.8)], [])            # hold
    f.update([lane(0.8)], [])            # stale, estimate cleared
    est = f.update([lane(0.8)], [])
    assert (est.offset, est.status) == (0.8, LANE_VISION)


@pytest.mark.software
def test_lane_cm_is_none_until_both_scale_values_are_set():
    assert LaneFilter(Phase3Config(cm_per_px=0.05)).update([lane(0.5)], []).offset_cm is None


@pytest.mark.software
def test_lane_cm_undoes_the_half_width_normalization():
    cfg = Phase3Config(lane_roi_width_px=480, cm_per_px=0.05)
    assert LaneFilter(cfg).update([lane(0.5)], []).offset_cm == pytest.approx(0.5 * 240 * 0.05)


# =============================================================================
# Software: HeadingTracker
# =============================================================================
@pytest.mark.software
def test_heading_is_zero_while_on_vision():
    h = HeadingTracker(Phase3Config())
    assert h.update(LANE_VISION, 50.0, 0.1, []) == 0.0


@pytest.mark.software
def test_heading_integrates_yaw_minus_bias_while_vision_is_lost():
    h = HeadingTracker(Phase3Config(gyro_bias_dps=1.0))
    h.update(LANE_HOLD, 11.0, 0.1, [])
    assert h.update(LANE_HOLD, 11.0, 0.1, []) == pytest.approx(2.0)


@pytest.mark.software
def test_heading_resets_when_vision_returns():
    h = HeadingTracker(Phase3Config())
    h.update(LANE_HOLD, 20.0, 0.5, [])
    assert h.update(LANE_VISION, 20.0, 0.5, []) == 0.0


@pytest.mark.software
def test_heading_is_clamped():
    h = HeadingTracker(Phase3Config(heading_limit_deg=10.0))
    assert h.update(LANE_STALE, 1000.0, 0.5, []) == 10.0


@pytest.mark.software
def test_heading_holds_when_no_yaw_is_available():
    h = HeadingTracker(Phase3Config())
    h.update(LANE_HOLD, 10.0, 0.1, [])
    assert h.update(LANE_HOLD, None, 0.1, []) == pytest.approx(1.0)


# =============================================================================
# Software: TrafficClassifier
# =============================================================================
@pytest.mark.software
def test_traffic_needs_two_of_three_frames_to_change_state():
    t = TrafficClassifier(Phase3Config(vote_window=3))
    red = [det("traffic_light", "red")]
    assert feed(t, [red, red]) == [GO, STOP]


@pytest.mark.software
def test_traffic_one_bad_frame_does_not_release_a_stop():
    t = TrafficClassifier(Phase3Config(vote_window=3))
    red = [det("traffic_light", "red")]
    assert feed(t, [red, red, [], red])[-2:] == [STOP, STOP]


@pytest.mark.software
def test_traffic_three_way_split_keeps_the_previous_state():
    t = TrafficClassifier(Phase3Config(vote_window=3))
    red, yellow = [det("traffic_light", "red")], [det("traffic_light", "yellow")]
    states = feed(t, [red, red, yellow, []])
    assert states[-1] == STOP    # window is red, yellow, go: no majority


@pytest.mark.software
def test_traffic_maps_yellow_to_caution():
    t = TrafficClassifier(Phase3Config(vote_window=1))
    assert feed(t, [[det("traffic_light", "yellow")]]) == [CAUTION]


@pytest.mark.software
def test_traffic_below_gate_counts_as_no_light():
    t = TrafficClassifier(Phase3Config(vote_window=1, min_confidence_traffic=0.5))
    assert feed(t, [[det("traffic_light", "red", conf=0.4)]]) == [GO]


@pytest.mark.software
def test_traffic_ignores_other_detection_types():
    t = TrafficClassifier(Phase3Config(vote_window=1))
    assert feed(t, [[det("stop_sign"), det("lane_boundary")]]) == [GO]


# =============================================================================
# Software: StopSignClassifier
# =============================================================================
@pytest.mark.software
def test_sign_needs_a_majority_of_the_window():
    s = StopSignClassifier(Phase3Config(vote_window=3))
    sign = [det("stop_sign")]
    assert feed(s, [sign, sign]) == [False, True]


@pytest.mark.software
def test_sign_below_gate_is_ignored():
    s = StopSignClassifier(Phase3Config(vote_window=1, min_confidence_sign=0.5))
    assert feed(s, [[det("stop_sign", conf=0.3)]]) == [False]


# =============================================================================
# Software: SensorSample
# =============================================================================
@pytest.mark.software
def test_from_imu_reads_a_valid_frame():
    frame = SimpleNamespace(valid=True, mean_yaw_rate_dps=3.0, peak_lateral_accel=-0.5)
    s = SensorSample.from_imu(frame, wheel_speed_mps=0.2)
    assert (s.yaw_rate_dps, s.lateral_accel_mps2, s.wheel_speed_mps) == (3.0, -0.5, 0.2)


@pytest.mark.software
def test_from_imu_invalid_frame_gives_no_readings():
    frame = SimpleNamespace(valid=False, mean_yaw_rate_dps=None, peak_lateral_accel=None)
    assert SensorSample.from_imu(frame) == SensorSample()


# =============================================================================
# Software: Phase3Processor
# =============================================================================
@pytest.mark.software
def test_packet_carries_the_phase2_stamp():
    pkt, dbg = Phase3Processor().process(p2(frame_id=7, ts=350))
    assert (pkt.frame_id, pkt.timestamp_ms) == (7, 350)
    assert (dbg["frame_id"], dbg["timestamp_ms"]) == (7, 350)


@pytest.mark.software
def test_packet_is_frozen():
    pkt, _ = Phase3Processor().process(p2())
    with pytest.raises(FrozenInstanceError):
        pkt.lane_offset = 1.0


@pytest.mark.software
def test_missing_sensors_pass_through_as_zero():
    pkt, _ = Phase3Processor().process(p2(), None)
    assert (pkt.yaw_rate, pkt.lateral_accel, pkt.wheel_speed) == (0.0, 0.0, 0.0)


@pytest.mark.software
def test_sensors_pass_through():
    s = SensorSample(yaw_rate_dps=2.0, lateral_accel_mps2=0.3, wheel_speed_mps=0.1)
    pkt, _ = Phase3Processor().process(p2(), s)
    assert (pkt.yaw_rate, pkt.lateral_accel, pkt.wheel_speed) == (2.0, 0.3, 0.1)


@pytest.mark.software
def test_none_phase2_raises():
    with pytest.raises(ValueError, match="None"):
        Phase3Processor().process(None)


@pytest.mark.software
def test_dt_is_clamped_on_a_long_gap():
    proc = Phase3Processor(Phase3Config(max_dt_s=0.5))
    proc.process(p2(ts=0, lanes=[lane(ts=0)]))
    _, dbg = proc.process(p2(frame_id=2, ts=5000, lanes=[lane(frame_id=2, ts=5000)]))
    assert dbg["dt"] == 0.5


@pytest.mark.software
def test_heading_integrates_across_frames_once_lane_drops():
    proc = Phase3Processor(Phase3Config(hold_max_frames=5))
    yaw = SensorSample(yaw_rate_dps=10.0)
    proc.process(p2(frame_id=1, ts=0, lanes=[lane(frame_id=1, ts=0)]), yaw)
    proc.process(p2(frame_id=2, ts=100, lanes=[]), yaw)
    pkt, _ = proc.process(p2(frame_id=3, ts=200, lanes=[]), yaw)
    assert pkt.lane_status == LANE_HOLD
    assert pkt.heading_error == pytest.approx(2.0)


@pytest.mark.software
def test_red_light_sequence_stops_then_releases():
    proc = Phase3Processor(Phase3Config(vote_window=3))
    red = [det("traffic_light", "red")]
    green = [det("traffic_light", "green")]
    states = [proc.process(p2(d))[0].drive_state for d in (red, red, red, green, green)]
    assert states == [GO, STOP, STOP, STOP, GO]