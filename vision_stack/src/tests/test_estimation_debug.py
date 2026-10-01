"""
test_estimation_debug.py  --  src/estimation_debug.py

The debug twin is held to the production Phase3Processor first: identical
packets and log over the synthetic drive and over a randomized Phase 2
stream. Then each stage's record is checked on the cases the Phase 3 video
has to show: a detection accepted vs gated, a lane jump rejected then held
into stale, a stop-line distance held, the heading reset.

--software  Records, parity and timings. No camera, no IMU.
"""
import random
from dataclasses import replace

import pytest

from src.estimation.estimation import (
    GO, LANE_HOLD, LANE_STALE, LANE_VISION, STOP, Phase3Config, Phase3Processor, SensorSample,
)
from src.debugger.estimation_debug import (
    JUMP_GATE, NO_RESULT, UNUSABLE_MODE, TracedPhase3Processor,
)
from src.perception.phase2_out import Phase2Output
from src.perception.stop_line_distance import StopLineResult
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, drive_sequence, same
from src.tests.test_estimation import det, lane

CFG = Phase3Config(max_offset_jump=0.3, hold_max_frames=2, vote_window=3)

def line(px, detected=True, fid=1, ts=100, cm=None):
    """Real StopLineResult; distance_px is None when not detected, as Phase 2 reports it."""
    if not detected:
        return StopLineResult(False, None, None, None, None, None, False, 0.0, 0, fid, ts)
    return StopLineResult(True, px, 50.0, 100.0, 300.0, 0.0, False, 0.7, 1, fid, ts,
                          distance_cm=cm, proximity=0.5)

def phase2(fid, ts, dets=(), lanes=None, lines=()):
    """Phase2Output at (fid, ts), with every piece restamped to it; lanes None is one centered result."""
    lanes = [lane()] if lanes is None else lanes
    return Phase2Output([replace(d, frame_id=fid, timestamp=ts) for d in dets],
                        [replace(r, frame_id=fid, timestamp_ms=ts) for r in lanes], fid, ts,
                        stop_line_results=[replace(r, frame_id=fid, timestamp_ms=ts) for r in lines])

def frames(*items):
    """Phase2Outputs 50 ms apart; each item is (dets, lanes, lines), lines None for none."""
    return [phase2(i, i * 50, dets, lanes, lines or ()) for i, (dets, lanes, lines) in enumerate(items)]

def run(stream, cfg=CFG, sensors=None):
    proc = TracedPhase3Processor(cfg)
    return [proc.process(o, sensors) for o in stream]


# =============================================================================
# Parity with the production processor
# =============================================================================

def _random_stream(n, seed):
    """Lanes that jump, drop and go unusable; lights of every color; signs and lines around the gates; stalls."""
    rng = random.Random(seed)
    out, ts = [], 0
    for i in range(n):
        ts += rng.choice([50, 50, 50, 900])                    # an occasional stall past max_dt_s
        lanes = rng.choice([[], [lane(rng.uniform(-1, 1))], [lane(rng.uniform(-0.2, 0.2))],
                            [lane(0.0, mode=rng.choice(["none", "single_uncalibrated"]))]])
        dets = [det("traffic_light", rng.choice(("red", "yellow", "green", "unknown")), rng.random())
                for _ in range(rng.randint(0, 1))]
        dets += [det("stop_sign", "stop", rng.random()) for _ in range(rng.randint(0, 2))]
        lines = [line(rng.uniform(0, 80), rng.random() < 0.6, cm=rng.choice([None, rng.uniform(5, 50)]))
                 for _ in range(rng.randint(0, 1))]
        out.append(phase2(i, ts, dets, lanes, lines))
    return out


@pytest.mark.software
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("cfg", [CFG, Phase3Config(), Phase3Config(max_offset_jump=None, vote_window=5)])
def test_packets_and_log_match_the_production_processor(seed, cfg):
    prod, traced = Phase3Processor(cfg), TracedPhase3Processor(cfg)
    rng = random.Random(seed)
    for o in _random_stream(300, seed):
        sensors = SensorSample(yaw_rate_dps=rng.choice([None, rng.uniform(-30, 30)]),
                               left_wheel_cps=rng.choice([None, 0.0, rng.uniform(-400, 400)]),
                               right_wheel_cps=rng.choice([None, 0.0, rng.uniform(-400, 400)]))
        p_pkt, p_dbg = prod.process(o, sensors)
        t_pkt, t_dbg = traced.process(o, sensors)
        same(p_pkt, t_pkt, f"f{o.frame_id} packet")
        same(p_dbg, {k: t_dbg[k] for k in p_dbg}, f"f{o.frame_id} debug")
        # The twin's lane verdict is the production status: accepted exactly when on vision
        assert t_dbg["lane"]["accepted"] == (t_pkt.lane_status == LANE_VISION)


@pytest.mark.software
def test_packets_match_over_the_synthetic_drive():
    prod, traced = Phase3Processor(), TracedPhase3Processor()
    for sf in drive_sequence():
        phase2 = run_chain(sf.frame, sf.frame_id, sf.timestamp_ms, SCENE_CONFIG).phase2
        same(prod.process(phase2, sf.sensors)[0], traced.process(phase2, sf.sensors)[0],
             f"{sf.segment} f{sf.frame_id}")


@pytest.mark.software
def test_every_stage_is_timed_and_the_record_keys_are_present():
    _, dbg = run(frames(([], None, None)))[0]
    assert tuple(dbg["timings_ms"]) == TracedPhase3Processor.STAGES
    assert all(v >= 0.0 for v in dbg["timings_ms"].values())
    for stage in ("lane", "heading", "traffic", "stop_sign", "stop_line"):
        assert isinstance(dbg[stage], dict)


# =============================================================================
# Traffic and stop sign: accepted vs gated, vote buffer
# =============================================================================

@pytest.mark.software
def test_a_detection_above_the_gate_passes_and_one_below_is_gated():
    stream = frames(([det("traffic_light", "red", 0.9)], None, None),
                    ([det("traffic_light", "red", 0.2)], None, None),
                    ([det("traffic_light", "red", 0.9)], None, None))
    recs = [dbg["traffic"] for _, dbg in run(stream)]
    first = recs[0]["detections"][0]
    assert (first["label"], first["confidence"], first["gate"], first["passed"]) == ("red", 0.9, 0.40, True)
    assert first["bbox"] == (0, 0, 2, 4) and first["source_rect"] == (0, 0, 10, 10)
    assert recs[1]["detections"][0]["passed"] is False
    assert [r["raw_vote"] for r in recs] == [STOP, GO, STOP]
    assert [r["buffer"] for r in recs] == [(STOP,), (STOP, GO), (STOP, GO, STOP)]
    assert [r["state"] for r in recs] == [GO, GO, STOP]      # 2 of 3 needed
    assert recs[0]["window"] == 3


@pytest.mark.software
def test_the_gate_is_inclusive_as_in_the_classifier():
    cfg = Phase3Config(min_confidence_sign=0.45)
    _, dbg = run(frames(([det("stop_sign", "stop", 0.45)], None, None)), cfg)[0]
    assert dbg["stop_sign"]["detections"][0]["passed"] is True
    assert dbg["stop_sign"]["raw_vote"] is True


@pytest.mark.software
def test_each_vote_records_only_its_own_detection_type():
    _, dbg = run(frames(([det("traffic_light", "green", 0.9), det("stop_sign", "stop", 0.3),
                          det("lane_boundary", "solid", 0.9)], None, None)))[0]
    assert [d["label"] for d in dbg["traffic"]["detections"]] == ["green"]
    assert [(d["passed"], d["gate"]) for d in dbg["stop_sign"]["detections"]] == [(False, 0.45)]
    assert dbg["stop_sign"]["raw_vote"] is False
    assert "detections" not in dbg["stop_line"]


# =============================================================================
# Lane: accepted, jump rejected, hold into stale
# =============================================================================

@pytest.mark.software
def test_a_lane_jump_is_rejected_then_held_into_stale_and_reseeded():
    stream = frames(([], [lane(0.0)], None), ([], [lane(0.8)], None), ([], [lane(0.8)], None),
                    ([], [lane(0.8)], None), ([], [lane(0.8)], None))
    recs = [dbg["lane"] for _, dbg in run(stream)]
    assert [(r["accepted"], r["reason"], r["status"], r["missed"]) for r in recs] == [
        (True, None, LANE_VISION, 0),
        (False, JUMP_GATE, LANE_HOLD, 1),
        (False, JUMP_GATE, LANE_HOLD, 2),
        (False, JUMP_GATE, LANE_STALE, 3),       # hold_max 2 exceeded: EMA cleared
        (True, None, LANE_VISION, 0),            # re-seeded, no jump to gate against
    ]
    assert recs[1]["jump"] == pytest.approx(0.8) and recs[1]["max_jump"] == 0.3
    assert (recs[1]["raw_offset"], recs[1]["offset"]) == (0.8, 0.0)   # output held while input moved
    assert recs[1]["ema_before"] == recs[1]["ema_after"] == 0.0
    assert (recs[3]["ema_before"], recs[3]["ema_after"]) == (0.0, None)
    assert (recs[4]["ema_before"], recs[4]["jump"], recs[4]["ema_after"]) == (None, None, 0.8)
    assert all(r["hold_max"] == 2 for r in recs)


@pytest.mark.software
def test_lane_reasons_for_no_result_and_an_unusable_mode():
    recs = [dbg["lane"] for _, dbg in run(frames(([], [], None), ([], [lane(0.5, mode="none")], None)))]
    assert [(r["reason"], r["raw_offset"], r["mode"]) for r in recs] == [
        (NO_RESULT, None, None), (UNUSABLE_MODE, 0.5, "none")]
    assert recs[0]["status"] == LANE_STALE and recs[0]["ema_before"] is None


# =============================================================================
# Stop line: seen vs held
# =============================================================================

@pytest.mark.software
def test_a_stop_line_distance_is_held_through_a_missed_frame():
    stream = frames(([], None, [line(30.0, cm=40.0)]), ([], None, [line(20.0, cm=30.0)]),
                    ([], None, [line(0.0, detected=False)]), ([], None, [line(0.0, detected=False)]))
    recs = [dbg["stop_line"] for _, dbg in run(stream)]
    assert [(r["seen"], r["held"], r["state"]) for r in recs] == [
        (True, False, False), (True, False, True), (False, True, True), (False, False, False)]
    assert (recs[1]["measured_px"], recs[1]["reported_px"], recs[1]["reported_cm"]) == (20.0, 20.0, 30.0)
    assert (recs[2]["measured_px"], recs[2]["reported_px"], recs[2]["reported_cm"]) == (None, 20.0, 30.0)
    assert recs[3]["reported_px"] is None
    assert [r["buffer"] for r in recs][-1] == (True, False, False)
    assert recs[0]["raw_vote"] is True and recs[2]["raw_vote"] is False


# =============================================================================
# Heading
# =============================================================================

@pytest.mark.software
def test_heading_resets_on_vision_and_integrates_off_it():
    stream = frames(([], [lane(0.0)], None), ([], [], None), ([], [], None))
    proc = TracedPhase3Processor(CFG)
    recs = []
    for o, yaw in zip(stream, (10.0, 10.0, None)):
        recs.append(proc.process(o, SensorSample(yaw_rate_dps=yaw))[1]["heading"])
    assert [(r["reset"], r["integrated"]) for r in recs] == [(True, False), (False, True), (False, False)]
    assert recs[1]["dt"] == pytest.approx(0.05) and recs[1]["heading"] == pytest.approx(0.5)
    assert recs[2]["yaw_rate"] is None and recs[2]["heading"] == pytest.approx(0.5)
