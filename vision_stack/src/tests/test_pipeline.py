"""
test_pipeline.py  --  src/pipeline.py

The pipeline declares its own flow through the production twins; the
linkers run the same flow through the debug versions. These hold the two to
identical results at both handoffs, under configs that move every stage's
tuning, so a stage that is skipped, reordered or handed the wrong config
shows up as a field that differs.

--software  perceive() against run_chain() field by field on the shared
            scenes and the gate sweep; estimate() and step() against
            phase3_linker over a drive sequence, Phase 2 checked first on
            every frame; config reach for both phases; opt-in timing; the
            import boundary. No camera.
"""
import subprocess
import sys
from dataclasses import fields, replace

import numpy as np
import pytest

import src.perception.color_branch as cb
import src.perception.feature_fusion as ff
import src.perception.geometry as geo
import src.perception.lane_offset as lo
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.estimation import LANE_HOLD, LANE_STALE, LANE_VISION, Phase3Processor
from src.params import LANE_BOUNDARY, PIPELINE_ROOT, STOP_SIGN, TRAFFIC_LIGHT
from src.perception.color_branch import ColorConfig
from src.phase2_linker import run_chain
from src.phase3_linker import run_phase3_chain
from src.pipeline import Pipeline
from src.tests.scenes import (
    ALT_CONFIG, ALT_ESTIMATION, SCENE_CONFIG, SCENES, SWEEP, differs, drive_sequence, same, sweep_frame,
)

# MEASURED undistorts; synthetic frames come out warped, but both paths warp
# them identically, so it still pins the undistortion path's parity
CONFIGS = {
    "default": PipelineConfig(),
    "scene": SCENE_CONFIG,
    "scene_color_off": replace(SCENE_CONFIG, color=ColorConfig()),
    "alt": ALT_CONFIG,
    "measured": MEASURED,
}
GROUPS = ("preprocess", "roi", "geometry", "color", "lane_offset")


def _stamp(i: int) -> tuple[int, int]:
    """A frame stamp no stage could produce by accident."""
    return 1000 + i, 70_000 + 53 * i


# =============================================================================
# Parity with run_chain
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("config_name", CONFIGS)
@pytest.mark.parametrize("scene", SCENES)
def test_perceive_matches_run_chain(scene, config_name):
    config = CONFIGS[config_name]
    fid, ts = _stamp(len(scene))
    same(Pipeline(config).perceive(SCENES[scene], fid, ts),
         run_chain(SCENES[scene], fid, ts, config).phase2, "phase2")


@pytest.mark.software
@pytest.mark.parametrize("config_name", ["default", "scene", "alt"])
def test_perceive_matches_run_chain_across_the_gate_sweep(config_name):
    config = CONFIGS[config_name]
    pipeline = Pipeline(config)
    for i, (level, width, noise) in enumerate(SWEEP):
        frame, (fid, ts) = sweep_frame(level, width, noise), _stamp(i)
        same(pipeline.perceive(frame, fid, ts), run_chain(frame, fid, ts, config).phase2,
             f"level={level} width={width} noise={noise}")


@pytest.mark.software
def test_scenes_reach_every_output_through_the_pipeline():
    """Guards the parity tests: the scenes must reach every detection type and lane mode."""
    types, modes = set(), set()
    for config in CONFIGS.values():
        pipeline = Pipeline(config)
        for frame in SCENES.values():
            p2 = pipeline.perceive(frame, 1, 50)
            types |= {d.type for d in p2.detections}
            modes |= {r.mode for r in p2.lane_offset_results}
    assert {LANE_BOUNDARY, STOP_SIGN, TRAFFIC_LIGHT} <= types
    assert {"two_boundary", "none"} <= modes and len(modes) >= 3


# =============================================================================
# Config reach: ALT_CONFIG must be able to catch a stage ignoring its config
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("group", GROUPS)
def test_alt_config_moves_every_stage_group(group):
    assert getattr(ALT_CONFIG, group) != getattr(SCENE_CONFIG, group)


@pytest.mark.software
@pytest.mark.parametrize("group", GROUPS)
def test_each_alt_group_changes_the_phase2_output(group):
    """
    Swapping only this group's tuning changes the Phase 2 output on some
    scene. Without that, parity under ALT_CONFIG couldn't tell a stage that
    reads its config from one that ignores it.
    """
    swapped = replace(SCENE_CONFIG, **{group: getattr(ALT_CONFIG, group)})
    frames = list(SCENES.values()) + [sweep_frame(*p) for p in SWEEP]
    assert any(differs(run_chain(f, 1, 50, swapped).phase2, run_chain(f, 1, 50, SCENE_CONFIG).phase2)
               for f in frames), f"ALT_CONFIG.{group} changes nothing on any scene"


# =============================================================================
# Stage timing (opt-in)
# =============================================================================

@pytest.mark.software
def test_timing_is_off_by_default():
    pipeline = Pipeline(SCENE_CONFIG)
    pipeline.perceive(SCENES["two_boundary"], 1, 50)
    assert pipeline.last_timings_ms == {}


@pytest.mark.software
def test_timing_on_records_run_chains_stages_and_changes_nothing():
    frame = SCENES["intersection"]
    timed, plain = Pipeline(SCENE_CONFIG, timing=True), Pipeline(SCENE_CONFIG)
    same(timed.perceive(frame, 3, 150), plain.perceive(frame, 3, 150))
    first = timed.last_timings_ms
    assert list(first) == list(run_chain(frame, 3, 150, SCENE_CONFIG).timings_ms)
    assert all(v >= 0.0 for v in first.values())
    timed.perceive(frame, 4, 200)
    assert timed.last_timings_ms is not first          # a caller holding the old dict keeps it


# =============================================================================
# Contract
# =============================================================================

@pytest.mark.software
def test_default_config_is_measured():
    assert Pipeline().config is MEASURED


@pytest.mark.software
def test_perceive_carries_the_stamp_and_leaves_the_frame_untouched():
    frame = SCENES["intersection"]
    before = frame.copy()
    p2 = Pipeline(MEASURED).perceive(frame, 424242, 987_654_321)
    assert np.array_equal(frame, before)
    assert (p2.frame_id, p2.timestamp_ms) == (424242, 987_654_321)
    assert p2.detections and p2.lane_offset_results
    for d in p2.detections:                     # DetectionObject names its stamp "timestamp"
        assert (d.frame_id, d.timestamp) == (424242, 987_654_321)
    for r in p2.lane_offset_results:
        assert (r.frame_id, r.timestamp_ms) == (424242, 987_654_321)


@pytest.mark.software
def test_perceive_runs_no_debug_function(monkeypatch):
    """The debug versions are wired to raise; the pipeline must not notice."""
    def boom(*_, **__):
        raise AssertionError("debug path called")
    for module, name in ((geo, "run_geometry_stage"), (geo, "extract_lane_candidates"),
                         (geo, "extract_sign_candidates"), (geo, "_extract_lane_candidates"),
                         (geo, "_extract_sign_candidates"), (cb, "run_color_stage"),
                         (cb, "extract_traffic_light_candidates"), (cb, "_blobs_to_candidates"),
                         (lo, "compute_lane_offset"), (lo, "_usable"), (lo, "_single_sided"),
                         (ff, "fuse_detections"), (ff, "_best_candidate")):
        monkeypatch.setattr(module, name, boom)
    pipeline = Pipeline(SCENE_CONFIG, timing=True)
    for frame in SCENES.values():
        pipeline.perceive(frame, 1, 50)


@pytest.mark.software
def test_pipeline_imports_no_linker_or_debugger():
    probe = ("import sys, src.pipeline; "
             "print(sorted(m for m in sys.modules if m.startswith('src.')))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=PIPELINE_ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert not [m for m in eval(out) if m.startswith(("src.debugger", "src.phase2_linker",
                                                       "src.phase3_linker"))]


# =============================================================================
# Phase 3: estimate() and step() against phase3_linker
# =============================================================================

# (Phase 2 config, Phase 3 config). cm_scale reaches lane_offset_cm and the
# derived lane ROI width; alt moves every field of both
PACKET_CASES = {
    "measured": (SCENE_CONFIG, MEASURED_ESTIMATION),
    "cm_scale": (SCENE_CONFIG, replace(MEASURED_ESTIMATION, cm_per_px=0.05)),
    "alt": (ALT_CONFIG, ALT_ESTIMATION),
}
DRIVE = drive_sequence()


def _linker_processor(config, estimation, first_frame):
    """The Phase3Processor phase3_linker.run() builds: the lane ROI width from the first frame when cm_per_px is set."""
    if estimation.cm_per_px is not None and estimation.lane_roi_width_px is None:
        lane_w = run_chain(first_frame, 0, 0, config).roi.lane_rect[2]
        estimation = replace(estimation, lane_roi_width_px=int(lane_w))
    return Phase3Processor(estimation)

def _packets(pipeline, sequence=DRIVE):
    """step() over a sequence: every packet with Phase 3's debug beside it."""
    out = []
    for sf in sequence:
        out.append((pipeline.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors),
                    pipeline.last_estimation_debug))
    return out


@pytest.mark.software
@pytest.mark.parametrize("case", PACKET_CASES)
def test_packets_match_phase3_linker_over_a_drive(case):
    """
    Frame by frame: Phase 2 parity first, since a smoothed, voted packet
    hides most Phase 2 differences; then the packet and Phase 3's debug.
    """
    config, estimation = PACKET_CASES[case]
    pipeline = Pipeline(config, estimation)
    processor = _linker_processor(config, estimation, DRIVE[0].frame)
    for sf in DRIVE:
        where = f"{sf.segment} f{sf.frame_id}"
        res = run_phase3_chain(sf.frame, sf.frame_id, sf.timestamp_ms, processor, sf.sensors, config)
        p2 = pipeline.perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
        same(p2, res.chain.phase2, f"{where} phase2")
        same(pipeline.estimate(p2, sf.sensors), res.packet, f"{where} packet")
        same(pipeline.last_estimation_debug, res.p3_debug, f"{where} p3_debug")


@pytest.mark.software
@pytest.mark.parametrize("case", PACKET_CASES)
def test_step_is_perceive_then_estimate(case):
    config, estimation = PACKET_CASES[case]
    split = Pipeline(config, estimation)
    stepped = _packets(Pipeline(config, estimation))
    for sf, (packet, debug) in zip(DRIVE, stepped):
        same(split.estimate(split.perceive(sf.frame, sf.frame_id, sf.timestamp_ms), sf.sensors),
             packet, f"{sf.segment} f{sf.frame_id}")
        same(split.last_estimation_debug, debug)


@pytest.mark.software
def test_drive_sequence_moves_every_phase3_output():
    """Guards the packet parity test: the drive must change every vote, status and integrator."""
    seen = {k: set() for k in ("drive", "stop", "lane", "log")}
    headings, cm = [], []
    for case in PACKET_CASES.values():
        for (packet, debug), sf in zip(_packets(Pipeline(*case)), DRIVE):
            assert (packet.frame_id, packet.timestamp_ms) == (sf.frame_id, sf.timestamp_ms)
            seen["drive"].add(packet.drive_state)
            seen["stop"].add(packet.stop_sign_detected)
            seen["lane"].add(packet.lane_status)
            seen["log"] |= {entry.split("]")[0] + "]" for entry in debug["log"]}
            headings.append(packet.heading_error)
            cm.append(packet.lane_offset_cm)
    assert seen["drive"] == {"go", "caution", "stop"}
    assert seen["stop"] == {True, False}
    assert seen["lane"] == {LANE_VISION, LANE_HOLD, LANE_STALE}
    assert {"[TRAFFIC]", "[SIGN]", "[LANE]", "[DT]", "[HEADING]"} <= seen["log"]
    assert max(abs(h) for h in headings) > 10.0
    assert any(c is not None for c in cm) and any(c is None for c in cm)


ALT_FIELDS = [f.name for f in fields(ALT_ESTIMATION)
              if getattr(ALT_ESTIMATION, f.name) != getattr(MEASURED_ESTIMATION, f.name)]

@pytest.mark.software
def test_alt_estimation_moves_every_field_but_the_derived_width():
    assert set(ALT_FIELDS) == {f.name for f in fields(ALT_ESTIMATION)} - {"lane_roi_width_px"}

@pytest.mark.software
@pytest.mark.parametrize("field", ALT_FIELDS)
def test_each_alt_estimation_field_changes_the_packets(field):
    """Without this, packet parity under ALT_ESTIMATION couldn't catch Phase 3 ignoring a field."""
    swapped = replace(MEASURED_ESTIMATION, **{field: getattr(ALT_ESTIMATION, field)})
    assert differs(_packets(Pipeline(SCENE_CONFIG, swapped)),
                   _packets(Pipeline(SCENE_CONFIG, MEASURED_ESTIMATION))), \
        f"ALT_ESTIMATION.{field} changes nothing over the drive"


@pytest.mark.software
@pytest.mark.parametrize("config", [SCENE_CONFIG, ALT_CONFIG], ids=["scene", "alt"])
def test_cm_scale_lane_width_is_the_one_phase3_linker_takes(config):
    pipeline = Pipeline(config, replace(MEASURED_ESTIMATION, cm_per_px=0.05))
    assert pipeline.estimation.lane_roi_width_px == run_chain(DRIVE[0].frame, 0, 0, config).roi.lane_rect[2]
    assert pipeline.processor._cfg is pipeline.estimation


@pytest.mark.software
def test_cm_scale_refuses_a_frame_of_another_size():
    small = np.zeros((240, 320, 3), np.uint8)
    with pytest.raises(ValueError, match="cm scale"):
        Pipeline(SCENE_CONFIG, replace(MEASURED_ESTIMATION, cm_per_px=0.05)).perceive(small, 1, 50)
    Pipeline(SCENE_CONFIG).perceive(small, 1, 50)          # no cm scale: no size to hold it to


@pytest.mark.software
def test_phase3_state_is_built_once_and_kept():
    pipeline = Pipeline(SCENE_CONFIG)
    processor = pipeline.processor
    assert isinstance(processor, Phase3Processor) and pipeline.last_estimation_debug is None
    assert pipeline.estimation is MEASURED_ESTIMATION and Pipeline().estimation is MEASURED_ESTIMATION
    _packets(pipeline, DRIVE[:3])
    assert pipeline.processor is processor
    assert pipeline.last_estimation_debug["frame_id"] == DRIVE[2].frame_id


@pytest.mark.software
def test_step_timing_adds_phase3_only_when_on():
    sf = DRIVE[0]
    timed, plain = Pipeline(SCENE_CONFIG, timing=True), Pipeline(SCENE_CONFIG)
    same(timed.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors),
         plain.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors))
    stages = list(run_chain(sf.frame, 0, 0, SCENE_CONFIG).timings_ms)
    assert list(timed.last_timings_ms) == stages + ["phase3"]
    assert timed.last_timings_ms["phase3"] >= 0.0
    assert plain.last_timings_ms == {}


# =============================================================================
# Known defect: stop lines reach the lane offset (Section 6)
# =============================================================================

@pytest.mark.software
@pytest.mark.xfail(strict=True, reason="horizontal lines pass the lane gates until Section 6 "
                                       "separates them; remove this marker when it lands")
@pytest.mark.parametrize("scene", ["stop_line_short", "stop_line_wide", "stop_line_between_marks"])
def test_a_stop_line_leaves_the_lane_offset_alone(scene):
    pipeline = Pipeline(SCENE_CONFIG)
    clear = pipeline.perceive(SCENES["two_boundary"], 1, 50).lane_offset_results[0]
    crossed = pipeline.perceive(SCENES[scene], 1, 50).lane_offset_results[0]
    assert (crossed.mode, crossed.left_x, crossed.right_x) == (clear.mode, clear.left_x, clear.right_x)
