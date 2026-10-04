"""
test_pipeline.py  --  src/pipeline.py

The pipeline declares its own flow through the production twins; the
linkers run the same flow through the debug versions. These hold the two to
identical results at both handoffs, under configs that move every stage's
tuning, so a stage that is skipped, reordered or handed the wrong config
shows up as a field that differs.

--software  perceive() against run_chain() field by field on the shared
            scenes and the gate sweep; estimate() against phase3_linker over
            a drive sequence, Phase 2 checked first on every frame; step()
            against navigation_linker over a course that moves every
            navigation rule, command by command; config reach for both
            phases; the contract guard; opt-in timing; the import boundary.
            No camera or motors.
"""
import subprocess
import sys
import time
from dataclasses import fields, replace

import numpy as np
import pytest

import src.perception.color_branch as cb
import src.perception.feature_fusion as ff
import src.perception.geometry as geo
import src.perception.lane_offset as lo
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.estimation.estimation import LANE_HOLD, LANE_STALE, LANE_VISION, Phase3Processor
from src.params import LANE_BOUNDARY, PIPELINE_ROOT, STOP_SIGN, TRAFFIC_LIGHT
from src.perception.color_branch import ColorConfig
from src.phase2_linker import run_chain
import src.navigation_linker as nl
from src.navigation.navigation import (
    BRAKE, RULE_END_OF_COURSE, RULE_INTERSECTION, RULE_LANE_KEEPING, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT,
    Command, Navigation,
)
from src.navigation.route import Route
from src.phase3_linker import make_processor, run_phase3_chain
from src.pipeline import Pipeline
from src.tests.scenes import (
    ALT_CONFIG, ALT_ESTIMATION, GREEN_LAMP, RED_LAMP, SCENE_CONFIG, SCENES, SWEEP, SYNTHETIC_GROUND, SYNTHETIC_STOP_LINE_TABLE,
    YELLOW_LAMP, course_sequence, differs, drive_sequence, scene, same, sweep_frame,
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
GROUPS = ("preprocess", "roi", "geometry", "color", "lane_offset", "stop_line", "ground")


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
    assert any(Pipeline(SCENE_CONFIG).perceive(f, 1, 50).stop_line_results[0].detected for f in SCENES.values())


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


@pytest.mark.software
def test_alt_lane_horizontal_filter_alone_changes_the_phase2_output():
    """Like the stop-line filter: the geometry group guard can pass on its Canny change alone."""
    lane = SCENE_CONFIG.geometry.lane
    alt = ALT_CONFIG.geometry.lane
    swapped = replace(SCENE_CONFIG, geometry=replace(SCENE_CONFIG.geometry, lane=replace(
        lane, horizontal_edge_deg=alt.horizontal_edge_deg, horizontal_min_run_px=alt.horizontal_min_run_px,
        horizontal_band_px=alt.horizontal_band_px)))
    assert any(differs(run_chain(f, 1, 50, swapped).phase2, run_chain(f, 1, 50, SCENE_CONFIG).phase2)
               for f in SCENES.values())


@pytest.mark.software
def test_alt_stop_line_filter_alone_changes_the_phase2_output():
    """The geometry group guard can pass on its Canny change alone; the stop-line detector needs its own."""
    swapped = replace(SCENE_CONFIG, geometry=replace(SCENE_CONFIG.geometry,
                                                     stop_line=ALT_CONFIG.geometry.stop_line))
    assert ALT_CONFIG.geometry.stop_line != SCENE_CONFIG.geometry.stop_line
    assert any(differs(run_chain(f, 1, 50, swapped).phase2, run_chain(f, 1, 50, SCENE_CONFIG).phase2)
               for f in SCENES.values())


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
    "ground": (replace(SCENE_CONFIG, ground=SYNTHETIC_GROUND), MEASURED_ESTIMATION),
    "stop_line_table": (replace(SCENE_CONFIG, stop_line_table=SYNTHETIC_STOP_LINE_TABLE), MEASURED_ESTIMATION),
}
DRIVE = drive_sequence()



def _packets(pipeline, sequence=DRIVE):
    """step() over a sequence: every packet (last_packet) with Phase 3's debug beside it."""
    out = []
    for sf in sequence:
        pipeline.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors)
        out.append((pipeline.last_packet, pipeline.last_estimation_debug))
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
    sf0 = DRIVE[0]
    processor = make_processor(sf0.frame, sf0.frame_id, sf0.timestamp_ms, config, estimation)
    for sf in DRIVE:
        where = f"{sf.segment} f{sf.frame_id}"
        res = run_phase3_chain(sf.frame, sf.frame_id, sf.timestamp_ms, processor, sf.sensors, config)
        p2 = pipeline.perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
        same(p2, res.chain.phase2, f"{where} phase2")
        same(pipeline.estimate(p2, sf.sensors), res.packet, f"{where} packet")
        # The traced debug is a superset: production's fields must match it exactly
        same(pipeline.last_estimation_debug,
             {k: res.p3_debug[k] for k in pipeline.last_estimation_debug}, f"{where} p3_debug")


@pytest.mark.software
@pytest.mark.parametrize("case", PACKET_CASES)
def test_step_is_perceive_estimate_then_navigate(case):
    config, estimation = PACKET_CASES[case]
    split, stepped = Pipeline(config, estimation), Pipeline(config, estimation)
    for sf in DRIVE:
        cmd = stepped.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors)
        packet = split.estimate(split.perceive(sf.frame, sf.frame_id, sf.timestamp_ms), sf.sensors)
        where = f"{sf.segment} f{sf.frame_id}"
        same(packet, stepped.last_packet, where)
        same(split.last_estimation_debug, stepped.last_estimation_debug, where)
        assert split.navigate(packet) == cmd and isinstance(cmd, Command), where
        same(split.navigation.record, stepped.navigation.record, where)


@pytest.mark.software
@pytest.mark.parametrize("lamp, color", [(RED_LAMP, "red"), (YELLOW_LAMP, "yellow"), (GREEN_LAMP, "green")])
def test_each_scene_lamp_is_seen_as_its_color_under_the_calibrated_bands(lamp, color):
    """The scenes run on calibration/hsv_ranges.json: a lamp it no longer covers would read as no light, which looks like go."""
    lights = [d for d in run_chain(scene(lights=(lamp,), lamp_radius=12), 0, 0, SCENE_CONFIG).fusion.detections
              if d.type == "traffic_light"]
    assert [d.label_detail for d in lights] == [color] and lights[0].confidence >= 0.9, lights


@pytest.mark.software
def test_drive_sequence_moves_every_phase3_output():
    """Guards the packet parity test: the drive must change every vote, status and integrator."""
    seen = {k: set() for k in ("drive", "stop", "lane", "log", "line")}
    headings, cm, line_px = [], [], []
    for case in PACKET_CASES.values():
        for (packet, debug), sf in zip(_packets(Pipeline(*case)), DRIVE):
            assert (packet.frame_id, packet.timestamp_ms) == (sf.frame_id, sf.timestamp_ms)
            seen["drive"].add(packet.drive_state)
            seen["stop"].add(packet.stop_sign_detected)
            seen["line"].add(packet.stop_line_detected)
            line_px.append(packet.stop_line_distance_px)
            seen["lane"].add(packet.lane_status)
            seen["log"] |= {entry.split("]")[0] + "]" for entry in debug["log"]}
            headings.append(packet.heading_error)
            cm.append(packet.lane_offset_cm)
    assert seen["drive"] == {"go", "caution", "stop"}
    assert seen["stop"] == {True, False}
    assert seen["lane"] == {LANE_VISION, LANE_HOLD, LANE_STALE}
    assert {"[TRAFFIC]", "[SIGN]", "[LANE]", "[DT]", "[HEADING]", "[STOPLINE]"} <= seen["log"]
    assert seen["line"] == {True, False}
    assert 0.0 in line_px and len({d for d in line_px if d is not None}) >= 3
    assert max(abs(h) for h in headings) > 10.0
    assert any(c is not None for c in cm) and any(c is None for c in cm)


@pytest.mark.software
def test_the_drive_reports_cm_only_with_a_ground_plane():
    scene_cm = [p.stop_line_distance_cm for p, _ in _packets(Pipeline(*PACKET_CASES["measured"]))]
    ground_cm = [p.stop_line_distance_cm for p, _ in _packets(Pipeline(*PACKET_CASES["ground"]))]
    assert all(cm is None for cm in scene_cm)
    assert 0.0 in ground_cm and len({cm for cm in ground_cm if cm}) >= 3


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
def test_step_timing_adds_phase3_and_navigation_only_when_on():
    sf = DRIVE[0]
    timed, plain = Pipeline(SCENE_CONFIG, timing=True), Pipeline(SCENE_CONFIG)
    same(timed.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors),
         plain.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors))
    stages = list(run_chain(sf.frame, 0, 0, SCENE_CONFIG).timings_ms)
    assert list(timed.last_timings_ms) == stages + ["phase3", "navigation"]
    assert timed.last_timings_ms["phase3"] >= 0.0 and timed.last_timings_ms["navigation"] >= 0.0
    assert plain.last_timings_ms == {}


@pytest.mark.software
def test_each_stage_time_leaves_out_the_wait_between_stages():
    sf = DRIVE[0]
    timed = Pipeline(SCENE_CONFIG, timing=True)
    p2 = timed.perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
    time.sleep(0.05)
    packet = timed.estimate(p2, sf.sensors)
    time.sleep(0.05)
    timed.navigate(packet)
    assert timed.last_timings_ms["phase3"] < 40.0 and timed.last_timings_ms["navigation"] < 40.0


# =============================================================================
# Navigation: step() against navigation_linker
# =============================================================================

COURSE = course_sequence()
COURSE_ROUTE = Route(("left",))
COURSE_CONFIGS = {"scene": SCENE_CONFIG, "alt": ALT_CONFIG}     # alt also sees the synthetic stop sign
ALL_RULES = {RULE_LANE_KEEPING, RULE_INTERSECTION, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, RULE_END_OF_COURSE}


class _CourseSource:
    """live_view FrameSource stand-in replaying a sequence; current is the frame last read."""
    fps, label = 20, "course"

    def __init__(self, sequence):
        self.frames, self.current = iter(sequence), None

    def read(self):
        self.current = next(self.frames, None)
        sf = self.current
        return None if sf is None else (sf.frame, sf.frame_id, sf.timestamp_ms)

    def close(self):
        pass


class _CourseSensors:
    """phase3_linker.Sensors stand-in: the current frame's readings (none before the first frame)."""
    def __init__(self, source):
        self.source = source

    def read(self):
        return (None if self.source.current is None else self.source.current.sensors), None

    def stop(self):
        pass


class _Motor:
    """MotorController stand-in: the Command each frame drove, in order."""
    def __init__(self):
        self.commands = []

    def drive(self, left, right):
        self.commands.append(Command(left, right))

    def brake(self):
        self.commands.append(BRAKE)

    def stop(self):
        pass


class _Watched(Navigation):
    """Navigation keeping each packet and record, so the linker's run can be compared frame by frame."""
    def reset(self):
        super().reset()
        self.packets, self.records = [], []

    def update(self, packet):
        cmd = super().update(packet)
        self.packets.append(packet)
        self.records.append(self.record)
        return cmd


def _pipeline_course(config, sequence=COURSE):
    """step() over the course until Navigation finishes: (pipeline, commands, packets, records)."""
    pipeline = Pipeline(config, route=COURSE_ROUTE)
    cmds, packets, records = [], [], []
    for sf in sequence:
        cmds.append(pipeline.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors))
        packets.append(pipeline.last_packet)
        records.append(pipeline.navigation.record)
        if pipeline.finished:
            break
    return pipeline, cmds, packets, records


@pytest.mark.software
@pytest.mark.parametrize("case", COURSE_CONFIGS)
def test_commands_match_navigation_linker_over_a_course(case, tmp_path):
    """Frame by frame: the packet the navigator saw, its record, and the command the motors got; then how the run ended."""
    config = COURSE_CONFIGS[case]
    source, motor, nav = _CourseSource(COURSE), _Motor(), _Watched(route=COURSE_ROUTE, gyro_bias_dps=MEASURED_ESTIMATION.gyro_bias_dps)
    report = nl.run(source, _CourseSensors(source), motor, nav, config, MEASURED_ESTIMATION,
                    out_dir=str(tmp_path / "run"), max_run_s=1e9, render=False)
    pipeline, cmds, packets, records = _pipeline_course(config)
    assert len(motor.commands) == len(cmds)
    for i, sf in enumerate(COURSE[:len(cmds)]):
        where = f"{sf.segment} f{sf.frame_id}"
        same(packets[i], nav.packets[i], f"{where} packet")
        same(records[i], nav.records[i], f"{where} record")
        assert cmds[i] == motor.commands[i], where
    assert report["ended_by"] == nl.END_COURSE and pipeline.finished
    assert (report["outcome"], report["end_step"]) == (pipeline.navigation.outcome, pipeline.navigation.end_step)


@pytest.mark.software
def test_the_course_moves_every_navigation_rule_and_finishes():
    """Guards the command parity test: every rule and intersection stage decides some frame, the route is counted, the run finishes."""
    seen, stages = set(), set()
    for config in COURSE_CONFIGS.values():
        pipeline, cmds, _, records = _pipeline_course(config)
        seen |= {r["rule"] for r in records}
        stages |= {r.get("stage") for r in records if r["rule"] == RULE_INTERSECTION}
        assert BRAKE in cmds and any(not c.brake for c in cmds)
        assert pipeline.finished and (pipeline.navigation.outcome, pipeline.navigation.end_step) == ("finished", 1)
        assert len(cmds) < len(COURSE)                       # it ended on the course, not at the sequence's end
    assert seen == ALL_RULES
    assert stages == {"to_line", "turn", "exit"}                    # the route's left turn is driven on the gyro


@pytest.mark.software
def test_a_command_that_breaks_the_contract_is_braked_and_says_why():
    pipeline = Pipeline(SCENE_CONFIG)
    pipeline.navigation.update = lambda packet: Command(0.1, 2.0)
    sf = DRIVE[0]
    assert pipeline.step(sf.frame, sf.frame_id, sf.timestamp_ms, sf.sensors) == BRAKE
    assert len(pipeline.last_problems) == 2
    pipeline.navigation.update = lambda packet: Command(0.5, 0.5)
    assert pipeline.navigate(pipeline.last_packet) == Command(0.5, 0.5) and pipeline.last_problems == []


@pytest.mark.software
def test_the_route_and_the_gyro_bias_reach_navigation():
    route = Route(("right", "straight"), "stop_line")
    pipeline = Pipeline(SCENE_CONFIG, replace(MEASURED_ESTIMATION, gyro_bias_dps=1.5), route=route)
    assert pipeline.navigation.progress.route is route
    intersection = dict(pipeline.navigation.rules)[RULE_INTERSECTION]
    assert intersection.gyro_bias_dps == 1.5
    assert Pipeline(SCENE_CONFIG).navigation.progress.route == Route()


@pytest.mark.software
def test_navigation_state_is_built_once_and_finished_follows_it():
    pipeline = Pipeline(SCENE_CONFIG)
    navigation = pipeline.navigation
    assert pipeline.last_packet is None and pipeline.last_problems == [] and not pipeline.finished
    _packets(pipeline, DRIVE[:3])
    assert pipeline.navigation is navigation and pipeline.last_packet.frame_id == DRIVE[2].frame_id
    dict(navigation.rules)[RULE_END_OF_COURSE]._end("finished")
    assert pipeline.finished


# =============================================================================
# Stop lines and the lane offset
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("scene", ["stop_line_short", "stop_line_wide", "intersection"])
def test_a_stop_line_clear_of_the_lane_lines_leaves_the_lane_offset_alone(scene):
    """The stop line passes the lane gates as a lane candidate; lane_offset must skip it."""
    pipeline = Pipeline(SCENE_CONFIG)
    clear = pipeline.perceive(SCENES["two_boundary"], 1, 50).lane_offset_results[0]
    p2 = pipeline.perceive(SCENES[scene], 1, 50)
    crossed = p2.lane_offset_results[0]
    assert (crossed.mode, crossed.left_x, crossed.right_x) == (clear.mode, clear.left_x, clear.right_x)
    assert p2.stop_line_results[0].detected


@pytest.mark.software
@pytest.mark.parametrize("scene", ["stop_line_touching", "stop_line_between_marks", "stop_line_touching_left",
                                   "stop_line_far", "stop_line_clipped", "stop_line_tilted", "two_stop_lines"])
def test_a_stop_line_touching_the_lane_lines_keeps_the_lane_and_is_measured(scene):
    """The lane detector's edges lose the stop line, so it no longer joins the lane lines into one contour."""
    p2 = Pipeline(SCENE_CONFIG).perceive(SCENES[scene], 1, 50)
    lane = p2.lane_offset_results[0]
    assert lane.mode == "two_boundary"
    assert abs(lane.left_x - 149.5) <= 1.0 and abs(lane.right_x - 289.5) <= 1.0
    assert p2.stop_line_results[0].detected


@pytest.mark.software
@pytest.mark.parametrize("tape", [6, 20, 30])
def test_the_lane_stays_all_the_way_up_to_a_stop_line_that_touches_it(tape):
    """
    The approach, frame by frame, down to the robot on the line, with real
    tape widths. Wide tape still shifts the anchors (up to half a tape) while
    the line is in view: the piece below it has no top edge, so its sides
    trace apart. Pinned so the shift can't grow unnoticed.
    """
    pipeline = Pipeline(SCENE_CONFIG)
    for y in (5, 20, 35, 50, 65, 76):
        frame = scene(marks=(150, 290), mark_width=tape, stop_line=(150 - tape // 2 - 10, 290 + tape // 2 + 10, y),
                      stop_line_thickness=10)
        lane = pipeline.perceive(frame, 1, 50).lane_offset_results[0]
        assert lane.mode == "two_boundary", f"stop line at y={y}"
        assert abs(lane.left_x - 150) <= tape / 2 + 1 and abs(lane.right_x - 290) <= tape / 2 + 1, f"y={y}"


@pytest.mark.software
def test_the_lane_filter_off_brings_back_the_blind_lane():
    """The filter is what keeps the lane: without it the touching stop line blinds it again."""
    off = replace(SCENE_CONFIG, geometry=replace(SCENE_CONFIG.geometry,
                                                 lane=replace(SCENE_CONFIG.geometry.lane, horizontal_edge_deg=None)))
    p2 = Pipeline(off).perceive(SCENES["stop_line_touching"], 1, 50)
    assert p2.lane_offset_results[0].mode == "none" and p2.stop_line_results[0].detected


@pytest.mark.software
def test_a_horizontal_blob_too_short_for_a_stop_line_still_moves_the_offset():
    """
    Known limit, pinned so a change to it is deliberate: shorter than any
    stop line, so no stop line is found for lane_offset to skip it by, yet
    it passes the lane gates.
    """
    p2 = Pipeline(SCENE_CONFIG).perceive(SCENES["horizontal_blob"], 1, 50)
    assert not p2.stop_line_results[0].detected
    assert p2.lane_offset_results[0].right_x != 289.5
