"""
test_parallel.py  --  src/perception/parallel.py through Pipeline.perceive and run_chain

Parallel perception gives the same Phase 2 output as serial, frame by frame,
over the synthetic drive, under both tunings; color and obstacle really run
on the p2-side thread; the timings: branch_wait in place of color and
obstacle (so they still add up), the side thread's own times beside them,
every name known to stage_timing; an error on the side thread reaches the
caller; the switch: explicit beats the run's choice beats params, and
--parallel / --serial set the run's choice.

--software  No hardware.
"""
import argparse
import threading

import pytest

import src.perception.parallel as par
import src.pipeline as pipeline_mod
from src.analysis import stage_timing as st
from src.phase2_linker import run_chain
from src.pipeline import Pipeline
from src.tests.scenes import ALT_CONFIG, SCENE_CONFIG, drive_sequence, same


@pytest.fixture(autouse=True)
def no_run_choice():
    par.set_default(None)
    yield
    par.set_default(None)


@pytest.mark.software
@pytest.mark.parametrize("config", [SCENE_CONFIG, ALT_CONFIG], ids=["scene", "alt"])
def test_parallel_gives_the_same_phase_2_output_frame_by_frame(config):
    serial, parallel = Pipeline(config, parallel=False), Pipeline(config, parallel=True)
    for sf in drive_sequence():
        a = serial.perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
        b = parallel.perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
        same(a, b, f"{sf.segment} f{sf.frame_id}")


@pytest.mark.software
def test_the_debug_chain_matches_too_and_keeps_every_debug_dict():
    for sf in drive_sequence()[::7]:
        a = run_chain(sf.frame, sf.frame_id, sf.timestamp_ms, SCENE_CONFIG, trace=True, parallel=False)
        b = run_chain(sf.frame, sf.frame_id, sf.timestamp_ms, SCENE_CONFIG, trace=True, parallel=True)
        same(a.phase2, b.phase2, f"f{sf.frame_id}")
        assert set(a.traffic_debug) == set(b.traffic_debug) and set(a.obstacle_debug) == set(b.obstacle_debug)


@pytest.mark.software
def test_color_and_obstacle_run_on_the_side_thread(monkeypatch):
    seen = {}
    real_color, real_obstacle = pipeline_mod.detect_color, pipeline_mod.estimate_obstacle

    def color(*a, **k):
        seen["color"] = threading.current_thread().name
        return real_color(*a, **k)

    def obstacle(*a, **k):
        seen["obstacle"] = threading.current_thread().name
        return real_obstacle(*a, **k)
    monkeypatch.setattr(pipeline_mod, "detect_color", color)
    monkeypatch.setattr(pipeline_mod, "estimate_obstacle", obstacle)
    sf = drive_sequence()[0]
    Pipeline(SCENE_CONFIG, parallel=True).perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
    assert all(name.startswith(par.SIDE_THREAD_NAME) for name in seen.values()) and len(seen) == 2
    seen.clear()
    Pipeline(SCENE_CONFIG, parallel=False).perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
    assert set(seen.values()) == {threading.main_thread().name}


@pytest.mark.software
def test_the_timings_still_add_up_and_name_the_wait():
    sf = drive_sequence()[0]
    p = Pipeline(SCENE_CONFIG, timing=True, parallel=True)
    p.perceive(sf.frame, sf.frame_id, sf.timestamp_ms)
    assert list(p.last_timings_ms) == ["preprocess", "roi", "geometry", "lane_offset", "stop_line", "branch_wait",
                                       "fusion", "package"]
    assert set(p.last_side_ms) == {"color", "obstacle"} and all(v >= 0 for v in p.last_side_ms.values())
    chain = run_chain(sf.frame, sf.frame_id, sf.timestamp_ms, SCENE_CONFIG, parallel=True)
    assert "branch_wait" in chain.timings_ms and "color" not in chain.timings_ms
    assert set(chain.side_timings_ms) == {"color", "obstacle"}
    assert all(name in st.STAGE_ORDER and name not in st.EXTERNAL_STAGES for name in chain.timings_ms)
    assert run_chain(sf.frame, sf.frame_id, sf.timestamp_ms, SCENE_CONFIG, parallel=False).side_timings_ms == {}


@pytest.mark.software
def test_an_error_on_the_side_thread_reaches_the_caller(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("color broke")
    monkeypatch.setattr(pipeline_mod, "detect_color", broken)
    sf = drive_sequence()[0]
    with pytest.raises(RuntimeError, match="color broke"):
        Pipeline(SCENE_CONFIG, parallel=True).perceive(sf.frame, sf.frame_id, sf.timestamp_ms)


@pytest.mark.software
def test_explicit_beats_the_runs_choice_beats_params(monkeypatch):
    monkeypatch.setattr(par, "PARALLEL_PERCEPTION", False)
    assert par.enabled() is False
    par.set_default(True)
    assert par.enabled() is True and par.enabled(False) is False
    assert Pipeline(SCENE_CONFIG).parallel is True


@pytest.mark.software
@pytest.mark.parametrize("argv, choice", [([], None), (["--parallel"], True), (["--serial"], False)])
def test_the_command_line_sets_the_runs_choice(argv, choice, monkeypatch):
    monkeypatch.setattr(par, "PARALLEL_PERCEPTION", False)
    ap = argparse.ArgumentParser()
    par.add_arguments(ap)
    assert par.apply(ap.parse_args(argv)) is bool(choice)
    assert par._default is choice
    with pytest.raises(SystemExit):
        ap.parse_args(["--parallel", "--serial"])


@pytest.mark.software
def test_the_comparison_runs_every_mode_on_every_frame_and_reports_the_gain():
    from src.analysis import parallel_compare as pc
    frames = [(sf.frame, sf.frame_id, sf.timestamp_ms) for sf in drive_sequence()[:12]]
    modes = [("serial cv1", False, 1), ("parallel cv1", True, 1)]
    res = pc.measure(frames, modes, block=4, config=SCENE_CONFIG)
    assert {r["frames"] for r in res.values()} == {12}
    assert res["serial cv1"]["branch_wait_ms"] is None and res["parallel cv1"]["branch_wait_ms"]["p50"] >= 0
    assert set(res["parallel cv1"]["side_ms"]) == {"color", "obstacle"}
    text = "\n".join(pc.report(res))
    assert "parallel cv1:" in text and "ms per frame at p50 against serial" in text


@pytest.mark.software
def test_frames_are_loaded_into_memory_skipping_drops():
    from src.analysis import parallel_compare as pc
    items = iter([("a", 0, 0), (None, None, None), ("b", 1, 50), None])
    src = type("S", (), {"read": lambda self: next(items)})()
    assert pc.load_frames(src, 10) == [("a", 0, 0), ("b", 1, 50)]
