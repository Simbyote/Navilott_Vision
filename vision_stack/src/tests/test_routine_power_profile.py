"""
test_routine_power_profile.py  --  src/routines/power_profile.py

The CPU's busy share from /proc/stat; a stage's numbers from its samples
(sag against rest, drain as a fitted slope, CPU, heat, under-voltage and
throttling); the stages run on a fake clock with fake loads and readings:
every stage in order, sampled every SAMPLE_S for stage_s, a redo repeating
its stage, the sag against the latest rest, a load that fails, samples.csv
with each attempt, the ADC released; readings from the Pi's own sources;
no ADC; the criteria at their limits; the real loads with fake hardware.

--software  Fake clock, loads, ADC and hardware.
"""
import csv
import sys
import threading
import types

import pytest

import src.routines.harness as h
import src.routines.power_profile as pp
from src.diagnostics.battery import Power
from src.tests.test_routines import Person, no_conditions


@pytest.mark.software
def test_cpu_busy_from_proc_stat(tmp_path):
    p = tmp_path / "stat"
    p.write_text("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n")
    a = pp.read_cpu_times(p)
    assert a == (150, 1000)                                    # idle 800 + iowait 50 out of 1000
    p.write_text("cpu  200 0 100 850 50 0 0 0 0 0\n")
    assert pp.cpu_busy(a, pp.read_cpu_times(p)) == 75.0     # 150 more busy of 200 more
    assert pp.cpu_busy(a, a) is None and pp.cpu_busy(None, a) is None
    assert pp.read_cpu_times(tmp_path / "missing") is None


def sample(t, v=11.9, cpu=20.0, temp=50.0, uv=0, thr=0, stage="x"):
    return {"stage": stage, "t_s": t, "battery_v": v, "cpu_pct": cpu, "temp_c": temp, "under_voltage": uv,
            "throttled": thr}


@pytest.mark.software
def test_a_stages_row_sag_drain_cpu_heat_and_flags():
    s = [sample(t, v=11.9 - 0.001 * t, cpu=10.0 + t, temp=50.0 + t / 10) for t in range(0, 61, 10)]
    r = pp.stage_row("pipeline", s, rest_v=12.0)
    assert (r["stage"], r["seconds"], r["v_min"], r["sag_v"]) == ("pipeline", 60.0, 11.84, 0.13)
    assert r["drain_mv_min"] == pytest.approx(60.0)            # 1 mV/s falling
    assert (r["cpu_pct"], r["temp_max_c"], r["under_voltage"], r["throttled"]) == (40.0, 56.0, 0, 0)
    flagged = pp.stage_row("full", [sample(0, uv=1), sample(1, thr=1), sample(2, v=None)], None)
    assert (flagged["under_voltage"], flagged["throttled"], flagged["sag_v"]) == (1, 1, None)
    empty = pp.stage_row("rest", [sample(0, v=None, cpu=None, temp=None)], 12.0)
    assert (empty["v_mean"], empty["drain_mv_min"], empty["cpu_pct"], empty["temp_max_c"]) == (None, None, None, None)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


class Loads(dict):
    """A load per stage: records it ran and for how long it was asked to; fail makes one raise."""
    def __init__(self, fail=None):
        super().__init__({n: self._make(n) for n, _ in pp.STAGES})
        self.ran, self.fail = [], fail

    def _make(self, name):
        def load(ctx, seconds, stop):
            self.ran.append((name, seconds))
            if name == self.fail:
                raise OSError("no camera")
            stop.wait(5.0)
        return load


def reader(volts):
    """Scripted readings: the volts drop by the stage's load; nothing else changes."""
    state = {"v": iter(volts)}

    def read():
        return {"battery_v": next(state["v"], 11.0), "cpu_pct": 30.0, "temp_c": 55.0, "under_voltage": 0,
                "throttled": 0}
    return read


def routine(loads=None, volts=(), released=None):
    clock = Clock()
    r = pp.PowerProfile(loads=loads or Loads(), read=reader(volts), clock=clock, sleep=clock.sleep)
    if released is not None:
        r._power = type("P", (), {"cleanup": lambda s: released.append(1)})()
    return r


def go(r, person, tmp_path, options=None, trials=None):
    return h.run_routine(r, person.console(), tmp_path / "out", trials=trials, options=options or {"stage_s": 2},
                         conditions_fn=no_conditions)


ENTERS = tuple(x for _ in pp.STAGES for x in ("", ""))       # Enter to start each stage, Enter to keep it


@pytest.mark.software
def test_every_stage_runs_in_order_sampled_for_its_time(tmp_path):
    loads, released = Loads(), []
    volts = [12.0] * 4 + [11.9] * 4 + [11.8] * 4 + [11.6] * 4 + [11.5] * 4      # 4 samples a 2 s stage
    res = go(routine(loads, volts, released), Person(*ENTERS), tmp_path)
    assert [n for n, _ in loads.ran] == [n for n, _ in pp.STAGES] and loads.ran[0][1] == 2.0
    rows = __import__("json").loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert [(r["stage"], r["v_mean"], r["sag_v"]) for r in rows] == [
        ("rest", 12.0, None), ("camera", 11.9, 0.1), ("pipeline", 11.8, 0.2), ("motors", 11.6, 0.4),
        ("full", 11.5, 0.5)]
    assert res["verdict"] == h.PASS and released == [1]
    with open(tmp_path / "out" / "samples.csv") as f:
        samples = list(csv.DictReader(f))
    assert len(samples) == 20 and samples[0]["attempt"] == "1" and samples[-1]["stage"] == "full"
    assert res["state"]["samples_count"] == 20 and "samples" not in res["state"]


@pytest.mark.software
def test_a_redo_repeats_its_stage_and_the_sag_is_against_the_latest_rest(tmp_path):
    loads = Loads()
    volts = [11.0] * 4 + [12.0] * 4 + [11.7] * 4
    res = go(routine(loads, volts), Person("", "r", "", "", "", ""), tmp_path, trials=2)
    assert [n for n, _ in loads.ran] == ["rest", "rest", "camera"]
    rows = res["criteria"] and __import__("json").loads((tmp_path / "out" / "results.json").read_text())["rows"]
    assert [(r["stage"], r["sag_v"]) for r in rows] == [("rest", None), ("camera", 0.3)]
    with open(tmp_path / "out" / "samples.csv") as f:
        assert [s["attempt"] for s in csv.DictReader(f)][::4] == ["1", "2", "3"]


@pytest.mark.software
def test_a_failing_load_is_said_and_recorded_and_its_stage_still_sampled(tmp_path):
    p = Person("", "", "", "")
    res = go(routine(Loads(fail="camera"), [12.0] * 8), p, tmp_path, trials=2)
    assert "the camera load failed: OSError('no camera')" in p.said()
    assert res["state"]["load_errors"] == {"camera": "OSError('no camera')"} and res["trials_kept"] == 2


@pytest.mark.software
@pytest.mark.parametrize("v_min, uv, thr, passed", [
    (Power.VOLTAGE_WARNING, 0, 0, [True, True, True]),
    (Power.VOLTAGE_WARNING - 0.01, 0, 0, [False, True, True]),
    (11.0, 1, 0, [True, False, True]),
    (11.0, 0, 1, [True, True, False]),
    (None, 0, 0, [False, True, True]),
])
def test_the_criteria_at_their_limits(v_min, uv, thr, passed):
    rows = [{"stage": "rest", "v_min": 9.0, "under_voltage": 0, "throttled": 0},          # rest isn't a load
            {"stage": "full", "v_min": v_min, "under_voltage": uv, "throttled": thr}]
    assert [c["passed"] for c in pp.PowerProfile().judge(rows)] == passed


@pytest.mark.software
def test_readings_from_the_pis_own_sources(monkeypatch):
    import src.diagnostics.system_monitor as sm
    monkeypatch.setattr(sm, "sample", lambda: {"temp_c": 61.0, "under_voltage": True, "freq_capped": False,
                                               "throttled": False, "soft_temp_limit": True})
    times = iter([(100, 1000), (150, 1100)])
    monkeypatch.setattr(pp, "read_cpu_times", lambda: next(times))
    r = pp.PowerProfile()
    r._cpu_prev = pp.read_cpu_times()
    r._power = type("P", (), {"voltage_raw": lambda s: 11.8765})()
    assert r._read_pi() == {"battery_v": 11.877, "cpu_pct": 50.0, "temp_c": 61.0, "under_voltage": 1, "throttled": 1}
    r._power = type("P", (), {"voltage_raw": lambda s: (_ for _ in ()).throw(OSError("no ACK"))})()
    monkeypatch.setattr(pp, "read_cpu_times", lambda: None)
    assert r._read_pi()["battery_v"] is None


@pytest.mark.software
def test_without_the_adc_it_says_so_and_still_runs(tmp_path, monkeypatch):
    def no_adc():
        raise ImportError("No module named 'board'")
    r = pp.PowerProfile(loads=Loads(), power_factory=no_adc)
    p = Person()
    ctx = h.Context(p.console(), tmp_path)
    r.setup(ctx)
    assert "no battery ADC" in p.said() and r._power is None and r._read == r._read_pi
    assert ctx.options == {"stage_s": pp.STAGE_S, "duty": pp.BASE_SPEED}


# =============================================================================
# The real loads, on fake hardware
# =============================================================================

@pytest.mark.software
def test_the_motors_load_keeps_refreshing_the_duty_then_stops(monkeypatch):
    log = []

    class Motor:
        def __init__(self, pi):
            log.append("open")

        def drive(self, left, right):
            log.append(("drive", left, right))

        def stop(self):
            log.append("stop")
    monkeypatch.setitem(sys.modules, "pigpio", types.SimpleNamespace(
        pi=lambda: types.SimpleNamespace(stop=lambda: log.append("pi stop"))))
    import src.peripherals.drive as drive
    monkeypatch.setattr(drive, "MotorController", Motor)
    stop = threading.Event()
    ctx = types.SimpleNamespace(options={"duty": 0.35})
    t = threading.Thread(target=pp._motors, args=(ctx, 1.0, stop))
    t.start()
    while sum(1 for e in log if isinstance(e, tuple)) < 3:
        pass
    stop.set()
    t.join(2.0)
    assert log[0] == "open" and log[1] == ("drive", 0.35, 0.35) and log[-2:] == ["stop", "pi stop"]


@pytest.mark.software
def test_the_pipeline_load_runs_dry_until_the_stage_ends(monkeypatch, tmp_path):
    import src.linker_io as lio
    import src.navigation_linker as nl
    seen = {}
    monkeypatch.setattr(lio, "open_rig", lambda camera, motors, button: (seen.update(motors=motors) or ("s", "se", "m", None)))

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when):
        seen.update(out=out_dir, cap=max_run_s, motors_on=motors_on, render=render, before=stop_when({}))
        stop.set()
        seen["after"] = stop_when({})
    monkeypatch.setattr(nl, "run", run)
    stop = threading.Event()
    pp._pipeline(types.SimpleNamespace(out_dir=tmp_path), 30.0, stop)
    assert seen == {"motors": False, "out": str(tmp_path / "pipeline"), "cap": 35.0, "motors_on": False,
                    "render": False, "before": None, "after": "stage over"}


@pytest.mark.software
def test_rest_camera_and_full_loads(monkeypatch):
    stop = threading.Event()
    stop.set()
    pp._rest(None, 10.0, stop)                                  # returns at once once stopped
    import src.debugger.live_view as lv
    log = []

    class Cam:
        def __init__(self, w, h, fps, controls):
            log.append(("cam", w, h, fps))

        def read(self):
            log.append("read")
            stop.set()

        def close(self):
            log.append("closed")
    monkeypatch.setattr(lv, "CameraFrameSource", Cam)
    stop.clear()
    pp._camera(None, 1.0, stop)
    assert log[1:] == ["read", "closed"]
    calls = []
    monkeypatch.setattr(pp, "_motors", lambda ctx, s, st: calls.append("wheels") or st.wait(2.0))
    monkeypatch.setattr(pp, "_pipeline", lambda ctx, s, st: calls.append("pipeline"))
    stop.clear()
    pp._full(None, 1.0, stop)
    assert sorted(calls) == ["pipeline", "wheels"] and stop.is_set()
