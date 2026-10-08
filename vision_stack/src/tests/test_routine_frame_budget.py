"""
test_routine_frame_budget.py  --  src/routines/frame_budget.py

The nearest-rank p95; BudgetWatch over generated nav.csv rows: Phases 2+3
against the budget, latency, the frame rate over the rows' span, ending at
the time; the Sampler on a thread with fake readers: mean and max CPU,
largest memory, hottest reading, a throttle flag new during the run (or
set at the end) counted and one latched before it not, no vcgencmd; the
routine on a fake rig, run and sampler: one run per trial with the motors
off, the row, the rig closed when stopping at Enter, the sampler stopped
when the run fails; each criterion at its limit.

--software  A scripted tester, a fake rig, run and sampler.
"""
import json
import time

import pytest

import src.routines.frame_budget as fb
import src.routines.harness as h
from src.routines import ROUTINES
from src.tests.test_routines import Person, no_conditions


def rows(seconds=10.0, fps=25.0, work=30.0, latency=40.0, slow_every=0, slow_work=60.0):
    n = int(seconds * fps) + 1
    return [{"t": round(k / fps, 4), "phase2_ms": (slow_work if slow_every and k % slow_every == 0 else work) - 1.0,
             "phase3_ms": 1.0, "latency_ms": latency} for k in range(n)]


def feed(watch, rs):
    for r in rs:
        why = watch(r)
        if why:
            return why
    return None


# =============================================================================
# The numbers
# =============================================================================

@pytest.mark.software
def test_p95_nearest_rank():
    assert fb.p95(list(range(1, 101))) == 95 and fb.p95([3.0]) == 3.0 and fb.p95([]) is None
    assert fb.p95(list(range(1, 21))) == 19 and fb.p95([5, 1, 3]) == 5


@pytest.mark.software
def test_the_watch_counts_budget_latency_and_rate():
    w = fb.BudgetWatch(seconds=4.0)
    assert feed(w, rows(seconds=10.0, slow_every=10)) == fb.TIME_UP
    assert w.frames == 101 and w.t == 4.0 and w.fps() == 25.0
    assert w.over_budget_pct() == pytest.approx(round(100 * 11 / 101, 1))
    assert max(w.work_ms) == 60.0 and w.latency_ms[0] == 40.0
    late = fb.BudgetWatch(seconds=1e9)
    feed(late, [{**r, "t": r["t"] + 100.0} for r in rows(seconds=2.0)])
    assert late.fps() == 25.0                                     # over the rows' own span


@pytest.mark.software
def test_the_budget_is_strictly_over_and_missing_values_skip():
    w = fb.BudgetWatch(seconds=100.0)
    feed(w, [{"t": 0.0, "phase2_ms": 49.0, "phase3_ms": 1.0, "latency_ms": ""},
             {"t": 0.1, "phase2_ms": "", "phase3_ms": 1.0, "latency_ms": 10},
             {"t": 0.2, "phase2_ms": 49.5, "phase3_ms": 1.0, "latency_ms": "x"}])
    assert w.frames == 2 and w.over_budget_pct() == 50.0 and w.latency_ms == [10.0]
    assert fb.BudgetWatch(1.0).fps() is None and fb.BudgetWatch(1.0).over_budget_pct() is None
    one = fb.BudgetWatch(1.0)
    one({"t": 0.0, "phase2_ms": 1, "phase3_ms": 1})
    assert one.fps() is None


class Reader:
    def __init__(self, samples):
        self.samples = list(samples)

    def __call__(self):
        return self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]


def flags(before, after):
    seq = iter([before, after])
    return lambda: next(seq)


NONE = {"throttled": 0, "throttled_occurred": 0, "freq_capped": 0, "freq_capped_occurred": 0,
        "soft_temp_limit": 0, "soft_temp_limit_occurred": 0, "under_voltage": 0, "under_voltage_occurred": 0}


@pytest.mark.software
def test_the_sampler_reads_on_its_thread():
    s = fb.Sampler(read=Reader([{"cpu_pct": 100.0, "rss_mb": 150.0, "temp_c": 60.0},
                                {"cpu_pct": 50.0, "rss_mb": 180.04, "temp_c": 62.5},
                                {"cpu_pct": 60.0, "rss_mb": 170.0, "temp_c": 61.0},
                                {"cpu_pct": None, "rss_mb": None, "temp_c": None}]),
                   throttled=flags(NONE, NONE), period_s=0.001).start()
    time.sleep(0.05)
    out = s.stop()
    assert len(s.samples) >= 4
    assert (out["cpu_mean_pct"], out["cpu_max_pct"], out["rss_max_mb"], out["temp_max_c"]) == (70.0, 100.0, 180.0, 62.5)
    assert out["throttled"] == 0 and out["under_voltage"] == 0
    n = len(s.samples)
    time.sleep(0.01)
    assert len(s.samples) == n                  # stopped


@pytest.mark.software
def test_the_sampler_with_no_samples():
    out = fb.Sampler(read=Reader([{}]), throttled=flags({}, {}), period_s=60.0).start().stop()
    assert out == {"cpu_mean_pct": None, "cpu_max_pct": None, "rss_max_mb": None, "temp_max_c": None,
                   "throttled": None, "under_voltage": None}


@pytest.mark.software
@pytest.mark.parametrize("before, after, throttled, under", [
    (NONE, {**NONE, "freq_capped_occurred": 1}, 1, 0),                                  # new during the run
    ({**NONE, "throttled_occurred": 1}, {**NONE, "throttled_occurred": 1}, 0, 0),        # latched before it
    ({**NONE, "throttled_occurred": 1}, {**NONE, "throttled": 1, "throttled_occurred": 1}, 1, 0),   # on now
    (NONE, {**NONE, "under_voltage_occurred": 1}, 0, 1),
    (NONE, {**NONE, "soft_temp_limit_occurred": 1}, 1, 0),
])
def test_throttling_new_during_the_run(before, after, throttled, under):
    out = fb.Sampler(read=Reader([{}]), throttled=flags(before, after), period_s=60.0).start().stop()
    assert (out["throttled"], out["under_voltage"]) == (throttled, under)


# =============================================================================
# The routine
# =============================================================================

class Rig:
    def __init__(self, log):
        self.log = log

    def __call__(self, camera, motors, button):
        self.log.append(("open", camera, motors, button))
        log = self.log
        return (type("S", (), {"close": lambda s: log.append("source closed")})(),
                type("Se", (), {"stop": lambda s: log.append("sensors stopped")})(),
                type("M", (), {"stop": lambda s: log.append("motors stopped")})(), None)


def fake_run(scripts, log, fail=False):
    attempts = iter(scripts)

    def run(source, sensors, motor, nav, config, p3, out_dir, system, max_run_s, motors_on, render, stop_when):
        log.append(("run", out_dir.rsplit("/", 1)[-1], max_run_s, motors_on, render))
        if fail:
            raise RuntimeError("camera gone")
        return {"ended_by": feed(stop_when, next(attempts)) or "ended early (lane lost)"}
    return run


GOOD = {"cpu_mean_pct": 55.0, "cpu_max_pct": 80.0, "rss_max_mb": 210.0, "temp_max_c": 61.0, "throttled": 0,
        "under_voltage": 0}


def sampler_of(log, result=GOOD):
    class FakeSampler:
        def start(self):
            log.append("sampling")
            return self

        def stop(self):
            log.append("sampled")
            return dict(result)
    return FakeSampler


def go(scripts, person, tmp_path, log=None, result=GOOD, fail=False, **options):
    log = [] if log is None else log
    r = fb.FrameBudget(open_rig=Rig(log), navigation_run=fake_run(scripts, log, fail),
                       sampler=sampler_of(log, result))
    return h.run_routine(r, person.console(), tmp_path / "out", options=options, conditions_fn=no_conditions)


@pytest.mark.software
def test_three_runs_pass(tmp_path):
    log = []
    res = go([rows(seconds=12)] * 3, Person(*[""] * 6), tmp_path, log, seconds=10)
    assert res["verdict"] == h.PASS, res["criteria"]
    assert log[:4] == [("open", True, False, False), "sampling", ("run", "attempt_01", 20.0, False, False), "sampled"]
    row = json.loads((tmp_path / "out" / "results.json").read_text())["rows"][0]
    assert (row["seconds"], row["frames"], row["fps"], row["over_budget_pct"]) == (10.0, 251, 25.0, 0.0)
    assert (row["work_p95_ms"], row["latency_p95_ms"], row["cpu_mean_pct"], row["rss_max_mb"]) == (30.0, 40.0, 55.0, 210.0)
    assert row["ended_by"] == fb.TIME_UP and len(res["criteria"]) == 18
    assert res["criteria"][0]["name"] == "frames per second (run 1)"


@pytest.mark.software
def test_the_default_run_length(tmp_path):
    log = []
    go([rows(seconds=61)], Person("", ""), tmp_path, log)
    assert next(e for e in log if e[0] == "run")[2] == fb.SECONDS + 10.0 and fb.SECONDS == 60.0


@pytest.mark.software
def test_stopping_at_enter_closes_the_rig_and_never_samples(tmp_path):
    log = []
    res = go([], Person("q"), tmp_path, log)
    assert res["stopped_early"] and log[1:] == ["source closed", "sensors stopped", "motors stopped"]


@pytest.mark.software
def test_a_failed_run_still_stops_the_sampler(tmp_path):
    log = []
    with pytest.raises(RuntimeError):
        go([], Person(""), tmp_path, log, fail=True)
    assert log[-1] == "sampled"


@pytest.mark.software
def test_registered():
    assert ROUTINES["frame-budget"] is fb.FrameBudget and fb.FrameBudget.trials == 3
    assert (fb.TARGET_FPS, fb.BUDGET_MS, fb.LATENCY_MS, fb.CPU_MAX_PCT, fb.RSS_MAX_MB) == (20.0, 50.0, 50.0, 70.0, 400.0)


# =============================================================================
# The criteria
# =============================================================================

def one(tmp_path, script, result=GOOD):
    res = go([script], Person("", ""), tmp_path, result=result, seconds=10)
    return [c["name"] for c in res["criteria"] if not c["passed"]]


@pytest.mark.software
@pytest.mark.parametrize("script, result, failing", [
    (rows(seconds=12, fps=20.0), GOOD, []),
    (rows(seconds=12, fps=19.5), GOOD, ["frames per second"]),
    (rows(seconds=12, slow_every=25), GOOD, []),                                  # 11 of 251: 4.4%
    (rows(seconds=12, slow_every=10), GOOD, ["frames over the 50 ms budget"]),
    (rows(seconds=12, latency=50.0), GOOD, []),
    (rows(seconds=12, latency=50.1), GOOD, ["p95 frame to motor command"]),
    (rows(seconds=12), {**GOOD, "cpu_mean_pct": 70.0}, []),
    (rows(seconds=12), {**GOOD, "cpu_mean_pct": 70.1}, ["CPU busy, mean"]),
    (rows(seconds=12), {**GOOD, "rss_max_mb": 400.0}, []),
    (rows(seconds=12), {**GOOD, "rss_max_mb": 400.1}, ["memory, largest"]),
    (rows(seconds=8), GOOD, ["ran the whole time"]),
    (rows(seconds=12), {**GOOD, "cpu_mean_pct": None}, ["CPU busy, mean"]),
])
def test_each_criterion(tmp_path, script, result, failing):
    assert one(tmp_path, script, result) == failing
