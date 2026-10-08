"""
test_sched_latency.py  --  src/diagnostics/sched_latency.py

The Python probe on a fake clock (each wake-up's lateness, the grid kept,
no sleep when already late); cyclictest's command line per policy and its
-h output (several threads summed, the summary rows, a failure); the
kernel's preemption model; percentiles from a histogram with overflows;
each finding at its threshold; measure() without root, without cyclictest,
with a failing one and with both; the summary, --compare, the folder and
the figure.

--software  Canned cyclictest output and a fake clock. No root or rt-tests.
--hardware  5 s of the Python probe on the Pi, and cyclictest when root.
"""
import csv
import json
import os

import pytest

import src.diagnostics.sched_latency as sl

CYCLIC = """\
# /dev/cpu_dma_latency set to 0us
# Histogram
000000 000000\t000000
000005 000100\t000050
000006 000700\t000800
000009 000150\t000140
000041 000001\t000000
000052 000000\t000001
# Total: 000000951 000000991
# Min Latencies: 00005 00005
# Avg Latencies: 00006 00007
# Max Latencies: 00041 00052
# Histogram Overflows: 00000 00000
# Histogram Overflow at cycle number:
# Thread 0:
# Thread 1:
"""


class FakeClock:
    """perf_counter and sleep together: sleep advances the clock by the delay plus a scripted lateness."""
    def __init__(self, late_us):
        self.now, self.late, self.sleeps = 0.0, list(late_us), []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s + (self.late.pop(0) if self.late else 0.0) / 1e6


# =============================================================================
# Measuring
# =============================================================================

@pytest.mark.software
def test_the_python_probe_measures_each_wake_ups_lateness_on_a_fixed_grid():
    clock = FakeClock([50.0, 120.0, 30.0, 2000.0, 10.0])
    got = sl.python_latency(0.005, 1000, clock=clock, sleep=clock.sleep)
    # five 1 ms marks; the 4th woke 2 ms late, past the 5th, which is taken at once: 1000 us late
    assert [round(v, 3) for v in got] == [50.0, 120.0, 30.0, 2000.0, 1000.0]
    # the grid is kept: each sleep aims at the next 1 ms mark, not 1 ms after waking; none when already late
    assert [round(s * 1e6) for s in clock.sleeps] == [1000, 950, 880, 970]


@pytest.mark.software
def test_cyclictest_runs_on_every_core_at_each_policy():
    other = sl.cyclictest_argv("other", 30.0)
    assert other[:4] == ["cyclictest", "-m", "-q", "-S"] and "-i1000" in other and "-D30s" in other
    assert f"-h{sl.HIST_MAX_US}" in other and other[-2:] == ["-p0", "--policy=other"]
    assert sl.cyclictest_argv("fifo", 5.0, 500)[-2:] == [f"-p{sl.FIFO_PRIORITY}", "--policy=fifo"]


@pytest.mark.software
def test_cyclictest_output_is_summed_over_its_threads():
    p = sl.parse_cyclictest(CYCLIC)
    assert p["hist"] == {5: 150, 6: 1500, 9: 290, 41: 1, 52: 1}
    assert (p["total"], p["min"], p["avg"], p["max"], p["overflows"]) == (1942, 5, 6.5, 52, 0)
    assert sl.parse_cyclictest("cyclictest: unable to set priority\n") is None


@pytest.mark.software
def test_run_cyclictest_reports_why_it_couldnt():
    ok = lambda argv, **kw: type("P", (), {"returncode": 0, "stdout": CYCLIC, "stderr": ""})()      # noqa: E731
    parsed, raw = sl.run_cyclictest("fifo", 1.0, run=ok)
    assert parsed["max"] == 52 and raw == CYCLIC
    bad = lambda argv, **kw: type("P", (), {"returncode": 1, "stdout": "", "stderr": "mlockall failed"})()  # noqa: E731
    assert sl.run_cyclictest("fifo", 1.0, run=bad) == (None, "cyclictest failed (1): mlockall failed")

    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])
    assert "sudo apt install rt-tests" in sl.run_cyclictest("other", 1.0, run=missing)[1]


@pytest.mark.software
@pytest.mark.parametrize("version, model", [
    ("#1 SMP PREEMPT_RT Debian 6.6.51-1+rpt3", "PREEMPT_RT"),
    ("#1 SMP PREEMPT Debian 1:6.6.51-1+rpt3 (2024-10-08)", "PREEMPT"),
    ("#1 SMP PREEMPT_DYNAMIC Fri", "PREEMPT_DYNAMIC"),
    ("#1 SMP Fri", "none (voluntary or server)"),
])
def test_the_preemption_model_comes_from_uname_v(version, model):
    assert sl.preemption_model(version) == model


# =============================================================================
# Statistics and findings
# =============================================================================

@pytest.mark.software
def test_percentiles_from_a_histogram_with_overflows_as_late_as_the_max():
    hist = {k: 1 for k in range(1, 1001)}                       # 1..1000 us, one each
    s = sl.case_stats(hist)
    assert (s["n"], s["min"], s["p50"], s["p99"], s["p999"], s["max"]) == (1000, 1, 500, 990, 999, 1000)
    tail = sl.case_stats({5: 997}, overflows=3, max_us=25000)    # 0.3% past the histogram
    assert (tail["p99"], tail["p999"], tail["max"], tail["n"]) == (5, 25000, 25000, 1000)
    assert sl.case_stats({}) is None
    assert sl.hist_of([0.2, 1.9, 1.1, 7.0]) == {0: 1, 1: 2, 7: 1}


def res_of(**cases):
    full = lambda m, p99=None: {"n": 1000, "min": 3, "p50": 8, "p99": p99 if p99 is not None else m // 2,   # noqa: E731
                                "p999": m, "max": m, "overflows": 0}
    return {"cases": {k: full(*v) if isinstance(v, tuple) else full(v) for k, v in cases.items()},
            "preemption": "PREEMPT_RT"}


@pytest.mark.software
def test_a_worst_case_over_a_tenth_of_the_imu_period_is_a_finding_and_over_it_a_missed_tick():
    at = sl.findings(res_of(python=1000))
    assert at == ["python: the worst wake-up was 1.0 ms late, 10% of the IMU's 10 ms period and 2% of the 50 ms frame"]
    assert sl.findings(res_of(python=999)) == ["every case stayed under 1 ms, a tenth of the tightest deadline (the "
                                               "IMU's): Linux's jitter is small against the robot's soft deadlines"]
    missed = sl.findings(res_of(python=12000))[0]
    assert "12.0 ms late, 120% of the IMU's" in missed and missed.endswith("a 100 Hz loop would miss a tick")


@pytest.mark.software
def test_real_time_priority_and_python_overheads_are_findings_at_their_factor():
    f = sl.findings(res_of(other=(600, 40), fifo=(300, 20)))
    assert any("real-time priority cut the kernel's worst case 2x (600 -> 300 us)" in x for x in f)
    assert not any("real-time" in x for x in sl.findings(res_of(other=(599, 40), fifo=(300, 20))))
    g = sl.findings(res_of(python=(800, 80), other=(600, 40)))
    assert any("Python adds to the kernel's wake-up: p99 80 us against cyclictest's 40 us" in x for x in g)
    assert not any("Python adds" in x for x in sl.findings(res_of(python=(800, 79), other=(600, 40))))


@pytest.mark.software
def test_a_kernel_without_preempt_rt_says_an_rt_kernel_would_help():
    r = res_of(python=100)
    r["preemption"] = "PREEMPT"
    assert sl.findings(r)[-1].startswith("the kernel isn't PREEMPT_RT")


# =============================================================================
# measure(), output and the command line
# =============================================================================

def fake_python(seconds, interval_us):
    return [10.0] * 990 + [500.0] * 9 + [1500.0]


@pytest.mark.software
def test_measure_without_root_runs_python_only_and_says_why():
    res = sl.measure(1.0, "idle", root=False, python_fn=fake_python)
    assert set(res["cases"]) == {"python"} and res["cases"]["python"]["max"] == 1500
    assert res["skipped"] == {"other": "needs root (sudo)", "fifo": "needs root (sudo)"}


@pytest.mark.software
def test_measure_with_root_runs_cyclictest_at_both_policies_and_keeps_a_failure(monkeypatch):
    calls = []

    def cyclic(policy, seconds, interval_us):
        calls.append(policy)
        return (sl.parse_cyclictest(CYCLIC), "") if policy == "other" else (None, "cyclictest failed (1): x")
    res = sl.measure(2.0, "loaded", root=True, python_fn=fake_python, cyclic_fn=cyclic)
    assert calls == ["other", "fifo"]
    assert res["cases"]["other"]["max"] == 52 and res["cases"]["other"]["p50"] == 6
    assert res["skipped"] == {"fifo": "cyclictest failed (1): x"}
    monkeypatch.setattr(sl.shutil, "which", lambda name: None)
    none = sl.measure(1.0, root=True, python_fn=fake_python)
    assert none["skipped"]["other"].startswith("cyclictest isn't installed")


@pytest.mark.software
def test_write_summary_compare_and_cli(tmp_path, monkeypatch, capsys):
    cyclic = lambda policy, s, i: (sl.parse_cyclictest(CYCLIC), "")          # noqa: E731
    idle = sl.measure(1.0, "idle", root=True, python_fn=fake_python, cyclic_fn=cyclic)
    lines = sl.write(tmp_path / "idle", idle, draw=False)
    text = "\n".join(lines)
    assert "python     1000      10      10      10      500     1500     3.0%    15.0%" in text
    assert "other      1942       5       6       9       41       52     0.1%     0.5%" in text
    assert "single-digit to tens of microseconds" in text
    with open(tmp_path / "idle" / "histogram.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["us_late", "python", "other", "fifo"] and ["5", "0", "150", "150"] in rows
    saved = json.loads((tmp_path / "idle" / "sched_latency.json").read_text())
    assert "hist" not in saved and saved["cases"]["fifo"]["max"] == 52
    loaded = {**idle, "label": "loaded", "cases": {**idle["cases"], "python": {**idle["cases"]["python"], "p99": 900,
                                                                               "max": 4000}}}
    sl.write(tmp_path / "loaded", loaded, draw=False)
    assert sl.cli(["--compare", str(tmp_path / "idle"), str(tmp_path / "loaded")]) == 0
    out = capsys.readouterr().out
    assert "idle vs loaded" in out and "python            10          900         1500         4000" in out
    assert sl.cli(["--compare", str(tmp_path / "idle"), str(tmp_path / "nope")]) == 2
    monkeypatch.setattr(sl, "measure", lambda s, label: {**idle, "label": label})
    assert sl.cli(["--label", "x", "--out", str(tmp_path / "x"), "--no-figure"]) == 0
    assert (tmp_path / "x" / "summary.txt").read_text().startswith("scheduling latency  x")


@pytest.mark.software
def test_the_tail_is_the_share_of_wake_ups_at_least_each_lateness():
    xs, ys = sl.tail({0: 50, 5: 40, 300: 9, 2000: 1})
    assert xs == [1, 5, 300, 2000]                                  # 0 us drawn at 1 on the log axis
    assert ys == [1.0, 0.5, 0.1, 0.01]


@pytest.mark.software
def test_the_figure_draws_each_case_and_skips_without_data(tmp_path):
    pytest.importorskip("matplotlib")
    res = sl.measure(1.0, "idle", root=True, python_fn=fake_python,
                     cyclic_fn=lambda p, s, i: (sl.parse_cyclictest(CYCLIC), ""))
    png = sl.figure(res, tmp_path / "f.png")
    assert png.exists() and png.stat().st_size > 10_000
    assert sl.figure({"hist": {}, "label": ""}, tmp_path / "g.png") is None


# =============================================================================
# On the Pi
# =============================================================================

@pytest.mark.hardware
def test_scheduling_latency_on_the_pi(artifacts):
    res = sl.measure(5.0, "hw-test")
    sl.write(artifacts.path / "sched", res, draw=False)
    assert res["cases"]["python"]["n"] >= 4000
    if os.geteuid() == 0 and "other" not in res["skipped"]:
        assert res["cases"]["other"]["n"] > 0
