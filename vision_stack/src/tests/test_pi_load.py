"""
test_pi_load.py  --  src/analysis/pi_load.py

Recordings are written with diagnostics.monitor.write() itself, so the two
tools can't drift apart in format. Known answers: the process's total CPU,
main's p95 and the busiest thread's share; core loads; flags latched since
boot; every finding on its own and none on a clean recording; lining a
run's frames up on the monotonic clock and attributing its slow frames to a
clock drop, a busy main thread or neither; and the command line.

--software  CSV and JSON files in a temp folder. No Pi needed.
"""
import csv
import json

import pytest

import src.analysis.pi_load as pl
import src.diagnostics.monitor as mon
from src.diagnostics.system_monitor import THROTTLE_BITS
from src.navigation_linker import NAV_FIELDS

STEP = 0.5          # sampling interval of the hand-made recordings


def trow(t, tid, name, cpu, core=0, vol=10.0, invol=1.0):
    return {"elapsed_s": t, "tid": tid, "name": name, "state": "S", "core": core, "cpu_pct": cpu,
            "user_pct": cpu, "sys_pct": 0.0, "vol_ctx_s": vol, "invol_ctx_s": invol}


def srow(t, **kw):
    r = {"elapsed_s": t, "temp_c": 55.0, "cpu_mhz": 1000.0, "throttled_raw": "0x0", "rss_mb": 150.0,
         "mem_available_mb": 250.0, "load_1m": 1.0}
    r.update({k: 0 for k in THROTTLE_BITS})
    r.update({f"{k}_occurred": 0 for k in THROTTLE_BITS})
    r.update(kw)
    return {k: r.get(k) for k in mon.SYSTEM_COLUMNS}


def recording(tmp_path, n=20, main=50.0, hub=2.0, gst=20.0, hub_vol=100.0, main_invol=1.0, main_core=None,
              cores=(40.0, 30.0, 20.0, 10.0), system=None, t0=1000.0):
    """A diagnostics folder: main, sensor-hub and a GStreamer thread over n samples, written by monitor.write()."""
    threads, core_rows = [], []
    for i in range(1, n + 1):
        t = round(i * STEP, 3)
        threads += [trow(t, 100, "main", main if not callable(main) else main(i),
                         core=main_core(i) if main_core else 0, invol=main_invol),
                    trow(t, 101, "sensor-hub", hub, core=1, vol=hub_vol),
                    trow(t, 102, "src", gst, core=2)]
        core_rows += [{"elapsed_s": t, "core": c, "busy_pct": b}
                      for c, b in enumerate(cores(i) if callable(cores) else cores)]
    rec = {"threads": threads, "cores": core_rows,
           "system": system if system is not None else [srow(float(s)) for s in range(int(n * STEP) + 1)],
           "interrupted": False, "elapsed_s": n * STEP, "t0_monotonic": t0}
    folder = tmp_path / "diag"
    mon.write(str(folder), {"pid": 100, "command": "python3 -m src.main", "interval_s": STEP, "cores": 4}, rec)
    return folder


def run(folder, **kw):
    d = pl.load(str(folder), kw.pop("run", None))
    return pl.analyze(d["threads"], d["cores"], d["system"], d["nav"], d["meta"].get("t0_monotonic"), d["nav_t0"])


# =============================================================================
# The numbers
# =============================================================================

@pytest.mark.software
def test_process_load_core_load_and_the_busiest_thread(tmp_path):
    res = run(recording(tmp_path, main=lambda i: 40.0 if i <= 10 else 80.0))
    p = res["process"]
    assert p["samples"] == 20 and p["total_cpu_pct"]["mean"] == pytest.approx(60.0 + 22.0)
    assert p["main_cpu_p95"] == pytest.approx(80.0) and p["busiest"] == "main"
    assert p["main_share"] == p["busiest_share"] == pytest.approx(60.0 / 82.0, abs=1e-3)
    assert res["cores"] == {"0": {"mean": 40.0, "max": 40.0}, "1": {"mean": 30.0, "max": 30.0},
                            "2": {"mean": 20.0, "max": 20.0}, "3": {"mean": 10.0, "max": 10.0}}
    assert [t["name"] for t in res["threads"]] == ["main", "src", "sensor-hub"]


@pytest.mark.software
def test_one_spike_is_not_cpu_bound_the_p95_is_what_counts(tmp_path):
    res = run(recording(tmp_path, main=lambda i: 100.0 if i == 7 else 40.0))
    assert res["process"]["main_cpu_p95"] < 50.0
    assert not any("CPU-bound" in f for f in res["findings"])


@pytest.mark.software
def test_the_busiest_thread_is_named_whichever_it_is(tmp_path):
    res = run(recording(tmp_path, main=5.0, gst=90.0))
    assert res["process"]["busiest"] == "src"
    assert any("one thread (src)" in f for f in res["findings"]), res["findings"]


@pytest.mark.software
def test_threads_sharing_a_name_are_summed_per_sample():
    rows = [trow(0.5, 7, "pigpio-cb", 3.0), trow(0.5, 8, "pigpio-cb", 4.0), trow(1.0, 7, "pigpio-cb", 1.0)]
    times, cpu = pl.by_sample(rows)
    assert times.tolist() == [0.5, 1.0] and cpu["pigpio-cb"].tolist() == [7.0, 1.0]


@pytest.mark.software
def test_a_clean_recording_has_no_findings_and_a_short_one_only_a_note(tmp_path):
    res = run(recording(tmp_path))
    assert res["findings"] == [] and any("too short to judge memory" in n for n in res["notes"])


# =============================================================================
# Findings
# =============================================================================

def throttled_system(n=10, **kw):
    return [srow(float(s), **kw) for s in range(n + 1)]


@pytest.mark.software
@pytest.mark.parametrize("kw, words", [
    ({"main": 95.0}, "CPU-bound"),
    ({"main": 90.0, "gst": 5.0}, "mostly serial"),
    ({"hub_vol": 50.0}, "sensor-hub woke 50"),
    ({"main_invol": 80.0}, "preempted 80"),
    ({"main_core": lambda i: i % 2}, "changes core"),
    ({"cores": (95.0, 5.0, 5.0, 5.0)}, "unevenly loaded"),
    ({"system": throttled_system(temp_c=77.0)}, "77.0 C"),
    ({"system": [srow(float(s), cpu_mhz=600.0 if s > 5 else 1000.0) for s in range(11)]}, "clock dropped to 600"),
    ({"system": throttled_system(under_voltage=1, under_voltage_occurred=1)}, "throttling: under voltage"),
    ({"system": throttled_system(under_voltage_occurred=1)}, "latched since boot, not during this run: under_voltage"),
])
def test_each_finding_on_its_own(tmp_path, kw, words):
    found = run(recording(tmp_path, **kw))["findings"]
    assert any(words in f for f in found), found


@pytest.mark.software
def test_a_flag_active_during_the_run_isnt_also_reported_as_only_latched(tmp_path):
    found = run(recording(tmp_path, system=throttled_system(under_voltage=1, under_voltage_occurred=1)))["findings"]
    assert not any("latched since boot" in f for f in found)


# =============================================================================
# Lining up a run
# =============================================================================

def nav_run(tmp_path, t0, intervals_ms):
    """A navigation run folder: nav.csv frames at the given intervals, report.json with its t0."""
    folder = tmp_path / "nav"
    folder.mkdir()
    t, rows = 0.0, []
    for i, iv in enumerate(intervals_ms):
        t += iv / 1000.0
        rows.append({**{f: "" for f in NAV_FIELDS}, "frame_id": i, "t": round(t, 4)})
    with open(folder / "nav.csv", "w", newline="") as f:
        w = csv.DictWriter(f, NAV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    (folder / "report.json").write_text(json.dumps({"run": {"t0_monotonic": t0}}))
    return folder


@pytest.mark.software
@pytest.mark.parametrize("cause", ["clock", "cpu", "neither"])
def test_slow_frames_are_put_down_to_what_they_coincided_with(tmp_path, cause):
    # The recording starts at 1000.0 s, the run 1 s later; frames every 50 ms, slow (120 ms) from 6 s to 8 s
    intervals = [50.0] * 100 + [120.0] * 17 + [50.0] * 40
    slow_window = lambda t: 6.0 <= t <= 8.5                                         # noqa: E731
    system = [srow(float(s), cpu_mhz=600.0 if cause == "clock" and slow_window(s) else 1000.0) for s in range(13)]
    main = (lambda i: 95.0 if slow_window(i * STEP) else 40.0) if cause == "cpu" else 40.0
    folder = recording(tmp_path, n=24, main=main, system=system, t0=1000.0)
    res = run(folder, run=str(nav_run(tmp_path, 1001.0, intervals)))
    a = res["aligned"]
    assert a["frames"] > 140 and a["late"] == 17
    words = {"clock": "clock drops", "cpu": "busy frame loop", "neither": "neither a clock drop"}[cause]
    assert any(words in f for f in res["findings"]), res["findings"]
    if cause == "cpu":
        assert a["late_main_cpu"] == pytest.approx(95.0) and a["normal_main_cpu"] < 50.0


# Startup: imports and the camera opening load main for the recording's first
# 6 s; the run starts at 7 s (t0 = 1007) and lasts 12 s; the recording goes on 3 s more
STARTUP_S, RUN_START_S, RUN_S, RECORDING_N = 6.0, 7.0, 12.0, 44


def startup_recording(tmp_path, **kw):
    startup = lambda i: i * STEP <= STARTUP_S                                       # noqa: E731
    args = dict(n=RECORDING_N, main=lambda i: 100.0 if startup(i) else 40.0,
                cores=lambda i: (95.0, 5.0, 5.0, 5.0) if startup(i) else (40.0, 35.0, 30.0, 25.0),
                system=[srow(float(s), cpu_mhz=600.0 if s <= STARTUP_S else 1000.0, rss_mb=50.0 + s)
                        for s in range(int(RECORDING_N * STEP) + 1)])
    args.update(kw)
    return recording(tmp_path, t0=1000.0, **args)


@pytest.mark.software
def test_with_the_run_its_own_time_is_judged_and_startup_left_out(tmp_path):
    folder = startup_recording(tmp_path)
    alone = run(folder)
    assert any("CPU-bound" in f for f in alone["findings"]) and any("clock dropped" in f for f in alone["findings"])
    assert alone["cores"]["0"]["max"] == 95.0                                      # startup's core 0 is in it
    res = run(folder, run=str(nav_run(tmp_path, 1000.0 + RUN_START_S, [50.0] * int(RUN_S * 20))))
    w = res["window"]
    assert w["used"] and (w["start_s"], w["end_s"]) == pytest.approx((RUN_START_S, RUN_START_S + RUN_S), abs=0.06)
    assert w["samples"] == pytest.approx(RUN_S / STEP, abs=1) and w["recording_s"] == RECORDING_N * STEP
    assert res["process"]["main_cpu_p95"] == pytest.approx(40.0)
    main = next(t for t in res["threads"] if t["name"] == "main")
    assert main["cpu_mean"] == pytest.approx(40.0) and main["samples"] == w["samples"]
    assert res["cores"]["0"] == {"mean": 40.0, "max": 40.0}
    assert res["system"]["cpu_mhz"]["min"] == 1000.0 and res["system"]["duration_s"] == pytest.approx(RUN_S, abs=1.0)
    assert res["system"]["memory"]["rss_start_mb"] >= 50.0 + RUN_START_S, "memory from the run's start, not boot"
    for words in ("CPU-bound", "clock dropped", "unevenly loaded"):
        assert not any(words in f for f in res["findings"]), (words, res["findings"])
    assert report_text(res).startswith("window    the run's own 12.0 s (7.0-19.0 s of the 22.0 s recording")


@pytest.mark.software
def test_a_run_too_short_for_its_own_window_is_judged_over_the_whole_recording_and_says_so(tmp_path):
    folder = startup_recording(tmp_path)
    res = run(folder, run=str(nav_run(tmp_path, 1000.0 + RUN_START_S, [50.0] * 20)))     # 1 s: 2 samples
    assert not res["window"]["used"] and res["window"]["samples"] < pl.MIN_WINDOW_SAMPLES
    assert any("CPU-bound" in f for f in res["findings"])
    assert any("judged over the whole recording" in n for n in res["notes"])
    assert not report_text(res).startswith("window")


@pytest.mark.software
def test_the_figure_shades_the_run_judged(tmp_path):
    pytest.importorskip("matplotlib")
    folder = startup_recording(tmp_path)
    nav = nav_run(tmp_path, 1000.0 + RUN_START_S, [50.0] * int(RUN_S * 20))
    assert pl.main([str(folder), "--run", str(nav)]) == 0
    assert (folder / "pi_load.png").is_file()
    assert json.loads((folder / "pi_load.json").read_text())["window"]["used"]


# =============================================================================
# How the threads share the Pi
# =============================================================================

@pytest.mark.software
def test_python_and_native_threads_are_told_apart_by_name():
    for name in ("main", "sensor-hub", "frame-recorder", "motor-watchdog", "system-monitor", "pigpio-cb"):
        assert pl.is_python_thread(name), name
    for name in ("task0", "CameraManager", "IPAProxyRPi", "python3", "pool-1", "python3-ust"):
        assert not pl.is_python_thread(name), name


@pytest.mark.software
def test_the_split_sums_each_side_per_sample():
    rows = [trow(0.5, 1, "main", 60.0), trow(0.5, 2, "sensor-hub", 5.0), trow(0.5, 3, "task0", 30.0),
            trow(0.5, 4, "python3", 20.0), trow(0.5, 5, "python3", 15.0),
            trow(1.0, 1, "main", 90.0), trow(1.0, 3, "task0", 10.0)]
    split = pl.python_native(rows)
    assert split["t"].tolist() == [0.5, 1.0]
    assert split["python"].tolist() == [65.0, 90.0] and split["native"].tolist() == [65.0, 10.0]
    assert split["python_threads"] == ["main", "sensor-hub"] and split["native_threads"] == ["python3", "task0"]


@pytest.mark.software
def test_lanes_follow_each_thread_s_core_busiest_first_and_leave_out_idle_ones():
    rows = [trow(0.5, 1, "main", 60.0, core=0), trow(1.0, 1, "main", 70.0, core=2), trow(1.5, 1, "main", 65.0, core=2),
            trow(0.5, 7, "python3", 20.0, core=1), trow(1.0, 7, "python3", 25.0, core=1),
            trow(0.5, 8, "python3", 30.0, core=3), trow(1.0, 8, "python3", 30.0, core=3),
            trow(0.5, 9, "pool-1", 0.0), trow(1.0, 9, "pool-1", 0.4)]
    lanes = pl.thread_lanes(rows)
    assert lanes["step"] == 0.5 and lanes["idle"] == 1
    assert [lane["label"] for lane in lanes["lanes"]] == ["main", "python3 8", "python3 7"]   # by total CPU
    main = lanes["lanes"][0]
    assert main["python"] and main["t"].tolist() == [0.5, 1.0, 1.5] and main["core"].tolist() == [0, 2, 2]
    assert main["cpu"].tolist() == [60.0, 70.0, 65.0] and not lanes["lanes"][1]["python"]


@pytest.mark.software
def test_the_figure_draws_the_split_and_the_lanes(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    folder = startup_recording(tmp_path)
    d = pl.load(str(folder))
    res = pl.analyze(d["threads"], d["cores"], d["system"])
    drawn = {}
    real_savefig = plt.Figure.savefig

    def keep(fig, *a, **kw):
        drawn["fig"] = fig
        return real_savefig(fig, *a, **kw)
    plt.Figure.savefig = keep
    try:
        assert pl.figure(d["threads"], d["cores"], d["system"], res, "t", tmp_path / "f.png") is not None
    finally:
        plt.Figure.savefig = real_savefig
    axes = drawn["fig"].axes
    labels = [t.get_text() for a in axes for t in a.get_yticklabels()]
    assert {"main", "sensor-hub", "src"} <= set(labels), "a lane per busy thread"
    legends = " ".join(t.get_text() for a in axes if a.get_legend() for t in a.get_legend().get_texts())
    assert "Python threads" in legends and "native threads" in legends and "core 0" in legends
    plot_axes = [a for a in axes if a.get_label() != "<colorbar>"]
    widths = {round(a.get_position().width, 3) for a in plot_axes if a.get_position().width > 0.1}
    assert len(widths) == 1, f"every panel the same width on the shared time axis: {widths}"


def report_text(res):
    return "\n".join(pl.report_lines(res))


@pytest.mark.software
def test_frames_outside_the_recording_are_left_out(tmp_path):
    folder = recording(tmp_path, n=4, t0=1000.0)                                    # 2 s of recording
    a = run(folder, run=str(nav_run(tmp_path, 1010.0, [50.0] * 20)))["aligned"]     # the run starts 10 s later
    assert a == {"frames": 0}


# =============================================================================
# The command line
# =============================================================================

@pytest.mark.software
def test_the_command_line_writes_its_summary_and_lines_up_a_run(tmp_path, capsys):
    folder = recording(tmp_path)
    nav = nav_run(tmp_path, 1001.0, [50.0] * 60)
    assert pl.main([str(folder), "--run", str(nav)]) == 0
    out = capsys.readouterr().out
    assert "process" in out and "aligned" in out and "findings" in out
    res = json.loads((folder / "pi_load.json").read_text())
    assert res["aligned"]["frames"] > 0 and res["process"]["busiest"] == "main"


@pytest.mark.software
def test_a_run_without_t0_still_analyzes_the_recording_and_says_why_it_isnt_aligned(tmp_path, capsys):
    folder = recording(tmp_path)
    nav = nav_run(tmp_path, None, [50.0] * 10)
    assert pl.main([str(folder), "--run", str(nav)]) == 0
    assert "can't line the run up" in capsys.readouterr().err
    assert json.loads((folder / "pi_load.json").read_text())["aligned"] is None


@pytest.mark.software
def test_something_other_than_a_recording_is_an_error(tmp_path):
    (tmp_path / "threads.csv").write_text("a,b\n1,2\n")
    assert pl.main([str(tmp_path)]) == 1
