"""
test_routine_report.py  --  src/analysis/routine_report.py

--software  Margins for each criterion kind; which fields are numeric and
            which change; analyze() on a folder the real harness wrote,
            judged by the real routine; a figure for every routine (its own
            plot where it has one); nav_run beside each attempt with a
            nav.csv; the newest routine folder; the command line.
"""
import csv
import json

import pytest

import src.routines.harness as h
from src.analysis import routine_report as rr
from src.routines import ROUTINES
from src.routines.power_profile import SAMPLE_FIELDS
from src.tests.test_nav_run import lane
from src.navigation_linker import NAV_FIELDS
from src.tests.test_routines import no_conditions


def canned(name, rows, out, options=None, state=None):
    """The real routine's harness run and judge(), with trial() returning rows in turn."""
    base = ROUTINES[name]

    class Canned(base):
        def setup(self, ctx):
            ctx.state.update(state or {})

        def teardown(self, ctx):
            pass

        def trial(self, ctx, i):
            return rows[i]

    folder = out / f"routine_{name}_20261011_100000"
    h.run_routine(Canned(), h.Console(inp=lambda p: "", out=lambda s: None), folder, trials=len(rows),
                  options=options or {}, conditions_fn=no_conditions)
    return folder


ROWS = {
    "tape-check": [{"reading_cm": v} for v in (20.0, 20.2, 19.9)],
    "stop-distance": [{"gap_cm": g, "reported_cm": 1.0, "speed_cps": 1100.0, "battery_v": v, "reason": "stop_sign",
                       "ended_by": "stopped at the line"} for g, v in ((3.5, 11.9), (4.1, 11.8), (6.4, 11.4))],
    "lane-offset": [{"planned_cm": p, "measured_cm": p + 0.1, "reported_cm": p + 0.5, "error_cm": 0.4, "sd_cm": 0.2,
                     "vision_pct": 100.0, "frames": 100, "lane_width_px": 280.0, "offset": p / 15} for p in (-2, 0, 2)],
    "detect-range": [{"gap_cm": g, "state": s, "frames": 60, "seen_pct": 0.0 if s == "off" else 100.0 - g,
                      "wrong_frames": 0, "voted": s, "expected": s, "result": "ok" if g < 30 else "missed"}
                     for g in (0, 30) for s in ("red", "yellow", "green", "off")],
    "recover-offset": [{"planned_cm": p, "start_cm": p, "end_cm": 0.4, "recovered": abs(p) < 8, "robot_start_cm": p,
                        "recover_s": abs(p) * 0.4 if abs(p) < 8 else None, "overshoot_cm": 0.2, "mean_abs_cm": 1.0,
                        "max_abs_cm": abs(p) + 0.5, "lane_lost_s": 0.0, "ended_by": "held", "battery_v": 11.7}
                       for p in (0, 3, -8)],
    "power-profile": [{"stage": s, "seconds": 30, "v_mean": 11.9 - g, "v_min": 11.8 - g, "sag_v": g, "drain_mv_min": 5,
                       "cpu_pct": 40, "temp_max_c": 55, "under_voltage": 0, "throttled": 0}
                      for s, g in (("rest", 0.0), ("motors", 0.6), ("full", 0.8))],
    "imu-check": [{"part": "rest", "direction": "", "seconds": 30, "bias_dps": -1.05, "noise_dps": 0.05,
                   "drift_deg": 0.8, "heading_deg": None, "error_deg": None, "reads": 3000, "rate_hz": 100,
                   "read_ms_p95": 1.2, "read_errors": 0, "temp_c": 30}]
                  + [{"part": "turn", "direction": d, "seconds": 5, "bias_dps": None, "noise_dps": None,
                      "drift_deg": None, "heading_deg": 90 + e, "error_deg": e, "reads": 500, "rate_hz": 100,
                      "read_ms_p95": 1.2, "read_errors": 0, "temp_c": 30} for d, e in (("right", 1.5), ("left", 12.0))],
    "frame-budget": [{"seconds": 60, "frames": 900, "fps": 15.1, "over_budget_pct": 22.0, "work_p95_ms": 61.0,
                      "latency_p95_ms": 58.0, "cpu_mean_pct": 64.0, "cpu_max_pct": 88.0, "rss_max_mb": 210.0,
                      "temp_max_c": 61.0, "throttled": 0, "under_voltage": 0, "ended_by": "time up"}],
    "course-run": [{"completed": True, "ended_by": "end of course", "stopwatch_s": 95.2, "robot_s": 95.0,
                    "clock_diff_s": 0.2, "steps_done": 6, "steps_planned": 6, "touches": 0, "violations": 0,
                    "holds": 3, "reds": 1, "lane_lost_s": 0.0, "contract": 0, "fps": 15.0,
                    "battery_start_v": 11.9, "battery_end_v": 11.6}],
    "figure-eight": [{"minutes": 5, "ended_by": "time up", "intersections": 40, "laps_robot": 5, "laps_seen": 5,
                      "touches": 0, "turns_off_gyro": 0, "turns_off_heading": 1, "left_mean_deg": 88.0,
                      "right_mean_deg": 91.0, "lap_s_mean": 58.0, "lane_lost_s": 0.0, "contract": 0, "fps": 15.0,
                      "battery_start_v": 11.9, "battery_end_v": 11.4}],
}
ROWS["obstacle-check"] = (
    [{"gap_cm": 10.0, "state": "present", "frames": 40, "seen_pct": 100.0, "voted": True, "reported_cm": None,
      "error_cm": None, "clipped_pct": 100.0, "would_brake": True, "result": "ok"}]
    + [{"gap_cm": g, "state": "present", "frames": 40, "seen_pct": 100.0, "voted": True, "reported_cm": g + e,
        "error_cm": e, "clipped_pct": 0.0, "would_brake": False, "result": "ok" if abs(e) <= 3 else "off"}
       for g, e in ((15.0, 0.5), (20.0, 1.5), (30.0, 4.0))]
    + [{"gap_cm": None, "state": "clear", "frames": 40, "seen_pct": 0.0, "voted": False, "reported_cm": None,
        "error_cm": None, "clipped_pct": 0.0, "would_brake": False, "result": "ok"}])
ROWS["obstacle-stop"] = (
    [{"state": "object", "gap_cm": g, "reported_cm": 14.0, "clipped": False, "brakes": 1, "stopped": True,
      "resume_s": 1.05, "speed_cps": 1200.0, "battery_v": 11.8, "ended_by": "drove on"} for g in (9.5, 11.0)]
    + [{"state": "clear", "gap_cm": None, "reported_cm": None, "clipped": False, "brakes": 0, "stopped": False,
        "resume_s": None, "speed_cps": 1210.0, "battery_v": 11.7, "ended_by": "clear lane driven"}])
ROWS["stop-sign"] = (
    [{"sign": "present", "sign_seen": True, "stops": 1, "hold_s": h_, "resumed": True, "across": True,
      "battery_v": 11.8, "ended_by": "across the intersection"} for h_ in (2.0, 2.1)]
    + [{"sign": "absent", "sign_seen": False, "stops": 0, "hold_s": None, "resumed": False, "across": True,
        "battery_v": 11.7, "ended_by": "across the intersection"}])
OPTIONS = {"detect-range": {"target": "light", "gaps": "0,30", "need_cm": 20.0, "frames": 60}}


@pytest.mark.software
@pytest.mark.parametrize("op, limit, value, expected", [
    ("<=", 2.0, 0.5, 1.5), ("<=", 2.0, 3.6, -1.6), (">=", 20.0, 15.1, -4.9), (">=", 5, 5, 0),
    ("within", [2.0, 6.0], 4.4, 1.6), ("within", [2.0, 6.0], 6.5, -0.5), ("within", [2.0, 6.0], 2.5, 0.5),
    ("<=", 0, None, None)])
def test_the_margin_is_how_far_inside_the_limit_a_value_sits(op, limit, value, expected):
    got = rr.margin({"op": op, "limit": limit, "value": value})
    assert got == (None if expected is None else pytest.approx(expected))


@pytest.mark.software
def test_numeric_fields_skip_text_bools_and_the_harness_columns_and_varying_splits_the_constant_ones():
    rows = [{"trial": 1, "t_s": 0.0, "gap_cm": 3.0, "state": "red", "ok": True, "frames": 60, "maybe": None},
            {"trial": 2, "t_s": 5.0, "gap_cm": 4.5, "state": "green", "ok": False, "frames": 60, "maybe": 2.0}]
    fields = rr.numeric_fields(rows)
    assert fields == ["gap_cm", "frames", "maybe"]
    assert rr.varying(rows, fields) == (["gap_cm"], {"frames": 60.0, "maybe": 2.0})


@pytest.mark.software
def test_analyze_reads_the_harness_folder_with_the_routines_own_verdict(tmp_path):
    folder = canned("stop-distance", ROWS["stop-distance"], tmp_path)
    result, rows = rr.load(folder)
    res = rr.analyze(result, rows, folder)
    assert (res["routine"], res["verdict"], res["trials_kept"]) == ("stop-distance", "FAIL", 3)
    by = {c["name"]: c for c in res["criteria"]}
    assert by["mean gap"]["passed"] and by["mean gap"]["margin"] == pytest.approx(min(4.667 - 2, 6 - 4.667), abs=1e-3)
    assert not by["spread (max - min)"]["passed"] and by["spread (max - min)"]["margin"] == pytest.approx(2 - 2.9)
    assert res["fields"]["gap_cm"]["mean"] == pytest.approx(14.0 / 3)
    assert "reason" not in res["fields"] and res["attempts"] == []


@pytest.mark.software
def test_every_routine_has_canned_rows_here_and_every_key_plot_is_a_routine():
    assert set(ROWS) == set(ROUTINES)
    assert set(rr.KEY_PLOTS) <= set(ROUTINES)


@pytest.mark.software
@pytest.mark.parametrize("name", sorted(ROUTINES))
def test_every_routine_draws_its_report(tmp_path, name):
    pytest.importorskip("matplotlib")
    folder = canned(name, ROWS[name], tmp_path, options=OPTIONS.get(name))
    assert rr.main([str(folder), "--no-attempts"]) == 0
    assert (folder / "routine_report.png").stat().st_size > 0
    res = json.loads((folder / "routine_report.json").read_text())
    assert res["routine"] == name and len(res["criteria"]) == len(json.loads((folder / "results.json").read_text())["criteria"])


@pytest.mark.software
def test_the_power_profile_plot_reads_the_samples_when_there_are_some(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    folder = canned("power-profile", ROWS["power-profile"], tmp_path)
    with open(folder / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, SAMPLE_FIELDS)
        w.writeheader()
        for stage in ("rest", "motors"):
            w.writerows({"attempt": 1, "stage": stage, "t_s": t, "battery_v": 11.8 - t / 100, "cpu_pct": 40,
                         "temp_c": 50, "under_voltage": 0, "throttled": 0} for t in range(5))
    result, rows = rr.load(folder)
    fig, ax = plt.subplots()
    rr.KEY_PLOTS["power-profile"](ax, rows, result, folder)
    assert [ln.get_label() for ln in ax.get_lines()][:2] == ["rest", "motors"]     # one line per stage
    plt.close(fig)


@pytest.mark.software
def test_detect_range_spreads_its_states_so_equal_reads_stay_visible(tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    folder = canned("detect-range", ROWS["detect-range"], tmp_path, options=OPTIONS["detect-range"])
    result, rows = rr.load(folder)
    fig, ax = plt.subplots()
    rr.KEY_PLOTS["detect-range"](ax, rows, result, folder)
    xs = {round(float(x), 2) for c in ax.collections for x, _ in c.get_offsets()}
    assert len([x for x in xs if abs(x) < 2]) == 4                 # four states at the 0 cm gap, apart
    plt.close(fig)


@pytest.mark.software
def test_each_attempt_with_a_nav_csv_gets_nav_runs_report(tmp_path):
    folder = canned("tape-check", ROWS["tape-check"], tmp_path)
    for name in ("attempt_01", "attempt_02"):
        (folder / name).mkdir()
        with open(folder / name / "nav.csv", "w", newline="") as f:
            w = csv.DictWriter(f, NAV_FIELDS)
            w.writeheader()
            w.writerows(lane(20))
    (folder / "attempt_03").mkdir()                                  # a redo with no run in it
    got = rr.attempt_reports(folder)
    assert [a for a, _, err in got] == ["attempt_01", "attempt_02"] and all(err is None for _, _, err in got)
    assert (folder / "attempt_01" / "nav_run.json").is_file()
    assert rr.analyze(*rr.load(folder), folder)["attempts"] == ["attempt_01", "attempt_02"]


@pytest.mark.software
def test_with_no_argument_the_newest_routine_folder_is_used(tmp_path):
    older = canned("tape-check", ROWS["tape-check"], tmp_path / "a")
    newer = canned("lane-offset", ROWS["lane-offset"], tmp_path / "b")
    import os
    os.utime(older / "results.json", (1, 1))
    assert rr.find_run(None, roots=(tmp_path / "a", tmp_path / "b")) == newer
    assert rr.find_run(str(newer / "results.json")) == newer


@pytest.mark.software
def test_a_folder_that_isnt_a_routines_is_an_error(tmp_path, capsys):
    assert rr.main([str(tmp_path)]) == 1
    assert "no results.json" in capsys.readouterr().err
