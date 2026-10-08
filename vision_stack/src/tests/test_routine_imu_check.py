"""
test_routine_imu_check.py  --  src/routines/imu_check.py

rest_stats and heading over scripted readings: bias, noise, drift with the
configured bias, rate, read time, errors, a drain's closing reading left
out; the routine on a fake sensor hub, motors, sleep and temperature: the
ten parts in order with turns alternating left and right, a redo repeating
the same part, the settings, the vibration noise against rest and the
motors failing, the hub stopped even after q; the summary's advice (bias,
scale, a flipped IMU_YAW_SIGN); the criteria one at a time and at their
limits.

--software  A scripted tester, a fake hub, motors, sleep and temperature.
"""
import json

import pytest

import src.routines.harness as h
import src.routines.imu_check as ic
from src.config import GYRO_BIAS_DPS
from src.peripherals.sensing import SensorBatch, SensorReading
from src.routines import ROUTINES
from src.tests.test_routines import Person, no_conditions

HZ = 100


def reads(yaw, seconds=1.0, t0=0.0, fail=(), ms=1.0):
    """Readings at HZ over seconds: yaw(k) or a constant; indexes in fail read None (a failed read)."""
    n = int(round(seconds * HZ)) + 1
    f = yaw if callable(yaw) else (lambda k: yaw)
    return [SensorReading(t0 + k / HZ, None if k in fail else f(k), 0.0, None, None, ms + k % 3) for k in range(n)]


def rest(bias=GYRO_BIAS_DPS, noise=0.05, seconds=1.0, **kw):
    return reads(lambda k: bias + (noise if k % 2 else -noise), seconds, **kw)


def turn(deg, seconds=1.0, bias=GYRO_BIAS_DPS, **kw):
    return reads(bias + deg / seconds, seconds, **kw)


class Hub:
    """Each drain returns the next scripted readings; [] once they run out."""
    def __init__(self, batches):
        self.batches, self.drains, self.stopped = list(batches), 0, False

    def drain(self):
        self.drains += 1
        return SensorBatch(self.batches.pop(0) if self.batches else [], None, 0)

    def stop(self):
        self.stopped = True


def script(rest_reads=None, turns=None, vib_reads=None):
    """The drains of a full routine: rest (discard, part), each turn (start, end), vibration (discard, part)."""
    rest_reads = rest() if rest_reads is None else rest_reads
    turns = [(-90.0 if k % 2 == 0 else 90.0) for k in range(ic.TURNS)] if turns is None else turns
    out = [[], rest_reads]
    for d in turns:
        out += [[], d if isinstance(d, list) else turn(d)]
    return out + [[], rest(noise=0.15) if vib_reads is None else vib_reads]


class Motors:
    def __init__(self, fail=None):
        self.calls, self.fail, self.stopped = [], fail, None

    def __call__(self, ctx, seconds, stop):
        self.calls.append(seconds)
        if self.fail:
            raise self.fail
        stop.wait(5.0)
        self.stopped = stop.is_set()


def routine(batches, motors=None):
    sleeps = []
    r = ic.ImuCheck(hub=Hub(batches), motors=motors or Motors(), sleep=sleeps.append, temperature=lambda: 51.5)
    r.sleeps = sleeps
    return r


def go(r, person, tmp_path, **options):
    return h.run_routine(r, person.console(), tmp_path / "out", options=options, conditions_fn=no_conditions)


def everyone(n=28):
    return Person(*[""] * n)


def results(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())


# =============================================================================
# The numbers
# =============================================================================

@pytest.mark.software
def test_rest_stats_bias_noise_drift_rate_time_and_errors():
    rs = rest(bias=-1.0, noise=0.1, seconds=2.0, fail={5}) + [SensorReading(2.5, None, None, 3, 4, None)]
    s = ic.rest_stats(rs, configured_bias=-1.1)
    assert s["reads"] == 201 and s["rate_hz"] == 100.0 and s["read_errors"] == 1
    assert s["bias_dps"] == pytest.approx(-1.0, abs=0.001) and s["noise_dps"] == pytest.approx(0.1, abs=0.001)
    assert s["drift_deg"] == pytest.approx(0.2, abs=0.01)             # 0.1 deg/s over 2 s
    assert s["read_ms_p95"] == 3.0


@pytest.mark.software
def test_rest_stats_of_nothing_and_of_one_reading():
    assert ic.rest_stats([]) == {"reads": 0, "rate_hz": None, "read_errors": 0, "read_ms_p95": None,
                                 "bias_dps": None, "noise_dps": None, "drift_deg": None}
    one = ic.rest_stats(reads(-1.1, seconds=0.0))
    assert (one["reads"], one["rate_hz"], one["bias_dps"], one["noise_dps"], one["drift_deg"]) == (1, None, -1.1, None, 0.0)


@pytest.mark.software
def test_heading_integrates_net_of_bias_and_skips_failed_reads():
    assert ic.heading(turn(90.0, bias=-1.1), -1.1) == pytest.approx(90.0)
    assert ic.heading(turn(-45.0, seconds=0.5, bias=0.0), 0.0) == pytest.approx(-45.0)
    assert ic.heading(turn(90.0, fail={50}), GYRO_BIAS_DPS) == pytest.approx(90.0)   # bridged across the gap
    assert ic.heading(reads(2.0, seconds=0.0), 0.0) == 0.0
    assert ic.heading(reads(lambda k: float(k)), 0.0) == pytest.approx(50.0)    # a ramp: trapezoids, not steps


@pytest.mark.software
def test_the_registry_has_it():
    assert ROUTINES["imu-check"] is ic.ImuCheck and ic.ImuCheck.trials == 10
    assert set(ic.ImuCheck.settings) == {"rest_s", "vib_s", "duty"}             # duty: read by power_profile._motors


# =============================================================================
# The routine
# =============================================================================

@pytest.mark.software
def test_rest_eight_alternating_turns_and_vibration_pass(tmp_path):
    motors = Motors()
    r = routine(script(), motors)
    person = everyone()
    res = go(r, person, tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    rows = results(tmp_path)["rows"]
    assert [x["part"] for x in rows] == ["rest"] + ["turn"] * 8 + ["vibration"]
    assert [x["direction"] for x in rows[1:9]] == ["left", "right"] * 4
    assert rows[0]["bias_dps"] == pytest.approx(GYRO_BIAS_DPS, abs=0.001) and rows[0]["temp_c"] == 51.5
    assert rows[0]["seconds"] == ic.REST_S and abs(rows[0]["drift_deg"]) < 0.01
    assert (rows[1]["heading_deg"], rows[1]["error_deg"], rows[1]["reads"], rows[1]["seconds"]) == (-90.0, 0.0, 101, 1.0)
    assert rows[2]["heading_deg"] == 90.0
    assert r.sleeps == [ic.REST_S, ic.VIB_S] and motors.calls == [ic.VIB_S] and motors.stopped
    state = results(tmp_path)["state"]
    assert state["scale_left"] == 1.0 and state["scale_right"] == 1.0 and "sign_flipped" not in state
    assert state["vibration_noise_x"] == pytest.approx(3.0, abs=0.05)
    assert rows[9]["noise_dps"] == pytest.approx(0.15, abs=0.002)
    asked, said = "\n".join(person.prompts), person.said()
    assert asked.index("LEFT") < asked.index("RIGHT") < asked.index("WHEELS UP")
    assert "put GYRO_BIAS_DPS" not in said and "left turns read 100.0%" in said
    assert r._hub.stopped


@pytest.mark.software
def test_the_settings_set_the_parts_lengths(tmp_path):
    r = routine(script())
    go(r, everyone(), tmp_path, rest_s=5, vib_s=2.5)
    assert r.sleeps == [5.0, 2.5]
    assert results(tmp_path)["rows"][0]["seconds"] == 5.0


@pytest.mark.software
def test_a_redo_repeats_the_same_part_and_direction(tmp_path):
    batches = script()
    batches[2:2] = [[], turn(-80.0)]            # the first turn's bumped attempt, then the real one
    answers = ["", ""] + ["", "", "r"] + [""] * 26
    res = go(routine(batches), Person(*answers), tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    rows = results(tmp_path)["rows"]
    assert len(rows) == 10 and [x["direction"] for x in rows[1:3]] == ["left", "right"]
    assert rows[1]["heading_deg"] == -90.0


@pytest.mark.software
def test_turns_use_the_measured_rest_bias(tmp_path):
    batches = script(rest_reads=rest(bias=-1.2), turns=[turn(d, bias=-1.2) for d in [-90.0, 90.0] * 4])
    res = go(routine(batches), everyone(), tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    assert results(tmp_path)["rows"][1]["heading_deg"] == -90.0


@pytest.mark.software
def test_a_turn_before_any_rest_uses_the_configured_bias(tmp_path):
    r = routine([[], turn(91.25)])
    ctx = h.Context(console=everyone().console(), out_dir=tmp_path, options={})
    r.setup(ctx)
    row = r.trial(ctx, 2)
    assert (row["direction"], row["heading_deg"], row["error_deg"]) == ("right", 91.25, 1.25)


@pytest.mark.software
def test_the_summary_says_the_bias_to_configure(tmp_path):
    person = everyone()
    res = go(routine(script(rest_reads=rest(bias=-0.8))), person, tmp_path)
    assert [c["name"] for c in res["criteria"] if not c["passed"]][0] == "bias off the configured"
    assert "measured bias -0.800 deg/s" in person.said() and "put GYRO_BIAS_DPS = -0.80 in config.py" in person.said()


@pytest.mark.software
def test_the_summary_says_the_scale_and_a_flipped_sign(tmp_path):
    person = everyone()
    go(routine(script(turns=[95.0, -95.0] * 4)), person, tmp_path)
    state = results(tmp_path)["state"]
    assert state["sign_flipped"] and state["scale_left"] == pytest.approx(95 / 90, abs=1e-4)
    assert "flip IMU_YAW_SIGN" in person.said() and "left turns read 105.6%" in person.said()
    one_way = everyone()
    go(routine(script(turns=[90.0, 90.0] * 4)), one_way, tmp_path / "b")
    assert "flip IMU_YAW_SIGN" not in one_way.said()


@pytest.mark.software
def test_motors_failing_are_said_and_kept(tmp_path):
    person = everyone()
    res = go(routine(script(), Motors(fail=RuntimeError("no pigpiod"))), person, tmp_path)
    assert res["verdict"] == h.PASS
    assert "the motors failed" in person.said()
    assert "no pigpiod" in results(tmp_path)["state"]["vibration_error"]


@pytest.mark.software
def test_stopping_at_the_first_prompt_stops_the_hub_and_says_nothing_measured(tmp_path):
    r = routine(script())
    person = Person("q")
    res = go(r, person, tmp_path)
    assert res["stopped_early"] and r._hub.stopped and "measured bias" not in person.said()


# =============================================================================
# The criteria
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("kw, failing", [
    ({"rest_reads": rest(bias=GYRO_BIAS_DPS + 0.25)}, "bias off the configured"),
    ({"rest_reads": rest(bias=GYRO_BIAS_DPS + 0.05, seconds=60.0)}, "rest drift (configured bias)"),
    ({"rest_reads": rest(bias=GYRO_BIAS_DPS - 0.05, seconds=60.0)}, "rest drift (configured bias)"),
    ({"turns": [-93.5, 90.0] + [-90.0, 90.0] * 3}, "turns off 90 by over 3 deg or the wrong way"),
    ({"turns": [-90.0, 93.5] + [-90.0, 90.0] * 3}, "turns off 90 by over 3 deg or the wrong way"),
    ({"turns": [90.0, 90.0] + [-90.0, 90.0] * 3}, "turns off 90 by over 3 deg or the wrong way"),
    ({"turns": [turn(-90.0, fail={7})] + [90.0] + [-90.0, 90.0] * 3}, "read errors"),
    ({"vib_reads": rest(fail={3})}, "read errors"),
])
def test_each_criterion_fails_on_its_own(tmp_path, kw, failing):
    res = go(routine(script(**kw)), everyone(), tmp_path)
    assert res["verdict"] == h.FAIL
    assert [c["name"] for c in res["criteria"] if not c["passed"]] == [failing]


@pytest.mark.software
def test_the_limits_pass(tmp_path):
    kw = {"rest_reads": rest(bias=GYRO_BIAS_DPS + 0.0333, seconds=60.0),   # 0.03 deg/s off, 2.0 deg over the minute
          "turns": [-87.0, 93.0] + [-90.0, 90.0] * 3}
    kw["turns"] = [turn(d, bias=GYRO_BIAS_DPS + 0.0333) for d in kw["turns"]]
    res = go(routine(script(**kw)), everyone(), tmp_path)
    assert res["verdict"] == h.PASS, res["criteria"]
    rest_row = results(tmp_path)["rows"][0]
    assert abs(rest_row["drift_deg"]) <= ic.DRIFT_MAX_DEG
    assert [x["error_deg"] for x in results(tmp_path)["rows"][1:3]] == [3.0, 3.0]


@pytest.mark.software
def test_the_latest_rest_is_judged():
    rows = [{"part": "rest", "bias_dps": 0.0, "drift_deg": 0.0, "read_errors": 0},
            {"part": "rest", "bias_dps": GYRO_BIAS_DPS, "drift_deg": 0.0, "read_errors": 0}]
    assert all(c["passed"] for c in ic.ImuCheck().judge(rows))
    assert ic.ImuCheck().judge([])[0]["value"] is None
