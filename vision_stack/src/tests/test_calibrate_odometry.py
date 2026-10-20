"""
test_calibrate_odometry.py  --  src/scripts/calibrate_odometry.py

A trial: counts moved per cm, the wheels' mismatch, refused when a wheel
didn't count forward or the distance isn't positive. The summary: cm per
count from every trial's counts over every trial's cm, each wheel, the
spread. The run: push trials (prompts, a failed trial redone, q stops),
drive trials (drive, brake, ask the distance), the JSON written or not.

--software  Fake encoders and motors.
"""
import json

import pytest

import src.scripts.calibrate_odometry as co


class Wheels:
    """counts() from a script of readings; a motor that logs."""
    def __init__(self, *readings):
        self.readings, self.calls, self.closed = list(readings), [], False

    def counts(self):
        return self.readings.pop(0)

    def drive(self, left, right):
        self.calls.append(("drive", left, right))

    def brake(self):
        self.calls.append(("brake",))


def asker(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


@pytest.mark.software
def test_a_trial_is_counts_per_cm_with_the_wheels_mismatch():
    t = co.trial(100.0, (0, 0), (4700, 4800))
    assert t == {"cm": 100.0, "left": 4700, "right": 4800, "counts_per_cm": 47.5, "mismatch_pct": 2.11}
    for cm, after in ((0.0, (10, 10)), (100.0, (0, 50)), (100.0, (-40, -40))):
        with pytest.raises(ValueError):
            co.trial(cm, (0, 0), after)


@pytest.mark.software
def test_the_summary_pools_every_trial():
    s = co.summarize([co.trial(100.0, (0, 0), (4700, 4700)), co.trial(50.0, (0, 0), (2400, 2400))])
    assert s["counts_per_cm"] == pytest.approx(7100 / 150, abs=1e-4)
    assert s["cm_per_count"] == pytest.approx(150 / 7100, abs=1e-6)
    assert s["spread_pct"] == pytest.approx(100 * (48 - 47) / (7100 / 150), abs=0.01)
    with pytest.raises(ValueError):
        co.summarize([])


@pytest.mark.software
def test_push_trials_redo_a_failed_one_and_stop_on_q():
    w, said = Wheels((0, 0), (4700, 4750), (5000, 5000), (5000, 5000), (5000, 5000), (9700, 9800)), []
    got = co.run(w.counts, asker("", "", "", "", "", "", "q"), said.append, cm=100.0, trials=3)
    assert [(t["left"], t["right"]) for t in got] == [(4700, 4750), (4700, 4800)]
    assert any("redo this trial" in s for s in said)                         # the push that didn't count


@pytest.mark.software
def test_drive_trials_drive_brake_and_ask_how_far():
    w = Wheels((0, 0), (3000, 2900), (3000, 2900), (6000, 5900))
    slept = []
    got = co.run(w.counts, asker("", "64"), print, trials=1, motor=w, drive_s=0.35, duty=0.4, sleep=slept.append)
    assert got[0]["cm"] == 64.0
    # the command re-sent inside the motor watchdog's 0.5 s, then a brake, then a coast before counting
    assert w.calls == [("drive", 0.4, 0.4)] * 4 + [("brake",)]
    assert sum(slept[:4]) == pytest.approx(0.35) and max(slept[:4]) <= co.DRIVE_RESEND_S


@pytest.mark.software
def test_main_writes_the_scale_the_pipeline_reads(tmp_path):
    from src.estimation.odometry import load_odometry
    out, w = tmp_path / "odometry.json", Wheels((0, 0), (4700, 4700))
    closed = []
    rc = co.main(["--trials", "1", "--out", str(out)], hardware=lambda: (w.counts, None, lambda: closed.append(1)),
                 ask=asker("", ""), say=lambda s: None)
    assert rc == 0 and closed == [1]
    data = json.loads(out.read_text())
    assert data["mode"] == "push" and load_odometry(out) == data["cm_per_count"] == pytest.approx(1 / 47, abs=1e-6)
    rc = co.main(["--trials", "1", "--out", str(tmp_path / "none.json")], hardware=lambda: (w.counts, None, lambda: None),
                 ask=asker("q"), say=lambda s: None)
    assert rc == 1 and not (tmp_path / "none.json").exists()


@pytest.mark.software
def test_the_report_says_how_far_the_measurement_is_from_the_nominal_scale():
    # 700 counts over 10 turns' 103.67 cm: exactly the nominal 3.3 cm wheel at 700 counts/turn
    s = co.summarize([co.trial(103.67, (0, 0), (7000, 7000))])
    n = co.against_nominal(s)
    assert n["vs_nominal_pct"] == pytest.approx(0.0, abs=0.1) and n["counts_per_rev"] == pytest.approx(700, abs=1)
    assert "vs nominal     +0%" in co.report(s) or "vs nominal     -0%" in co.report(s)
    squashed = co.summarize([co.trial(100.0, (0, 0), (7000, 7000))])          # 3.5% short: a 3.18 cm wheel
    assert co.against_nominal(squashed)["effective_diameter_cm"] == pytest.approx(3.183, abs=0.002)
    wrong = co.summarize([co.trial(103.67, (0, 0), (2350, 2350))])            # a third the counts: not 100:1
    assert "the gear ratio or the decoding" in co.report(wrong)
