"""
test_routines.py  --  src/routines/ (harness.py, tape_check.py, __main__.py)

The console with a scripted tester: defaults, q and Ctrl-D stopping, numbers
re-asked until they parse and sit in range, the after-trial choices.
Criteria for each operator and a missing value; stats; the verdicts. The
conditions and the needs check. run_routine() through every path a tester
can take: keep, redo, discard, q inside a trial, q after one, Ctrl-C, an
error (the folder still written, teardown always run), and the folder's
files. The tape check's criterion at its threshold, and the command line.

--software  A scripted console and fake readings. No hardware.
"""
import csv
import json

import pytest

import src.routines.harness as h
from src.routines import ROUTINES
from src.routines.__main__ import main
from src.routines.tape_check import MAX_SPREAD_CM, TapeCheck


class Person:
    """The person at the keyboard: answers in order; out collects everything said."""
    def __init__(self, *answers):
        self.answers, self.prompts, self.out = list(answers), [], []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        a = self.answers.pop(0)
        if a is KeyboardInterrupt:
            raise KeyboardInterrupt
        return a

    def console(self):
        return h.Console(inp=self, out=self.out.append)

    def said(self):
        return "\n".join(self.out)


def no_conditions(tester="", notes=""):
    return {"time": "t", "commit": "abc1234", "uncommitted_changes": False, "host": "pi", "tester": tester,
            "notes": notes, "temp_c": 50.0, "cpu_mhz": 1000.0, "throttled": "0x0", "battery_v": 11.9}


# =============================================================================
# The console
# =============================================================================

@pytest.mark.software
def test_ask_strips_takes_the_default_and_stops_on_q_or_end_of_input():
    t = Person("  Mike ", "", "Q")
    c = t.console()
    assert c.ask("Name") == "Mike"
    assert c.ask("Mat", default="the course") == "the course" and t.prompts[1] == "Mat [the course]: "
    with pytest.raises(h.Quit):
        c.ask("Anything")
    with pytest.raises(h.Quit):
        c.ask("More")                                               # end of input: EOFError


@pytest.mark.software
def test_numbers_are_asked_again_until_they_parse_and_sit_in_range():
    t = Person("abc", "400", "-1", "12,5")
    c = t.console()
    assert c.ask_number("Gap", lo=0, hi=300, unit="cm") == 12.5
    assert t.prompts[0] == "Gap (0..300 cm): "
    assert "'abc' isn't a number" in t.said() and "400 is outside 0..300" in t.said() and "-1 is outside" in t.said()
    assert h.Console(inp=lambda p: "7").ask_number("x") == 7.0
    open_top = Person("1e9")
    assert open_top.console().ask_number("x", lo=0) == 1e9 and open_top.prompts[0] == "x (0..): "


@pytest.mark.software
def test_after_a_trial_enter_keeps_r_redoes_d_discards_and_anything_else_is_asked_again():
    t = Person("", "k", "r", "x", "d", "q")
    c = t.console()
    assert [c.after_trial() for _ in range(4)] == [h.KEEP, h.KEEP, h.REDO, h.DISCARD]
    assert "Enter, r, d or q" in t.said()
    with pytest.raises(h.Quit):
        c.after_trial()


# =============================================================================
# Criteria and verdicts
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("value, op, limit, passed", [
    (0.5, "<=", 0.5, True), (0.51, "<=", 0.5, False), (95.0, ">=", 95.0, True), (94.9, ">=", 95.0, False),
    (2.0, "within", (-2.0, 2.0), True), (-2.1, "within", (-2.0, 2.0), False), (None, "<=", 1.0, False),
])
def test_criteria_at_their_limits(value, op, limit, passed):
    assert h.criterion("x", value, op, limit, "cm")["passed"] is passed


@pytest.mark.software
def test_criterion_text_and_a_bad_operator():
    assert h.criterion("spread", 0.3, "<=", 0.5, "cm")["text"] == "spread: 0.3 cm (needs <= 0.5 cm)"
    assert h.criterion("offset", -1.25, "within", (-2, 2), "cm")["text"] == "offset: -1.25 cm (needs -2..2 cm)"
    assert h.criterion("detected", None, ">=", 95, "%")["text"] == "detected: -- % (needs >= 95 %)"
    with pytest.raises(ValueError):
        h.criterion("x", 1.0, "==", 1.0)


@pytest.mark.software
def test_stats_and_verdicts():
    assert h.stats([1.0, 2.0, 3.0]) == {"n": 3, "mean": 2.0, "sd": 1.0, "min": 1.0, "max": 3.0}
    assert h.stats([4.0])["sd"] == 0.0 and h.stats([]) is None and h.stats([None, 2])["n"] == 1
    ok, bad = h.criterion("a", 1, "<=", 2), h.criterion("b", 3, "<=", 2)
    assert h.verdict([ok], True) == h.PASS and h.verdict([ok, bad], True) == h.FAIL
    assert h.verdict([ok], False) == h.INCOMPLETE and h.verdict([], True) == h.RECORDED


# =============================================================================
# Conditions and needs
# =============================================================================

@pytest.mark.software
def test_conditions_record_the_code_the_machine_the_tester_and_the_pi():
    c = h.conditions("Ana", "bright room", battery=lambda: 11.84,
                     system=lambda: {"temp_c": 48.5, "cpu_mhz": 1000.0, "throttled_raw": "0x0"})
    assert (c["tester"], c["notes"], c["battery_v"], c["temp_c"], c["throttled"]) == ("Ana", "bright room", 11.84, 48.5, "0x0")
    assert len(c["commit"] or "") >= 7 and isinstance(c["uncommitted_changes"], bool) and c["host"]


@pytest.mark.software
def test_battery_volts_from_the_adc_or_none_without_it(monkeypatch):
    from src.diagnostics import battery_run
    cleaned = []

    class Power:
        def voltage_raw(self):
            return 11.876

        def cleanup(self):
            cleaned.append(1)
    monkeypatch.setattr(battery_run, "open_battery", lambda say: Power())
    assert h.read_battery_volts() == 11.88 and cleaned == [1]
    monkeypatch.setattr(battery_run, "open_battery", lambda say: None)
    assert h.read_battery_volts() is None

    class Broken(Power):
        def voltage_raw(self):
            raise OSError("no ACK")
    monkeypatch.setattr(battery_run, "open_battery", lambda say: Broken())
    assert h.read_battery_volts() is None and cleaned == [1, 1]


@pytest.mark.software
def test_needs_say_what_to_do_when_missing(tmp_path):
    checks = {"pigpiod": (lambda: False, "start it"), "other": (lambda: True, "")}
    assert h.check_needs(("pigpiod", "other"), checks) == ["start it"]
    assert h.check_needs(("nope",), checks) == ["unknown need 'nope'"]
    (tmp_path / "12").mkdir()
    (tmp_path / "12" / "comm").write_text("pigpiod\n")
    (tmp_path / "13").mkdir()                                          # gone before its comm was read
    assert h._process_running("pigpiod", tmp_path) and not h._process_running("pigpiod", tmp_path / "x")


# =============================================================================
# run_routine()
# =============================================================================

class Probe(h.Routine):
    """A routine that records its calls: each trial asks for one number."""
    name, title, question, requirement, trials = "probe", "Probe", "Does it work?", "P9", 3
    fields = ("value",)
    instructions = "Set it up."

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def setup(self, ctx):
        self.calls.append("setup")

    def trial(self, ctx, i):
        self.calls.append(f"trial {i}")
        self.attempts = getattr(self, "attempts", []) + [ctx.attempt]
        if i == self.fail_at:
            raise RuntimeError("motor driver fault")
        return {"value": ctx.console.ask_number("Value")}

    def teardown(self, ctx):
        self.calls.append("teardown")

    def judge(self, rows):
        self.calls.append(f"judge {len(rows)}")
        return [h.criterion("mean", h.stats(r["value"] for r in rows)["mean"], "<=", 5.0)]


def run(routine, person, tmp_path, **kw):
    clock = iter(range(100))
    return h.run_routine(routine, person.console(), tmp_path / "out", conditions_fn=no_conditions,
                         clock=lambda: float(next(clock)), **kw)


@pytest.mark.software
def test_a_full_run_keeps_every_trial_and_judges_them(tmp_path):
    r, t = Probe(), Person("1", "", "2", "", "3", "")
    res = run(r, t, tmp_path, tester="Ana", notes="dim")
    assert r.calls == ["setup", "trial 0", "trial 1", "trial 2", "teardown", "judge 3"]
    assert res["verdict"] == h.PASS and res["trials_kept"] == 3 and not res["stopped_early"]
    said = t.said()
    assert "=== Probe ===" in said and "Question: Does it work?" in said and "Verifies: P9" in said
    assert "Set it up." in said and "--- trial 3 of 3 ---" in said and "  value 2" in said
    with open(tmp_path / "out" / "trials.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(x["trial"], x["value"]) for x in rows] == [("1", "1.0"), ("2", "2.0"), ("3", "3.0")]
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["verdict"] == "PASS" and saved["conditions"]["start"]["tester"] == "Ana" and len(saved["rows"]) == 3
    text = (tmp_path / "out" / "summary.txt").read_text()
    assert text.startswith("Probe: PASS") and "PASS  mean: 2 (needs <= 5)" in text and "notes: dim" in text


@pytest.mark.software
def test_redo_runs_the_trial_again_and_discard_drops_it(tmp_path):
    r, t = Probe(), Person("9", "r", "1", "", "8", "d", "2", "", "3", "")
    res = run(r, t, tmp_path)
    assert res["trials_kept"] == 3 and [x["value"] for x in json.loads(
        (tmp_path / "out" / "results.json").read_text())["rows"]] == [1.0, 2.0, 3.0]
    assert "again" in t.said() and "discarded" in t.said()
    # five attempts for three kept: the redo and the discard ran trial 0 and trial 1 again
    assert [c for c in r.calls if c.startswith("trial")] == ["trial 0", "trial 0", "trial 1", "trial 1", "trial 2"]
    assert r.attempts == [0, 1, 2, 3, 4]                                    # ctx.attempt counts every start


@pytest.mark.software
def test_q_inside_a_trial_stops_and_keeps_what_was_done(tmp_path):
    r, t = Probe(), Person("1", "", "q")
    res = run(r, t, tmp_path)
    assert res["verdict"] == h.INCOMPLETE and res["trials_kept"] == 1 and res["stopped_early"]
    assert r.calls[-2:] == ["teardown", "judge 1"]
    assert "stopped early" in (tmp_path / "out" / "summary.txt").read_text()


@pytest.mark.software
def test_q_after_a_trial_keeps_that_trial_then_stops(tmp_path):
    res = run(Probe(), Person("1", "", "2", "q"), tmp_path)
    assert res["trials_kept"] == 2 and res["verdict"] == h.INCOMPLETE


@pytest.mark.software
@pytest.mark.parametrize("answers", [("1", "", KeyboardInterrupt), ("1", KeyboardInterrupt)])
def test_ctrl_c_stops_like_q_with_teardown_and_the_folder(tmp_path, answers):
    r = Probe()
    res = run(r, Person(*answers), tmp_path)
    assert res["stopped_early"] and "teardown" in r.calls and (tmp_path / "out" / "summary.txt").exists()


@pytest.mark.software
def test_an_error_is_recorded_after_teardown_and_then_raised(tmp_path):
    r = Probe(fail_at=1)
    with pytest.raises(RuntimeError, match="motor driver fault"):
        run(r, Person("1", ""), tmp_path)
    assert r.calls == ["setup", "trial 0", "trial 1", "teardown", "judge 1"]
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["verdict"] == h.INCOMPLETE and "motor driver fault" in saved["error"]


@pytest.mark.software
def test_a_stop_before_any_trial_judges_nothing_and_trials_can_be_overridden(tmp_path):
    r = Probe()
    res = run(r, Person("q"), tmp_path)
    assert res["criteria"] == [] and "judge 0" not in " ".join(r.calls)
    res = run(Probe(), Person("1", ""), tmp_path, trials=1)
    assert res["trials_planned"] == 1 and res["verdict"] == h.PASS


@pytest.mark.software
def test_a_routine_without_criteria_is_recorded(tmp_path):
    class Characterize(Probe):
        def judge(self, rows):
            return []
    res = run(Characterize(), Person("1", "", "2", "", "3", ""), tmp_path)
    assert res["verdict"] == h.RECORDED
    assert "none: a characterization" in (tmp_path / "out" / "summary.txt").read_text()


# =============================================================================
# The tape check and the command line
# =============================================================================

@pytest.mark.software
def test_the_tape_check_passes_a_spread_up_to_its_limit(tmp_path):
    at = run(TapeCheck(), Person("pen", "15.0", "", "15.5", "", "15.2", "", "15.1", "", "15.3", ""), tmp_path)
    assert at["verdict"] == h.PASS and at["criteria"][0]["value"] == pytest.approx(MAX_SPREAD_CM)
    assert at["state"] == {"what": "pen"}
    over = run(TapeCheck(), Person("", "15.0", "", "15.6", "", "15.2", "", "15.1", "", "15.3", ""), tmp_path / "b")
    assert over["verdict"] == h.FAIL and over["state"] == {"what": "a fixed distance"}


@pytest.mark.software
def test_the_command_line_lists_runs_and_refuses(tmp_path, monkeypatch):
    assert "tape-check" in ROUTINES
    t = Person()
    assert main(["--list"], t.console()) == 0 and "tape-check" in t.said()
    assert main([], Person().console()) == 2
    t = Person()
    assert main(["nope"], t.console()) == 2 and "no routine 'nope'" in t.said()
    monkeypatch.setattr(h, "conditions", no_conditions)
    import src.routines.__main__ as m
    monkeypatch.setattr(m, "run_routine", lambda r, c, out, n, tester, notes, options: h.run_routine(
        r, c, out, n, tester, notes, options, conditions_fn=no_conditions))
    t = Person("Ignacio", "pen", "15.0", "", "15.1", "", "15.0", "", "15.1", "", "15.0", "")
    assert main(["tape-check", "--out", str(tmp_path / "a")], t.console()) == 0
    assert json.loads((tmp_path / "a" / "results.json").read_text())["conditions"]["start"]["tester"] == "Ignacio"
    t = Person("pen", "15.0", "", "17.0", "", "q")
    assert main(["tape-check", "--tester", "Ana", "--out", str(tmp_path / "b")], t.console()) == 1
    monkeypatch.setattr(TapeCheck, "needs", ("pigpiod",))
    monkeypatch.setattr(m, "check_needs", lambda needs: ["the GPIO daemon isn't running: make pigpiod"])
    t = Person()
    assert main(["tape-check"], t.console()) == 2 and "make pigpiod" in t.said()


@pytest.mark.software
def test_settings_parse_as_numbers_or_text_and_only_the_routines_own():
    from src.routines.__main__ import parse_settings
    known = {"stage_s": "", "duty": "", "mode": ""}
    assert parse_settings(["stage_s=30", "duty=0.35", "mode=quiet", " duty = 1e-1 "], known) == \
        {"stage_s": 30, "duty": 0.1, "mode": "quiet"}
    assert isinstance(parse_settings(["stage_s=30"], known)["stage_s"], int)        # a count stays a whole number
    with pytest.raises(ValueError, match="no setting 'speed'; this routine has: duty, mode, stage_s"):
        parse_settings(["speed=1"], known)
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        parse_settings(["stage_s"], known)


@pytest.mark.software
def test_the_command_line_passes_settings_and_lists_them(tmp_path, monkeypatch):
    import src.routines.__main__ as m
    seen = {}
    monkeypatch.setattr(m, "run_routine", lambda r, c, out, n, tester, notes, options: seen.update(options=options)
                        or {"verdict": "RECORDED"})
    monkeypatch.setattr(m, "check_needs", lambda needs: [])
    assert main(["power-profile", "--tester", "x", "--set", "stage_s=5", "--out", str(tmp_path)], Person().console()) == 0
    assert seen["options"] == {"stage_s": 5}
    p = Person()
    assert main(["power-profile", "--tester", "x", "--set", "nope=1"], p.console()) == 2 and "no setting 'nope'" in p.said()
    p = Person()
    main(["--list"], p.console())
    assert "--set stage_s=..." in p.said() and "--set max_s=..." in p.said()
