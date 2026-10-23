"""Tape check: how repeatable the tester's own hand measurement is.

Purpose:
    Every accuracy routine compares the robot with a hand measurement, and
    can't judge it more finely than that measurement's spread. Each tester
    measures one fixed distance a few times, taking the tape away in
    between; the spread is their floor, recorded with their name. It also
    rehearses the prompts with no hardware.

Main package:
    TapeCheck: the routine. Pass: the readings span at most MAX_SPREAD_CM.
"""
from src.routines.harness import Routine, criterion, stats

MAX_SPREAD_CM = 0.5         # a reading spread over this is coarser than the tightest tolerance in requirements.md


class TapeCheck(Routine):
    name = "tape-check"
    title = "Tape check: hand-measurement repeatability"
    question = "How much do this tester's tape readings of one fixed distance vary?"
    requirement = ""
    trials = 5
    fields = ("reading_cm",)
    instructions = """\
Pick one fixed distance on the course and mark both ends: e.g. from a stop
line's near edge to a tape mark about 15 cm away. Each trial: lay the tape,
read it to the nearest millimetre, then take the tape away before the next."""

    def setup(self, ctx):
        ctx.state["what"] = ctx.console.ask("What are you measuring (a few words)", default="a fixed distance")

    def trial(self, ctx, i):
        return {"reading_cm": ctx.console.ask_number("Tape reading", lo=0.0, hi=300.0, unit="cm")}

    def judge(self, rows):
        s = stats(r["reading_cm"] for r in rows)
        return [criterion("spread (max - min)", s["max"] - s["min"], "<=", MAX_SPREAD_CM, "cm")]
