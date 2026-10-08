"""Routine harness: a hardware test a teammate runs by following prompts, judged PASS or FAIL.

Purpose:
    A routine answers one question about the robot on the real hardware
    (how far from the line does it stop; how much does the pack sag with
    the motors on) by repeating a trial and comparing what it measured, or
    what the tester measured by hand, with pass criteria agreed up front.
    The people running it shouldn't need to know which linker, which flags
    or where files go: the routine prompts each step on the console, asks
    for the hand measurements, checks the answers, and writes one folder
    with every trial, the conditions it ran under, and a verdict. This
    module is what every routine shares; a routine itself (routines/*.py)
    is a question, its trial and its criteria.

    A routine is built from a test request card (docs/routines/request_card.md):
    the question, the requirement it closes, how ground truth is measured,
    the conditions, how many trials, the pass criteria.

Main package:
    Console: the prompts. ask(), ask_number() (checked against limits),
        wait(); "q" at any prompt stops the routine cleanly.
    Routine: the base class: name, title, question, requirement, trials,
        fields, needs, instructions; setup(), trial(), teardown(), judge().
    criterion(): one pass criterion's result.
    conditions(): what the run was under: time, commit, host, tester,
        temperature, clock, throttling, battery volts.
    run_routine(): the whole flow; check_needs(): what the hardware must
        have running first; stats(): n, mean, sd, min, max for judge().

Flow:
    1. needs checked (pigpiod running...); the routine's question and
       instructions shown; conditions recorded.
    2. Each trial: the routine's trial() (its prompts and measurements),
       its row shown, then keep it, redo it, discard it or stop.
    3. teardown() whatever happened; conditions again at the end; judge()
       on the kept trials.
    4. trials.csv, results.json and summary.txt in runs/routine_<name>_<time>/;
       the verdict PASS, FAIL, INCOMPLETE (stopped early) or RECORDED
       (no criteria: a characterization).
"""
import csv
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

PASS, FAIL, INCOMPLETE, RECORDED = "PASS", "FAIL", "INCOMPLETE", "RECORDED"
QUIT_WORDS = ("q", "quit")
KEEP, REDO, DISCARD = "keep", "redo", "discard"


class Quit(Exception):
    """The tester stopped the routine at a prompt."""


# =============================================================================
# The console
# =============================================================================

class Console:
    """
    Prompts on the terminal. input and output are injectable, so tests
    script the tester's answers.
    """
    def __init__(self, inp=input, out=print):
        self._in, self._out = inp, out

    def say(self, text: str = "") -> None:
        self._out(text)

    def ask(self, prompt: str, default: str | None = None) -> str:
        """A line of text; default for an empty answer. q stops the routine."""
        hint = f" [{default}]" if default not in (None, "") else ""
        try:
            answer = self._in(f"{prompt}{hint}: ").strip()
        except EOFError:
            raise Quit from None
        if answer.lower() in QUIT_WORDS:
            raise Quit
        return answer if answer or default is None else default

    def ask_number(self, prompt: str, lo: float | None = None, hi: float | None = None, unit: str = "") -> float:
        """A number, asked again until it parses and sits within [lo, hi]."""
        rng = ("" if lo is None and hi is None else
               f" ({'' if lo is None else lo}..{'' if hi is None else hi}{' ' + unit if unit else ''})")
        while True:
            text = self.ask(f"{prompt}{rng}")
            try:
                value = float(text.replace(",", "."))
            except ValueError:
                self.say(f"  '{text}' isn't a number; try again (q stops)")
                continue
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                self.say(f"  {value:g} is outside {lo}..{hi}; try again (q stops)")
                continue
            return value

    def wait(self, prompt: str = "Press Enter to go on") -> None:
        """Wait for Enter; q stops the routine."""
        self.ask(prompt, default="")

    def after_trial(self) -> str:
        """Keep, redo or discard the trial just run; q stops (the trial is kept first)."""
        while True:
            answer = self.ask("Enter = keep, r = redo, d = discard, q = keep and stop", default="").lower()
            if answer in ("", "k"):
                return KEEP
            if answer == "r":
                return REDO
            if answer == "d":
                return DISCARD
            self.say("  Enter, r, d or q")


# =============================================================================
# Routines and criteria
# =============================================================================

def criterion(name: str, value: float | None, op: str, limit, unit: str = "") -> dict:
    """
    One pass criterion: value against limit with op ("<=", ">=", or
    "within": limit is (lo, hi)). A value of None (nothing to judge) fails.
    """
    if value is None:
        passed = False
    elif op == "<=":
        passed = value <= limit
    elif op == ">=":
        passed = value >= limit
    elif op == "within":
        passed = limit[0] <= value <= limit[1]
    else:
        raise ValueError(f"unknown op {op!r}")
    u = f" {unit}" if unit else ""
    needs = f"{limit[0]:g}..{limit[1]:g}" if op == "within" else f"{op} {limit:g}"
    got = "--" if value is None else f"{value:.3g}"
    return {"name": name, "value": None if value is None else round(value, 4), "op": op,
            "limit": list(limit) if op == "within" else limit, "unit": unit, "passed": passed,
            "text": f"{name}: {got}{u} (needs {needs}{u})"}


def stats(values) -> dict | None:
    """n, mean, sd (sample), min, max of the numbers given; None without any."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v)}


class Routine:
    """
    A routine: subclass it, fill the fields, write trial() (and judge()).

    name: The command-line name (make routine-<name>).
    title, question: What the routine is and the one question it answers.
    requirement: The requirements.md row it verifies (P3, D4...), or "".
    trials: How many trials by default (--trials overrides).
    fields: The columns a trial's row has, in order.
    needs: What must be running first, checked by check_needs: "pigpiod".
    instructions: What the tester sets up before the first trial.
    settings: The routine's own options ({name: description}), given as
        --set name=value and read from ctx.options.
    """
    name = ""
    title = ""
    question = ""
    requirement = ""
    trials = 5
    fields: tuple = ()
    needs: tuple = ()
    instructions = ""
    settings: dict = {}             # {name: what it is}: the routine's own --set KEY=VALUE options

    def setup(self, ctx: "Context") -> None:
        """Before the first trial: open hardware, ask set-up questions."""

    def trial(self, ctx: "Context", i: int) -> dict:
        """
        One trial: prompt, measure, return a row with the fields. i is the
        trial being filled (0-based): a redo runs the same i again.
        """
        raise NotImplementedError

    def teardown(self, ctx: "Context") -> None:
        """After the last trial, or a stop or an error: release hardware."""

    def judge(self, rows: list[dict]) -> list[dict]:
        """The pass criteria (criterion()) on the kept rows; none: a characterization."""
        return []


@dataclass
class Context:
    """
    What a routine's methods get: the console, its folder, the options, a
    scratch dict, and attempt: how many trials have been started before
    this one (redone and discarded ones too), for naming per-attempt files.
    """
    console: Console
    out_dir: Path
    options: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    attempt: int = 0


# =============================================================================
# Conditions and needs
# =============================================================================

def _git(*args) -> str | None:
    try:
        p = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                           cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def read_battery_volts() -> float | None:
    """One resting reading of the pack, or None without the ADC (a laptop, or no I2C)."""
    from src.diagnostics import battery_run
    power = battery_run.open_battery(say=lambda s: None)
    if power is None:
        return None
    try:
        return round(power.voltage_raw(), 2)
    except Exception:                   # an unreadable ADC: the run goes on without it
        return None
    finally:
        power.cleanup()


def conditions(tester: str = "", notes: str = "", battery=read_battery_volts, system=None) -> dict:
    """
    What the run is under: when, which code (commit, and whether it had
    uncommitted changes), which machine, who, the SoC's temperature, clock
    and throttling, and the pack's volts. Every reading is optional.
    """
    if system is None:
        from src.diagnostics.system_monitor import sample
        system = sample
    sysrow = system()
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain")
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit,
            "uncommitted_changes": None if dirty is None else bool(dirty), "host": platform.node(),
            "tester": tester, "notes": notes, "temp_c": sysrow.get("temp_c"), "cpu_mhz": sysrow.get("cpu_mhz"),
            "throttled": sysrow.get("throttled_raw"), "battery_v": battery()}


def _process_running(name: str, proc: Path = Path("/proc")) -> bool:
    for entry in proc.iterdir() if proc.is_dir() else ():
        if entry.name.isdigit():
            try:
                if (entry / "comm").read_text().strip() == name:
                    return True
            except OSError:
                continue
    return False


# What each need checks, and what to do when it's missing
NEEDS = {"pigpiod": (lambda: _process_running("pigpiod"), "the GPIO daemon isn't running: make pigpiod")}


def check_needs(needs: tuple, checks: dict = NEEDS) -> list[str]:
    """What's missing, as instructions; [] when everything the routine needs is there."""
    out = []
    for n in needs:
        ok, why = checks.get(n, (lambda: False, f"unknown need {n!r}"))
        if not ok():
            out.append(why)
    return out


# =============================================================================
# Running a routine
# =============================================================================

def verdict(criteria: list[dict], complete: bool) -> str:
    """PASS / FAIL on the criteria; INCOMPLETE when stopped early; RECORDED with no criteria."""
    if not complete:
        return INCOMPLETE
    if not criteria:
        return RECORDED
    return PASS if all(c["passed"] for c in criteria) else FAIL


def run_routine(routine: Routine, console: Console, out_dir, trials: int | None = None, tester: str = "",
                notes: str = "", options: dict | None = None, conditions_fn=conditions, clock=time.monotonic) -> dict:
    """
    The whole routine, prompts to verdict; returns the result and writes
    the folder. Stopping at a prompt (q, or Ctrl-D) keeps what was done.
    An error in the routine is recorded as the verdict's reason and raised
    after the folder is written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = trials or routine.trials
    ctx = Context(console, out, dict(options or {}))
    say = console.say
    say(f"\n=== {routine.title or routine.name} ===")
    say(f"Question: {routine.question}")
    if routine.requirement:
        say(f"Verifies: {routine.requirement} (docs/requirements.md)")
    say(f"{n} trials. At any prompt, q stops and keeps what's done.")
    if routine.instructions:
        say("\n" + routine.instructions.strip())
    start = conditions_fn(tester, notes)
    rows, stopped, error = [], False, None
    t0 = clock()
    try:
        routine.setup(ctx)
        while len(rows) < n:
            say(f"\n--- trial {len(rows) + 1} of {n} ---")
            try:
                row = routine.trial(ctx, len(rows))
            except (Quit, KeyboardInterrupt):
                stopped = True
                break
            ctx.attempt += 1
            say("  " + ", ".join(f"{k} {_fmt(row.get(k))}" for k in routine.fields))
            try:
                choice = console.after_trial()
            except (Quit, KeyboardInterrupt):
                rows.append({"trial": len(rows) + 1, "t_s": round(clock() - t0, 2), **row})
                stopped = True
                break
            if choice == KEEP:
                rows.append({"trial": len(rows) + 1, "t_s": round(clock() - t0, 2), **row})
            elif choice == DISCARD:
                say("  discarded")
            else:
                say("  again")
    except (Quit, KeyboardInterrupt):  # Ctrl-C stops like q: teardown runs, the folder is written
        stopped = True
    except Exception as exc:            # the folder still records what happened
        error = exc
    finally:
        try:
            routine.teardown(ctx)
        except Exception as exc:
            error = error or exc
    end = conditions_fn(tester, notes)
    criteria = routine.judge(rows) if rows else []
    complete = len(rows) >= n and error is None
    result = {"routine": routine.name, "title": routine.title, "question": routine.question,
              "requirement": routine.requirement, "trials_planned": n, "trials_kept": len(rows),
              "stopped_early": stopped, "error": None if error is None else repr(error),
              "verdict": verdict(criteria, complete), "criteria": criteria,
              "conditions": {"start": start, "end": end}, "options": ctx.options, "state": ctx.state}
    _write(out, routine, rows, result)
    for line in summary_lines(result, rows, routine):
        say(line)
    say(f"\nwrote {out}")
    if error is not None:
        raise error
    return result


def _fmt(v) -> str:
    return "--" if v is None or v == "" else (f"{v:.4g}" if isinstance(v, float) else str(v))


def summary_lines(result: dict, rows: list[dict], routine: Routine) -> list[str]:
    s, e = result["conditions"]["start"], result["conditions"]["end"]
    lines = [f"\n{result['title'] or result['routine']}: {result['verdict']}",
             f"  question: {result['question']}"]
    if result["requirement"]:
        lines.append(f"  verifies: {result['requirement']}")
    lines.append(f"  trials: {result['trials_kept']} kept of {result['trials_planned']} planned"
                 + (" (stopped early)" if result["stopped_early"] else "")
                 + (f"; error: {result['error']}" if result["error"] else ""))
    lines += ["", "criteria"] + [f"  {'PASS' if c['passed'] else 'FAIL'}  {c['text']}" for c in result["criteria"]]
    if not result["criteria"]:
        lines.append("  none: a characterization (the numbers are the result)")
    lines += ["", "trials", "  " + "  ".join(["trial", *routine.fields])]
    lines += ["  " + "  ".join([str(r["trial"]), *(_fmt(r.get(k)) for k in routine.fields)]) for r in rows]
    lines += ["", f"conditions: {s['time']}, commit {s['commit'] or '--'}"
              + (" (with uncommitted changes)" if s["uncommitted_changes"] else "")
              + f", host {s['host']}, tester {s['tester'] or '--'}",
              f"  battery {_fmt(s['battery_v'])} -> {_fmt(e['battery_v'])} V, temperature "
              f"{_fmt(s['temp_c'])} -> {_fmt(e['temp_c'])} C, throttled {s['throttled'] or '--'} -> {e['throttled'] or '--'}"]
    if s["notes"]:
        lines.append(f"  notes: {s['notes']}")
    return lines


def _write(out: Path, routine: Routine, rows: list[dict], result: dict) -> None:
    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["trial", "t_s", *routine.fields], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "results.json", "w") as f:
        json.dump({**result, "rows": rows}, f, indent=2, default=str)
    (out / "summary.txt").write_text("\n".join(summary_lines(result, rows, routine)).lstrip("\n") + "\n")


def default_out_dir(name: str, runs_dir=None) -> Path:
    from src.params import RUNS_DIR
    return Path(runs_dir or RUNS_DIR) / f"routine_{name}_{time.strftime('%Y%m%d_%H%M%S')}"

