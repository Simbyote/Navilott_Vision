"""Command line: python3 -m src.routines NAME [--trials N] [--tester WHO] [--notes TEXT] [--out DIR] | --list"""
import argparse
import sys

from src.routines import ROUTINES
from src.routines.harness import Console, check_needs, default_out_dir, run_routine

_CLI_HELP = """\
Run a hardware test routine: it prompts every step, asks for any hand
measurement, and writes runs/routine_<name>_<time>/ with every trial, the
conditions and a PASS / FAIL verdict. At any prompt, q stops and keeps what's
done. docs/guides/routines.md has each routine; make routines lists them.
"""


def parse_settings(pairs: list[str], known: dict) -> dict:
    """
    --set KEY=VALUE pairs as a dict, numbers as numbers, checked against
    the routine's settings ({name: what it is}).

    Raises:
        ValueError: A pair without "=", or a setting the routine doesn't have.
    """
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or not key:
            raise ValueError(f"{pair!r}: expected KEY=VALUE")
        if key not in known:
            raise ValueError(f"no setting {key!r}; this routine has: " + (", ".join(sorted(known)) or "none"))
        try:
            out[key] = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
        except ValueError:
            out[key] = raw
    return out


def main(argv=None, console=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m src.routines", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="the routine")
    ap.add_argument("--list", action="store_true", help="list the routines")
    ap.add_argument("--trials", type=int, default=None, metavar="N", help="trials (default: the routine's)")
    ap.add_argument("--tester", default=None, help="who runs it (asked if not given)")
    ap.add_argument("--notes", default="", help="anything about this run worth keeping (lighting, mat...)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a routine's own setting, e.g. --set stage_s=30 (each routine lists its own)")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    console = console or Console()
    if args.list or not args.name:
        console.say("routines (make routine-<name>):")
        for name, r in sorted(ROUTINES.items()):
            console.say(f"  {name:<16} {r.question}" + (f"  [{r.requirement}]" if r.requirement else ""))
            for key, what in r.settings.items():
                console.say(f"  {'':<16}   --set {key}=...  {what}")
        return 0 if args.list else 2
    routine = ROUTINES.get(args.name)
    if routine is None:
        console.say(f"no routine {args.name!r}; make routines lists them")
        return 2
    try:
        options = parse_settings(args.set, routine.settings)
    except ValueError as exc:
        console.say(str(exc))
        return 2
    missing = check_needs(routine.needs)
    if missing:
        console.say("can't start: " + "; ".join(missing))
        return 2
    tester = args.tester if args.tester is not None else console.ask("Your name", default="")
    out = args.out or default_out_dir(routine.name)
    result = run_routine(routine(), console, out, args.trials, tester, args.notes, options)
    return {"PASS": 0, "RECORDED": 0}.get(result["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
