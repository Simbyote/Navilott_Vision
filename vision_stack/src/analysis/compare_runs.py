"""Compare two runs: every number that changed between them, largest change first.

Purpose:
    After a tuning change or a new commit, what moved? Every numeric value in
    two runs' JSON summaries becomes a dotted key (stage_ms.p95, ...) and those
    past a threshold are listed. Direction isn't judged; only the reader knows
    whether lower is better.

Main package:
    compare() -> base, new, delta and percent per key, written to compare.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from src.analysis.common import fmt

THRESHOLD_PCT = 10.0
SKIP_FILES = frozenset({"run_meta.json", "config.json"})

_CLI_HELP = """\
Every number that changed between two runs' JSON summaries, largest change
first. Writes compare.csv to --out (default: the current folder).

Run from vision_stack/, venv active:
    python3 -m src.analysis.compare_runs artifacts/<old> artifacts/<new>
    python3 -m src.analysis.compare_runs old/summary.json new/summary.json
"""


def flatten(obj, prefix: str = "") -> dict:
    """
    Numeric leaves as {dotted.key: float}, list items by index.

    Skips bools, strings, and number lists longer than 8 (frame indices, not summaries).
    """
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        if len(obj) > 8 and all(isinstance(v, (int, float)) for v in obj):
            return out
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix[:-1]] = float(obj)
    return out


def compare(base: dict, new: dict, threshold_pct: float = THRESHOLD_PCT) -> list:
    """
    Rows {key, base, new, delta, pct, flag} for every key, flagged first, then by size.

    pct is None when base is 0; a one-sided key is flagged with the other side None.
    """
    rows = []
    for key in sorted(set(base) | set(new)):
        b, n = base.get(key), new.get(key)
        if b is None or n is None:
            rows.append({"key": key, "base": b, "new": n, "delta": None, "pct": None, "flag": True})
            continue
        delta = n - b
        pct = None if b == 0 else 100.0 * delta / abs(b)
        flag = (abs(pct) >= threshold_pct) if pct is not None else delta != 0
        rows.append({"key": key, "base": b, "new": n, "delta": delta, "pct": pct, "flag": flag})
    # Within each group, changes with no percentage go on top
    rows.sort(key=lambda r: (not r["flag"],
                             -(abs(r["pct"]) if r["pct"] is not None else float("inf"))))
    return rows


def pairs(a: Path, b: Path) -> list:
    """[(label, base json, new json)] for two files, or matching paths under two folders."""
    if a.is_file() and b.is_file():
        return [(a.name, a, b)]
    if a.is_dir() and b.is_dir():
        out = []
        for fa in sorted(a.rglob("*.json")):
            rel = fa.relative_to(a)
            if fa.name in SKIP_FILES:
                continue
            fb = b / rel
            if fb.is_file():
                out.append((str(rel), fa, fb))
        if not out:
            raise FileNotFoundError(f"no JSON files at matching paths in {a} and {b}")
        return out
    raise FileNotFoundError("give two JSON files or two folders")


def commit_of(folder: Path) -> str:
    meta = folder / "run_meta.json"
    if not meta.is_file():
        return "unknown"
    git = json.loads(meta.read_text()).get("git") or {}
    return f"{git.get('commit') or 'unknown'}{' (dirty)' if git.get('dirty') else ''}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.compare_runs", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", help="older run: folder or JSON file")
    p.add_argument("new", help="newer run: folder or JSON file")
    p.add_argument("--threshold", type=float, default=THRESHOLD_PCT,
                   help=f"flag changes of at least this percent (default {THRESHOLD_PCT:g})")
    p.add_argument("--all", action="store_true", help="print unflagged keys too")
    p.add_argument("--out", default=".", help="folder for compare.csv (default: current)")
    args = p.parse_args(argv)

    a, b = Path(args.base), Path(args.new)
    try:
        matched = pairs(a, b)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if a.is_dir():
        print(f"base {a}  commit {commit_of(a)}")
        print(f"new  {b}  commit {commit_of(b)}")

    table = []
    for label, fa, fb in matched:
        rows = compare(flatten(json.loads(fa.read_text())), flatten(json.loads(fb.read_text())),
                       args.threshold)
        flagged = [r for r in rows if r["flag"]]
        print(f"\n{label}: {len(flagged)} of {len(rows)} values changed by "
              f">= {args.threshold:g}%")
        for r in (rows if args.all else flagged):
            pct = "" if r["pct"] is None else f"{r['pct']:+.1f}%"
            print(f"  {r['key']:<44} {fmt(r['base'], 3):>10} -> {fmt(r['new'], 3):>10}  {pct:>8}")
        table += [(label, r["key"], r["base"], r["new"], r["delta"], r["pct"], int(r["flag"]))
                  for r in rows]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "key", "base", "new", "delta", "pct", "flagged"])
        w.writerows(table)
    print(f"\nwrote {out / 'compare.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
