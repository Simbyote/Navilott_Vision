"""Every analysis a pytest --hardware run can feed, in one go.

Purpose:
    A hardware run (artifacts/<YYYYMMDD_HHMMSS>) holds one folder per test,
    several with a CSV some tool reads, and each tool takes one CSV per call.
    This finds every input under the run and runs each tool that reads it,
    so each figure lands next to the CSV it came from. Nothing new is
    computed: these are the tools' own outputs.

Main package:
    report(): hw_report.txt in the run folder, per tool and CSV the headline
    lines and the figures (plan() gives the (tool, CSV) pairs).

Flow:
    1. For each tool in TOOLS, every file under the run named one of its CSV_NAMES.
    2. tool.main([csv]); its printout kept, the figures and JSON it wrote listed.
    3. hw_report.txt.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

from src.analysis import common, gate_rejections, jitter, soak, stability, stage_timing

# Each tool and the CSVs it can read from a hardware run. nav.csv and p3.csv
# are left out: a hardware run has none, and nav_run reads navigation runs
TOOLS = {
    "stage_timing": (stage_timing, ("stage_timing.csv", "stages.csv")),
    "jitter": (jitter, ("frames.csv", "stage_timing.csv", "stages.csv")),
    "stability": (stability, ("lane_offset_timing.csv", "stages.csv")),
    "gate_rejections": (gate_rejections, gate_rejections.CSV_NAMES),
    "soak": (soak, ("system.csv",)),
}
OUTPUT_SUFFIXES = (".png", ".json")
REPORT = "hw_report.txt"

_CLI_HELP = """\
Run every analysis tool on every CSV of a pytest --hardware run, and list
what each wrote in hw_report.txt in the run folder.

    python3 -m src.analysis.hw_report                         newest artifacts/ run
    python3 -m src.analysis.hw_report artifacts/20261012_172045
    make hw-report RUN=artifacts/20261012_172045
"""


def newest_run(root: Path = common.ARTIFACTS_DIR) -> Path:
    """The newest artifacts/<stamp> folder."""
    runs = sorted((p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: p.name) if Path(root).is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no hardware runs in {root}")
    return runs[-1]


def _fits(name: str, csv: Path) -> bool:
    """
    Whether the tool can use this CSV. Only jitter shares a name with an
    unrelated file: frame_meta's frames.csv holds sensor timestamps, not a
    loop's intervals, and has no column jitter reads.
    """
    if name != "jitter":
        return True
    try:
        return common.interval_ms(common.Table(csv, 0)) is not None
    except (OSError, ValueError):
        return False


def plan(run: Path) -> list[tuple[str, Path]]:
    """(tool name, CSV) for every CSV under run that a tool reads and can use, in TOOLS order, then path order."""
    out = []
    for name, (_, names) in TOOLS.items():
        for csv_name in names:
            out += [(name, p) for p in sorted(Path(run).rglob(csv_name)) if _fits(name, p)]
    return out


def _outputs(folder: Path) -> dict:
    return {p: p.stat().st_mtime for p in folder.iterdir() if p.suffix in OUTPUT_SUFFIXES}


def run_one(name: str, csv: Path) -> dict:
    """One tool on one CSV: {"tool", "csv", "code", "lines" (its printout), "wrote" (new or updated files)}."""
    module = TOOLS[name][0]
    before = _outputs(csv.parent)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            code = module.main([str(csv)])
        except Exception as e:                  # one tool's crash shouldn't stop the rest
            print(f"ERROR: {type(e).__name__}: {e}")
            code = 1
    after = _outputs(csv.parent)
    wrote = sorted(p for p, t in after.items() if before.get(p) != t)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip() and not ln.startswith("wrote ")]
    return {"tool": name, "csv": csv, "code": code, "lines": lines, "wrote": wrote}


def report(run: Path) -> tuple[list[dict], Path]:
    """Every planned run; writes REPORT in the run folder. (results, report path)."""
    run = Path(run)
    results = [run_one(name, csv) for name, csv in plan(run)]
    text = [f"hardware run {run.name}: {len(results)} analyses", ""]
    for r in results:
        rel = r["csv"].relative_to(run)
        text.append(f"[{r['tool']}] {rel}  " + ("ok" if r["code"] == 0 else "FAILED"))
        text += [f"    {ln}" for ln in r["lines"][-8:]]
        text += [f"    -> {p.relative_to(run)}" for p in r["wrote"]]
        text.append("")
    path = run / REPORT
    path.write_text("\n".join(text) + "\n")
    return results, path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.hw_report", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="artifacts/<stamp> folder (default: newest)")
    args = p.parse_args(argv)
    try:
        run = Path(args.run) if args.run else newest_run()
        if not run.is_dir():
            raise FileNotFoundError(f"{run} is not a folder")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    results, path = report(run)
    figures = sum(p.suffix == ".png" for r in results for p in r["wrote"])
    failed = [r for r in results if r["code"] != 0]
    for r in results:
        print(f"{'ok ' if r['code'] == 0 else 'ERR'} {r['tool']:<16} {r['csv'].relative_to(run)}"
              + ("" if r["code"] == 0 else f"  ({r['lines'][-1] if r['lines'] else 'no output'})"))
    print(f"\n{len(results)} analyses, {figures} figures, {len(failed)} failed; see {path}")
    return 0 if results and not failed else (1 if not results else 2)


if __name__ == "__main__":
    sys.exit(main())
