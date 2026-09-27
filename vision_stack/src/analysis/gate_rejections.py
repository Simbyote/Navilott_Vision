"""Gate rejections: which filter throws away the most candidates, per detector.

Purpose:
    Each detector passes candidates through a chain of gates (area, aspect,
    intensity, ...), and every candidate lands in exactly one bucket: the
    gate that rejected it, or accepted. When a detector finds too little,
    the gate with the largest share is the first threshold to look at; when
    it finds too much, the accepted share says how permissive the chain is.

Main package:
    analyze()            per group: candidates seen, and per gate the total,
                         share of seen and frames it fired on, largest first
    gate_rejections.png  one bar chart per group, largest gate at the top

Input:
    geometry_timing.csv (groups lane, sign) and color_timing.csv (groups red,
    yellow, green) from the hardware tests; any CSV works whose groups have a
    <group>_seen column and <group>_<gate> count columns. Give files, run
    folders (searched recursively) or nothing for the newest of each.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table

CSV_NAMES = ("geometry_timing.csv", "color_timing.csv")
ACCEPTED = "accepted"

_CLI_HELP = """\
Which gate rejects the most candidates, per detector. Writes
gate_rejections.png and gate_rejections.json next to the first input.

Run from vision_stack/, venv active:
    python3 -m src.analysis.gate_rejections                       newest of each
    python3 -m src.analysis.gate_rejections artifacts/<run>       one hardware run
    python3 -m src.analysis.gate_rejections a.csv b.csv           specific files
"""


def groups(columns) -> dict:
    """{group: [gate columns]} for every <group>_seen column; *_ms columns are never gates."""
    out = {}
    for c in columns:
        if c.endswith("_seen"):
            g = c[:-len("_seen")]
            out[g] = [k for k in columns
                      if k.startswith(g + "_") and k != c and not k.endswith("_ms")]
    return out


def analyze(table: Table) -> dict:
    """
    Outputs:
        {group: {
            "frames", "seen" (total candidates),
            "frames_with_candidates",
            "gates": [{"gate", "count", "share", "frames"}] rejections by
                     count, largest first, then accepted last,
            "unbucketed": seen minus the bucket totals (0 when the counts
                          are consistent),
        }}
    """
    found = groups(table.columns)
    if not found:
        raise ValueError(f"{table.path.name}: no <group>_seen columns")
    out = {}
    for g, cols in found.items():
        seen = np.nan_to_num(table.numeric(f"{g}_seen"))
        total_seen = float(seen.sum())
        gates = []
        for c in cols:
            v = np.nan_to_num(table.numeric(c))
            gates.append({"gate": c[len(g) + 1:], "count": int(v.sum()),
                          "share": float(v.sum() / total_seen) if total_seen else 0.0,
                          "frames": int(np.count_nonzero(v))})
        rejects = sorted((x for x in gates if x["gate"] != ACCEPTED), key=lambda x: -x["count"])
        accepted = [x for x in gates if x["gate"] == ACCEPTED]
        out[g] = {
            "frames": len(table),
            "seen": int(total_seen),
            "frames_with_candidates": int(np.count_nonzero(seen)),
            "gates": rejects + accepted,
            "unbucketed": int(total_seen - sum(x["count"] for x in gates)),
        }
    return out


def figure(results: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None or not results:
        return None
    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 1.2 + 0.42 * sum(
        len(results[n]["gates"]) for n in names) + 0.6 * len(names)), squeeze=False)
    for ax, g in zip(axes[:, 0], names):
        r = results[g]
        gates = list(reversed(r["gates"]))              # largest ends up on top
        labels = [x["gate"] for x in gates]
        shares = [100 * x["share"] for x in gates]
        colors = ["#2ca02c" if x["gate"] == ACCEPTED else "#1f77b4" for x in gates]
        ax.barh(labels, shares, color=colors)
        for i, x in enumerate(gates):
            ax.text(shares[i], i, f" {x['count']}", va="center", fontsize=7)
        ax.set_xlim(0, max(shares + [1]) * 1.18)
        ax.set_title(f"{g}: {r['seen']} candidates over {r['frames']} frames",
                     fontsize=9, loc="left")
        ax.set_xlabel("% of candidates seen")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def resolve(inputs) -> list:
    """CSV paths from files and folders, or the newest of each CSV_NAMES when empty."""
    paths = []
    for arg in inputs or []:
        p = Path(arg)
        if p.is_file():
            paths.append(p)
            continue
        hits = [common.find_csv(p, name) for name in CSV_NAMES if list(p.rglob(name))]
        if not hits:
            raise FileNotFoundError(f"no {' or '.join(CSV_NAMES)} in {p}")
        paths += hits
    if not inputs:
        for name in CSV_NAMES:
            try:
                paths.append(common.find_csv(None, name))
            except FileNotFoundError:
                pass
        if not paths:
            raise FileNotFoundError(f"no {' or '.join(CSV_NAMES)} found")
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.gate_rejections",
                                description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="CSVs or run folders (default: newest of each)")
    p.add_argument("--out", help="output folder (default: next to the first CSV)")
    args = p.parse_args(argv)

    try:
        results = {}
        paths = resolve(args.inputs)
        for path in paths:
            results.update(analyze(Table(path)))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else paths[0].parent
    common.write_json(out / "gate_rejections.json", results)
    fig = figure(results, "Gate rejections", out / "gate_rejections.png")

    for g, r in results.items():
        print(f"{g}: {r['seen']} candidates in {r['frames_with_candidates']}/{r['frames']} frames")
        for x in r["gates"]:
            print(f"  {x['gate']:<14} {x['count']:>7}  {100 * x['share']:5.1f}%  "
                  f"({x['frames']} frames)")
        if r["unbucketed"]:
            print(f"  WARNING: {r['unbucketed']} candidates in no bucket")
    print(f"wrote {out / 'gate_rejections.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
