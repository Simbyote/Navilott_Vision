"""
combine_runs.py

Purpose:
    Merge many bench-test CSV pairs (frames + contours) from separate runs
    into one combined dataset for side-by-side comparison, without typing
    a long file list on the command line: point it at zip archives,
    directories, or individual CSVs and it discovers everything inside.

    Run identity is derived from the data itself, not from folder-name
    parsing: every frames.csv row already carries
    fix_roi_inset..fix_anchor_halves, so the active combo (e.g. "OD") is
    reconstructed directly from those columns. Only the human-readable
    paramtag/timestamp label is taken from the filename (best-effort,
    falls back to the raw stem if it doesn't match the
    <paramtag>__<timestamp> convention from sweep_fix.sh/sweep_fix_combo.sh).

Usage:
    python3 vision_stack/scripts/combine_runs.py --out combined/od_calib \\
        vision_stack/logs/T3Logs/OD.zip \\
        vision_stack/logs/T3Logs/O \\
        some_loose_run_frames.csv

    Produces:
        <out>_frames.csv     every input frames.csv concatenated, tagged
                              with run_id/combo/paramtag/run_timestamp
        <out>_contours.csv   same, for contours (join back on
                              run_id + frame_id). Omitted if no run had
                              DEBUG_CONTOURS on
        <out>_summary.csv    one row per run: quick side-by-side stats
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

# =============================================================================
# Constants
# =============================================================================
FRAMES_SUFFIX = "_frames.csv"
CONTOURS_SUFFIX = "_contours.csv"

# Canonical order matches Config.FIX_NAMES / flags_str() in config.py
FIX_COLS_CANONICAL = [
    ("fix_roi_inset", "I"),
    ("fix_trapezoid_mask", "T"),
    ("fix_orientation_filt", "O"),
    ("fix_dashed_dilate", "D"),
    ("fix_anchor_halves", "A"),
]

# Matches the <paramtag>__<timestamp> convention from sweep_fix.sh /
# sweep_fix_combo.sh, e.g. "oa65_dh12__20260829-0047"
PARAMTAG_RE = re.compile(r"^(?P<paramtag>.+)__(?P<timestamp>\d{8}-\d{4})$")

# =============================================================================
# Discovery
# =============================================================================
@dataclass
class FoundFile:
    """
    One CSV location (disk path or an entry inside a zip), opened lazily so
    nothing is read from disk/zip until it's actually needed

    label: human-readable origin, shown in warnings
    opener: () -> file-like object readable by pandas.read_csv
    """
    label: str
    opener: Callable[[], object]


def _iter_disk_csvs(root: str):
    if os.path.isfile(root):
        yield root
    else:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(FRAMES_SUFFIX) or f.endswith(CONTOURS_SUFFIX):
                    yield os.path.join(dirpath, f)


def _iter_zip_csvs(zip_path: str):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(FRAMES_SUFFIX) or name.endswith(CONTOURS_SUFFIX):
                yield name


def discover(inputs: List[str]) -> Dict[str, dict]:
    """
    Purpose:
        Walk every input (zip / directory / loose file) and pair up
        *_frames.csv with its matching *_contours.csv

    Outputs:
        dict keyed by a unique stem_key (path-based, so two runs that
        happen to share a paramtag/timestamp in different folders don't
        collide) -> {"stem": str, "frames": FoundFile|None, "contours": FoundFile|None}
    """
    runs: Dict[str, dict] = {}

    def register(kind: str, stem_key: str, stem: str, found: FoundFile) -> None:
        entry = runs.setdefault(stem_key, {"stem": stem, "frames": None, "contours": None})
        entry[kind] = found

    for inp in inputs:
        if inp.lower().endswith(".zip"):
            for internal in _iter_zip_csvs(inp):
                is_frames = internal.endswith(FRAMES_SUFFIX)
                suffix_len = len(FRAMES_SUFFIX) if is_frames else len(CONTOURS_SUFFIX)
                stem_key = internal[:-suffix_len]        # keeps internal folder for uniqueness
                stem = os.path.basename(stem_key)

                def make_opener(zp: str = inp, nm: str = internal) -> Callable[[], object]:
                    def _open():
                        with zipfile.ZipFile(zp) as zf:
                            return io.BytesIO(zf.read(nm))
                    return _open

                register("frames" if is_frames else "contours", stem_key, stem,
                          FoundFile(label=f"{inp}:{internal}", opener=make_opener()))
        else:
            for path in _iter_disk_csvs(inp):
                fname = os.path.basename(path)
                is_frames = fname.endswith(FRAMES_SUFFIX)
                suffix_len = len(FRAMES_SUFFIX) if is_frames else len(CONTOURS_SUFFIX)
                stem = fname[:-suffix_len]
                stem_key = os.path.join(os.path.dirname(path), stem)

                def make_opener(p: str = path) -> Callable[[], object]:
                    return lambda: open(p, "rb")

                register("frames" if is_frames else "contours", stem_key, stem,
                          FoundFile(label=path, opener=make_opener()))

    return runs

# =============================================================================
# Combine
# =============================================================================
def _derive_combo(frames_df: pd.DataFrame) -> str:
    """
    Reconstruct the active-fix combo letters (canonical I T O D A order)
    directly from the frames data, rather than trusting a folder name
    """
    if frames_df.empty:
        return "NONE"
    first = frames_df.iloc[0]
    combo = "".join(letter for col, letter in FIX_COLS_CANONICAL if bool(first.get(col, 0)))
    return combo or "NONE"


def build_combined(runs: Dict[str, dict]):
    """
    Purpose:
        Load every discovered run, tag each row with run identity columns,
        and concatenate into combined frames/contours DataFrames plus a
        one-row-per-run summary table

    Outputs:
        (combined_frames, combined_contours, summary) — combined_contours
        is empty if no run had a contours file
    """
    frame_dfs: List[pd.DataFrame] = []
    contour_dfs: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    for stem_key, entry in sorted(runs.items()):
        stem = entry["stem"]
        f_found: Optional[FoundFile] = entry["frames"]
        c_found: Optional[FoundFile] = entry["contours"]

        if f_found is None:
            print(f"WARN: '{stem_key}' has a contours file but no matching "
                  f"frames file — skipped", file=sys.stderr)
            continue

        fdf = pd.read_csv(f_found.opener())
        combo = _derive_combo(fdf)

        m = PARAMTAG_RE.match(stem)
        paramtag, timestamp = (m.group("paramtag"), m.group("timestamp")) if m else (stem, "")
        run_id = f"{combo}__{stem}"

        fdf.insert(0, "run_id", run_id)
        fdf.insert(1, "combo", combo)
        fdf.insert(2, "paramtag", paramtag)
        fdf.insert(3, "run_timestamp", timestamp)
        frame_dfs.append(fdf)

        summary_rows.append({
            "run_id": run_id,
            "combo": combo,
            "paramtag": paramtag,
            "run_timestamp": timestamp,
            "n_frames": len(fdf),
            "avg_frame_time_ms": round(fdf["frame_time_ms"].mean(), 2),
            "max_frame_time_ms": round(fdf["frame_time_ms"].max(), 2),
            "budget_exceeded_pct": round(100 * fdf["budget_exceeded"].mean(), 2),
            "avg_offset": round(fdf["offset"].mean(), 4),
            "avg_lane_confidence": round(fdf["lane_confidence"].mean(), 3),
            "intersection_trigger_count": int(fdf["intersection_trigger"].sum()),
            "pole_misclassified_total": int(fdf["pole_misclassified"].sum()),
            "wall_edge_detected_total": int(fdf["wall_edge_detected"].sum()),
            "dashed_reject_center_total": int(fdf["dashed_reject_center"].sum()),
            "anchor_wrong_half_count": int(fdf["anchor_wrong_half"].sum()),
            "dashed_line_dropped_count": int(fdf["dashed_line_dropped"].sum()),
        })

        if c_found is not None:
            cdf = pd.read_csv(c_found.opener())
            cdf.insert(0, "run_id", run_id)
            cdf.insert(1, "combo", combo)
            contour_dfs.append(cdf)
        else:
            print(f"NOTE: '{stem_key}' has no contours file "
                  f"(DEBUG_CONTOURS was likely off) — frames only", file=sys.stderr)

    combined_frames = pd.concat(frame_dfs, ignore_index=True) if frame_dfs else pd.DataFrame()
    combined_contours = pd.concat(contour_dfs, ignore_index=True) if contour_dfs else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return combined_frames, combined_contours, summary

# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Combine bench-run CSV pairs (zips, directories, or loose files) "
                     "into one dataset for side-by-side comparison"
    )
    ap.add_argument(
        "inputs", nargs="+",
        help="zip archives, directories, and/or individual *_frames.csv/*_contours.csv "
             "files, in any mix",
    )
    ap.add_argument(
        "--out", required=True, metavar="PREFIX",
        help="writes PREFIX_frames.csv, PREFIX_contours.csv (if any), PREFIX_summary.csv",
    )
    args = ap.parse_args()

    runs = discover(args.inputs)
    if not runs:
        print("No *_frames.csv / *_contours.csv files found in the given inputs.",
              file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    combined_frames, combined_contours, summary = build_combined(runs)

    combined_frames.to_csv(f"{args.out}_frames.csv", index=False)
    if not combined_contours.empty:
        combined_contours.to_csv(f"{args.out}_contours.csv", index=False)
    summary.to_csv(f"{args.out}_summary.csv", index=False)

    print(f"Combined {len(summary)} run(s) -> {args.out}_frames.csv "
          f"({len(combined_frames)} rows)")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()