"""
plot_runs.py

Purpose:
    Generate static comparison plots from a frames.csv (+ optional
    contours.csv) produced by run_pipeline.py's --csv logging, or from
    combine_runs.py's combined output.

    Single-run and multi-run inputs are handled the same way: if a
    run_id column is present (combine_runs.py output), each run gets its
    own line/color and a legend; if not (a raw run_pipeline.py CSV pair),
    the whole file is treated as one run.

Usage:
    python3 vision_stack/scripts/plot_runs.py \\
        --frames vision_stack/logs/T3Logs/OD/oa65_dh12__20260829-0047_frames.csv \\
        --contours vision_stack/logs/T3Logs/OD/oa65_dh12__20260829-0047_contours.csv \\
        --out vision_stack/logs/plots/OD_oa65_dh12

    --out is a DIRECTORY (created if missing), so plots from different
    runs/prefixes land in their own folder instead of colliding in one
    flat directory. Produces, inside that directory:
        timing.png              frame_time_ms per frame, budget-exceeded frames marked
        offset_confidence.png   lane offset + lane_confidence over time
        tags.png                failure-mode tag totals, bar chart
        contours.png            area vs aspect, accepted vs rejected (only if --contours given)
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: no display needed, just write PNGs
import matplotlib.pyplot as plt

# =============================================================================
# Constants
# =============================================================================
TAG_COLUMNS = [
    "pole_misclassified", "wall_edge_detected", "dashed_reject_center",
    "anchor_wrong_half", "dashed_line_dropped",
]


def _runs(df: pd.DataFrame):
    """
    Splits into (run_id, sub_df) pairs. Falls back to a single implicit
    run when run_id isn't present (a raw, non-combined CSV)
    """
    if "run_id" in df.columns:
        return list(df.groupby("run_id"))
    return [("run", df)]


def _multi_run(df: pd.DataFrame) -> bool:
    return "run_id" in df.columns and df["run_id"].nunique() > 1

# =============================================================================
# Plots
# =============================================================================
def plot_timing(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for run_id, g in _runs(df):
        ax.plot(g["frame_id"], g["frame_time_ms"], linewidth=1, label=run_id)
        exceeded = g[g["budget_exceeded"] == 1]
        if not exceeded.empty:
            ax.scatter(exceeded["frame_id"], exceeded["frame_time_ms"],
                       color="red", s=12, zorder=5)
    ax.set_xlabel("frame_id")
    ax.set_ylabel("frame_time_ms")
    ax.set_title("Frame time per frame (red = budget exceeded)")
    if _multi_run(df):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timing.png"), dpi=150)
    plt.close(fig)


def plot_offset_confidence(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for run_id, g in _runs(df):
        axes[0].plot(g["frame_id"], g["offset"], linewidth=1, label=run_id)
        axes[1].plot(g["frame_id"], g["lane_confidence"], linewidth=1, label=run_id)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("offset")
    axes[1].set_ylabel("lane_confidence")
    axes[1].set_xlabel("frame_id")
    if _multi_run(df):
        axes[0].legend(fontsize=8)
    fig.suptitle("Lane offset and confidence over time")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "offset_confidence.png"), dpi=150)
    plt.close(fig)


def plot_tags(df: pd.DataFrame, out_dir: str) -> None:
    rows = []
    for run_id, g in _runs(df):
        row = {"run_id": run_id}
        for col in TAG_COLUMNS:
            row[col] = g[col].sum()
        rows.append(row)
    tag_df = pd.DataFrame(rows).set_index("run_id")

    fig, ax = plt.subplots(figsize=(9, 5))
    tag_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("total occurrences")
    ax.set_title("Failure-mode tag totals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tags.png"), dpi=150)
    plt.close(fig)


def plot_contours(cdf: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    accepted = cdf[cdf["accepted"] == 1]
    rejected = cdf[cdf["accepted"] == 0]
    ax.scatter(rejected["aspect"], rejected["area"], s=6, alpha=0.4,
               color="tab:red", label="rejected")
    ax.scatter(accepted["aspect"], accepted["area"], s=6, alpha=0.4,
               color="tab:green", label="accepted")
    ax.set_xlabel("aspect ratio")
    ax.set_ylabel("area")
    ax.set_yscale("log")
    ax.set_title("Contour area vs aspect ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "contours.png"), dpi=150)
    plt.close(fig)

# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot a frames.csv (+ optional contours.csv), single-run or combined"
    )
    ap.add_argument("--frames", required=True, metavar="PATH",
                    help="a *_frames.csv from run_pipeline.py or combine_runs.py")
    ap.add_argument("--contours", default=None, metavar="PATH",
                    help="matching *_contours.csv (optional — skips the contour plot if omitted)")
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="directory to write plots into (created if missing) — "
                         "e.g. vision_stack/logs/plots/OD_oa65_dh12")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fdf = pd.read_csv(args.frames)
    plot_timing(fdf, args.out)
    plot_offset_confidence(fdf, args.out)
    plot_tags(fdf, args.out)

    if args.contours:
        cdf = pd.read_csv(args.contours)
        plot_contours(cdf, args.out)

    print(f"Plots written to {args.out}/")


if __name__ == "__main__":
    main()