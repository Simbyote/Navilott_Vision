"""Detection range: how far away a stop sign or traffic light is still detected, reliably.

Purpose:
    Navigation needs warning: the distance at which a sign is first seen,
    minus the robot's stopping distance, is its margin. Place the target at
    measured distances in front of the robot, record a short still run at
    each, and give the runs with their distances. The result is the
    detection rate at every distance and the reliable range: the farthest
    distance out to which every measured distance, starting from the
    nearest, is detected in at least --threshold of frames.

Main package:
    analyze()              per target and distance: frames, detection rate,
                           longest run of missed frames; reliable range,
                           farthest distance seen at all
    detection_range.png    detection rate against distance, per target,
                           with the threshold and the reliable range marked

Input:
    One run per distance, p3.csv (p2_stop, p2_traffic) or test_feature_fusion's
    fusion_timing.csv (n_stop_sign, n_traffic_light); a frame counts as
    detected when its count is above zero. Distances come from a manifest
    CSV with columns distance_cm,run (run relative to the manifest), or from
    --at DISTANCE_CM RUN, repeated. Measure to the target from the same
    point on the robot every time, such as the lens.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, runs_of

CSV_NAMES = ("p3.csv", "fusion_timing.csv")
TARGETS = {                 # target: count columns, in order of preference
    "stop_sign": ("p2_stop", "n_stop_sign"),
    "traffic_light": ("p2_traffic", "n_traffic_light"),
}
THRESHOLD = 0.9
WARMUP_FRAMES = 5
COLORS = {"stop_sign": "#d62728", "traffic_light": "#2ca02c"}

_CLI_HELP = """\
Detection rate against measured distance, and the reliable range, for
stop signs and traffic lights. Writes detection_range.png and
detection_range.json.

Record one still run per distance, then either list them in a manifest
(columns distance_cm,run) or pass them directly:
    python3 -m src.analysis.detection_range distances.csv
    python3 -m src.analysis.detection_range --at 20 runs/A --at 40 runs/B --at 60 runs/C
"""


def detections(table: Table, targets) -> dict:
    """{target: bool array, detected per frame} for each target the table has a column for."""
    out = {}
    for target in targets:
        col = table.first_with_values(*TARGETS[target])
        if col is not None:
            out[target] = np.nan_to_num(table.numeric(col)) > 0
    return out


def analyze(points, threshold: float = THRESHOLD) -> dict:
    """
    Inputs:
        points: [(distance_cm, {target: detected bool array})]
        threshold: detection rate that counts as reliable

    Outputs:
        {target: {
            "distances": [{"distance_cm", "frames", "rate", "longest_miss"}]
                         nearest first,
            "reliable_range_cm": farthest distance with every distance from
                                 the nearest out to it at or above threshold;
                                 None when the nearest already falls short,
            "farthest_seen_cm": farthest distance detected in any frame, or None,
            "reliable_distances_cm": every distance at or above threshold,
        }}
    """
    if not points:
        raise ValueError("no distances given")
    targets = sorted({t for _, d in points for t in d})
    if not targets:
        raise ValueError("no stop sign or traffic light columns in any run")
    out = {}
    for target in targets:
        rows = []
        for distance, det in sorted(points, key=lambda p: p[0]):
            if target not in det:
                continue
            hits = np.asarray(det[target], dtype=bool)
            if hits.size == 0:
                raise ValueError(f"{target} at {distance:g} cm has no frames")
            misses = [n for hit, _, n in runs_of(hits.tolist()) if not hit]
            rows.append({"distance_cm": float(distance), "frames": int(hits.size),
                         "rate": float(hits.mean()), "longest_miss": max(misses, default=0)})
        reliable = None
        for r in rows:
            if r["rate"] < threshold:
                break
            reliable = r["distance_cm"]
        seen = [r["distance_cm"] for r in rows if r["rate"] > 0]
        out[target] = {"distances": rows, "reliable_range_cm": reliable,
                       "farthest_seen_cm": max(seen) if seen else None,
                       "reliable_distances_cm": [r["distance_cm"] for r in rows if r["rate"] >= threshold]}
    return out


def figure(result: dict, threshold: float, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    for target, r in result.items():
        d = [x["distance_cm"] for x in r["distances"]]
        rate = [100 * x["rate"] for x in r["distances"]]
        color = COLORS.get(target, "#1f77b4")
        name = target.replace("_", " ")
        ax.plot(d, rate, marker="o", color=color, label=name)
        if r["reliable_range_cm"] is not None:
            ax.axvline(r["reliable_range_cm"], color=color, linestyle=":", linewidth=1)
            ax.text(r["reliable_range_cm"], 3, f" {name}\n reliable to {r['reliable_range_cm']:g} cm",
                    color=color, fontsize=7, va="bottom")
    ax.axhline(100 * threshold, color="black", linestyle="--", linewidth=1,
               label=f"{100 * threshold:.0f}% threshold")
    ax.set_ylim(0, 105)
    ax.set_xlabel("distance to target (cm)")
    ax.set_ylabel("frames detected (%)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.detection_range", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", nargs="?", help="CSV with columns distance_cm,run")
    p.add_argument("--at", nargs=2, action="append", metavar=("DISTANCE_CM", "RUN"), default=[],
                   help="one distance and its run; repeat per distance")
    p.add_argument("--targets", nargs="+", choices=list(TARGETS), default=None,
                   help="targets to report (default: those detected in any run, or all if none)")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help=f"detection rate that counts as reliable (default {THRESHOLD})")
    p.add_argument("--skip", type=int, default=WARMUP_FRAMES, help="drop the first N frames of each run")
    p.add_argument("--out", help="output folder (default: the manifest's, or the current one)")
    args = p.parse_args(argv)

    try:
        runs = common.read_manifest(args.manifest, "distance_cm") if args.manifest else []
        runs += [(float(d), Path(r)) for d, r in args.at]
        if not runs:
            raise ValueError("no distances: give a manifest or --at DISTANCE_CM RUN")
        points = [(d, detections(Table(common.find_csv(r, CSV_NAMES), args.skip), list(TARGETS)))
                  for d, r in runs]
        targets = args.targets or [t for t in TARGETS if any(det.get(t, np.zeros(0)).any()
                                                              for _, det in points)] or list(TARGETS)
        points = [(d, {t: v for t, v in det.items() if t in targets}) for d, det in points]
        res = analyze(points, args.threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (Path(args.manifest).parent if args.manifest else Path("."))
    common.write_json(out / "detection_range.json", {"threshold": args.threshold, "targets": res})
    fig = figure(res, args.threshold, "Detection range", out / "detection_range.png")

    for target, r in res.items():
        print(f"{target.replace('_', ' ')}:")
        for x in r["distances"]:
            print(f"  {x['distance_cm']:>6.0f} cm  {100 * x['rate']:5.1f}%  of {x['frames']} frames, "
                  f"longest miss {x['longest_miss']}")
        rel = r["reliable_range_cm"]
        print(f"  reliable to {rel:g} cm" if rel is not None
              else "  not reliable even at the nearest distance")
        if r["farthest_seen_cm"] is not None and r["farthest_seen_cm"] != rel:
            print(f"  seen at all out to {r['farthest_seen_cm']:g} cm")
    print(f"wrote {out / 'detection_range.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
