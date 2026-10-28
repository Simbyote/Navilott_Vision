"""Obstacle map: several objects through the obstacle-check routine, side by side.

Purpose:
    One obstacle-check run says how one object read at each gap; the question
    is which objects the detector can't see, and why. This lays every run out
    as an object-by-gap grid: seen (with the cm it reported) or missed, and
    for a miss the gate that refused it, found by replaying the run's saved
    look at that gap through Phase 2. The tallest blob left after the opening,
    by gap, shows how far each object sits from the height gate.

Main package:
    analyze(runs) -> dict; obstacle_map.png and obstacle_map.json in the folder holding the runs.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from src.analysis import common
from src.analysis.common import Table

RUN_GLOB = "routine_obstacle-check_*"
GATE_WORDS = {"small": "too small", "narrow": "too narrow", "short": "too short", "far": "too far",
              None: "nothing left", "passed": "passed"}
SEEN, MISSED, ALARM, CLEAR_OK = "#1D5F8C", "#E8EDF1", "#B26A00", "#E8EDF1"
SHADES = ("#1E63B5", "#2A9A88", "#7B4F8C", "#C62828", "#5F6B76", "#8A6D3B", "#E0A100")

_CLI_HELP = """\
Several obstacle-check runs (one object each) as one grid: which gaps each
object was seen at, the cm it reported, and for a miss the gate that refused
it, by replaying the saved look through Phase 2 (--no-replay skips that).
Writes obstacle_map.png and obstacle_map.json in the folder.

Name each run's object with ARGS="--set object=can" when recording it, or end
its folder name in "(can)".

Run from vision_stack/:
    python3 -m src.analysis.obstacle_map runs/                  every obstacle-check run in it
    python3 -m src.analysis.obstacle_map runs/routine_obstacle-check_A runs/routine_obstacle-check_B
    make obstacle-map RUN=runs/obstacle_day
"""


def run_folders(args: list[str]) -> list[Path]:
    out = []
    for a in args or [str(common.RUNS_DIR)]:
        p = Path(a)
        if (p / "trials.csv").is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(q for q in p.glob(RUN_GLOB) if (q / "trials.csv").is_file())
    if not out:
        raise FileNotFoundError(f"no obstacle-check run (trials.csv) in {', '.join(args or [str(common.RUNS_DIR)])}")
    return out


def object_name(folder: Path) -> str:
    res = folder / "results.json"
    name = ""
    if res.is_file():
        name = (json.loads(res.read_text()).get("options") or {}).get("object") or ""
    if not name:
        m = re.search(r"\((.*)\)$", folder.name)
        name = m.group(1) if m else folder.name.replace("routine_obstacle-check_", "")
    return name.replace("_", " ")


def picture(folder: Path, gap: float) -> Path | None:
    """The last saved look at this gap, straight ahead (no lateral offset)."""
    found = sorted(folder.glob(f"attempt_*_{gap:g}cm.jpg"))
    return found[-1] if found else None


def replay(path: Path, chain) -> dict:
    """The tallest blob after the opening and the gate that decided it, through Phase 2."""
    frame = cv2.imread(str(path))
    if frame is None:
        return {"gate": "unreadable", "blob_h": None}
    res = chain(frame)
    blobs = res.obstacle_debug.get("blobs") or []
    if res.obstacle is not None and res.obstacle.detected:
        return {"gate": "passed", "blob_h": max((b["bbox"][3] for b in blobs), default=None)}
    if not blobs:
        return {"gate": None, "blob_h": 0}
    tallest = max(blobs, key=lambda b: b["bbox"][3])
    return {"gate": tallest["gate"], "blob_h": tallest["bbox"][3]}


def default_chain(frame):
    from src.config import MEASURED
    from src.phase2_linker import run_chain
    return run_chain(frame, 0, 0, MEASURED, parallel=False)


def _true(v: str) -> bool:
    return str(v).strip().lower() == "true"


def analyze(folders: list[Path], chain=default_chain, do_replay: bool = True) -> dict:
    objects, gaps = [], set()
    for f in folders:
        t = Table(f / "trials.csv")
        state, gap, lat = t.text("state"), t.numeric("gap_cm"), t.numeric("lateral_cm")
        voted, rep, err, clip = t.text("voted"), t.numeric("reported_cm"), t.numeric("error_cm"), t.numeric("clipped_pct")
        cells = {}
        for i in range(len(t)):
            if state[i] == "present" and np.isfinite(gap[i]) and not (np.isfinite(lat[i]) and lat[i] != 0):
                cells[float(gap[i])] = {"seen": _true(voted[i]),
                                        "reported_cm": None if not np.isfinite(rep[i]) else float(rep[i]),
                                        "error_cm": None if not np.isfinite(err[i]) else float(err[i]),
                                        "clipped": bool(np.isfinite(clip[i]) and clip[i] >= 50)}
                gaps.add(float(gap[i]))
        clear = [_true(voted[i]) for i in range(len(t)) if state[i] == "clear"]
        for g, c in cells.items():
            pic = picture(f, g) if do_replay else None
            c.update(replay(pic, chain) if pic else {"gate": "no picture" if do_replay else "", "blob_h": None})
        objects.append({"object": object_name(f), "folder": str(f), "cells": cells,
                        "clear_trials": len(clear), "false_alarms": sum(clear)})
    gaps = sorted(gaps)
    res = {"gaps": gaps, "objects": objects}
    seen = [g for o in objects for g, c in o["cells"].items() if c["seen"]]
    res["farthest_seen_cm"] = max(seen) if seen else None
    res["never_seen"] = [o["object"] for o in objects if not any(c["seen"] for c in o["cells"].values())]
    res["false_alarms"] = sum(o["false_alarms"] for o in objects)
    errs = [c["error_cm"] for o in objects for c in o["cells"].values() if c["seen"] and c["error_cm"] is not None]
    res["error_range_cm"] = [min(errs), max(errs)] if errs else None
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    if res["never_seen"]:
        out.append(f"never seen at any gap: {', '.join(res['never_seen'])}")
    gates = {}
    for o in res["objects"]:
        for c in o["cells"].values():
            if not c["seen"] and c.get("gate") not in ("", "no picture", "passed", "unreadable"):
                gates[GATE_WORDS.get(c["gate"], c["gate"])] = gates.get(GATE_WORDS.get(c["gate"], c["gate"]), 0) + 1
    if gates:
        top = max(gates.items(), key=lambda kv: kv[1])[0]
        why = ("nothing stood out from the floor or survived the opening: too little contrast or colour, or "
               "narrower than open_px" if top == "nothing left" else f"the '{top}' gate refused the most")
        out.append(f"misses by why: {', '.join(f'{k} {v}' for k, v in sorted(gates.items(), key=lambda kv: -kv[1]))}; "
                   + why)
    if res["false_alarms"]:
        out.append(f"{res['false_alarms']} false alarms on a clear lane")
    if res["error_range_cm"]:
        lo, hi = res["error_range_cm"]
        out.append(f"distance error when seen: {lo:+.1f} to {hi:+.1f} cm")
    return out


def figure(res: dict, title: str, out_path, min_height_px: float):
    plt = common.pyplot()
    if plt is None:
        return None
    objs, gaps = res["objects"], res["gaps"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 1.2 + 0.55 * len(objs) + 3.4),
                                  gridspec_kw={"height_ratios": [0.55 * len(objs) + 0.6, 3.2]})
    cols = [f"{g:g} cm" for g in gaps] + ["clear lane"]
    for i, o in enumerate(objs):
        for j, g in enumerate(gaps + [None]):
            if g is None:
                color = ALARM if o["false_alarms"] else CLEAR_OK
                text, tc = (f"{o['false_alarms']} alarms" if o["false_alarms"] else f"0 of {o['clear_trials']}"), "#1F2933"
            else:
                c = o["cells"].get(g)
                if c is None:
                    color, text, tc = "white", "", "#1F2933"
                elif c["seen"]:
                    color, tc = SEEN, "white"
                    text = "seen, clipped" if c["clipped"] or c["reported_cm"] is None else f"seen {c['reported_cm']:.1f}"
                else:
                    color, tc = MISSED, "#5F6B76"
                    text = GATE_WORDS.get(c.get("gate"), c.get("gate") or "missed") or "missed"
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.86, color=color))
            ax.text(j + 0.47, i + 0.43, text, ha="center", va="center", fontsize=7, color=tc)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(objs), 0)
    ax.set_xticks([j + 0.47 for j in range(len(cols))], cols, fontsize=8)
    ax.set_yticks([i + 0.43 for i in range(len(objs))], [o["object"] for o in objs], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, pad=22)
    any_h = False
    for k, o in enumerate(objs):
        hs = [o["cells"].get(g, {}).get("blob_h") for g in gaps]
        if any(h is not None for h in hs):
            any_h = True
            ax2.plot(gaps, [np.nan if h is None else h for h in hs], "-o", color=SHADES[k % len(SHADES)], lw=1.2,
                     ms=4, label=o["object"])
    ax2.axhline(min_height_px, color=ALARM, ls="--", lw=0.9)
    ax2.text(gaps[-1] if gaps else 0, min_height_px + 3, f"height gate {min_height_px:g} px", color=ALARM,
             fontsize=7, ha="right", va="bottom")
    ax2.set_xticks(gaps)
    ax2.set_xlabel("object's gap from the robot's front edge (cm)", fontsize=8)
    ax2.set_ylabel("tallest blob left (px; 0 = none)", fontsize=8)
    if any_h:
        ax2.legend(fontsize=7, frameon=False, ncol=3)
    else:
        ax2.text(0.5, 0.5, "no replay (--no-replay, or no saved looks)", transform=ax2.transAxes, ha="center",
                 color="#5F6B76", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.25)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    lines = [f"{len(res['objects'])} objects at {', '.join(f'{g:g}' for g in res['gaps'])} cm; farthest seen "
             f"{res['farthest_seen_cm'] if res['farthest_seen_cm'] is not None else '--'} cm"]
    for o in res["objects"]:
        cells = "  ".join(
            f"{g:g}:" + ("seen" if c["seen"] else GATE_WORDS.get(c.get("gate"), c.get("gate") or "missed"))
            for g, c in sorted(o["cells"].items()))
        lines.append(f"  {o['object']:<14} {cells}  | clear {o['false_alarms']} of {o['clear_trials']}")
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.obstacle_map", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="obstacle-check run folders, or folders holding them (default: runs/)")
    p.add_argument("--no-replay", action="store_true", help="don't replay the saved looks for the refusing gate")
    p.add_argument("--out", help="output folder (default: the first argument's folder)")
    args = p.parse_args(argv)
    try:
        folders = run_folders(args.runs)
        res = analyze(folders, do_replay=not args.no_replay)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    first = Path(args.runs[0]) if args.runs else Path(common.RUNS_DIR)
    out = Path(args.out) if args.out else (first if not (first / "trials.csv").is_file() else first.parent)
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "obstacle_map.json", res)
    from src.config import MEASURED
    fig = figure(res, f"Obstacles: {len(res['objects'])} objects", out / "obstacle_map.png",
                 MEASURED.obstacle.min_height_px)
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'obstacle_map.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
