"""Light clips: parked recordings of the course light cycling, scored per colour and per change, for TP-03.

Purpose:
    D3 asks whether each light state is read right at the stop line. Instead
    of setting one state at a time (detect-range, target=light), the robot is
    parked at a gap for ~30 s while the light cycles, recording frames; the
    frames are labelled with what the light really showed (make label). Each
    clip is then scored per colour, frame by frame (Phase 2's colour) and as
    voted (the drive state navigation acts on: yellow voted stop is right,
    agreed 2026-11-02), and every change of the light is timed: how many
    frames the vote took to follow it.

    The vote needs 2 of its 3 frames to change, so the frames just after a
    change aren't scored as voted (evaluate.py's settle); they are what the
    response time measures. A change to green or off is a release: navigation
    holds red RED_MEMORY_MS by design, so it is reported, not judged.

Main package:
    score_clip(readings, labels) -> dict with per-colour rows and changes;
    light_clips.csv (one row per clip and colour, TP-03's columns A-K) and
    light_changes.csv in --out.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from src.analysis import common
from src.analysis.evaluate import (FRAME_MS, LIGHTS, UNCHECKED, VOTED_FOR, find_frames, frames_in, read_labels,
                                   readings_from_log, replay, truth_for)

STOPPING = ("red", "yellow")
TO_STOP, RELEASE, STOP_TO_STOP = "to stop", "release", "stop to stop"
CLIP_FIELDS = ("run", "condition", "gap_cm", "clip_s", "changes", "light", "frames", "frame_right_pct",
               "wrong_colour_frames", "voted_right_pct", "voted_wrong_frames")
CHANGE_FIELDS = ("run", "condition", "gap_cm", "fid", "from", "to", "kind", "response_frames", "response_ms")

_CLI_HELP = """\
Parked recordings of the light cycling (one gap and lighting per clip),
labelled with make label, scored per colour and per change for TP-03.

Record a clip (motors off), label it, then score:
    python3 -m src.navigation_linker --camera --no-motors --max-run-s 30
    make label RUN=runs/nav_<time> ARGS="--draft"     then make label RUN=... to check
    python3 -m src.analysis.light_clips runs/light_day
    make light-clips RUN=runs/light_day

Set range_cm (the gap) and condition (the lighting) on every labels.csv row.
Clips are replayed from their frames; --from-log scores p3.csv instead (no
frames needed, but the log keeps only the vote's colour, so per-frame and
voted then agree by construction).
"""


def ok_vote(truth: str, vote: str) -> bool:
    """What navigation was told is right: yellow voted stop passes (agreed 2026-11-02)."""
    return vote == VOTED_FOR[truth] or (truth == "yellow" and vote == "stop")


def wrong_vote(truth: str, vote: str) -> bool:
    """The hazard: going on red or yellow, or stopping for green or off."""
    return (truth in STOPPING and vote == "go") or (truth not in STOPPING and vote in ("stop", "caution"))


def kind_of(before: str, after: str) -> str:
    if after in STOPPING:
        return STOP_TO_STOP if before in STOPPING else TO_STOP
    return RELEASE if before in STOPPING else "go to go"


def _mode(values):
    c = Counter(v for v in values if v not in (None, ""))
    return c.most_common(1)[0][0] if c else None


def score_clip(readings: list[dict], labels: list[dict], settle: int = 2, run: str = "") -> dict:
    """readings in evaluate.replay()'s form, labels from read_labels (unchecked rows already out)."""
    pairs = []                                  # (fid, truth, said colour or None, vote)
    for r in readings:
        lb = truth_for(labels, r["fid"])
        if lb is not None and lb.get("light") in LIGHTS:
            pairs.append((r["fid"], lb["light"], r.get("light"), r.get("vote_drive") or "go", lb))
    gap = _mode(p[4].get("range_cm") for p in pairs)
    cond = _mode(p[4].get("condition") for p in pairs) or ""
    changes, since, voted = [], settle + 1, []
    for i, (fid, truth, said, vote, _) in enumerate(pairs):
        if i and (truth != pairs[i - 1][1] or fid != pairs[i - 1][0] + 1):
            since = 0
            if fid == pairs[i - 1][0] + 1 and truth != pairs[i - 1][1]:
                follow = next((j for j in range(i, len(pairs)) if ok_vote(truth, pairs[j][3])
                               or pairs[j][1] != truth), None)
                reached = follow is not None and pairs[follow][1] == truth and ok_vote(truth, pairs[follow][3])
                n = pairs[follow][0] - fid if reached else None
                changes.append({"run": run, "condition": cond, "gap_cm": gap, "fid": fid, "from": pairs[i - 1][1],
                                "to": truth, "kind": kind_of(pairs[i - 1][1], truth), "response_frames": n,
                                "response_ms": None if n is None else n * FRAME_MS})
        since += 1
        if since > settle:
            voted.append((truth, vote))
    rows = []
    for colour in LIGHTS:
        mine = [p for p in pairs if p[1] == colour]
        if not mine:
            continue
        right = sum((p[2] == colour) if colour != "off" else p[2] is None for p in mine)
        wrong = sum(p[2] not in (None, colour) for p in mine)
        v = [vt for t, vt in voted if t == colour]
        rows.append({"run": run, "condition": cond, "gap_cm": gap,
                     "clip_s": round(len(readings) * FRAME_MS / 1000, 1), "changes": len(changes), "light": colour,
                     "frames": len(mine), "frame_right_pct": round(100 * right / len(mine), 1),
                     "wrong_colour_frames": wrong,
                     "voted_right_pct": round(100 * sum(ok_vote(colour, x) for x in v) / len(v), 1) if v else None,
                     "voted_wrong_frames": sum(wrong_vote(colour, x) for x in v)})
    return {"rows": rows, "changes": changes}


def clip_folders(args) -> list[Path]:
    found = []
    for a in args or [str(common.RUNS_DIR)]:
        p = Path(a)
        if (p / "labels.csv").is_file():
            found.append(p)
        elif p.is_dir():
            found += sorted(x.parent for x in p.rglob("labels.csv"))
    if not found:
        raise FileNotFoundError(f"no labelled clip (labels.csv) in {', '.join(args or [str(common.RUNS_DIR)])}")
    return sorted(set(found))


def readings_for(folder: Path, from_log: bool) -> list[dict]:
    if from_log:
        log = folder / "p3.csv"
        if not log.is_file():
            raise FileNotFoundError(f"no p3.csv in {folder}")
        return readings_from_log(log)
    return replay(frames_in(find_frames(folder)), sequence=True)


def _write(path: Path, fields, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.light_clips", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="clip folders (labels.csv inside), or folders holding them")
    p.add_argument("--from-log", action="store_true", help="score p3.csv instead of replaying the frames")
    p.add_argument("--out", help="output folder (default: the first argument)")
    a = p.parse_args(argv)
    try:
        from src.config import MEASURED_ESTIMATION
        settle = MEASURED_ESTIMATION.vote_window - 1
        folders = clip_folders(a.runs)
        rows, changes = [], []
        for f in folders:
            labels = [lb for lb in read_labels(f / "labels.csv") if lb.get("condition") != UNCHECKED]
            res = score_clip(readings_for(f, a.from_log), labels, settle, run=f.name)
            if not res["rows"]:
                print(f"{f.name}: no frame labelled with a light, skipped", file=sys.stderr)
            rows += res["rows"]
            changes += res["changes"]
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else (Path(a.runs[0]) if a.runs else Path(common.RUNS_DIR))
    out = out.parent if (out / "labels.csv").is_file() and not a.out else out
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "light_clips.csv", CLIP_FIELDS, rows)
    _write(out / "light_changes.csv", CHANGE_FIELDS, changes)
    for r in rows:
        print(f"{r['run']} {r['gap_cm']} cm {r['condition']}: {r['light']:<6} {r['frames']:>4} frames, "
              f"frame {r['frame_right_pct']}%, voted {r['voted_right_pct']}%, {r['voted_wrong_frames']} voted wrong")
    print(f"wrote {out / 'light_clips.csv'}\nwrote {out / 'light_changes.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
