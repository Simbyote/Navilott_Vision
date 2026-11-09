"""Light report: what a navigation run's traffic light read frame by frame, what the vote made of it, and when it braked.

Purpose:
    One frame's reading is noisy (a missed light reads as go, a ring reads the
    wrong color); the 3-frame vote is what navigation acts on, and the light
    rule brakes only at a stop line. Three strips on one time axis show where
    the three part ways, and a close-up of the longest hold shows how it began
    and ended.

Main package:
    analyze(nav, video) -> dict; light_report.png and light_report.json beside
    nav.csv. nav_video.csv (written when the run's video is rendered) adds each
    frame's own reading; without it only the vote and the braking show.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table, runs_of

CSV_NAME, VIDEO_CSV = "nav.csv", "nav_video.csv"
RULE = "traffic_light"
STATE_COLORS = {"stop": "#C62828", "caution": "#E0A100", "go": "#2E7D32"}
NONE_COLOR, HOLD_COLOR = "#C9D1D8", "#1F2933"
LABEL_STATE = {"red": "stop", "yellow": "caution", "green": "go"}
# A red reading with the vote at go for this many frames each side is a stray the vote filtered
STRAY_WINDOW = 12
ZOOM_PAD_S = 3.0

_CLI_HELP = """\
The traffic light over a navigation run: each frame's reading, the 3-frame
vote and when the light rule braked, on one time axis, with a close-up of the
longest hold. Writes light_report.png and light_report.json next to nav.csv.
Render the run first (make render) for each frame's own reading.

Run from vision_stack/:
    python3 -m src.analysis.light_report                      newest navigation run
    python3 -m src.analysis.light_report runs/nav_<time>
    make light-report RUN=runs/nav_<time>
"""


def frame_reads(nav: Table, video: Table | None) -> tuple[list, list]:
    """(raw state per nav frame or None without a reading, detected per frame), matched by frame_id."""
    n = len(nav)
    if video is None or not video.has("traffic_raw"):
        return [None] * n, [None] * n
    by_id = {}
    ids, raw = video.numeric("frame_id"), video.text("traffic_raw")
    dets = video.text("traffic_detections") if video.has("traffic_detections") else [""] * len(video)
    for i, r, d in zip(ids, raw, dets):
        if np.isfinite(i):
            passed = [x.split(":")[0] for x in d.split(";") if x and not x.endswith(":gated")]
            by_id[int(i)] = (r or None, bool(passed))
    fid = nav.numeric("frame_id")
    got = [by_id.get(int(f)) if np.isfinite(f) else None for f in fid]
    return [g[0] if g else None for g in got], [g[1] if g else None for g in got]


def holds(t: np.ndarray, braking: np.ndarray) -> list[dict]:
    """Each run of light-rule frames: start and end time, frames."""
    out, i = [], 0
    for value, start, length in runs_of(list(braking)):
        if value:
            out.append({"t0": float(t[start]), "t1": float(t[start + length - 1]), "frames": int(length),
                        "start": int(start)})
        i += length
    return out


def analyze(nav: Table, video: Table | None = None) -> dict:
    for c in ("t", "drive_state", "rule"):
        if not nav.has(c):
            raise ValueError(f"{nav.path} isn't a nav.csv (no {c})")
    t = nav.numeric("t")
    vote = nav.text("drive_state")
    braking = np.array([r == RULE for r in nav.text("rule")])
    raw, seen = frame_reads(nav, video)
    has_raw = any(r is not None for r in raw)
    hs = holds(t, braking)
    res = {"frames": len(nav), "duration_s": float(np.nanmax(t) - np.nanmin(t)) if len(nav) else 0.0,
           "has_raw": has_raw, "vote_share": {s: vote.count(s) / max(len(vote), 1) for s in STATE_COLORS},
           "holds": hs, "brake_frames": int(braking.sum())}
    if has_raw:
        known = [s for s in seen if s is not None]
        res["no_light_share"] = (sum(not s for s in known) / len(known)) if known else None
        strays = []
        for i, r in enumerate(raw):
            if r == "stop" and not braking[i]:
                around = vote[max(0, i - STRAY_WINDOW):i + 1 + STRAY_WINDOW]
                if all(v == "go" for v in around):
                    strays.append(i)
        res["stray_reds"] = len(strays)
        res["stray_red_times"] = [float(t[i]) for i in strays]
        res["raw_flips"] = int(sum(1 for a, b in zip(raw, raw[1:]) if a and b and a != b))
        res["vote_flips"] = int(sum(1 for a, b in zip(vote, vote[1:]) if a != b))
    in_hold_go = sum(1 for i, b in enumerate(braking) if b and vote[i] == "go")
    res["go_votes_in_holds"] = int(in_hold_go)
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    if res.get("no_light_share") is not None and res["no_light_share"] > 0.1:
        out.append(f"no light passed the gates on {100 * res['no_light_share']:.0f}% of frames: those read as go")
    if res.get("stray_reds"):
        out.append(f"{res['stray_reds']} red readings inside green stretches (outside any hold): the vote kept "
                   "them from braking; if they cluster, look at what's red in the traffic ROI")
    if res["go_votes_in_holds"]:
        out.append(f"the vote said go on {res['go_votes_in_holds']} frames while the light rule held the robot: "
                   "the release time kept a dropped frame from letting it go")
    if res.get("vote_flips") and res.get("raw_flips") and res["raw_flips"] > 3 * res["vote_flips"]:
        out.append(f"the reading changed {res['raw_flips']} times, the vote {res['vote_flips']}: the vote is doing its job")
    if not res["holds"]:
        out.append("the light rule never braked in this run")
    return out


def _strips(ax, t, raw, seen, vote, braking, has_raw):
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.05
    width = np.append(np.diff(t), dt)        # each frame lasts until the next, so the strip has no gaps
    width = np.where(np.isfinite(width) & (width > 0), width, dt)
    rows = ([("one frame", raw, seen)] if has_raw else []) + [("3-frame vote", vote, None)]
    for k, (label, states, det) in enumerate(rows):
        y = len(rows) - k
        for st, c in STATE_COLORS.items():
            m = np.array([s == st and (d is None or d) for s, d in zip(states, det or [None] * len(states))])
            ax.bar(t[m], 0.7, width=width[m], bottom=y - 0.35, color=c, lw=0, align="edge")
        if det is not None:
            m = np.array([d is False for d in det])
            ax.bar(t[m], 0.7, width=width[m], bottom=y - 0.35, color=NONE_COLOR, lw=0, align="edge")
    ax.bar(t[braking], 0.5, width=width[braking], bottom=-0.25, color=HOLD_COLOR, lw=0, align="edge")
    ax.set_yticks(list(range(len(rows), 0, -1)) + [0], [r[0] for r in rows] + ["light braking"], fontsize=8)
    ax.set_ylim(-0.5, len(rows) + 0.75)
    return len(rows)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def figure(nav: Table, video: Table | None, res: dict, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    t = nav.numeric("t")
    vote = nav.text("drive_state")
    braking = np.array([r == RULE for r in nav.text("rule")])
    raw, seen = frame_reads(nav, video)
    longest = max(res["holds"], key=lambda h: h["frames"]) if res["holds"] else None
    fig, axes = plt.subplots(2 if longest else 1, 1, figsize=(11, 4.6 if longest else 2.4), squeeze=False)
    axes = axes[:, 0]
    top = _strips(axes[0], t, raw, seen, vote, braking, res["has_raw"])
    for ts in res.get("stray_red_times", []):
        axes[0].plot(ts, top + 0.5, "v", color="#B26A00", ms=4)
    axes[0].set_xlim(np.nanmin(t), np.nanmax(t))
    axes[0].set_title(title, loc="left", fontsize=10)
    note = "grey: no light found, which reads as go; ▼ a red reading inside a green stretch" if res["has_raw"] else \
        "no nav_video.csv: render the run (make render) for each frame's own reading"
    axes[0].set_xlabel(f"time (s); {note}", fontsize=8)
    if longest:
        _strips(axes[1], t, raw, seen, vote, braking, res["has_raw"])
        axes[1].set_xlim(longest["t0"] - ZOOM_PAD_S, longest["t1"] + ZOOM_PAD_S)
        axes[1].set_xlabel(f"time (s): the longest hold, {longest['t1'] - longest['t0']:.1f} s from {longest['t0']:.1f} s",
                           fontsize=8)
    for a in axes:
        a.tick_params(labelsize=7)
        a.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    vs = res["vote_share"]
    lines = [f"{res['frames']} frames over {res['duration_s']:.1f} s; the vote: go {100 * vs['go']:.0f}%, "
             f"caution {100 * vs['caution']:.0f}%, stop {100 * vs['stop']:.0f}%",
             f"light holds: {len(res['holds'])} ({res['brake_frames']} frames)"
             + "".join(f"\n  {h['t0']:.1f}-{h['t1']:.1f} s, {h['frames']} frames" for h in res["holds"])]
    if res["has_raw"]:
        lines.append(f"each frame's reading: no light on {100 * (res['no_light_share'] or 0):.0f}%, "
                     f"{res['stray_reds']} stray reds, {res['raw_flips']} changes (the vote {res['vote_flips']})")
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.light_report", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="navigation run folder or nav.csv (default: the newest)")
    p.add_argument("--out", help="output folder (default: next to nav.csv)")
    args = p.parse_args(argv)
    try:
        path = common.find_csv(args.run, CSV_NAME)
        nav = Table(path)
        video = Table(path.parent / VIDEO_CSV) if (path.parent / VIDEO_CSV).is_file() else None
        res = analyze(nav, video)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "light_report.json", res)
    fig = figure(nav, video, res, f"Traffic light: {path.parent.name}", out / "light_report.png")
    print(path)
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'light_report.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
