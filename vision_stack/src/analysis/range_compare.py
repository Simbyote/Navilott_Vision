"""Range compare: several detect-range runs, by condition, round and gap, beside the camera's metadata.

Purpose:
    One detect-range run says how a target read at each gap from one spot. The
    questions across a day of them are which backgrounds it fails against, how
    the same spot changed from round to round, how detection falls off with
    distance, and why. Each saved look of a present sign is measured the way
    the detector sees it: the sign is the sign-shaped blob that differs between
    the present look and the absent look at the same gap (the robot hasn't
    moved), and its redness (R - max(G, B)) is set against the threshold the
    detector would pick in that look (Otsu over the sign ROI, never under
    min_redness). A sign under its threshold isn't in the red mask at all,
    whatever its shape. The saved look is the last of a look's frames, as a
    JPEG and before undistortion, so the margin is an estimate of what the
    detector saw. frame_meta runs named for the same spots add lux,
    exposure and gain per condition.

Main package:
    analyze(runs, conditions, metas) -> dict; range_compare.png, .json and a
    tidy range_compare.csv (one row per look, for a spreadsheet) in --out.
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

from src.analysis import common
from src.analysis.common import Table

RUN_GLOB = "routine_detect-range_*"
META_GLOB = "meta_*"
ABSENT = ("absent", "off")
DIFF_MIN = 35               # a pixel changed between the present and absent looks (max over B, G, R)
MIN_BLOB_PX = 30            # smaller is a speck, not a sign
ASPECT = (0.5, 2.0)         # a sign is about as wide as tall; wires and tape are not
MIN_FILL = 0.45             # blob area / its box: an octagon fills ~0.83
MIN_SIGN_REDNESS = 5.0      # a changed blob redder than this can be the sign; a person or a shadow isn't
WINDOW_CM = (20.0, 45.0)    # where a posted sign is in view on the course; closer, it has left the frame
SHADES = ("#1E63B5", "#C62828", "#2A9A88", "#7B4F8C", "#E0A100", "#5F6B76", "#8A6D3B")
LOOSE_FIELDS = ("run", "time", "condition", "round", "sign", "gap_cm", "state", "frames", "seen_pct",
                "wrong_frames", "voted", "result", "target_found", "target_px", "target_redness",
                "threshold", "margin", "target_above_pct", "target_v")

_CLI_HELP = """\
Compare several detect-range runs taken at different spots or times: detection
by condition and round, detection by gap, false reads with the target away,
the sign's redness against the threshold the detector picks, and the camera's
lux / exposure / gain per condition from frame_meta runs.

Name the runs in a conditions CSV (columns: run, condition; optional round,
sign, meta), where run is the folder name or its time stamp:
    run,condition,round,sign,meta
    20261102_121725,Start,2,,meta_20261102_135402-STARTING-POSITION
Without one, a folder name ending in "(window)" names its condition. Rounds
default to each condition's 1st, 2nd, 3rd run in time order. A meta folder is
matched to a condition by name when the CSV doesn't give one.

Run from vision_stack/:
    python3 -m src.analysis.range_compare runs/sign_day --conditions runs/sign_day/conditions.csv --meta runs/meta_day
    make range-compare RUN=runs/sign_day ARGS="--conditions runs/sign_day/conditions.csv --meta runs/meta_day"
"""


def run_folders(args: list[str], skipped: list | None = None) -> list[Path]:
    """Runs with a trials.csv; an empty one (a run cut off, or a full disk) goes in skipped."""
    found = []
    for a in args or [str(common.RUNS_DIR)]:
        p = Path(a)
        if (p / "trials.csv").is_file():
            found.append(p)
        elif p.is_dir():
            found += sorted(q for q in p.rglob(RUN_GLOB) if (q / "trials.csv").is_file())
    out = [f for f in found if (f / "trials.csv").stat().st_size > 0]
    if skipped is not None:
        skipped += [f.name for f in found if f not in out]
    if not out:
        raise FileNotFoundError(f"no detect-range run (trials.csv) in {', '.join(args or [str(common.RUNS_DIR)])}")
    return sorted(out, key=lambda f: stamp(f))


def stamp(folder: Path) -> str:
    m = re.search(r"(\d{8}_\d{6})", folder.name)
    return m.group(1) if m else folder.name


def read_conditions(path) -> dict:
    """{stamp or folder name: row}; empty without a file."""
    if not path:
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "run" not in rows[0]:
        raise ValueError(f"{path}: needs a 'run' column")
    out = {}
    for r in rows:
        key = (r.get("run") or "").strip()
        if key:
            out[stamp(Path(key))] = {k: (v or "").strip() for k, v in r.items()}
    return out


def run_info(folder: Path, conditions: dict) -> dict:
    row = conditions.get(stamp(folder), {})
    res = folder / "results.json"
    meta = json.loads(res.read_text()) if res.is_file() and res.stat().st_size else {}
    start = (meta.get("conditions") or {}).get("start") or {}
    cond = row.get("condition")
    if not cond:
        m = re.search(r"\((.*)\)$", folder.name)
        cond = m.group(1) if m else (start.get("notes") or "unlabelled")
    return {"run": folder.name, "time": start.get("time") or stamp(folder), "condition": cond,
            "round": row.get("round") or "", "sign": row.get("sign") or "", "meta": row.get("meta") or "",
            "target": (meta.get("options") or {}).get("target") or "", "commit": start.get("commit") or ""}


def picture(folder: Path, trial: int, gap: float, state: str) -> Path | None:
    p = folder / f"attempt_{trial:02d}_{gap:g}cm_{state}.jpg"
    if p.is_file():
        return p
    found = sorted(folder.glob(f"attempt_*_{gap:g}cm_{state}.jpg"))
    return found[-1] if found else None


def min_redness() -> float:
    from src.config import MEASURED
    return float(MEASURED.geometry.sign.min_redness)


def sign_rect(shape) -> tuple[int, int, int, int]:
    from src.perception.roi_crop import SIGN
    h, w = shape[:2]
    return int(SIGN.x0 * w), int(SIGN.y0 * h), int(SIGN.x1 * w), int(SIGN.y1 * h)


def redness(bgr: np.ndarray) -> np.ndarray:
    """geometry_sign's: R - max(G, B), saturating at 0."""
    b, g, r = cv2.split(bgr)
    return cv2.subtract(r, cv2.max(g, b))


def target_look(path: Path | None, away: Path | None, floor: float) -> dict:
    """
    The sign in a present look: the most red sign-shaped blob among what
    changed against the absent look at the same gap. Its median redness, the
    threshold the detector would use on this ROI (Otsu, never under floor),
    the margin between them, and the share of the sign over the threshold.
    """
    none = {"target_found": False, "target_px": 0, "target_redness": None, "threshold": None, "margin": None,
            "target_above_pct": None, "target_v": None}
    frame = cv2.imread(str(path)) if path else None
    other = cv2.imread(str(away)) if away else None
    if frame is None or other is None or other.shape != frame.shape:
        return none
    x0, y0, x1, y1 = sign_rect(frame.shape)
    roi = frame[y0:y1, x0:x1]
    red = redness(roi)
    t, _ = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t), floor)
    diff = (cv2.absdiff(roi, other[y0:y1, x0:x1]).max(axis=2) >= DIFF_MIN).astype(np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(diff)
    best, best_red = None, -1.0
    for i in range(1, n):
        bw, bh, area = st[i, 2], st[i, 3], st[i, 4]
        if area < MIN_BLOB_PX or not ASPECT[0] <= bw / max(bh, 1) <= ASPECT[1] or area / float(bw * bh) < MIN_FILL:
            continue
        med = float(np.median(red[lab == i]))
        if med > best_red:
            best, best_red = i, med
    if best is None or best_red < MIN_SIGN_REDNESS:
        return none
    px = lab == best
    vals = red[px].astype(float)
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2][px]
    return {"target_found": True, "target_px": int(px.sum()), "target_redness": float(np.median(vals)),
            "threshold": round(t, 1), "margin": round(float(np.median(vals)) - t, 1),
            "target_above_pct": round(100.0 * float((vals >= t).mean()), 1), "target_v": float(np.median(v))}


def _true(v) -> bool:
    return str(v).strip().lower() == "true"


def looks(folder: Path, info: dict, floor: float) -> list[dict]:
    t = Table(folder / "trials.csv")
    trial, gap, frames = t.numeric("trial"), t.numeric("gap_cm"), t.numeric("frames")
    seen, wrong = t.numeric("seen_pct"), t.numeric("wrong_frames")
    state, voted, result = t.text("state"), t.text("voted"), t.text("result")
    out = []
    for i in range(len(t)):
        row = {k: info[k] for k in ("run", "time", "condition", "round", "sign")}
        row.update({"gap_cm": float(gap[i]), "state": state[i], "frames": int(frames[i]) if np.isfinite(frames[i]) else 0,
                    "seen_pct": float(seen[i]) if np.isfinite(seen[i]) else 0.0,
                    "wrong_frames": int(wrong[i]) if np.isfinite(wrong[i]) else 0,
                    "voted": _true(voted[i]), "result": result[i]})
        if state[i] not in ABSENT:
            away = next((picture(folder, int(trial[j]), float(gap[j]), state[j]) for j in range(len(t))
                         if state[j] in ABSENT and gap[j] == gap[i]), None)
            row.update(target_look(picture(folder, int(trial[i]), float(gap[i]), state[i]), away, floor))
        else:
            row.update({k: None for k in LOOSE_FIELDS[12:]})
        out.append(row)
    return out


def assign_rounds(infos: list[dict]) -> None:
    """Blank rounds become each condition's 1st, 2nd... run in time order."""
    count = {}
    for info in sorted(infos, key=lambda i: i["time"]):
        count[info["condition"]] = count.get(info["condition"], 0) + 1
        if not info["round"]:
            info["round"] = str(count[info["condition"]])


def meta_folders(args: list[str]) -> list[Path]:
    out = []
    for a in args or []:
        p = Path(a)
        if (p / "frames.csv").is_file():
            out.append(p)
        elif p.is_dir():
            out += sorted(q for q in p.rglob(META_GLOB) if (q / "frames.csv").is_file())
    return out


def meta_summary(folder: Path) -> dict:
    t = Table(folder / "frames.csv")
    out = {"folder": folder.name, "frames": len(t)}
    for k in ("lux", "exposure_us", "analogue_gain", "digital_gain", "colour_gain_r", "colour_gain_b", "colour_temp_k"):
        if t.has(k):
            v = t.numeric(k)
            v = v[np.isfinite(v)]
            out[k] = float(np.median(v)) if v.size else None
    if out.get("exposure_us") and out.get("analogue_gain"):
        out["exposure_x_gain"] = out["exposure_us"] * out["analogue_gain"] * (out.get("digital_gain") or 1.0)
    first = t.text("t_s")[0] if t.has("t_s") and len(t) else ""
    out["t_s"] = first
    return out


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def match_meta(condition: str, metas: list[Path], named: str = "") -> Path | None:
    if named:
        hit = [m for m in metas if m.name == named or stamp(m) == stamp(Path(named))]
        return hit[0] if hit else None
    words = _words(condition)
    hits = [m for m in metas if words and all(any(w in mw or mw.startswith(w) for mw in _words(m.name)) for w in words)]
    return hits[0] if len(hits) == 1 else None


def rate(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(1 for r in rows if r["voted"])
    return {"k": k, "n": n, "pct": round(100.0 * k / n, 1) if n else None,
            "mean_seen_pct": round(statistics.fmean(r["seen_pct"] for r in rows), 1) if n else None}


def in_window(r: dict, window) -> bool:
    return window[0] <= r["gap_cm"] <= window[1]


def analyze(folders: list[Path], conditions: dict | None = None, metas: list[Path] | None = None,
            window=WINDOW_CM, floor: float | None = None) -> dict:
    floor = min_redness() if floor is None else floor
    infos = [run_info(f, conditions or {}) for f in folders]
    assign_rounds(infos)
    rows = []
    for f, info in zip(folders, infos):
        rows += looks(f, info, floor)
    present = [r for r in rows if r["state"] not in ABSENT]
    absent = [r for r in rows if r["state"] in ABSENT]
    conds = list(dict.fromkeys(i["condition"] for i in sorted(infos, key=lambda i: i["time"])))
    gaps = sorted({r["gap_cm"] for r in rows})
    by_cond = {}
    for c in conds:
        pr = [r for r in present if r["condition"] == c]
        ab = [r for r in absent if r["condition"] == c]
        ms = [r["margin"] for r in pr if r["margin"] is not None and in_window(r, window)]
        by_cond[c] = {"window": rate([r for r in pr if in_window(r, window)]),
                      "all_gaps": rate(pr),
                      "rounds": {rd: rate([r for r in pr if r["round"] == rd and in_window(r, window)])
                                 for rd in sorted({r["round"] for r in pr}, key=_round_key)},
                      "by_gap": {g: rate([r for r in pr if r["gap_cm"] == g]) for g in gaps},
                      "absent_looks": len(ab), "absent_wrong_votes": sum(1 for r in ab if r["voted"]),
                      "absent_wrong_frames": sum(r["wrong_frames"] for r in ab),
                      "absent_frames": sum(r["frames"] for r in ab),
                      "margin_median": float(np.median(ms)) if ms else None,
                      "meta": None}
    metas = metas or []
    for c in conds:
        named = next((i["meta"] for i in infos if i["condition"] == c and i["meta"]), "")
        m = match_meta(c, metas, named)
        if m is not None:
            by_cond[c]["meta"] = meta_summary(m)
    res = {"runs": infos, "conditions": conds, "gaps": gaps, "window_cm": list(window), "min_redness": floor,
           "by_condition": by_cond,
           "by_gap": {g: rate([r for r in present if r["gap_cm"] == g]) for g in gaps},
           "looks": rows, "commits": sorted({i["commit"] for i in infos if i["commit"]})}
    res["findings"] = findings(res)
    return res


def _round_key(r: str):
    return (0, int(r)) if r.isdigit() else (1, r)


def findings(res: dict) -> list[str]:
    out = []
    bc, w = res["by_condition"], res["window_cm"]
    ranked = sorted(((c, d["window"]) for c, d in bc.items() if d["window"]["n"]), key=lambda kv: kv[1]["pct"])
    if ranked:
        lo, hi = ranked[0], ranked[-1]
        out.append(f"{w[0]:g}-{w[1]:g} cm, voted present: worst {lo[0]} {lo[1]['k']}/{lo[1]['n']}, "
                   f"best {hi[0]} {hi[1]['k']}/{hi[1]['n']}")
    near = [r for r in res["looks"] if r["state"] not in ABSENT and r["gap_cm"] < w[0]]
    if near:
        found = sum(1 for r in near if r["target_found"])
        out.append(f"closer than {w[0]:g} cm: a sign-shaped red blob in {found} of {len(near)} saved looks; "
                   "where none, the sign has left the frame (a range limit, not a detector miss)")
    under = [(c, d["margin_median"]) for c, d in bc.items() if d["margin_median"] is not None and d["margin_median"] < 0]
    if under:
        out.append("the sign's median redness under the detector's threshold (margin): "
                   + ", ".join(f"{c} {m:+.0f}" for c, m in under) + "; only its reddest pixels reach the red mask, "
                   "so its outline breaks up and detection turns hit-or-miss")
    looks_in = [r for r in res["looks"] if r["margin"] is not None and in_window(r, w)]
    if looks_in:
        above = [r for r in looks_in if r["margin"] >= 0]
        below = [r for r in looks_in if r["margin"] < 0]
        out.append(f"{w[0]:g}-{w[1]:g} cm looks voted present: {sum(r['voted'] for r in above)}/{len(above)} "
                   f"with the sign over its threshold, {sum(r['voted'] for r in below)}/{len(below)} under it")
    alarms = [(c, d["absent_wrong_votes"], d["absent_wrong_frames"]) for c, d in bc.items() if d["absent_wrong_frames"]]
    if alarms:
        out.append("target away, wrong reads: " + ", ".join(f"{c} {v} votes / {f} frames" for c, v, f in alarms))
    if len(res["commits"]) > 1:
        out.append(f"runs span {len(res['commits'])} commits: {', '.join(res['commits'])}; compare rounds with care")
    elif res["commits"]:
        out.append(f"every run on commit {res['commits'][0]}: changes between rounds are the scene, not the code")
    with_meta = [c for c, d in bc.items() if d["meta"]]
    if with_meta:
        out.append(f"frame_meta matched for {len(with_meta)} of {len(bc)} conditions; one sample each, "
                   "taken at its own time, not during every run")
    return out


def figure(res: dict, out_path, title: str):
    plt = common.pyplot()
    if plt is None:
        return None
    conds, gaps, bc, w = res["conditions"], res["gaps"], res["by_condition"], res["window_cm"]
    color = {c: SHADES[i % len(SHADES)] for i, c in enumerate(conds)}
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.22)

    # 1 detection by condition across rounds
    ax = fig.add_subplot(gs[0, 0])
    rounds = sorted({r for d in bc.values() for r in d["rounds"]}, key=_round_key)
    for c in conds:
        ys = [bc[c]["rounds"].get(r, {}).get("pct") for r in rounds]
        ax.plot(range(len(rounds)), [np.nan if y is None else y for y in ys], "-o", color=color[c], lw=1.6, ms=5,
                label=c)
        for x, r in enumerate(rounds):
            d = bc[c]["rounds"].get(r)
            if d and d["n"]:
                ax.annotate(f"{d['k']}/{d['n']}", (x, d["pct"]), textcoords="offset points", xytext=(6, -3),
                            fontsize=6, color=color[c])
    ax.set_xticks(range(len(rounds)), [f"round {r}" for r in rounds], fontsize=8)
    ax.set_ylim(-5, 105)
    ax.set_ylabel(f"present looks voted present, {w[0]:g}-{w[1]:g} cm (%)", fontsize=8)
    ax.set_title("Detection by condition, round to round", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # 2 detection by distance
    ax = fig.add_subplot(gs[0, 1])
    for c in conds:
        ys = [bc[c]["by_gap"][g]["mean_seen_pct"] for g in gaps]
        ax.plot(gaps, [np.nan if y is None else y for y in ys], "-o", color=color[c], lw=1, ms=3, alpha=0.75, label=c)
    allg = [res["by_gap"][g]["mean_seen_pct"] for g in gaps]
    ax.plot(gaps, [np.nan if y is None else y for y in allg], "-s", color="#1F2933", lw=2.4, ms=5, label="all runs")
    for g in gaps:
        d = res["by_gap"][g]
        if d["n"]:
            ax.text(g, -9, f"n={d['n']}", ha="center", fontsize=6, color="#5F6B76")
    ax.axvspan(w[0], w[1], color="#E8EDF1", zorder=0)
    ax.set_ylim(-14, 105)
    ax.set_xticks(gaps)
    ax.set_xlabel("gap before the stop line (cm)", fontsize=8)
    ax.set_ylabel("mean frames seeing the target (%)", fontsize=8)
    ax.set_title("Average detection by distance (shaded: the judged window)", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    # 3 grid of every run x gap
    ax = fig.add_subplot(gs[1, :])
    runs = sorted(res["runs"], key=lambda i: i["time"])
    for i, info in enumerate(runs):
        for j, g in enumerate(gaps):
            pr = [r for r in res["looks"] if r["run"] == info["run"] and r["gap_cm"] == g and r["state"] not in ABSENT]
            ab = [r for r in res["looks"] if r["run"] == info["run"] and r["gap_cm"] == g and r["state"] in ABSENT]
            if not pr:
                continue
            r = pr[0]
            shade = plt.cm.Blues(0.15 + 0.75 * r["seen_pct"] / 100.0)
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.86, color=shade))
            txt = f"{r['seen_pct']:.0f}%" + (" v" if r["voted"] else "")
            if r["target_found"] is False and g < w[0]:
                txt += "\nno sign in view"
            ax.text(j + 0.47, i + 0.43, txt, ha="center", va="center", fontsize=6,
                    color="white" if r["seen_pct"] > 55 else "#1F2933")
            if ab and ab[0]["wrong_frames"]:
                ax.add_patch(plt.Rectangle((j, i), 0.94, 0.86, fill=False, ec="#C62828", lw=1.6 if ab[0]["voted"] else 0.8))
    ax.set_xlim(0, len(gaps))
    ax.set_ylim(len(runs), 0)
    ax.set_xticks([j + 0.47 for j in range(len(gaps))], [f"{g:g} cm" for g in gaps], fontsize=8)
    ax.set_yticks([i + 0.43 for i in range(len(runs))],
                  [f"{i['time'][11:16]}  {i['condition']} r{i['round']}" + (f" ({i['sign']})" if i["sign"] else "")
                   for i in runs], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Every run: % of frames seeing the target (v = voted present); red edge = wrong frames with it "
                 "away (thick: a wrong vote)", loc="left", fontsize=9, pad=20)

    # 4 the target's own red brightness against detection
    ax = fig.add_subplot(gs[2, 0])
    for c in conds:
        pts = [(r["margin"], r["seen_pct"]) for r in res["looks"]
               if r["condition"] == c and r["margin"] is not None and in_window(r, w)]
        if pts:
            ax.scatter(*zip(*pts), s=22, color=color[c], alpha=0.8, label=c)
    ax.axvline(0, color="#B26A00", ls="--", lw=1)
    ax.text(1, 50, "detector's threshold", color="#B26A00", fontsize=7, rotation=90)
    ax.set_xlabel("sign redness minus the threshold the detector picked (R - max(G,B), median)", fontsize=8)
    ax.set_ylabel("frames seeing it (%)", fontsize=8)
    ax.set_ylim(-5, 105)
    ax.set_title(f"Why: is the sign red enough to enter the mask? ({w[0]:g}-{w[1]:g} cm)", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # 5 camera metadata per condition beside detection
    ax = fig.add_subplot(gs[2, 1])
    with_meta = [c for c in conds if bc[c]["meta"]]
    if with_meta:
        x = np.arange(len(with_meta))
        lux = [bc[c]["meta"].get("lux") or 0 for c in with_meta]
        gain = [bc[c]["meta"].get("analogue_gain") or 0 for c in with_meta]
        ax.bar(x - 0.2, lux, 0.38, color="#9FB3C8", label="lux (camera's estimate)")
        ax.set_ylabel("lux", fontsize=8)
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, gain, 0.38, color="#5F6B76", label="analogue gain")
        ax2.set_ylabel("analogue gain", fontsize=8)
        for i, c in enumerate(with_meta):
            d, v = bc[c]["window"], bc[c]["margin_median"]
            ax.text(i, max(lux) * 1.04 if lux else 0, f"{d['k']}/{d['n']} voted\nmargin {v:+.0f}" if v is not None
                    else f"{d['k']}/{d['n']} voted", ha="center", fontsize=7, color=color[c])
        ax.set_xticks(x, with_meta, fontsize=8)
        ax.set_ylim(0, max(lux) * 1.3 if lux and max(lux) else 1)
        ax2.set_ylim(0, max(gain) * 1.6 if gain and max(gain) else 1)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
        ax2.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "no frame_meta runs matched (--meta)", transform=ax.transAxes, ha="center", color="#5F6B76")
    ax.set_title("Camera metadata per condition (one frame_meta sample each)", loc="left", fontsize=10)
    for a in fig.axes:
        a.tick_params(labelsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_rows(path, rows: list[dict]) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LOOSE_FIELDS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in LOOSE_FIELDS})
    return path


def report_lines(res: dict) -> list[str]:
    w = res["window_cm"]
    lines = [f"{len(res['runs'])} runs, {len(res['conditions'])} conditions, gaps {', '.join(f'{g:g}' for g in res['gaps'])} cm",
             f"  {'condition':<22} {'voted ' + f'{w[0]:g}-{w[1]:g}':>12} {'all gaps':>9}  rounds{'':<14} "
             f"{'away: votes/frames':>19} {'margin':>7} {'lux':>6} {'gain':>5}"]
    for c in res["conditions"]:
        d = res["by_condition"][c]
        rounds = " ".join(f"r{r}:{x['k']}/{x['n']}" for r, x in d["rounds"].items())
        m = d["meta"] or {}
        lines.append(f"  {c:<22} {d['window']['k']:>5}/{d['window']['n']:<6} {d['all_gaps']['k']:>4}/{d['all_gaps']['n']:<4}  "
                     f"{rounds:<20} {d['absent_wrong_votes']:>8}/{d['absent_wrong_frames']:<10} "
                     f"{common.fmt(d['margin_median'], 0):>7} {common.fmt(m.get('lux'), 0):>6} "
                     f"{common.fmt(m.get('analogue_gain'), 2):>5}")
    lines.append("  by gap (all runs, mean % of frames seeing it): "
                 + "  ".join(f"{g:g}:{res['by_gap'][g]['mean_seen_pct']:.0f}" for g in res["gaps"]
                             if res["by_gap"][g]["n"]))
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.range_compare", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="detect-range run folders, or folders holding them (default: runs/)")
    p.add_argument("--conditions", help="CSV naming each run's condition (run, condition[, round, sign, meta])")
    p.add_argument("--meta", action="append", default=[], help="frame_meta run folders, or a folder holding them")
    p.add_argument("--window", default=f"{WINDOW_CM[0]:g}-{WINDOW_CM[1]:g}",
                   help="the gaps judged for the rate, cm (default %(default)s: where the sign is in view)")
    p.add_argument("--out", help="output folder (default: the first argument's folder)")
    args = p.parse_args(argv)
    try:
        lo, hi = (float(x) for x in args.window.split("-"))
        skipped = []
        folders = run_folders(args.runs, skipped)
        res = analyze(folders, read_conditions(args.conditions), meta_folders(args.meta), window=(lo, hi))
        if skipped:
            res["skipped"] = skipped
            res["findings"].append(f"skipped, empty trials.csv (cut off or a full disk): {', '.join(skipped)}")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    first = Path(args.runs[0]) if args.runs else Path(common.RUNS_DIR)
    out = Path(args.out) if args.out else (first if not (first / "trials.csv").is_file() else first.parent)
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "range_compare.json", {k: v for k, v in res.items() if k != "looks"})
    write_rows(out / "range_compare.csv", res["looks"])
    target = next((i["target"] for i in res["runs"] if i["target"]), "target")
    fig = figure(res, out / "range_compare.png", f"Detect range ({target}): {len(res['runs'])} runs across "
                 f"{len(res['conditions'])} conditions")
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'range_compare.json'}\nwrote {out / 'range_compare.csv'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
