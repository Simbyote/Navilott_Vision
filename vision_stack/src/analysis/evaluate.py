"""Labelled replay: each detector scored against what was really in front of the camera.

Purpose:
    The routines judge one condition, live, and are gone; this scores recorded
    frames against labels, so a tuning change can be measured on the same
    frames before and after. Every frame goes through the robot's own
    Pipeline (Phases 2 and 3) under MEASURED; each detector is scored per
    frame (Phase 2's detection through Phase 3's confidence gate) and, for a
    recorded sequence, as voted (the packet navigation acts on). The vote
    needs 2 of its 3 frames to change, so the frames just after the truth
    changes aren't held against it: that's its latency, not a wrong read.

Labels (labels.csv beside the frames, or --labels): ranges of frame ids,
    first and last inclusive; a blank cell isn't scored, "none" means not there:
        first,last,sign,light,line_cm,obstacle_cm,offset_cm,range_cm,condition
        0,140,no,off,none,none,0,,room light
        141,220,yes,red,18,none,,60,glare
    sign: yes / no. light: red / yellow / green / off (nothing lit in view).
    line_cm, obstacle_cm: distance from the robot's front edge, the stop-line
    table's reference; yes (there, distance not measured: presence only); or none. offset_cm: + = robot right of the lane's
    center. range_cm: how far the sign or light is, for range curves.
    condition: any tag (glare, dim, angled) to break the scores down by;
    "unchecked" (a draft of what the robot read) is never scored.
    Lines starting with # are comments. src/scripts/label_frames.py writes it.

    A routine folder (detect-range, obstacle-check, lane-offset) labels
    itself: each look's saved frame carries its gap and state in its name.
    Single frames score per frame only: a vote needs a sequence.

Main package:
    evaluate(frames, labels) -> per detector: counts, precision and recall with
    95% intervals, light confusion matrices, distance errors, broken down by
    range and condition; findings. eval.json and eval.png beside the run.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.params import STOP_SIGN, TRAFFIC_LIGHT

LABEL_FIELDS = ("first", "last", "sign", "light", "line_cm", "obstacle_cm", "offset_cm", "range_cm", "condition")
LIGHTS = ("red", "yellow", "green", "off")
NONE = "none"
# label_frames --draft tags what the robot read with this; it's never scored until a person accepts it
UNCHECKED = "unchecked"
# A stop line or obstacle that's there, its distance not measured: scored for presence, not error
PRESENT = "yes"
FRAME_MS = 50
# A detector's distance is within this of the tape: lane offset P3's 2 cm, the routines' 3 cm for the rest
TOLERANCE_CM = {"line": 3.0, "obstacle": 3.0, "offset": 2.0}
RANGE_BINS_CM = (0, 15, 30, 45, 60, 90, 1e9)
DIST_BINS_CM = (0, 5, 10, 15, 20, 30, 1e9)
# Below these, a share says little: findings call it out rather than judge it
MIN_POSITIVES = 20
VOTED_FOR = {"red": "stop", "yellow": "caution", "green": "go", "off": "go"}
_ROUTINE_FILE = re.compile(r"attempt_\d+_(?P<rest>.+)\.(?:jpg|png)$")

_CLI_HELP = """\
Score each detector against labels on recorded frames: how often it finds what's
there (recall), how often what it finds is real (precision), how far off its
distances are, and where it fails (by range and condition).

    make eval RUN=runs/nav_<time>                      labels.csv in the run folder
    make eval RUN=runs/nav_<time> BASE=runs/nav_<time>/eval.json   against an earlier score
    make eval RUN=runs/routine_detect-range_<time>     a routine labels itself
    make label RUN=runs/nav_<time>                     write the labels, frame by frame

Record frames with any linker (make phase2 / phase3 / navigate save frames/), label
them once, and rescore after every tuning change: the same frames, before and after.
"""


# ---- labels ----

def _num_or_none(v: str, field: str, where: str):
    v = v.strip().lower()
    if v == "":
        return None
    if v == NONE:
        return NONE
    if v == PRESENT and field in ("line_cm", "obstacle_cm"):
        return PRESENT
    try:
        return float(v)
    except ValueError:
        what = "a number, 'yes' or 'none'" if field in ("line_cm", "obstacle_cm") else "a number or 'none'"
        raise ValueError(f"{where}: {field} {v!r} is not {what}") from None


def parse_label(row: dict, where: str) -> dict:
    """One labels.csv row, checked; blank fields are left out (not scored)."""
    out = {}
    try:
        out["first"], out["last"] = int(row["first"]), int(row["last"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: first and last must be frame ids") from None
    if out["last"] < out["first"]:
        raise ValueError(f"{where}: last {out['last']} is before first {out['first']}")
    sign = (row.get("sign") or "").strip().lower()
    if sign:
        if sign not in ("yes", "no"):
            raise ValueError(f"{where}: sign {sign!r} is not yes or no")
        out["sign"] = sign == "yes"
    light = (row.get("light") or "").strip().lower()
    if light:
        if light not in LIGHTS:
            raise ValueError(f"{where}: light {light!r} is not one of {', '.join(LIGHTS)}")
        out["light"] = light
    for f in ("line_cm", "obstacle_cm", "offset_cm", "range_cm"):
        v = _num_or_none(row.get(f) or "", f, where)
        if v is not None:
            if f in ("offset_cm", "range_cm") and v == NONE:
                raise ValueError(f"{where}: {f} must be a number")
            out[f] = v
    cond = (row.get("condition") or "").strip()
    if cond:
        out["condition"] = cond
    return out


def read_labels(path) -> list[dict]:
    """labels.csv's rows, checked; ValueError naming the line for a bad value or overlapping ranges."""
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: no labels")
    reader = csv.DictReader(lines)
    missing = [f for f in ("first", "last") if f not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{path}: no {' or '.join(missing)} column")
    labels = [parse_label(r, f"{path.name} row {i + 1}") for i, r in enumerate(reader)]
    spans = sorted((lb["first"], lb["last"]) for lb in labels)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if b0 <= a1:
            raise ValueError(f"{path.name}: frames {b0}-{min(a1, b1)} are labelled twice")
    return labels


def truth_for(labels: list[dict], fid: int) -> dict | None:
    for lb in labels:
        if lb["first"] <= fid <= lb["last"]:
            return lb
    return None


# ---- frames ----

def frames_in(folder) -> list[tuple[int, Path]]:
    """(frame id, path) in order: the recorder's NNNNNN.jpg names give the id (other images beside them
    are skipped), else every image by position."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not files:
        raise FileNotFoundError(f"no frames in {folder}")
    numbered = [p for p in files if p.stem.isdigit()]
    if numbered:                        # the recorder's frames; anything else beside them is a report
        return [(int(p.stem), p) for p in numbered]
    return list(enumerate(files))


def routine_kind(folder) -> str | None:
    m = re.match(r"routine_(detect-range|obstacle-check|lane-offset)_", Path(folder).name)
    return m.group(1) if m else None


def labels_from_routine(folder) -> list[tuple[Path, dict]]:
    """A routine's saved looks with the truth their names carry; [] if the folder isn't one that labels itself."""
    folder = Path(folder)
    kind = routine_kind(folder)
    if kind is None:
        return []
    options = {}
    if (folder / "results.json").is_file():
        options = json.loads((folder / "results.json").read_text()).get("options", {})
    out = []
    for p in sorted(folder.glob("attempt_*.jpg")):
        m = _ROUTINE_FILE.match(p.name)
        if not m:
            continue
        parts = m.group("rest").split("_")
        cm = [float(x[:-2]) for x in parts if re.fullmatch(r"[-+]?[\d.]+cm", x)]
        state = parts[-1]
        truth = {"condition": options.get("object", "") or ""}
        if kind == "detect-range":
            target = options.get("target", "light")
            if not cm:
                continue
            gap = cm[0]
            if target == "light" and state in LIGHTS:
                truth.update(light=state, range_cm=gap)
            elif target == "sign" and state in ("present", "absent"):
                truth.update(sign=state == "present", range_cm=gap)
            elif target == "line" and state == "present":
                truth.update(line_cm=gap)
            else:
                continue
        elif kind == "obstacle-check":
            truth["obstacle_cm"] = NONE if state == "clear" else cm[0]
        else:
            if not cm:
                continue
            truth["offset_cm"] = cm[0]
        out.append((p, truth))
    return out


# ---- replay ----

def replay(items, config=None, estimation=None, sequence: bool = True) -> list[dict]:
    """
    items: (frame id, path or image). Each frame through Pipeline.perceive, and for a sequence
    estimate too, in order. Per frame: Phase 2's reading through Phase 3's gates, and the packet.
    """
    import cv2
    from src.config import MEASURED, MEASURED_ESTIMATION
    from src.pipeline import Pipeline
    config = config or MEASURED
    estimation = estimation or MEASURED_ESTIMATION
    pipe = None
    out = []
    for fid, src in items:
        frame = cv2.imread(str(src)) if isinstance(src, (str, Path)) else src
        if frame is None:
            continue
        if pipe is None or not sequence:
            pipe = Pipeline(config, estimation, frame_size=frame.shape[:2], parallel=False)
        ts = fid * FRAME_MS
        p2 = pipe.perceive(frame, fid, ts)
        rec = {"fid": fid, **frame_reading(p2, estimation)}
        if sequence:
            pk = pipe.estimate(p2)
            rec.update(vote_sign=pk.stop_sign_detected, vote_drive=pk.drive_state,
                       vote_line=pk.stop_line_detected, vote_line_cm=pk.stop_line_distance_cm,
                       vote_obstacle=pk.obstacle_detected, vote_obstacle_cm=pk.obstacle_distance_cm,
                       vote_offset_cm=pk.lane_offset_cm, lane_status=pk.lane_status)
        out.append(rec)
    return out


def frame_reading(p2, estimation) -> dict:
    """One frame's reading as Phase 3 takes it in, before any vote."""
    sign = any(d.type == STOP_SIGN and d.confidence >= estimation.min_confidence_sign for d in p2.detections)
    light = None
    for d in p2.detections:
        if d.type == TRAFFIC_LIGHT and d.confidence >= estimation.min_confidence_traffic:
            light = d.label_detail
    sl = p2.stop_line_results[0] if p2.stop_line_results else None
    ob = p2.obstacle_results[0] if p2.obstacle_results else None
    lo = p2.lane_offset_results[0] if p2.lane_offset_results else None
    return {"sign": sign, "light": light,
            "line": bool(sl and sl.detected), "line_cm": sl.distance_cm if sl and sl.detected else None,
            "obstacle": bool(ob and ob.detected), "obstacle_cm": ob.distance_cm if ob and ob.detected else None,
            "offset": None if lo is None or lo.mode == "none" else lo.offset, "lane_mode": lo.mode if lo else "none"}


# ---- scoring ----

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval for a share k of n; it stays honest at 0, n and small n."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def share(k: int, n: int) -> dict:
    return {"k": k, "n": n, "share": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def _bin(v: float, edges) -> str:
    for a, b in zip(edges, edges[1:]):
        if a <= v < b:
            return f"{a:g}+" if b >= 1e9 else f"{a:g}-{b:g}"
    return "?"


def score_presence(pairs: list[tuple[bool, bool, dict]]) -> dict:
    """(truth, said, label) per frame: counts, precision, recall, false alarms; recall by range and condition."""
    tp = sum(t and s for t, s, _ in pairs)
    fn = sum(t and not s for t, s, _ in pairs)
    fp = sum(s and not t for t, s, _ in pairs)
    tn = sum(not t and not s for t, s, _ in pairs)
    by_range, by_cond = {}, {}
    for t, s, lb in pairs:
        if t and "range_cm" in lb:
            by_range.setdefault(_bin(lb["range_cm"], RANGE_BINS_CM), [0, 0])
            by_range[_bin(lb["range_cm"], RANGE_BINS_CM)][0] += s
            by_range[_bin(lb["range_cm"], RANGE_BINS_CM)][1] += 1
        c = lb.get("condition")
        if c:
            d = by_cond.setdefault(c, [0, 0, 0, 0])      # tp, positives, fp, negatives
            if t:
                d[0] += s
                d[1] += 1
            else:
                d[2] += s
                d[3] += 1
    return {"frames": len(pairs), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": share(tp, tp + fn), "precision": share(tp, tp + fp), "false_alarm": share(fp, fp + tn),
            "by_range": {k: share(*v) for k, v in sorted(by_range.items(), key=lambda kv: float(kv[0].split("-")[0].rstrip("+")))},
            "by_condition": {k: {"recall": share(v[0], v[1]), "false_alarm": share(v[2], v[3])}
                             for k, v in sorted(by_cond.items())}}


def score_light(pairs: list[tuple[str, str | None, dict]]) -> dict:
    """(truth, said, label): the 4 x 4 confusion matrix (said 'none' for no gated light) and accuracy."""
    said_keys = ("red", "yellow", "green", NONE)
    m = {t: {s: 0 for s in said_keys} for t in LIGHTS}
    right = 0
    for t, s, _ in pairs:
        s_ = s if s in said_keys else NONE
        m[t][s_] += 1
        right += (s_ == t) or (t == "off" and s_ == NONE)
    # a red, yellow or green read where something else is lit or nothing is: the hazard
    wrong_colour = sum(m[t][s] for t in LIGHTS for s in ("red", "yellow", "green") if s != t)
    missed = sum(m[t][NONE] for t in ("red", "yellow", "green"))
    by_range = {}
    for t, s, lb in pairs:
        if t != "off" and "range_cm" in lb:
            d = by_range.setdefault(_bin(lb["range_cm"], RANGE_BINS_CM), [0, 0])
            d[0] += s == t
            d[1] += 1
    return {"frames": len(pairs), "matrix": m, "accuracy": share(right, len(pairs)),
            "wrong_colour": share(wrong_colour, len(pairs)), "missed_lit": share(missed, sum(
                sum(m[t].values()) for t in ("red", "yellow", "green"))),
            "by_range": {k: share(*v) for k, v in sorted(by_range.items(), key=lambda kv: float(kv[0].split("-")[0].rstrip("+")))}}


def score_drive(pairs: list[tuple[str, str]]) -> dict:
    """(expected drive state, voted): what navigation was told about the light."""
    keys = ("stop", "caution", "go")
    m = {t: {s: 0 for s in keys} for t in keys}
    for t, s in pairs:
        m[t][s if s in keys else "go"] += 1
    right = sum(m[k][k] for k in keys)
    ran = sum(m[t]["go"] for t in ("stop", "caution"))
    return {"frames": len(pairs), "matrix": m, "accuracy": share(right, len(pairs)),
            "would_run_it": share(ran, sum(sum(m[t].values()) for t in ("stop", "caution")))}


def score_distance(pairs: list[tuple[object, bool, float | None, dict]], tol: float) -> dict:
    """(truth cm or 'none', detected, reported cm, label): detection where it's there, false alarms where not, error."""
    pres = [(t is not NONE, s, lb) for t, s, _, lb in pairs]
    out = score_presence(pres)
    err, by = [], {}
    for t, s, cm, lb in pairs:
        if t is NONE or t is PRESENT or not s or cm is None:
            continue
        e = cm - float(t)
        err.append(e)
        by.setdefault(_bin(float(t), DIST_BINS_CM), []).append(e)
    out["error_cm"] = _err(err, tol)
    out["error_by_distance"] = {k: _err(v, tol) for k, v in sorted(by.items(), key=lambda kv: float(kv[0].split("-")[0].rstrip("+")))}
    out["points"] = [(float(t), cm) for t, s, cm, _ in pairs if t not in (NONE, PRESENT) and s and cm is not None]
    out["no_cm"] = sum(1 for t, s, cm, _ in pairs if t not in (NONE, PRESENT) and s and cm is None)
    return out


def _err(e: list[float], tol: float) -> dict:
    if not e:
        return {"n": 0}
    a = np.array(e)
    return {"n": len(a), "mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
            "mae": round(float(np.abs(a).mean()), 2), "p95_abs": round(float(np.percentile(np.abs(a), 95)), 2),
            "within_tol": share(int((np.abs(a) <= tol).sum()), len(a)), "tol_cm": tol}


def score_offset(pairs: list[tuple[float, float | None, float | None]]) -> dict:
    """(truth cm, frame offset normalized, voted cm). In cm when Phase 3 has a scale, else a fit of the normalized offset."""
    seen = [(t, o, c) for t, o, c in pairs if o is not None]
    out = {"frames": len(pairs), "seen": share(len(seen), len(pairs))}
    cm = [(t, c) for t, _, c in seen if c is not None]
    if cm:
        out["error_cm"] = _err([c - t for t, c in cm], TOLERANCE_CM["offset"])
        out["points"] = cm
        out["units"] = "cm"
    if len({t for t, _, _ in seen}) >= 2:
        t = np.array([s[0] for s in seen])
        o = np.array([s[1] for s in seen])
        a, b = np.polyfit(o, t, 1)
        resid = t - (a * o + b)
        out["fit"] = {"cm_per_unit": round(float(a), 2), "bias_cm": round(float(b), 2),
                      "residual_std_cm": round(float(resid.std()), 2)}
        out.setdefault("points", [(float(ti), float(oi)) for ti, oi in zip(t, o)])
        out.setdefault("units", "normalized")
    return out


def evaluate(readings: list[dict], truth_of, settle: int | None = None) -> dict:
    """
    readings from replay(); truth_of(fid) -> the label dict or None. Scores whatever the labels cover.
    The vote isn't scored for settle frames after a field's truth changes (default: Phase 3's window less 1).
    """
    if settle is None:
        from src.config import MEASURED_ESTIMATION
        settle = MEASURED_ESTIMATION.vote_window - 1
    sign_f, sign_v, light_f, drive_v, line_f, line_v, ob_f, ob_v, off = ([] for _ in range(9))
    labelled = 0
    last, since = {}, {}

    def steady(field, value) -> bool:
        presence = value if field in ("sign", "light") else value is not NONE
        if last.get(field, presence) != presence:
            since[field] = 0
        last[field] = presence
        since[field] = since.get(field, settle) + 1
        return since[field] > settle

    for r in readings:
        lb = truth_of(r["fid"])
        if lb is None:
            last.clear()
            continue
        labelled += 1
        voted = "vote_sign" in r
        if "sign" in lb:
            sign_f.append((lb["sign"], r["sign"], lb))
            if voted and steady("sign", lb["sign"]):
                sign_v.append((lb["sign"], r["vote_sign"], lb))
        if "light" in lb:
            light_f.append((lb["light"], r["light"], lb))
            if voted and steady("light", lb["light"]):
                drive_v.append((VOTED_FOR[lb["light"]], r["vote_drive"]))
        if "line_cm" in lb:
            line_f.append((lb["line_cm"], r["line"], r["line_cm"], lb))
            if voted and steady("line_cm", lb["line_cm"]):
                line_v.append((lb["line_cm"], r["vote_line"], r["vote_line_cm"], lb))
        if "obstacle_cm" in lb:
            ob_f.append((lb["obstacle_cm"], r["obstacle"], r["obstacle_cm"], lb))
            if voted and steady("obstacle_cm", lb["obstacle_cm"]):
                ob_v.append((lb["obstacle_cm"], r["vote_obstacle"], r["vote_obstacle_cm"], lb))
        if "offset_cm" in lb:
            off.append((lb["offset_cm"], r["offset"], r.get("vote_offset_cm")))
    res = {"frames": len(readings), "labelled": labelled, "vote_settle_frames": settle}
    if sign_f:
        res["sign"] = {"frame": score_presence(sign_f), **({"voted": score_presence(sign_v)} if sign_v else {})}
    if light_f:
        res["light"] = {"frame": score_light(light_f), **({"voted": score_drive(drive_v)} if drive_v else {})}
    if line_f:
        res["line"] = {"frame": score_distance(line_f, TOLERANCE_CM["line"]),
                       **({"voted": score_distance(line_v, TOLERANCE_CM["line"])} if line_v else {})}
    if ob_f:
        res["obstacle"] = {"frame": score_distance(ob_f, TOLERANCE_CM["obstacle"]),
                           **({"voted": score_distance(ob_v, TOLERANCE_CM["obstacle"])} if ob_v else {})}
    if off:
        res["offset"] = score_offset(off)
    res["findings"] = findings(res)
    return res


# ---- words ----

def _pct(s: dict) -> str:
    if not s or s.get("share") is None:
        return "–"
    lo, hi = s["ci95"]
    return f"{100 * s['share']:.0f}% ({s['k']}/{s['n']}; 95% {100 * lo:.0f}–{100 * hi:.0f})"


def findings(res: dict) -> list[str]:
    out = []
    if res["labelled"] == 0:
        return ["no frame is labelled: check the label ranges against the frame ids"]
    for name, what in (("sign", "stop sign"), ("line", "stop line"), ("obstacle", "obstacle")):
        if name not in res:
            continue
        f = res[name]["frame"]
        if f["tp"] + f["fn"] and f["tp"] + f["fn"] < MIN_POSITIVES:
            out.append(f"{what}: only {f['tp'] + f['fn']} labelled frames with one there; the recall's interval "
                       "is wide: label more before judging it")
        if f["fp"]:
            out.append(f"{what}: {f['fp']} false alarms in {f['fp'] + f['tn']} frames with none there "
                       f"({_pct(f['false_alarm'])}); the frames are listed by condition")
        weak = [(k, v) for k, v in f["by_range"].items() if v["n"] >= 5 and v["share"] < 0.5]
        if weak:
            out.append(f"{what}: found in under half the frames at {', '.join(k for k, _ in weak)} cm: the range ends there")
        for c, v in f["by_condition"].items():
            if v["recall"]["n"] >= 5 and v["recall"]["share"] is not None and f["recall"]["share"] \
                    and v["recall"]["share"] < f["recall"]["share"] - 0.2:
                out.append(f"{what}: under '{c}' recall falls to {_pct(v['recall'])}")
        e = f.get("error_cm", {})
        if e.get("n") and abs(e["mean"]) > e["tol_cm"] / 2:
            out.append(f"{what}: distances read {e['mean']:+.1f} cm off on average: a bias, so recalibrate "
                       "the stop-line table before tuning the detector")
        if f.get("no_cm"):
            out.append(f"{what}: {f['no_cm']} detections had no cm (no table or homography for this lens)")
    if "light" in res:
        L = res["light"]["frame"]
        if L["wrong_colour"]["k"]:
            out.append(f"traffic light: {L['wrong_colour']['k']} frames read a colour that wasn't lit: "
                       "the hazard case, see the matrix")
        v = res["light"].get("voted")
        if v and v["would_run_it"]["k"]:
            out.append(f"traffic light: the vote said go on {_pct(v['would_run_it'])} of red or yellow frames")
    if "offset" in res:
        o = res["offset"]
        if "error_cm" in o and o["error_cm"]["n"]:
            e = o["error_cm"]
            out.append(f"lane offset: {e['mae']:.1f} cm mean error, {_pct(e['within_tol'])} within ±2 cm (P3)")
        elif "fit" in o:
            out.append(f"lane offset: no cm scale in Phase 3; the fit reads {o['fit']['cm_per_unit']} cm per unit, "
                       f"bias {o['fit']['bias_cm']:+.1f} cm, scatter {o['fit']['residual_std_cm']} cm")
    return out


def report_lines(res: dict) -> list[str]:
    lines = [f"{res['frames']} frames, {res['labelled']} labelled"]
    for name, what in (("sign", "stop sign"), ("line", "stop line"), ("obstacle", "obstacle")):
        if name not in res:
            continue
        for how, s in res[name].items():
            lines.append(f"{what:<12} {how:<6} recall {_pct(s['recall'])}, precision {_pct(s['precision'])}, "
                         f"false alarms {_pct(s['false_alarm'])}")
            e = s.get("error_cm", {})
            if e.get("n"):
                lines.append(f"{'':<19} error {e['mean']:+.2f} ± {e['std']:.2f} cm, |error| p95 {e['p95_abs']:.2f}, "
                             f"within ±{e['tol_cm']:g} cm {_pct(e['within_tol'])}")
            if s["by_range"]:
                lines.append(f"{'':<19} by range: " + ", ".join(f"{k} cm {_pct(v)}" for k, v in s["by_range"].items()))
    if "light" in res:
        L = res["light"]["frame"]
        lines.append(f"traffic light frame  accuracy {_pct(L['accuracy'])}, wrong colour {_pct(L['wrong_colour'])}, "
                     f"lit but missed {_pct(L['missed_lit'])}")
        lines.append("    truth \\ read     red  yellow  green  none")
        for t, row in L["matrix"].items():
            lines.append(f"    {t:<15}" + "".join(f"{row[s]:>7}" for s in ("red", "yellow", "green", NONE)))
        if "voted" in res["light"]:
            v = res["light"]["voted"]
            lines.append(f"traffic light voted  accuracy {_pct(v['accuracy'])}, go on a red or yellow {_pct(v['would_run_it'])}")
    if "offset" in res:
        o = res["offset"]
        lines.append(f"lane offset          seen {_pct(o['seen'])}"
                     + (f", error {o['error_cm']['mean']:+.2f} ± {o['error_cm']['std']:.2f} cm" if o.get("error_cm", {}).get("n") else "")
                     + (f", fit {o['fit']['cm_per_unit']} cm/unit, bias {o['fit']['bias_cm']:+.2f} cm" if "fit" in o else ""))
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def compare_lines(res: dict, base: dict) -> list[str]:
    """The headline numbers against an earlier eval.json on the same frames."""
    def get(r, *keys):
        for k in keys:
            if not isinstance(r, dict) or k not in r:
                return None
            r = r[k]
        return r
    out = []
    for path, label in ((("sign", "frame", "recall", "share"), "stop sign recall"),
                        (("sign", "frame", "false_alarm", "share"), "stop sign false alarms"),
                        (("light", "frame", "accuracy", "share"), "light accuracy"),
                        (("light", "frame", "wrong_colour", "share"), "light wrong colour"),
                        (("line", "frame", "recall", "share"), "stop line recall"),
                        (("line", "frame", "error_cm", "mae"), "stop line |error| cm"),
                        (("obstacle", "frame", "recall", "share"), "obstacle recall"),
                        (("obstacle", "frame", "false_alarm", "share"), "obstacle false alarms"),
                        (("offset", "error_cm", "mae"), "lane offset |error| cm")):
        a, b = get(base, *path), get(res, *path)
        if a is None or b is None:
            continue
        pct = path[-1] == "share"
        f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.2f}")
        out.append(f"  {label:<24} {f(a):>7} -> {f(b):<7} ({'+' if b >= a else ''}{100 * (b - a) if pct else b - a:.1f}"
                   f"{' pts' if pct else ''})")
    if base.get("labelled") != res.get("labelled"):
        out.append(f"  (labelled frames {base.get('labelled')} -> {res.get('labelled')}: not the same set)")
    return out


def figure(res: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    panels = [k for k in ("light", "sign", "line", "obstacle", "offset") if k in res]
    if not panels:
        return None
    cols = min(len(panels), 3)
    rows_ = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.3 * rows_), squeeze=False)
    for ax, k in zip(axes.flat, panels):
        if k == "light":
            m = res["light"]["frame"]["matrix"]
            said = ("red", "yellow", "green", NONE)
            a = np.array([[m[t][s] for s in said] for t in LIGHTS], float)
            norm = a / np.maximum(a.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(LIGHTS)):
                for j in range(len(said)):
                    ax.text(j, i, f"{int(a[i, j])}", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "#222", fontsize=8)
            ax.set_xticks(range(4), said)
            ax.set_yticks(range(4), LIGHTS)
            ax.set_xlabel("read (Phase 2 through the gate)")
            ax.set_ylabel("truth")
            ax.set_title("Traffic light, frame by frame", fontsize=9, loc="left")
            ax.grid(False)
        elif k == "sign":
            f = res["sign"]["frame"]
            br = f["by_range"]
            if br:
                xs = list(br)
                ys = [v["share"] for v in br.values()]
                lo = [v["share"] - v["ci95"][0] for v in br.values()]
                hi = [v["ci95"][1] - v["share"] for v in br.values()]
                ax.bar(xs, ys, color="#1E63B5", yerr=[lo, hi], capsize=3)
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("range (cm)")
                ax.set_ylabel("share of frames found")
            else:
                ax.bar(["recall", "false alarms"], [f["recall"]["share"] or 0, f["false_alarm"]["share"] or 0],
                       color=["#1E63B5", "#C62828"])
                ax.set_ylim(0, 1.05)
            ax.set_title(f"Stop sign: recall {_pct(f['recall']).split(' (')[0]}, false alarms "
                         f"{_pct(f['false_alarm']).split(' (')[0]}", fontsize=9, loc="left")
        elif k in ("line", "obstacle"):
            f = res[k]["frame"]
            pts = f["points"]
            if pts:
                t, c = zip(*pts)
                ax.plot(t, c, ".", ms=3, color="#1E63B5", alpha=0.5)
                lim = [0, max(max(t), max(c)) * 1.1]
                ax.plot(lim, lim, color="#555", lw=0.7)
                tol = f["error_cm"]["tol_cm"]
                ax.fill_between(lim, [x - tol for x in lim], [x + tol for x in lim], color="#2A9A88", alpha=0.12)
            else:
                ax.text(0.5, 0.5, "nothing to compare in cm:\n" + (f"{f['no_cm']} found without a cm\n(no table for this lens)"
                        if f.get("no_cm") else "no labelled distance found"), ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#555")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_xlabel("tape (cm)")
            ax.set_ylabel("reported (cm)")
            ax.set_title(f"{'Stop line' if k == 'line' else 'Obstacle'}: found {_pct(f['recall']).split(' (')[0]}, "
                         f"false alarms {_pct(f['false_alarm']).split(' (')[0]}", fontsize=9, loc="left")
        else:
            o = res["offset"]
            if o.get("points"):
                t, c = zip(*o["points"])
                ax.plot(t, c, ".", ms=3, color="#1E63B5", alpha=0.5)
                if o["units"] == "cm":
                    lim = [min(t + c) - 1, max(t + c) + 1]
                    ax.plot(lim, lim, color="#555", lw=0.7)
                    ax.fill_between(lim, [x - 2 for x in lim], [x + 2 for x in lim], color="#2A9A88", alpha=0.12)
            ax.set_xlabel("ruler (cm, + = robot right)")
            ax.set_ylabel(f"reported ({o.get('units', '')})")
            ax.set_title("Lane offset", fontsize=9, loc="left")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---- command line ----

def find_frames(run: Path) -> Path:
    """run/frames, or run itself if it holds the recorder's numbered frames (not a run's report PNGs)."""
    run = Path(run)
    frames = run / "frames"
    if frames.is_dir() and any(q.suffix.lower() in (".jpg", ".png") for q in frames.iterdir()):
        return frames
    if run.is_dir() and any(q.suffix.lower() in (".jpg", ".png") and q.stem.isdigit() for q in run.iterdir()):
        return run
    raise FileNotFoundError(f"no frames in {run / 'frames'}: the recorder writes them there "
                            "(copy the frames/ folder with the run)")


def run_eval(run: Path, labels_path: Path | None = None, config=None, estimation=None) -> tuple[dict, str]:
    """Scores for a run folder (its frames and labels.csv) or a routine folder; (result, what was scored)."""
    run = Path(run)
    auto = labels_from_routine(run) if labels_path is None else []
    if auto:
        readings = replay([(i, p) for i, (p, _) in enumerate(auto)], config, estimation, sequence=False)
        truth = {i: {"first": i, "last": i, **t} for i, (_, t) in enumerate(auto)}
        return evaluate(readings, truth.get), f"{routine_kind(run)} looks, labelled by their names"
    frames = find_frames(run)
    lp = labels_path or next((p for p in (run / "labels.csv", frames / "labels.csv") if p.is_file()), None)
    if lp is None:
        raise FileNotFoundError(f"no labels.csv in {run}: write one (make label RUN={run}) or pass --labels")
    labels = read_labels(lp)
    unchecked = sum(lb["last"] - lb["first"] + 1 for lb in labels if lb.get("condition") == UNCHECKED)
    labels = [lb for lb in labels if lb.get("condition") != UNCHECKED]
    items = frames_in(frames)
    readings = replay(items, config, estimation, sequence=True)
    res = evaluate(readings, lambda fid: truth_for(labels, fid))
    if unchecked:
        res["unchecked_frames"] = unchecked
        res["findings"].insert(0, f"{unchecked} frames are still an unchecked draft and weren't scored: "
                                  "review them (make label: n, k)")
    return res, f"{len(items)} frames, labels {lp.name}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.evaluate", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="a run folder with frames/ and labels.csv, or a detect-range / obstacle-check / "
                               "lane-offset routine folder")
    p.add_argument("--labels", help="labels.csv elsewhere")
    p.add_argument("--base", help="an earlier eval.json on the same frames, to compare")
    p.add_argument("--out", help="output folder (default: the run folder)")
    args = p.parse_args(argv)
    run = Path(args.run)
    try:
        res, what = run_eval(run, Path(args.labels) if args.labels else None)
        base = json.loads(Path(args.base).read_text()) if args.base else None
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    from src.utils import code_version
    res["code"] = code_version()
    res["scored"] = what
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "eval.json", res)
    fig = figure(res, f"Labelled replay: {run.name}", out / "eval.png")
    print(f"{run}: {what}")
    print("\n".join(report_lines(res)))
    if base:
        print("against the base:")
        print("\n".join(compare_lines(res, base)))
    print(f"wrote {out / 'eval.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
