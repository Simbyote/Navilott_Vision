#!/usr/bin/env python3
"""Traffic-light sweep: try many HSV floors and blob gates on labelled frames, keep what reads them right.

Purpose:
    calibrate_lamps measures a lamp as one bright disc with a glow around it
    and puts each band between the two. The course's light isn't that: bare
    LEDs, a white center with a thin colored ring, reflections in the panel
    above them and a second light nearby (2026-10-06), and its measured bands
    passed nothing. This doesn't measure the lamp. It runs the color branch
    itself on frames you've labelled (which lamp was lit) with each setting
    in a grid, and scores what Phase 3 would see: a lamp of the lit color at
    or above its confidence gate is a hit, any other color there a false
    reading. The best settings are the ones that read the most frames right.

    Hue spans stay as they are in hsv_ranges.json; a lamp's hue is rarely
    the problem. What's swept is what took hand-tuning: each color's S and V
    floors, and the blob gates (min_area, ref_area, roundness, core).

Main package:
    parse_label(): "red=runs/nav_x:108-200" -> (label, path, frame span).
    read_labelled(): a run folder's frames in a span, spread to a limit.
    color_score(): one color's hits and false readings under one setting.
    fusion_score(): what the frame reads as (the highest-confidence lamp at
        the gate), against its label, over all frames.
    sweep_bands(), sweep_blob(): one pass over each grid.
    main(): the command line.

Flow:
    1. Read each label's frames; cut the traffic ROI as the robot does.
    2. Start from calibration/hsv_ranges.json and MEASURED's blob gates.
    3. ROUNDS times: each color's (S, V) floor scored on its own (hits minus
       FALSE_WEIGHT x false readings), then the blob gates on whole frames.
       Ties go to the setting whose weakest right reading is strongest,
       then to the middle of those, so the choice isn't on the edge of what
       works.
    4. Print each color's best floors, the frames read right before and
       after, the bands and the BlobFilter; --write saves the bands.
"""
import argparse
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.config import MEASURED, MEASURED_ESTIMATION
from src.params import HSV_RANGES_PATH
from src.perception.color_branch import (BlobFilter, ColorRange, HSVRanges, find_traffic_light_candidates,
                                         load_hsv_ranges)
from src.scripts.calibrate_lamps import COLORS, IMAGE_SUFFIXES, traffic_roi, write_ranges

OFF = "off"                     # label for frames with no lamp lit: anything read there is false
LABELS = COLORS + (OFF,)

# Frames used per label, spread across its span: enough to see a lamp drop
# out, few enough for the Pi (~0.14 ms per detection on the course's ROI)
FRAMES_PER_LABEL = 60
# Grid of floors per color. V does most of the work: a lit LED's ring reads
# V 200+, the board and unlit lenses 150-215 (2026-10-06)
V_FLOORS = (80, 100, 120, 140, 160, 180, 200, 215, 230)
S_FLOORS = (20, 40, 60, 90, 120, 150)
# Blob gates: lamp sizes from bare LEDs (10-35 px^2) to diffused lamps (300-800)
MIN_AREAS = (4.0, 8.0, 15.0, 30.0)
REF_AREAS = (20.0, 30.0, 60.0, 150.0, 350.0)
MIN_ROUNDNESS = (0.2, 0.35, 0.5)
MIN_CORE_PX = (0, 3)
# A false reading costs this many hits: a red read at a green light stops the
# robot, a missed one is caught by the traffic rule's memory
FALSE_WEIGHT = 2
ROUNDS = 2

_CLI_HELP = """\
Sweep the traffic-light HSV floors and blob gates on frames where you know
which lamp was lit, and keep the settings that read the most of them right.
Record with the robot where it stops at the light, motors off:

    make nav-dry MAX_RUN_S=30

then label the frames (frame numbers, from the file names) and sweep:

    python3 -m src.scripts.sweep_lamps green=runs/nav_X:0-86 yellow=runs/nav_X:87-107 red=runs/nav_X:108-200
    python3 -m src.scripts.sweep_lamps red=runs/lamp_red green=runs/lamp_green    whole folders
    ... off=runs/nav_X:300-350          frames with no lamp lit: anything read there is false
    ... --write                         also save the bands to calibration/hsv_ranges.json

The BlobFilter it prints goes in config.py (MEASURED's _TRAFFIC_LIGHT_BLOB).
Leave a few frames out of each span where the light is changing.
"""


# =============================================================================
# Labelled frames
# =============================================================================

def parse_label(arg: str) -> tuple[str, str, tuple[int, int] | None]:
    """'label=path' or 'label=path:A-B' (frames A to B, inclusive) -> (label, path, span or None)."""
    label, sep, rest = arg.partition("=")
    if not sep or label not in LABELS or not rest:
        raise ValueError(f"{arg!r}: expected label=path[:A-B] with label one of {', '.join(LABELS)}")
    m = re.fullmatch(r"(.+):(\d+)-(\d+)", rest)
    if not m:
        return label, rest, None
    a, b = int(m.group(2)), int(m.group(3))
    if a > b:
        raise ValueError(f"{arg!r}: the span's start is after its end")
    return label, m.group(1), (a, b)


def _frame_number(path: Path, index: int) -> int:
    """The frame number in the file name (navigation_linker writes 000123.jpg); else its place in the folder."""
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else index


def read_labelled(path: str, span: tuple[int, int] | None, limit: int = FRAMES_PER_LABEL) -> list[np.ndarray]:
    """An image, or a folder's images (searched below it) within span, spread to at most limit."""
    p = Path(path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        if span is not None:
            files = [f for i, f in enumerate(files) if span[0] <= _frame_number(f, i) <= span[1]]
        if len(files) > limit:
            files = [files[i] for i in np.linspace(0, len(files) - 1, limit).round().astype(int)]
    else:
        raise FileNotFoundError(f"{path}: no such image or folder")
    frames = [img for img in (cv2.imread(str(f)) for f in files) if img is not None]
    if not frames:
        where = f" in frames {span[0]}-{span[1]}" if span else ""
        raise FileNotFoundError(f"{path}: no readable .jpg or .png frames{where}")
    return frames


# =============================================================================
# Scoring
# =============================================================================

def detect(roi: np.ndarray, hsv: HSVRanges, blob: BlobFilter, gate: float) -> list:
    """The color branch's candidates at or above Phase 3's gate."""
    return [c for c in find_traffic_light_candidates(roi, hsv, blob) if c.confidence >= gate]


def color_score(samples: list, color: str, hsv: HSVRanges, blob: BlobFilter, gate: float) -> dict:
    """
    One color on its own: hits (a lamp of this color on its frames), false
    readings (one on any other frame), and the lowest hit confidence.
    """
    hits = false = n = 0
    low = None
    for label, roi in samples:
        mine = [c.confidence for c in detect(roi, hsv, blob, gate) if c.label == color]
        if label == color:
            n += 1
            if mine:
                hits += 1
                low = max(mine) if low is None else min(low, max(mine))
        elif mine:
            false += 1
    return {"hits": hits, "n": n, "false": false, "low": low, "score": hits - FALSE_WEIGHT * false}


def reading(roi: np.ndarray, hsv: HSVRanges, blob: BlobFilter, gate: float) -> tuple[str, float | None]:
    """What the frame reads as: the highest-confidence lamp at the gate (fusion's pick), or OFF."""
    cands = detect(roi, hsv, blob, gate)
    if not cands:
        return OFF, None
    best = max(cands, key=lambda c: c.confidence)
    return best.label, best.confidence


def fusion_score(samples: list, hsv: HSVRanges, blob: BlobFilter, gate: float) -> dict:
    """Whole frames: label -> Counter of readings; right, wrong (a color that wasn't lit), missed; lowest right confidence."""
    table = {label: Counter() for label in LABELS}
    right = wrong = missed = 0
    low = None
    for label, roi in samples:
        got, conf = reading(roi, hsv, blob, gate)
        table[label][got] += 1
        if got == label:
            right += 1
            if conf is not None:
                low = conf if low is None else min(low, conf)
        elif got == OFF:
            missed += 1
        else:
            wrong += 1
    return {"table": table, "right": right, "wrong": wrong, "missed": missed, "low": low,
            "score": right - FALSE_WEIGHT * wrong}


def _best(scored: list):
    """
    The best of [(setting, score dict), ...] in grid order: the highest
    score; among those, the strongest weakest right reading; among those,
    the middle one, away from the edge of what works.
    """
    top = max(r["score"] for _, r in scored)
    tied = [(k, r) for k, r in scored if r["score"] == top]
    margin = max(r["low"] or 0.0 for _, r in tied)
    tied = [k for k, r in tied if (r["low"] or 0.0) == margin]
    return tied[len(tied) // 2]


# =============================================================================
# The sweep
# =============================================================================

def with_floor(hsv: HSVRanges, color: str, s: int, v: int) -> HSVRanges:
    """hsv with this color's S and V floors set (both red bands for red); hue and caps kept."""
    def floor(r: ColorRange) -> ColorRange:
        return ColorRange((r.lower[0], s, v), r.upper)
    if color == "red":
        return replace(hsv, red_low=floor(hsv.red_low), red_high=floor(hsv.red_high))
    return replace(hsv, **{color: floor(getattr(hsv, color))})


def sweep_bands(samples: list, hsv: HSVRanges, blob: BlobFilter, gate: float) -> tuple[HSVRanges, dict]:
    """Each labelled color's floors on its own; returns the new ranges and, per color, every (s, v) -> score."""
    tables = {}
    for color in COLORS:
        if not any(label == color for label, _ in samples):
            continue                                 # not lit in any frame: nothing to fit it to
        grid = [(s, v) for v in V_FLOORS for s in S_FLOORS]
        scores = {sv: color_score(samples, color, with_floor(hsv, color, *sv), blob, gate) for sv in grid}
        s, v = _best([(sv, scores[sv]) for sv in grid])
        hsv = with_floor(hsv, color, s, v)
        tables[color] = scores
    return hsv, tables


def blob_grid() -> list[BlobFilter]:
    return [BlobFilter(min_area=a, max_area=3000.0, ref_area=r, min_roundness=rd, min_core_px=k)
            for a in MIN_AREAS for r in REF_AREAS if r > a for rd in MIN_ROUNDNESS for k in MIN_CORE_PX]


def sweep_blob(samples: list, hsv: HSVRanges, gate: float) -> tuple[BlobFilter, list]:
    """Every blob gate in the grid on whole frames; returns the best (ties: highest lowest-confidence, then middle)."""
    scores = [(b, fusion_score(samples, hsv, b, gate)) for b in blob_grid()]
    return _best(scores), scores


def bands_json(hsv: HSVRanges) -> dict:
    """hsv_ranges.json entries for every band."""
    return {k: {"lower": list(getattr(hsv, k).lower), "upper": list(getattr(hsv, k).upper)}
            for k in ("red_low", "red_high", "yellow", "green")}


# =============================================================================
# Command line
# =============================================================================

def _table_lines(score: dict) -> list[str]:
    head = "  lit \\ read as  " + "".join(f"{c:>8}" for c in LABELS)
    rows = [f"  {label:15}" + "".join(f"{score['table'][label][c]:8d}" for c in LABELS)
            for label in LABELS if sum(score["table"][label].values())]
    low = "-" if score["low"] is None else f"{score['low']:.2f}"
    return [head, *rows, f"  right {score['right']}, wrong color {score['wrong']}, missed {score['missed']}; "
                         f"lowest right confidence {low} (gate {MEASURED_ESTIMATION.min_confidence_traffic})"]


def main(argv: list[str] | None = None, config=MEASURED, say=print) -> int:
    p = argparse.ArgumentParser(prog="sweep_lamps", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", nargs="+", metavar="LABEL=PATH[:A-B]",
                   help=f"{', '.join(LABELS)} = an image or a run folder, optionally frames A to B")
    p.add_argument("--hsv", default=str(HSV_RANGES_PATH), help="the bands to start from (their hues are kept)")
    p.add_argument("--frames", type=int, default=FRAMES_PER_LABEL, help="frames used per label")
    p.add_argument("--write", action="store_true", help="save the swept bands to --out")
    p.add_argument("--out", default=str(HSV_RANGES_PATH))
    args = p.parse_args(argv)
    try:
        samples = [(label, np.ascontiguousarray(traffic_roi(f, config)))
                   for label, path, span in map(parse_label, args.labels)
                   for f in read_labelled(path, span, args.frames)]
        hsv = load_hsv_ranges(args.hsv)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        say(f"ERROR: {exc}")
        return 2
    if not any(label in COLORS for label, _ in samples):
        say("ERROR: no lamp frames: label at least one of red, yellow, green")
        return 2

    gate = MEASURED_ESTIMATION.min_confidence_traffic
    counts = Counter(label for label, _ in samples)
    say("frames: " + ", ".join(f"{k} {counts[k]}" for k in LABELS if counts[k]))
    blob = config.color.blob
    before = fusion_score(samples, hsv, blob, gate)
    say("\nnow (hsv_ranges.json, MEASURED's blob gates):")
    for line in _table_lines(before):
        say(line)

    tables = {}
    for _ in range(ROUNDS):
        hsv, tables = sweep_bands(samples, hsv, blob, gate)
        blob, _scores = sweep_blob(samples, hsv, gate)

    for color, scores in tables.items():
        chosen = tuple(getattr(hsv, "red_low" if color == "red" else color).lower[1:])
        ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])[:5]
        if chosen not in dict(ranked):
            ranked.append((chosen, scores[chosen]))
        say(f"\n{color}: best floors (S, V) on its own, hits / frames, false readings, lowest hit confidence")
        for (s, v), r in ranked:
            low = "-" if r["low"] is None else f"{r['low']:.2f}"
            mark = "   <- chosen" if (s, v) == chosen else ""
            say(f"  S {s:3d}  V {v:3d}   {r['hits']:3d} / {r['n']:3d}   false {r['false']:3d}   {low}{mark}")
    after = fusion_score(samples, hsv, blob, gate)
    say("\nswept:")
    for line in _table_lines(after):
        say(line)

    say("\nbands (hsv_ranges.json):")
    entries = bands_json(hsv)
    for k, e in entries.items():
        say(f"  {k:9s} lower {e['lower']}  upper {e['upper']}")
    say(f"blob gates (config.py, _TRAFFIC_LIGHT_BLOB):\n  BlobFilter(min_area = {blob.min_area}, max_area = "
        f"{blob.max_area}, ref_area = {blob.ref_area}, min_roundness = {blob.min_roundness}, "
        f"min_core_px = {blob.min_core_px})")
    if after["wrong"]:
        say(f"\nWARNING: {after['wrong']} frame(s) still read as a color that wasn't lit. Something in the "
            "traffic ROI looks like a lamp (a reflection, another light): narrow roi_crop.TRAFFIC around it")
    if after["low"] is not None and after["low"] < gate + 0.1:
        say(f"WARNING: the weakest right reading is {after['low']:.2f}, near the {gate} gate: a dimmer room may "
            "drop it. Record there too and sweep both")
    if args.write:
        try:
            write_ranges(args.out, entries)
        except ValueError as exc:
            say(f"ERROR: {exc}")
            return 1
        say(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
