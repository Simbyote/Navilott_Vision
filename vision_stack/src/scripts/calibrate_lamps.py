#!/usr/bin/env python3
"""Traffic-light lamp calibration: frames of each lit lamp -> the HSV bands the color branch detects them with.

Purpose:
    The color branch (perception/color_branch.py) finds a lamp as the
    pixels inside its color's HSV band. A lamp on camera isn't one color: a
    bright core, a ring of the lamp's real color around it, and a glow that
    spreads past the lamp, still tinted but dimmer and paler. A band that
    takes the glow measures the glow, not the lamp. This takes frames the
    robot recorded with one lamp lit, runs them through the robot's own
    preprocess and traffic-ROI crop, and measures the lamp (a disc around
    the brightest blob) against its glow (a ring outside it). Each band's
    hue spans the lamp's; its S and V minimums sit halfway between the
    lamp and the glow, so the band keeps the one and drops the other. It
    says when no S or V threshold separates them (try a darker exposure),
    when two colors' hues overlap, and how large each lamp's blob is under
    its new band, for BlobFilter's areas. It suggests no band at all when
    what it found has no color to measure (a blown-out white spot, or not
    the lamp), and says when two lamps were found at the same spot. --write puts the measured colors
    into calibration/hsv_ranges.json, leaving the others as they were.

Main package:
    traffic_roi(): a recorded frame's traffic ROI, as the pipeline cuts it.
    measure_lamp(): one frame's lamp and glow statistics and suggested band.
    combine(): the median band over a lamp's frames.
    bands_for(): a color's band(s) in hsv_ranges.json form (red: both halves).
    blob_area(): the lamp's largest blob under a band, as the color branch measures it.
    hue_overlaps(): pairs of colors whose hue spans meet.
    write_ranges(): merge bands into hsv_ranges.json, checked by its loader.
    main(): the command line.

Flow:
    1. For each color=path given, read its frames (an image, or a folder
       such as a navigation_linker run, searched for .jpg / .png).
    2. Each frame: preprocess, crop the traffic ROI, find the brightest blob,
       measure the lamp disc and the glow ring around it.
    3. The median band over the frames; the blob area under it.
    4. Print the bands, the warnings and the areas; --write merges them.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import FrameData
from src.config import MEASURED
from src.params import HSV_RANGES_PATH
from src.perception.color_branch import load_hsv_ranges
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois

COLORS = ("red", "yellow", "green")
# A lamp's disc around its brightest blob: the lamps measure 200-300 px^2 in
# the robot's 480x270 frames (2 cm lamps, 2026-10-04), a disc of radius 8-10;
# 8 stays inside the lamp
LAMP_RADIUS_PX = 8
# The glow ring, in lamp radii: clear of the lamp's edge, close enough to be its glow
GLOW_INNER, GLOW_OUTER = 1.5, 3.0
BRIGHT_MARGIN = 5           # V within this of the frame's maximum is the brightest blob
HUE_MARGIN = 4              # widen the lamp's hue span (5th-95th percentile) by this each side
# S and V thresholds: halfway between the lamp's 10th percentile and the glow's 90th
LAMP_PCT, GLOW_PCT = 10, 90
FRAMES_PER_LAMP = 15        # frames read from a folder, spread across it
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# What a lamp with a color to measure looks like. The 2026-10-04 green lamp's
# ring read S ~50; a clipped white spot reads S under ~10 and its hue is
# noise (the first home run, 2026-10-04: lamp S median 3, hue spread 0-114)
MIN_LAMP_S = 25             # the lamp disc's median saturation
MAX_HUE_SPREAD = 30         # the lamp disc's 5th-95th percentile hue span, OpenCV units (60 degrees)
# A band comes from the frames that found a colored lamp, when at least this
# share of them did; the others found something white and brighter, which is
# reported. On 2026-10-04 the first frame found the red lamp (S 214) and most
# of the others a white spot, and the median over all of them hid the lamp
MIN_COLORED_SHARE = 0.5

_CLI_HELP = """\
Calibrate the traffic-light HSV bands from frames of each lamp, lit the way
it is on the course. Record a few seconds per lamp with the robot where it
stops at the light:

    python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 5 --no-render --out runs/lamp_red

then (any colors, each an image or a run folder):

    python3 -m src.scripts.calibrate_lamps red=runs/lamp_red yellow=runs/lamp_yellow green=runs/lamp_green
    python3 -m src.scripts.calibrate_lamps ... --write        also update calibration/hsv_ranges.json
"""


# =============================================================================
# Measuring
# =============================================================================

def traffic_roi(frame_bgr: np.ndarray, config=MEASURED) -> np.ndarray:
    """The traffic ROI the color branch reads: preprocess (undistortion included), then the crop."""
    pre = preprocess_frame(FrameData(frame_bgr, 0, 0), config.preprocess)
    return crop_rois(pre, config.roi).traffic_roi


def _unwrap(h: np.ndarray) -> np.ndarray:
    """Hues made continuous across red's wrap at 180 when most sit near it: those below 90 move up by 180."""
    return np.where(h < 90, h + 180, h) if np.mean((h < 20) | (h > 160)) > 0.5 else h


def _pcts(a: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in np.percentile(a, (5, 50, 95))]


def measure_lamp(roi_bgr: np.ndarray) -> dict:
    """
    One frame's lamp: the disc of LAMP_RADIUS_PX around the brightest blob's
    middle, against the ring GLOW_INNER-GLOW_OUTER radii out.

    Outputs:
        {"center": (x, y) in ROI px, "lamp" / "glow": {"h", "s", "v"} 5/50/95th
         percentiles (h unwrapped past 180 for red), "band": {"h_lo", "h_hi",
         "s_min", "v_min"} (h_hi may pass 179 for red), "s_gap", "v_gap": the
         lamp's 10th percentile minus the glow's 90th (> 0: that channel separates them)}
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    _, _, stats, cents = cv2.connectedComponentsWithStats((v >= int(v.max()) - BRIGHT_MARGIN).astype(np.uint8))
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y = (float(c) for c in cents[big])
    yy, xx = np.ogrid[:hsv.shape[0], :hsv.shape[1]]
    d = np.hypot(yy - y, xx - x)
    lamp = hsv[d <= LAMP_RADIUS_PX].astype(float)
    glow = hsv[(d > GLOW_INNER * LAMP_RADIUS_PX) & (d <= GLOW_OUTER * LAMP_RADIUS_PX)].astype(float)
    h_lamp, h_glow = _unwrap(lamp[:, 0]), _unwrap(glow[:, 0])
    if h_lamp.max() >= 180 and h_glow.max() < 180:     # red: put the glow on the same side of the wrap
        h_glow = np.where(h_glow < 90, h_glow + 180, h_glow)
    gap = {c: float(np.percentile(lamp[:, i], LAMP_PCT) - np.percentile(glow[:, i], GLOW_PCT)) for c, i in (("s", 1), ("v", 2))}
    mid = {c: float(np.percentile(lamp[:, i], LAMP_PCT) + np.percentile(glow[:, i], GLOW_PCT)) / 2 for c, i in (("s", 1), ("v", 2))}
    return {"center": (round(x, 1), round(y, 1)), "roi_size": (hsv.shape[1], hsv.shape[0]),
            "s_median": float(np.median(lamp[:, 1])), "v_median": float(np.median(lamp[:, 2])),
            "hue_spread": float(np.percentile(h_lamp, 95) - np.percentile(h_lamp, 5)),
            "lamp": {"h": _pcts(h_lamp), "s": _pcts(lamp[:, 1]), "v": _pcts(lamp[:, 2])},
            "glow": {"h": _pcts(h_glow), "s": _pcts(glow[:, 1]), "v": _pcts(glow[:, 2])},
            "band": {"h_lo": float(np.percentile(h_lamp, 5)) - HUE_MARGIN,
                     "h_hi": float(np.percentile(h_lamp, 95)) + HUE_MARGIN,
                     "s_min": mid["s"], "v_min": mid["v"]},
            "s_gap": gap["s"], "v_gap": gap["v"]}


def combine(measures: list[dict]) -> dict:
    """The median band, gaps and color checks over a lamp's frames."""
    med = lambda key: float(np.median([m["band"][key] for m in measures]))       # noqa: E731
    return {"band": {k: med(k) for k in ("h_lo", "h_hi", "s_min", "v_min")},
            "s_median": float(np.median([m["s_median"] for m in measures])),
            "hue_spread": float(np.median([m["hue_spread"] for m in measures])),
            "center": tuple(float(np.median([m["center"][i] for m in measures])) for i in (0, 1)),
            "s_gap": float(np.median([m["s_gap"] for m in measures])),
            "v_gap": float(np.median([m["v_gap"] for m in measures])),
            "frames": len(measures), "first": measures[0]}


def no_color(res: dict) -> str | None:
    """Why the spot measured has no color to calibrate, or None when it has one."""
    if res["s_median"] < MIN_LAMP_S:
        return (f"the spot found is white, not colored (median saturation {res['s_median']:.0f}, "
                f"a lamp's is over {MIN_LAMP_S}): the lamp is blown out, or the brightest thing in the "
                "traffic ROI isn't the lamp")
    if res["hue_spread"] > MAX_HUE_SPREAD:
        return (f"the spot's hue spans {res['hue_spread']:.0f} (one color spans under {MAX_HUE_SPREAD}): "
                "it isn't one lamp's color")
    return None


def is_colored(m: dict) -> bool:
    """True when one frame's spot is a lamp with a color: saturated, and of one hue."""
    return m["s_median"] >= MIN_LAMP_S and m["hue_spread"] <= MAX_HUE_SPREAD


def colored_frames(measures: list[dict]) -> list[dict] | None:
    """The frames that found a colored lamp, when they're at least MIN_COLORED_SHARE of them; else None."""
    colored = [m for m in measures if is_colored(m)]
    return colored if colored and len(colored) >= MIN_COLORED_SHARE * len(measures) else None


def at_edge(m: dict) -> bool:
    """True when the lamp's disc runs past the traffic ROI's border: part of the lamp is out of it."""
    (x, y), (w, h) = m["center"], m["roi_size"]
    return min(x, y, w - 1 - x, h - 1 - y) < LAMP_RADIUS_PX


def same_spot(results: dict) -> list[tuple[str, str]]:
    """Pairs of colors whose lamps were found within a lamp radius of each other: one of them isn't its lamp."""
    names = sorted(results)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if np.hypot(*np.subtract(results[a]["center"], results[b]["center"])) < LAMP_RADIUS_PX]


def bands_for(color: str, band: dict) -> dict:
    """
    The color's entries for hsv_ranges.json. Red always gets both halves:
    a span crossing 180 splits at the wrap; one that doesn't leaves the
    other half a single hue at the edge (0 or 180), still red.
    """
    s, v = int(round(np.clip(band["s_min"], 0, 255))), int(round(np.clip(band["v_min"], 0, 255)))
    lo, hi = band["h_lo"], band["h_hi"]
    if lo >= 180:                                         # unwrapped, but all past the wrap: back to the low side
        lo, hi = lo - 180, hi - 180
    entry = lambda a, b: {"lower": [int(a), s, v], "upper": [int(b), 255, 255]}      # noqa: E731
    if color != "red":
        return {color: entry(max(0, np.floor(lo)), min(179, np.ceil(hi)))}
    if hi >= 180:                                         # the span crosses the wrap
        return {"red_high": entry(max(0, np.floor(min(lo, 179))), 180),
                "red_low": entry(0, min(179, np.ceil(hi - 180)))}
    if lo < 90:                                           # all on the low side
        return {"red_low": entry(max(0, np.floor(lo)), min(179, np.ceil(hi))), "red_high": entry(180, 180)}
    return {"red_high": entry(max(0, np.floor(lo)), 180), "red_low": entry(0, 0)}


def blob_area(roi_bgr: np.ndarray, entries: dict) -> float:
    """The largest blob under the color's band(s), measured as the color branch does (outer contour's area)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for e in entries.values():
        mask |= cv2.inRange(hsv, np.array(e["lower"]), np.array(e["upper"]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max((cv2.contourArea(c) for c in contours), default=0.0)


def _spans(color: str, entries: dict) -> list[tuple[int, int]]:
    return [(e["lower"][0], e["upper"][0]) for e in entries.values() if e["lower"][0] < e["upper"][0] or color != "red"]


def hue_overlaps(per_color: dict) -> list[tuple[str, str]]:
    """Pairs of colors whose hue spans share a hue: a lamp of one could pass as the other."""
    out = []
    names = sorted(per_color)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if any(alo <= bhi and blo <= ahi for alo, ahi in _spans(a, per_color[a])
                   for blo, bhi in _spans(b, per_color[b])):
                out.append((a, b))
    return out


# =============================================================================
# Frames, the file and the command line
# =============================================================================

def read_frames(path: str, limit: int = FRAMES_PER_LAMP) -> list[np.ndarray]:
    """An image, or up to limit images spread across a folder (searched below it)."""
    p = Path(path)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
        if len(files) > limit:
            files = [files[i] for i in np.linspace(0, len(files) - 1, limit).round().astype(int)]
    else:
        raise FileNotFoundError(f"{path}: no such image or folder")
    frames = [img for img in (cv2.imread(str(f)) for f in files) if img is not None]
    if not frames:
        raise FileNotFoundError(f"{path}: no readable .jpg or .png frames")
    return frames


def write_ranges(path, entries: dict) -> None:
    """Merge entries into hsv_ranges.json, keeping the bands not measured; refused unless its loader accepts the result."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.update(entries)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(data, indent=4) + "\n")
    try:
        load_hsv_ranges(str(tmp))
    except (KeyError, ValueError) as exc:
        tmp.unlink()
        raise ValueError(f"not written, the result doesn't load: {exc!r}") from None
    tmp.replace(path)


def _parse(args: list[str]) -> dict:
    out = {}
    for a in args:
        color, sep, path = a.partition("=")
        if not sep or color not in COLORS or not path:
            raise ValueError(f"{a!r}: expected color=path with color one of {', '.join(COLORS)}")
        out[color] = path
    return out


def main(argv: list[str] | None = None, config=MEASURED, say=print) -> int:
    p = argparse.ArgumentParser(prog="calibrate_lamps", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lamps", nargs="+", metavar="COLOR=PATH", help="red, yellow or green = an image or a run folder")
    p.add_argument("--write", action="store_true", help="merge the measured bands into --out")
    p.add_argument("--out", default=str(HSV_RANGES_PATH))
    p.add_argument("--frames", type=int, default=FRAMES_PER_LAMP, help="frames read per folder")
    args = p.parse_args(argv)
    try:
        lamps = _parse(args.lamps)
        rois = {c: [traffic_roi(f, config) for f in read_frames(path, args.frames)] for c, path in lamps.items()}
    except (ValueError, FileNotFoundError) as exc:
        say(f"ERROR: {exc}")
        return 2

    entries, warnings, results = {}, [], {}
    for color, frames in rois.items():
        measures = [measure_lamp(r) for r in frames]
        colored = colored_frames(measures)
        say(f"\n{color}: {len(measures)} frame(s); per frame, the brightest spot in the traffic ROI:")
        for i, m in enumerate(measures):
            kind = "colored" if is_colored(m) else "white / mixed"
            say(f"  {i:3d}  at ({m['center'][0]:5.1f}, {m['center'][1]:5.1f})  S {m['s_median']:5.0f}  "
                f"V {m['v_median']:5.0f}  {kind}")
        res = results[color] = combine(colored or measures)
        first = res["first"]
        say(f"  {'colored frames' if colored else 'all frames'}, the first of them:")
        say("          H p5/p50/p95      S p5/p50/p95      V p5/p50/p95")
        for part in ("lamp", "glow"):
            say(f"  {part}  " + "  ".join(" ".join(f"{x:5.0f}" for x in first[part][c]) for c in "hsv"))
        n_colored = sum(map(is_colored, measures))
        why = None if colored else (no_color(res) or f"only {n_colored} of {len(measures)} frames found a "
                                    "colored lamp, under half: the brightest thing in the traffic ROI is "
                                    "mostly something else")
        if why:
            say(f"  no band: {why}")
            warnings.append(f"{color}: no band suggested; see above")
            continue
        if len(colored) < len(measures):
            others = [m["center"] for m in measures if not is_colored(m)]
            warnings.append(f"{color}: only {len(colored)} of {len(measures)} frames found the colored lamp; the "
                            f"rest found something white and brighter (near {others[0]}): block it, or the band "
                            "may miss the lamp in a run. The band is from the colored frames")
        if at_edge(first):
            warnings.append(f"{color}: the lamp sits at the edge of the traffic ROI ({first['center'][0]:.0f}, "
                            f"{first['center'][1]:.0f} of {first['roi_size'][0]}x{first['roi_size'][1]}): part of "
                            "it is outside. Move the robot back or aim the camera so the light sits inside it")
        e = bands_for(color, res["band"])
        entries[color] = e
        used = [r for r, m in zip(frames, measures) if colored is None or is_colored(m)]
        area = float(np.median([blob_area(r, e) for r in used]))
        for key, val in e.items():
            say(f"  {key:9s} lower {val['lower']}  upper {val['upper']}")
        say(f"  blob area under it: {area:.0f} px^2 (median over the frames)")
        if res["s_gap"] <= 0 and res["v_gap"] <= 0:
            warnings.append(f"{color}: the lamp and its glow overlap in both S and V: no band separates them. "
                            "Darken the exposure and measure again")
        elif res["s_gap"] <= 0 or res["v_gap"] <= 0:
            warnings.append(f"{color}: only {'V' if res['s_gap'] <= 0 else 'S'} separates the lamp from its glow")
    for a, b in same_spot(results):
        warnings.append(f"the {a} and {b} lamps were found at the same spot: the brightest thing in the traffic "
                        "ROI isn't the lit lamp (a reflection, a light behind it), or one diffuser covers both")
    for a, b in hue_overlaps(entries):
        warnings.append(f"{a} and {b} share hues: a {a} lamp could pass for {b}. Darken the exposure "
                        "(an overexposed red goes orange) and measure again")
    say("")
    for w in warnings:
        say(f"WARNING: {w}")

    merged = {k: v for e in entries.values() for k, v in e.items()}
    if args.write and not merged:
        say("nothing written: no lamp gave a band")
        return 1
    if args.write:
        try:
            write_ranges(args.out, merged)
        except ValueError as exc:
            say(f"ERROR: {exc}")
            return 1
        say(f"wrote {', '.join(sorted(merged))} into {args.out}")
    else:
        say("not written: add --write to put these into " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
