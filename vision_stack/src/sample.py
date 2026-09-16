#!/usr/bin/env python3
"""
contact_sheet.py

Purpose:
    Tiles geometry-branch debug overlays into labelled contact sheets so a few
    thousand frames can be reviewed by eye in a handful of images instead of
    one PNG at a time.

Modes:
    accepted  one cell per frame, the accepted-contour overlay
    contours  one cell per frame, the all-contour overlay
    edges     one cell per frame, the raw Canny edges
    pair      one cell per frame, contours stacked above accepted (see what
              each gate removed, frame by frame)

Usage:
    python contact_sheet.py                          # 48 frames, accepted overlays
    python contact_sheet.py --mode pair --count 24   # before/after comparison
    python contact_sheet.py --cols 6 --scale 0.75    # denser sheet
    python contact_sheet.py --dirs path/to/Sample1   # one directory only
"""
import argparse
import os
import sys

import cv2
import numpy as np

DEFAULT_SAMPLE_DIRS = (
    "vision_stack/frames/Sample1",
    "vision_stack/frames/Sample2",
    "vision_stack/frames/Sample3",
)

SUFFIXES = {
    "accepted": "_gb_lane_accepted.png",
    "contours": "_gb_lane_contours.png",
    "edges": "_gb_lane_edges.png",
}

LABEL_H = 16
PAD = 3
BG = (20, 20, 20)
FG = (235, 235, 235)
RULE = (70, 70, 70)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def collect(results_dir: str, suffix: str) -> dict:
    """Map stem -> path for every file in results_dir ending with suffix."""
    if not os.path.isdir(results_dir):
        return {}
    return {
        f.removesuffix(suffix): os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(suffix)
    }


def sample_evenly(items: list, count: int) -> list:
    """Take count items spread across the whole list, not the first count.

    Frames are sequential, so the first N all look alike; an even spread
    covers the varied lighting and road geometry of the full track.
    """
    if count <= 0 or count >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def as_bgr(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def label_cell(img: np.ndarray, text: str) -> np.ndarray:
    """Add a label strip beneath a cell image."""
    h, w = img.shape[:2]
    out = np.full((h + LABEL_H, w, 3), BG, np.uint8)
    out[:h] = img
    cv2.putText(out, text, (3, h + LABEL_H - 4), FONT, 0.36, FG, 1, cv2.LINE_AA)
    return out


def build_cell(stem: str, paths: dict, mode: str, scale: float):
    """Build one labelled cell, or None if a required image is missing."""
    if mode == "pair":
        top = paths["contours"].get(stem)
        bot = paths["accepted"].get(stem)
        if top is None or bot is None:
            return None
        a, b = cv2.imread(top), cv2.imread(bot)
        if a is None or b is None:
            return None
        a, b = as_bgr(a), as_bgr(b)
        if a.shape[1] != b.shape[1]:
            return None
        rule = np.full((1, a.shape[1], 3), RULE, np.uint8)
        cell = np.vstack([a, rule, b])
    else:
        p = paths[mode].get(stem)
        if p is None:
            return None
        cell = cv2.imread(p)
        if cell is None:
            return None
        cell = as_bgr(cell)

    if scale != 1.0:
        cell = cv2.resize(cell, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    return label_cell(cell, stem)


def montage(cells: list, cols: int) -> np.ndarray:
    """Tile equal-sized cells into a padded grid."""
    ch, cw = cells[0].shape[:2]
    rows = -(-len(cells) // cols)
    sheet = np.full((rows * (ch + PAD) + PAD, cols * (cw + PAD) + PAD, 3),
                    BG, np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        y = PAD + r * (ch + PAD)
        x = PAD + c * (cw + PAD)
        sheet[y:y + ch, x:x + cw] = cell
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=list(DEFAULT_SAMPLE_DIRS),
                    help="sample directories containing a results/ subdirectory")
    ap.add_argument("--mode", default="accepted",
                    choices=["accepted", "contours", "edges", "pair"])
    ap.add_argument("--count", type=int, default=48,
                    help="frames per sheet, spread evenly (0 = all)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="cell scale factor, e.g. 0.5 to halve")
    ap.add_argument("--out-dir", default="vision_stack/frames/contact_sheets")
    args = ap.parse_args()

    needed = ["contours", "accepted"] if args.mode == "pair" else [args.mode]
    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    for sample_dir in args.dirs:
        results_dir = os.path.join(sample_dir, "results")
        paths = {k: collect(results_dir, SUFFIXES[k]) for k in needed}

        stems = sorted(set.intersection(*(set(paths[k]) for k in needed)))
        if not stems:
            print(f"[SKIP] no {args.mode} overlays in {results_dir}")
            continue

        picked = sample_evenly(stems, args.count)
        cells = [c for c in (build_cell(s, paths, args.mode, args.scale)
                             for s in picked) if c is not None]
        if not cells:
            print(f"[SKIP] no readable overlays in {results_dir}")
            continue

        # Cells must be identical size to tile; drop any odd ones out.
        shape = cells[0].shape
        mixed = sum(1 for c in cells if c.shape != shape)
        if mixed:
            print(f"[WARN] {mixed} cells with mismatched size dropped "
                  f"in {sample_dir}")
            cells = [c for c in cells if c.shape == shape]

        sheet = montage(cells, args.cols)
        tag = os.path.basename(os.path.normpath(sample_dir))
        out = os.path.join(args.out_dir, f"{tag}_{args.mode}_sheet.png")
        cv2.imwrite(out, sheet)
        written.append(out)
        print(f"[OK] {tag}: {len(cells)}/{len(stems)} frames -> {out} "
              f"({sheet.shape[1]}x{sheet.shape[0]})")

    if not written:
        print("\nNothing written. Run geometry.py first to produce overlays.")
        return 1
    print(f"\nDone. {len(written)} sheet(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())