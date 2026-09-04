"""
view_frame_data.py

Purpose:
    Pull every debug image whose filename contains a given tag (e.g. "lane_accepted") out of a
    dataset's results/ dump and make them inspectable in one place:

      1. copy   -> all matches gathered into <sample_dir>/review/<tag>/ so an image viewer can be
                   arrow-keyed through them in frame order
      2. sheet  -> contact sheets tiling N matches per page with the frame stem labeled under each,
                   written to <sample_dir>/review/<tag>_sheet_XX.png

Usage:
    python view_frame_data.py                 # uses MATCH_TAG below
    python view_frame_data.py sign_accepted   # one-off override

Only SAMPLE_DIRS and the config block below need editing between datasets.
"""

import os
import re
import sys
import shutil

import cv2
import numpy as np

# =============================================================================
# Config
# =============================================================================
SAMPLE_DIRS = [
    "vision_stack/frames/Sample1",
    "vision_stack/frames/Sample2",
    "vision_stack/frames/Sample3",
]

MATCH_TAG = "lane_accepted"      # substring matched against filenames in results/
RESULTS_SUBDIR = "results"
REVIEW_SUBDIR = "review"
MODE = "both"                    # "copy" | "sheet" | "both"

GRID_COLS = 4                    # cells per contact sheet row
PER_SHEET = 16                   # cells per sheet page; keeps sheets viewer-friendly
CELL_W = 480                     # cell width in px; cell height derived from image aspect
LABEL_H = 22                     # label strip height under each cell
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

PAD_BGR = (32, 32, 32)           # letterbox fill
SHEET_BGR = (18, 18, 18)         # sheet background / empty cells
BORDER_BGR = (70, 70, 70)
TEXT_BGR = (230, 230, 230)


# =============================================================================
# Helpers
# =============================================================================
def natural_key(name):
    """Sort frame_2 before frame_10 instead of lexicographically."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def short_label(filename, tag):
    """Strip the shared tag suffix so labels stay compact (frame_0007_gb_lane_accepted -> frame_0007)."""
    stem = os.path.splitext(filename)[0]
    head = stem.split(tag)[0].rstrip("_")
    return head if head else stem


def find_matches(results_dir, tag):
    """Return [(label, path), ...] for every image in results_dir containing tag."""
    matches = []
    for f in sorted(os.listdir(results_dir), key=natural_key):
        if os.path.splitext(f)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        if tag not in f:
            continue
        matches.append((short_label(f, tag), os.path.join(results_dir, f)))
    return matches


def cell_height(paths, cell_w, probe=8):
    """Derive one cell height from the median aspect of the first few images, so ROI crops tile tightly."""
    ratios = []
    for p in paths[:probe]:
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            ratios.append(h / float(w))
    if not ratios:
        return cell_w
    return int(np.clip(cell_w * float(np.median(ratios)), 120, 1200))


def letterbox(img, w, h):
    """Fit img into a w x h canvas without distorting aspect ratio."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    scale = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.full((h, w, 3), PAD_BGR, dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_cell(img, label, cell_w, cell_h):
    """One labeled, bordered cell of fixed size."""
    cell = np.full((cell_h + LABEL_H, cell_w, 3), SHEET_BGR, dtype=np.uint8)
    cell[:cell_h] = letterbox(img, cell_w, cell_h)
    cv2.putText(cell, label[:44], (6, cell_h + LABEL_H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_BGR, 1, cv2.LINE_AA)
    cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h + LABEL_H - 1), BORDER_BGR, 1)
    return cell


def build_sheet(cells, cols):
    """Tile equally sized cells into a grid, padding the last row with blanks."""
    blank = np.full_like(cells[0], SHEET_BGR)
    rows = []
    for i in range(0, len(cells), cols):
        row = list(cells[i:i + cols])
        row += [blank] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else MATCH_TAG

    total_matched = 0
    total_sheets = 0
    total_fail = 0

    for sample_dir in SAMPLE_DIRS:
        results_dir = os.path.join(sample_dir, RESULTS_SUBDIR)
        if not os.path.isdir(results_dir):
            print(f"[SKIP] No results dir: {results_dir}")
            continue

        matches = find_matches(results_dir, tag)
        if not matches:
            print(f"[SKIP] No '{tag}' images in {results_dir}")
            continue

        review_dir = os.path.join(sample_dir, REVIEW_SUBDIR)
        os.makedirs(review_dir, exist_ok=True)

        # --- copy branch -----------------------------------------------------
        if MODE in ("copy", "both"):
            copy_dir = os.path.join(review_dir, tag)
            os.makedirs(copy_dir, exist_ok=True)
            for _, path in matches:
                shutil.copy2(path, os.path.join(copy_dir, os.path.basename(path)))
            print(f"[OK] copied {len(matches)} -> {copy_dir}")

        # --- sheet branch ----------------------------------------------------
        if MODE in ("sheet", "both"):
            cell_h = cell_height([p for _, p in matches], CELL_W)
            page = 0
            for start in range(0, len(matches), PER_SHEET):
                chunk = matches[start:start + PER_SHEET]
                cells = []
                for label, path in chunk:
                    img = cv2.imread(path)
                    if img is None:
                        print(f"[FAIL] Could not read: {path}")
                        total_fail += 1
                        continue
                    cells.append(make_cell(img, label, CELL_W, cell_h))
                if not cells:
                    continue

                sheet_path = os.path.join(review_dir, f"{tag}_sheet_{page:02d}.png")
                cv2.imwrite(sheet_path, build_sheet(cells, GRID_COLS))
                print(f"[OK] sheet {page:02d}: {len(cells)} frames -> {sheet_path}")
                page += 1
                total_sheets += 1

        total_matched += len(matches)

    print(f"\nDone. tag='{tag}'  {total_matched} matched, "
          f"{total_sheets} sheets, {total_fail} failed.")