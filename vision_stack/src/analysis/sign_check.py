"""Why a stop sign in view isn't detected: which gate, and what relaxing it would find.

Purpose:
    The detector gates red blobs largest first and takes the first that
    passes. This replays frames through Phases 1-2, gates every blob on its
    own, then relaxes one gate at a time, naming the gate that brings the sign
    back (docs/guides/analysis.md, "Stop sign misses").

Main package:
    sign_check.csv (per frame: blobs, the largest's gate and values, ROI edges
    touched, hidden passes, relaxed passes) and sign_check.json (totals).

Usage (from vision_stack/, on the Pi or a laptop with the frames):
    python3 -m src.analysis.sign_check --frames runs/nav_<time>/frames
    python3 -m src.analysis.sign_check --video runs/nav_<time>/nav.avi
    python3 -m src.analysis.sign_check --frames runs/routine_detect-range_<time>    the routine's saved looks
    make sign-check ARGS="--frames runs/nav_<time>/frames"
"""
import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.config import MEASURED
from src.params import RUNS_DIR

# (name, SignContourFilter changes), one gate relaxed at a time
RELAXED = (
    ("vertices 7-10", {"min_vertices": 7, "max_vertices": 10}),
    ("solidity 0.7", {"min_solidity": 0.70}),
    ("no aspect", {"aspect_range": None}),
    ("no extent", {"extent_range": None}),
    ("no letters", {"min_letter_share": None}),
    ("no ghost gates", {"aspect_range": None, "extent_range": None, "min_letter_share": None}),
)


def gate_of(contour, sign_filter, sign_color_roi) -> tuple[str | None, dict]:
    """(the refusing gate or None, what was measured) for one blob on its own."""
    import cv2
    from src.perception import geometry as g
    area = cv2.contourArea(contour)
    got = {"area": round(area, 1)}
    if area < sign_filter.min_area or area > sign_filter.max_area:
        return "area", got
    hull, hull_area, approx = g._sign_polygon(contour, sign_filter)
    if hull_area <= 0:
        return "hull", got
    got["vertices"] = len(approx)
    if not sign_filter.min_vertices <= len(approx) <= sign_filter.max_vertices:
        return "vertices", got
    got["solidity"] = round(area / hull_area, 3)
    if got["solidity"] < sign_filter.min_solidity:
        return "solidity", got
    gate, shape = g._shape_gate(contour, area, hull, sign_color_roi, sign_filter)
    got.update(shape)
    return gate, got


def blobs(sign_color_roi, sign_filter) -> list:
    """The red mask's external contours at or over min_area, largest first."""
    import cv2
    from src.perception import geometry as g
    mask, _ = g._red_mask(g._redness(sign_color_roi), sign_filter)
    sized = [c for c in g._contours(mask) if cv2.contourArea(c) >= sign_filter.min_area]
    return sorted(sized, key=cv2.contourArea, reverse=True)


def edges_touched(contour, shape) -> str:
    """ROI sides the blob's box touches, joined by "+"; "" for none."""
    import cv2
    x, y, w, h = cv2.boundingRect(contour)
    rows, cols = shape[:2]
    sides = [("top", y <= 0), ("bottom", y + h >= rows), ("left", x <= 0), ("right", x + w >= cols)]
    return "+".join(s for s, hit in sides if hit)


def check_frame(sign_color_roi, sign_filter) -> dict:
    """
    One frame's sign ROI.

    hidden: the largest is refused but a smaller blob passes (missed before
    2026-10-20). relaxed: for an undetected frame, the RELAXED names that pass a blob.
    """
    found = blobs(sign_color_roi, sign_filter)
    row = {"blobs": len(found), "largest_area": None, "largest_gate": None, "measured": {}, "cut": "",
           "detected": False, "hidden": False, "relaxed": []}
    if not found:
        return row
    gate, got = gate_of(found[0], sign_filter, sign_color_roi)
    row.update(largest_area=got["area"], largest_gate=gate, measured=got,
               cut=edges_touched(found[0], sign_color_roi.shape))
    if gate is None:
        row["detected"] = True
        return row
    row["hidden"] = any(gate_of(c, sign_filter, sign_color_roi)[0] is None for c in found[1:])
    row["detected"] = row["hidden"]                         # the detector goes on to the smaller blobs
    if not row["detected"]:
        row["relaxed"] = [name for name, change in RELAXED
                          if any(gate_of(c, replace(sign_filter, **change), sign_color_roi)[0] is None
                                 for c in found)]
    return row


def summarize(rows: list[dict]) -> dict:
    red = [r for r in rows if r["blobs"]]
    refused = [r for r in red if not r["detected"]]
    largest_refused = [r for r in red if r["largest_gate"] is not None]
    by_gate = {}
    for r in largest_refused:
        by_gate[r["largest_gate"]] = by_gate.get(r["largest_gate"], 0) + 1
    return {"frames": len(rows), "with_red_blob": len(red), "detected": sum(r["detected"] for r in red),
            "found_behind_larger": sum(r["hidden"] for r in red),
            "refused_cut_by_roi": sum(bool(r.get("cut")) for r in refused),
            "refused_by_gate": dict(sorted(by_gate.items(), key=lambda kv: -kv[1])),
            "relaxed_adds": {name: sum(name in r["relaxed"] for r in refused) for name, _ in RELAXED}}


def report(s: dict) -> list[str]:
    lines = [f"{s['frames']} frames, {s['with_red_blob']} with a red blob in the sign ROI, "
             f"{s['detected']} detected as the robot runs",
             f"found behind a larger, refused red blob (missed before 2026-10-20): {s['found_behind_larger']} frames",
             f"refused while cut off by the ROI's edge (the sign partly outside it): {s['refused_cut_by_roi']} frames",
             "the largest blob refused by (the sign, or something red beside it):"]
    lines += [f"  {g:<16}{n:6}" for g, n in s["refused_by_gate"].items()] or ["  (none)"]
    lines.append("frames detected with one gate relaxed:")
    lines += [f"  {name:<16}{n:+6}" for name, n in s["relaxed_adds"].items()]
    return lines


def main(argv=None) -> int:
    from src.debugger.live_view import DirectoryFrameSource, VideoFrameSource
    from src.phase2_linker import run_chain
    ap = argparse.ArgumentParser(prog="sign_check", description="Why a stop sign in view isn't detected.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Usage")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", metavar="DIR")
    src.add_argument("--video", metavar="PATH")
    ap.add_argument("--out", default=None, metavar="DIR")
    a = ap.parse_args(argv)

    source = DirectoryFrameSource(a.frames) if a.frames else VideoFrameSource(a.video)
    sign_filter = MEASURED.geometry.sign
    rows = []
    try:
        while (item := source.read()) is not None:
            frame, fid, ts = item
            if frame is None:
                continue
            chain = run_chain(frame, fid, ts, MEASURED)
            rows.append({"frame_id": fid, **check_frame(chain.roi.sign_color_roi, sign_filter)})
    finally:
        source.close()
    s = summarize(rows)
    print(f"{source.label}:")
    print("\n".join(report(s)))
    out = Path(a.out) if a.out else RUNS_DIR / f"sign_check_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sign_check.csv", "w") as f:
        f.write("frame_id,blobs,largest_area,largest_gate,cut,vertices,solidity,aspect,extent,letters,detected,hidden,"
                "relaxed\n")
        for r in rows:
            m = r["measured"]
            f.write(",".join(str(v) for v in (
                r["frame_id"], r["blobs"], "" if r["largest_area"] is None else r["largest_area"],
                r["largest_gate"] or ("" if not r["blobs"] else "passed"), r["cut"],
                m.get("vertices", ""), m.get("solidity", ""), m.get("aspect", ""), m.get("extent", ""),
                m.get("letters", ""), int(r["detected"]), int(r["hidden"]), "|".join(r["relaxed"]))) + "\n")
    (out / "sign_check.json").write_text(json.dumps({"source": source.label, "summary": s}, indent=2) + "\n")
    print(f"wrote {out / 'sign_check.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
