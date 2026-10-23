"""Serial against parallel perception, on the same frames.

Purpose:
    perception/parallel.py runs Phase 2's color and obstacle branches on a
    side thread while the main thread runs geometry. Whether that wins on
    the Pi depends on how much of each branch is inside OpenCV (which
    releases the GIL), so it's measured rather than assumed
    (docs/guides/analysis.md, "Parallel perception").

Main package:
    compare.json in --out: per mode, Phase 2 wall ms per frame, the main
    thread's branch_wait, the side thread's own times, and CPU in cores;
    the verdict is parallel p50 against serial p50.

Flow:
    1. Frames loaded into memory (--frames, a video, or --camera), so file
       reads don't add to the timings.
    2. Pipeline.perceive() in blocks alternating serial and parallel, at
       each --opencv-threads count.
    3. Statistics per mode; print, JSON.

Usage (from vision_stack/, on the Pi, with the robot's pipeline stopped if --camera):
    python3 -m src.analysis.parallel_compare --frames runs/nav_<time>/frames
    python3 -m src.analysis.parallel_compare --camera --count 200
    python3 -m src.analysis.parallel_compare --frames DIR --opencv-threads 1,2
    make compare-parallel ARGS="--frames runs/nav_<time>/frames"
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import MEASURED
from src.params import FPS, FRAME_H, FRAME_W, OPENCV_THREADS, RUNS_DIR
from src.pipeline import Pipeline

DEFAULT_COUNT = 200
BLOCK = 20                     # frames per block before switching mode
WARMUP = 5                     # frames per mode run first and not counted


def load_frames(source, count: int) -> list:
    """Up to count (frame, frame_id, timestamp_ms) from a live_view FrameSource, held in memory."""
    frames = []
    while len(frames) < count:
        item = source.read()
        if item is None:
            break
        if item[0] is None:
            continue
        frames.append(item)
    return frames


def _pct(values, q):
    return round(float(np.percentile(values, q)), 2) if values else None


def measure(frames: list, modes: list[tuple[str, bool, int]], block: int = BLOCK, config=MEASURED,
            clock=time.perf_counter, cpu=time.process_time) -> dict:
    """
    Every mode over every frame, in alternating blocks.

    Inputs:
        frames: (frame, frame_id, timestamp_ms) held in memory.
        modes: (label, parallel, opencv_threads) each.
    Outputs:
        {label: {frames, wall_ms {p50, p95, max}, branch_wait_ms {p50, p95},
        side_ms {color, obstacle: p50}, cpu_cores}}.
    """
    pipes = {label: Pipeline(config, timing=True, parallel=par) for label, par, _ in modes}
    walls = {label: [] for label, _, _ in modes}
    waits = {label: [] for label, _, _ in modes}
    side = {label: {"color": [], "obstacle": []} for label, _, _ in modes}
    busy = {label: [0.0, 0.0] for label, _, _ in modes}            # cpu s, wall s
    for label, _, threads in modes:                                   # first-call allocation, untimed
        cv2.setNumThreads(threads)
        for f, fid, ts in frames[:WARMUP]:
            pipes[label].perceive(f, fid, ts)
    for start in range(0, len(frames), block):
        for label, _, threads in modes:
            cv2.setNumThreads(threads)
            p = pipes[label]
            c0, w0 = cpu(), clock()
            for f, fid, ts in frames[start:start + block]:
                t0 = clock()
                p.perceive(f, fid, ts)
                walls[label].append((clock() - t0) * 1000.0)
                if "branch_wait" in p.last_timings_ms:
                    waits[label].append(p.last_timings_ms["branch_wait"])
                    for k in side[label]:
                        side[label][k].append(p.last_side_ms.get(k, 0.0))
            busy[label][0] += cpu() - c0
            busy[label][1] += clock() - w0
    out = {}
    for label, par, threads in modes:
        w = walls[label]
        out[label] = {
            "parallel": par, "opencv_threads": threads, "frames": len(w),
            "wall_ms": {"p50": _pct(w, 50), "p95": _pct(w, 95), "max": round(max(w), 2) if w else None},
            "branch_wait_ms": {"p50": _pct(waits[label], 50), "p95": _pct(waits[label], 95)} if par else None,
            "side_ms": {k: _pct(v, 50) for k, v in side[label].items()} if par else None,
            "cpu_cores": round(busy[label][0] / busy[label][1], 2) if busy[label][1] > 0 else None,
        }
    return out


def _cell(v, width: int = 8, fmt: str = ".1f") -> str:
    return f"{v:>{width}{fmt}}" if v is not None else f"{'-':>{width}}"


def report(results: dict) -> list[str]:
    """The comparison as printed lines, with each parallel mode against the serial one at the same thread count."""
    lines = [f"{'mode':<22}{'p50 ms':>8}{'p95':>8}{'max':>8}{'wait p50':>10}{'color':>8}{'obst':>8}{'cores':>7}"]
    for label, r in results.items():
        w, wait, side = r["wall_ms"], r["branch_wait_ms"] or {}, r["side_ms"] or {}
        lines.append(f"{label:<22}{_cell(w['p50'])}{_cell(w['p95'])}{_cell(w['max'])}"
                     f"{_cell(wait.get('p50'), 10)}{_cell(side.get('color'))}{_cell(side.get('obstacle'))}"
                     f"{_cell(r['cpu_cores'], 7, '.2f')}")
    for label, r in results.items():
        if not r["parallel"]:
            continue
        base = next((b for b in results.values() if not b["parallel"] and b["opencv_threads"] == r["opencv_threads"]), None)
        if base and base["wall_ms"]["p50"] and r["wall_ms"]["p50"]:
            gain = base["wall_ms"]["p50"] - r["wall_ms"]["p50"]
            lines.append(f"{label}: {gain:+.1f} ms per frame at p50 against serial "
                         f"({100.0 * gain / base['wall_ms']['p50']:+.0f}%)")
    return lines


def main(argv=None) -> int:
    from src.debugger.live_view import CameraFrameSource, DirectoryFrameSource, VideoFrameSource
    ap = argparse.ArgumentParser(prog="parallel_compare", description="Serial against parallel perception, same frames.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Usage")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true")
    src.add_argument("--video", metavar="PATH")
    src.add_argument("--frames", metavar="DIR")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT, help="frames held in memory (default %(default)s)")
    ap.add_argument("--opencv-threads", default=str(OPENCV_THREADS), metavar="N[,N]",
                    help="OpenCV thread counts to try each mode at (default %(default)s)")
    ap.add_argument("--out", default=None, metavar="DIR")
    a = ap.parse_args(argv)

    source = (CameraFrameSource(FRAME_W, FRAME_H, FPS) if a.camera else
              VideoFrameSource(a.video) if a.video else DirectoryFrameSource(a.frames))
    try:
        frames = load_frames(source, a.count)
    finally:
        source.close()
    if len(frames) <= WARMUP:
        print(f"only {len(frames)} frames: need more than {WARMUP}")
        return 1
    threads = [int(t) for t in a.opencv_threads.split(",") if t.strip()]
    modes = [(f"{'parallel' if par else 'serial'} cv{t}", par, t) for t in threads for par in (False, True)]
    print(f"{len(frames)} frames in memory from {source.label}; modes: {', '.join(m[0] for m in modes)}")
    results = measure(frames, modes)
    cv2.setNumThreads(OPENCV_THREADS)
    print("\n".join(report(results)))
    out = Path(a.out) if a.out else RUNS_DIR / f"parallel_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps({"source": source.label, "frames": len(frames), "modes": results},
                                                 indent=2) + "\n")
    print(f"wrote {out / 'compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
