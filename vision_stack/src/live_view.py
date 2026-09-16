#!/usr/bin/env python3
"""
live_view.py

Video Pipeline Runner

Purpose:
    Runs a video source through every stage built so far and shows what the
    pipeline decided, frame by frame:

        CameraSource -> preprocess_frame -> crop_rois -> run_geometry_stage
                     -> fuse_detections (optional) -> compute_lane_offset

    Overlays come from debug_video.annotate(), so the window and the recorded
    video show the same picture. Output is written whether or not a display is
    attached, which is the point: a run on the robot can be reviewed later.

Sources:
    --camera            live capture through capture.CameraSource
    --video PATH        a recorded file
    --frames DIR        an image sequence, sorted by filename

Output (--out DIR, default vision_stack/runs/<timestamp>):
    run.avi             annotated video
    run.csv             per-frame decision log from DebugVideoWriter
    stages.csv          per-frame stage timings and counts
    summary.txt         mode histogram, availability, blind runs, timings

Display:
    On by default when a display is present. Falls back to headless
    automatically if the window cannot open, so the same command works over
    ssh and on the bench. q quits, space pauses, s saves a still.

Fusion:
    feature_fusion needs color_branch for TrafficLightCandidate. If that import
    fails the run continues with lane offset only and says so, rather than
    taking the whole pipeline down for a branch that is not wired yet.

Notes:
    Stage order is not defined here. It comes from pipeline_linker.run_chain(),
    so the linker's tests and this runner cannot drift apart.
"""
import os
import sys
import time
import csv
from dataclasses import dataclass

import cv2
import numpy as np

from capture import CameraSource, CaptureError, FrameData
from pipeline_linker import run_chain, PipelineConfig, MEASURED
from lane_offset import _usable
import debug_video as dv

# Optional: the color branch may not be wired yet
try:
    from feature_fusion import fuse_detections
    FUSION_AVAILABLE = True
    FUSION_ERROR = None
except Exception as exc:                      # ImportError, or color_branch missing
    FUSION_AVAILABLE = False
    FUSION_ERROR = exc

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_SIZE = (480, 360)       # (w, h)
DEFAULT_FPS = 20
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
STEERING_MODES = ("two_boundary", "left_only", "right_only")

# =============================================================================
# Frame Sources
# =============================================================================
class FrameSource:
    """
    Common interface over the three sources.

    read() yields (frame_bgr, frame_id, timestamp_ms) or None when exhausted.
    Only CameraSource mints its own stamp; file and directory sources
    synthesize one from the frame index and the nominal frame interval, so a
    replay carries the same fields a live run would.
    """
    def __init__(self, label, fps):
        self.label = label
        self.fps = fps
        self._i = 0

    def _stamp(self):
        fid = self._i
        self._i += 1
        return fid, int(fid * 1000.0 / max(self.fps, 1))

    def read(self):
        raise NotImplementedError

    def close(self):
        pass

class CameraFrameSource(FrameSource):
    """Live capture. Keeps capture.py's stamp rather than making a new one."""
    def __init__(self, width, height, fps):
        super().__init__("camera", fps)
        self.cam = CameraSource(width, height, fps)
        self.cam.open()

    def read(self):
        fd = self.cam.read()
        if fd is None:
            return None, None, None          # transient drop; caller continues
        return fd.frame, fd.frame_id, fd.timestamp_ms

    def close(self):
        self.cam.release()

class VideoFrameSource(FrameSource):
    def __init__(self, path, fps):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise CaptureError(f"could not open video {path}")
        native = cap.get(cv2.CAP_PROP_FPS)
        super().__init__(os.path.basename(path),
                         fps or (native if native and native > 1 else DEFAULT_FPS))
        self.cap = cap

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        fid, ts = self._stamp()
        return frame, fid, ts

    def close(self):
        self.cap.release()

class DirectoryFrameSource(FrameSource):
    def __init__(self, path, fps):
        super().__init__(os.path.basename(os.path.normpath(path)), fps)
        self.files = [
            os.path.join(path, f) for f in sorted(os.listdir(path))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not self.files:
            raise CaptureError(f"no images in {path}")
        self._n = 0

    def read(self):
        while self._n < len(self.files):
            frame = cv2.imread(self.files[self._n])
            self._n += 1
            if frame is not None:
                fid, ts = self._stamp()
                return frame, fid, ts
        return None

# =============================================================================
# Candidate Gating, for the overlay
# =============================================================================
def candidate_gates(geometry, config):
    """
    Purpose:
        Pair every raw lane candidate with the gate that rejected it, or None
        if it passed, in the form debug_video.annotate() expects

    Notes:
        Re-runs _usable() per candidate rather than parsing the debug log,
        because a passing candidate logs nothing and the entries cannot be
        aligned back to their candidates by position
    """
    pairs = []
    for cand in geometry.lane_candidates:
        scratch = []
        ok = _usable(cand, config, scratch)
        gate = None
        if not ok and scratch:
            parts = scratch[-1].split()
            gate = parts[1] if len(parts) > 1 else "rejected"
        pairs.append((cand.bbox, gate))
    return pairs

# =============================================================================
# Run Statistics
# =============================================================================
@dataclass
class RunStats:
    """Accumulates what the summary reports."""
    frames: int = 0
    drops: int = 0
    accepted: int = 0
    usable: int = 0
    detections: int = 0
    blind_run: int = 0
    _cur_blind: int = 0

    def __post_init__(self):
        self.modes = {}
        self.gates = {}
        self.offsets = []
        self.stage_ms = {}

    def update(self, result, dbg, gates, n_detections, timings):
        self.frames += 1
        self.modes[result.mode] = self.modes.get(result.mode, 0) + 1
        self.usable += result.boundary_count
        self.accepted += dbg.get("raw_count", 0)
        self.detections += n_detections

        for _bbox, gate in gates:
            if gate:
                self.gates[gate] = self.gates.get(gate, 0) + 1
        for entry in dbg.get("log", ()):
            for tag in ("MERGE", "SPAN"):
                if entry.startswith(f"[{tag}]"):
                    self.gates[tag] = self.gates.get(tag, 0) + 1

        if result.mode in STEERING_MODES:
            self.offsets.append(result.offset)
            self._cur_blind = 0
        else:
            self._cur_blind += 1
            self.blind_run = max(self.blind_run, self._cur_blind)

        for name, ms in timings.items():
            self.stage_ms.setdefault(name, []).append(ms)

    def report(self):
        """Render the summary as a list of lines."""
        out = []
        n = max(self.frames, 1)
        out.append(f"frames processed        {self.frames}")
        out.append(f"dropped reads           {self.drops}")
        out.append(f"candidates accepted     {self.accepted}  "
                   f"({self.accepted/n:.2f} per frame)")
        out.append(f"usable as boundaries    {self.usable}  "
                   f"({self.usable/n:.2f} per frame, need 2.00)")
        if self.accepted:
            out.append(f"survival rate           "
                       f"{100*self.usable/self.accepted:.1f}%")
        if FUSION_AVAILABLE:
            out.append(f"fused detections        {self.detections}  "
                       f"({self.detections/n:.2f} per frame)")

        out.append("")
        out.append(f"[MODES] {self.frames} frames:")
        for mode in ("two_boundary", "left_only", "right_only",
                     "single_uncalibrated", "none"):
            c = self.modes.get(mode, 0)
            flag = "   <-- never" if c == 0 else ""
            out.append(f" {mode:<22}{c:6}  ({100*c/n:5.1f}%){flag}")

        steering = sum(self.modes.get(m, 0) for m in STEERING_MODES)
        out.append("")
        out.append(f"[AVAILABILITY] {steering}/{self.frames} frames produced a "
                   f"steering signal ({100*steering/n:.1f}%)")
        out.append(f"[BLIND] longest run without one: {self.blind_run} frames")

        if self.gates:
            out.append("")
            out.append("[LOSSES] why a candidate or a pair was not used:")
            for gate, c in sorted(self.gates.items(), key=lambda kv: -kv[1]):
                out.append(f" {gate:<22}{c:6}")

        if self.offsets:
            s = sorted(self.offsets)
            out.append("")
            out.append(f"[OFFSETS] min {s[0]:+.3f}  med {s[len(s)//2]:+.3f}  "
                       f"max {s[-1]:+.3f}")

        if self.stage_ms:
            out.append("")
            out.append("[TIMING] per stage, ms:")
            total_med = 0.0
            for name, vals in self.stage_ms.items():
                v = sorted(vals)
                med, p95 = v[len(v)//2], v[min(int(0.95*len(v)), len(v)-1)]
                total_med += med
                out.append(f" {name:<22} med {med:6.1f}  p95 {p95:6.1f}")
            out.append(f" {'TOTAL (median)':<22}     {total_med:6.1f}  "
                       f"-> {1000/max(total_med, 0.001):5.1f} FPS")
        return out

# =============================================================================
# Runner
# =============================================================================
def run(source, config, out_dir, display=True, scale=1, stride=1,
        limit=None, fps=DEFAULT_FPS):
    """
    Purpose:
        Pull frames from source, run the chain, draw, record, and optionally
        show. Returns RunStats

    Notes:
        A dropped read is counted and skipped rather than ending the run.
        CameraSource returns None for a transient failure and raises
        CaptureError only when the pipeline is actually dead
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = RunStats()
    writer = dv.DebugVideoWriter(os.path.join(out_dir, "run.avi"),
                                 fps=fps, stride=stride)
    stage_f = open(os.path.join(out_dir, "stages.csv"), "w", newline="")
    stage_csv = csv.writer(stage_f)
    stage_csv.writerow(["frame_id", "timestamp_ms", "preprocess_ms", "roi_ms",
                        "geometry_ms", "fusion_ms", "lane_offset_ms",
                        "total_ms", "accepted", "usable", "mode", "offset"])

    window = "pipeline"
    can_display = display
    if can_display:
        try:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        except Exception as exc:
            print(f"  display unavailable ({exc}); continuing headless")
            can_display = False

    paused = False
    stills = 0
    try:
        while True:
            if limit is not None and stats.frames >= limit:
                break

            try:
                item = source.read()
            except CaptureError as exc:
                print(f"  capture failed: {exc}")
                break
            if item is None:
                break
            frame, frame_id, timestamp_ms = item
            if frame is None:
                stats.drops += 1
                continue

            t0 = time.perf_counter()
            chain = run_chain(frame, frame_id, timestamp_ms, config)
            t_chain = (time.perf_counter() - t0) * 1000.0

            n_detections, t_fusion = 0, 0.0
            if FUSION_AVAILABLE:
                tf = time.perf_counter()
                try:
                    fusion, _fdbg = fuse_detections(chain.geometry, [], chain.roi)
                    n_detections = len(fusion.detections)
                except Exception as exc:
                    print(f"  fusion failed on frame {frame_id}: {exc}")
                t_fusion = (time.perf_counter() - tf) * 1000.0

            gates = candidate_gates(chain.geometry, config.lane_offset)
            result, dbg = chain.offset, chain.offset_debug

            timings = {"chain": t_chain, "fusion": t_fusion}
            stats.update(result, dbg, gates, n_detections, timings)

            stage_csv.writerow([
                frame_id, timestamp_ms, "", "", "", round(t_fusion, 2), "",
                round(t_chain + t_fusion, 2),
                dbg.get("raw_count", 0), result.boundary_count,
                result.mode, result.offset,
            ])

            img = writer.write(frame, result, dbg, chain.roi.lane_rect,
                               gates, scale, (frame.shape[1], frame.shape[0]))

            if can_display and img is not None:
                cv2.imshow(window, img)
                while True:
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key == ord("q"):
                        return stats
                    if key == ord(" "):
                        paused = not paused
                        if not paused:
                            break
                    elif key == ord("s"):
                        p = os.path.join(out_dir, f"still_{stills:03d}.png")
                        cv2.imwrite(p, img)
                        print(f"  saved {p}")
                        stills += 1
                    elif not paused:
                        break
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        writer.close()
        stage_f.close()
        source.close()
        if can_display:
            cv2.destroyAllWindows()
    return stats

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    def flag(name, default=None):
        if name not in sys.argv:
            return default
        i = sys.argv.index(name)
        if len(sys.argv) > i + 1 and not sys.argv[i + 1].startswith("--"):
            return sys.argv[i + 1]
        return True

    if len(sys.argv) == 1 or "--help" in sys.argv:
        print(__doc__)
        print("  --width N --height N --fps N   capture size and rate")
        print("  --scale N                      overlay magnification")
        print("  --stride N                     record every Nth frame")
        print("  --limit N                      stop after N frames")
        print("  --no-display                   force headless")
        sys.exit(0)

    fps = int(flag("--fps", DEFAULT_FPS))
    width = int(flag("--width", DEFAULT_SIZE[0]))
    height = int(flag("--height", DEFAULT_SIZE[1]))

    if flag("--camera"):
        source = CameraFrameSource(width, height, fps)
    elif flag("--video"):
        source = VideoFrameSource(flag("--video"), fps)
    elif flag("--frames"):
        source = DirectoryFrameSource(flag("--frames"), fps)
    else:
        print("Pick a source: --camera, --video PATH, or --frames DIR")
        sys.exit(2)

    out_dir = flag("--out", os.path.join(
        "vision_stack", "runs", time.strftime("%Y%m%d_%H%M%S")))

    print(f"source   {source.label}")
    print(f"output   {out_dir}")
    if not FUSION_AVAILABLE:
        print(f"fusion   disabled ({type(FUSION_ERROR).__name__}: {FUSION_ERROR})")
        print("         lane offset still runs; wire color_branch to enable it")

    stats = run(
        source,
        config = MEASURED,
        out_dir = out_dir,
        display = not flag("--no-display", False),
        scale = int(flag("--scale", 1)),
        stride = int(flag("--stride", 1)),
        limit = int(flag("--limit")) if flag("--limit") else None,
        fps = fps,
    )

    lines = stats.report()
    print("\n" + "\n".join(lines))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"source: {source.label}\n")
        f.write(f"fusion: {'on' if FUSION_AVAILABLE else 'off'}\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/run.avi, run.csv, stages.csv, summary.txt")