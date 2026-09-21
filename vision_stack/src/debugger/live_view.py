"""
live_view.py

Live View: run a frame source through the pipeline and show / record what it
decided, frame by frame.

Purpose:
    Pulls frames from a source, hands each one to a process callable, draws
    the decision with debug_video, records it, and optionally shows it:

        FrameSource -> process(frame) -> overlay -> DebugVideoWriter
                                                 -> Display (optional)

    Overlays come from debug_video.annotate(), so the window and the recorded
    video show the same picture. Output is written whether or not a display
    is attached, which is the point: a run on the robot can be reviewed later.

    Every picture is a view. The lane view always runs; further views
    (--views stop,traffic) run beside it. Each gets its own window entry,
    video, CSV and summary section, so one target can be tuned at a time.

Layers (top depends on bottom, never the reverse):
    cli()             argument parsing, source construction, summary file
    run()             the loop: read, process, then every view in turn
    Display           window, keys, headless fallback
    StageLog          stages.csv
    RunStats          run-wide counts and stage timings; each view reports its
                      own statistics
    FrameSource       camera / video file / image directory

This module holds nothing specific to one view. The views are debug_lane,
debug_stop and debug_traffic; the interface they share is in debug_video.

Dependencies:
    Imports debug_video, the three view modules and capture. It does NOT
    import the linker: the stage order is injected as `process`, so
    phase2_linker can import this module without a cycle. Any callable that
    returns an object with .geometry, .roi.lane_rect, .offset and
    .offset_debug will do, which is what the lane view reads.

Sources (cli):
    --camera            live capture through capture.CameraSource
    --video PATH        a recorded file
    --frames DIR        an image sequence, sorted by filename

Output (--out DIR, default runs/<timestamp>):
    run.avi / run.csv   the lane view: annotated video and decision log
    run_<view>.avi/.csv one pair per extra view, e.g. run_stop.avi
    stages.csv          per-frame timings and counts
    summary.txt         run counts, one section per view (the lane section has
                        the mode histogram, availability and blind runs), then
                        stage timings

Views (--views a,b):
    lane                always on (debug_lane)
    stop                stop-sign detector on the sign ROI (debug_stop)
    traffic             color branch on the traffic ROI: the three HSV masks
                        and every blob (debug_traffic). Needs --hsv PATH; the
                        color branch is off without calibrated ranges
    A view is any object with the interface in debug_video. Register it in
    VIEWS to make it selectable.
    Views that need the per-contour trace get it from run_live_view, which
    turns trace on.

Display:
    On by default. Falls back to headless automatically if the window cannot
    open, so the same command works over ssh and on the bench.
    q quits, space pauses, s saves a still of the current view,
    v or 1-9 switches view.

Fusion:
    Fusion and Phase 2 packaging run inside the chain, not here. This module
    reads chain.fusion, chain.phase2 and chain.timings_ms when the process
    callable provides them, and counts fused detections and per-stage timings
    from them. A process callable without them still works: fusion columns
    stay empty and the chain is timed as one unit.
"""
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

import cv2

from src.capture.camera import CameraSource, CaptureError
import src.debugger.debug_video as dv
import src.debugger.debug_lane as debug_lane
import src.debugger.debug_stop as debug_stop
import src.debugger.debug_traffic as debug_traffic

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_SIZE = (480, 360)       # (w, h)
DEFAULT_FPS = 12
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# Extra views selectable with --views. Each is built with conf_threshold=. The
# lane view is not here: it always runs, and needs the lane config
VIEWS = {"stop": debug_stop.StopView, "traffic": debug_traffic.TrafficView}

# =============================================================================
# Frame Sources
# =============================================================================
class FrameSource:
    """
    Common interface over the three sources.

    read() yields (frame_bgr, frame_id, timestamp_ms), None when exhausted,
    or (None, None, None) for a transient drop the caller should skip.
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
    """fps=None uses the file's own rate."""
    def __init__(self, path, fps=None):
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
    def __init__(self, path, fps=DEFAULT_FPS):
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
# Run Statistics
# =============================================================================
@dataclass
class RunStats:
    """
    What the run as a whole reports: frame counts, fused detections and stage
    timings. Everything about a target is reported by its view, and lands in
    `sections`.
    """
    frames: int = 0
    drops: int = 0
    detections: int = 0
    fusion_seen: bool = False       # True once any chain result carried fusion

    def __post_init__(self):
        self.stage_ms = {}
        self.sections = {}          # view name -> report lines

    def update(self, n_detections, timings):
        """n_detections is None when the chain result carried no fusion."""
        self.frames += 1
        if n_detections is not None:
            self.fusion_seen = True
            self.detections += n_detections
        for name, ms in timings.items():
            self.stage_ms.setdefault(name, []).append(ms)

    def report(self):
        """Render the summary as a list of lines: counts, views, timing."""
        out = []
        n = max(self.frames, 1)
        out.append(f"frames processed        {self.frames}")
        out.append(f"dropped reads           {self.drops}")
        if self.fusion_seen:
            out.append(f"fused detections        {self.detections}  "
                       f"({self.detections/n:.2f} per frame)")

        for lines in self.sections.values():
            out.append("")
            out.extend(lines)

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
# Outputs
# =============================================================================
class StageLog:
    """
    stages.csv: one row per processed frame.

    Per-stage columns come from the chain's timings_ms and stay blank for a
    process callable that does not provide them; total_ms is always the
    measured wall time of the whole process call.

    accepted, usable, mode and offset are the lane signal, the pipeline's
    headline output, kept here so one file lines it up with the timings. The
    lane view's own run.csv has the full decision log.
    """
    FIELDS = ("frame_id", "timestamp_ms", "preprocess_ms", "roi_ms",
              "geometry_ms", "fusion_ms", "lane_offset_ms", "total_ms",
              "accepted", "usable", "mode", "offset", "detections", "color_ms")

    def __init__(self, path):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.FIELDS)

    def write(self, frame_id, timestamp_ms, timings, t_total, chain,
              n_detections):
        result, dbg = chain.offset, chain.offset_debug
        ms = lambda name: round(timings[name], 2) if name in timings else ""
        self._w.writerow([
            frame_id, timestamp_ms, ms("preprocess"), ms("roi"),
            ms("geometry"), ms("fusion"), ms("lane_offset"),
            round(t_total, 2),
            dbg.get("raw_count", 0), result.boundary_count,
            result.mode, result.offset,
            "" if n_detections is None else n_detections,
            ms("color"),
        ])

    def close(self):
        self._f.close()

class Display:
    """
    Preview window with pause, still capture and view switching.

    names: the views, in order; the first is shown to start with.
    Falls back to headless if the window cannot open (no display, or a
    headless OpenCV build). show() returns False when the user asks to quit.
    """
    WINDOW = "pipeline"

    def __init__(self, enabled, still_dir, names=("lane",)):
        self.still_dir = still_dir
        self.names = list(names)
        self.current = 0
        self.enabled = enabled
        self.paused = False
        self._stills = 0
        if enabled and self._no_display_server():
            print("  no display server (DISPLAY unset); continuing headless")
            self.enabled = False
        if self.enabled:
            try:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
            except Exception as exc:
                print(f"  display unavailable ({exc}); continuing headless")
                self.enabled = False

    @staticmethod
    def _no_display_server():
        """
        Qt builds of OpenCV abort the whole process, not raise, when there is
        no X/Wayland server, so the try/except below never sees it over ssh.
        """
        return (sys.platform.startswith("linux")
                and not os.environ.get("DISPLAY")
                and not os.environ.get("WAYLAND_DISPLAY"))

    def _draw(self, shots):
        img = shots.get(self.names[self.current])
        if img is not None:
            cv2.imshow(self.WINDOW, img)
        return img

    def show(self, shots):
        """shots: {view name: image} for the views rendered this frame."""
        if not self.enabled:
            return True
        img = self._draw(shots)
        if img is None:
            return True
        while True:
            key = cv2.waitKey(0 if self.paused else 1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord(" "):
                self.paused = not self.paused
                if not self.paused:
                    return True
            elif key == ord("s"):
                name = self.names[self.current]
                p = os.path.join(self.still_dir, f"still_{name}_{self._stills:03d}.png")
                cv2.imwrite(p, img)
                print(f"  saved {p}")
                self._stills += 1
            elif key == ord("v") or ord("1") <= key <= ord("9"):
                if key == ord("v"):
                    self.current = (self.current + 1) % len(self.names)
                elif key - ord("1") < len(self.names):
                    self.current = key - ord("1")
                shown = self._draw(shots)
                if shown is not None:
                    img = shown
            elif not self.paused:
                return True

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()

# =============================================================================
# Runner
# =============================================================================
def _video_name(view):
    """The lane view keeps the historical run.avi / run.csv names."""
    return "run.avi" if view.name == "lane" else f"run_{view.name}.avi"

def run(source, process, lane_config, out_dir, display=True, scale=1,
        stride=1, limit=None, fps=None, views=()):
    """
    Purpose:
        Pull frames from source, run them through process, then run every view
        on the result: draw, record, and optionally show. Returns RunStats

    Inputs:
        source: FrameSource
        process: callable (frame_bgr, frame_id, timestamp_ms) -> chain result
                 exposing .geometry, .roi.lane_rect, .offset, .offset_debug
        lane_config: the lane_offset config, used by the lane view to classify
                 candidates; must be the one process() runs with
        out_dir: created if missing
        fps: recorded video rate; None uses source.fps
        views: extra views to run beside the lane view (see VIEWS). Each is
               observed on every frame, but rendered and recorded only on the
               frames the stride keeps

    Notes:
        A dropped read is counted and skipped rather than ending the run.
        CameraSource returns None for a transient failure and raises
        CaptureError only when the pipeline is actually dead
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = RunStats()
    views = [debug_lane.LaneView(lane_config), *views]
    stage_log = StageLog(os.path.join(out_dir, "stages.csv"))
    writers = [
        dv.ViewWriter(os.path.join(out_dir, _video_name(v)), v.CSV_FIELDS,
                      fps=fps or source.fps, stride=stride)
        for v in views
    ]
    window = Display(display, out_dir, [v.name for v in views])

    try:
        while limit is None or stats.frames < limit:
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
            chain = process(frame, frame_id, timestamp_ms)
            t_total = (time.perf_counter() - t0) * 1000.0

            fusion = getattr(chain, "fusion", None)
            n_detections = None if fusion is None else len(fusion.detections)
            timings = dict(getattr(chain, "timings_ms", None)
                           or {"chain": t_total})

            stats.update(n_detections, timings)
            stage_log.write(frame_id, timestamp_ms, timings, t_total,
                            chain, n_detections)

            shots = {}
            for v, w in zip(views, writers):
                data = v.extract(chain, frame)
                v.observe(data)
                if not w.take():
                    continue
                shot = v.render(data, scale)
                w.push(shot, v.row(data))
                shots[v.name] = shot

            if shots and not window.show(shots):
                break
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        for w in writers:
            w.close()
        stage_log.close()
        source.close()
        window.close()
    for v in views:
        stats.sections[v.name] = v.report()
    return stats

# =============================================================================
# Command line
# =============================================================================
def write_summary(path, source, stats):
    with open(path, "w") as f:
        f.write(f"source: {source.label}\n")
        f.write(f"fusion: {'on' if stats.fusion_seen else 'off'}\n\n")
        f.write("\n".join(stats.report()) + "\n")

def cli(runner, argv=None):
    """
    Purpose:
        Parse arguments, open the chosen source, call runner, write the
        summary. Returns a process exit code

    Inputs:
        runner: callable (source, out_dir=, display=, scale=, stride=,
                limit=, views=, hsv_path=) -> RunStats. phase2_linker.run_live_view
                fits, which is how the linker supplies the stage order
        argv: argument list, default sys.argv[1:]
    """
    ap = argparse.ArgumentParser(
        prog="live_view", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true")
    src.add_argument("--video", metavar="PATH")
    src.add_argument("--frames", metavar="DIR")
    ap.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    ap.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    ap.add_argument("--fps", type=int, default=None,
                    help="capture/replay rate (video files default to their own)")
    ap.add_argument("--scale", type=int, default=1, help="overlay magnification")
    ap.add_argument("--stride", type=int, default=1, help="record every Nth frame")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--views", default="", metavar="A,B",
                    help="extra views beside the lane view: "
                         + ", ".join(sorted(VIEWS)))
    ap.add_argument("--stop-threshold", type=float, default=None, metavar="C",
                    help="stop-sign confidence needed downstream; candidates "
                         "below it show amber in the stop view")
    ap.add_argument("--traffic-threshold", type=float, default=None, metavar="C",
                    help="traffic-light confidence needed downstream; blobs "
                         "below it show amber in the traffic view")
    ap.add_argument("--hsv", default=None, metavar="PATH",
                    help="calibrated HSV ranges JSON; switches the color "
                         "branch on (off without it)")
    ap.add_argument("--no-display", action="store_true", help="force headless")
    ap.add_argument("--out", default=None, metavar="DIR")

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        ap.print_help()
        return 0
    args = ap.parse_args(argv)

    names = [n for n in args.views.split(",") if n and n != "lane"]
    unknown = [n for n in names if n not in VIEWS]
    if unknown:
        print(f"unknown view {', '.join(unknown)}; choose from "
              f"{', '.join(sorted(VIEWS))}")
        return 2
    thresholds = {"stop": args.stop_threshold, "traffic": args.traffic_threshold}
    views = [VIEWS[n](conf_threshold=thresholds.get(n)) for n in names]
    if "traffic" in names and not args.hsv:
        print("note: the traffic view needs --hsv PATH; without calibrated "
              "ranges the color branch is off and the view stays empty")

    try:
        if args.camera:
            source = CameraFrameSource(args.width, args.height,
                                       args.fps or DEFAULT_FPS)
        elif args.video:
            source = VideoFrameSource(args.video, args.fps)
        else:
            source = DirectoryFrameSource(args.frames, args.fps or DEFAULT_FPS)
    except (CaptureError, OSError) as exc:
        print(f"source error: {exc}")
        return 2

    out_dir = args.out or os.path.join(
        "vision_stack", "runs", time.strftime("%Y%m%d_%H%M%S"))

    print(f"source   {source.label}")
    print(f"output   {out_dir}")

    stats = runner(source, out_dir=out_dir, display=not args.no_display,
                   scale=args.scale, stride=args.stride, limit=args.limit,
                   views=views, hsv_path=args.hsv)

    print("\n" + "\n".join(stats.report()))
    write_summary(os.path.join(out_dir, "summary.txt"), source, stats)
    print(f"\nwrote {out_dir}: " + ", ".join(sorted(os.listdir(out_dir))))
    return 0