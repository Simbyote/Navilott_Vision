"""Phase 1-3 linker: the debug pipeline for capture, perception and estimation, reported as text and video.

Purpose:
    The estimation test harness. Phase 3's lane filter, dropout hold, heading
    tracker and votes only mean something across frames, so this runs the
    whole chain and shows each packet next to the Phase 2 input it came from;
    a bad packet can then be traced to bad input or bad filtering. It is the
    instrumented twin of pipeline.Pipeline: Phases 1-2 come from
    phase2_linker.run_chain() and Phase 3 from
    estimation_debug.TracedPhase3Processor, which record every decision and
    time every stage, and the tests hold both to Pipeline's packets. It runs
    the chain itself, live or from a replay, rather than reading
    phase2_linker's recordings. The Phase 3 video (debug_phase3) is recorded
    and shown the way phase2_linker's views are.

Main package:
    Phase3Result: one frame's ChainResult, the EstimationPacket handed to
    Navigation, Phase 3's debug (the per-stage records and timings), and
    per-phase timings.

Flow:
    FrameSource -> run_chain() -> TracedPhase3Processor.process() -> EstimationPacket
                   (Phases 1-2)   (Phase 3)
    Each result goes to p3.csv, the event tracker, the run statistics and,
    unless turned off, Phase3View (p3_debug.avi / .csv and the window);
    summary.txt is written when the source ends or the run is interrupted.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from src.capture.camera import CaptureError
from src.perception.color_branch import load_hsv_ranges
from src.params import FPS, FRAME_H, FRAME_W, RUNS_DIR, STOP_SIGN, TRAFFIC_LIGHT
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.phase2_linker import ChainResult, run_chain
import src.debugger.debug_video as dv
from src.debugger.debug_phase3 import Phase3View
from src.debugger.live_view import (
    CameraFrameSource, VideoFrameSource, DirectoryFrameSource, Display, stage_timing_report,
)
from src.estimation.estimation import (
    LANE_VISION, LANE_HOLD, LANE_STALE,
    EstimationPacket, Phase3Config, Phase3Processor, SensorSample, with_lane_roi_width,
)
from src.debugger.estimation_debug import TracedPhase3Processor
from src.peripherals.sensing import Sensors           # re-exported: older imports read it from here

# --help text. Kept apart from the module docstring, which documents the code.
_CLI_HELP = """\
Run capture -> perception -> estimation and report what estimation decided,
frame by frame, next to the Phase 2 input it came from, as text and as video.

Sources:
    --camera        live capture through capture.CameraSource
    --video PATH    a recorded clip, e.g. a live_view run.avi
    --frames DIR    an image sequence, sorted by filename
    Replays are deterministic, so estimation config changes can be compared
    on the same footage. Replay timestamps come from the nominal frame rate.

Sensors:
    --imu reads the MPU-6050 and feeds it to Phase 3 each frame.
    --encoders reads the wheel encoders (needs sudo pigpiod) and passes each
    wheel's counts per second through to the packet. Both are read together
    at 100 Hz by src/peripherals/sensing.py's SensorHub, as in production, and grouped
    per frame; yaw reads + = turning right. Without them, Phase 3
    runs with no sensors: heading holds at 0 and the pass-through fields read
    0.0. Each driver is imported only with its flag, so replays run off the Pi.

Output (--out DIR, default <root>/runs/p3_<timestamp>):
    console      one status line every --print-every frames, plus an event
                 line on every lane_status, drive_state, stop_sign or
                 stop_line change
    p3.csv       every frame: timings, the Phase 2 lane and stop-line input,
                 the packet, and Phase 3's debug log
    p3_debug.avi the Phase 3 video, every frame (--no-video turns it off):
                 the Phase 2 lane overlay with traffic lights and stop signs
                 green if they passed Phase 3's gate, amber if not; the lane
                 filter's raw vs filtered offset, status, hold counter and
                 rejection reason; the three vote buffers; and a 5 s timeline
    p3_debug.csv every frame's Phase 3 decisions: lane reason, EMA before and
                 after, vote buffers, stop line seen or held
    summary.txt  timing percentiles, lane status and mode histograms, longest
                 hold and stale runs, offset statistics while on vision, the
                 [PHASE 3] decision counts, and [TIMING] per stage (Phase 2
                 stages, Phase 3 stages, render; render is not in the total
                 or the P2+P3 budget)

Display:
    On by default, as in phase2_linker; falls back to headless if the window
    can't open. q quits, space pauses, s saves a still. --no-display forces
    headless.

Examples (from the repo root):
    python3 -m src.phase3_linker --video run.avi
    python3 -m src.phase3_linker --camera --imu --fps 20
    python3 -m src.phase3_linker --camera --imu --encoders
    python3 -m src.phase3_linker --camera --limit 200 --print-every 1
    python3 -m src.phase3_linker --camera --no-display --no-video   # text and timing only

Sign convention:
    lane_offset    + = robot RIGHT of lane center, so steer left
    heading_error  + = robot has turned RIGHT since the last vision frame
                   (per estimation; check against the IMU's own axis note)
"""


@dataclass(frozen=True)
class Phase3Result:
    """One frame through all three phases."""
    chain: ChainResult              # every Phase 1-2 stage output
    packet: EstimationPacket        # handed to Navigation
    p3_debug: dict                  # the processor's debug; TracedPhase3Processor adds per-stage records and timings_ms
    timings_ms: dict = field(default_factory=dict)      # capture, phase2, phase3, total; render added by run() when it draws

def run_phase3_chain(
        frame_bgr: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        processor: Phase3Processor,
        sensors: SensorSample | None = None,
        config: PipelineConfig = MEASURED,
        capture_ms: float = 0.0,
    ) -> Phase3Result:
    """
    Run one frame through Phases 2 and 3.

    Inputs:
        frame_bgr, frame_id, timestamp_ms: As the frame source delivered them.
        processor: This run's Phase3Processor, normally a
            TracedPhase3Processor (run() builds one). Stateful: pass the
            same one every frame, in order.
        sensors: Readings for this frame window; None runs without sensors.
        config: Phase 2 tuning.
        capture_ms: Time spent in source.read(). Phase 1 happens there,
            before this call, so its time comes in rather than being measured.

    Outputs:
        Phase3Result.
    """
    t0 = time.perf_counter()
    chain = run_chain(frame_bgr, frame_id, timestamp_ms, config)
    t1 = time.perf_counter()
    packet, p3_debug = processor.process(chain.phase2, sensors)
    t2 = time.perf_counter()

    p2_ms = (t1 - t0) * 1000.0
    p3_ms = (t2 - t1) * 1000.0
    timings = {
        "capture": capture_ms,
        "phase2": p2_ms,
        "phase3": p3_ms,
        "total": capture_ms + p2_ms + p3_ms,
    }
    return Phase3Result(chain, packet, p3_debug, timings)


def _fmt(v, spec: str, none: str = "--") -> str:
    """Format an optional number."""
    return none if v is None else format(v, spec)

def status_line(res: Phase3Result) -> str:
    """
    One readable line: timing | Phase 2 lane input | Phase 3 output.

    Example:
        f0412 t=20.61s P1=6.1 P2=31.8 P3=0.9ms | two_boundary L=150 R=290 n=2
        c=.82 raw=+0.045 | off=+0.031 vision hd=+0.0 | go stop=F
    """
    pk, off, t = res.packet, res.chain.offset, res.timings_ms
    cm = f" ({pk.lane_offset_cm:+.1f}cm)" if pk.lane_offset_cm is not None else ""
    return (
        f"f{pk.frame_id:04d} t={pk.timestamp_ms / 1000.0:6.2f}s "
        f"P1={t['capture']:4.1f} P2={t['phase2']:4.1f} P3={t['phase3']:3.1f}ms | "
        f"{off.mode:<19} L={_fmt(off.left_x, '3.0f')} R={_fmt(off.right_x, '3.0f')} "
        f"n={off.boundary_count} c={off.confidence:.2f} raw={off.offset:+.3f} | "
        f"off={pk.lane_offset:+.3f}{cm} {pk.lane_status:<6} hd={pk.heading_error:+.1f} | "
        f"{pk.drive_state} stop={'T' if pk.stop_sign_detected else 'F'} "
        f"line={_fmt(pk.stop_line_distance_px, '.0f')}"
        + (f"/{pk.stop_line_distance_cm:.1f}cm" if pk.stop_line_distance_cm is not None else "")
    )


class EventTracker:
    """
    Reports changes in the packet fields Navigation acts on: lane_status,
    drive_state, stop_sign_detected and stop_line_detected. counts holds
    transitions per field.
    """
    def __init__(self) -> None:
        self._prev = None
        self.counts: Counter = Counter()

    def update(self, res: Phase3Result) -> list[str]:
        """
        One line per field that changed since the previous frame; none on the first frame.

        A lane_status change also names the Phase 2 lane mode behind it, since
        that's usually the question when the filter drops to hold.
        """
        pk = res.packet
        now = {
            "lane": pk.lane_status,
            "drive": pk.drive_state,
            "stop_sign": "T" if pk.stop_sign_detected else "F",
            "stop_line": "T" if pk.stop_line_detected else "F",
        }
        lines = []
        if self._prev is not None:
            for key, value in now.items():
                if value != self._prev[key]:
                    self.counts[key] += 1
                    why = f" (p2 mode={res.chain.offset.mode})" if key == "lane" else ""
                    lines.append(f">>> f{pk.frame_id:04d} {key}: "
                                 f"{self._prev[key]} -> {value}{why}")
        self._prev = now
        return lines


CSV_COLUMNS = (
    "frame_id", "timestamp_ms", "dt_s",
    "capture_ms", "phase2_ms", "phase3_ms", "total_ms",
    "p2_mode", "p2_offset", "p2_left_x", "p2_right_x", "p2_lane_width_px",
    "p2_conf", "p2_boundary_count", "p2_detections", "p2_traffic", "p2_stop",
    "p2_stop_line_px",
    "lane_offset", "lane_offset_cm", "lane_status", "heading_error",
    "drive_state", "stop_sign_detected", "stop_line_detected", "stop_line_distance_px",
    "yaw_rate", "lateral_accel",
    "p3_log",
    # Appended, so no earlier column moves
    "p2_stop_line_cm", "stop_line_distance_cm",
    "left_wheel_cps", "right_wheel_cps",
)

class CsvLog:
    """p3.csv: every field of every frame, one row per Phase3Result, in CSV_COLUMNS order."""
    def __init__(self, path: str) -> None:
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(CSV_COLUMNS)

    def write(self, res: Phase3Result) -> None:
        pk, off, t = res.packet, res.chain.offset, res.timings_ms
        dets = res.chain.phase2.detections
        self._w.writerow((
            pk.frame_id, pk.timestamp_ms, res.p3_debug.get("dt"),
            f"{t['capture']:.2f}", f"{t['phase2']:.2f}",
            f"{t['phase3']:.3f}", f"{t['total']:.2f}",
            off.mode, f"{off.offset:.4f}", off.left_x, off.right_x,
            off.lane_width_px, f"{off.confidence:.3f}", off.boundary_count,
            len(dets),
            sum(d.type == TRAFFIC_LIGHT for d in dets),
            sum(d.type == STOP_SIGN for d in dets),
            res.chain.stop_line.distance_px,        # blank when Phase 2 saw no line
            pk.lane_offset, pk.lane_offset_cm, pk.lane_status,
            pk.heading_error, pk.drive_state, int(pk.stop_sign_detected),
            int(pk.stop_line_detected), pk.stop_line_distance_px,
            pk.yaw_rate, pk.lateral_accel,
            " | ".join(res.p3_debug.get("log", [])),
            res.chain.stop_line.distance_cm, pk.stop_line_distance_cm,     # blank without a ground homography
            pk.left_wheel_cps, pk.right_wheel_cps,
        ))

    def close(self) -> None:
        self._f.close()


class Phase3Stats:
    """
    Run-wide statistics for the summary.

    Offset statistics cover only frames on vision, so a held value repeated
    for seven frames doesn't shrink the spread. With the robot parked
    centered, the offset std is the measurement noise floor.

    budget_ms: Per-frame processing budget, normally one frame period.
    """
    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms
        self.frames = 0
        self.timings = {k: [] for k in ("capture", "phase2", "phase3", "total")}
        self.stage_ms = {}                  # stage -> per-frame ms: Phase 2 stages, Phase 3 stages, render
        self.status = Counter()
        self.modes = Counter()
        self.vision_offsets = []
        self.over_budget = 0
        self.longest = {LANE_HOLD: 0, LANE_STALE: 0}    # longest unbroken run, frames
        self._run_status, self._run_len = None, 0
        self._t_start = time.perf_counter()

    def update(self, res: Phase3Result) -> None:
        """Fold one frame into the counts, timings and run lengths."""
        self.frames += 1
        for k, v in res.timings_ms.items():
            self.timings.setdefault(k, []).append(v)
        stages = {**res.chain.timings_ms, **res.p3_debug.get("timings_ms", {})}
        if "render" in res.timings_ms:
            stages["render"] = res.timings_ms["render"]
        for k, v in stages.items():
            self.stage_ms.setdefault(k, []).append(v)
        # Capture time on a live camera includes waiting for the next frame,
        # so the processing budget is judged on Phases 2 and 3 only
        if res.timings_ms["phase2"] + res.timings_ms["phase3"] > self.budget_ms:
            self.over_budget += 1

        status = res.packet.lane_status
        self.status[status] += 1
        self.modes[res.chain.offset.mode] += 1
        if status == LANE_VISION:
            self.vision_offsets.append(res.packet.lane_offset)

        self._run_len = self._run_len + 1 if status == self._run_status else 1
        self._run_status = status
        if status in self.longest:
            self.longest[status] = max(self.longest[status], self._run_len)

    def report(self) -> list[str]:
        """The summary as lines: frames and budget, timing percentiles, lane status, lane modes, offsets."""
        n = max(self.frames, 1)
        wall = time.perf_counter() - self._t_start
        lines = [
            f"frames           {self.frames}",
            f"wall time        {wall:.1f} s  ({self.frames / wall:.1f} FPS effective)"
            if wall > 0 else "wall time        0 s",
            f"over budget      {self.over_budget} of {self.frames} "
            f"(P2+P3 > {self.budget_ms:.1f} ms)",
            "",
            "timing (ms)      mean     p50     p95     max",
        ]
        for k, v in self.timings.items():
            if v:
                a = np.asarray(v)
                lines.append(f"  {k:<14}{a.mean():6.1f}  {np.percentile(a, 50):6.1f}"
                             f"  {np.percentile(a, 95):6.1f}  {a.max():6.1f}")

        lines += ["", "lane status"]
        for s in (LANE_VISION, LANE_HOLD, LANE_STALE):
            lines.append(f"  {s:<14}{self.status[s]:6d}  {100.0 * self.status[s] / n:5.1f}%")
        lines.append(f"  longest hold   {self.longest[LANE_HOLD]} frames")
        lines.append(f"  longest stale  {self.longest[LANE_STALE]} frames")

        lines += ["", "phase 2 lane mode"]
        for mode, count in self.modes.most_common():
            lines.append(f"  {mode:<20}{count:6d}  {100.0 * count / n:5.1f}%")

        lines += ["", "lane offset while on vision (+ = right of center)"]
        if self.vision_offsets:
            a = np.asarray(self.vision_offsets)
            lines.append(f"  mean {a.mean():+.4f}  std {a.std():.4f}  "
                         f"min {a.min():+.4f}  max {a.max():+.4f}")
        else:
            lines.append("  no frames on vision")

        if self.stage_ms:
            lines += [""] + stage_timing_report(self.stage_ms, exclude=("render",))
        return lines


def make_processor(frame, fid: int, ts: int, config: PipelineConfig,
                   p3_config: Phase3Config) -> TracedPhase3Processor:
    """
    The traced Phase 3 processor for a run, built on its first frame.

    With cm_per_px set, lane_offset_cm needs the lane ROI width in px, taken
    from config's lane ROI at the first frame's size, as the pipeline does
    (with_lane_roi_width); fid and ts aren't used.
    """
    return TracedPhase3Processor(with_lane_roi_width(p3_config, config.roi, frame.shape[:2]))


def run(
        source,
        config: PipelineConfig = MEASURED,
        p3_config: Phase3Config = MEASURED_ESTIMATION,
        use_imu: bool = False,
        out_dir: str = str(RUNS_DIR / "p3"),
        print_every: int = 20,
        verbose: bool = False,
        limit: int | None = None,
        video: bool = True,
        display: bool = False,
        scale: int = 1,
        use_encoders: bool = False,
    ) -> Phase3Stats:
    """
    Run every frame from source through all three phases.

    Inputs:
        source: A live_view FrameSource: camera, video file or image directory.
        config: Phase 2 tuning.
        p3_config: Phase 3 tuning; defaults to MEASURED_ESTIMATION. When
            cm_per_px is set and lane_roi_width_px isn't, the width is taken
            from the first frame's lane ROI.
        use_imu: Start the IMU and feed it to Phase 3.
        use_encoders: Start the wheel encoders (peripherals.drive, through
            pigpio) and feed their counts per second to Phase 3.
        print_every: Status line every N frames; 0 prints events only.
        verbose: Also print Phase 3's per-frame debug log.
        limit: Stop after this many frames; None runs until the source ends.
        video: Record p3_debug.avi and p3_debug.csv, every frame.
        display: Show the same picture in a window (q quits, space pauses,
            s saves a still); falls back to headless without a display.
        scale: Magnification of the video and window.

    Outputs:
        Phase3Stats.

    Side effects:
        Writes p3.csv and summary.txt into out_dir (created if missing), and
        the video when on; may open a window; prints to the console, and
        closes the source. Ctrl-C ends the run early; the CSV, a playable
        video and the summary are still written.
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = Phase3Stats(budget_ms=1000.0 / max(source.fps, 1))
    events = EventTracker()
    log = CsvLog(os.path.join(out_dir, "p3.csv"))
    sensors = Sensors(use_imu, use_encoders)
    processor = None
    view = Phase3View(config.lane_offset, source.fps) if (video or display) else None
    writer = (dv.ViewWriter(os.path.join(out_dir, "p3_debug.avi"), Phase3View.CSV_FIELDS,
                            fps=source.fps) if video else None)
    window = Display(display, out_dir, [Phase3View.name])

    try:
        while limit is None or stats.frames < limit:
            t0 = time.perf_counter()
            item = source.read()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            if item is None:
                break                               # source exhausted
            frame, fid, ts = item
            if frame is None:
                continue                            # transient camera drop

            sample = sensors.sample()
            if processor is None:
                processor = make_processor(frame, fid, ts, config, p3_config)

            res = run_phase3_chain(frame, fid, ts, processor, sample, config, capture_ms)
            quit_ = False
            if view is not None:
                data = view.extract(res, frame)
                view.observe(data)
                t0 = time.perf_counter()
                img = view.render(data)
                if writer is not None:
                    writer.push(img, view.row(data))
                res.timings_ms["render"] = view.last_render_ms = (time.perf_counter() - t0) * 1000.0
                quit_ = not window.show({view.name: img})
            stats.update(res)
            log.write(res)

            for line in events.update(res):
                print(line)
            if print_every and stats.frames % print_every == 0:
                print(status_line(res))
            if verbose:
                for entry in res.p3_debug.get("log", []):
                    print(f"    {entry}")
            if quit_:
                break

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        log.close()
        if writer is not None:
            writer.close()              # released here so Ctrl-C still leaves a playable file
        window.close()
        sensors.stop()
        source.close()

    summary = stats.report()
    if view is not None:
        summary += [""] + view.report()
    summary += ["", f"transitions      lane {events.counts['lane']}  "
                    f"drive {events.counts['drive']}  "
                    f"stop_sign {events.counts['stop_sign']}  "
                    f"stop_line {events.counts['stop_line']}"]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"source {source.label}\n\n" + "\n".join(summary) + "\n")
    print("\n" + "\n".join(summary))
    return stats


def cli(argv: list[str] | None = None) -> int:
    """
    Parse arguments, open the source, and run.

    Inputs:
        argv: Argument list; None reads sys.argv[1:]. Empty prints help.

    Outputs:
        Process exit code: 0 on success, 2 if the source can't be opened.
    """
    ap = argparse.ArgumentParser(
        prog="phase3_linker", description=_CLI_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true")
    src.add_argument("--video", metavar="PATH")
    src.add_argument("--frames", metavar="DIR")
    ap.add_argument("--width", type=int, default=FRAME_W)
    ap.add_argument("--height", type=int, default=FRAME_H)
    ap.add_argument("--fps", type=int, default=None,
                    help="capture/replay rate (video files default to their own)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--hsv", default=None, metavar="PATH",
                    help="HSV ranges JSON to use instead of MEASURED's "
                         "(calibration/hsv_ranges.json)")
    ap.add_argument("--imu", action="store_true", help="feed the MPU-6050 to Phase 3")
    ap.add_argument("--encoders", action="store_true",
                    help="feed the wheel encoders to Phase 3 (needs sudo pigpiod)")
    ap.add_argument("--gyro-bias", type=float, default=MEASURED_ESTIMATION.gyro_bias_dps, metavar="DPS",
                    help=f"gyro Z at rest, + = right, subtracted before integrating "
                         f"(default {MEASURED_ESTIMATION.gyro_bias_dps}, config.GYRO_BIAS_DPS)")
    ap.add_argument("--cm-per-px", type=float, default=None, metavar="S",
                    help="hand-measured ground scale; fills lane_offset_cm")
    ap.add_argument("--print-every", type=int, default=None, metavar="N",
                    help="status line every N frames (default: once a second); "
                         "0 prints events only")
    ap.add_argument("--verbose", action="store_true",
                    help="print Phase 3's per-frame debug log")
    ap.add_argument("--no-video", action="store_true",
                    help="don't record p3_debug.avi / .csv")
    ap.add_argument("--no-display", action="store_true", help="force headless")
    ap.add_argument("--scale", type=int, default=1, help="magnify the video and window")
    ap.add_argument("--out", default=None, metavar="DIR")

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        ap.print_help()
        return 0
    args = ap.parse_args(argv)

    try:
        if args.camera:
            source = CameraFrameSource(args.width, args.height, args.fps or FPS)
        elif args.video:
            source = VideoFrameSource(args.video, args.fps)
        else:
            source = DirectoryFrameSource(args.frames, args.fps or FPS)
    except (CaptureError, OSError) as exc:
        print(f"source error: {exc}")
        return 2

    config = MEASURED
    if args.hsv:
        config = replace(config, color=replace(config.color, hsv_ranges=load_hsv_ranges(args.hsv)))
    p3_config = replace(MEASURED_ESTIMATION, gyro_bias_dps=args.gyro_bias, cm_per_px=args.cm_per_px)

    out_dir = args.out or str(RUNS_DIR / ("p3_" + time.strftime("%Y%m%d_%H%M%S")))
    print_every = args.print_every if args.print_every is not None \
        else max(1, int(round(source.fps)))

    print(f"source   {source.label} @ {source.fps:.0f} FPS")
    print(f"output   {out_dir}")
    names = [n for n, on in (("IMU", args.imu), ("encoders", args.encoders)) if on]
    print(f"sensors  {' + '.join(names) or 'none'}   "
          f"color branch {'off' if config.color.hsv_ranges is None else 'on'}   "
          f"cm/px {args.cm_per_px if args.cm_per_px else 'uncalibrated'}\n")

    run(source, config, p3_config, args.imu, out_dir, print_every,
        args.verbose, args.limit, video=not args.no_video,
        display=not args.no_display, scale=args.scale, use_encoders=args.encoders)
    return 0

if __name__ == "__main__":
    sys.exit(cli())