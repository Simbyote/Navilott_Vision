"""Phase 1-3 linker: capture, perception and estimation in one headless process, reported as text.

Purpose:
    The estimation test harness. Phase 3's lane filter, dropout hold, heading
    tracker and votes only mean something across frames, so this runs the
    whole chain and shows each packet next to the Phase 2 input it came from;
    a bad packet can then be traced to bad input or bad filtering. Phases 1-2
    come from phase2_linker.run_chain(), the only place that order is
    written, so the two linkers can't drift. It runs the chain itself, live or
    from a replay, rather than reading phase2_linker's recordings, and it
    doesn't draw or record video; live_view is the visual debugger.

Main package:
    Phase3Result: one frame's ChainResult, the EstimationPacket handed to
    Navigation, Phase 3's debug summary, and per-phase timings.

Flow:
    FrameSource -> run_chain() -> Phase3Processor.process() -> EstimationPacket
                   (Phases 1-2)   (Phase 3)
    Each result goes to p3.csv, the event tracker and the run statistics;
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
from src.perception.color_branch import ColorConfig, load_hsv_ranges
from src.params import FPS, FRAME_H, FRAME_W, RUNS_DIR, STOP_SIGN, TRAFFIC_LIGHT
from src.phase2_linker import MEASURED, ChainResult, PipelineConfig, run_chain
from src.debugger.live_view import (
    CameraFrameSource, VideoFrameSource, DirectoryFrameSource,
)
from src.estimation import (
    LANE_VISION, LANE_HOLD, LANE_STALE,
    EstimationPacket, Phase3Config, Phase3Processor, SensorSample,
)

# --help text. Kept apart from the module docstring, which documents the code.
_CLI_HELP = """\
Run capture -> perception -> estimation headless and report what estimation
decided, frame by frame, next to the Phase 2 input it came from.

Sources:
    --camera        live capture through capture.CameraSource
    --video PATH    a recorded clip, e.g. a live_view run.avi
    --frames DIR    an image sequence, sorted by filename
    Replays are deterministic, so estimation config changes can be compared
    on the same footage. Replay timestamps come from the nominal frame rate.

Sensors:
    --imu starts the MPU-6050 reader and feeds it to Phase 3 each frame.
    Without it, Phase 3 runs with no sensors: heading holds at 0 and the
    pass-through fields read 0.0. The IMU driver is imported only with --imu,
    so replays run off the Pi.

Output (--out DIR, default <root>/runs/p3_<timestamp>):
    console      one status line every --print-every frames, plus an event
                 line on every lane_status, drive_state or stop_sign change
    p3.csv       every frame: timings, the Phase 2 lane input, the packet,
                 and Phase 3's debug log
    summary.txt  timing percentiles, lane status and mode histograms, longest
                 hold and stale runs, and offset statistics while on vision

Examples (from the repo root):
    python3 -m src.phase3_linker --video run.avi
    python3 -m src.phase3_linker --camera --imu --fps 20
    python3 -m src.phase3_linker --camera --limit 200 --print-every 1

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
    p3_debug: dict                  # Phase3Processor's debug summary: dt, log, ...
    timings_ms: dict = field(default_factory=dict)      # capture, phase2, phase3, total

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
        processor: This run's Phase3Processor. Stateful: pass the same one
            every frame, in order.
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
        f"{pk.drive_state} stop={'T' if pk.stop_sign_detected else 'F'}"
    )


class EventTracker:
    """
    Reports changes in the packet fields Navigation acts on: lane_status,
    drive_state and stop_sign_detected. counts holds transitions per field.
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
    "lane_offset", "lane_offset_cm", "lane_status", "heading_error",
    "drive_state", "stop_sign_detected", "yaw_rate", "lateral_accel",
    "wheel_speed", "p3_log",
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
            pk.lane_offset, pk.lane_offset_cm, pk.lane_status,
            pk.heading_error, pk.drive_state, int(pk.stop_sign_detected),
            pk.yaw_rate, pk.lateral_accel, pk.wheel_speed,
            " | ".join(res.p3_debug.get("log", [])),
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
            self.timings[k].append(v)
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
        return lines


class _NoSensors:
    """Stand-in when --imu isn't given: every frame runs without sensors."""
    def sample(self) -> SensorSample | None:
        return None

    def stop(self) -> None:
        pass

class _ImuSensors:
    """
    The MPU-6050 through peripherals.imu, imported here so replays never load
    the board drivers. Uses IMU_I2C_ADDRESS and IMU_RATE_HZ from params, and
    doesn't calibrate, so any gyro bias correction comes from --gyro-bias.
    """
    def __init__(self) -> None:
        from src.peripherals.imu import IMUReader
        self._imu = IMUReader()
        self._imu.start()
        time.sleep(0.1)
        self._imu.snapshot()        # drop what accumulated during startup

    def sample(self) -> SensorSample:
        return SensorSample.from_imu(self._imu.snapshot())

    def stop(self) -> None:
        self._imu.stop()


def run(
        source,
        config: PipelineConfig = MEASURED,
        p3_config: Phase3Config = Phase3Config(),
        use_imu: bool = False,
        out_dir: str = str(RUNS_DIR / "p3"),
        print_every: int = 20,
        verbose: bool = False,
        limit: int | None = None,
    ) -> Phase3Stats:
    """
    Run every frame from source through all three phases.

    Inputs:
        source: A live_view FrameSource: camera, video file or image directory.
        config: Phase 2 tuning.
        p3_config: Phase 3 tuning. When cm_per_px is set and
            lane_roi_width_px isn't, the width is taken from the first
            frame's lane ROI.
        use_imu: Start the IMU and feed it to Phase 3.
        print_every: Status line every N frames; 0 prints events only.
        verbose: Also print Phase 3's per-frame debug log.
        limit: Stop after this many frames; None runs until the source ends.

    Outputs:
        Phase3Stats.

    Side effects:
        Writes p3.csv and summary.txt into out_dir (created if missing),
        prints to the console, and closes the source. Ctrl-C ends the run
        early; the CSV and summary are still written.
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = Phase3Stats(budget_ms=1000.0 / max(source.fps, 1))
    events = EventTracker()
    log = CsvLog(os.path.join(out_dir, "p3.csv"))
    sensors = _ImuSensors() if use_imu else _NoSensors()
    processor = None

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
                # Built on the first frame so the lane ROI width is known
                if p3_config.cm_per_px is not None and p3_config.lane_roi_width_px is None:
                    lane_w = run_chain(frame, fid, ts, config).roi.lane_rect[2]
                    p3_config = replace(p3_config, lane_roi_width_px=int(lane_w))
                processor = Phase3Processor(p3_config)

            res = run_phase3_chain(frame, fid, ts, processor, sample, config, capture_ms)
            stats.update(res)
            log.write(res)

            for line in events.update(res):
                print(line)
            if print_every and stats.frames % print_every == 0:
                print(status_line(res))
            if verbose:
                for entry in res.p3_debug.get("log", []):
                    print(f"    {entry}")

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        log.close()
        sensors.stop()
        source.close()

    summary = stats.report()
    summary += ["", f"transitions      lane {events.counts['lane']}  "
                    f"drive {events.counts['drive']}  "
                    f"stop_sign {events.counts['stop_sign']}"]
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
                    help="calibrated HSV ranges JSON; switches the color branch on")
    ap.add_argument("--imu", action="store_true", help="feed the MPU-6050 to Phase 3")
    ap.add_argument("--gyro-bias", type=float, default=0.0, metavar="DPS",
                    help="gyro Z reading at standstill, subtracted before integrating")
    ap.add_argument("--cm-per-px", type=float, default=None, metavar="S",
                    help="hand-measured ground scale; fills lane_offset_cm")
    ap.add_argument("--print-every", type=int, default=None, metavar="N",
                    help="status line every N frames (default: once a second); "
                         "0 prints events only")
    ap.add_argument("--verbose", action="store_true",
                    help="print Phase 3's per-frame debug log")
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
        config = replace(config, color=ColorConfig(load_hsv_ranges(args.hsv),
                                                   config.color.blob))
    p3_config = Phase3Config(gyro_bias_dps=args.gyro_bias, cm_per_px=args.cm_per_px)

    out_dir = args.out or str(RUNS_DIR / ("p3_" + time.strftime("%Y%m%d_%H%M%S")))
    print_every = args.print_every if args.print_every is not None \
        else max(1, int(round(source.fps)))

    print(f"source   {source.label} @ {source.fps:.0f} FPS")
    print(f"output   {out_dir}")
    print(f"sensors  {'IMU' if args.imu else 'none'}   "
          f"color branch {'on' if args.hsv else 'off'}   "
          f"cm/px {args.cm_per_px if args.cm_per_px else 'uncalibrated'}\n")

    run(source, config, p3_config, args.imu, out_dir, print_every,
        args.verbose, args.limit)
    return 0

if __name__ == "__main__":
    sys.exit(cli())