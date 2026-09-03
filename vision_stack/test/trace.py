"""
trace.py

Per-Frame Diagnostic Tracer

Purpose:
    Emit a complete, record of what every pipeline stage decided
    on a given frame, so that an incorrect action can be traced back to the
    stage that produced it rather than inferred from the motor command.

    Every frame block is terminated by a trailer carrying the frame id and
    three timestamps (monotonic, wall clock, and the pipeline's own
    timestamp_ms), so blocks stay attributable when stdout interleaves with
    logging output from other threads.

WHY THIS IS NOT JUST print()
    The loop budget is 66.6 ms at 15 FPS and the robot is a Pi Zero 2W.
    An unbuffered write() to a terminal over SSH BLOCKS when the socket send
    buffer fills, and that block lands inside the frame budget. A tracer that
    prints 14 lines per frame with 14 separate write() calls will change the
    timing of the thing it is measuring.

    Three mitigations are built in:

      1. ONE write() per frame. Lines accumulate in a list and are joined and
         emitted as a single syscall.
      2. Optional file sink. Point the tracer at a file, `tail -f` it from a
         second tmux pane, and the terminal is fully off the critical path.
      3. Ring-buffer + trigger mode. Full detail for the last N frames is held
         in memory and only emitted when an anomaly fires. Normal frames cost
         one short line. This is the recommended mode for on-track running.

    The tracer times itself. `self.overhead_ms` is the cost of the last frame's
    formatting and I/O, and it is printed in the trailer. If it is not small
    relative to the budget, the measurement is lying to you.

LEVELS
    OFF     0   no output; every method early-returns on an int comparison
    FRAME   1   one line per frame (roughly what run_pipeline already logs)
    STAGE   2   one block per frame, one line per pipeline stage
    DETAIL  3   STAGE plus per-candidate and per-rejection detail

USAGE
    tracer = FrameTracer(level=TraceLevel.STAGE, sink="trace.log")
    ...
    tracer.begin(frame_id, t_frame_start, timestamp_ms)
    tracer.t_capture(ret, frame_bgr)
    tracer.mark("capture")
    ...
    tracer.end(nav_packet)
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Deque, Dict, List, Optional


# =============================================================================
# Levels
# =============================================================================
class TraceLevel(IntEnum):
    OFF    = 0
    FRAME  = 1
    STAGE  = 2
    DETAIL = 3


# =============================================================================
# Anomaly reasons
# =============================================================================
# A frame is "interesting" if any of these fire. In ring-buffer mode these are
# what trigger a dump of the preceding frames.

ANOM_NO_LANE       = "NO_LANE"        # lane mode resolved to none
ANOM_GATE_REJECT   = "GATE_REJECT"    # rate gate threw away a usable estimate
ANOM_CONF_REJECT   = "CONF_REJECT"    # lane confidence below min_confidence_lane
ANOM_DEAD_RECKON   = "DEAD_RECKON"    # packet not valid; offset is held
ANOM_STALE         = "STALE"          # age exceeded deadreck_max_frames
ANOM_WIDTH_REJECT  = "WIDTH_REJECT"   # a pair existed and failed the width band
ANOM_SINGLE_ANCHOR = "SINGLE_ANCHOR"  # offset inferred, not measured
ANOM_BUDGET        = "BUDGET"         # frame time exceeded the loop budget
ANOM_CAPTURE_FAIL  = "CAPTURE_FAIL"   # cap.read() returned False
ANOM_SATURATED     = "SATURATED"      # PD correction hit the clamp
ANOM_REVERSE       = "REVERSE"        # a wheel was commanded negative
ANOM_STEER_MISMATCH = "STEER_MISMATCH" # offset sign and wheel diff disagree

_ANOM_ALL = frozenset({
    ANOM_NO_LANE, ANOM_GATE_REJECT, ANOM_CONF_REJECT, ANOM_DEAD_RECKON,
    ANOM_STALE, ANOM_WIDTH_REJECT, ANOM_SINGLE_ANCHOR, ANOM_BUDGET,
    ANOM_CAPTURE_FAIL, ANOM_SATURATED, ANOM_REVERSE, ANOM_STEER_MISMATCH,
})


# =============================================================================
# Config
# =============================================================================
@dataclass
class TraceConfig:
    """
    level: TraceLevel; OFF disables every method on an int compare
    sink: "stdout" | "stderr" | a filesystem path
    every_n: emit a full block only every Nth frame (1 = every frame).
             The one-line FRAME summary is unaffected.
    ring_frames: when > 0, run in trigger mode. Full blocks are held in a ring
             of this depth and emitted only when an anomaly fires. 0 disables.
    trigger_on: set of anomaly codes that cause a ring dump. Empty = all.
    color: ANSI colour. Auto-disabled when the sink is not a tty.
    max_candidates: cap per-candidate DETAIL lines so one noisy frame cannot
             produce hundreds of lines.
    """
    level:          TraceLevel = TraceLevel.STAGE
    sink:           str        = "stdout"
    every_n:        int        = 1
    ring_frames:    int        = 0
    trigger_on:     frozenset  = _ANOM_ALL
    color:          bool       = True
    max_candidates: int        = 12


# =============================================================================
# Colour
# =============================================================================
class _C:
    RESET = "\033[0m"
    DIM   = "\033[2m"
    BOLD  = "\033[1m"
    RED   = "\033[31m"
    GRN   = "\033[32m"
    YEL   = "\033[33m"
    BLU   = "\033[34m"
    MAG   = "\033[35m"
    CYN   = "\033[36m"


_NOCOLOR = {k: "" for k in dir(_C) if not k.startswith("_")}


# =============================================================================
# Tracer
# =============================================================================
class FrameTracer:
    """
    One instance per run. begin() at the top of the loop, end() at the bottom.

    Thread safety: none. This is for the single-threaded time-sliced loop. If
    Phase 2 is ever split across cores, give each worker its own tracer and
    merge on frame_id.
    """

    # -- box drawing, ASCII only so it survives a dumb terminal over Pi Connect
    _TOP = "+-"
    _MID = "| "
    _BOT = "+-"

    def __init__(self, config: Optional[TraceConfig] = None, **kwargs):
        # Allow FrameTracer(level=..., sink=...) as a shorthand
        if config is None:
            config = TraceConfig(**kwargs)
        self._cfg = config

        self._buf: List[str] = []
        self._ring: Deque[List[str]] = deque(maxlen=max(config.ring_frames, 1))

        self._frame_id: int = -1
        self._t_begin: float = 0.0
        self._timestamp_ms: int = 0
        self._t_last_mark: float = 0.0
        self._t_prev_frame_begin: Optional[float] = None
        self._marks: List[tuple] = []
        self._anomalies: List[str] = []

        self.overhead_ms: float = 0.0
        self.frames_emitted: int = 0
        self.frames_suppressed: int = 0

        # Sink
        self._own_fh = False
        if config.sink == "stdout":
            self._fh = sys.stdout
        elif config.sink == "stderr":
            self._fh = sys.stderr
        else:
            d = os.path.dirname(config.sink)
            if d:
                os.makedirs(d, exist_ok=True)
            self._fh = open(config.sink, "w", buffering=1)   # line buffered
            self._own_fh = True

        is_tty = False
        try:
            is_tty = self._fh.isatty()
        except Exception:
            pass
        self._c = _C if (config.color and is_tty) else _NOCOLOR

    # =====================================================================
    # Properties
    # =====================================================================
    @property
    def enabled(self) -> bool:
        return self._cfg.level > TraceLevel.OFF

    @property
    def detail(self) -> bool:
        return self._cfg.level >= TraceLevel.DETAIL

    # =====================================================================
    # Frame lifecycle
    # =====================================================================
    def begin(self, frame_id: int, t_frame_start: float, timestamp_ms: int) -> None:
        """
        Purpose:
            Open a frame record.

        Inputs:
            frame_id: capture-loop frame counter
            t_frame_start: time.perf_counter() taken at the top of the loop
            timestamp_ms: the pipeline's own int(t_frame_start * 1000.0)
        """
        if self._cfg.level == TraceLevel.OFF:
            return
        self._buf = []
        self._marks = []
        self._anomalies = []
        self._t_prev_frame_begin = self._t_begin or None
        self._frame_id = frame_id
        self._t_begin = t_frame_start
        self._timestamp_ms = timestamp_ms
        self._t_last_mark = t_frame_start

    def mark(self, stage: str) -> None:
        """
        Purpose:
            Record the wall time consumed since the previous mark.
            Call immediately after each pipeline stage returns.
        """
        if self._cfg.level == TraceLevel.OFF:
            return
        now = time.perf_counter()
        self._marks.append((stage, (now - self._t_last_mark) * 1000.0))
        self._t_last_mark = now

    def anomaly(self, code: str) -> None:
        """
        Purpose:
            Flag this frame as interesting. Safe to call more than once with
            the same code.
        """
        if self._cfg.level == TraceLevel.OFF:
            return
        if code not in self._anomalies:
            self._anomalies.append(code)

    # =====================================================================
    # Internal line helpers
    # =====================================================================
    def _sec(self, tag: str, colour: str = "") -> None:
        c = self._c
        self._buf.append(f"{self._MID}{colour}{tag:<5}{c['RESET']} ")

    def _add(self, tag: str, text: str, colour: str = "") -> None:
        c = self._c
        col = getattr(_C, colour, "") if (colour and self._c is _C) else ""
        self._buf.append(f"{self._MID}{col}{tag:<6}{c['RESET']}{text}")

    def _sub(self, text: str) -> None:
        c = self._c
        self._buf.append(f"{self._MID}      {c['DIM']}{text}{c['RESET']}")

    # =====================================================================
    # Stage recorders
    # =====================================================================
    def t_capture(self, ret: bool, frame) -> None:
        """
        Purpose:
            Record the Phase 1 capture result.
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        if not ret or frame is None:
            self.anomaly(ANOM_CAPTURE_FAIL)
            self._add("CAP", "read FAILED - frame dropped", "RED")
            return
        h, w = frame.shape[:2]
        ch = frame.shape[2] if frame.ndim == 3 else 1
        self._add("CAP", f"{w}x{h}x{ch} {frame.dtype} "
                         f"mean={float(frame.mean()):6.2f} "
                         f"min={int(frame.min()):3d} max={int(frame.max()):3d}")

    def t_roi(self, roi_result) -> None:
        """
        Purpose:
            Record the three ROI rectangles in source-frame pixels.

        Inputs:
            roi_result: roi_crop.ROICropResult
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        sh, sw = roi_result.source_shape[:2]
        self._add("ROI", f"src={sw}x{sh}  "
                         f"lane={_rect(roi_result.lane_rect)}  "
                         f"traf={_rect(roi_result.traffic_rect)}  "
                         f"sign={_rect(roi_result.sign_rect)}")

    def t_geometry(self, geo_result, reject_counts: Optional[Dict[str, int]] = None,
                   lane_roi_shape: Optional[tuple] = None) -> None:
        """
        Purpose:
            Record what the geometry branch produced and, if the instrumented
            build is in use, why every other contour was thrown away.

        Inputs:
            geo_result: geometry.GeometryBranchResult
            reject_counts: dict from the patched _extract_lane_candidates.
                Keys: seen, area, degenerate, too_few_pts, aspect, span,
                      intensity, accepted
            lane_roi_shape: (H, W) of the lane ROI, for edge-margin context
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        n_lane = len(geo_result.lane_candidates)
        n_sign = len(geo_result.sign_candidates)
        colour = "" if n_lane else "YEL"
        self._add("GEO", f"lane_cand={n_lane:2d}  sign_cand={n_sign:2d}", colour)

        if reject_counts:
            seen = reject_counts.get("seen", 0)
            acc = reject_counts.get("accepted", 0)
            drops = "  ".join(
                f"{k}={v}" for k, v in reject_counts.items()
                if k not in ("seen", "accepted") and v
            ) or "none"
            self._sub(f"contours={seen:3d} -> accepted={acc:2d}   rejected: {drops}")

        if self.detail:
            for i, c in enumerate(geo_result.lane_candidates[:self._cfg.max_candidates]):
                x, y, w, h = c.bbox
                cx = x + w / 2.0
                self._sub(f"lane[{i}] bbox=({x:3d},{y:3d},{w:3d},{h:3d}) "
                          f"cx={cx:6.1f} conf={c.confidence:.4f} "
                          f"pts={len(c.contour):3d}")
            for i, c in enumerate(geo_result.sign_candidates[:self._cfg.max_candidates]):
                x, y, w, h = c.bbox
                self._sub(f"sign[{i}] bbox=({x:3d},{y:3d},{w:3d},{h:3d}) "
                          f"verts={c.vertex_count} conf={c.confidence:.4f}")

    def t_color(self, tl_candidates) -> None:
        """
        Purpose:
            Record traffic-light candidates from the colour branch.
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        n = len(tl_candidates)
        labels = ",".join(f"{c.label}:{c.confidence:.2f}" for c in tl_candidates[:6]) or "-"
        self._add("COL", f"tl_cand={n:2d}  [{labels}]")

    def t_fusion(self, detections, summary: dict) -> None:
        """
        Purpose:
            Record the fused detection set and what fusion discarded.

        Inputs:
            detections: list[feature_fusion.DetectionObject]
            summary: the debug_summary dict returned by fuse_detections()
                Keys: frame_id, timestamp_ms, counts, total, discarded,
                      suppressed, log
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        counts = ", ".join(f"{k}={v}" for k, v in summary.get("counts", {}).items()) or "none"
        disc = summary.get("discarded", 0)
        supp = summary.get("suppressed", 0)
        colour = "YEL" if (disc or supp) else ""
        self._add("FUS", f"total={summary.get('total', len(detections)):2d}  {counts}  "
                         f"discarded={disc} suppressed={supp}", colour)
        if self.detail:
            for entry in summary.get("log", [])[:self._cfg.max_candidates]:
                self._sub(str(entry))

    def t_offset(self, r, frame_width: int, conf_threshold: float) -> None:
        """
        Purpose:
            Record the single most important decision in the pipeline: how the
            lane_boundary set was reduced to one signed offset.

        Inputs:
            r: lane_offset.LaneOffsetResult
            frame_width: the width passed to compute_lane_offset()
            conf_threshold: the gate that produced r.boundary_count
        """
        if self._cfg.level < TraceLevel.STAGE:
            return

        if r.mode == "none":
            self.anomaly(ANOM_NO_LANE)
        elif r.mode == "width_rejected":
            self.anomaly(ANOM_WIDTH_REJECT)
        elif r.mode in ("left_only", "right_only"):
            self.anomaly(ANOM_SINGLE_ANCHOR)

        colour = {"two_boundary": "GRN", "none": "RED"}.get(r.mode, "YEL")
        fc = frame_width / 2.0
        lx = f"{r.left_x:6.1f}" if r.left_x is not None else "  ----"
        rx = f"{r.right_x:6.1f}" if r.right_x is not None else "  ----"
        wpx = f"{r.lane_width_px:6.1f}" if r.lane_width_px is not None else "  ----"

        self._add("OFF", f"mode={r.mode:<14} off={r.offset:+.4f}  conf={r.confidence:.4f}  "
                         f"n_gated={r.boundary_count:2d} (>= {conf_threshold:.2f})", colour)

        if r.lane_width_px is not None:
            centre = (r.left_x + r.right_x) / 2.0
            self._sub(f"L={lx}  R={rx}  width={wpx}px  lane_ctr={centre:6.1f}  "
                      f"frame_ctr={fc:6.1f}  delta={centre - fc:+6.1f}px")
        else:
            self._sub(f"L={lx}  R={rx}  width=  ----   "
                      f"frame_ctr={fc:6.1f}  (single-anchor inference)")

        # Sign-convention restatement. This is the field most often
        # misread when a run goes the wrong way.
        if r.mode != "none":
            steer = "RIGHT" if r.offset > 0 else ("LEFT" if r.offset < 0 else "STRAIGHT")
            side = "LEFT of" if r.offset > 0 else ("RIGHT of" if r.offset < 0 else "ON")
            self._sub(f"interpretation: robot is {side} lane centre -> steer {steer}")

    def t_seam(self, p3_input) -> None:
        """
        Purpose:
            Record exactly what crosses the Phase 2 / Phase 3 boundary.
            If this line and the OFF line ever disagree, the seam is broken.

        Inputs:
            p3_input: contracts.Phase2Snapshot
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        le = p3_input.lane
        self._add("SEAM", f"lane.offset_norm={le.offset_norm:+.4f} mode={le.mode:<14} "
                          f"conf={le.confidence:.4f} valid={_b(le.valid)} "
                          f"n_det={len(p3_input.detections)}")
        if self.detail:
            for d in p3_input.detections[:self._cfg.max_candidates]:
                self._sub(f"det {d.type:<13} {d.label:<10} "
                          f"pos=({d.position_x:6.1f},{d.position_y:6.1f}) "
                          f"conf={d.confidence:.4f}")

    def t_imu(self, imu_frame, sensor_sample) -> None:
        """
        Purpose:
            Record the sensor window backing the dead-reckoning fallback.

        Inputs:
            imu_frame: imu.IMUFrame
            sensor_sample: contracts.SensorSample
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        colour = "" if imu_frame.valid else "YEL"
        yaw = imu_frame.mean_yaw_rate_dps
        lat = imu_frame.peak_lateral_accel
        self._add("IMU", f"n={imu_frame.sample_count:3d} valid={_b(imu_frame.valid)}  "
                         f"yaw={_f(yaw, '%+7.2f')} deg/s  "
                         f"lat_acc={_f(lat, '%+6.2f')} m/s2  "
                         f"wheel={_f(sensor_sample.wheel_speed, '%+5.2f')}", colour)

    def t_phase3(self, packet, lane_diag: Optional[dict] = None,
                 processor: Optional[Any] = None) -> None:
        """
        Purpose:
            Record the Phase 3 output and, on an instrumented build, the gate
            decisions that produced it.

        Inputs:
            packet: contracts.EstimationPacket
            lane_diag: the dict returned by the patched _filter_lane(), which
                carries the extra keys: raw, usable, reject, ema_prev, dt
            processor: the Phase3Processor, for vote-buffer introspection
        """
        if self._cfg.level < TraceLevel.STAGE:
            return

        if not packet.lane_offset_valid:
            self.anomaly(ANOM_DEAD_RECKON)
        if packet.lane_offset_stale:
            self.anomaly(ANOM_STALE)

        colour = "GRN" if packet.lane_offset_valid else ("RED" if packet.lane_offset_stale else "YEL")
        self._add("P3", f"off={packet.lane_offset:+.4f} m  norm={packet.lane_offset_norm:+.4f}  "
                        f"valid={_b(packet.lane_offset_valid)} age={packet.lane_offset_age:2d} "
                        f"stale={_b(packet.lane_offset_stale)} mode={packet.lane_mode:<14}", colour)
        self._sub(f"heading={packet.heading_error:+7.2f} deg src={packet.heading_source:<6}  "
                  f"drive={packet.drive_state:<7} stop_sign={_b(packet.stop_sign_detected)}")

        if lane_diag:
            reject = lane_diag.get("reject")
            if reject == "rate_gate":
                self.anomaly(ANOM_GATE_REJECT)
            elif reject == "confidence":
                self.anomaly(ANOM_CONF_REJECT)
            prev = lane_diag.get("ema_prev")
            self._sub(f"filter: raw={_f(lane_diag.get('raw'), '%+.4f')} -> "
                      f"ema={packet.lane_offset_norm:+.4f}  "
                      f"prev_ema={_f(prev, '%+.4f')}  "
                      f"usable={_b(lane_diag.get('usable', False))}  "
                      f"reject={reject or '-':<12} dt={lane_diag.get('dt', 0.0) * 1000.0:5.1f}ms")
            if reject == "rate_gate":
                self._sub(f"        rate gate: |{_f(lane_diag.get('raw'), '%+.4f')} - "
                          f"{_f(lane_diag.get('gate_last'), '%+.4f')}| = "
                          f"{_f(lane_diag.get('gate_delta'), '%.4f')} > "
                          f"limit {_f(lane_diag.get('gate_limit'), '%.4f')}")

        if processor is not None and self.detail:
            try:
                self._sub(f"vote drive={list(processor._drive_state_buf)} "
                          f"stop={list(processor._stop_sign_buf)}")
            except AttributeError:
                pass

    def t_pd(self, left: float, right: float, diag: Optional[dict] = None,
             packet=None) -> None:
        """
        Purpose:
            Record the wheel command and the terms behind it, and assert that
            the commanded turn agrees with the direction the offset sign asked
            for.

        Inputs:
            left, right: the values actually handed to _drive()
            diag: PDController.last_diag from the instrumented build. Keys:
                ramp_frac, ramped_base, error, derivative, correction,
                correction_raw, clamped, held
            packet: contracts.EstimationPacket, for the steer cross-check

        Notes:
            The turn label assumes the standard differential-drive convention:
            a robot turns toward its SLOWER wheel. If your chassis is geared or
            mounted such that this is inverted, change _turn_of() and nothing
            else -- do not "fix" it by flipping a gain.
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        if left < 0.0 or right < 0.0:
            self.anomaly(ANOM_REVERSE)

        turn = _turn_of(left, right)

        colour = "RED" if (left < 0.0 or right < 0.0) else ""
        self._add("PD", f"L={left:+.4f}  R={right:+.4f}   diff={right - left:+.4f}  "
                        f"-> {turn}", colour)

        # ---- steer consistency: intent (offset sign) vs command (wheel diff)
        if packet is not None and packet.lane_offset_valid:
            intent = ("RIGHT" if packet.lane_offset_norm > 1e-4
                      else "LEFT" if packet.lane_offset_norm < -1e-4 else "STRAIGHT")
            if intent != "STRAIGHT" and turn != "STRAIGHT" and intent != turn:
                self.anomaly(ANOM_STEER_MISMATCH)
                self._sub(f"!! STEER MISMATCH: offset {packet.lane_offset_norm:+.4f} "
                          f"asks for {intent}, wheels command {turn}")

        if diag:
            if diag.get("clamped"):
                self.anomaly(ANOM_SATURATED)
            self._sub(f"ramp={diag.get('ramp_frac', 0.0):.3f} "
                      f"base={diag.get('ramped_base', 0.0):.4f}  "
                      f"err={_f(diag.get('error'), '%+.4f')} "
                      f"deriv={_f(diag.get('derivative'), '%+.4f')}  "
                      f"corr_raw={_f(diag.get('correction_raw'), '%+.4f')} -> "
                      f"corr={_f(diag.get('correction'), '%+.4f')}"
                      f"{'  [CLAMPED]' if diag.get('clamped') else ''}"
                      f"{'  [HELD]' if diag.get('held') else ''}")

    def note(self, text: str, colour: str = "") -> None:
        """
        Purpose:
            Free-form line for anything not covered by a stage recorder.
        """
        if self._cfg.level < TraceLevel.STAGE:
            return
        self._add("NOTE", text, colour)

    # =====================================================================
    # Emit
    # =====================================================================
    def end(self, packet=None, budget_ms: Optional[float] = None) -> float:
        """
        Purpose:
            Close the frame, append the timing line and the id/timestamp
            trailer, and emit in a single write().

        Inputs:
            packet: contracts.EstimationPacket, or None if the frame aborted
            budget_ms: loop budget; exceeding it raises the BUDGET anomaly

        Output:
            proc_ms: measured processing time for the frame, excluding the
                     tracer's own overhead
        """
        if self._cfg.level == TraceLevel.OFF:
            return 0.0

        t_end = time.perf_counter()
        proc_ms = (t_end - self._t_begin) * 1000.0
        dt_ms = ((self._t_begin - self._t_prev_frame_begin) * 1000.0
                 if self._t_prev_frame_begin else 0.0)

        if budget_ms is not None and proc_ms > budget_ms:
            self.anomaly(ANOM_BUDGET)

        c = self._c

        # ---- timing line
        if self._cfg.level >= TraceLevel.STAGE and self._marks:
            parts = " ".join(f"{name}={ms:5.1f}" for name, ms in self._marks)
            self._add("TIME", parts)

        # ---- trailer: frame id + timestamps. Requested format.
        wall = time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"
        anom = ",".join(self._anomalies) if self._anomalies else "-"
        anom_col = _C.RED if (self._anomalies and self._c is _C) else ""
        fps = (1000.0 / dt_ms) if dt_ms > 0 else 0.0

        trailer = (
            f"{self._BOT}f={self._frame_id:06d}  "
            f"ts_ms={self._timestamp_ms}  "
            f"mono={self._t_begin:12.4f}s  "
            f"wall={wall}  "
            f"dt={dt_ms:6.2f}ms ({fps:5.2f} fps)  "
            f"proc={proc_ms:6.2f}ms  "
            f"trace={self.overhead_ms:5.2f}ms  "
            f"{anom_col}flags={anom}{c['RESET']}"
        )

        # ---- level 1: single summary line, nothing else
        if self._cfg.level == TraceLevel.FRAME:
            line = self._frame_line(packet, proc_ms, dt_ms, anom)
            self._write(line + "\n")
            self.frames_emitted += 1
            self.overhead_ms = (time.perf_counter() - t_end) * 1000.0
            return proc_ms

        header = (f"{self._TOP}frame {self._frame_id:06d} "
                  + "-" * max(4, 96 - len(str(self._frame_id))))
        block = [header] + self._buf + [trailer, ""]

        fired = bool(self._anomalies) and bool(
            set(self._anomalies) & set(self._cfg.trigger_on))

        if self._cfg.ring_frames > 0:
            # Trigger mode: hold the block, emit only on an anomaly.
            self._ring.append(block)
            if fired:
                out = []
                for b in self._ring:
                    out.extend(b)
                self._ring.clear()
                self._write("\n".join(out) + "\n")
                self.frames_emitted += 1
            else:
                self._write(self._frame_line(packet, proc_ms, dt_ms, anom) + "\n")
                self.frames_suppressed += 1
        elif (self._frame_id % max(self._cfg.every_n, 1)) == 0 or fired:
            self._write("\n".join(block) + "\n")
            self.frames_emitted += 1
        else:
            self.frames_suppressed += 1

        self.overhead_ms = (time.perf_counter() - t_end) * 1000.0
        return proc_ms

    def _frame_line(self, packet, proc_ms: float, dt_ms: float, anom: str) -> str:
        """
        Purpose:
            The compact one-line-per-frame form, always terminated by the
            frame id and timestamp.
        """
        if packet is None:
            return (f"f={self._frame_id:06d} ABORTED "
                    f"proc={proc_ms:6.2f}ms flags={anom} "
                    f"ts_ms={self._timestamp_ms}")
        return (
            f"f={self._frame_id:06d} "
            f"off={packet.lane_offset:+.4f} n={packet.lane_offset_norm:+.4f} "
            f"{'OK ' if packet.lane_offset_valid else 'DR '}"
            f"age={packet.lane_offset_age:02d}{' STALE' if packet.lane_offset_stale else '      '} "
            f"head={packet.heading_error:+6.2f} src={packet.heading_source:<6} "
            f"drive={packet.drive_state:<7} mode={packet.lane_mode:<14} "
            f"proc={proc_ms:6.2f}ms dt={dt_ms:6.2f}ms flags={anom} "
            f"ts_ms={packet.timestamp_ms}"
        )

    def _write(self, text: str) -> None:
        try:
            self._fh.write(text)
        except (BrokenPipeError, ValueError):
            pass

    # =====================================================================
    # Shutdown
    # =====================================================================
    def close(self) -> None:
        """
        Purpose:
            Flush any frames still held in the ring and release the sink.
            Call from the pipeline's finally block.
        """
        if self._cfg.level == TraceLevel.OFF:
            return
        if self._ring:
            out = []
            for b in self._ring:
                out.extend(b)
            self._ring.clear()
            self._write("\n".join(out) + "\n")
        self._write(f"# tracer: emitted={self.frames_emitted} "
                    f"suppressed={self.frames_suppressed}\n")
        try:
            self._fh.flush()
        finally:
            if self._own_fh:
                self._fh.close()


# =============================================================================
# Formatting helpers
# =============================================================================
def _b(v: Optional[bool]) -> str:
    """Fixed-width boolean so columns line up across frames."""
    return "T" if v else "F"


def _f(v: Optional[float], fmt: str = "%.4f") -> str:
    """Format a float that may be None, at the width the format implies."""
    if v is None:
        width = 0
        for ch in fmt:
            if ch.isdigit():
                width = width * 10 + int(ch)
            elif ch == ".":
                break
        return "-".rjust(max(width, 4))
    return fmt % v


def _turn_of(left: float, right: float, eps: float = 1e-6) -> str:
    """
    Purpose:
        Name the commanded turn from a differential wheel pair.
        A differential-drive robot turns toward its SLOWER wheel.
    """
    if abs(left - right) <= eps:
        return "STRAIGHT"
    return "RIGHT" if left > right else "LEFT"


def _rect(r: Optional[tuple]) -> str:
    if not r:
        return "----"
    x, y, w, h = r
    return f"({x:3d},{y:3d},{w:3d},{h:3d})"