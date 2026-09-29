"""Phase 3 view: what Phase 2 handed to Phase 3, and what Phase 3 did with it, frame by frame.

Purpose:
    phase3_linker's video. Phase 3 only acts across frames (EMA, jump gate,
    hold, votes), so a wrong packet is either bad input or a filter doing
    what it was told. This view puts both on one picture: the Phase 2 lane
    overlay (debug_lane.annotate, as in run.avi) with each traffic light and
    stop sign re-colored by its Phase 3 fate, then the lane filter's decision,
    the three votes and a scrolling timeline where "output held while the
    input moved" shows as a flat filtered trace over a moving raw one.
    It reads the records TracedPhase3Processor leaves in the debug dict;
    nothing here re-derives a decision.

Main package:
    Phase3View: implements the view interface in debug_video, on a
    phase3_linker.Phase3Result instead of a chain result.

    Top        camera frame under the lane overlay's header (padded, so the
               header hides nothing), lane overlay; traffic / sign boxes green when
               they passed the gate, amber below it, labeled "conf >= gate"
    Lane bar   [-1, 1]: raw offset (hollow; red when gated), filtered offset
               (filled); background by status; hold counter n/max and reason
    Votes      traffic, stop sign, stop line: the vote buffer as cells
               (oldest left), this frame's raw vote and the state
    Timeline   the last ~5 s: status band, raw and filtered offset, then
               drive state, stop sign and stop line tracks

Flow:
    extract() pulls the Phase3Result apart, observe() adds the frame to the
    timeline and the run statistics, render() draws the four panels.
"""
from collections import Counter, deque

import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.debugger.debug_lane import C_LANE, HEADER_H, annotate, candidate_gates
from src.estimation import CAUTION, GO, LANE_HOLD, LANE_STALE, LANE_VISION, STOP
from src.estimation_debug import NO_RESULT, UNUSABLE_MODE

STATUS_BG = {LANE_VISION: (30, 70, 30), LANE_HOLD: (0, 70, 95), LANE_STALE: (40, 30, 100)}
STATUS_FG = {LANE_VISION: dv.C_USABLE, LANE_HOLD: dv.C_AMBER, LANE_STALE: dv.C_RED}
DRIVE_COLORS = {GO: dv.C_USABLE, CAUTION: dv.C_AMBER, STOP: dv.C_RED}
C_DARK = (70, 70, 70)
TIMELINE_S = 5.0


def _raw_measured(lane: dict) -> bool:
    """True when this frame's raw offset is a measurement (a usable mode), gated or not."""
    return lane["reason"] not in (NO_RESULT, UNUSABLE_MODE)

def _cell_color(kind: str, value) -> tuple[int, int, int]:
    
    if kind == "traffic":
        return DRIVE_COLORS.get(value, dv.C_GRAY)
    if kind == "stop_sign":
        return dv.C_RED if value else C_DARK
    return dv.C_WHITE if value else C_DARK         # stop line: white like tape


class Phase3View:
    """
    Phase 3 view; implements the view interface in debug_video.

    lane_config: The LaneOffsetConfig the chain ran with, for the lane
        overlay's candidate gates (debug_lane.candidate_gates).
    fps: The run's frame rate; sizes the timeline to TIMELINE_S.
    last_render_ms: Set by the caller after each render; the next frame's
        corner shows it, since a frame can't show its own render time.
    """
    name = "phase3"
    CSV_FIELDS = (
        "frame_id", "timestamp_ms",
        "lane_mode", "lane_raw", "lane_accepted", "lane_reason", "lane_jump",
        "ema_before", "ema_after", "missed", "hold_max", "lane_status", "lane_offset",
        "yaw_rate", "dt", "heading", "heading_reset",
        "traffic_detections", "traffic_raw", "traffic_buffer", "drive_state",
        "sign_detections", "sign_raw", "sign_buffer", "stop_sign",
        "line_seen", "line_measured_px", "line_held", "line_buffer", "line_state",
        "line_reported_px", "line_reported_cm",
    )

    def __init__(self, lane_config=None, fps: float = dv.DEFAULT_FPS) -> None:
        self.lane_config = lane_config
        self.history: deque = deque(maxlen=max(2, int(round(fps * TIMELINE_S))))
        self.last_render_ms: float | None = None
        self._frames = 0
        self._reasons = Counter()
        self._status = Counter()
        self._held_moving = 0
        self._gated = Counter()
        self._line_held = 0

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def extract(self, res, frame=None) -> dict:
        """
        This frame's data from a Phase3Result.

        Inputs:
            res: phase3_linker.Phase3Result; p3_debug must come from
                TracedPhase3Processor (the lane, heading, traffic, stop_sign
                and stop_line records).
            frame: The source frame; None uses the chain's own.
        """
        chain, dbg = res.chain, res.p3_debug
        frame = chain.frame.frame if frame is None else frame
        return {
            "frame": frame,
            "chain": chain,
            "packet": res.packet,
            "dbg": dbg,
            "timings": dict(res.timings_ms),
            "gates": (candidate_gates(chain.geometry, self.lane_config)
                      if self.lane_config is not None else []),
        }

    def observe(self, data) -> None:
        """Add the frame to the timeline and the run statistics."""
        dbg, pk = data["dbg"], data["packet"]
        lane = dbg["lane"]
        self.history.append({
            "raw": lane["raw_offset"] if _raw_measured(lane) else None,
            "gated": lane["reason"] is not None and _raw_measured(lane),
            "offset": pk.lane_offset,
            "status": pk.lane_status,
            "drive": pk.drive_state,
            "sign": pk.stop_sign_detected,
            "line": pk.stop_line_detected,
        })
        self._frames += 1
        self._status[pk.lane_status] += 1
        if lane["reason"] is not None:
            self._reasons[lane["reason"]] += 1
            if pk.lane_status == LANE_HOLD and _raw_measured(lane):
                self._held_moving += 1
        for kind in ("traffic", "stop_sign"):
            self._gated[kind] += sum(not d["passed"] for d in dbg[kind]["detections"])
        self._line_held += bool(dbg["stop_line"]["held"])

    def row(self, data) -> list:
        """One CSV row, in CSV_FIELDS order."""
        d, pk = data["dbg"], data["packet"]
        lane, hd, tr, sg, sl = d["lane"], d["heading"], d["traffic"], d["stop_sign"], d["stop_line"]
        opt = lambda v: "" if v is None else v
        dets = lambda rec: ";".join(f"{x['label']}:{x['confidence']:.3f}:{'pass' if x['passed'] else 'gated'}"
                                    for x in rec["detections"])
        buf = lambda rec: "".join(str(v)[0].upper() for v in rec["buffer"])    # G/C/S, T/F
        return [
            pk.frame_id, pk.timestamp_ms,
            opt(lane["mode"]), opt(lane["raw_offset"]), int(lane["accepted"]), opt(lane["reason"]),
            opt(lane["jump"] and round(lane["jump"], 4)), opt(lane["ema_before"]), opt(lane["ema_after"]),
            lane["missed"], lane["hold_max"], lane["status"], lane["offset"],
            opt(hd["yaw_rate"]), hd["dt"], hd["heading"], int(hd["reset"]),
            dets(tr), tr["raw_vote"], buf(tr), tr["state"],
            dets(sg), int(sg["raw_vote"]), buf(sg), int(sg["state"]),
            int(sl["seen"]), opt(sl["measured_px"]), int(sl["held"]), buf(sl), int(sl["state"]),
            opt(sl["reported_px"]), opt(sl["reported_cm"]),
        ]

    def report(self) -> list[str]:
        """summary.txt's [PHASE 3] section."""
        n = max(self._frames, 1)
        out = [f"[PHASE 3] {self._frames} frames"]
        for s in (LANE_VISION, LANE_HOLD, LANE_STALE):
            out.append(f"  {s:<22}{self._status[s]:6}  ({100 * self._status[s] / n:5.1f}%)")
        if self._reasons:
            out.append(" lane frames not accepted, by reason:")
            for r, c in self._reasons.most_common():
                out.append(f"  {r:<22}{c:6}")
        out.append(f" held while the raw offset was measured  {self._held_moving}")
        out.append(f" detections below the gate               traffic {self._gated['traffic']}  "
                   f"stop sign {self._gated['stop_sign']}")
        out.append(f" frames with a held stop-line distance   {self._line_held}")
        return out

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def render(self, data, scale: int = 1) -> np.ndarray:
        """The four panels stacked, as wide as the frame at this scale."""
        s = max(1, int(scale))
        fs, th, lh = 0.38 * s, max(1, s // 2), 16 * s
        top = self._top(data, s, fs, th, lh)
        W = top.shape[1]
        panels = [top, self._lane_bar(data, W, s, fs, th, lh),
                  self._votes(data, W, s, fs, th, lh), self._timeline(W, s, fs, th, lh)]
        return np.vstack(panels)

    def _top(self, data, s, fs, th, lh) -> np.ndarray:
        """The lane overlay from run.avi, traffic and sign boxes by fate, and the corner timings."""
        chain, frame = data["chain"], data["frame"]
        # annotate() draws its header over the frame's top rows, where the
        # traffic ROI is; pad the frame so the header covers nothing
        pad = HEADER_H
        framed = np.vstack([np.zeros((pad, frame.shape[1], 3), np.uint8), frame])
        lx, ly, lw, lh_ = chain.roi.lane_rect
        img = annotate(framed, chain.offset, chain.offset_debug, (lx, ly + pad, lw, lh_),
                       data["gates"], s, (framed.shape[1], framed.shape[0]))
        for kind in ("traffic", "stop_sign"):
            for d in data["dbg"][kind]["detections"]:
                (rx, ry, _, _), (bx, by, bw, bh) = d["source_rect"], d["bbox"]
                ry += pad
                color = dv.C_USABLE if d["passed"] else dv.C_AMBER
                p0, p1 = ((rx + bx) * s, (ry + by) * s), ((rx + bx + bw) * s, (ry + by + bh) * s)
                cv2.rectangle(img, p0, p1, color, th + 1)
                label = f"{d['label'] or kind} {d['confidence']:.2f} {'>=' if d['passed'] else '<'} {d['gate']:.2f}"
                dv.draw_text(img, label, (p0[0], p1[1] + lh - 3 * s), color, fs, th)

        pk, t = data["packet"], data["timings"]
        ms = lambda k: f"{t[k]:.1f}" if k in t else "--"
        render = "--" if self.last_render_ms is None else f"{self.last_render_ms:.1f}"
        # The header's third line, right side: annotate's tags fill it from the left
        text = (f"t={pk.timestamp_ms / 1000.0:.2f}s  P1 {ms('capture')}  P2 {ms('phase2')}  "
                f"P3 {ms('phase3')}  render(prev) {render} ms")
        (tw, _), _ = cv2.getTextSize(text, dv.FONT, fs, th)
        dv.draw_text(img, text, (img.shape[1] - tw - 6 * s, 3 * lh - 3 * s), dv.C_GRAY, fs, th)
        return img

    def _lane_bar(self, data, W, s, fs, th, lh) -> np.ndarray:
        """Raw and filtered offset on [-1, 1] over the status color, with the hold counter and reason."""
        lane = data["dbg"]["lane"]
        H = 4 * lh
        img = np.full((H, W, 3), STATUS_BG.get(lane["status"], C_DARK), np.uint8)

        raw = f"{lane['raw_offset']:+.3f}" if _raw_measured(lane) else "--"
        head = (f"lane {lane['status']}  filtered {lane['offset']:+.3f}  raw {raw}"
                f" ({lane['mode'] or 'no result'})  "
                + (f"hold {lane['missed']}/{lane['hold_max']}" if lane["missed"] <= lane["hold_max"]
                   else f"missed {lane['missed']} > hold {lane['hold_max']}"))
        dv.draw_text(img, head, (6 * s, lh - 3 * s), STATUS_FG.get(lane["status"], dv.C_WHITE), fs, th)
        reason = lane["reason"]
        if reason is not None:
            if reason == "jump_gate":
                reason += f" {lane['jump']:+.3f} > {lane['max_jump']}"
            dv.draw_text(img, f"not accepted: {reason}", (6 * s, 2 * lh - 3 * s), dv.C_RED, fs, th)

        x0, x1, y = 12 * s, W - 12 * s, 3 * lh
        dv.draw_offset_axis(img, x0, x1, y, 5 * s, dv.C_WHITE)
        if lane["status"] != LANE_STALE:
            cv2.circle(img, (dv.offset_x(lane["offset"], x0, x1), y), 5 * s, C_LANE, -1)
        if lane["raw_offset"] is not None and _raw_measured(lane):
            color = dv.C_WHITE if lane["accepted"] else dv.C_RED
            cv2.circle(img, (dv.offset_x(lane["raw_offset"], x0, x1), y), 7 * s, color, th + 1)
        return img

    def _votes(self, data, W, s, fs, th, lh) -> np.ndarray:
        """One row per vote: its buffer as cells, oldest left, then the raw vote and the state."""
        d = data["dbg"]
        rows = (("traffic", "traffic"), ("stop_sign", "sign"), ("stop_line", "line"))
        rh = lh + 4 * s
        img = np.zeros((rh * len(rows) + 4 * s, W, 3), np.uint8)
        cw, ch, gap, label_w = 14 * s, 11 * s, 3 * s, 52 * s
        for i, (kind, label) in enumerate(rows):
            rec = d[kind]
            y = 2 * s + i * rh
            dv.draw_text(img, label, (6 * s, y + ch), dv.C_GRAY, fs, th)
            buf = rec["buffer"]
            for k in range(rec["window"]):
                x = label_w + k * (cw + gap)
                if k < len(buf):
                    cv2.rectangle(img, (x, y), (x + cw, y + ch), _cell_color(kind, buf[k]), -1)
                else:
                    cv2.rectangle(img, (x, y), (x + cw, y + ch), C_DARK, 1)
            x = label_w + rec["window"] * (cw + gap) + 8 * s
            state = rec["state"]
            text = f"raw {rec['raw_vote']}  -> {state}"
            if kind == "stop_line":
                if rec["held"]:
                    text += f"   held {rec['reported_px']:.0f}px"
                elif rec["reported_px"] is not None:
                    text += f"   seen {rec['reported_px']:.0f}px"
                if rec["reported_cm"] is not None:
                    text += f" / {rec['reported_cm']:.1f}cm"
                color = dv.C_AMBER if rec["held"] else (dv.C_WHITE if state else dv.C_GRAY)
            else:
                color = _cell_color(kind, state) if state not in (False, None) else dv.C_GRAY
                gated = sum(not x["passed"] for x in rec["detections"])
                if gated:
                    text += f"   {gated} below gate"
            dv.draw_text(img, text, (x, y + ch), color, fs, th)
        return img

    def _timeline(self, W, s, fs, th, lh) -> np.ndarray:
        """The last TIMELINE_S: status band with raw and filtered offset, then the three state tracks."""
        bh, tk = 64 * s, 7 * s
        H = bh + 3 * (tk + 2 * s) + lh
        img = np.zeros((H, W, 3), np.uint8)
        hist = list(self.history)
        n = self.history.maxlen
        dx = W / n
        mid = bh // 2
        yv = lambda v: int(mid - max(-1.0, min(1.0, v)) * (mid - 3 * s))
        x_of = lambda i: int(W - (len(hist) - i) * dx)      # newest at the right edge

        for i, h in enumerate(hist):
            cv2.rectangle(img, (x_of(i), 0), (int(x_of(i) + dx), bh), STATUS_BG[h["status"]], -1)
        cv2.line(img, (0, mid), (W, mid), dv.C_GRAY, 1)

        for i in range(1, len(hist)):
            a, b = hist[i - 1], hist[i]
            xa, xb = int(x_of(i - 1) + dx / 2), int(x_of(i) + dx / 2)
            if a["raw"] is not None and b["raw"] is not None:
                cv2.line(img, (xa, yv(a["raw"])), (xb, yv(b["raw"])), dv.C_GRAY, 1, cv2.LINE_AA)
            if a["status"] != LANE_STALE and b["status"] != LANE_STALE:
                cv2.line(img, (xa, yv(a["offset"])), (xb, yv(b["offset"])), C_LANE, th + 1, cv2.LINE_AA)
        for i, h in enumerate(hist):
            if h["gated"]:
                cv2.circle(img, (int(x_of(i) + dx / 2), yv(h["raw"])), 2 * s, dv.C_RED, -1)

        tracks = (("drive", lambda h: DRIVE_COLORS.get(h["drive"], dv.C_GRAY)),
                  ("sign", lambda h: dv.C_RED if h["sign"] else None),
                  ("line", lambda h: dv.C_WHITE if h["line"] else None))
        for r, (label, color_of) in enumerate(tracks):
            y = bh + 2 * s + r * (tk + 2 * s)
            for i, h in enumerate(hist):
                c = color_of(h)
                if c is not None:
                    cv2.rectangle(img, (x_of(i), y), (int(x_of(i) + dx), y + tk), c, -1)
            dv.draw_text(img, label, (4 * s, y + tk), dv.C_GRAY, fs * 0.8, th)

        dv.draw_text(img, f"last {TIMELINE_S:.0f} s   raw (gray, red = gated)   filtered (yellow)",
                     (4 * s, lh - 4 * s), dv.C_GRAY, fs * 0.8, th)
        return img
