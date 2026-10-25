"""Lane view: what compute_lane_offset() decided, drawn on the frame it decided it on.

Purpose:
    Anchor selection and gating reviewed frame by frame after a run, with no
    monitor on the robot. LaneView (live_view) and DebugVideoWriter (a loop
    without a chain result) share one drawing. Only candidate_gates() imports
    lane_offset, at its first call, so no import cycle is possible.

Main package:
    annotate(): raw candidates (green usable, red with their gate), left
    (cyan) and right (magenta) boundaries, robot (gray) and lane center
    (yellow, red when clamped), under a mode and offset header.
"""
import re
from collections import Counter

import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.params import (
    FRAME_H, FRAME_W, MODE_LEFT_ONLY, MODE_NONE, MODE_RIGHT_ONLY,
    MODE_SINGLE_UNCALIBRATED, MODE_TWO_BOUNDARY,
)

FRAME_SIZE = (FRAME_W, FRAME_H)      # (w, h) the lane_rect coordinates assume

STEERING_MODES = (MODE_TWO_BOUNDARY, MODE_LEFT_ONLY, MODE_RIGHT_ONLY)
LOG_TAGS = ("BLIND", "MERGE", "SPAN", "ONE-SIDED", "UNCALIBRATED", "REJECT")

MODE_COLORS = {                               # BGR
    MODE_TWO_BOUNDARY:        (80, 210, 80),
    MODE_LEFT_ONLY:           (0, 200, 255),
    MODE_RIGHT_ONLY:          (0, 200, 255),
    MODE_SINGLE_UNCALIBRATED: (0, 120, 255),
    MODE_NONE:                (160, 160, 160),
}
TAG_COLORS = {"BLIND": (60, 60, 255), "MERGE": (60, 60, 255),
              "SPAN": (60, 60, 255)}          # everything else amber
GATE_SHORT = {"confidence": "conf", "proximity": "prox", "length_px": "len",
              "width_px": "wid", "mean_intensity": "int", "stop_line": "on stop line"}

HEADER_H = 52                                 # px at scale 1, over the frame's top rows
C_LEFT   = (255, 255, 0)                      # cyan
C_RIGHT  = (255, 0, 255)                      # magenta
C_LANE   = (0, 255, 255)                      # yellow

C_GRAY, C_WHITE, C_USABLE = dv.C_GRAY, dv.C_WHITE, dv.C_USABLE
C_RED, C_AMBER, FONT = dv.C_RED, dv.C_AMBER, dv.FONT
_text = dv.draw_text

LANE_CSV_FIELDS = ("frame_id", "timestamp_ms", "mode", "offset", "left_x",
                   "right_x", "lane_width_px", "confidence", "raw_count",
                   "usable_count") + LOG_TAGS

_TAG_RE = re.compile(r"^\[([A-Z-]+)\]")


def log_tags(log: list[str]) -> Counter:
    return Counter(m.group(1) for m in map(_TAG_RE.match, log) if m)


def load_source_frame(path: str, lane_rect: tuple[int, int, int, int],
                      frame_size: tuple[int, int] = FRAME_SIZE) -> np.ndarray | None:
    """
    A saved frame as BGR at frame_size, or None if unreadable.

    A lane-ROI crop is placed back at lane_rect on black, so one offset fits both
    kinds of dump; other sizes are resized.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    x, y, w, h = lane_rect
    if img.shape[:2] == (h, w):
        canvas = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)
        canvas[y:y + h, x:x + w] = img
        return canvas
    if (img.shape[1], img.shape[0]) != frame_size:
        img = cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)
    return img


def _blank(lane_rect, frame_size):
    img = np.full((frame_size[1], frame_size[0], 3), 25, np.uint8)
    x, y, w, h = lane_rect
    img[y:y + h, x:x + w] = 55
    return img


def _csv_row(result, dbg):
    tags = log_tags(dbg.get("log", ()))
    opt = lambda v: "" if v is None else v
    return [
        result.frame_id, result.timestamp_ms, result.mode,
        result.offset, opt(result.left_x), opt(result.right_x),
        opt(result.lane_width_px), result.confidence,
        dbg.get("raw_count", ""), result.boundary_count,
        *(tags.get(t, 0) for t in LOG_TAGS),
    ]


def candidate_gates(geometry, config) -> list[tuple[tuple[int, int, int, int], str | None]]:
    """
    [(bbox, gate)] per raw candidate, gate None when it passed; stop-line members get "stop_line".

    Re-runs lane_offset's _usable(), since a passing candidate logs nothing and the
    log can't be aligned back. config must be the chain's, or the gates differ.
    """
    # Private to lane_offset; imported here so nothing else depends on it
    from src.perception.lane_offset import _on_stop_line, _usable

    pairs = []
    stop_lines = getattr(geometry, "stop_line_candidates", [])
    for cand in geometry.lane_candidates:
        if _on_stop_line(cand, stop_lines, config.stop_line_overlap) is not None:
            pairs.append((cand.bbox, "stop_line"))
            continue
        scratch = []
        ok = _usable(cand, config, scratch)
        gate = None
        if not ok and scratch:
            parts = scratch[-1].split()
            gate = parts[1] if len(parts) > 1 else "rejected"
        pairs.append((cand.bbox, gate))
    return pairs


def annotate(frame: np.ndarray | None, result, dbg: dict,
             lane_rect: tuple[int, int, int, int], candidates=(), scale: int = 1,
             frame_size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """
    Annotated copy at frame_size * scale; frame None draws on a blank canvas.

    lane_rect shifts lane-ROI coordinates into the frame.
    """
    s = max(1, int(scale))
    base = _blank(lane_rect, frame_size) if frame is None else frame
    img = (cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
           if s > 1 else base.copy())
    fs, th = 0.38 * s, max(1, s // 2)
    rx, ry, rw, rh = lane_rect

    def P(x, y):
        return int(round((rx + x) * s)), int(round((ry + y) * s))

    top, bot = P(0, 0)[1], P(0, rh - 1)[1]
    cv2.rectangle(img, P(0, 0), P(rw - 1, rh - 1), C_GRAY, 1)

    for bbox, gate in candidates:
        bx, by, bw, bh = bbox
        color = C_USABLE if gate is None else C_RED
        cv2.rectangle(img, P(bx, by), P(bx + bw - 1, by + bh - 1), color, 1)
        if gate is not None:
            lx, ly = P(bx, by)
            _text(img, GATE_SHORT.get(gate, gate), (lx, ly - 3 * s),
                  C_RED, fs * 0.9, th)

    for ax, _w in dbg.get("anchors", ()):
        x0, _ = P(ax, 0)
        cv2.line(img, (x0, bot), (x0, bot - 10 * s), C_USABLE, th + 1)

    cx = rw / 2.0
    cxp = P(cx, 0)[0]
    cv2.line(img, (cxp, top), (cxp, bot), C_GRAY, 1)
    if result.mode in STEERING_MODES:
        # offset = (cx - lane_center) / cx  ->  lane_center = cx * (1 - offset)
        clamped = abs(result.offset) >= 1.0
        lc = P(cx * (1.0 - result.offset), 0)[0]
        ym = (top + bot) // 2
        color = C_RED if clamped else C_LANE
        cv2.line(img, (lc, top), (lc, bot), color, 1)
        cv2.circle(img, (lc, ym), 4 * s, color, -1)
        cv2.arrowedLine(img, (cxp, ym), (lc, ym), color, th, tipLength=0.15)

    for val, color, tag in ((result.left_x, C_LEFT, "L"),
                            (result.right_x, C_RIGHT, "R")):
        if val is None:
            continue
        xp = P(val, 0)[0]
        cv2.line(img, (xp, top), (xp, bot), color, th + 1)
        _text(img, f"{tag}{val:.0f}", (xp + 3 * s, top + 12 * s), color, fs, th)

    W = img.shape[1]
    hh = HEADER_H * s
    cv2.rectangle(img, (0, 0), (W - 1, hh), (0, 0, 0), -1)
    lh = 16 * s

    _text(img, f"#{result.frame_id}  {result.mode}", (6 * s, lh - 3 * s),
          MODE_COLORS.get(result.mode, C_WHITE), fs * 1.15, th)

    off_s = (f"offset {result.offset:+.3f}" if result.mode in STEERING_MODES
             else "offset  --")
    (tw, _), _ = cv2.getTextSize(off_s, FONT, fs * 1.15, th)
    _text(img, off_s, (W - tw - 6 * s, lh - 3 * s), C_WHITE, fs * 1.15, th)

    # offset gauge, right side, -1 .. +1
    gx0, gx1, gy = W - 130 * s, W - 8 * s, 2 * lh - 6 * s
    dv.draw_offset_axis(img, gx0, gx1, gy, 4 * s)
    if result.mode in STEERING_MODES:
        mx = dv.offset_x(result.offset, gx0, gx1)
        cv2.circle(img, (mx, gy), 4 * s,
                   C_RED if abs(result.offset) >= 1.0 else C_LANE, -1)

    stats = f"raw {dbg.get('raw_count', 0)}  usable {result.boundary_count}"
    if result.lane_width_px is not None:
        stats += f"  width {result.lane_width_px:.0f}px"
    stats += f"  conf {result.confidence:.2f}"
    _text(img, stats, (6 * s, 2 * lh - 3 * s), C_WHITE, fs, th)

    x = 6 * s
    for tag, n in log_tags(dbg.get("log", ())).items():
        label = f"{tag}x{n}" if n > 1 else tag
        _text(img, label, (x, 3 * lh - 3 * s), TAG_COLORS.get(tag, C_AMBER),
              fs, th)
        x += cv2.getTextSize(label, FONT, fs, th)[0][0] + 10 * s

    return img


class LaneView:
    """Lane view for debug_video. lane_config None skips candidate boxes; the decision still draws."""
    name = "lane"
    CSV_FIELDS = LANE_CSV_FIELDS

    def __init__(self, lane_config=None):
        self.lane_config = lane_config
        self._frames = 0
        self._accepted = 0
        self._usable = 0
        self._modes = {}
        self._gates = {}
        self._offsets = []
        self._blind_run = 0
        self._cur_blind = 0

    def extract(self, chain, frame=None):
        gates = (candidate_gates(chain.geometry, self.lane_config)
                 if self.lane_config is not None else [])
        size = FRAME_SIZE if frame is None else (frame.shape[1], frame.shape[0])
        return {
            "frame": frame,
            "frame_size": size,
            "result": chain.offset,
            "dbg": chain.offset_debug,
            "lane_rect": chain.roi.lane_rect,
            "gates": gates,
        }

    def observe(self, data):
        result, dbg = data["result"], data["dbg"]
        self._frames += 1
        self._modes[result.mode] = self._modes.get(result.mode, 0) + 1
        self._usable += result.boundary_count
        self._accepted += dbg.get("raw_count", 0)

        for _bbox, gate in data["gates"]:
            if gate:
                self._gates[gate] = self._gates.get(gate, 0) + 1
        for entry in dbg.get("log", ()):
            for tag in ("MERGE", "SPAN"):
                if entry.startswith(f"[{tag}]"):
                    self._gates[tag] = self._gates.get(tag, 0) + 1

        if result.mode in STEERING_MODES:
            self._offsets.append(result.offset)
            self._cur_blind = 0
        else:
            self._cur_blind += 1
            self._blind_run = max(self._blind_run, self._cur_blind)

    def report(self):
        n = max(self._frames, 1)
        out = [f"[LANE] {self._frames} frames"]
        out.append(f"candidates accepted     {self._accepted}  "
                   f"({self._accepted/n:.2f} per frame)")
        out.append(f"usable as boundaries    {self._usable}  "
                   f"({self._usable/n:.2f} per frame, need 2.00)")
        if self._accepted:
            out.append(f"survival rate           "
                       f"{100*self._usable/self._accepted:.1f}%")

        out.append("")
        out.append(f"[MODES] {self._frames} frames:")
        for mode in (*STEERING_MODES, MODE_SINGLE_UNCALIBRATED, MODE_NONE):
            c = self._modes.get(mode, 0)
            flag = "   <-- never" if c == 0 else ""
            out.append(f" {mode:<22}{c:6}  ({100*c/n:5.1f}%){flag}")

        steering = sum(self._modes.get(m, 0) for m in STEERING_MODES)
        out.append("")
        out.append(f"[AVAILABILITY] {steering}/{self._frames} frames produced a "
                   f"steering signal ({100*steering/n:.1f}%)")
        out.append(f"[BLIND] longest run without one: {self._blind_run} frames")

        if self._gates:
            out.append("")
            out.append("[LOSSES] why a candidate or a pair was not used:")
            for gate, c in sorted(self._gates.items(), key=lambda kv: -kv[1]):
                out.append(f" {gate:<22}{c:6}")

        if self._offsets:
            s = sorted(self._offsets)
            out.append("")
            out.append(f"[OFFSETS] min {s[0]:+.3f}  med {s[len(s)//2]:+.3f}  "
                       f"max {s[-1]:+.3f}")
        return out

    def render(self, data, scale=1):
        return annotate(data["frame"], data["result"], data["dbg"],
                        data["lane_rect"], data["gates"], scale,
                        data["frame_size"])

    def row(self, data):
        return _csv_row(data["result"], data["dbg"])


class DebugVideoWriter(dv.ViewWriter):
    """
    The lane view's video and CSV for a loop with stage outputs but no chain result.

    csv_path "auto" puts it beside the video; stride keeps every Nth frame.
    """
    CSV_FIELDS = LANE_CSV_FIELDS

    def __init__(self, path, fps=dv.DEFAULT_FPS, fourcc=dv.FOURCC,
                 csv_path="auto", stride=1):
        super().__init__(path, self.CSV_FIELDS, fps, fourcc, csv_path, stride)

    def write(self, frame, result, dbg, lane_rect, candidates=(), scale=1,
              frame_size=FRAME_SIZE):
        """Annotate and write one frame; the image, or None if skipped."""
        if not self.take():
            return None

        img = annotate(frame, result, dbg, lane_rect, candidates, scale,
                       frame_size)
        row = _csv_row(result, dbg) if self.csv_path else None
        return self.push(img, row)