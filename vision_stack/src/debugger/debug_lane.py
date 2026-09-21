"""
debug_lane.py -- Lane view for the debug tooling

Draws what compute_lane_offset() decided onto the frame it decided it on and
writes the result to disk, so anchor selection and gating can be reviewed
frame-by-frame after a run with no monitor on the robot.

Two ways in, both built on the same drawing:
    LaneView         the view live_view runs (extract / observe / render / row
                     / report), the same interface as debug_stop and
                     debug_traffic
    DebugVideoWriter annotate() plus the decision CSV, for a loop that has the
                     stage outputs in hand and no chain result (run_pipeline.py)

Inputs are the stage's own outputs, unchanged:
    result      LaneOffsetResult
    dbg         debug_summary dict ("log", "anchors", "raw_count", ...)
    lane_rect   roi.lane_rect, to shift lane-ROI x/y into frame coordinates
    candidates  [(bbox, gate)] for every raw candidate; gate is None when
                the candidate passed, else the gate name from its [REJECT]

Overlay:
    header      frame id, mode, offset + gauge, counts, decision tags
    ROI box     lane_rect outline
    candidates  green box = usable, red box + gate = rejected
    anchors     short green tick at the foot of every usable anchor
    boundaries  selected left_x (cyan) / right_x (magenta), full ROI height
    centers     ROI center = robot (gray), lane center implied by the
                offset (yellow; red when the offset is clamped at +/-1)

Importing this module does not import lane_offset.py. Only candidate_gates()
does, at its first call, so annotate() and DebugVideoWriter work anywhere the
stage outputs do, and no import cycle with the stage is possible.
"""
import re
from collections import Counter

import cv2
import numpy as np

import src.debugger.debug_video as dv

# =============================================================================
# Configuration
# =============================================================================
FRAME_SIZE   = (480, 360)      # (w, h) the lane_rect coordinates assume

STEERING_MODES = ("two_boundary", "left_only", "right_only")
LOG_TAGS = ("BLIND", "MERGE", "SPAN", "ONE-SIDED", "UNCALIBRATED", "REJECT")

MODE_COLORS = {                               # BGR
    "two_boundary":        (80, 210, 80),
    "left_only":           (0, 200, 255),
    "right_only":          (0, 200, 255),
    "single_uncalibrated": (0, 120, 255),
    "none":                (160, 160, 160),
}
TAG_COLORS = {"BLIND": (60, 60, 255), "MERGE": (60, 60, 255),
              "SPAN": (60, 60, 255)}          # everything else amber
GATE_SHORT = {"confidence": "conf", "proximity": "prox", "length_px": "len",
              "width_px": "wid", "mean_intensity": "int"}

C_LEFT   = (255, 255, 0)                      # cyan
C_RIGHT  = (255, 0, 255)                      # magenta
C_LANE   = (0, 255, 255)                      # yellow

# Shared palette and label helper, from the generic module
C_GRAY, C_WHITE, C_USABLE = dv.C_GRAY, dv.C_WHITE, dv.C_USABLE
C_RED, C_AMBER, FONT = dv.C_RED, dv.C_AMBER, dv.FONT
_text = dv.draw_text

LANE_CSV_FIELDS = ("frame_id", "timestamp_ms", "mode", "offset", "left_x",
                   "right_x", "lane_width_px", "confidence", "raw_count",
                   "usable_count") + LOG_TAGS

_TAG_RE = re.compile(r"^\[([A-Z-]+)\]")

# =============================================================================
# Helpers
# =============================================================================
def log_tags(log):
    """Counter of the [TAG] that opens each debug log entry."""
    return Counter(m.group(1) for m in map(_TAG_RE.match, log) if m)

def load_source_frame(path, lane_rect, frame_size=FRAME_SIZE):
    """
    Read a saved frame as a BGR image at frame_size.

    A saved lane-ROI crop (lane_rect sized) is placed back at lane_rect on a
    black canvas, so the same lane_rect offset works for both kinds of dump.
    Returns None if the file cannot be read.
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
    """One decision-log row, in LANE_CSV_FIELDS order."""
    tags = log_tags(dbg.get("log", ()))
    opt = lambda v: "" if v is None else v
    return [
        result.frame_id, result.timestamp_ms, result.mode,
        result.offset, opt(result.left_x), opt(result.right_x),
        opt(result.lane_width_px), result.confidence,
        dbg.get("raw_count", ""), result.boundary_count,
        *(tags.get(t, 0) for t in LOG_TAGS),
    ]

def candidate_gates(geometry, config):
    """
    Purpose:
        Pair every raw lane candidate with the gate that rejected it, or None
        if it passed, in the form annotate() expects

    Inputs:
        geometry: GeometryBranchResult
        config: the LaneOffsetConfig the chain ran with

    Notes:
        Re-runs lane_offset's _usable() per candidate rather than parsing the
        debug log, because a passing candidate logs nothing and the entries
        cannot be aligned back to their candidates by position.

        _usable is private to lane_offset. It is imported here, at the first
        call, so that nothing else in this module depends on lane_offset. A
        public classify function in lane_offset would remove the need.
    """
    from src.perception.lane_offset import _usable

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
# Overlay
# =============================================================================
def annotate(frame, result, dbg, lane_rect, candidates=(), scale=1,
             frame_size=FRAME_SIZE):
    """
    Return an annotated copy of frame at frame_size * scale.

    frame None draws on a blank canvas, so candidate geometry and decisions
    can still be reviewed without source images.
    """
    s = max(1, int(scale))
    base = _blank(lane_rect, frame_size) if frame is None else frame
    img = (cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
           if s > 1 else base.copy())
    fs, th = 0.38 * s, max(1, s // 2)
    rx, ry, rw, rh = lane_rect

    def P(x, y):
        """lane-ROI coordinates -> output pixel"""
        return int(round((rx + x) * s)), int(round((ry + y) * s))

    top, bot = P(0, 0)[1], P(0, rh - 1)[1]
    cv2.rectangle(img, P(0, 0), P(rw - 1, rh - 1), C_GRAY, 1)

    # --- raw candidates ------------------------------------------------------
    for bbox, gate in candidates:
        bx, by, bw, bh = bbox
        color = C_USABLE if gate is None else C_RED
        cv2.rectangle(img, P(bx, by), P(bx + bw - 1, by + bh - 1), color, 1)
        if gate is not None:
            lx, ly = P(bx, by)
            _text(img, GATE_SHORT.get(gate, gate), (lx, ly - 3 * s),
                  C_RED, fs * 0.9, th)

    # --- usable anchor feet --------------------------------------------------
    for ax, _w in dbg.get("anchors", ()):
        x0, _ = P(ax, 0)
        cv2.line(img, (x0, bot), (x0, bot - 10 * s), C_USABLE, th + 1)

    # --- robot (ROI center) and implied lane center --------------------------
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

    # --- selected boundaries -------------------------------------------------
    for val, color, tag in ((result.left_x, C_LEFT, "L"),
                            (result.right_x, C_RIGHT, "R")):
        if val is None:
            continue
        xp = P(val, 0)[0]
        cv2.line(img, (xp, top), (xp, bot), color, th + 1)
        _text(img, f"{tag}{val:.0f}", (xp + 3 * s, top + 12 * s), color, fs, th)

    # --- header --------------------------------------------------------------
    W = img.shape[1]
    hh = 52 * s
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
    cv2.line(img, (gx0, gy), (gx1, gy), C_GRAY, 1)
    for v in (-1.0, 0.0, 1.0):
        tx = int(gx0 + (v + 1) / 2 * (gx1 - gx0))
        cv2.line(img, (tx, gy - 4 * s), (tx, gy + 4 * s), C_GRAY, 1)
    if result.mode in STEERING_MODES:
        v = max(-1.0, min(1.0, result.offset))
        mx = int(gx0 + (v + 1) / 2 * (gx1 - gx0))
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

# =============================================================================
# View
# =============================================================================
class LaneView:
    """
    Lane view. Implements the view interface live_view runs:

        name, CSV_FIELDS
        extract(chain, frame) -> data   pull this target's data out of a chain
                                        result; the lane view is the one that
                                        draws on the source frame
        observe(data)                   accumulate run statistics
        render(data, scale) -> img      the annotated picture
        row(data) -> list               one CSV row
        report() -> [str]               lines for summary.txt

    lane_config: the LaneOffsetConfig the chain ran with, used to say which
                 gate rejected each candidate. None draws no candidate boxes
                 (the decision, anchors and boundaries still draw)
    """
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

    # -- data -----------------------------------------------------------------
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

    # -- statistics -----------------------------------------------------------
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
        for mode in ("two_boundary", "left_only", "right_only",
                     "single_uncalibrated", "none"):
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

    # -- picture and row --------------------------------------------------------
    def render(self, data, scale=1):
        return annotate(data["frame"], data["result"], data["dbg"],
                        data["lane_rect"], data["gates"], scale,
                        data["frame_size"])

    def row(self, data):
        return _csv_row(data["result"], data["dbg"])

# =============================================================================
# Writer
# =============================================================================
class DebugVideoWriter(dv.ViewWriter):
    """
    The lane view's video and CSV, for a loop that has the stage outputs and
    no chain result. live_view does not use this: it runs LaneView through
    ViewWriter like every other view.

    csv_path: "auto" -> video path with .csv; None disables the sidecar
    stride:   keep every Nth frame, for CPU relief on the robot
    """
    CSV_FIELDS = LANE_CSV_FIELDS

    def __init__(self, path, fps=dv.DEFAULT_FPS, fourcc=dv.FOURCC,
                 csv_path="auto", stride=1):
        super().__init__(path, self.CSV_FIELDS, fps, fourcc, csv_path, stride)

    def write(self, frame, result, dbg, lane_rect, candidates=(), scale=1,
              frame_size=FRAME_SIZE):
        """Annotate and write one frame. Returns the image, or None if skipped."""
        if not self.take():
            return None

        img = annotate(frame, result, dbg, lane_rect, candidates, scale,
                       frame_size)
        row = _csv_row(result, dbg) if self.csv_path else None
        return self.push(img, row)