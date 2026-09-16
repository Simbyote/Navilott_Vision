"""
debug_video.py -- Debug video logging for lane_offset.py decisions

Draws what compute_lane_offset() decided onto the frame it decided it on and
writes the result to disk, so anchor selection and gating can be reviewed
frame-by-frame after a run with no monitor on the robot.

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

Nothing here imports lane_offset.py, so run_pipeline.py can use the same
writer in the live loop.
"""
import csv
import os
import re
from collections import Counter

import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
FOURCC       = "MJPG"          # cheapest OpenCV encoder on ARM; use .avi
DEFAULT_FPS  = 20.0
FRAME_SIZE   = (480, 360)      # (w, h) the lane_rect coordinates assume
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

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

C_GRAY   = (140, 140, 140)
C_WHITE  = (235, 235, 235)
C_USABLE = (90, 255, 90)                    # green; white vanishes on tape
C_LEFT   = (255, 255, 0)                      # cyan
C_RIGHT  = (255, 0, 255)                      # magenta
C_LANE   = (0, 255, 255)                      # yellow
C_RED    = (60, 60, 255)
C_AMBER  = (0, 190, 255)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

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

def _text(img, s, org, color, fs, th=1):
    # Black outline first so labels stay readable over white tape
    cv2.putText(img, s, org, FONT, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, fs, color, th, cv2.LINE_AA)

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
# Writer
# =============================================================================
class DebugVideoWriter:
    """
    Opens on the first frame so the output size comes from real data.

    csv_path: "auto" -> video path with .csv; None disables the sidecar
    stride:   keep every Nth frame, for CPU relief on the robot. The overlay
              frame id is the real one, so skipped frames show as gaps
    """
    CSV_FIELDS = ("frame_id", "timestamp_ms", "mode", "offset", "left_x",
                  "right_x", "lane_width_px", "confidence", "raw_count",
                  "usable_count") + LOG_TAGS

    def __init__(self, path, fps=DEFAULT_FPS, fourcc=FOURCC,
                 csv_path="auto", stride=1):
        self.path = path
        self.fps = fps
        self.fourcc = fourcc
        self.stride = max(1, int(stride))
        self.csv_path = (os.path.splitext(path)[0] + ".csv"
                         if csv_path == "auto" else csv_path)
        self.frames_written = 0
        self._seen = 0
        self._vw = None
        self._size = None
        self._csv_f = None
        self._csv = None

    def _open(self, w, h):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._vw = cv2.VideoWriter(self.path,
                                   cv2.VideoWriter_fourcc(*self.fourcc),
                                   self.fps, (w, h))
        # A missing codec otherwise fails silently and writes nothing
        if not self._vw.isOpened():
            raise RuntimeError(f"VideoWriter could not open {self.path} "
                               f"with fourcc {self.fourcc}")
        self._size = (w, h)
        if self.csv_path:
            self._csv_f = open(self.csv_path, "w", newline="")
            self._csv = csv.writer(self._csv_f)
            self._csv.writerow(self.CSV_FIELDS)

    def write(self, frame, result, dbg, lane_rect, candidates=(), scale=1,
              frame_size=FRAME_SIZE):
        """Annotate and write one frame. Returns the image, or None if skipped."""
        self._seen += 1
        if (self._seen - 1) % self.stride:
            return None

        img = annotate(frame, result, dbg, lane_rect, candidates, scale,
                       frame_size)
        h, w = img.shape[:2]
        if self._vw is None:
            self._open(w, h)
        if (w, h) != self._size:
            # VideoWriter drops mismatched frames without an error
            img = cv2.resize(img, self._size, interpolation=cv2.INTER_AREA)
        self._vw.write(img)
        self.frames_written += 1

        if self._csv:
            tags = log_tags(dbg.get("log", ()))
            opt = lambda v: "" if v is None else v
            self._csv.writerow([
                result.frame_id, result.timestamp_ms, result.mode,
                result.offset, opt(result.left_x), opt(result.right_x),
                opt(result.lane_width_px), result.confidence,
                dbg.get("raw_count", ""), result.boundary_count,
                *(tags.get(t, 0) for t in LOG_TAGS),
            ])
        return img

    def close(self):
        if self._vw is not None:
            self._vw.release()
            self._vw = None
        if self._csv_f is not None:
            self._csv_f.close()
            self._csv_f = self._csv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

# =============================================================================
# Self-test
#
#   python3 debug_video.py
# =============================================================================
if __name__ == "__main__":
    import sys
    from types import SimpleNamespace

    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results")
    out = os.path.join(RESULTS_DIR, "debug_video_selftest.avi")
    LANE_RECT = (24, 252, 432, 108)
    N = 30
    failed = 0

    with DebugVideoWriter(out) as vw:
        for i in range(N):
            res = SimpleNamespace(
                offset=round((i - N / 2) / (N / 2.5), 4),
                left_x=150.0 + i, right_x=300.0 + i, lane_width_px=150.0,
                confidence=0.6, boundary_count=2, mode="two_boundary",
                frame_id=i, timestamp_ms=i * 50)
            dbg = {"raw_count": 3, "anchors": [(150.0 + i, 0.6), (300.0 + i, 0.6)],
                   "log": ["[REJECT] confidence 0.1 < 0.3"]}
            cands = [((148 + i, 10, 5, 95), None), ((298 + i, 10, 5, 95), None),
                     ((60, 30, 20, 20), "confidence")]
            vw.write(None, res, dbg, LANE_RECT, cands, scale=2)

    cap = cv2.VideoCapture(out)
    got = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    ok = got == N
    failed += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  video frame count {got}/{N}")

    with open(os.path.splitext(out)[0] + ".csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ok = len(rows) == N and all(r["REJECT"] == "1" for r in rows)
    failed += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  sidecar rows {len(rows)}/{N}, tags counted")

    sys.exit(1 if failed else 0)