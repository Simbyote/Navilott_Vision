"""
debug_video.py -- Shared pieces of the debug views

Everything here is view-agnostic. Each view lives in its own module and uses
these to record and to draw:

    debug_lane.py      lane_offset decisions, on the source frame
    debug_stop.py      stop-sign detector, on the sign ROI
    debug_traffic.py   color branch, on the traffic ROI

Provides:
    ViewWriter    an .avi and an optional CSV sidecar for one view
    draw_text     outlined label, readable over white tape
    palette       C_GRAY, C_WHITE, C_USABLE, C_RED, C_AMBER (BGR), and FONT
    FOURCC, DEFAULT_FPS

The view interface (live_view runs any object that has these):

    name                        short id, used in file names and the window
    CSV_FIELDS                  header row of the view's CSV
    extract(chain, frame=None)  pull this view's data out of a chain result.
                                frame is the source frame; views that draw on
                                their own ROI ignore it
    observe(data)               accumulate run statistics, every frame
    render(data, scale) -> img  the picture
    row(data) -> list           one CSV row
    report() -> [str]           lines for summary.txt

Nothing here imports a pipeline stage or another view.

Names that used to live here (annotate, DebugVideoWriter, load_source_frame,
log_tags and the lane constants) moved to debug_lane.py. Importing them from
this module still works and warns; update the import and drop the shim at the
bottom once nothing uses it.
"""
import csv
import os

import cv2

# =============================================================================
# Configuration
# =============================================================================
FOURCC       = "MJPG"          # cheapest OpenCV encoder on ARM; use .avi
DEFAULT_FPS  = 20.0

C_GRAY   = (140, 140, 140)
C_WHITE  = (235, 235, 235)
C_USABLE = (90, 255, 90)                    # green; white vanishes on tape
C_RED    = (60, 60, 255)
C_AMBER  = (0, 190, 255)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# =============================================================================
# Helpers
# =============================================================================
def draw_text(img, s, org, color, fs, th=1):
    """Outlined label, shared by every debug view."""
    # Black outline first so labels stay readable over white tape
    cv2.putText(img, s, org, FONT, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, fs, color, th, cv2.LINE_AA)

_text = draw_text

# =============================================================================
# Writer
# =============================================================================
class ViewWriter:
    """
    Recorder for one debug view: an .avi and an optional CSV sidecar.

    Opens on the first frame so the output size comes from real data. Call
    take() first, so a frame the stride skips costs no rendering, then push()
    the finished image and its CSV row.

    csv_fields: header row for the sidecar; empty disables it
    csv_path:   "auto" -> video path with .csv; None disables the sidecar
    stride:     keep every Nth frame, for CPU relief on the robot. The overlay
                frame id is the real one, so skipped frames show as gaps
    """
    def __init__(self, path, csv_fields=(), fps=DEFAULT_FPS, fourcc=FOURCC,
                 csv_path="auto", stride=1):
        self.path = path
        self.fps = fps
        self.fourcc = fourcc
        self.stride = max(1, int(stride))
        self.csv_fields = tuple(csv_fields)
        if not self.csv_fields:
            self.csv_path = None
        else:
            self.csv_path = (os.path.splitext(path)[0] + ".csv"
                             if csv_path == "auto" else csv_path)
        self.frames_written = 0
        self._seen = 0
        self._vw = None
        self._size = None
        self._csv_f = None
        self._csv = None

    def take(self):
        """Count one incoming frame. True if the stride keeps it."""
        self._seen += 1
        return (self._seen - 1) % self.stride == 0

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
            self._csv.writerow(self.csv_fields)

    def push(self, img, row=None):
        """Write one finished image and, if given, its CSV row. Returns the image."""
        h, w = img.shape[:2]
        if self._vw is None:
            self._open(w, h)
        if (w, h) != self._size:
            # VideoWriter drops mismatched frames without an error
            img = cv2.resize(img, self._size, interpolation=cv2.INTER_AREA)
        self._vw.write(img)
        self.frames_written += 1
        if self._csv and row is not None:
            self._csv.writerow(row)
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
# Compatibility shim: names that moved to debug_lane.py
# =============================================================================
_MOVED_TO_DEBUG_LANE = frozenset((
    "annotate", "DebugVideoWriter", "load_source_frame", "log_tags",
    "FRAME_SIZE", "STEERING_MODES", "LOG_TAGS", "MODE_COLORS", "TAG_COLORS",
    "GATE_SHORT", "C_LEFT", "C_RIGHT", "C_LANE",
))

def __getattr__(name):
    """Forward the moved names, with a warning, so old imports keep working."""
    if name in _MOVED_TO_DEBUG_LANE:
        import warnings
        import vision_stack.src.debugger.debug_lane as debug_lane
        warnings.warn(f"debug_video.{name} moved to debug_lane.{name}",
                      DeprecationWarning, stacklevel=2)
        return getattr(debug_lane, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")