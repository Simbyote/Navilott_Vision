"""Shared recording, drawing and interface pieces for the debug views.

Purpose:
    Everything view-agnostic. Each view (debug_lane on the source frame,
    debug_stop on the sign ROI, debug_traffic on the traffic ROI) records
    through ViewWriter and draws with draw_text and the shared palette.
    Nothing here imports a pipeline stage or another view.

Main package:
    The view interface. live_view runs any object that has:

        name                        short id, used in file names and the window
        CSV_FIELDS                  header row of the view's CSV
        extract(chain, frame=None)  pull this view's data out of a chain result;
                                    frame is the source frame, ignored by views
                                    that draw on their own ROI
        observe(data)               accumulate run statistics, every frame
        render(data, scale) -> img  the picture
        row(data) -> list           one CSV row
        report() -> [str]           lines for summary.txt

Flow:
    Per frame, live_view calls extract() and observe() on every view, then
    ViewWriter.take(); kept frames go through render() and row() into
    ViewWriter.push().
"""
import csv
import os

import cv2
import numpy as np

from src.params import FPS

FOURCC       = "MJPG"          # cheapest OpenCV encoder on ARM; use .avi
DEFAULT_FPS  = float(FPS)

# BGR
C_GRAY   = (140, 140, 140)
C_WHITE  = (235, 235, 235)
C_USABLE = (90, 255, 90)                    # green; white vanishes on tape
C_RED    = (60, 60, 255)
C_AMBER  = (0, 190, 255)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# pass: accepted and at or above conf_threshold (if set); low: accepted but below it,
# so the threshold is what hides it; reject: failed a gate, labeled with the gate and value
STATE_COLORS = {"pass": C_USABLE, "low": C_AMBER, "reject": C_RED}
REPORT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)    # summary.txt: share of frames passing at each
GAP_PX = 6                                            # between panels, before zoom

def draw_text(img: np.ndarray, s: str, org: tuple[int, int], color: tuple[int, int, int],
              fs: float, th: int = 1) -> None:
    """Label with a black outline underneath, so it stays readable over white tape."""
    cv2.putText(img, s, org, FONT, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, fs, color, th, cv2.LINE_AA)

_text = draw_text

class CandidateView:
    """
    Base for views that grade a detector's candidates against a downstream confidence threshold.

    A subclass provides _entries(data): one dict per candidate the detector
    looked at, each with at least gate (None when accepted) and confidence.
    The pass / low / reject grading, the per-frame summary, the render sizing
    and the threshold report are shared. debug_stop and debug_traffic build on it.

    conf_threshold: The confidence a candidate has to reach downstream. None
        draws no threshold and no amber state.
    zoom: Extra magnification on top of the run's scale; the ROIs are small.
    """
    def __init__(self, conf_threshold: float | None = None, zoom: int = 2) -> None:
        self.conf_threshold = conf_threshold
        self.zoom = max(1, int(zoom))

    def _entries(self, data: dict) -> list[dict]:
        raise NotImplementedError

    def _state(self, e: dict) -> str:
        """pass, low or reject; see STATE_COLORS."""
        if e["gate"] is not None:
            return "reject"
        thr = self.conf_threshold
        if thr is not None and (e["confidence"] or 0.0) < thr:
            return "low"
        return "pass"

    def _summary(self, data: dict) -> dict:
        """This frame's entries, best accepted entry, and pass / low / reject counts."""
        entries = self._entries(data)
        accepted = [e for e in entries if e["gate"] is None]
        states = [self._state(e) for e in entries]
        best = max(accepted, key=lambda e: e["confidence"] or 0.0, default=None)
        return {
            "entries": entries,
            "best": best,
            "passed": states.count("pass"),
            "low": states.count("low"),
            "rejected": states.count("reject"),
        }

    def _metrics(self, scale: int) -> tuple[int, float, int, int, int, int]:
        """(s, fs, th, lh, hh, fh): pixel scale, font scale, thickness, line, header and footer heights."""
        s = max(1, int(scale)) * self.zoom
        fs = max(0.34, 0.18 * s)
        th = 1 if s < 4 else 2
        lh = int(38 * fs)
        hh, fh = 2 * lh + 8, lh + 6
        return s, fs, th, lh, hh, fh

    def _header(self, sm: dict) -> tuple[tuple[int, int, int], str]:
        """Title color (green if any passed, amber if only low, else white) and the counts text."""
        color = (C_USABLE if sm["passed"]
                 else C_AMBER if sm["low"] else C_WHITE)
        counts = f"pass {sm['passed']}"
        if self.conf_threshold is not None:
            counts += f"  low {sm['low']}"
        counts += f"  rejected {sm['rejected']}"
        return color, counts

    def _threshold_report(self, best_conf: list[float], n: int) -> list[str]:
        """summary.txt lines: best confidence per frame, and the share of frames passing at each REPORT_THRESHOLDS."""
        if not best_conf:
            return []
        s = sorted(best_conf)
        return [
            f" best confidence per frame: min {s[0]:.3f}  "
            f"med {s[len(s) // 2]:.3f}  max {s[-1]:.3f}",
            " frames that would pass at a threshold of:",
            "  " + "   ".join(
                f"{t:.2f} -> {100 * sum(1 for c in s if c >= t) / n:.0f}%"
                for t in REPORT_THRESHOLDS),
        ]


class ViewWriter:
    """
    Recorder for one debug view: an .avi and an optional CSV sidecar.

    Opens on the first frame so the output size comes from real data. Call
    take() first, so a frame the stride skips costs no rendering, then push()
    the finished image and its CSV row.

    csv_fields: Header row for the sidecar; empty disables it.
    csv_path: "auto" puts it beside the video with a .csv extension; None disables it.
    stride: Keep every Nth frame, for CPU relief on the robot. The overlay
        frame id is the real one, so skipped frames show as gaps.
    """
    def __init__(self, path: str, csv_fields=(), fps: float = DEFAULT_FPS,
                 fourcc: str = FOURCC, csv_path: str | None = "auto", stride: int = 1) -> None:
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

    def take(self) -> bool:
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

    def push(self, img: np.ndarray, row: list | None = None) -> np.ndarray:
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