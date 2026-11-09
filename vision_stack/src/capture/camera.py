"""Camera acquisition and per-frame identity for Phase 1.

Purpose:
    Owns the libcamera / GStreamer pipeline and is the one place frame ids
    and timestamps are minted. The ISP hands over BGR; the frame is turned
    upright here rather than in the pipeline (phase1_capture.md).

Main package:
    FrameData: one upright BGR frame, its id and its monotonic arrival time.
"""
import time
import warnings
import cv2
import numpy as np
from dataclasses import dataclass

from src.params import CAMERA_CONTROLS, CAMERA_ROTATE_180, FPS, FRAME_H, FRAME_W, MAX_FPS, MIN_FPS, SENSOR_CONFIG

# The names from gst-inspect-1.0 libcamerasrc on the robot (2026-10-04, libcamera v0.7.2);
# enum values from libcamera's control definitions. A typo or a bad value fails here, by
# name, not as GStreamer's "unable to start pipeline". name: (kind, values or (lo, hi), what)
CAMERA_CONTROLS_SPEC = {
    "ae-enable": ("bool", None, "auto exposure on; for fixed exposure use the manual modes below instead"),
    "ae-exposure-mode": ("enum", ("normal", "short", "long", "custom"),
                         "how auto exposure splits shutter and gain: short favors a fast shutter (less blur)"),
    "ae-constraint-mode": ("enum", ("normal", "highlight", "shadows", "custom"),
                           "what auto exposure protects: highlight keeps bright parts (windows, LEDs) "
                           "from clipping, shadows lifts dark ones"),
    "exposure-value": ("float", (-8.0, 8.0), "auto exposure's target in stops: +1 twice as bright, -1 half"),
    "exposure-time-mode": ("enum", ("auto", "manual"), "manual holds exposure-time"),
    "exposure-time": ("int", (1, 1_000_000), "shutter in microseconds; at 20 FPS at most ~50000"),
    "analogue-gain-mode": ("enum", ("auto", "manual"), "manual holds analogue-gain"),
    "analogue-gain": ("float", (1.0, 64.0), "sensor gain, 1.0 = none; more is brighter and noisier"),
    "digital-gain": ("float", (1.0, 64.0), "gain after the sensor"),
    "ae-flicker-period": ("int", (1, 1_000_000),
                          "mains flicker to avoid, microseconds: 8333 under 60 Hz lights, 10000 under 50 Hz"),
    "awb-enable": ("bool", None, "auto white balance on; off is untried on this camera, which has no "
                   "colour-gains control to set the gains by hand"),
    "awb-mode": ("enum", ("auto", "incandescent", "tungsten", "fluorescent", "indoor", "daylight",
                          "cloudy", "custom"),
                 "the color temperatures auto white balance picks from; a preset bounds it, so window "
                 "daylight can't drag the colors as far"),
    "saturation": ("float", (0.0, 32.0), "color strength, 1.0 normal, 0 gray"),
    "sharpness": ("float", (0.0, 16.0), "edge sharpening, 1.0 normal"),
}
CAMERA_CONTROL_NAMES = frozenset(CAMERA_CONTROLS_SPEC)


def orient(frame: np.ndarray) -> np.ndarray:
    """
    Rotated 180 degrees when CAMERA_ROTATE_180, else the same array. A videoflip in the
    pipeline read the ISP's uncached buffer pixel by pixel and took a whole core (2026-10-20).
    """
    return cv2.rotate(frame, cv2.ROTATE_180) if CAMERA_ROTATE_180 else frame


def _now_ms() -> int:
    """Monotonic, so NTP steps don't move it; only differences mean anything."""
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class FrameData:
    frame: np.ndarray    # (H, W, 3) uint8 BGR, as delivered by the appsink
    frame_id: int        # failed reads don't use one
    timestamp_ms: int    # monotonic, at the appsink pull


class CaptureError(RuntimeError):
    pass


class CameraSource:
    """frame_id never repeats or skips for the life of the instance, across reopens too."""
    def __init__(
            self,
            width: int,
            height: int,
            fps: int,
            max_consecutive_failures: int | None = None,
            controls: dict | None = None,
        ) -> None:
        """
        Doesn't touch the camera until open(). max_consecutive_failures defaults to fps
        (~1 s of dead camera). ValueError for an unknown control name.
        """
        if fps < MIN_FPS:
            warnings.warn(
                f"fps={fps} is below MIN_FPS={MIN_FPS}; control updates "
                "may be too sparse for lane following",
                RuntimeWarning, stacklevel=2,
            )
        elif fps > MAX_FPS:
            warnings.warn(
                f"fps={fps} is above MAX_FPS={MAX_FPS}; the Pi Zero 2 W "
                "is unlikely to sustain it",
                RuntimeWarning, stacklevel=2,
            )
        self.width, self.height, self.fps = width, height, fps
        self.controls = dict(CAMERA_CONTROLS if controls is None else controls)
        format_controls(self.controls)
        self.max_consecutive_failures = (
            fps if max_consecutive_failures is None else max_consecutive_failures
        )
        self._cap = None
        self._consecutive_failures = 0
        # Not reset by open(), so a reopen never reissues ids already logged
        self._frame_id = 0

    def open(self) -> "CameraSource":
        """CaptureError if GStreamer can't open the pipeline."""
        pipeline = build_gst_pipeline(self.width, self.height, self.fps, controls=self.controls)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise CaptureError("Failed to open camera pipeline")
        return self

    def read(self) -> FrameData | None:
        """
        One stamped frame, or None for a failed read within the budget; None uses no
        frame_id. CaptureError before open() or once the budget is spent.
        """
        if self._cap is None:
            raise CaptureError("read() called before open()")

        ok, frame = self._cap.read()
        timestamp_ms = _now_ms()   # arrival, not packaging

        if not ok or frame is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_consecutive_failures:
                raise CaptureError(
                    f"{self._consecutive_failures} consecutive failed reads "
                    "— camera pipeline appears dead"
                )
            return None

        self._consecutive_failures = 0
        data = FrameData(
            frame = orient(frame),
            frame_id = self._frame_id,
            timestamp_ms = timestamp_ms,
        )
        self._frame_id += 1
        return data

    def release(self) -> None:
        """Safe to call repeatedly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoSink:
    """Every frame must match the configured size."""
    def __init__(
            self,
            filename: str,
            fps: int,
            width: int,
            height: int,
            fourcc: str = "XVID"
        ) -> None:
        """No file until open(). fps is the playback rate written, not a throttle."""
        self.filename, self.fps = filename, fps
        self.size = (width, height)
        self.fourcc = fourcc
        self._out = None

    def open(self) -> "VideoSink":
        """Creates or overwrites the file. CaptureError for a bad path or codec."""
        cc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._out = cv2.VideoWriter(self.filename, cc, self.fps, self.size)
        if not self._out.isOpened():
            raise CaptureError(f"Failed to open writer: {self.filename}")
        return self

    def write(self, frame: np.ndarray) -> None:
        """CaptureError before open() or for a frame of another size."""
        if self._out is None:
            raise CaptureError("write() called before open()")
        if frame.shape[:2][::-1] != self.size:   # shape is (H, W); size is (W, H)
            raise CaptureError(
                f"VideoSink: expected {self.size}, got {frame.shape[:2][::-1]}"
            )
        self._out.write(frame)

    def release(self) -> None:
        """Safe to call repeatedly."""
        if self._out is not None:
            self._out.release()
            self._out = None


def check_control(name: str, value) -> str | None:
    """Why value doesn't suit the control, or None. An enum also takes its index."""
    kind, allowed, _ = CAMERA_CONTROLS_SPEC[name]
    if kind == "bool":
        return None if isinstance(value, bool) else f"{name} takes true or false, not {value!r}"
    if kind == "enum":
        if isinstance(value, str) and value.lower() in allowed:
            return None
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(allowed):
            return None
        return f"{name} takes one of {', '.join(allowed)}, not {value!r}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} takes a number, not {value!r}"
    if kind == "int" and not isinstance(value, int):
        return f"{name} takes a whole number, not {value!r}"
    lo, hi = allowed
    return None if lo <= value <= hi else f"{name} takes {lo} to {hi}, not {value!r}"


def format_controls(controls: dict) -> str:
    """name=value pairs for the pipeline string. ValueError names an unknown control or a bad value."""
    unknown = sorted(set(controls) - CAMERA_CONTROL_NAMES)
    if unknown:
        raise ValueError(f"unknown camera control(s) {', '.join(unknown)}; "
                         f"known: {', '.join(sorted(CAMERA_CONTROL_NAMES))}")
    problems = [p for p in (check_control(k, v) for k, v in controls.items()) if p]
    if problems:
        raise ValueError("bad camera control value: " + "; ".join(problems))
    fmt = lambda v: ("true" if v else "false") if isinstance(v, bool) else str(v)      # noqa: E731
    return " ".join(f"{k}={fmt(v)}" for k, v in controls.items())


def describe_controls(current: dict | None = None) -> list[str]:
    """One line per control: its values, what it does, and what params.CAMERA_CONTROLS sets."""
    current = CAMERA_CONTROLS if current is None else current
    lines = []
    for name, (kind, allowed, what) in CAMERA_CONTROLS_SPEC.items():
        values = ("true | false" if kind == "bool" else " | ".join(allowed) if kind == "enum"
                  else f"{allowed[0]} .. {allowed[1]}")
        now = f"   [set: {current[name]}]" if name in current else ""
        lines += [f"{name:<20} {values}{now}", f"{'':<20} {what}"]
    return lines


def parse_controls(pairs: list[str] | None) -> dict:
    """--camera-control KEY=VALUE strings as a dict: booleans and numbers parsed, enums kept as text."""
    out = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or not key or not raw:
            raise ValueError(f"{pair!r}: expected KEY=VALUE, e.g. exposure-value=-1")
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key] = value
    format_controls(out)
    return out


def build_gst_pipeline(
        width: int = FRAME_W,
        height: int = FRAME_H,
        fps: int = FPS,
        color_space: str = "BGR",
        controls: dict | None = None,
    ) -> str:
    """The capture pipeline string for cv2.VideoCapture. ValueError for an unknown control."""
    # format:        BGR straight from the ISP, so no videoconvert
    # sensor-config: pinned mode so field of view doesn't change with output size
    # no videoflip:  orient() turns the frames
    # appsink:       newest frame only, so a slow consumer never reads a backlog
    ctl = format_controls(CAMERA_CONTROLS if controls is None else controls)
    return (
        f'libcamerasrc sensor-config="{SENSOR_CONFIG}"{" " + ctl if ctl else ""} ! '
        f"video/x-raw,format={color_space},width={width},height={height},framerate={fps}/1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )