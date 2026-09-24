"""Camera acquisition and per-frame identity for Phase 1.

Purpose:
    Owns the libcamera/GStreamer camera pipeline and is the single origin of
    per-frame identity. No downstream stage mints a frame id or timestamp;
    each references what capture assigned. Also provides a file sink for
    writing frames to video.

Main package:
    FrameData: one BGR frame plus its sequence number and monotonic arrival
    time. Ids are unique and gap-free for the life of a CameraSource, even
    across reopens.

Flow:
    1. Build the libcamera -> GStreamer -> OpenCV pipeline string.
    2. Open the capture and pull frames, stamping each with id and time.
    3. Absorb isolated read failures; raise once the failure budget is spent.
    4. Optionally write frames to a video file.
"""
import time
import warnings
import cv2
import numpy as np
from dataclasses import dataclass

from src.params import CAMERA_ROTATE_180, FPS, FRAME_H, FRAME_W, MAX_FPS, MIN_FPS, SENSOR_CONFIG


def _now_ms() -> int:
    """Monotonic ms timestamp. Immune to NTP wall-clock steps; origin is arbitrary."""
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class FrameData:
    """Captured frame and its identity. Created only by CameraSource.read()."""
    frame: np.ndarray    # (H, W, 3) uint8 BGR, as delivered by the appsink
    frame_id: int        # 0-based count of delivered frames; failed reads don't consume one
    timestamp_ms: int    # monotonic ms at appsink pull; meaningful only as a difference within one run


class CaptureError(RuntimeError):
    """Raised on any capture or recording failure: open, dead camera, or frame-size mismatch."""

class CameraSource:
    """
    Owns the camera handle and assigns frame identity.

    frame_id never repeats or skips for the life of the instance, including
    across release() and open().
    """
    def __init__(
            self,
            width: int,
            height: int,
            fps: int,
            max_consecutive_failures: int | None = None
        ) -> None:
        """
        Configure the source. The camera is not touched until open().

        Inputs:
            width, height: Output frame size in px.
            fps: Requested frame rate. Also sets the default failure budget.
                Values outside [MIN_FPS, MAX_FPS] are accepted but warn.
            max_consecutive_failures: Consecutive failed reads tolerated
                before read() raises. Defaults to fps (~1 s of dead camera).
                Lower it to fail fast; raise it to ride out longer stalls.
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
        self.max_consecutive_failures = (
            fps if max_consecutive_failures is None else max_consecutive_failures
        )
        self._cap = None
        self._consecutive_failures = 0
        # Deliberately not reset by open(): a reopen after a recovery must
        # not reissue ids that already appeared in the logs.
        self._frame_id = 0

    def open(self) -> "CameraSource":
        """
        Start the camera pipeline.

        Outputs:
            self, so construction and open() can be chained.

        Side effects:
            Claims the camera through libcamera.

        Raises:
            CaptureError: If GStreamer cannot open the pipeline.
        """
        pipeline = build_gst_pipeline(self.width, self.height, self.fps)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise CaptureError("Failed to open camera pipeline")
        return self

    def read(self) -> FrameData | None:
        """
        Pull one frame from the camera and stamp it with its identity.

        Outputs:
            FrameData on success. None when a single read failed but the
            failure budget isn't exhausted. A None consumes no frame_id, so
            the caller can `continue` without creating a gap or a duplicate.

        Side effects:
            Advances frame_id on success; updates the consecutive-failure count.

        Raises:
            CaptureError: If called before open(), or once the failure
                budget is exhausted.
        """
        if self._cap is None:
            raise CaptureError("read() called before open()")

        ok, frame = self._cap.read()
        timestamp_ms = _now_ms()   # before the branch so it marks arrival, not packaging

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
            frame = frame,
            frame_id = self._frame_id,
            timestamp_ms = timestamp_ms,
        )
        self._frame_id += 1
        return data

    def release(self) -> None:
        """Release the camera. Safe to call repeatedly or before open()."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

class VideoSink:
    """Writes BGR frames to a video file. Every frame must match the configured size."""
    def __init__(
            self,
            filename: str,
            fps: int,
            width: int,
            height: int,
            fourcc: str = "XVID"
        ) -> None:
        """
        Configure the sink. No file is created until open().

        Inputs:
            filename: Output path. The extension must suit the codec
                (XVID -> .avi).
            fps: Playback rate recorded in the file. It does not throttle writes.
            width, height: Expected frame size in px; write() rejects anything else.
            fourcc: Four-character codec code.
        """
        self.filename, self.fps = filename, fps
        self.size = (width, height)
        self.fourcc = fourcc
        self._out = None

    def open(self) -> "VideoSink":
        """
        Create the output file and start the encoder.

        Outputs:
            self, so construction and open() can be chained.

        Side effects:
            Creates or overwrites the file at filename.

        Raises:
            CaptureError: If the writer cannot be opened (bad path or
                unsupported codec).
        """
        cc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._out = cv2.VideoWriter(self.filename, cc, self.fps, self.size)
        if not self._out.isOpened():
            raise CaptureError(f"Failed to open writer: {self.filename}")
        return self

    def write(self, frame: np.ndarray) -> None:
        """
        Append one frame to the file.

        Inputs:
            frame: BGR image matching the configured (width, height).

        Side effects:
            File I/O.

        Raises:
            CaptureError: If called before open(), or the frame size doesn't
                match the configured size.
        """
        if self._out is None:
            raise CaptureError("write() called before open()")
        if frame.shape[:2][::-1] != self.size:   # shape is (H, W); size is (W, H)
            raise CaptureError(
                f"VideoSink: expected {self.size}, got {frame.shape[:2][::-1]}"
            )
        self._out.write(frame)

    def release(self) -> None:
        """Finalize and close the file. Safe to call repeatedly or before open()."""
        if self._out is not None:
            self._out.release()
            self._out = None


def build_gst_pipeline(
        width: int = FRAME_W,
        height: int = FRAME_H,
        fps: int = FPS,
        color_space: str = "BGR"
    ) -> str:
    """
    Build the libcamera -> GStreamer -> OpenCV capture pipeline string.

    Inputs:
        width, height: Output frame size in px, scaled from the fixed
            SENSOR_CONFIG mode.
        fps: Requested frame rate.
        color_space: Raw format handed to OpenCV. Downstream stages assume "BGR".

    Outputs:
        Pipeline string for cv2.VideoCapture(..., cv2.CAP_GSTREAMER).
    """
    # sensor-config: pinned mode so field of view doesn't change with output size
    # videoflip:     see CAMERA_ROTATE_180
    # appsink:       hold only the newest frame so a slow consumer never reads a stale backlog
    flip = "videoflip method=rotate-180 ! " if CAMERA_ROTATE_180 else ""
    return (
        f'libcamerasrc sensor-config="{SENSOR_CONFIG}" ! '
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        f"{flip}"
        f"video/x-raw,format={color_space} ! "
        "appsink drop=true max-buffers=1 sync=false"
    )