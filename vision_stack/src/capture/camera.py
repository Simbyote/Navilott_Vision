'''
capture.py

Phase 1: Video Capture

Purpose:
    Owns the camera pipeline and is the single origin of per-frame identity.
    Every frame delivered by CameraSource.read() arrives as a FrameData
    carrying the pixels, a sequence number, and an arrival timestamp.
    No stage downstream mints either value; they reference what capture
    assigned.

Clock:
    timestamp_ms comes from time.monotonic_ns(), so it never steps backward
    when NTP corrects the wall clock. The origin is arbitrary. The value is
    meaningful only as a difference against another timestamp_ms from the
    same process run, which is all the pipeline uses it for.
'''
import time
import cv2
import numpy as np
from dataclasses import dataclass


def _now_ms() -> int:
    """
    Purpose:
        Monotonic millisecond timestamp
    """
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class FrameData:
    """
    Captured frame and its identity.

    frame: (H, W, 3) uint8 BGR image as delivered by the appsink
    frame_id: sequence number, 0-based, counts frames delivered
    timestamp_ms: monotonic ms sampled immediately after the frame was
                  pulled from the appsink
    """
    frame: np.ndarray
    frame_id: int
    timestamp_ms: int


class CaptureError(RuntimeError):
    """Raised when the camera pipeline cannot be opened."""

class CameraSource:
    """
    Purpose:
        Encapsulates a camera source for video capture using OpenCV and GStreamer.

    """
    def __init__(
            self, 
            width, 
            height, 
            fps, 
            max_consecutive_failures=None
        ):
        """
        Purpose:
            Initializes the CameraSource with specified width, height, fps, and optional max_consecutive_failures.
        Inputs:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            max_consecutive_failures: Maximum number of consecutive frame read failures before giving up (default
        """
        self.width, self.height, self.fps = width, height, fps
        self.max_consecutive_failures = max_consecutive_failures or fps  # ~1s
        self._cap = None
        self._consecutive_failures = 0
        # Counts delivered frames for the life of this instance. Deliberately
        # not reset by open(): a reopen after a recovery must not reissue ids
        # that already appeared in the logs.
        self._frame_id = 0

    def open(self):
        """
        Purpose:
            Opens the camera source using a GStreamer pipeline.
        Raises:
            CaptureError: If the camera pipeline cannot be opened.
        """
        pipeline = build_gst_pipeline(self.width, self.height, self.fps)
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise CaptureError("Failed to open camera pipeline")
        return self
    
    def read(self):
        """
        Purpose:
            Pull one frame from the camera and stamp it with its identity.

        Raises:
            CaptureError: If the camera source is not opened, or if there
            are too many consecutive read failures.

        Returns:
            FrameData on success, or None when a single read failed and the
            failure budget has not been exhausted. A None costs no frame_id,
            so the caller can `continue` without creating a gap or a
            duplicate.
        """
        if self._cap is None:
            raise CaptureError("read() called before open()")

        ok, frame = self._cap.read()
        timestamp_ms = _now_ms()   # sampled at arrival, before any branching

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

    def release(self):
        """
        Purpose:
            Releases the camera source and frees resources.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

class VideoSink:
    """
    Purpose:
        Encapsulates a video sink for writing frames to a video file using OpenCV.
    """
    def __init__(
            self, 
            filename, 
            fps, 
            width, 
            height, 
            fourcc="XVID"
        ):
        """
        Purpose:
            Initializes the VideoSink with specified filename, fps, width, height, and fourcc codec.
        Inputs:
            filename: Output video file name
            fps: Frames per second
            width: Frame width in pixels
            height: Frame height in pixels
            fourcc: FourCC code for video codec (default: "XVID")
        """
        self.filename, self.fps = filename, fps
        self.size = (width, height)
        self.fourcc = fourcc
        self._out = None

    def open(self):
        """
        Purpose:
            Opens the video sink for writing frames to the specified file.
        Raises:
            CaptureError: If the video writer cannot be opened.
        """
        cc = cv2.VideoWriter_fourcc(*self.fourcc)
        self._out = cv2.VideoWriter(self.filename, cc, self.fps, self.size)
        if not self._out.isOpened():
            raise CaptureError(f"Failed to open writer: {self.filename}")
        return self

    def write(self, frame):
        """
            Purpose:
                Writes a frame to the video sink.
            Inputs:
                frame: The frame to be written to the video file.
            Raises:
                CaptureError: If the frame size does not match the expected size.
        """
        if self._out is None:
            return
        if frame.shape[:2][::-1] != self.size:
            raise CaptureError(
                f"VideoSink: expected {self.size}, got {frame.shape[:2][::-1]}"
            )
        self._out.write(frame)

    def release(self):
        """
        Purpose:
            Releases the video sink and frees resources.
        """
        if self._out is not None:
            self._out.release()
            self._out = None


def build_gst_pipeline(
        width: int = 480, 
        height: int = 360, 
        fps: int = 15, 
        color_space: str = "BGR"
    ) -> str:
    """
        Purpose:
            Builds a libcamera -> GStreamer -> OpenCV pipeline string
        
        Inputs:
            width: frame width in pixels
            height: frame height in pixels
            fps: frames per second
            color_space: color space for the video frames
        
        Outputs:
            GStreamer pipeline string to pass to cv2.VideoCapture() for Phase 1 camera acquisition
        """
    return (
        'libcamerasrc sensor-config="sensor/config,width=1920,height=1080,depth=10" ! '
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "videoflip method=rotate-180 ! "
        f"video/x-raw,format={color_space} ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
