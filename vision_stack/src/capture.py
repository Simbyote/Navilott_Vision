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
        "libcamerasrc ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "videoflip method=rotate-180 ! "
        f"video/x-raw,format={color_space} ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

    
# =============================================================================
# Self-test
#
#   python3 capture.py              logic tests only, runs anywhere
#   python3 capture.py --camera     also exercises the real IMX219
#
# Exits non-zero on any failure so it can gate a commit.
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    import tempfile
    import time
    import traceback
    import numpy as np
 
    W, H, FPS = 480, 360, 15
 
    _results = []
 
    def check(name, fn):
        """Run one test, record pass/fail, never abort the suite."""
        try:
            fn()
        except Exception:
            _results.append((name, False))
            print(f"  FAIL  {name}")
            for line in traceback.format_exc().strip().splitlines()[-2:]:
                print(f"        {line.strip()}")
        else:
            _results.append((name, True))
            print(f"  ok    {name}")
 
    def expect_raises(exc_type, fn):
        try:
            fn()
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")
 
    class _FakeCapture:
        """
        Stands in for cv2.VideoCapture so the failure-counting logic can be
        exercised with no camera present. Scripted with a list of bools:
        True -> return a valid frame, False -> return (False, None).
        """
        def __init__(self, script):
            self._script = list(script)
 
        def read(self):
            ok = self._script.pop(0) if self._script else False
            return (True, np.zeros((H, W, 3), np.uint8)) if ok else (False, None)
 
        def isOpened(self):
            return True
 
        def release(self):
            pass
 
    # -------------------------------------------------------------------------
    # build_gst_pipeline
    # -------------------------------------------------------------------------
    def t_pipeline_is_a_string():
        s = build_gst_pipeline(W, H, FPS)
        assert isinstance(s, str), f"expected str, got {type(s).__name__}"
        assert "(" not in s and "'" not in s, f"tuple artifact in pipeline: {s}"
 
    def t_format_cap_after_videoconvert():
        s = build_gst_pipeline(W, H, FPS)
        assert s.index("videoconvert") < s.index("format=BGR"), (
            "format cap must sit downstream of videoconvert; libcamerasrc "
            "cannot negotiate BGR on its source pad"
        )
 
    def t_caps_have_no_stray_whitespace():
        for chunk in build_gst_pipeline(W, H, FPS).split(" ! "):
            if chunk.startswith("video/x-raw"):
                assert ", " not in chunk, f"space inside caps struct: {chunk!r}"
 
    def t_pipeline_element_count():
        s = build_gst_pipeline(W, H, FPS)
        assert len(s.split(" ! ")) == 6, f"expected 6 elements, got {s.split(' ! ')}"
 
    def t_pipeline_interpolates_params():
        s = build_gst_pipeline(320, 240, 30)
        for token in ("width=320", "height=240", "framerate=30/1"):
            assert token in s, f"{token} missing from {s}"
 
    # -------------------------------------------------------------------------
    # CameraSource -- no hardware needed
    # -------------------------------------------------------------------------
    def t_read_before_open_raises():
        cam = CameraSource(W, H, FPS)
        expect_raises(CaptureError, cam.read)
 
    def t_release_before_open_is_safe():
        CameraSource(W, H, FPS).release()
 
    def t_release_is_idempotent():
        cam = CameraSource(W, H, FPS)
        cam._cap = _FakeCapture([])
        cam.release()
        cam.release()
        assert cam._cap is None
 
    def t_transient_failure_returns_none():
        cam = CameraSource(W, H, FPS, max_consecutive_failures=5)
        cam._cap = _FakeCapture([False, False])
        assert cam.read() is None
        assert cam.read() is None
        assert cam._consecutive_failures == 2
 
    def t_sustained_failure_raises():
        cam = CameraSource(W, H, FPS, max_consecutive_failures=3)
        cam._cap = _FakeCapture([False] * 10)
        assert cam.read() is None
        assert cam.read() is None
        expect_raises(CaptureError, cam.read)   # third failure trips it
 
    def t_counter_resets_after_good_frame():
        cam = CameraSource(W, H, FPS, max_consecutive_failures=3)
        cam._cap = _FakeCapture([False, False, True, False, False])
        cam.read(); cam.read()
        assert cam._consecutive_failures == 2
        assert cam.read() is not None
        assert cam._consecutive_failures == 0, "counter must reset on success"
        cam.read(); cam.read()                  # must not trip: streak restarted
 
    def t_default_failure_budget_is_about_one_second():
        cam = CameraSource(W, H, FPS)
        assert cam.max_consecutive_failures == FPS

    # -------------------------------------------------------------------------
    # FrameData contract -- no hardware needed
    # -------------------------------------------------------------------------
    def t_read_returns_framedata():
        cam = CameraSource(W, H, FPS)
        cam._cap = _FakeCapture([True])
        fd = cam.read()
        assert isinstance(fd, FrameData), f"got {type(fd).__name__}"
        assert fd.frame.shape == (H, W, 3), f"got {fd.frame.shape}"
        assert isinstance(fd.frame_id, int)
        assert isinstance(fd.timestamp_ms, int)

    def t_frame_ids_start_at_zero_and_increment():
        cam = CameraSource(W, H, FPS)
        cam._cap = _FakeCapture([True] * 4)
        ids = [cam.read().frame_id for _ in range(4)]
        assert ids == [0, 1, 2, 3], f"got {ids}"

    def t_dropped_frames_do_not_consume_an_id():
        """A None must not burn an id, or two frames share one downstream."""
        cam = CameraSource(W, H, FPS, max_consecutive_failures=10)
        cam._cap = _FakeCapture([True, False, False, True])
        first = cam.read()
        assert cam.read() is None
        assert cam.read() is None
        second = cam.read()
        assert (first.frame_id, second.frame_id) == (0, 1), (
            f"got {first.frame_id} then {second.frame_id}"
        )

    def t_timestamps_are_monotonic():
        cam = CameraSource(W, H, FPS)
        cam._cap = _FakeCapture([True] * 5)
        stamps = [cam.read().timestamp_ms for _ in range(5)]
        assert stamps == sorted(stamps), f"timestamp went backward: {stamps}"

    def t_framedata_is_immutable():
        """Downstream stages reference the stamp; none of them may rewrite it."""
        fd = FrameData(np.zeros((H, W, 3), np.uint8), 7, 1234)
        expect_raises(Exception, lambda: setattr(fd, "frame_id", 8))

    def t_reopen_does_not_reissue_ids():
        cam = CameraSource(W, H, FPS)
        cam._cap = _FakeCapture([True, True])
        cam.read(); cam.read()
        cam.release()
        cam._cap = _FakeCapture([True])          # stand in for a reopen
        assert cam.read().frame_id == 2, "ids restarted after reopen"
 
    # -------------------------------------------------------------------------
    # VideoSink
    # -------------------------------------------------------------------------
    def t_sink_write_before_open_is_noop():
        VideoSink("unused.avi", FPS, W, H).write(np.zeros((H, W, 3), np.uint8))
 
    def t_sink_rejects_wrong_size():
        path = os.path.join(tempfile.mkdtemp(), "size.avi")
        sink = VideoSink(path, FPS, W, H).open()
        try:
            expect_raises(
                CaptureError,
                lambda: sink.write(np.zeros((H // 2, W, 3), np.uint8)),
            )
        finally:
            sink.release()
 
    def t_sink_roundtrip():
        path = os.path.join(tempfile.mkdtemp(), "roundtrip.avi")
        sink = VideoSink(path, FPS, W, H).open()
        try:
            for i in range(30):
                f = np.zeros((H, W, 3), np.uint8)
                f[:, :, i % 3] = 255
                sink.write(f)
        finally:
            sink.release()
 
        assert os.path.getsize(path) > 0, "writer produced a 0-byte file"
        back = cv2.VideoCapture(path)
        try:
            assert back.isOpened(), "written file could not be reopened"
            n = 0
            while back.read()[0]:
                n += 1
        finally:
            back.release()
        assert n == 30, f"wrote 30 frames, read back {n}"
 
    def t_sink_release_is_idempotent():
        path = os.path.join(tempfile.mkdtemp(), "idem.avi")
        sink = VideoSink(path, FPS, W, H).open()
        sink.release()
        sink.release()
        assert sink._out is None
 
    # -------------------------------------------------------------------------
    # Hardware -- only with --camera
    # -------------------------------------------------------------------------
    def t_camera_opens_and_reads():
        cam = CameraSource(W, H, FPS).open()
        try:
            fd = None
            for _ in range(10):                 # allow a few startup drops
                fd = cam.read()
                if fd is not None:
                    break
            assert fd is not None, "no frame in first 10 reads"
            assert fd.frame.shape == (H, W, 3), f"expected {(H, W, 3)}, got {fd.frame.shape}"
            assert fd.frame.dtype == np.uint8, f"expected uint8, got {fd.frame.dtype}"
            assert fd.frame_id == 0, f"first delivered frame must be id 0, got {fd.frame_id}"
        finally:
            cam.release()
 
    def t_camera_sustains_rate():
        cam = CameraSource(W, H, FPS).open()
        try:
            n, dropped, t0 = 0, 0, time.perf_counter()
            while n < 45:
                if cam.read() is None:
                    dropped += 1
                else:
                    n += 1
            measured = n / (time.perf_counter() - t0)
        finally:
            cam.release()
        print(f"        measured {measured:.1f} FPS, {dropped} dropped")
 
    def t_camera_frames_are_not_constant():
        """A frozen or black sensor still returns ok=True -- check for signal."""
        cam = CameraSource(W, H, FPS).open()
        try:
            frames = [fd.frame for fd in (cam.read() for _ in range(20))
                      if fd is not None]
        finally:
            cam.release()
        assert len(frames) >= 2, "need two frames to compare"
        assert frames[0].std() > 1.0, "frame is flat -- lens cap on, or sensor dead?"
        deltas = [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                  for a, b in zip(frames, frames[1:])]
        assert max(deltas) > 0.0, "every frame identical -- appsink may be stalled"
 
    print("\nbuild_gst_pipeline")
    check("returns a plain string", t_pipeline_is_a_string)
    check("format cap follows videoconvert", t_format_cap_after_videoconvert)
    check("caps contain no stray whitespace", t_caps_have_no_stray_whitespace)
    check("pipeline has 6 elements", t_pipeline_element_count)
    check("parameters interpolate", t_pipeline_interpolates_params)
 
    print("\nCameraSource")
    check("read() before open() raises", t_read_before_open_raises)
    check("release() before open() is safe", t_release_before_open_is_safe)
    check("release() is idempotent", t_release_is_idempotent)
    check("transient failure returns None", t_transient_failure_returns_none)
    check("sustained failure raises", t_sustained_failure_raises)
    check("counter resets after good frame", t_counter_resets_after_good_frame)
    check("default budget is ~1 second", t_default_failure_budget_is_about_one_second)

    print("\nFrameData")
    check("read() returns a FrameData", t_read_returns_framedata)
    check("ids start at 0 and increment", t_frame_ids_start_at_zero_and_increment)
    check("a dropped frame costs no id", t_dropped_frames_do_not_consume_an_id)
    check("timestamps never go backward", t_timestamps_are_monotonic)
    check("FrameData is immutable", t_framedata_is_immutable)
    check("reopen does not reissue ids", t_reopen_does_not_reissue_ids)
 
    print("\nVideoSink")
    check("write() before open() is a no-op", t_sink_write_before_open_is_noop)
    check("wrong frame size raises", t_sink_rejects_wrong_size)
    check("30 frames written and read back", t_sink_roundtrip)
    check("release() is idempotent", t_sink_release_is_idempotent)
 
    if "--camera" in sys.argv:
        print("\nHardware")
        check("opens and delivers a frame", t_camera_opens_and_reads)
        check("sustains target frame rate", t_camera_sustains_rate)
        check("frames carry real signal", t_camera_frames_are_not_constant)
    else:
        print("\nHardware: skipped (pass --camera on the Pi to include)")
 
    passed = sum(1 for _, ok in _results if ok)
    print(f"\n{passed}/{len(_results)} passed")
    sys.exit(0 if passed == len(_results) else 1)