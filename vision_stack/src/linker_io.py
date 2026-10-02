"""What the driving linkers share: opening the rig, the frame recorder, the stand-in motors, and each frame's chain record.

Purpose:
    maneuver_linker, navigation_linker and intersection_linker all open the
    same rig (a camera or a replay, the sensors, the motors, the start
    button), drive the robot while recording every frame for a video
    rendered afterwards, and offer a dry run with the motors off. Those
    pieces live here, so no linker repeats them or imports another one for
    them. phase3_linker opens its own: no motors or button, sensors by flag.

Main package:
    open_rig(...) -> Rig(source, sensors, motor, system); raises one of
        OPEN_ERRORS after releasing whatever had opened.
    release(*things): stop() or close() each that opened.
    countdown(): the console "starting in 3, 2, 1" when there's no button.
    FrameRecorder: put(frame_id, frame, record) on a background thread;
        close() flushes. dropped, written.
    NoMotors: drive() / brake() / stop() that do nothing (--no-motors).
    chain_record(res): one frame's Phase 1-3 result without its images,
        what debug_maneuver.as_result() redraws the Phase 3 view from.

Flow (opening the rig): the camera with the sensors, or a video / frame
    folder without; the motors and the start button only with the camera
    and when asked for, NoMotors otherwise.

Flow (a recording):
    1. FrameRecorder(out_dir) makes frames/ and opens records.pkl.
    2. Per frame: put() copies the frame and queues it with its record; a
       full queue drops it rather than block the control loop.
    3. close() writes what's queued, then reports any error the thread hit.
"""
import os
import pickle
import queue
import threading
import time
from typing import NamedTuple

import cv2

from src.capture.camera import CaptureError
from src.debugger.debug_maneuver import FRAMES_DIR, RECORDS_FILE, frame_path
from src.debugger.live_view import CameraFrameSource, DirectoryFrameSource, VideoFrameSource
from src.params import FPS, FRAME_H, FRAME_W
from src.peripherals.sensing import Sensors

JPEG_QUALITY = 90       # frames are only a video background; decisions come from the records
QUEUE_FRAMES = 64       # ~2.5 s at 25 FPS of slack before the recorder drops frames
COUNTDOWN_S = 3         # seconds to step back from the robot when there's no start button
# What a rig that won't open raises: a camera or file that won't open, no
# pigpiod, a missing module off the Pi. Anything else is a bug, not hardware
OPEN_ERRORS = (CaptureError, OSError, RuntimeError, ImportError)


# =============================================================================
# The rig
# =============================================================================

class Rig(NamedTuple):
    """What a driving linker runs on. sensors and system are None when not opened."""
    source: object
    sensors: object
    motor: object
    system: object


def release(*things) -> None:
    """Stop (sensors, motors) or close (sources) each thing that opened; None is skipped."""
    for thing in things:
        if thing is not None:
            (thing.stop if hasattr(thing, "stop") else thing.close)()


def open_rig(camera: bool = False, video: str | None = None, frames: str | None = None,
             fps: int | None = None, size: tuple[int, int] = (FRAME_W, FRAME_H),
             motors: bool = True, button: bool = True) -> Rig:
    """
    Open a driving linker's source, sensors, motors and start button.

    Inputs:
        camera: The robot's camera, with the IMU and encoders on the sensor
            hub. Else video (a recording) or frames (a folder) is replayed,
            with no sensors.
        fps: Capture or replay rate; None is FPS, or a video's own rate.
        size: Capture (width, height).
        motors: The real motors (pigpio, needs sudo pigpiod); only with the
            camera. NoMotors otherwise.
        button: The start button (peripherals.system.System); only with the
            camera. None otherwise.
    Outputs:
        Rig(source, sensors, motor, system).
    Raises:
        One of OPEN_ERRORS, once whatever had opened is released.
    """
    source = sensors = motor = system = None
    try:
        if camera:
            source = CameraFrameSource(size[0], size[1], fps or FPS)
            sensors = Sensors(imu=True, encoders=True)
        elif video:
            source = VideoFrameSource(video, fps)
        else:
            source = DirectoryFrameSource(frames, fps or FPS)
        if camera and motors:
            import pigpio
            from src.peripherals.drive import MotorController
            motor = MotorController(pigpio.pi())
        else:
            motor = NoMotors()
        if camera and button:
            from src.peripherals.system import System
            system = System()
    except OPEN_ERRORS:
        release(motor, sensors, source)
        raise
    return Rig(source, sensors, motor, system)


def countdown(seconds: int = COUNTDOWN_S) -> None:
    """With no start button: "starting in 3", "2", "1", a second apart."""
    for n in range(seconds, 0, -1):
        print(f"  starting in {n}")
        time.sleep(1.0)


# =============================================================================
# Recording
# =============================================================================


class FrameRecorder:
    """
    Writes each frame's JPEG and pickled record on a background thread, so the
    control loop never waits on the disk.

    A full queue drops the frame (counted in dropped) rather than blocking:
    the video loses a frame, the robot doesn't lose a control step. The CSVs
    are written in the loop and are always complete.
    """
    def __init__(self, out_dir: str, maxsize: int = QUEUE_FRAMES) -> None:
        self.out_dir = out_dir
        os.makedirs(os.path.join(out_dir, FRAMES_DIR), exist_ok=True)
        self.dropped = self.written = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._records = open(os.path.join(out_dir, RECORDS_FILE), "wb")
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._work, name="frame-recorder", daemon=True)
        self._thread.start()

    def put(self, frame_id: int, frame, record: dict) -> None:
        """Queue one frame and its record; copies the frame, since the camera may reuse its buffer."""
        try:
            self._q.put_nowait((frame_id, frame.copy(), record))
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        try:
            while (item := self._q.get()) is not None:
                fid, frame, record = item
                cv2.imwrite(frame_path(self.out_dir, fid), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                pickle.dump(record, self._records, protocol=pickle.HIGHEST_PROTOCOL)
                self.written += 1
        except BaseException as exc:            # reported by close(); the loop must not die with it
            self._error = exc

    def close(self) -> None:
        """Flush everything queued, then close. Raises what the thread hit, if anything."""
        self._q.put(None)
        self._thread.join()
        self._records.close()
        if self._error is not None:
            raise RuntimeError(f"frame recorder failed: {self._error!r}") from self._error


class NoMotors:
    """--no-motors: commands are logged in the run's CSV but never sent."""
    def drive(self, left: float, right: float) -> None:
        pass

    def brake(self) -> None:
        pass

    def stop(self) -> None:
        pass


def chain_record(res) -> dict:
    """
    What debug_maneuver.as_result() needs to redraw a frame's Phase 3 view,
    without the images the chain carries.

    Inputs:
        res: run_phase3_chain()'s result for this frame.
    Outputs:
        frame_id, timestamp_ms, geometry, offset, offset_debug, lane_rect,
        packet, p3_debug, timings.
    """
    chain = res.chain
    return {"frame_id": res.packet.frame_id, "timestamp_ms": res.packet.timestamp_ms,
            "geometry": chain.geometry, "offset": chain.offset, "offset_debug": chain.offset_debug,
            "lane_rect": chain.roi.lane_rect, "packet": res.packet, "p3_debug": res.p3_debug,
            "timings": dict(res.timings_ms)}
