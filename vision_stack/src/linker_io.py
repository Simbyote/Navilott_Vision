"""What the driving linkers share: opening the rig, the frame recorder, the stand-in motors.

Purpose:
    maneuver_linker, navigation_linker and intersection_linker open the same
    rig, record every frame for a video rendered afterwards, and offer a dry
    run, so those pieces live here once.

Main package:
    Rig, from open_rig(): the source, sensors, motors and start button.
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
from src.os_threads import name_os_thread
from src.params import CAMERA_CONTROLS, FPS, FRAME_H, FRAME_W
from src.peripherals.sensing import Sensors

JPEG_QUALITY = 90       # frames are only the video's background
QUEUE_FRAMES = 64       # ~2.5 s of slack
COUNTDOWN_S = 3         # to step back, without a start button
# A rig that won't open; anything else is a bug, not hardware
OPEN_ERRORS = (CaptureError, OSError, RuntimeError, ImportError)


class Rig(NamedTuple):
    """sensors and system are None when not opened."""
    source: object
    sensors: object
    motor: object
    system: object


def release(*things) -> None:
    for thing in things:
        if thing is not None:
            (thing.stop if hasattr(thing, "stop") else thing.close)()


def open_rig(camera: bool = False, video: str | None = None, frames: str | None = None,
             fps: int | None = None, size: tuple[int, int] = (FRAME_W, FRAME_H),
             motors: bool = True, button: bool = True, camera_controls: dict | None = None) -> Rig:
    """
    The camera with the IMU and encoders, or a replay of video or frames without sensors.
    Real motors (pigpio) and the button only with the camera; NoMotors otherwise.
    Raises one of OPEN_ERRORS once whatever had opened is released.
    """
    source = sensors = motor = system = None
    try:
        if camera:
            source = CameraFrameSource(size[0], size[1], fps or FPS, {**CAMERA_CONTROLS, **(camera_controls or {})})
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
    for n in range(seconds, 0, -1):
        print(f"  starting in {n}")
        time.sleep(1.0)


class FrameRecorder:
    """
    Writes each frame's JPEG and record on a thread, so the loop never waits on the disk.
    A full queue drops the frame (counted) rather than block the loop.
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
        """Copies the frame: the camera may reuse its buffer."""
        try:
            self._q.put_nowait((frame_id, frame.copy(), record))
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        name_os_thread("frame-recorder")
        try:
            while (item := self._q.get()) is not None:
                fid, frame, record = item
                cv2.imwrite(frame_path(self.out_dir, fid), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                pickle.dump(record, self._records, protocol=pickle.HIGHEST_PROTOCOL)
                self.written += 1
        except BaseException as exc:            # close() reports it; the loop must not die
            self._error = exc

    def close(self) -> None:
        """Flush, then raise whatever the thread hit."""
        self._q.put(None)
        self._thread.join()
        self._records.close()
        if self._error is not None:
            raise RuntimeError(f"frame recorder failed: {self._error!r}") from self._error


class NoRecorder:
    """--no-record, to see whether the recorder's writes are behind a stall."""
    dropped = written = 0

    def put(self, frame_id: int, frame, record: dict) -> None:
        pass

    def close(self) -> None:
        pass


class NoMotors:
    """--no-motors: commands are logged, never sent."""
    def drive(self, left: float, right: float) -> None:
        pass

    def brake(self) -> None:
        pass

    def stop(self) -> None:
        pass


def chain_record(res) -> dict:
    """
    What debug_maneuver needs to redraw a frame's Phase 3 view, without the images; stages has each stage's ms.
    """
    chain = res.chain
    return {"frame_id": res.packet.frame_id, "timestamp_ms": res.packet.timestamp_ms,
            "geometry": chain.geometry, "offset": chain.offset, "offset_debug": chain.offset_debug,
            "lane_rect": chain.roi.lane_rect, "obstacle": chain.obstacle, "obstacle_rect": chain.roi.obstacle_rect,
            "packet": res.packet, "p3_debug": res.p3_debug, "stages": dict(chain.timings_ms),
            "timings": dict(res.timings_ms)}
