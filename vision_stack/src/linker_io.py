"""What the driving linkers share: the frame recorder, the stand-in motors, and each frame's chain record.

Purpose:
    maneuver_linker, navigation_linker and intersection_linker all drive the
    robot while recording every frame for a video rendered afterwards, and
    all offer a dry run with the motors off. Those pieces live here, so no
    linker imports another one for them.

Main package:
    FrameRecorder: put(frame_id, frame, record) on a background thread;
        close() flushes. dropped, written.
    NoMotors: drive() / brake() / stop() that do nothing (--no-motors).
    chain_record(res): one frame's Phase 1-3 result without its images,
        what debug_maneuver.as_result() redraws the Phase 3 view from.

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

import cv2

from src.debugger.debug_maneuver import FRAMES_DIR, RECORDS_FILE, frame_path

JPEG_QUALITY = 90       # frames are only a video background; decisions come from the records
QUEUE_FRAMES = 64       # ~2.5 s at 25 FPS of slack before the recorder drops frames


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
