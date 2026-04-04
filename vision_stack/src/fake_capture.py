"""
fake_capture.py
===============
Camera-free frame source for pipeline development on the Pi.

Loads images from disk and cycles through them at a target rate,
yielding (frame_id, timestamp_ms, frame) tuples identical to what
the live GStreamer source would produce.

Usage:
    from fake_capture import FakeCapture

    src = FakeCapture(
        image_dirs=["vision_stack/sample_img/duckietown/s1"],
        target_fps=30,
        resize_to=(640, 480),   # match real capture resolution
        loop=True,
    )
    for frame_id, timestamp_ms, frame in src:
        output = run_phase2_on_frame(frame, frame_id, timestamp_ms, cfg)
"""

import cv2
import time
import glob
import os
import numpy as np
from typing import Optional

class FakeCapture:
    """
    Cycles through a list of image files at a target frame rate.
    Mimics the (frame_id, timestamp_ms, frame) contract of a live capture loop.
    """

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        image_dirs: list,
        target_fps: float = 30.0,
        resize_to: tuple = (640, 480),  # (W, H)
        loop: bool = True,
        max_frames: Optional[int] = None,         # None for infinite, or a positive integer to limit total frames
    ):
        self.target_fps   = target_fps
        self.resize_to    = resize_to   # (W, H) to match cv2 convention
        self.loop         = loop
        self.max_frames   = max_frames
        self._frame_delay = 1.0 / target_fps

        self._paths = self._collect_images(image_dirs)
        if not self._paths:
            raise FileNotFoundError(
                f"FakeCapture: no images found in {image_dirs}"
            )
        print(f"[FakeCapture] {len(self._paths)} images loaded, "
              f"target={target_fps} FPS, resize={resize_to}")

    def _collect_images(self, dirs: list) -> list:
        paths = []
        for d in dirs:
            for ext in self.IMAGE_EXTENSIONS:
                paths += sorted(glob.glob(os.path.join(d, f"*{ext}")))
                paths += sorted(glob.glob(os.path.join(d, f"*{ext.upper()}")))
        return paths

    def __iter__(self):
        frame_id   = 0
        img_index  = 0
        start_time = time.time()

        while True:
            if self.max_frames is not None and frame_id >= self.max_frames:
                break

            # Wrap around when we've exhausted the image list
            if img_index >= len(self._paths):
                if not self.loop:
                    break
                img_index = 0

            # Load and resize
            path  = self._paths[img_index]
            frame = cv2.imread(path)
            if frame is None:
                print(f"[FakeCapture] WARNING: could not read {path}, skipping")
                img_index += 1
                continue

            W, H = self.resize_to
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

            timestamp_ms = int(time.time() * 1000)

            yield frame_id, timestamp_ms, frame

            frame_id  += 1
            img_index += 1

            # Rate limiting — sleep only the remaining budget for this frame
            elapsed    = time.time() - start_time
            expected   = frame_id * self._frame_delay
            sleep_time = expected - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)