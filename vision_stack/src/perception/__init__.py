"""Phase 2 perception: every per-frame image stage.

Purpose:
    Importing any stage imports this package first, so it is where the
    process-wide OpenCV setting lives: cv2.setNumThreads(OPENCV_THREADS),
    set before the first frame by main.py, every linker, the replays and
    the tests alike (params.py has the measurements behind the number).

Flow:
    1. On first import: OpenCV's parallel operations use OPENCV_THREADS
       threads, the calling one included.
"""
import cv2

from src.params import OPENCV_THREADS

cv2.setNumThreads(OPENCV_THREADS)
