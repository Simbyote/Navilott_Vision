"""Phase 2 perception: every per-frame image stage, and OpenCV's thread count for the process.

Purpose:
    Importing any stage imports this package first, so it is where the
    process-wide OpenCV setting lives. main.py, every linker, the replays
    and the tests all get the same thread count before the first frame
    (params.OPENCV_THREADS has the measurements behind the number).

Main package:
    None: the stages are the package's modules.

Flow:
    1. On first import, cv2.setNumThreads(OPENCV_THREADS).
"""
import cv2

from src.params import OPENCV_THREADS

cv2.setNumThreads(OPENCV_THREADS)
