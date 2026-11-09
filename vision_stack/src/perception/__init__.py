"""Phase 2 perception: the per-frame image stages.

Importing any stage imports this package first, so it sets OpenCV's thread
count for the whole process (params.OPENCV_THREADS) before the first frame.
"""
import cv2

from src.params import OPENCV_THREADS

cv2.setNumThreads(OPENCV_THREADS)
