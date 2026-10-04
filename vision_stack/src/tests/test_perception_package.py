"""
test_perception_package.py  --  src/perception/__init__.py

--software  Importing any perception stage sets OpenCV's thread count to
            params.OPENCV_THREADS, in a fresh interpreter so nothing earlier
            in the session has set it. No camera.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from src.params import OPENCV_THREADS

ROOT = Path(__file__).resolve().parents[2]


def threads_after(code):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return int(r.stdout.strip().splitlines()[-1])


@pytest.mark.software
@pytest.mark.parametrize("module", ["src.perception", "src.perception.preprocess", "src.perception.geometry"])
def test_importing_a_perception_stage_sets_opencvs_thread_count(module):
    code = f"import cv2; cv2.setNumThreads(7); import {module}; print(cv2.getNumThreads())"
    assert threads_after(code) == OPENCV_THREADS


@pytest.mark.software
def test_the_production_pipeline_gets_it_too():
    # main.py runs Pipeline.step(); importing the pipeline must be enough
    assert threads_after("import cv2; cv2.setNumThreads(7); import src.pipeline; print(cv2.getNumThreads())") \
        == OPENCV_THREADS
