"""
Test-mode selection and shared fixtures for the vision stack.

    pytest                         -> --software (default)
    pytest --software              contract tests: no camera, deterministic, pass/fail
    pytest --hardware              characterization on the target: CSV / PNG / graphs
    pytest --hardware --replay=src/tests/data/frames   (use the = form: with a space,
                                   pytest treats DIR as a path to collect)
                                   same, but fed from recorded frames instead of the camera
    pytest --hardware --record     capture tests also save their frames to
                                   tests/data/frames (the software dataset)
    pytest --frames=300            frames per hardware run (default 100)

Every test carries exactly one of @pytest.mark.software / @pytest.mark.hardware.
Tests outside the selected mode are deselected rather than skipped, so the
output only lists what actually ran.
"""
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import pytest

from src.capture.camera import CameraSource, CaptureError, FrameData
from src.params import FPS, FRAME_H, FRAME_W
from src.tests.artifacts import Artifacts

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data" / "frames"

CAMERA = dict(width=FRAME_W, height=FRAME_H, fps=FPS)    # keyword args for CameraSource


def pytest_addoption(parser):
    g = parser.getgroup("vision-pipeline")
    g.addoption("--software", action="store_true",
                help="run contract tests (default when no mode is given)")
    g.addoption("--hardware", action="store_true",
                help="run characterization tests that write artifacts")
    g.addoption("--frames", type=int, default=100,
                help="frames per hardware run (default: 100)")
    g.addoption("--replay", default=None, metavar="DIR",
                help="hardware tests read recorded frames from DIR instead of the camera")
    g.addoption("--record", action="store_true",
                help="capture test also saves frames to tests/data/frames")
    g.addoption("--artifact-dir", default="artifacts",
                help="root for hardware-run output (default: ./artifacts)")


def pytest_configure(config):
    config.addinivalue_line("markers", "software: contract test, no hardware needed")
    config.addinivalue_line("markers", "hardware: characterization test, writes artifacts")


def _selected_modes(config) -> set[str]:
    """Modes chosen on the command line; software when none is given."""
    modes = set()
    if config.getoption("--software"):
        modes.add("software")
    if config.getoption("--hardware"):
        modes.add("hardware")
    return modes or {"software"}


def pytest_collection_modifyitems(config, items):
    modes = _selected_modes(config)
    kept, dropped = [], []
    for item in items:
        item_modes = {m for m in ("software", "hardware") if item.get_closest_marker(m)}
        if item_modes and not (item_modes & modes):
            dropped.append(item)
        else:
            kept.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def load_recorded_frames(directory: Path, limit: int | None = None):
    """
    Yield FrameData from a directory of PNGs. If manifest.csv (frame_id,
    timestamp_ms) exists it supplies identity; otherwise ids are the sort
    order and timestamps are synthesized at the target fps.
    """
    directory = Path(directory)
    manifest = {}
    mpath = directory / "manifest.csv"
    if mpath.exists():
        with open(mpath, newline="") as f:
            for row in csv.DictReader(f):
                manifest[row["file"]] = (int(row["frame_id"]), int(row["timestamp_ms"]))
    period_ms = round(1000 / CAMERA["fps"])
    count = 0
    for i, png in enumerate(sorted(directory.glob("*.png"))):
        if limit is not None and count >= limit:
            return
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fid, ts = manifest.get(png.name, (i, i * period_ms))
        yield FrameData(frame=img, frame_id=fid, timestamp_ms=ts)
        count += 1


def _live_frames(n: int):
    """n frames from the real camera; skips the test if it can't open. Failed reads don't count."""
    src = CameraSource(**CAMERA)
    try:
        src.open()
    except CaptureError as e:
        pytest.skip(f"camera unavailable: {e}")
    try:
        got = 0
        while got < n:
            fd = src.read()          # raises CaptureError when the pipeline is dead
            if fd is None:
                continue
            got += 1
            yield fd
    finally:
        src.release()


@pytest.fixture
def frames(request):
    """
    Callable frames(n) -> iterator of FrameData. Live camera by default,
    recorded frames when --replay DIR is given. Hardware tests written
    against this fixture run unchanged on the Pi or on a desktop.
    """
    replay = request.config.getoption("--replay")

    def gen(n: int):
        if replay:
            return load_recorded_frames(Path(replay), limit=n)
        return _live_frames(n)
    return gen


@pytest.fixture(scope="session")
def dataset_frames():
    """Recorded dataset for software tests; empty list if none recorded yet."""
    if not DATA_DIR.exists():
        return []
    return list(load_recorded_frames(DATA_DIR))


def _git_state() -> dict:
    """Short commit hash and dirty flag for run_meta.json; None fields outside a repo."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=2).stdout.strip())
        return {"commit": rev or None, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


@pytest.fixture(scope="session")
def run_root(request):
    """Session output dir + run_meta.json. Only created if a hardware test asks."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(request.config.getoption("--artifact-dir")) / stamp
    root.mkdir(parents=True, exist_ok=True)
    Artifacts(root).json("run_meta.json", {
        "started": stamp,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "git": _git_state(),
        "camera_target": CAMERA,
        "frames_requested": request.config.getoption("--frames"),
        "replay": request.config.getoption("--replay"),
    })
    return root


@pytest.fixture
def artifacts(run_root, request):
    """Artifacts writer for this test's own subdirectory of the session run."""
    return Artifacts(run_root / request.node.name)