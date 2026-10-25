"""Small helpers shared across the pipeline's stages.

Kept in one place so copies can't drift, and dependency-free so any stage can
import them. Laps times stages; GcWatch freezes and times the garbage collector
for a control loop.
"""
import subprocess
import time
from pathlib import Path


def _git(*args) -> str | None:
    try:
        p = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5,
                           cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def code_version() -> dict:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain")
    return {"commit": commit, "uncommitted_changes": None if dirty is None else bool(dirty)}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def check_same_frame(geometry, roi, stage: str) -> None:
    """ValueError, naming stage, if either is None or their frame stamps disagree."""
    if geometry is None:
        raise ValueError(f"{stage}: geometry result is None")
    if roi is None:
        raise ValueError(f"{stage}: roi result is None")
    # A mismatch means candidates from two frames reached one call
    if (geometry.frame_id, geometry.timestamp_ms) != (roi.frame_id, roi.timestamp_ms):
        raise ValueError(
            f"{stage}: geometry stamp "
            f"{(geometry.frame_id, geometry.timestamp_ms)} does not match roi stamp "
            f"{(roi.frame_id, roi.timestamp_ms)} — candidates are from different frames"
        )


def check_frame_size(frame, size: tuple[int, int] | None, stage: str) -> None:
    """ValueError, naming stage, unless frame is size (H, W); None accepts any."""
    if size is not None and tuple(frame.shape[:2]) != tuple(size):
        raise ValueError(f"{stage}: frame is {tuple(frame.shape[:2])}, but the cm scale was set up for {tuple(size)}")


class Laps:
    """lap(name) records ms since the previous lap, mark() or start()."""
    __slots__ = ("times", "_mark")

    def __init__(self) -> None:
        self.times: dict = {}
        self._mark = time.perf_counter()

    def start(self) -> None:
        self.times = {}
        self._mark = time.perf_counter()

    def mark(self) -> None:
        self._mark = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.times[name] = (now - self._mark) * 1000.0
        self._mark = now


class GcWatch:
    """
    A full collection holds the GIL and stops every thread: 0.6-1.6 s frames on
    2026-10-12, past the motor watchdog. start() collects once and freezes everything
    alive, so later collections skip it; take() says how long the collector ran, so a
    slow frame can say whether it was the collector. tick() once per frame.
    """
    def __init__(self, disable: bool = False, collect_every: int = 200) -> None:
        self._t: float | None = None
        self._ms = 0.0
        self._gen = -1
        self._disable = disable
        self._every = max(1, collect_every)
        self._frames = 0

    def _callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t = time.perf_counter()
        elif self._t is not None:
            self._ms += (time.perf_counter() - self._t) * 1000.0
            self._gen = max(self._gen, info.get("generation", 0))
            self._t = None

    def start(self) -> "GcWatch":
        import gc
        gc.collect()
        gc.freeze()
        if self._disable:
            gc.disable()
        gc.callbacks.append(self._callback)
        return self

    def take(self) -> tuple[float, int]:
        """(collector ms since the last take, highest generation run; -1 for none)."""
        out = (round(self._ms, 2), self._gen)
        self._ms, self._gen = 0.0, -1
        return out

    def stop(self) -> None:
        import gc
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
        if self._disable:
            gc.enable()

    def tick(self, braked: bool = False) -> None:
        """
        With collection off, collect the youngest generation on a braked frame or every collect_every frames.
        """
        import gc
        self._frames += 1
        if self._disable and (braked or self._frames % self._every == 0):
            gc.collect(0)
