"""Small helpers shared across pipeline stages.

Purpose:
    Functions more than one module needs, kept in one place so the copies
    can't drift apart. Pure and dependency-free, so any stage can import it
    without pulling in anything else.

Main package:
    clamp(), check_same_frame(), check_frame_size().
    Laps: per-stage wall time, for the pipeline's opt-in stage timing.
    GcWatch: the garbage collector frozen before a control loop and timed in it.
"""
import time


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def check_same_frame(geometry, roi, stage: str) -> None:
    """
    Reject a missing geometry result or ROI crop, or a pair from different frames.

    Inputs:
        geometry, roi: Anything with frame_id and timestamp_ms, normally a
            GeometryBranchResult and the ROICropResult it was run on.
        stage: The caller's name, so the error says which stage refused.

    Raises:
        ValueError: If either is None, or their stamps disagree.
    """
    if geometry is None:
        raise ValueError(f"{stage}: geometry result is None")
    if roi is None:
        raise ValueError(f"{stage}: roi result is None")
    # Both descend from one ROICropResult, so a mismatch means candidates from
    # two frames reached one call and would be labeled with a frame they didn't come from
    if (geometry.frame_id, geometry.timestamp_ms) != (roi.frame_id, roi.timestamp_ms):
        raise ValueError(
            f"{stage}: geometry stamp "
            f"{(geometry.frame_id, geometry.timestamp_ms)} does not match roi stamp "
            f"{(roi.frame_id, roi.timestamp_ms)} — candidates are from different frames"
        )

def check_frame_size(frame, size: tuple[int, int] | None, stage: str) -> None:
    """
    Reject a frame of another size than the one a setting was built for.

    Inputs:
        frame: (H, W, ...) array.
        size: The (H, W) required; None accepts any size.
        stage: The caller's name, so the error says which stage refused.
    Raises:
        ValueError: If the sizes differ.
    """
    if size is not None and tuple(frame.shape[:2]) != tuple(size):
        raise ValueError(f"{stage}: frame is {tuple(frame.shape[:2])}, but the cm scale was set up for {tuple(size)}")


class Laps:
    """
    Wall time per stage, in ms, keyed by stage name.

    start() begins a frame with a fresh times dict; mark() restarts the
    clock without recording; lap(name) records the time since the previous
    lap, mark() or start() under name.
    """
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
    Python's garbage collector, frozen and timed for a control loop.

    Purpose:
        A full (generation 2) collection walks every live object while
        holding the GIL, so every thread stops: the 2026-10-12 runs had
        0.6-1.6 s frames in whatever stage was running (geometry, obstacle,
        capture) with the IMU thread's read stretched to 235 ms at the same
        moment, past the motor watchdog's 0.5 s. start() collects once and
        freezes everything alive (gc.freeze(): the imports, the configs, the
        pipeline), so later collections skip it; take() says how long the
        collector ran since the last call, so a slow frame can say whether it
        was the collector.

    Usage:
        watch = GcWatch().start()      # once, before the loop
        ms, gen = watch.take()         # each frame: collector ms since the last take, the highest generation run
        watch.stop()                   # after the loop
    """
    def __init__(self) -> None:
        self._t: float | None = None
        self._ms = 0.0
        self._gen = -1

    def _callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t = time.perf_counter()
        elif self._t is not None:
            self._ms += (time.perf_counter() - self._t) * 1000.0
            self._gen = max(self._gen, info.get("generation", 0))
            self._t = None

    def start(self) -> "GcWatch":
        """Collect, freeze what's alive, and start timing collections."""
        import gc
        gc.collect()
        gc.freeze()
        gc.callbacks.append(self._callback)
        return self

    def take(self) -> tuple[float, int]:
        """(ms the collector ran since the last take, the highest generation it collected; -1 for none)."""
        out = (round(self._ms, 2), self._gen)
        self._ms, self._gen = 0.0, -1
        return out

    def stop(self) -> None:
        """Stop timing; frozen objects stay frozen (harmless)."""
        import gc
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
