"""Small helpers shared across pipeline stages.

Purpose:
    Functions more than one module needs, kept in one place so the copies
    can't drift apart. Pure and dependency-free, so any stage can import it
    without pulling in anything else.

Main package:
    clamp(), check_same_frame(), check_frame_size().
    Laps: per-stage wall time, for the pipeline's opt-in stage timing.
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
