"""Small helpers shared across pipeline stages.

Purpose:
    Functions more than one module needs, kept in one place so the copies
    can't drift apart. Pure and dependency-free, so any stage can import it
    without pulling in anything else.

Main package:
    None; helper functions only.
"""


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