"""The parked robot's camera through Phases 1-3, for routines that judge what it sees without driving.

Purpose:
    detect-range and lane-offset place the robot, then ask what the
    pipeline makes of the scene for a second or two. Eyes holds the camera
    open for the whole routine and runs each look through the same chain a
    run uses (phase3_linker.run_phase3_chain, MEASURED tuning) with a fresh
    Phase 3 processor, so a look's votes and filters start from nothing,
    as they would arriving at a new scene. Each frame comes back as a plain
    dict of what the routines compare (Seen), and the look's last frame is
    kept to save beside the row, so a wrong answer can be looked at.

Main package:
    Eyes: open(), look(n) -> (list of Seen dicts, last frame),
        roi_width_px(frame), close().
    seen(res, p3): one Phase3Result as a Seen dict.
    save_frame(path, frame): the frame as a JPEG; False when it can't.
    parse_numbers(v): a --set list ("0,10,20", or one number) as floats.

Flow (a look):
    read frames until n have gone through (drops skipped, at most
    MAX_READS_PER_FRAME reads a frame) -> run_phase3_chain on each with this
    look's processor -> seen() per frame.
"""
from src.params import STOP_SIGN, TRAFFIC_LIGHT

MAX_READS_PER_FRAME = 3         # a camera that drops two reads in three has failed: the look ends short


def parse_numbers(v) -> list[float]:
    """A --set list: "0,10,20" or a single number, as floats."""
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in str(v).replace(";", ",").split(",") if x.strip()]


def seen(res, p3) -> dict:
    """
    What one frame's result says, for comparing with the scene.

    light: the colour of the frame's traffic light at or over Phase 3's
        confidence gate (what its vote counts), else None.
    sign: a stop sign at or over its gate this frame.
    drive_state, stop_sign: the votes after this frame.
    lane_offset, lane_status: the filtered offset ([-1, 1] of half the lane
        ROI, + = robot right of center) and its status.
    p2_offset, lane_mode, lane_width_px: Phase 2's raw lane measurement.
    total_ms: Phases 2 and 3's time on the frame.
    """
    light = sign = None
    for d in res.chain.phase2.detections:
        if d.type == TRAFFIC_LIGHT and d.confidence >= p3.min_confidence_traffic:
            light = d.label_detail
        elif d.type == STOP_SIGN and d.confidence >= p3.min_confidence_sign:
            sign = True
    pk, off = res.packet, res.chain.offset
    return {"light": light, "sign": bool(sign), "drive_state": pk.drive_state, "stop_sign": pk.stop_sign_detected,
            "lane_offset": pk.lane_offset, "lane_status": pk.lane_status, "p2_offset": off.offset,
            "lane_mode": off.mode, "lane_width_px": off.lane_width_px,
            "total_ms": res.timings_ms["phase2"] + res.timings_ms["phase3"]}


def save_frame(path, frame) -> bool:
    """The frame as a JPEG at path; False when there's no frame or it can't be written."""
    if frame is None:
        return False
    import cv2
    try:
        return bool(cv2.imwrite(str(path), frame))
    except cv2.error:
        return False


class Eyes:
    """
    The camera, held open, and looks through Phases 1-3.

    source: a live_view FrameSource; None opens the robot's camera at the
        pipeline's size and rate in open().
    config, p3: Phase 2 and Phase 3 tuning (MEASURED, MEASURED_ESTIMATION
        when None).
    """
    def __init__(self, source=None, config=None, p3=None):
        self.source, self.config, self.p3 = source, config, p3

    def open(self) -> "Eyes":
        from src.config import MEASURED, MEASURED_ESTIMATION
        self.config = self.config or MEASURED
        self.p3 = self.p3 or MEASURED_ESTIMATION
        if self.source is None:
            from src.debugger.live_view import CameraFrameSource
            from src.params import CAMERA_CONTROLS, FPS, FRAME_H, FRAME_W
            self.source = CameraFrameSource(FRAME_W, FRAME_H, FPS, CAMERA_CONTROLS)
        return self

    def look(self, n: int) -> tuple[list[dict], object]:
        """n frames through Phases 1-3 with a fresh processor: their Seen dicts and the last frame."""
        from src.phase3_linker import make_processor, run_phase3_chain
        out, last, processor = [], None, None
        for _ in range(n * MAX_READS_PER_FRAME):
            if len(out) >= n:
                break
            item = self.source.read()
            if item is None:
                break                                   # the source ended
            frame, fid, ts = item
            if frame is None:
                continue                                # a dropped frame
            if processor is None:
                processor = make_processor(frame, fid, ts, self.config, self.p3)
            res = run_phase3_chain(frame, fid, ts, processor, None, self.config)
            out.append(seen(res, self.p3))
            last = frame
        return out, last

    def roi_width_px(self, frame) -> int:
        """The lane ROI's width in px for this frame's size: what lane_offset's [-1, 1] is half of."""
        from src.perception.roi_crop import resolve
        return int(resolve(self.config.roi.lane, frame.shape[:2])[2])

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
