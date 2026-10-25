"""The parked robot's camera through Phases 1-3, for routines that judge what it sees without driving.

Purpose:
    detect-range, lane-offset and obstacle-check place the robot, then look
    for a second or two. Each look runs a run's chain (run_phase3_chain,
    MEASURED) with a fresh Phase 3 processor, so votes start from nothing as
    they would at a new scene.

Main package:
    Eyes.look(n) -> n Seen dicts and the last frame, saved beside the row.
"""
from src.params import STOP_SIGN, TRAFFIC_LIGHT

MAX_READS_PER_FRAME = 3         # two drops in three reads ends the look short


def parse_numbers(v) -> list[float]:
    """A --set list, "0,10,20" or one number, as floats."""
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in str(v).replace(";", ",").split(",") if x.strip()]


def seen(res, p3) -> dict:
    """
    One frame's result, for comparing with the scene.

    light and sign count only at or over Phase 3's gate; drive_state, stop_sign
    and the *_voted keys are votes after this frame. lane_offset is [-1, 1] of
    half the ROI, + = robot right; p2_* is Phase 2's raw measurement.
    obstacle_cm is None without cm, *_clipped when the base is below the view.
    stop_line_tilt (deg, + = right end nearer) is None unless seen this frame.
    """
    light = sign = None
    for d in res.chain.phase2.detections:
        if d.type == TRAFFIC_LIGHT and d.confidence >= p3.min_confidence_traffic:
            light = d.label_detail
        elif d.type == STOP_SIGN and d.confidence >= p3.min_confidence_sign:
            sign = True
    pk, off = res.packet, res.chain.offset
    ob = next((r for r in getattr(res.chain.phase2, "obstacle_results", []) if r.detected), None)
    line = getattr(res.chain, "stop_line", None)
    return {"light": light, "sign": bool(sign), "drive_state": pk.drive_state, "stop_sign": pk.stop_sign_detected,
            "lane_offset": pk.lane_offset, "lane_status": pk.lane_status, "p2_offset": off.offset,
            "lane_mode": off.mode, "lane_width_px": off.lane_width_px,
            "obstacle": ob is not None, "obstacle_cm": None if ob is None else ob.distance_cm,
            "obstacle_clipped": ob is not None and ob.clipped,
            "obstacle_voted": bool(getattr(pk, "obstacle_detected", False)),
            "obstacle_voted_cm": getattr(pk, "obstacle_distance_cm", None),
            "obstacle_voted_clipped": bool(getattr(pk, "obstacle_clipped", False)),
            "stop_line": bool(line is not None and line.detected),
            "stop_line_voted": bool(getattr(pk, "stop_line_detected", False)),
            "stop_line_tilt": getattr(pk, "stop_line_tilt_deg", None),
            "total_ms": res.timings_ms["phase2"] + res.timings_ms["phase3"]}


def save_frame(path, frame) -> bool:
    """JPEG at path; False with no frame or a failed write."""
    if frame is None:
        return False
    import cv2
    try:
        return bool(cv2.imwrite(str(path), frame))
    except cv2.error:
        return False


class Eyes:
    """source None opens the robot's camera in open(); config and p3 default to MEASURED tuning."""
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
        """Seen dicts for n frames through a fresh processor, and the last frame."""
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
        """Lane ROI width in px, what lane_offset's [-1, 1] is half of."""
        from src.perception.roi_crop import resolve
        return int(resolve(self.config.roi.lane, frame.shape[:2])[2])

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
