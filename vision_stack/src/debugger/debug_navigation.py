"""Navigation view: the navigation run's video, rendered after the run from what it recorded.

Purpose:
    navigation_linker draws nothing while the robot drives, so rendering
    can't slow the control loop. It records each frame's JPEG and a record of
    what the pipeline and the navigator decided; this module turns the run
    folder into video afterwards: the Phase 3 view (debug_phase3) exactly as
    the live decisions were, with a navigation strip under it. Nothing is
    re-run, so the video shows what happened, not a re-computation. The
    recording format (frames/, records.pkl) is maneuver_linker's, read with
    debug_maneuver's helpers.

Main package:
    render_run(): run folder in, nav.avi and nav_video.csv out.
    draw_strip(): the strip for one frame: DRIVE or BRAKE and why, which
    navigation rule decided it, the stop-line phase and the route step, both
    wheel duties as bars (with the stall duty marked), the steering and what
    it came from, and the packet fields the navigator steered by.

Flow:
    1. Read records.pkl in order; load each frame's JPEG (skipping any the
       recorder dropped).
    2. Rebuild the Phase3Result shape Phase3View reads; render it.
    3. Stack the navigation strip under it; record video and CSV.
"""
import os

import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.debugger.debug_maneuver import as_result, frame_path, read_records
from src.debugger.debug_phase3 import Phase3View
from src.navigation.navigation_contract import STALL_DUTY
from src.navigation.route import TURNS_TBD

VIDEO_FILE, VIDEO_CSV = "nav.avi", "nav_video.csv"
STRIP_LINES = 4


def _fmt(v, spec: str) -> str:
    return "--" if v in ("", None) else format(v, spec)


def _duty_bar(img, label: str, duty, y0: int, y1: int, x0: int, x1: int, s: int, fs: float, th: int) -> None:
    """One wheel's duty as a bar from the center: right of it forward, left reverse, stall duty marked."""
    lw = 16 * s
    dv.draw_text(img, label, (x0, y1 - 1 * s), dv.C_WHITE, fs, th)
    bx0, bx1 = x0 + lw, x1
    mid = (bx0 + bx1) // 2
    half = (bx1 - bx0) // 2
    cv2.rectangle(img, (bx0, y0), (bx1, y1), dv.C_GRAY, 1)
    for sign in (1, -1):
        x = mid + sign * int(STALL_DUTY * half)
        cv2.line(img, (x, y0), (x, y1), dv.C_AMBER, 1)
    cv2.line(img, (mid, y0), (mid, y1), dv.C_WHITE, 1)
    if duty not in ("", None) and duty != 0:
        end = mid + int(max(-1.0, min(1.0, duty)) * half)
        color = dv.C_USABLE if duty > 0 else dv.C_RED
        cv2.rectangle(img, (min(mid, end), y0 + 2 * s), (max(mid, end), y1 - 2 * s), color, -1)


def draw_strip(n: dict, width: int, scale: int = 1) -> np.ndarray:
    """
    The navigation strip for one frame.

    Inputs:
        n: The frame's record["nav"] from navigation_linker.
        width: Output width in px, the Phase 3 view's.
    """
    s = max(1, int(scale))
    fs, th, lh = 0.38 * s, max(1, s // 2), 16 * s
    img = np.zeros((STRIP_LINES * lh, width, 3), np.uint8)

    rule = (f"  [{n['rule']}{' / ' + n['phase'] if n.get('phase') else ''}"
            f"{' / step ' + n['step'] if n.get('step') else ''}{' TBD' if n.get('maneuver') in TURNS_TBD else ''}]") if n.get("rule") else ""
    if n.get("brake"):
        head, color = f"BRAKE  {n.get('reason', '')}{rule}", dv.C_RED
    else:
        head, color = f"DRIVE{rule}", dv.C_USABLE
    head += f"   t={_fmt(n.get('t'), '.2f')}s"
    dv.draw_text(img, head, (6 * s, lh - 3 * s), color, fs * 1.1, th)
    if n.get("event"):
        (tw, _), _ = cv2.getTextSize(head, dv.FONT, fs * 1.1, th)
        dv.draw_text(img, n["event"], (6 * s + tw + 12 * s, lh - 3 * s), dv.C_AMBER, fs, th)

    dv.draw_text(img, (f"cmd {_fmt(n.get('cmd_left'), '+.2f')}/{_fmt(n.get('cmd_right'), '+.2f')}  "
                       f"steer {_fmt(n.get('steer'), '+.3f')} ({n.get('source') or '--'})  "
                       f"lane {n.get('lane_status') or '--'} {_fmt(n.get('lane_offset'), '+.2f')}"
                       f"{'' if n.get('lane_offset_cm') in ('', None) else ' / ' + format(n['lane_offset_cm'], '+.1f') + 'cm'}  "
                       f"hdg {_fmt(n.get('heading_error'), '+.1f')}"),
                 (6 * s, 2 * lh - 3 * s), dv.C_WHITE, fs, th)

    x0, x1 = 6 * s, width - 6 * s
    xm = (x0 + x1) // 2
    _duty_bar(img, "L", n.get("cmd_left"), 2 * lh + 2 * s, 3 * lh - 2 * s, x0, xm - 6 * s, s, fs, th)
    _duty_bar(img, "R", n.get("cmd_right"), 2 * lh + 2 * s, 3 * lh - 2 * s, xm + 6 * s, x1, s, fs, th)

    line = n.get("stop_line_cm")
    dv.draw_text(img, (f"stop line {'--' if line in ('', None) else format(line, '.1f') + ' cm'}  "
                       f"light {n.get('drive_state') or '--'}  sign {'YES' if n.get('stop_sign') else 'no'}  "
                       f"cps {_fmt(n.get('left_cps'), '.0f')}/{_fmt(n.get('right_cps'), '.0f')}  "
                       f"latency {_fmt(n.get('latency_ms'), '.0f')} ms"),
                 (6 * s, 4 * lh - 4 * s), dv.C_GRAY, fs, th)
    return img


def render_run(out_dir: str, lane_config, fps: float, scale: int = 1, on_frame=None) -> dict:
    """
    Render a run folder's recording to nav.avi and nav_video.csv.

    Inputs:
        lane_config: The LaneOffsetConfig the run used, for the lane
            overlay's candidate gates.
        fps: The video's rate; the run's measured rate plays it at real speed.
        on_frame: Called with each finished image, e.g. to show it.

    Outputs:
        {"rendered": frames written, "missing": records whose JPEG wasn't
        found, "report": the Phase 3 view's summary lines}.

    Side effects:
        Writes nav.avi and nav_video.csv into out_dir.
    """
    view = Phase3View(lane_config, fps)
    writer = dv.ViewWriter(os.path.join(out_dir, VIDEO_FILE), Phase3View.CSV_FIELDS,
                           fps=fps, csv_path=os.path.join(out_dir, VIDEO_CSV))
    rendered = missing = 0
    try:
        for rec in read_records(out_dir):
            frame = cv2.imread(frame_path(out_dir, rec["frame_id"]))
            if frame is None:
                missing += 1
                continue
            data = view.extract(as_result(rec, frame), frame)
            view.observe(data)
            top = view.render(data, scale)
            img = np.vstack([top, draw_strip(rec["nav"], top.shape[1], scale)])
            writer.push(img, view.row(data))
            rendered += 1
            if on_frame is not None:
                on_frame(img)
    finally:
        writer.close()
    return {"rendered": rendered, "missing": missing, "report": view.report()}
