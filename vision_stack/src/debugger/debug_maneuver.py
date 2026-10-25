"""Maneuver view: the drive trial's video, rendered after the run from what it recorded.

Purpose:
    maneuver_linker draws nothing while moving, so rendering can't slow the
    control loop; it records JPEGs and a pickled record per frame. This
    renders the Phase 3 view with a maneuver strip, re-running nothing.

Main package:
    render_run(): run folder in, maneuver.avi and maneuver_video.csv out.
"""
import os
import pickle
from types import SimpleNamespace

import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.debugger.debug_phase3 import Phase3View
from src.maneuver import (
    ABORTED, DONE, FORWARD_STEPS, HOLD, PULSE_LEFT, PULSE_LEFT_REST, PULSE_RIGHT, PULSE_RIGHT_REST,
    SETTLE, TURN, TURN_SETTLE, ManeuverConfig,
)

C_TURN = (255, 255, 0)
STEP_COLORS = {**{s: dv.C_USABLE for s in FORWARD_STEPS},
               TURN: C_TURN, TURN_SETTLE: C_TURN, ABORTED: dv.C_RED, DONE: dv.C_WHITE, HOLD: (255, 0, 255),
               **{s: dv.C_AMBER for s in (SETTLE, PULSE_LEFT, PULSE_LEFT_REST, PULSE_RIGHT, PULSE_RIGHT_REST)}}
FRAMES_DIR, RECORDS_FILE = "frames", "records.pkl"


def frame_path(out_dir: str, frame_id: int) -> str:
    return os.path.join(out_dir, FRAMES_DIR, f"{frame_id:06d}.jpg")


def read_records(out_dir: str):
    """The run's records in order, stopping at the end or a record a crash cut short."""
    path = os.path.join(out_dir, RECORDS_FILE)
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                return


def as_result(rec: dict, frame: np.ndarray) -> SimpleNamespace:
    """A record in the shape Phase3View.extract() reads."""
    chain = SimpleNamespace(frame=SimpleNamespace(frame=frame), geometry=rec["geometry"],
                            offset=rec["offset"], offset_debug=rec["offset_debug"],
                            obstacle=rec.get("obstacle"),       # absent in older recordings
                            roi=SimpleNamespace(lane_rect=rec["lane_rect"], obstacle_rect=rec.get("obstacle_rect")))
    return SimpleNamespace(chain=chain, packet=rec["packet"], p3_debug=rec["p3_debug"],
                           timings_ms=rec["timings"])


def draw_strip(m: dict, cfg: ManeuverConfig, width: int, scale: int = 1) -> np.ndarray:
    """One frame's maneuver strip; cfg sets the leg length and turn target the bars run to."""
    s = max(1, int(scale))
    fs, th, lh = 0.38 * s, max(1, s // 2), 16 * s
    img = np.zeros((4 * lh, width, 3), np.uint8)
    step = m.get("step", "")
    fmt = lambda v, spec: "--" if v in ("", None) else format(v, spec)

    head = f"{step} {m['hold_point']}" if m.get("hold_point") else step
    head += f"   t={fmt(m.get('t'), '.2f')}s"
    dv.draw_text(img, head, (6 * s, lh - 3 * s), STEP_COLORS.get(step, dv.C_WHITE), fs * 1.1, th)
    if m.get("event"):
        (tw, _), _ = cv2.getTextSize(head, dv.FONT, fs * 1.1, th)
        color = dv.C_RED if m["event"].startswith("ABORT") else dv.C_GRAY
        dv.draw_text(img, m["event"], (6 * s + tw + 12 * s, lh - 3 * s), color, fs, th)

    cmd = "BRAKE" if m.get("brake") else f"{fmt(m.get('cmd_left'), '+.2f')}/{fmt(m.get('cmd_right'), '+.2f')}"
    dv.draw_text(img, (f"cmd {cmd}  "
                       f"cnt {fmt(m.get('c_counts'), '+.3f')} hdg {fmt(m.get('c_heading'), '+.3f')}  "
                       f"cps {fmt(m.get('left_cps'), '.0f')}/{fmt(m.get('right_cps'), '.0f')}  "
                       f"yaw {fmt(m.get('yaw_corrected'), '+.0f')}"),
                 (6 * s, 2 * lh - 3 * s), dv.C_WHITE, fs, th)

    x0, x1, y0, y1 = 6 * s, width - 6 * s, 3 * lh - 8 * s, 3 * lh + 2 * s
    cv2.rectangle(img, (x0, y0), (x1, y1), dv.C_GRAY, 1)
    frac = lambda v, top: max(0.0, min(1.0, v / top)) if top else 0.0
    if step in FORWARD_STEPS and m.get("leg_progress") not in ("", None):
        f = frac(m["leg_progress"], cfg.leg_counts)
        cv2.rectangle(img, (x0, y0), (x0 + int(f * (x1 - x0)), y1), dv.C_USABLE, -1)
        label = f"leg {m['leg_progress']:.0f} / {cfg.leg_counts} counts   heading {fmt(m.get('heading_deg'), '+.1f')} deg"
    elif step in (TURN, TURN_SETTLE) and m.get("turn_deg") not in ("", None):
        top = cfg.turn_target_deg + cfg.turn_tolerance_deg * 4       # room to show overshoot
        lo, hi = (cfg.turn_target_deg - cfg.turn_tolerance_deg, cfg.turn_target_deg + cfg.turn_tolerance_deg)
        cv2.rectangle(img, (x0 + int(frac(lo, top) * (x1 - x0)), y0),
                      (x0 + int(frac(hi, top) * (x1 - x0)), y1), (0, 90, 0), -1)
        cv2.rectangle(img, (x0, y0 + 2 * s), (x0 + int(frac(m["turn_deg"], top) * (x1 - x0)), y1 - 2 * s), C_TURN, -1)
        label = f"turn {m['turn_deg']:.1f} / {cfg.turn_target_deg:.0f} deg (pass +/- {cfg.turn_tolerance_deg:.0f})"
    else:
        label = ""
    if label:
        dv.draw_text(img, label, (x0 + 4 * s, 4 * lh - 4 * s), dv.C_WHITE, fs, th)
    return img


def render_run(out_dir: str, lane_config, cfg: ManeuverConfig, fps: float, scale: int = 1,
               on_frame=None) -> dict:
    """
    Render a run folder to maneuver.avi and maneuver_video.csv.

    fps at the run's measured rate plays it at real speed. Returns rendered,
    missing (no JPEG) and the Phase 3 view's report.
    """
    view = Phase3View(lane_config, fps)
    writer = dv.ViewWriter(os.path.join(out_dir, "maneuver.avi"), Phase3View.CSV_FIELDS,
                           fps=fps, csv_path=os.path.join(out_dir, "maneuver_video.csv"))
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
            img = np.vstack([top, draw_strip(rec["maneuver"], cfg, top.shape[1], scale)])
            writer.push(img, view.row(data))
            rendered += 1
            if on_frame is not None:
                on_frame(img)
    finally:
        writer.close()
    return {"rendered": rendered, "missing": missing, "report": view.report()}
