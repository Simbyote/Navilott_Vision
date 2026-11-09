"""Lane offset: does the robot's reported lane position match a ruler within 2 cm?

Purpose:
    P3 prompted: parked at measured offsets, reported cm against the ruler.
    The first, centered look sets the ground scale (LANE_CM / median lane
    width px) and the noise floor. Once it passes, recover-offset can trust
    the robot's offset.

Main package:
    LaneOffset (make routine-lane-offset); its summary gives config.py's cm_per_px.
"""
import statistics

from src.estimation.estimation import LANE_VISION
from src.routines.camera_look import Eyes, parse_numbers, save_frame
from src.routines.harness import Routine, criterion

POSITIONS_CM = "0,2,-2,4,-4"    # requirements.md P3: 0, +-2, +-4; + = robot right of the lane's center
LANE_CM = 14.0                  # course.md: lane width, line center to line center
FRAMES = 100                    # the procedure's --limit 100
TOLERANCE_CM = 2.0              # P3


def to_cm(offset: float | None, roi_width_px: int | None, cm_per_px: float | None) -> float | None:
    """lane_offset in cm as estimation's _to_cm does; None without a scale."""
    if offset is None or roi_width_px is None or cm_per_px is None:
        return None
    return offset * (roi_width_px / 2.0) * cm_per_px


def offset_look(frames: list[dict]) -> dict:
    """offset and offset_sd over frames on vision (None without), vision_pct, median lane_width_px."""
    on = [f["lane_offset"] for f in frames if f["lane_status"] == LANE_VISION]
    widths = [f["lane_width_px"] for f in frames if f.get("lane_width_px")]
    return {"frames": len(frames), "vision_pct": round(100.0 * len(on) / len(frames), 1) if frames else 0.0,
            "offset": statistics.fmean(on) if on else None,
            "offset_sd": statistics.stdev(on) if len(on) > 1 else None,
            "lane_width_px": statistics.median(widths) if widths else None}


class LaneOffset(Routine):
    name = "lane-offset"
    title = "Lane offset: reported position against a ruler"
    question = "Does the robot's reported position in the lane match a ruler, to within 2 cm?"
    requirement = "P3"
    trials = len(parse_numbers(POSITIONS_CM))
    fields = ("planned_cm", "measured_cm", "reported_cm", "error_cm", "sd_cm", "vision_pct", "frames",
              "lane_width_px", "offset")
    settings = {"positions": f"planned offsets in cm, + = right, the first 0 (default {POSITIONS_CM})",
                "lane_cm": f"the lane's width, line center to line center (default {LANE_CM:g})",
                "frames": f"frames per look (default {FRAMES})"}
    instructions = f"""\
Set up: a straight lane, both lines in view, lit as on the course. Nothing
drives. Measure from the lane's center (halfway between the two lines'
centers) to the robot's centerline, square to the lane: + when the robot sits
RIGHT of center. The first position is centered: it measures the scale
(lane {LANE_CM:g} cm wide) and the noise. Keep the robot pointing along the lane
at every position; only slide it sideways."""

    def __init__(self, eyes=None):
        self._eyes = eyes

    @staticmethod
    def _defaults(options: dict) -> None:
        options.setdefault("positions", POSITIONS_CM)
        options.setdefault("lane_cm", LANE_CM)
        options.setdefault("frames", FRAMES)
        if parse_numbers(options["positions"])[:1] != [0.0]:
            raise ValueError("the first position must be 0: it measures the scale")

    def plan(self, options):
        o = dict(options)
        self._defaults(o)
        return len(parse_numbers(o["positions"]))

    def setup(self, ctx):
        self._defaults(ctx.options)
        self._eyes = (self._eyes or Eyes()).open()
        ctx.state["cm_per_px"] = None

    def teardown(self, ctx):
        if self._eyes is not None:
            self._eyes.close()
        scale = ctx.state.get("cm_per_px")
        if scale is not None:
            ctx.console.say(f"\nground scale {scale:.5f} cm/px (lane {ctx.options['lane_cm']:g} cm = "
                            f"{ctx.state['lane_width_px']:g} px): set cm_per_px={scale:.5f} in "
                            "MEASURED_ESTIMATION (config.py) for lane_offset_cm in every run")

    def trial(self, ctx, i):
        planned = parse_numbers(ctx.options["positions"])[i]
        where = "centered in the lane" if planned == 0 else \
            f"{abs(planned):g} cm {'RIGHT' if planned > 0 else 'LEFT'} of center"
        ctx.console.wait(f"Robot {where}, pointing along the lane? Enter")
        measured = ctx.console.ask_number("Offset you measured (+ = right of center)", lo=-15.0, hi=15.0, unit="cm")
        frames, last = self._eyes.look(int(ctx.options["frames"]))
        s = offset_look(frames)
        save_frame(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}_{planned:+g}cm.jpg", last)
        roi = self._eyes.roi_width_px(last) if last is not None else None
        if i == 0:
            width = s["lane_width_px"]
            ctx.state["lane_width_px"] = width
            ctx.state["cm_per_px"] = None if not width else float(ctx.options["lane_cm"]) / width
            if not width:
                ctx.console.say("  no lane width measured: no scale, so no cm (are both lines in view?)")
        scale = ctx.state["cm_per_px"]
        reported, sd = to_cm(s["offset"], roi, scale), to_cm(s["offset_sd"], roi, scale)
        return {"planned_cm": planned, "measured_cm": measured,
                "reported_cm": None if reported is None else round(reported, 2),
                "error_cm": None if reported is None else round(reported - measured, 2),
                "sd_cm": None if sd is None else round(sd, 2), "vision_pct": s["vision_pct"], "frames": s["frames"],
                "lane_width_px": s["lane_width_px"],
                "offset": None if s["offset"] is None else round(s["offset"], 4)}

    def judge(self, rows):
        errors = [r["error_cm"] for r in rows]
        worst = None if any(e is None for e in errors) else max(abs(e) for e in errors)
        return [criterion("worst |reported - measured|", worst, "<=", TOLERANCE_CM, "cm")]
