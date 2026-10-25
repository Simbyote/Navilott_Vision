"""Detect range: how far before the stop line the robot reads the light, sign or line right.

Purpose:
    Parked at taped gaps before a stop line, for every state the tester can
    set (light red, yellow, green, off; sign posted or taken away). Each look
    reads OK, WRONG or MISSED; a wrong read anywhere is a hazard, a far miss
    only limits the range (docs/guides/routines.md section 2;
    docs/routines/requests/detect-range.md).

Main package:
    DetectRange (make routine-detect-range). With target=line at several
    angles, tilt_fit() gives intersection.py's TILT_ZERO_DEG and TILT_TO_HEADING.
"""
import statistics

from src.estimation.estimation import CAUTION, GO, STOP
from src.params import GREEN, RED, YELLOW
from src.routines.camera_look import Eyes, parse_numbers, save_frame
from src.routines.harness import Routine, criterion

LIGHT, SIGN, LINE = "light", "sign", "line"
OFF, PRESENT, ABSENT = "off", "present", "absent"
# The line can't be taken away between looks; test a false line as its own session (--notes)
STATES = {LIGHT: (RED, YELLOW, GREEN, OFF), SIGN: (PRESENT, ABSENT), LINE: (PRESENT,)}
EXPECTED = {RED: STOP, YELLOW: CAUTION, GREEN: GO, OFF: GO, PRESENT: True, ABSENT: False}
NOTHING = {LIGHT: GO, SIGN: False, LINE: False}     # what the vote falls back to when it sees nothing
GAPS_CM = "0,10,20,30,45"   # bumper to the stop line's near edge; 0 is on the line
ANGLES_DEG = "0"            # robot turned from square, + = right: "-10,-5,0,5,10" maps the tilt gate
OFFSETS_CM = "0"            # robot shifted from the lane's center, + = right
NEED_CM = 20.0              # every state read right from here in: navigation must know before the line
FRAMES = 40                 # about 2 s at 20 FPS: many times the vote window
SEEN_MIN_PCT = 50.0         # a lit light / posted sign must be seen in this share of frames
OK, MISSED, WRONG = "ok", "missed", "wrong"


def read_look(frames: list[dict], target: str, state: str) -> dict:
    """
    seen_pct: share of frames that saw what's there (for off / absent, anything,
    all wrong). wrong_frames saw what isn't there. voted is the last vote.
    """
    n = len(frames)
    tilt = None
    if target == LIGHT:
        here = state if state != OFF else None
        right = sum(1 for f in frames if here is not None and f["light"] == here)
        wrong = sum(1 for f in frames if f["light"] is not None and f["light"] != here)
        voted = frames[-1]["drive_state"] if frames else None
    elif target == LINE:
        right, wrong = sum(1 for f in frames if f.get("stop_line")), 0
        voted = frames[-1].get("stop_line_voted") if frames else None
        tilts = [f["stop_line_tilt"] for f in frames if f.get("stop_line_tilt") is not None]
        tilt = round(statistics.median(tilts), 2) if tilts else None
    else:
        here = state == PRESENT
        right = sum(1 for f in frames if here and f["sign"])
        wrong = sum(1 for f in frames if not here and f["sign"])
        voted = frames[-1]["stop_sign"] if frames else None
    lit = state not in (OFF, ABSENT)
    seen_pct = round(100.0 * (right if lit else wrong) / n, 1) if n else 0.0
    expected = EXPECTED[state]
    if voted == expected and (not lit or seen_pct >= SEEN_MIN_PCT):
        result = OK
    elif voted is not None and voted != expected and (voted != NOTHING[target] or
                                                       (n and 100.0 * wrong / n >= SEEN_MIN_PCT)):
        result = WRONG          # a red seen as green votes "go" like nothing seen: the frames tell
    else:
        result = MISSED
    return {"frames": n, "seen_pct": seen_pct, "wrong_frames": wrong, "voted": voted,
            "expected": expected, "tilt_deg": tilt, "result": result}


def tilt_fit(rows: list[dict]) -> dict | None:
    """
    Least-squares tilt = zero + slope x angle: TILT_ZERO_DEG = zero, TILT_TO_HEADING = 1 / slope.

    None with fewer than two angles or a tilt that doesn't move with angle.
    """
    pts = [(float(r["angle_deg"]), float(r["tilt_deg"])) for r in rows if r.get("tilt_deg") is not None]
    if len({a for a, _ in pts}) < 2:
        return None
    ma, mt = statistics.fmean(a for a, _ in pts), statistics.fmean(t for _, t in pts)
    sxx = sum((a - ma) ** 2 for a, _ in pts)
    slope = sum((a - ma) * (t - mt) for a, t in pts) / sxx
    if abs(slope) < 1e-6:
        return None
    return {"tilt_zero_deg": round(mt - slope * ma, 2), "tilt_to_heading": round(1.0 / slope, 2),
            "points": len(pts)}


def reliable_range(results: list[tuple[float, str]]) -> float | None:
    """Largest gap up to which every trial read OK; None if the nearest didn't."""
    best = None
    for gap in sorted({g for g, _ in results}):
        if any(r != OK for g, r in results if g == gap):     # sorted: the nearer gaps passed already
            break
        best = gap
    return best


def pose_text(angle: float, offset: float) -> str:
    if not angle and not offset:
        return "centered"
    turn = f"turned {abs(angle):g} deg {'right' if angle > 0 else 'left'}" if angle else "square"
    shift = f"{abs(offset):g} cm {'right' if offset > 0 else 'left'} of center" if offset else "centered"
    return f"{turn}, {shift}"


def pose_tag(angle: float, offset: float) -> str:
    return (f"_{angle:g}deg" if angle else "") + (f"_{offset:g}cm" if offset else "")


class DetectRange(Routine):
    name = "detect-range"
    title = "Detect range: the traffic light, stop sign or stop line, parked at gaps"
    question = ("From how far before the stop line does the robot read the traffic light (or stop sign, "
                "or the line) right, and from which angles and offsets?")
    requirement = "D3 (light colour), D2 (stop sign), the stop line"
    trials = len(parse_numbers(GAPS_CM)) * len(STATES[LIGHT])
    fields = ("gap_cm", "angle_deg", "offset_cm", "state", "frames", "seen_pct", "wrong_frames", "voted",
              "expected", "tilt_deg", "result")
    settings = {"target": "light, sign or line (default light)",
                "gaps": f"gaps before the stop line in cm, comma separated (default {GAPS_CM})",
                "angles": f"robot turned from square, deg, + = right, comma separated (default {ANGLES_DEG})",
                "offsets": f"robot shifted from the lane's center, cm, + = right (default {OFFSETS_CM})",
                "need_cm": f"every state must read right at gaps up to this (default {NEED_CM:g})",
                "frames": f"frames per look (default {FRAMES})"}
    instructions = """\
Set up: an intersection as on the course, its light (or stop sign) posted
where it will be on the day, the room lit as it will be. Mark the gaps on the
lane: bumper to the stop line's near edge. Motors stay off; nothing drives.
Each gap: place the robot on the mark, centered and pointing along the lane
(or at the angle and offset the prompt asks: turn it about the bumper's
middle; a protractor taped square to the line helps). Then, for each state
the prompt asks (light red, yellow, green, off; or the sign there and taken
away), set it and press Enter: the robot looks for ~2 s. With target=line
there is nothing to set: each pose is one look at the line.
Keep people and bright clothes out of the view while it looks."""

    def __init__(self, eyes=None):
        self._eyes, self._need = eyes, NEED_CM

    @staticmethod
    def _defaults(options: dict) -> None:
        options.setdefault("target", LIGHT)
        options.setdefault("gaps", GAPS_CM)
        options.setdefault("angles", ANGLES_DEG)
        options.setdefault("offsets", OFFSETS_CM)
        options.setdefault("need_cm", NEED_CM)
        options.setdefault("frames", FRAMES)
        if options["target"] not in STATES:
            raise ValueError(f"target must be light, sign or line, not {options['target']!r}")

    @staticmethod
    def _poses(options: dict) -> list[tuple[float, float]]:
        return [(a, o) for a in parse_numbers(options["angles"]) for o in parse_numbers(options["offsets"])]

    def plan(self, options):
        o = dict(options)
        self._defaults(o)
        return len(parse_numbers(o["gaps"])) * len(self._poses(o)) * len(STATES[o["target"]])

    def setup(self, ctx):
        self._defaults(ctx.options)
        self._need = float(ctx.options["need_cm"])
        self._eyes = (self._eyes or Eyes()).open()
        ctx.state["results"] = {}

    def teardown(self, ctx):
        if self._eyes is not None:
            self._eyes.close()
        done = list(ctx.state.get("results", {}).values())
        if not done:
            return
        rng = reliable_range([(d["gap_cm"], d["result"]) for d in done])
        ctx.state["range_cm"] = rng
        ctx.console.say("\nrange: " + ("not even at the nearest gap" if rng is None else
                                       f"every state read right up to {rng:g} cm before the line"))
        poses = list(dict.fromkeys((d["angle_deg"], d["offset_cm"]) for d in done))
        if len(poses) > 1:
            per = {}
            for a, o in poses:
                r = reliable_range([(d["gap_cm"], d["result"]) for d in done
                                    if (d["angle_deg"], d["offset_cm"]) == (a, o)])
                per[f"{a:g}deg {o:g}cm"] = r
                ctx.console.say(f"  {pose_text(a, o)}: " + ("not even at the nearest gap" if r is None
                                                             else f"up to {r:g} cm"))
            ctx.state["pose_range_cm"] = per
        fit = tilt_fit(done) if ctx.options["target"] == LINE else None
        if fit:
            ctx.state["tilt_fit"] = fit
            ctx.console.say(f"approach angle (navigation/intersection.py): TILT_ZERO_DEG = {fit['tilt_zero_deg']:g}, "
                            f"TILT_TO_HEADING = {fit['tilt_to_heading']:g}  ({fit['points']} looks)")

    def trial(self, ctx, i):
        target = ctx.options["target"]
        gaps, states, poses = parse_numbers(ctx.options["gaps"]), STATES[target], self._poses(ctx.options)
        per_gap = len(poses) * len(states)
        gap = gaps[i // per_gap]
        angle, offset = poses[(i % per_gap) // len(states)]
        state = states[i % len(states)]
        if i % len(states) == 0:
            ctx.console.wait(f"Robot's bumper {gap:g} cm before the stop line, {pose_text(angle, offset)}? Enter"
                             + (" looks" if target == LINE else ""))
        if target != LINE:
            what = {RED: "light RED", YELLOW: "light YELLOW", GREEN: "light GREEN", OFF: "light OFF (or covered)",
                    PRESENT: "the stop sign IN PLACE", ABSENT: "the stop sign TAKEN AWAY"}[state]
            ctx.console.wait(f"Set {what}, step out of view; Enter looks")
        frames, last = self._eyes.look(int(ctx.options["frames"]))
        row = {"gap_cm": gap, "angle_deg": angle, "offset_cm": offset, "state": state,
               **read_look(frames, target, state)}
        save_frame(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}_{gap:g}cm{pose_tag(angle, offset)}_{state}.jpg", last)
        ctx.state["results"][str(i)] = {k: row[k] for k in ("gap_cm", "angle_deg", "offset_cm", "state",
                                                            "tilt_deg", "result")}
        if row["result"] != OK:
            ctx.console.say(f"  {row['result'].upper()}: voted {row['voted']}, expected {row['expected']}")
        elif target == LINE and row["tilt_deg"] is not None:
            ctx.console.say(f"  tilt {row['tilt_deg']:g} deg")
        return row

    def judge(self, rows):
        need = self._need
        return [criterion(f"trials not read right within {need:g} cm",
                          sum(1 for r in rows if r["gap_cm"] <= need and r["result"] != OK), "<=", 0),
                criterion("wrong reads at any gap (something that isn't there)",
                          sum(1 for r in rows if r["result"] == WRONG), "<=", 0)]
