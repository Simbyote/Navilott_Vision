"""Detect range: from how far before the stop line does the robot read the traffic light (or the stop sign) right?

Purpose:
    D3 (light colour) and D2 (stop sign) have never been checked on the
    robot. Navigation acts on the votes, so this asks what the votes say
    with the robot parked at taped gaps before an intersection's stop line,
    the light or sign posted as on the course, for every state the tester
    can set: the light red, yellow, green and off; the sign there and taken
    away. Each trial is one look (camera_look.Eyes: Phases 1-3 with a fresh
    processor, `frames` frames, ~2 s): the share of frames that saw the
    right thing, the frames that saw something else, and the vote at the
    end. The look's last frame is saved beside the row.

    A trial reads OK when the final vote is the expected one and, for a lit
    light or a posted sign, at least SEEN_MIN_PCT of the frames saw it (a
    missed green votes "go" like a real one: the frames tell them apart).
    It's WRONG when the vote names something that isn't there (red read as
    yellow, a light that's off read as any colour, a sign that isn't there),
    or when it's wrong and at least SEEN_MIN_PCT of the frames saw a colour
    that isn't there: red read as green votes "go", as seeing nothing does.
    Anything else is MISSED (the vote fell back to "go" / no sign).

    Pass (first guesses: the course's sign and light positions are TBD in
    course.md): every trial at a gap up to `need_cm` reads OK; no WRONG
    read at any gap. A miss farther out only limits the range; a wrong read
    anywhere is a hazard (a red read as green runs the light). The summary
    gives the range: the largest gap up to which every trial read OK.

Main package:
    DetectRange: the routine (make routine-detect-range).
    read_look(seen, target, state): one look's numbers and its result.

Flow (per trial):
    trial i is gap gaps[i // len(states)], state states[i % len(states)]:
    the robot moved at each gap's first state; the tester sets the state;
    Enter; one look; the row and attempt_NN_<gap>cm_<state>.jpg.
"""
from src.estimation.estimation import CAUTION, GO, STOP
from src.params import GREEN, RED, YELLOW
from src.routines.camera_look import Eyes, parse_numbers, save_frame
from src.routines.harness import Routine, criterion

LIGHT, SIGN = "light", "sign"
OFF, PRESENT, ABSENT = "off", "present", "absent"
STATES = {LIGHT: (RED, YELLOW, GREEN, OFF), SIGN: (PRESENT, ABSENT)}
EXPECTED = {RED: STOP, YELLOW: CAUTION, GREEN: GO, OFF: GO, PRESENT: True, ABSENT: False}
NOTHING = {LIGHT: GO, SIGN: False}      # what the vote falls back to when it sees nothing
GAPS_CM = "0,10,20,30,45"   # bumper to the stop line's near edge; 0 is on the line
NEED_CM = 20.0              # every state read right from here in: navigation must know before the line
FRAMES = 40                 # about 2 s at 20 FPS: many times the vote window
SEEN_MIN_PCT = 50.0         # a lit light / posted sign must be seen in this share of frames
OK, MISSED, WRONG = "ok", "missed", "wrong"


def read_look(frames: list[dict], target: str, state: str) -> dict:
    """
    One look against the scene.

    seen_pct: share of frames that saw what's there (for off / absent:
        share that saw anything, which is all wrong).
    wrong_frames: frames that saw something that isn't there.
    voted: the vote after the last frame (None without frames).
    result: OK, MISSED or WRONG (module docstring).
    """
    n = len(frames)
    if target == LIGHT:
        here = state if state != OFF else None
        right = sum(1 for f in frames if here is not None and f["light"] == here)
        wrong = sum(1 for f in frames if f["light"] is not None and f["light"] != here)
        voted = frames[-1]["drive_state"] if frames else None
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
            "expected": expected, "result": result}


def reliable_range(results: list[tuple[float, str]]) -> float | None:
    """The largest gap up to which every trial (gap, result) read OK; None if the nearest didn't."""
    best = None
    for gap in sorted({g for g, _ in results}):
        if any(r != OK for g, r in results if g == gap):     # sorted: the nearer gaps passed already
            break
        best = gap
    return best


class DetectRange(Routine):
    name = "detect-range"
    title = "Detect range: the traffic light or stop sign, parked at gaps"
    question = "From how far before the stop line does the robot read the traffic light (or stop sign) right?"
    requirement = "D3 (light colour), D2 (stop sign)"
    trials = len(parse_numbers(GAPS_CM)) * len(STATES[LIGHT])
    fields = ("gap_cm", "state", "frames", "seen_pct", "wrong_frames", "voted", "expected", "result")
    settings = {"target": "light or sign (default light)",
                "gaps": f"gaps before the stop line in cm, comma separated (default {GAPS_CM})",
                "need_cm": f"every state must read right at gaps up to this (default {NEED_CM:g})",
                "frames": f"frames per look (default {FRAMES})"}
    instructions = """\
Set up: an intersection as on the course, its light (or stop sign) posted
where it will be on the day, the room lit as it will be. Mark the gaps on the
lane: bumper to the stop line's near edge. Motors stay off; nothing drives.
Each gap: place the robot on the mark, centered and pointing along the lane.
Then, for each state the prompt asks (light red, yellow, green, off; or the
sign there and taken away), set it and press Enter: the robot looks for ~2 s.
Keep people and bright clothes out of the view while it looks."""

    def __init__(self, eyes=None):
        self._eyes, self._need = eyes, NEED_CM

    @staticmethod
    def _defaults(options: dict) -> None:
        options.setdefault("target", LIGHT)
        options.setdefault("gaps", GAPS_CM)
        options.setdefault("need_cm", NEED_CM)
        options.setdefault("frames", FRAMES)
        if options["target"] not in STATES:
            raise ValueError(f"target must be light or sign, not {options['target']!r}")

    def plan(self, options):
        o = dict(options)
        self._defaults(o)
        return len(parse_numbers(o["gaps"])) * len(STATES[o["target"]])

    def setup(self, ctx):
        self._defaults(ctx.options)
        self._need = float(ctx.options["need_cm"])
        self._eyes = (self._eyes or Eyes()).open()
        ctx.state["results"] = {}

    def teardown(self, ctx):
        if self._eyes is not None:
            self._eyes.close()
        done = list(ctx.state.get("results", {}).values())
        if done:
            rng = reliable_range([(g, r) for g, _, r in done])
            ctx.state["range_cm"] = rng
            ctx.console.say("\nrange: " + ("not even at the nearest gap" if rng is None else
                                           f"every state read right up to {rng:g} cm before the line"))

    def trial(self, ctx, i):
        target = ctx.options["target"]
        gaps, states = parse_numbers(ctx.options["gaps"]), STATES[target]
        gap, state = gaps[i // len(states)], states[i % len(states)]
        if i % len(states) == 0:
            ctx.console.wait(f"Robot's bumper {gap:g} cm before the stop line, centered? Enter")
        what = {RED: "light RED", YELLOW: "light YELLOW", GREEN: "light GREEN", OFF: "light OFF (or covered)",
                PRESENT: "the stop sign IN PLACE", ABSENT: "the stop sign TAKEN AWAY"}[state]
        ctx.console.wait(f"Set {what}, step out of view; Enter looks")
        frames, last = self._eyes.look(int(ctx.options["frames"]))
        row = {"gap_cm": gap, "state": state, **read_look(frames, target, state)}
        save_frame(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}_{gap:g}cm_{state}.jpg", last)
        ctx.state["results"][str(i)] = (gap, state, row["result"])
        if row["result"] != OK:
            ctx.console.say(f"  {row['result'].upper()}: voted {row['voted']}, expected {row['expected']}")
        return row

    def judge(self, rows):
        need = self._need
        return [criterion(f"trials not read right within {need:g} cm",
                          sum(1 for r in rows if r["gap_cm"] <= need and r["result"] != OK), "<=", 0),
                criterion("wrong reads at any gap (something that isn't there)",
                          sum(1 for r in rows if r["result"] == WRONG), "<=", 0)]
