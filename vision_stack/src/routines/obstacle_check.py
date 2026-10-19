"""Obstacle check: does the parked robot see an object in its lane, how far away does it say it is, and nothing when the lane is clear?

Purpose:
    Obstacle detection (perception/obstacle.py) was tuned on one day's
    runs with one can. This checks it with the robot parked and an object
    placed at taped distances in its lane: the share of frames that saw it,
    the vote at the end, the distance it reported against the tape, and
    whether the obstacle rule would brake for it there. Trials with the lane
    clear (the object taken away) count false alarms. Each trial is one look
    (camera_look.Eyes: Phases 1-3 with a fresh processor, `frames` frames,
    ~2 s); the look's last frame is saved beside the row.

    The distance is the floor cm from the robot's front edge (the
    stop-line table's reference) to where the object meets the floor. The
    camera's view starts ~10.2 cm ahead, so an object nearer than that has
    no base in view: it reads clipped ("at the view bottom or nearer"),
    which the rule always brakes for; its distance isn't judged.

    A present trial reads OK when the vote says obstacle, at least
    SEEN_MIN_PCT of the frames saw it, and its reported distance (the
    median over the frames that measured it) is within tolerance_cm of
    the tape. MISSED: the vote never said so. OFF: seen, but the distance is
    out of tolerance. A clear trial reads OK unless the vote said obstacle
    (WRONG: a false alarm; the robot would brake on an empty lane).

    laterals places the object off the lane's middle as well: every gap
    at every lateral offset (cm, + = right of the middle), so the trials map
    how wide the path the detector watches really is. Only objects within
    in_path_cm of the middle (the robot's own path; 0 = the middle only)
    are judged: one farther out is outside the path, and seeing it or not
    is the map, not a fault. The summary gives, per gap, the widest offset
    up to which every object, both sides, read OK.

    Pass (first guesses, to refine after the first runs): every object up
    to need_cm seen and its distance within tolerance; no false alarm on a
    clear lane; the brake decision right at every gap more than
    BRAKE_MARGIN_CM from the rule's stop distance. The summary gives the
    range: the largest gap up to which every present trial read OK.

Main package:
    ObstacleCheck: the routine (make routine-obstacle-check).
    read_look(frames, gap_cm, present, tolerance_cm): one look's numbers and its result.
    seen_width(results): per gap, the widest offset every object read OK out to.

Flow (per trial):
    present trials, nearest gap first, each gap at every lateral offset,
    then `clear` clear-lane trials: the tester places (or removes) the
    object; Enter; one look; the row and attempt_NN_<gap>cm[_<lateral>cm].jpg
    (attempt_NN_clear.jpg).
"""
import statistics

from src.navigation.obstacle import OBSTACLE_STOP_CM
from src.routines.camera_look import Eyes, parse_numbers, save_frame
from src.routines.detect_range import reliable_range
from src.routines.harness import Routine, criterion

GAPS_CM = "10,15,20,25,30"  # front edge to where the object meets the floor; 10 is under the view's edge
LATERALS_CM = "0"           # object's center off the lane's middle, + = right: "-10,-5,0,5,10" maps the path
IN_PATH_CM = 0.0            # judged only this close to the middle: farther out is outside the robot's path
CLEAR_TRIALS = 3            # trials with the lane clear: false alarms
NEED_CM = 25.0              # every object seen and measured from here in
TOLERANCE_CM = 3.0          # reported distance against the tape
FRAMES = 40                 # about 2 s at 20 FPS: many times the vote window
SEEN_MIN_PCT = 50.0         # a present object must be seen in this share of frames
# The brake decision isn't judged this near the rule's stop distance: a
# tape and a report a cm apart there disagree on which side it falls
BRAKE_MARGIN_CM = 2.0
OK, MISSED, OFF, WRONG = "ok", "missed", "off", "wrong"
PRESENT, CLEAR = "present", "clear"


def would_brake(frame: dict, stop_cm: float = OBSTACLE_STOP_CM) -> bool:
    """The obstacle rule's decision on a frame's vote: an obstacle voted, and clipped or within stop_cm."""
    if not frame["obstacle_voted"]:
        return False
    if frame["obstacle_voted_clipped"]:
        return True
    cm = frame["obstacle_voted_cm"]
    return cm is not None and cm <= stop_cm


def read_look(frames: list[dict], gap_cm: float | None, present: bool, tolerance_cm: float = TOLERANCE_CM) -> dict:
    """
    One look against the scene.

    seen_pct: share of frames whose Phase 2 saw an obstacle.
    voted: the vote after the last frame (None without frames).
    reported_cm: the median distance over the frames that measured one (not
        clipped); None without any. error_cm: reported_cm - gap_cm.
    clipped_pct: share of frames whose obstacle ran off the view bottom.
    would_brake: the obstacle rule's decision on the last frame.
    result: OK, MISSED, OFF (present) or OK, WRONG (clear); see the module docstring.
    """
    n = len(frames)
    seen = [f for f in frames if f["obstacle"]]
    measured = [f["obstacle_cm"] for f in seen if not f["obstacle_clipped"] and f["obstacle_cm"] is not None]
    reported = round(statistics.median(measured), 1) if measured else None
    error = None if reported is None or gap_cm is None else round(reported - gap_cm, 1)
    voted = frames[-1]["obstacle_voted"] if frames else None
    seen_pct = round(100.0 * len(seen) / n, 1) if n else 0.0
    clipped_pct = round(100.0 * sum(1 for f in seen if f["obstacle_clipped"]) / n, 1) if n else 0.0
    if not present:
        result = WRONG if voted else OK
    elif not voted or seen_pct < SEEN_MIN_PCT:
        result = MISSED
    elif error is not None and abs(error) > tolerance_cm:
        result = OFF
    else:
        result = OK                 # measured within tolerance, or clipped (nearer than the view: no distance)
    return {"frames": n, "seen_pct": seen_pct, "voted": voted, "reported_cm": reported, "error_cm": error,
            "clipped_pct": clipped_pct, "would_brake": would_brake(frames[-1]) if frames else None, "result": result}


def seen_width(results: list[tuple[float, float, str]]) -> dict[float, float | None]:
    """
    Per gap, from (gap, lateral, result) present trials: the widest |lateral|
    up to which every trial at that gap, both sides, read OK; None if the
    middle-most didn't.
    """
    out = {}
    for gap in sorted({g for g, _, _ in results}):
        best = None
        for w in sorted({abs(lat) for g, lat, _ in results if g == gap}):
            if any(r != OK for g, lat, r in results if g == gap and abs(lat) <= w):
                break
            best = w
        out[gap] = best
    return out


class ObstacleCheck(Routine):
    name = "obstacle-check"
    title = "Obstacle check: an object in the lane, parked at gaps"
    question = ("Does the robot see an object in its lane, how far away does it say it is, "
                "and does it see nothing when the lane is clear?")
    requirement = "obstacle detection (stretch goal: stop for an obstacle)"
    trials = len(parse_numbers(GAPS_CM)) + CLEAR_TRIALS
    fields = ("gap_cm", "lateral_cm", "state", "frames", "seen_pct", "voted", "reported_cm", "error_cm", "clipped_pct",
              "would_brake", "result")
    settings = {"gaps": f"gaps from the front edge to the object, cm, comma separated (default {GAPS_CM})",
                "clear": f"trials with the lane clear (default {CLEAR_TRIALS})",
                "need_cm": f"every object seen and measured at gaps up to this (default {NEED_CM:g})",
                "tolerance_cm": f"reported distance against the tape (default {TOLERANCE_CM:g})",
                "laterals": f"object off the lane's middle, cm, + = right, comma separated (default {LATERALS_CM})",
                "in_path_cm": f"objects judged only this close to the middle (default {IN_PATH_CM:g})",
                "frames": f"frames per look (default {FRAMES})",
                "object": "what the object is, recorded (e.g. can, box)"}
    instructions = f"""\
Set up: a straight lane, the room lit as it will be on the day. Park the robot
centered in the lane, pointing along it, and leave it there: the object moves,
not the robot. Motors stay off; nothing drives.
Each gap: place the object in the middle of the lane (or as far off it as the
prompt says, measured to the object's center) with its nearest point that far
from the robot's front edge (tape along the lane's center), and press Enter:
the robot looks for ~2 s. Then the clear trials: take the object away
(and anything else off the lane) and press Enter. Gaps under ~10 cm sit below
the camera's view: they read clipped, and the rule brakes for them at
{OBSTACLE_STOP_CM:g} cm or nearer. Keep people and bright clothes out of view."""

    def __init__(self, eyes=None):
        self._eyes = eyes
        self._need, self._tol, self._in_path = NEED_CM, TOLERANCE_CM, IN_PATH_CM

    @staticmethod
    def _defaults(options: dict) -> None:
        options.setdefault("gaps", GAPS_CM)
        options.setdefault("clear", CLEAR_TRIALS)
        options.setdefault("need_cm", NEED_CM)
        options.setdefault("tolerance_cm", TOLERANCE_CM)
        options.setdefault("laterals", LATERALS_CM)
        options.setdefault("in_path_cm", IN_PATH_CM)
        options.setdefault("frames", FRAMES)
        options.setdefault("object", "")

    @staticmethod
    def _placements(options: dict) -> list[tuple[float, float]]:
        return [(g, lat) for g in sorted(parse_numbers(options["gaps"])) for lat in parse_numbers(options["laterals"])]

    def plan(self, options):
        o = dict(options)
        self._defaults(o)
        return len(self._placements(o)) + int(float(o["clear"]))

    def setup(self, ctx):
        self._defaults(ctx.options)
        self._need, self._tol = float(ctx.options["need_cm"]), float(ctx.options["tolerance_cm"])
        self._in_path = float(ctx.options["in_path_cm"])
        self._eyes = (self._eyes or Eyes()).open()
        ctx.state["results"] = {}

    def teardown(self, ctx):
        if self._eyes is not None:
            self._eyes.close()
        done = [(g, lat, r) for g, lat, r in ctx.state.get("results", {}).values() if g is not None]
        judged = [(g, r) for g, lat, r in done if abs(lat) <= self._in_path]
        if judged:
            rng = reliable_range(judged)
            ctx.state["range_cm"] = rng
            ctx.console.say("\nrange: " + ("not even at the nearest gap" if rng is None else
                                           f"every object seen and measured up to {rng:g} cm"))
        if len({lat for _, lat, _ in done}) > 1:
            width = seen_width(done)
            ctx.state["seen_width_cm"] = {f"{g:g}": w for g, w in width.items()}
            for g, w in width.items():
                ctx.console.say(f"  at {g:g} cm: " + ("missed even nearest the middle" if w is None
                                                      else f"seen out to {w:g} cm off the middle, both sides"))

    def trial(self, ctx, i):
        placements = self._placements(ctx.options)
        present = i < len(placements)
        gap, lateral = placements[i] if present else (None, None)
        if present:
            where = ("in the middle of the lane" if not lateral else
                     f"its center {abs(lateral):g} cm {'right' if lateral > 0 else 'left'} of the lane's middle")
            ctx.console.wait(f"Object {gap:g} cm from the front edge, {where}? Enter looks")
        else:
            ctx.console.wait("Lane CLEAR: object taken away, nothing on the lane? Enter looks")
        frames, last = self._eyes.look(int(ctx.options["frames"]))
        row = {"gap_cm": gap, "lateral_cm": lateral, "state": PRESENT if present else CLEAR,
               **read_look(frames, gap, present, self._tol)}
        tag = (f"{gap:g}cm" + (f"_{lateral:g}cm" if lateral else "")) if present else "clear"
        save_frame(ctx.out_dir / f"attempt_{ctx.attempt + 1:02d}_{tag}.jpg", last)
        ctx.state["results"][str(i)] = (gap, lateral, row["result"])
        if row["result"] != OK:
            ctx.console.say(f"  {row['result'].upper()}: voted {row['voted']}, seen {row['seen_pct']:g}%"
                            + ("" if row["reported_cm"] is None else f", reported {row['reported_cm']:g} cm"))
        return row

    def judge(self, rows):
        need, tol = self._need, self._tol
        in_path = [r for r in rows if r["state"] == CLEAR or abs(r.get("lateral_cm") or 0.0) <= self._in_path]
        near = [r for r in in_path if r["state"] == PRESENT and r["gap_cm"] <= need]
        errors = [abs(r["error_cm"]) for r in near if r["error_cm"] is not None]
        judged = [r for r in in_path if r["state"] == CLEAR or abs(r["gap_cm"] - OBSTACLE_STOP_CM) > BRAKE_MARGIN_CM]
        wrong_brake = sum(1 for r in judged
                          if r["would_brake"] != (r["state"] == PRESENT and r["gap_cm"] <= OBSTACLE_STOP_CM))
        return [criterion(f"objects within {need:g} cm not seen", sum(1 for r in near if r["result"] == MISSED),
                          "<=", 0),
                criterion(f"worst distance error within {need:g} cm", max(errors) if errors else None,
                          "<=", tol, "cm"),
                criterion("false alarms on a clear lane", sum(1 for r in rows if r["result"] == WRONG), "<=", 0),
                criterion(f"brake decisions wrong (gaps over {BRAKE_MARGIN_CM:g} cm from {OBSTACLE_STOP_CM:g} cm)",
                          wrong_brake, "<=", 0)]
