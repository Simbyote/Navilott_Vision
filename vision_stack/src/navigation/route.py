"""Route: the course plan, one maneuver per intersection, and how the run finishes.

Purpose:
    The robot runs a programmed course: at the first intersection turn
    left, at the second go straight, and so on. Every intersection has a
    stop line on the robot's approach, so each stop line passing under the
    view is one intersection (StopLineTracker.entered). The plan lives in a
    JSON file (config.ROUTE_PATH) so it can be changed between runs without
    touching code; it's read and checked once, at startup, before the start
    button, so a typo stops the run there and not halfway round the course:

        {"maneuvers": ["left", "straight", "right"], "finish": "edge"}

    How the run finishes:
        FINISH_EDGE: after the last maneuver, the lane running out (the
            mat's edge) is the finish. A lane lost before then is off
            course: the run ends early (end_of_course.py).
        FINISH_STOP_LINE: the first stop line after the last maneuver is the
            finish line: the robot stops at it.
    Intersections past the plan, with FINISH_EDGE, are crossed straight.

Main package:
    Route: the plan; load_route() reads and checks the file; RouteError.
    RouteProgress: where the run is in the plan, advanced once per intersection.
    MANEUVERS: STRAIGHT, LEFT, RIGHT. TURNS_TBD: maneuvers not built yet
        (Ignacio's turn logic), driven straight until they are.

Flow:
    1. load_route(path) at startup; describe() for the startup screen.
    2. RouteProgress.enter() on each intersection: the step it is.
    3. done / at_finish_line tell end_of_course how the run ends.
"""
import json
from dataclasses import dataclass
from pathlib import Path

STRAIGHT, LEFT, RIGHT = "straight", "left", "right"
MANEUVERS = (STRAIGHT, LEFT, RIGHT)
# Left and right turns are Ignacio's to build; until then they're driven straight and logged
TURNS_TBD = (LEFT, RIGHT)
FINISH_EDGE, FINISH_STOP_LINE = "edge", "stop_line"
FINISHES = (FINISH_EDGE, FINISH_STOP_LINE)
# RouteProgress.enter()'s kinds of intersection
KIND_MANEUVER, KIND_FINISH, KIND_EXTRA = "maneuver", "finish", "extra"


class RouteError(ValueError):
    """The route file is missing, malformed or names something unknown; the message says which."""


@dataclass(frozen=True)
class Route:
    """
    The course plan.

    maneuvers: One of MANEUVERS per intersection, in order.
    finish: FINISH_EDGE or FINISH_STOP_LINE.
    """
    maneuvers: tuple[str, ...] = ()
    finish: str = FINISH_EDGE

    def __post_init__(self):
        bad = [m for m in self.maneuvers if m not in MANEUVERS]
        if bad:
            raise RouteError(f"unknown maneuver(s) {bad}; each must be one of {list(MANEUVERS)}")
        if self.finish not in FINISHES:
            raise RouteError(f"unknown finish {self.finish!r}; must be one of {list(FINISHES)}")

    def describe(self) -> list[str]:
        """The startup screen's lines: the step count, each maneuver, and the finish."""
        n = len(self.maneuvers)
        lines = [f"Route: {n} maneuver{'s' if n != 1 else ''}"]
        lines += [f"  {i}. {m}{'  (TBD: driven straight)' if m in TURNS_TBD else ''}"
                  for i, m in enumerate(self.maneuvers, 1)]
        lines.append("  finish: " + ("the mat's edge after the last maneuver (the lane runs out)"
                                     if self.finish == FINISH_EDGE else
                                     f"stop at stop line {n + 1}, after the last maneuver"))
        return lines


def load_route(path) -> Route:
    """
    Read and check a route file.

    Inputs:
        path: The JSON file: {"maneuvers": [...], "finish": "edge" | "stop_line"}.
            Maneuver and finish names are case-insensitive; "finish" may be
            left out (FINISH_EDGE).
    Outputs:
        The Route.
    Raises:
        RouteError: Missing file, bad JSON, wrong shape or an unknown name,
            with a message saying which.
    """
    path = Path(path)
    if not path.is_file():
        raise RouteError(f"route file {path} not found")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RouteError(f"route file {path.name} isn't valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RouteError(f"route file {path.name} must hold an object with \"maneuvers\" and \"finish\"")
    unknown = sorted(set(data) - {"maneuvers", "finish"})
    if unknown:
        raise RouteError(f"route file {path.name}: unknown key(s) {unknown}; use \"maneuvers\" and \"finish\"")
    maneuvers = data.get("maneuvers")
    if not isinstance(maneuvers, list) or not all(isinstance(m, str) for m in maneuvers):
        raise RouteError(f"route file {path.name}: \"maneuvers\" must be a list of names, e.g. [\"left\", \"straight\"]")
    finish = data.get("finish", FINISH_EDGE)
    if not isinstance(finish, str):
        raise RouteError(f"route file {path.name}: \"finish\" must be \"edge\" or \"stop_line\"")
    return Route(tuple(m.strip().lower() for m in maneuvers), finish.strip().lower())


class RouteProgress:
    """
    Where a run is in its route.

    Attributes:
        step: Intersections entered so far.
        current: (step, kind, maneuver) for the latest intersection: kind
            KIND_MANEUVER (maneuver from the plan), KIND_FINISH (the finish
            line; maneuver None) or KIND_EXTRA (past the plan; straight).
            None before the first.
    """
    def __init__(self, route: Route | None = None) -> None:
        self.route = route or Route()
        self.reset()

    def reset(self) -> None:
        """Back to the start of the route."""
        self.step = 0
        self.current: tuple[int, str, str | None] | None = None

    @property
    def done(self) -> bool:
        """Every maneuver in the plan has been started."""
        return self.step >= len(self.route.maneuvers)

    @property
    def at_finish_line(self) -> bool:
        """The latest intersection is the finish line."""
        return self.current is not None and self.current[1] == KIND_FINISH

    def enter(self) -> tuple[int, str, str | None]:
        """
        One more intersection: what it is in the plan.

        Outputs:
            (step, kind, maneuver), also kept in current.
        """
        n = len(self.route.maneuvers)
        if self.step < n:
            self.current = (self.step + 1, KIND_MANEUVER, self.route.maneuvers[self.step])
        elif self.route.finish == FINISH_STOP_LINE and self.step == n:     # the first line past the plan
            self.current = (self.step + 1, KIND_FINISH, None)
        else:
            self.current = (self.step + 1, KIND_EXTRA, STRAIGHT)
        self.step += 1
        return self.current

    def label(self) -> str:
        """The latest intersection for logs and the video: "2/3 right", "4/3 finish", "5/3 extra"."""
        if self.current is None:
            return f"0/{len(self.route.maneuvers)}"
        step, kind, maneuver = self.current
        what = maneuver if kind == KIND_MANEUVER else kind
        return f"{step}/{len(self.route.maneuvers)} {what}"
