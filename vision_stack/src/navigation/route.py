"""Route: the course plan, one maneuver per intersection, and how the run finishes.

Purpose:
    Each stop line passing under the view is one intersection, and the plan
    says what to do at each. It is a JSON file (config.ROUTE_PATH) checked at
    startup, so a typo stops the run before the start, not halfway round.

Main package:
    Route: the maneuvers and the finish. RouteProgress tracks where a run is in it.
"""
import json
from dataclasses import dataclass
from pathlib import Path

STRAIGHT, LEFT, RIGHT = "straight", "left", "right"
MANEUVERS = (STRAIGHT, LEFT, RIGHT)
FINISH_EDGE, FINISH_STOP_LINE = "edge", "stop_line"
FINISHES = (FINISH_EDGE, FINISH_STOP_LINE)
KIND_MANEUVER, KIND_FINISH, KIND_EXTRA = "maneuver", "finish", "extra"


class RouteError(ValueError):
    """The message says what's wrong with the route file."""


@dataclass(frozen=True)
class Route:
    maneuvers: tuple[str, ...] = ()
    finish: str = FINISH_EDGE

    def __post_init__(self):
        bad = [m for m in self.maneuvers if m not in MANEUVERS]
        if bad:
            raise RouteError(f"unknown maneuver(s) {bad}; each must be one of {list(MANEUVERS)}")
        if self.finish not in FINISHES:
            raise RouteError(f"unknown finish {self.finish!r}; must be one of {list(FINISHES)}")

    def describe(self) -> list[str]:
        """The startup screen's lines."""
        n = len(self.maneuvers)
        lines = [f"Route: {n} maneuver{'s' if n != 1 else ''}"]
        lines += [f"  {i}. {m}" for i, m in enumerate(self.maneuvers, 1)]
        lines.append("  finish: " + ("the mat's edge after the last maneuver (the lane runs out)"
                                     if self.finish == FINISH_EDGE else
                                     f"stop at stop line {n + 1}, after the last maneuver"))
        return lines


def load_route(path) -> Route:
    """
    {"maneuvers": [...], "finish": "edge" | "stop_line"}, case-insensitive;
    finish defaults to edge. RouteError says what's wrong.
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
    step counts intersections entered. current is (step, kind, maneuver) for
    the latest: past the plan it's the finish line, then extra (straight).
    """
    def __init__(self, route: Route | None = None) -> None:
        self.route = route or Route()
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.current: tuple[int, str, str | None] | None = None

    @property
    def done(self) -> bool:
        return self.step >= len(self.route.maneuvers)

    @property
    def at_finish_line(self) -> bool:
        return self.current is not None and self.current[1] == KIND_FINISH

    def enter(self) -> tuple[int, str, str | None]:
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
        """For logs and the video: "2/3 right", "4/3 finish", "5/3 extra"."""
        if self.current is None:
            return f"0/{len(self.route.maneuvers)}"
        step, kind, maneuver = self.current
        what = maneuver if kind == KIND_MANEUVER else kind
        return f"{step}/{len(self.route.maneuvers)} {what}"
