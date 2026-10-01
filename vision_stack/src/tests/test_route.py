"""
test_route.py  --  src/navigation/route.py

The route file: a good one loads (names case-insensitive, finish optional),
every bad one is refused with a message saying what's wrong; the startup
description; RouteProgress through the plan: each maneuver in order, the
finish line after the plan, straight past it, done, labels, reset.
"""
import json

import pytest

from src.navigation.route import (
    FINISH_EDGE, FINISH_STOP_LINE, KIND_EXTRA, KIND_FINISH, KIND_MANEUVER, LEFT, MANEUVERS, RIGHT, STRAIGHT,
    TURNS_TBD, Route, RouteError, RouteProgress, load_route,
)


def write(tmp_path, data, raw=None):
    path = tmp_path / "route.json"
    path.write_text(raw if raw is not None else json.dumps(data))
    return path


# =============================================================================
# The file
# =============================================================================

@pytest.mark.software
def test_the_names():
    assert MANEUVERS == ("straight", "left", "right") and TURNS_TBD == ("left", "right")
    assert (FINISH_EDGE, FINISH_STOP_LINE) == ("edge", "stop_line")


@pytest.mark.software
@pytest.mark.software
def test_a_route_file_loads_with_its_maneuvers_in_order_and_its_finish(tmp_path):
    r = load_route(write(tmp_path, {"maneuvers": ["left", "straight", "right"], "finish": "stop_line"}))
    assert r == Route((LEFT, STRAIGHT, RIGHT), FINISH_STOP_LINE)


@pytest.mark.software
def test_names_are_case_insensitive_and_trimmed(tmp_path):
    r = load_route(write(tmp_path, {"maneuvers": [" Left", "STRAIGHT "], "finish": " Edge "}))
    assert r == Route((LEFT, STRAIGHT), FINISH_EDGE)


@pytest.mark.software
def test_finish_defaults_to_the_edge_and_an_empty_plan_is_allowed(tmp_path):
    assert load_route(write(tmp_path, {"maneuvers": []})) == Route((), FINISH_EDGE)


@pytest.mark.software
@pytest.mark.parametrize("data, raw, words", [
    (None, "{not json", "isn't valid JSON"),
    (["left"], None, "must hold an object"),
    ({"maneuvers": "left"}, None, "list of names"),
    ({"maneuvers": ["left", 2]}, None, "list of names"),
    ({}, None, "list of names"),
    ({"maneuvers": ["lfet"]}, None, "unknown maneuver"),
    ({"maneuvers": [], "finish": "wall"}, None, "unknown finish"),
    ({"maneuvers": [], "finish": 3}, None, "\"finish\" must be"),
    ({"maneuvers": [], "speed": 2}, None, "unknown key"),
])
def test_every_bad_file_is_refused_saying_why(tmp_path, data, raw, words):
    with pytest.raises(RouteError, match=words):
        load_route(write(tmp_path, data, raw))


@pytest.mark.software
def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(RouteError, match="not found"):
        load_route(tmp_path / "nope.json")


@pytest.mark.software
def test_a_route_built_in_code_is_checked_too():
    with pytest.raises(RouteError, match="unknown maneuver"):
        Route(("u-turn",))
    with pytest.raises(RouteError, match="unknown finish"):
        Route((), "cliff")


@pytest.mark.software
def test_the_startup_description_lists_every_step_and_the_finish():
    lines = Route((LEFT, STRAIGHT), FINISH_STOP_LINE).describe()
    assert lines[0] == "Route: 2 maneuvers"
    assert lines[1].startswith("  1. left") and "TBD" in lines[1]
    assert lines[2] == "  2. straight"
    assert "stop line 3" in lines[3]
    one = Route((RIGHT,)).describe()
    assert one[0] == "Route: 1 maneuver" and "mat's edge" in one[-1]


# =============================================================================
# Progress
# =============================================================================

@pytest.mark.software
def test_each_intersection_takes_the_next_maneuver_then_straight_past_the_plan():
    p = RouteProgress(Route((LEFT, RIGHT)))
    assert (p.step, p.current, p.done, p.label()) == (0, None, False, "0/2")
    assert p.enter() == (1, KIND_MANEUVER, LEFT) and not p.done and p.label() == "1/2 left"
    assert p.enter() == (2, KIND_MANEUVER, RIGHT) and p.done
    assert p.enter() == (3, KIND_EXTRA, STRAIGHT) and p.label() == "3/2 extra"
    assert not p.at_finish_line


@pytest.mark.software
def test_with_a_finish_line_the_first_line_past_the_plan_is_it():
    p = RouteProgress(Route((STRAIGHT,), FINISH_STOP_LINE))
    p.enter()
    assert p.enter() == (2, KIND_FINISH, None) and p.at_finish_line and p.label() == "2/1 finish"
    assert p.enter()[1] == KIND_EXTRA and not p.at_finish_line


@pytest.mark.software
def test_no_plan_is_done_from_the_start():
    p = RouteProgress()
    assert p.done and p.enter() == (1, KIND_EXTRA, STRAIGHT)


@pytest.mark.software
def test_reset_starts_the_route_over():
    p = RouteProgress(Route((LEFT,)))
    p.enter()
    p.reset()
    assert (p.step, p.current, p.done) == (0, None, False)
