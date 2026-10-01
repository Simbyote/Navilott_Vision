"""
test_navigation.py  --  src/navigation/navigation.py, the navigation subsystem's orchestrator

The rules in priority order over lane keeping: lane keeping when no rule
speaks; the first rule that speaks wins, but every rule sees every frame and
is told when a higher one already decided; record names the deciding part;
the contract checks and the intersection scenarios end to end; reset; the
contract re-exported.
"""
import pytest

import src.navigation.navigation as navigation
import src.navigation.navigation_contract as contract
from src.navigation.navigation import (
    RULE_END_OF_COURSE, RULE_INTERSECTION, RULE_LANE_KEEPING, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, Navigation,
)
from src.navigation.navigation_contract import BRAKE, Command, Navigator
from src.navigation.stop_line import APPROACH, CROSSING, IDLE, STOP_DELAY_MS
from src.tests.navigation_checks import (
    APPROACH_ROWS, INTERSECTION_CHECKS, check_commands, check_stale_lane_slows_then_stops, check_steers_toward_center,
    contract_problems, frames, intersection, packet,
)

MS = 50


class Says:
    """A rule stub: a fixed answer, logging what it was told."""
    def __init__(self, answer):
        self.answer, self.calls, self.resets, self.record = answer, [], 0, {"reason": "stub"}

    def update(self, packet, held=False):
        self.calls.append(held)
        return self.answer

    def reset(self):
        self.resets += 1


def stubbed(*answers):
    nav = Navigation()
    nav.rules = [(f"r{i}", Says(a)) for i, a in enumerate(answers)]
    return nav


def run(nav, case):
    """(packet, command, record) per frame of a case."""
    nav.reset()
    out = []
    for p in frames(case):
        out.append((p, nav.update(p), dict(nav.record)))
    return out


# =============================================================================
# The orchestration
# =============================================================================

@pytest.mark.software
def test_the_rules_in_priority_order():
    assert [name for name, _ in Navigation().rules] == [RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, RULE_INTERSECTION,
                                                        RULE_END_OF_COURSE]


@pytest.mark.software
def test_with_no_rule_speaking_lane_keeping_steers():
    nav = stubbed(None, None)
    cmd = nav.update(packet(lane_offset=0.3))
    assert cmd == nav.lane.update(packet(lane_offset=0.3))
    assert nav.record["rule"] == RULE_LANE_KEEPING and nav.record["source"] == "offset"


@pytest.mark.software
def test_the_first_rule_that_speaks_wins_and_every_rule_still_sees_the_frame():
    nav = stubbed(None, BRAKE, Command(0.4, 0.4))
    assert nav.update(packet()) == BRAKE
    first, second, third = (rule for _, rule in nav.rules)
    assert (first.calls, second.calls, third.calls) == ([False], [False], [True])     # held after the winner
    assert nav.record == {"rule": "r1", "phase": IDLE, "reason": "stub"}


@pytest.mark.software
def test_the_tracker_advances_before_the_rules_are_asked():
    nav, seen = Navigation(), []

    class Peek(Says):
        def update(self, packet, held=False):
            seen.append(nav.tracker.phase)
            return None
    nav.rules = [("peek", Peek(None))]
    nav.update(packet(stop_line_detected=True, stop_line_distance_px=10.0))
    assert seen == [APPROACH]


@pytest.mark.software
def test_reset_resets_the_tracker_every_rule_and_lane_keeping(monkeypatch):
    nav = stubbed(None, None)
    nav.update(packet(stop_line_detected=True, stop_line_distance_px=10.0))
    lane_resets = []
    monkeypatch.setattr(nav.lane, "reset", lambda: lane_resets.append(1))
    nav.reset()
    assert nav.tracker.phase == IDLE and nav.record == {} and lane_resets == [1]
    assert all(rule.resets == 1 for _, rule in nav.rules)


@pytest.mark.software
def test_a_given_lane_keeper_and_tracker_are_shared_with_the_rules():
    from src.navigation.lane_keeping import LaneKeepingNavigator
    from src.navigation.stop_line import StopLineTracker
    lane, tracker = LaneKeepingNavigator(base_speed=0.5), StopLineTracker()
    nav = Navigation(lane=lane, tracker=tracker, gyro_bias_dps=1.1)
    rules = dict(nav.rules)
    assert all(rules[r].tracker is tracker for r in (RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT, RULE_INTERSECTION))
    assert rules[RULE_INTERSECTION].lane is lane and rules[RULE_INTERSECTION].gyro_bias_dps == 1.1
    assert rules[RULE_END_OF_COURSE].lane is lane


@pytest.mark.software
def test_it_is_a_navigator_and_re_exports_the_contract():
    assert isinstance(Navigation(), Navigator)
    for name in ("BRAKE", "STALL_DUTY", "Command", "Navigator", "command_problems"):
        assert getattr(navigation, name) is getattr(contract, name)


# =============================================================================
# The contract, end to end
# =============================================================================

@pytest.mark.software
def test_every_command_is_valid():
    mixed = frames(intersection({"stop_sign_detected": True}, {"drive_state": "stop"}, after_frames=40)
                   + [{"lane_status": s, "lane_offset": o} for s in ("vision", "hold", "stale")
                      for o in (-1.0, 0.0, 1.0)])
    assert check_commands(Navigation(), mixed) == []


@pytest.mark.software
def test_steers_toward_center():
    assert check_steers_toward_center(Navigation()) == []


@pytest.mark.software
@pytest.mark.parametrize("check", INTERSECTION_CHECKS, ids=lambda c: c.__name__)
def test_every_intersection_check_passes(check):
    assert check(Navigation()) == []


@pytest.mark.software
def test_a_stale_lane_slows_then_stops():
    assert check_stale_lane_slows_then_stops(Navigation()) == []


@pytest.mark.software
def test_the_whole_contract():
    assert contract_problems(Navigation()) == []


@pytest.mark.software
def test_a_lost_lane_finishes_the_run_and_every_command_after_is_brake():
    nav = Navigation()
    out = run(nav, [{}] * 5 + [{"lane_status": "stale"}] * 30 + [{}] * 10)
    first = next(i for i, (_, _, rec) in enumerate(out) if rec.get("reason") == "end_of_course")
    assert nav.finished and all(cmd == BRAKE for _, cmd, _ in out[first:])
    assert {rec["rule"] for _, _, rec in out[5:first]} == {RULE_END_OF_COURSE}
    assert all(rec["rule"] == RULE_END_OF_COURSE for _, _, rec in out[first:])
    nav.reset()
    assert not nav.finished


@pytest.mark.software
def test_a_stale_lane_while_crossing_an_intersection_does_not_end_the_run():
    # No lane boundaries in the middle of an intersection: stale while the crossing drives
    case = intersection(after={"lane_status": "stale", "lane_mode": "none"}, after_frames=60) + [{}] * 5
    out = run(Navigation(), case)
    assert not any(rec.get("reason") == "end_of_course" for _, _, rec in out)


# =============================================================================
# Who decides, frame by frame
# =============================================================================

def rules_over(out):
    """The deciding rule per frame, collapsed to its runs."""
    runs = []
    for _, _, rec in out:
        if not runs or runs[-1] != rec["rule"]:
            runs.append(rec["rule"])
    return runs


@pytest.mark.software
def test_a_stop_sign_and_a_red_light_at_one_line_stop_then_wait_for_green():
    n = len(APPROACH_ROWS)
    red_ms = STOP_DELAY_MS + 3000                        # still red after the 2 s stop-sign hold
    case = intersection({"stop_sign_detected": True, "drive_state": "stop"}, after_frames=0)
    case += [{"drive_state": "stop", "lane_mode": "right_only"}] * (red_ms // MS)
    case += [{"lane_mode": "right_only"}] * 10 + [{}] * 10
    out = run(Navigation(), case)
    assert rules_over(out) == [RULE_LANE_KEEPING, RULE_INTERSECTION, RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT,
                               RULE_INTERSECTION, RULE_LANE_KEEPING]
    assert out[n][2]["phase"] == CROSSING
    assert all(cmd == BRAKE for _, cmd, rec in out if rec["rule"] in (RULE_STOP_SIGN, RULE_TRAFFIC_LIGHT))


@pytest.mark.software
def test_a_green_line_is_crossed_straight_then_lane_keeping_takes_over():
    case = intersection(after={"lane_mode": "right_only", "lane_offset": -0.9}, after_frames=40) + [{}] * 5
    out = run(Navigation(), case)
    assert rules_over(out) == [RULE_LANE_KEEPING, RULE_INTERSECTION, RULE_LANE_KEEPING]
    crossing = [cmd for _, cmd, rec in out if rec["rule"] == RULE_INTERSECTION]
    assert all(cmd == Command(0.4, 0.4) for cmd in crossing)          # straight, not chasing the offset



@pytest.mark.software
def test_once_finished_no_rule_is_asked_again():
    # A stop line passing under the view after the finish would make the intersection rule speak
    nav = Navigation()
    run(nav, [{}] * 5 + [{"lane_status": "stale"}] * 30)
    assert nav.finished
    asked = []
    for _, rule in nav.rules:
        real = rule.update
        rule.update = lambda p, held=False, real=real: asked.append(1) or real(p, held)
    for p in frames(intersection(after_frames=20), start=100):
        assert nav.update(p) == BRAKE and nav.record["rule"] == RULE_END_OF_COURSE
    assert asked == [] and nav.tracker.phase == IDLE
