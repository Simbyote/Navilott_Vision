"""
test_obstacle_rule.py  --  src/navigation/obstacle.py

The obstacle rule alone, then inside Navigation: what counts as within
reach (cm, then rows without cm, clipped always), braking until the
obstacle has been gone OBSTACLE_CLEAR_MS, not creeping toward one it
stopped short of, its priority over every other rule, and the lane it
hides not ending the run.
"""
import pytest

from src.navigation.navigation import RULE_END_OF_COURSE, RULE_LANE_KEEPING, RULE_OBSTACLE, Navigation
from src.navigation.navigation_contract import BRAKE
from src.navigation.obstacle import (
    OBSTACLE_CLEAR_MS, OBSTACLE_STOP_CM, OBSTACLE_STOP_PX, REASON_CLEARING, REASON_OBSTACLE, ObstacleRule,
    obstacle_near,
)
from src.navigation.end_of_course import END_STALE_MS
from src.tests.navigation_checks import check_brakes_for_an_obstacle_then_goes, forward, packet

MS = 50
CLEAR_FRAMES = OBSTACLE_CLEAR_MS // MS


def seen(cm=None, px=None, clipped=False, **fields):
    return {"obstacle_detected": True, "obstacle_distance_cm": cm, "obstacle_distance_px": px,
            "obstacle_clipped": clipped, **fields}


def drive(seq, rule=None):
    """seq: per-frame packet fields. Returns (rule, [(command, record)])."""
    rule = rule or ObstacleRule()
    out = []
    for i, fields in enumerate(seq):
        out.append((rule.update(packet(frame_id=i, timestamp_ms=i * MS, **fields)), dict(rule.record)))
    return rule, out


# =============================================================================
# Within reach
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("fields, near", [
    ({}, False),                                                    # nothing voted
    (seen(cm=OBSTACLE_STOP_CM, px=99.0), True),                     # cm decide, inclusive
    (seen(cm=OBSTACLE_STOP_CM + 0.01, px=0.0), False),              # cm decide over px
    (seen(px=OBSTACLE_STOP_PX), True),                              # no cm: rows decide, inclusive
    (seen(px=OBSTACLE_STOP_PX + 0.5), False),
    (seen(cm=40.0, px=90.0, clipped=True), True),                   # clipped is always within reach
    ({"obstacle_distance_cm": 1.0, "obstacle_clipped": True}, False),   # the vote says none: nothing is near
    (seen(), False),                                                # voted with no distance at all: not near
])
def test_what_counts_as_within_reach(fields, near):
    assert obstacle_near(packet(**fields)) is near


@pytest.mark.software
def test_the_distances_are_settable():
    p = packet(**seen(cm=20.0, px=50.0))
    assert not obstacle_near(p) and obstacle_near(p, stop_cm=20.0)
    assert obstacle_near(packet(**seen(px=50.0)), stop_px=50.0)


@pytest.mark.software
def test_the_constants():
    """The view's nearest floor is ~10.2 cm (the 2026-09-30 table's row 0): the stop sits just past it."""
    assert 10.2 < OBSTACLE_STOP_CM <= 15.0 and OBSTACLE_STOP_PX > 0 and OBSTACLE_CLEAR_MS >= 500


# =============================================================================
# Brake, wait, go
# =============================================================================

@pytest.mark.software
def test_far_obstacles_and_none_say_nothing():
    rule, out = drive([{}] * 3 + [seen(cm=30.0, px=80.0)] * 5)
    assert all(cmd is None and rec == {} for cmd, rec in out) and not rule.blocked


@pytest.mark.software
def test_it_brakes_from_the_frame_the_obstacle_is_within_reach():
    rule, out = drive([seen(cm=20.0)] * 2 + [seen(cm=11.0, px=15.0)] * 3)
    assert [cmd for cmd, _ in out] == [None, None, BRAKE, BRAKE, BRAKE] and rule.blocked
    assert out[2][1] == {"reason": REASON_OBSTACLE, "distance_cm": 11.0, "distance_px": 15.0}


@pytest.mark.software
def test_it_releases_after_the_obstacle_has_been_gone_the_clear_time():
    rule, out = drive([seen(clipped=True)] * 3 + [{}] * (CLEAR_FRAMES + 3))
    cmds = [cmd for cmd, _ in out]
    assert cmds[:3 + CLEAR_FRAMES] == [BRAKE] * (3 + CLEAR_FRAMES)
    assert cmds[3 + CLEAR_FRAMES:] == [None] * 3 and not rule.blocked
    assert out[4][1] == {"reason": REASON_CLEARING, "clear_ms": MS}


@pytest.mark.software
def test_one_that_stopped_it_holds_it_wherever_it_reads_until_it_is_gone():
    """Stopped just short, the obstacle reads a little past the stop distance: no creeping into it."""
    rule, out = drive([seen(cm=11.0)] + [seen(cm=13.5)] * 40)
    assert all(cmd == BRAKE for cmd, _ in out) and rule.blocked


@pytest.mark.software
def test_a_glimpse_during_the_clear_time_starts_it_over():
    half = CLEAR_FRAMES // 2
    rule, out = drive([seen(cm=10.5)] + [{}] * half + [seen(cm=30.0)] + [{}] * (CLEAR_FRAMES - 1) + [{}] * 2)
    cmds = [cmd for cmd, _ in out]
    assert cmds[:1 + half + 1 + CLEAR_FRAMES] == [BRAKE] * (1 + half + 1 + CLEAR_FRAMES)
    assert cmds[-1] is None


@pytest.mark.software
def test_reset_forgets_it():
    rule, _ = drive([seen(clipped=True)] * 3)
    rule.reset()
    assert not rule.blocked and rule.update(packet()) is None and rule.record == {}


# =============================================================================
# Inside Navigation
# =============================================================================

def run(nav, case):
    nav.reset()
    out = []
    for i, fields in enumerate(case):
        p = packet(frame_id=i, timestamp_ms=i * MS, **fields)
        out.append((nav.update(p), dict(nav.record)))
    return out


@pytest.mark.software
def test_it_is_the_first_rule_and_beats_a_stop_sign_at_the_line():
    nav = Navigation()
    assert nav.rules[0][0] == RULE_OBSTACLE
    near = seen(clipped=True, stop_sign_detected=True, drive_state="stop")
    out = run(nav, [near] * 5)
    assert all(cmd == BRAKE and rec["rule"] == RULE_OBSTACLE for cmd, rec in out)


@pytest.mark.software
def test_the_lane_it_hides_does_not_end_the_run():
    stale = seen(clipped=True, lane_status="stale", lane_mode="none")
    nav = Navigation()
    out = run(nav, [{}] * 5 + [stale] * (3 * END_STALE_MS // MS) + [{}] * (CLEAR_FRAMES + 5))
    assert not nav.finished
    assert {rec["rule"] for _, rec in out[5:5 + 3 * END_STALE_MS // MS]} == {RULE_OBSTACLE}
    assert out[-1][1]["rule"] == RULE_LANE_KEEPING and forward(out[-1][0])
    assert not any(rec["rule"] == RULE_END_OF_COURSE for _, rec in out)


@pytest.mark.software
def test_a_given_rule_is_the_one_navigation_asks():
    rule = ObstacleRule(stop_cm=30.0)
    nav = Navigation(obstacle=rule)
    assert nav.rules[0][1] is rule
    assert run(nav, [seen(cm=25.0)])[0][0] == BRAKE


@pytest.mark.software
def test_navigation_passes_the_obstacle_contract_check():
    assert check_brakes_for_an_obstacle_then_goes(Navigation()) == []


# =============================================================================
# In navigation_linker's record and video strip
# =============================================================================

@pytest.mark.software
def test_nav_csv_carries_the_voted_obstacle():
    import src.navigation_linker as nl
    assert "obstacle_cm" in nl.NAV_FIELDS
    assert nl.obstacle_cm(packet()) is None
    assert nl.obstacle_cm(packet(**seen(cm=14.5, px=40.0))) == 14.5
    assert nl.obstacle_cm(packet(**seen(cm=10.2, px=0.0, clipped=True))) == "view bottom"


@pytest.mark.software
@pytest.mark.parametrize("ob", [None, "", 14.5, "view bottom"])
def test_the_strip_draws_any_obstacle_record(ob):
    from src.debugger.debug_navigation import draw_strip
    img = draw_strip({"rule": RULE_OBSTACLE, "brake": 1, "reason": REASON_OBSTACLE, "obstacle_cm": ob}, 480)
    assert img.shape[1] == 480 and img.any()
