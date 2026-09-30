"""
test_lane_keeping.py  --  src/lane_keeping.py

LaneKeepingNavigator against the Navigation contract checks
(navigation_checks.py), then its own behavior: stop triggers, which offset it
steers by, the steering clamp and the stall-duty floor.
"""
import pytest

from src.lane_keeping import (
    BASE_SPEED, KP_CM, KP_HEADING, KP_NORM, MAX_STEERING_ADJ, STOP_LINE_THRESHOLD_CM,
    LaneKeepingNavigator, check_stop_line_trigger,
)
from src.navigation import BRAKE, STALL_DUTY, Command, Navigator, command_problems
from src.tests.navigation_checks import (
    check_commands, check_no_forward_on_stale, check_no_forward_on_stop,
    check_steers_toward_center, frames, packet,
)


def nav(**kw):
    return LaneKeepingNavigator(**kw)


# =============================================================================
# The contract
# =============================================================================

@pytest.mark.software
def test_it_is_a_navigator():
    assert isinstance(nav(), Navigator)


@pytest.mark.software
def test_every_command_is_valid():
    mixed = frames([{"lane_status": s, "lane_offset": o, "drive_state": d, "heading_error": h}
                    for s in ("vision", "hold", "stale") for o in (-1.0, -0.2, 0.0, 0.2, 1.0)
                    for d in ("go", "caution", "stop") for h in (-90.0, 0.0, 90.0)])
    assert check_commands(nav(), mixed) == []


@pytest.mark.software
def test_no_forward_drive_on_stop():
    assert check_no_forward_on_stop(nav()) == []


@pytest.mark.software
def test_steers_toward_center_both_ways():
    assert check_steers_toward_center(nav()) == []


@pytest.mark.software
@pytest.mark.xfail(strict=True, reason="open decision: it keeps driving on heading when the lane "
                                       "is stale; the contract says no forward drive on stale")
def test_no_forward_drive_on_a_stale_lane():
    assert check_no_forward_on_stale(nav()) == []


# =============================================================================
# Stop triggers
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("fields", [{"drive_state": "stop"}, {"stop_sign_detected": True},
                                    {"stop_line_detected": True, "stop_line_distance_cm": 1.0}])
def test_each_stop_trigger_brakes(fields):
    assert nav().update(packet(**fields)) == BRAKE


@pytest.mark.software
@pytest.mark.parametrize("detected, cm, fires", [
    (True, STOP_LINE_THRESHOLD_CM, True),           # at the threshold counts as reached
    (True, STOP_LINE_THRESHOLD_CM + 0.01, False),
    (True, 0.0, True),
    (True, None, False),                            # no ground homography: never fires
    (False, 1.0, False),                            # not voted: the distance alone doesn't count
])
def test_the_stop_line_fires_at_or_under_the_threshold(detected, cm, fires):
    p = packet(stop_line_detected=detected, stop_line_distance_cm=cm)
    assert check_stop_line_trigger(p) == fires
    assert (nav().update(p) == BRAKE) == fires


@pytest.mark.software
def test_the_threshold_is_configurable():
    p = packet(stop_line_detected=True, stop_line_distance_cm=5.0)
    assert nav().update(p) != BRAKE and nav(stop_line_threshold_cm=6.0).update(p) == BRAKE


@pytest.mark.software
def test_caution_still_drives():
    assert nav().update(packet(drive_state="caution")) == Command(BASE_SPEED, BASE_SPEED)


# =============================================================================
# Steering
# =============================================================================

@pytest.mark.software
def test_centered_drives_straight_at_the_base_speed():
    assert nav().update(packet()) == Command(BASE_SPEED, BASE_SPEED)


@pytest.mark.software
def test_on_vision_it_steers_by_cm_when_there_is_a_scale():
    cmd = nav().update(packet(lane_offset=0.9, lane_offset_cm=2.0))
    assert cmd.left == pytest.approx(BASE_SPEED - 2.0 * KP_CM)
    assert cmd.right == pytest.approx(BASE_SPEED + 2.0 * KP_CM)


@pytest.mark.software
def test_on_vision_without_a_scale_it_steers_by_the_normalized_offset():
    cmd = nav().update(packet(lane_offset=-0.2))
    assert cmd.left == pytest.approx(BASE_SPEED + 0.2 * KP_NORM)
    assert cmd.right == pytest.approx(BASE_SPEED - 0.2 * KP_NORM)


@pytest.mark.software
@pytest.mark.parametrize("status", ["hold", "stale"])
def test_without_vision_it_steers_against_the_heading_turned(status):
    # heading + = turned right, so it steers left: left slower
    cmd = nav().update(packet(lane_status=status, lane_offset=0.9, heading_error=5.0))
    assert cmd.left == pytest.approx(BASE_SPEED - 5.0 * KP_HEADING)
    assert cmd.right == pytest.approx(BASE_SPEED + 5.0 * KP_HEADING)


@pytest.mark.software
def test_an_unknown_lane_status_drives_straight():
    assert nav().update(packet(lane_status="unknown", lane_offset=1.0)) == Command(BASE_SPEED, BASE_SPEED)


@pytest.mark.software
def test_full_steering_is_clamped_and_stops_the_inner_wheel():
    cmd = nav().update(packet(lane_offset=1.0, lane_offset_cm=100.0))
    assert cmd.right == pytest.approx(BASE_SPEED + MAX_STEERING_ADJ)
    assert cmd.left == 0.0                          # 0.40 - 0.40: pivots on the inner wheel
    assert command_problems(cmd) == []


@pytest.mark.software
def test_a_slow_inner_wheel_is_lifted_to_the_stall_duty():
    cmd = nav(max_steering_adj=0.30).update(packet(lane_offset=1.0, lane_offset_cm=100.0))
    assert cmd.left == STALL_DUTY                   # 0.40 - 0.30 = 0.10 would stall
    assert command_problems(cmd) == []


@pytest.mark.software
def test_a_duty_that_would_reverse_is_lifted_to_the_reverse_stall_duty():
    cmd = nav(base_speed=0.25, max_steering_adj=0.40).update(packet(lane_offset=1.0, lane_offset_cm=100.0))
    assert cmd.left == -STALL_DUTY and cmd.right == pytest.approx(0.65)


@pytest.mark.software
@pytest.mark.parametrize("duty, out", [(0.0, 0.0), (5e-5, 0.0), (0.1, STALL_DUTY), (-0.1, -STALL_DUTY),
                                       (1.5, 1.0), (-1.5, -1.0), (0.5, 0.5)])
def test_sanitize_duty(duty, out):
    assert nav()._sanitize_duty(duty) == out


@pytest.mark.software
def test_base_speed_is_clamped_to_the_stall_duty_and_one():
    assert nav(base_speed=0.1).base_speed == STALL_DUTY and nav(base_speed=2.0).base_speed == 1.0


@pytest.mark.software
def test_reset_changes_nothing():
    n = nav()
    before = n.update(packet(lane_offset=0.3))
    n.reset()
    assert n.update(packet(lane_offset=0.3)) == before


@pytest.mark.software
def test_a_command_the_contract_rejects_becomes_a_brake(monkeypatch):
    # The last safety net: _sanitize_duty keeps every duty valid today, so break it on purpose
    n = nav()
    monkeypatch.setattr(n, "_sanitize_duty", lambda duty: 0.1)
    assert n.update(packet()) == BRAKE
