"""
test_intersection.py  --  src/navigation/intersection.py

The intersection rule against a real tracker and lane keeper: it drives
straight from the moment the line leaves the view, steers against the
heading turned (net of the gyro bias), ends after the line is reached on
both boundaries for TWO_BOUNDARY_FRAMES, at least one for
ONE_BOUNDARY_FRAMES, or MAX_CROSS_MS of driving; held frames count toward
none of them; active; reset. The route's turns: left and right from the
line on their duties until the gyro reaches TURN_TARGET_DEG their way (or
their time limit), then out on the new heading; held frames don't advance
a turn, and lane lines seen mid-turn don't end it.
"""
import pytest

from src.navigation.intersection import (
    LEFT_TURN, LEFT_TURN_MAX_MS, MAX_CROSS_MS, MAX_DT_MS, ONE_BOUNDARY_FRAMES, REASON_CROSSING, REASON_TURNING,
    RIGHT_TURN, RIGHT_TURN_MAX_MS, SOURCE_HEADING_HOLD, SOURCE_TURN, STAGE_EXIT, STAGE_TO_LINE, STAGE_TURN,
    TURN_TARGET_DEG, TWO_BOUNDARY_FRAMES, IntersectionRule,
)
from src.navigation.lane_keeping import BASE_SPEED, KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import Command, command_problems
from src.navigation.route import LEFT, RIGHT, STRAIGHT, Route, RouteProgress
from src.navigation.stop_line import STOP_DELAY_MS, StopLineTracker
from src.tests.navigation_checks import packet

MS = 50
LINE = [30.0, 15.0, 5.0]
LOST = len(LINE)
REACHED = LOST + STOP_DELAY_MS // MS


def drive(after, held=(), bias=0.0, line=LINE, ms=MS, maneuver=None):
    """
    The line, then one dict of fields per frame after it. held: frame indexes
    a higher rule holds. maneuver: a one-step route taking it at this line;
    None, no route (straight).
    """
    tracker = StopLineTracker()
    progress = RouteProgress(Route((maneuver,)) if maneuver else None)
    rule = IntersectionRule(tracker, LaneKeepingNavigator(), gyro_bias_dps=bias, progress=progress)
    seq = [{"stop_line_detected": True, "stop_line_distance_px": r, "lane_mode": "right_only"} for r in line] + after
    out = []
    for i, fields in enumerate(seq):
        p = packet(frame_id=i, timestamp_ms=i * ms, **fields)
        tracker.update(p)
        if tracker.entered:                         # as navigation.Navigation does
            progress.enter()
        out.append((rule.update(p, held=i in held), dict(rule.record)))
    return rule, out


def active(out):
    return [i for i, (cmd, _) in enumerate(out) if cmd is not None]


@pytest.mark.software
def test_the_constants():
    assert (TWO_BOUNDARY_FRAMES, ONE_BOUNDARY_FRAMES, MAX_CROSS_MS, MAX_DT_MS) == (3, 6, 3000, 500)


@pytest.mark.software
def test_nothing_to_say_without_a_line():
    _, out = drive([{}] * 40, line=[])
    assert not active(out)


@pytest.mark.software
def test_it_drives_straight_from_the_line_leaving_the_view():
    _, out = drive([{"lane_mode": "right_only", "lane_offset": -0.8}] * 10)
    assert active(out)[0] == LOST and out[LOST][0] == Command(BASE_SPEED, BASE_SPEED)   # ignores the offset
    assert out[LOST][1] == {"reason": REASON_CROSSING, "source": SOURCE_HEADING_HOLD, "steer": 0.0,
                            "heading_deg": 0.0, "stage": STAGE_TO_LINE, "step": "1/0 extra", "maneuver": STRAIGHT,
                            "turn_end": None}


@pytest.mark.software
def test_it_steers_against_the_heading_turned():
    _, out = drive([{"lane_mode": "right_only", "yaw_rate": 10.0}] * 11)      # turning right 10 deg/s
    heading = 10.0 * 10 * MS / 1000.0                                         # 10 frames after LOST
    cmd, rec = out[LOST + 10]
    assert rec["heading_deg"] == pytest.approx(heading)
    assert cmd.left == pytest.approx(BASE_SPEED - KP_HEADING * heading)       # steers left, back
    assert cmd.right == pytest.approx(BASE_SPEED + KP_HEADING * heading)


@pytest.mark.software
def test_the_gyro_bias_is_taken_off():
    _, out = drive([{"lane_mode": "right_only", "yaw_rate": 1.1}] * 20, bias=1.1)
    assert all(rec["heading_deg"] == 0.0 for _, rec in out[LOST:])


@pytest.mark.software
def test_a_long_packet_gap_integrates_as_at_most_max_dt():
    _, out = drive([{"lane_mode": "right_only", "yaw_rate": 10.0}] * 2, ms=2000)
    assert out[LOST + 1][1]["heading_deg"] == pytest.approx(10.0 * MAX_DT_MS / 1000.0)


@pytest.mark.software
def test_both_boundaries_before_the_line_is_reached_do_not_end_it():
    _, out = drive([{"lane_mode": "two_boundary"}] * (REACHED - LOST + 10))
    assert active(out)[:REACHED - LOST + 1] == list(range(LOST, REACHED + 1))


@pytest.mark.software
def test_both_boundaries_for_three_frames_after_the_line_end_it():
    after = [{"lane_mode": "none"}] * (REACHED - LOST + 5)
    after += [{"lane_mode": "two_boundary"}, {"lane_mode": "none"}]                 # one stray frame
    after += [{"lane_mode": "two_boundary"}] * TWO_BOUNDARY_FRAMES + [{"lane_mode": "two_boundary"}] * 3
    _, out = drive(after)
    end = LOST + (REACHED - LOST + 5) + 2 + TWO_BOUNDARY_FRAMES - 1
    assert active(out)[-1] == end - 1 and out[end] == (None, {})


@pytest.mark.software
@pytest.mark.parametrize("mode", ["right_only", "left_only"])
def test_one_boundary_for_one_boundary_frames_after_the_line_ends_it(mode):
    after = [{"lane_mode": "none"}] * (REACHED - LOST + 5)
    after += [{"lane_mode": mode}] * (ONE_BOUNDARY_FRAMES - 1) + [{"lane_mode": "none"}]   # one short of it
    after += [{"lane_mode": mode}] * (ONE_BOUNDARY_FRAMES + 3)
    _, out = drive(after)
    end = LOST + (REACHED - LOST + 5) + ONE_BOUNDARY_FRAMES + ONE_BOUNDARY_FRAMES - 1
    assert active(out)[-1] == end - 1 and out[end] == (None, {})


@pytest.mark.software
def test_one_and_two_boundary_frames_count_together_toward_one_boundary_frames():
    mixed = [{"lane_mode": m} for m in ("right_only", "two_boundary", "left_only")] * 2
    after = [{"lane_mode": "none"}] * (REACHED - LOST + 5) + mixed + [{"lane_mode": "none"}] * 5
    _, out = drive(after)
    end = LOST + (REACHED - LOST + 5) + ONE_BOUNDARY_FRAMES - 1
    assert active(out)[-1] == end - 1 and out[end] == (None, {})


@pytest.mark.software
def test_one_boundary_before_the_line_is_reached_does_not_end_it():
    _, out = drive([{"lane_mode": "right_only"}] * (REACHED - LOST + 10))
    assert active(out)[:REACHED - LOST + 1] == list(range(LOST, REACHED + 1))


@pytest.mark.software
def test_active_is_true_from_the_line_leaving_until_the_crossing_ends():
    tracker = StopLineTracker()
    rule = IntersectionRule(tracker, LaneKeepingNavigator())
    seen = []
    seq = ([{"stop_line_detected": True, "stop_line_distance_px": r, "lane_mode": "none"} for r in LINE]
           + [{"lane_mode": "none"}] * (REACHED - LOST) + [{"lane_mode": "two_boundary"}] * 5)
    for i, fields in enumerate(seq):
        p = packet(frame_id=i, timestamp_ms=i * MS, **fields)
        tracker.update(p)
        rule.update(p)
        seen.append(rule.active)
    assert seen[:LOST] == [False] * LOST and all(seen[LOST:REACHED + 1]) and not seen[-1]
    rule.reset()
    assert not rule.active


@pytest.mark.software
def test_it_gives_up_after_max_cross_ms_of_driving():
    after = [{"lane_mode": "none"}] * (REACHED - LOST + MAX_CROSS_MS // MS + 10)
    _, out = drive(after)
    end = REACHED + MAX_CROSS_MS // MS - 1          # the frame MAX_CROSS_MS of driving adds up on
    assert active(out)[-1] == end - 1 and out[end] == (None, {})


@pytest.mark.software
def test_held_frames_count_toward_neither_ending():
    hold = set(range(REACHED, REACHED + 60))                         # 3 s held at the line
    after = [{"lane_mode": "two_boundary"}] * (REACHED - LOST + 60) + [{"lane_mode": "none"}] * 100
    _, out = drive(after, held=hold)
    assert active(out)[-1] == REACHED + 60 + MAX_CROSS_MS // MS - 2      # the timeout, not the boundaries


@pytest.mark.software
def test_reset_stops_crossing():
    rule, out = drive([{"lane_mode": "right_only"}] * 5)
    assert active(out)
    rule.reset()
    rule.tracker.reset()                            # the line itself forgotten too
    assert rule.update(packet(timestamp_ms=10_000)) is None and rule.record == {}


@pytest.mark.software
@pytest.mark.parametrize("maneuver", [STRAIGHT, LEFT, RIGHT])
def test_the_record_names_the_route_step_and_maneuver(maneuver):
    _, out = drive([{"lane_mode": "none"}] * 3, maneuver=maneuver)
    assert (out[LOST][1]["step"], out[LOST][1]["maneuver"]) == (f"1/1 {maneuver}", maneuver)


# =============================================================================
# Turns
# =============================================================================

# Turning at 60 deg/s: TURN_TARGET_DEG takes ~1.4 s
YAW_DPS = 60.0
TURN_FRAMES = int(TURN_TARGET_DEG / YAW_DPS * 1000) // MS + 1


def turn_case(yaw, turn_frames=TURN_FRAMES, after=None):
    """Straight to the line, turning at yaw for turn_frames, then after (lane back by default)."""
    return ([{"lane_mode": "none"}] * (REACHED - LOST) + [{"lane_mode": "none", "yaw_rate": yaw}] * turn_frames
            + (after if after is not None else [{"lane_mode": "two_boundary"}] * 5))


def stages(out):
    return [rec.get("stage") for _, rec in out]


@pytest.mark.software
def test_the_turn_duties_are_ignacios_and_keep_the_contract():
    assert (LEFT_TURN, RIGHT_TURN) == (Command(0.36, 0.63), Command(0.45, 0.0))
    assert command_problems(LEFT_TURN) == command_problems(RIGHT_TURN) == []
    assert (TURN_TARGET_DEG, LEFT_TURN_MAX_MS, RIGHT_TURN_MAX_MS) == (85.0, 4100, 2400)


@pytest.mark.software
@pytest.mark.parametrize("maneuver, yaw, duties", [(LEFT, -YAW_DPS, LEFT_TURN), (RIGHT, YAW_DPS, RIGHT_TURN)])
def test_a_turn_starts_at_the_line_and_ends_on_the_gyro_target(maneuver, yaw, duties):
    _, out = drive(turn_case(yaw), maneuver=maneuver)
    st = stages(out)
    assert set(st[LOST:REACHED]) == {STAGE_TO_LINE}                          # straight to the line
    assert out[REACHED - 1][0] == Command(BASE_SPEED, BASE_SPEED)
    turning = [i for i, s in enumerate(st) if s == STAGE_TURN]
    assert turning[0] == REACHED and all(out[i][0] == duties for i in turning)
    assert out[turning[0]][1]["reason"] == REASON_TURNING and out[turning[0]][1]["source"] == SOURCE_TURN
    per_frame = YAW_DPS * MS / 1000                  # the frame reaching the target is already the exit's
    assert len(turning) * per_frame < TURN_TARGET_DEG <= (len(turning) + 1) * per_frame
    exit_ = [i for i, s in enumerate(st) if s == STAGE_EXIT]
    assert exit_ and exit_[0] == turning[-1] + 1 and out[-1] == (None, {})   # out on the lane


@pytest.mark.software
@pytest.mark.parametrize("maneuver, max_ms", [(LEFT, LEFT_TURN_MAX_MS), (RIGHT, RIGHT_TURN_MAX_MS)])
def test_a_turn_the_gyro_never_sees_ends_at_its_time_limit(maneuver, max_ms):
    _, out = drive(turn_case(0.0, turn_frames=max_ms // MS + 10), maneuver=maneuver)
    turning = [i for i, s in enumerate(stages(out)) if s == STAGE_TURN]
    assert turning[0] == REACHED and len(turning) == max_ms // MS - 1            # the frame reaching it exits


@pytest.mark.software
def test_turning_the_wrong_way_does_not_end_a_turn():
    _, out = drive(turn_case(+YAW_DPS, turn_frames=40), maneuver=LEFT)          # yaw to the right on a left turn
    assert STAGE_TURN in stages(out)[REACHED + 30:REACHED + 40]


@pytest.mark.software
def test_lane_lines_seen_mid_turn_do_not_end_it():
    case = ([{"lane_mode": "none"}] * (REACHED - LOST)
            + [{"lane_mode": "two_boundary", "yaw_rate": -YAW_DPS}] * TURN_FRAMES + [{"lane_mode": "none"}] * 5)
    _, out = drive(case, maneuver=LEFT)
    turning = [i for i, s in enumerate(stages(out)) if s == STAGE_TURN]
    assert len(turning) >= TURN_FRAMES - 1


@pytest.mark.software
def test_a_hold_at_the_line_delays_the_turn_without_using_its_time():
    hold = set(range(REACHED, REACHED + 40))                                     # 2 s at a stop sign
    case = turn_case(0.0, turn_frames=40 + LEFT_TURN_MAX_MS // MS + 5)
    _, out = drive(case, held=hold, maneuver=LEFT)
    turning = [i for i, s in enumerate(stages(out)) if s == STAGE_TURN]
    assert turning[0] == REACHED and len(turning) == 40 + LEFT_TURN_MAX_MS // MS - 1     # the held 2 s on top


@pytest.mark.software
def test_after_a_turn_the_exit_holds_the_heading_the_turn_ended_on():
    after = [{"lane_mode": "none", "yaw_rate": -10.0}] * 5 + [{"lane_mode": "two_boundary"}] * 3
    _, out = drive(turn_case(-YAW_DPS, after=after), maneuver=LEFT)
    exit_ = [(cmd, rec) for cmd, rec in out if rec.get("stage") == STAGE_EXIT]
    assert exit_[0][1]["heading_deg"] == 0.0 and exit_[0][1]["source"] == SOURCE_HEADING_HOLD
    assert exit_[-1][1]["heading_deg"] < 0 and exit_[-1][0].left > exit_[-1][0].right     # drifting left: steers right


@pytest.mark.software
def test_straight_and_no_route_never_turn():
    for maneuver in (STRAIGHT, None):
        _, out = drive(turn_case(-YAW_DPS), maneuver=maneuver)
        assert STAGE_TURN not in stages(out) and STAGE_EXIT in stages(out)
