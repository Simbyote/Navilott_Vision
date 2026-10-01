"""
test_intersection.py  --  src/navigation/intersection.py

The intersection rule against a real tracker and lane keeper: it drives
straight from the moment the line leaves the view, steers against the
heading turned (net of the gyro bias), ends on both boundaries for
TWO_BOUNDARY_FRAMES after the line is reached or after MAX_CROSS_MS of
driving; held frames count toward neither; reset.
"""
import pytest

from src.navigation.intersection import (
    MAX_CROSS_MS, MAX_DT_MS, REASON_CROSSING, SOURCE_HEADING_HOLD, TWO_BOUNDARY_FRAMES, IntersectionRule,
)
from src.navigation.lane_keeping import BASE_SPEED, KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import Command
from src.navigation.stop_line import STOP_DELAY_MS, StopLineTracker
from src.tests.navigation_checks import packet

MS = 50
LINE = [30.0, 15.0, 5.0]
LOST = len(LINE)
REACHED = LOST + STOP_DELAY_MS // MS


def drive(after, held=(), bias=0.0, line=LINE, ms=MS):
    """The line, then one dict of fields per frame after it. held: frame indexes a higher rule holds."""
    tracker = StopLineTracker()
    rule = IntersectionRule(tracker, LaneKeepingNavigator(), gyro_bias_dps=bias)
    seq = [{"stop_line_detected": True, "stop_line_distance_px": r, "lane_mode": "right_only"} for r in line] + after
    out = []
    for i, fields in enumerate(seq):
        p = packet(frame_id=i, timestamp_ms=i * ms, **fields)
        tracker.update(p)
        out.append((rule.update(p, held=i in held), dict(rule.record)))
    return rule, out


def active(out):
    return [i for i, (cmd, _) in enumerate(out) if cmd is not None]


@pytest.mark.software
def test_the_constants():
    assert (TWO_BOUNDARY_FRAMES, MAX_CROSS_MS, MAX_DT_MS) == (3, 4000, 500)


@pytest.mark.software
def test_nothing_to_say_without_a_line():
    _, out = drive([{}] * 40, line=[])
    assert not active(out)


@pytest.mark.software
def test_it_drives_straight_from_the_line_leaving_the_view():
    _, out = drive([{"lane_mode": "right_only", "lane_offset": -0.8}] * 10)
    assert active(out)[0] == LOST and out[LOST][0] == Command(BASE_SPEED, BASE_SPEED)   # ignores the offset
    assert out[LOST][1] == {"reason": REASON_CROSSING, "source": SOURCE_HEADING_HOLD, "steer": 0.0,
                            "heading_deg": 0.0}


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
    after = [{"lane_mode": "right_only"}] * (REACHED - LOST + 5)
    after += [{"lane_mode": "two_boundary"}, {"lane_mode": "right_only"}]           # one stray frame
    after += [{"lane_mode": "two_boundary"}] * TWO_BOUNDARY_FRAMES + [{"lane_mode": "two_boundary"}] * 3
    _, out = drive(after)
    end = LOST + (REACHED - LOST + 5) + 2 + TWO_BOUNDARY_FRAMES - 1
    assert active(out)[-1] == end - 1 and out[end] == (None, {})


@pytest.mark.software
def test_it_gives_up_after_max_cross_ms_of_driving():
    after = [{"lane_mode": "right_only"}] * (REACHED - LOST + MAX_CROSS_MS // MS + 10)
    _, out = drive(after)
    end = REACHED + MAX_CROSS_MS // MS - 1          # the frame MAX_CROSS_MS of driving adds up on
    assert active(out)[-1] == end - 1 and out[end] == (None, {})


@pytest.mark.software
def test_held_frames_count_toward_neither_ending():
    hold = set(range(REACHED, REACHED + 60))                         # 3 s held at the line
    after = [{"lane_mode": "two_boundary"}] * (REACHED - LOST + 60) + [{"lane_mode": "right_only"}] * 100
    _, out = drive(after, held=hold)
    assert active(out)[-1] == REACHED + 60 + MAX_CROSS_MS // MS - 2      # the timeout, not the boundaries


@pytest.mark.software
def test_reset_stops_crossing():
    rule, out = drive([{"lane_mode": "right_only"}] * 5)
    assert active(out)
    rule.reset()
    rule.tracker.reset()                            # the line itself forgotten too
    assert rule.update(packet(timestamp_ms=10_000)) is None and rule.record == {}
