"""
test_stop_line.py  --  src/navigation/stop_line.py

The stop-line tracker on hand-built packets: a line coming down the view and
passing under it, reached STOP_DELAY_MS later (at least a frame: CROSSING
lasts one) and only once; a line lost far away is flicker, not reached; a
re-sighting while crossing doesn't restart the delay; one after the line is
reached is a new line unless accept_new is off, as Navigation keeps it while
crossing; reset.
"""
import pytest

from src.navigation.stop_line import APPROACH, CROSSING, IDLE, NEAR_BOTTOM_ROWS, STOP_DELAY_MS, StopLineTracker
from src.tests.navigation_checks import packet, reach_frames

MS = 50


def feed(tracker, seq):
    """Feed (rows or None) per frame, MS apart; returns [(phase, reached)] after each."""
    out = []
    for i, rows in enumerate(seq):
        tracker.update(packet(frame_id=i, timestamp_ms=i * MS, stop_line_detected=rows is not None,
                              stop_line_distance_px=rows))
        out.append((tracker.phase, tracker.reached))
    return out


@pytest.mark.software
def test_the_constants_follow_the_calibration_and_the_runs():
    # 500 since 2026-10-12 (0 from 2026-10-07): it carries the robot into the
    # intersection, where the turn now starts without an advance
    assert STOP_DELAY_MS == 0 and NEAR_BOTTOM_ROWS == 25.0         # 0: the hanging light stays in view



@pytest.mark.software
def test_a_line_passing_under_the_view_is_reached_after_the_delay_once():
    t = StopLineTracker()
    out = feed(t, [None, 60.0, 30.0, 5.0] + [None] * 40)
    assert out[0] == (IDLE, False) and out[1][0] == APPROACH and out[3][0] == APPROACH
    assert out[4] == (CROSSING, False) and t.lost_ms is None
    reached = [i for i, (_, r) in enumerate(out) if r]
    assert reached == [4 + reach_frames(frame_ms=MS)]
    assert out[reached[0]][0] == IDLE and out[reached[0] + 1] == (IDLE, False)


@pytest.mark.software
def test_crossing_remembers_when_the_line_left():
    t = StopLineTracker()
    feed(t, [20.0, None])
    assert (t.phase, t.lost_ms, t.last_rows) == (CROSSING, MS, 20.0)


@pytest.mark.software
@pytest.mark.parametrize("last_rows, crosses", [(NEAR_BOTTOM_ROWS, True), (NEAR_BOTTOM_ROWS + 0.5, False),
                                                (0.0, True), (None, False)])
def test_only_a_line_last_seen_near_the_bottom_counts_as_passed(last_rows, crosses):
    t = StopLineTracker()
    out = feed(t, [60.0, last_rows, None])
    assert (out[-1][0] == CROSSING) == crosses
    if not crosses:
        assert out[-1][0] == IDLE and t.last_rows is None


@pytest.mark.software
def test_a_line_lost_far_away_is_never_reached():
    out = feed(StopLineTracker(), [60.0, 50.0] + [None] * 60)
    assert not any(r for _, r in out)


@pytest.mark.software
def test_a_flicker_while_crossing_does_not_restart_the_delay():
    out = feed(StopLineTracker(), [5.0, None, 3.0, None] + [None] * 40)
    assert [i for i, (_, r) in enumerate(out) if r] == [1 + reach_frames(frame_ms=MS)]


@pytest.mark.software
def test_the_delay_and_near_bottom_are_configurable():
    out = feed(StopLineTracker(delay_ms=200, near_bottom_rows=50.0), [40.0, None, None, None, None, None])
    assert [i for i, (_, r) in enumerate(out) if r] == [1 + reach_frames(200, MS)]


@pytest.mark.software
def test_the_next_line_is_tracked_after_one_is_reached():
    seq = [5.0, None] + [None] * 30 + [40.0, 10.0, None] + [None] * 30
    out = feed(StopLineTracker(), seq)
    assert len([r for _, r in out if r]) == 2


@pytest.mark.software
def test_reset_forgets_the_line():
    t = StopLineTracker()
    feed(t, [5.0, None])
    t.reset()
    assert (t.phase, t.reached, t.last_rows, t.lost_ms) == (IDLE, False, None, None)


@pytest.mark.software
def test_entered_marks_the_frame_the_line_leaves_the_view_once():
    seq = [60.0, 30.0, 5.0, None, None, None, 3.0, None]          # glimpsed again at 6, after it was reached

    def entered(accept_new_after_entering):
        t, out, crossing = StopLineTracker(), [], False
        for i, rows in enumerate(seq):
            t.update(packet(frame_id=i, timestamp_ms=i * MS, stop_line_detected=rows is not None,
                            stop_line_distance_px=rows), accept_new=not crossing or accept_new_after_entering)
            crossing = crossing or t.entered
            out.append(t.entered)
        return out
    # As Navigation runs it: no new line while the intersection is being crossed
    assert entered(False) == [False, False, False, True, False, False, False, False]
    # The tracker alone takes the glimpse for a new line once this one is reached: with no
    # delay (STOP_DELAY_MS = 0) nothing else stands between them, so Navigation's guard matters
    assert entered(True) == [False, False, False, True, False, False, False, STOP_DELAY_MS < 3 * MS]


@pytest.mark.software
def test_a_line_lost_far_away_never_enters():
    t, entered = StopLineTracker(), []
    for i, rows in enumerate([60.0, 50.0, None, None]):
        t.update(packet(frame_id=i, timestamp_ms=i * MS, stop_line_detected=rows is not None,
                        stop_line_distance_px=rows))
        entered.append(t.entered)
    assert not any(entered)
    t.reset()
    assert not t.entered


@pytest.mark.software
def test_with_accept_new_off_a_new_line_is_ignored_but_one_already_tracked_carries_on():
    t = StopLineTracker()
    t.update(packet(stop_line_detected=True, stop_line_distance_px=40.0, timestamp_ms=0), accept_new=False)
    assert t.phase == IDLE                                       # not taken up
    t.update(packet(stop_line_detected=True, stop_line_distance_px=40.0, timestamp_ms=50))
    assert t.phase == APPROACH
    t.update(packet(stop_line_detected=True, stop_line_distance_px=10.0, timestamp_ms=100), accept_new=False)
    t.update(packet(stop_line_detected=False, timestamp_ms=150), accept_new=False)
    assert t.phase == CROSSING and t.entered                    # the tracked line still leaves the view
    t.update(packet(timestamp_ms=150 + STOP_DELAY_MS), accept_new=False)
    assert t.reached


@pytest.mark.software
def test_the_reached_frame_says_when_its_line_first_came_into_view():
    t = StopLineTracker(delay_ms=0)
    seen = []
    for i, fields in enumerate([{}, {}] + [{"stop_line_detected": True, "stop_line_distance_px": r}
                                           for r in (30.0, 15.0, 5.0)] + [{}] * 3):
        t.update(packet(frame_id=i, timestamp_ms=i * 50, **fields))
        seen.append((t.phase, t.reached, t.seen_ms))
    assert seen[2] == (APPROACH, False, 100)                 # first sight
    reached = next(s for s in seen if s[1])
    assert reached == (IDLE, True, 100)                      # the line just reached, still named
    t.reset()
    assert t.seen_ms is None


# =============================================================================
# By distance, with odometry
# =============================================================================

def roll(tracker, sightings, after, step=3.0):
    """
    The robot rolling step cm a frame: sightings are the line's cm ahead on
    each frame it's seen (rows near the bottom), then `after` frames gone.
    Returns each frame's (odometry, reached, left_cm).
    """
    out = []
    for i in range(len(sightings) + after):
        cm = sightings[i] if i < len(sightings) else None
        p = packet(frame_id=i, timestamp_ms=i * MS, stop_line_detected=cm is not None,
                   stop_line_distance_px=None if cm is None else 10.0, stop_line_distance_cm=cm,
                   distance_cm=i * step)
        tracker.update(p)
        out.append((i * step, tracker.reached, tracker.left_cm))
    return out


def reached_at(out):
    return next((odo, left) for odo, reached, left in out if reached)


@pytest.mark.software
def test_by_distance_it_stops_nearest_the_mark_by_the_encoders():
    # the line 20 cm ahead at odometry 0, seen down to 11 cm; 3 cm a frame
    out = roll(StopLineTracker(reach_cm=5.0), [20.0, 17.0, 14.0, 11.0], 6)
    odo, left = reached_at(out)
    assert left == 5.0 and odo == 15.0                     # exactly on the mark when a frame lands there


@pytest.mark.software
def test_a_frame_stops_when_the_next_would_be_further_from_the_mark():
    # 4 cm a frame, the mark 4 cm before the line: a frame stops if what's left
    # is nearer the mark than what the next frame would leave
    out = roll(StopLineTracker(reach_cm=4.0), [20.0, 16.0, 12.0], 5, step=4.0)
    assert reached_at(out)[1] == 4.0
    out = roll(StopLineTracker(reach_cm=4.0), [21.5, 17.5, 13.5], 5, step=4.0)
    assert reached_at(out)[1] == 5.5                        # 5.5 is 1.5 over; the next, 1.5, is 2.5 under


@pytest.mark.software
def test_where_the_line_left_the_view_no_longer_moves_the_stop():
    # A line last seen 11 cm or 13 cm ahead: the timed stop carries the 2 cm
    # into where the robot stops; by distance every stop is within half a
    # frame's travel (1.5 cm) of the mark
    sightings = ([20.0, 17.0, 14.0, 11.0], [22.0, 19.0, 16.0, 13.0])
    timed = [reached_at(roll(StopLineTracker(), s, 8))[1] for s in sightings]
    measured = [reached_at(roll(StopLineTracker(reach_cm=2.0), s, 8))[1] for s in sightings]
    assert timed[1] - timed[0] == 2.0
    assert all(abs(left - 2.0) <= 1.5 for left in measured)


@pytest.mark.software
def test_a_held_distance_isnt_a_fresh_sighting():
    # Phase 3 repeats 14.0 through a missed frame while the robot moves on: the line stays where 14.0 put it
    out = roll(StopLineTracker(reach_cm=5.0), [20.0, 17.0, 14.0, 14.0], 6)
    assert out[3][2] == 11.0                               # 20 ahead of 0, the robot at 9: not 14 ahead of 9


@pytest.mark.software
@pytest.mark.parametrize("line_cm, odometry", [(None, True), (12.0, False)])
def test_without_a_line_cm_or_odometry_it_is_timed(line_cm, odometry):
    t = StopLineTracker(reach_cm=5.0)
    frames = [line_cm, line_cm, None, None, None, None]
    for i, cm in enumerate(frames):
        t.update(packet(frame_id=i, timestamp_ms=i * MS, stop_line_detected=i < 2, stop_line_distance_px=10.0 if i < 2 else None,
                        stop_line_distance_cm=cm if i < 2 else None, distance_cm=i * 3.0 if odometry else None))
        if t.reached:
            break
    assert t.reached and i == 2 + reach_frames(frame_ms=MS) and t.left_cm is None


@pytest.mark.software
def test_left_cm_is_logged_by_the_timed_stop_too():
    out = roll(StopLineTracker(), [20.0, 17.0, 14.0, 11.0], 3)       # reach_cm None: STOP_DELAY_MS
    odo, left = reached_at(out)
    assert left == pytest.approx(20.0 - odo) and out[0][2] == 20.0
