"""
test_stop_line.py  --  src/stop_line.py

The stop-line tracker on hand-built packets: a line coming down the view and
passing under it, reached exactly STOP_DELAY_MS later and only once; a line
lost far away is flicker, not reached; a re-sighting while crossing doesn't
restart the delay; reset.
"""
import pytest

from src.stop_line import APPROACH, CROSSING, IDLE, NEAR_BOTTOM_ROWS, STOP_DELAY_MS, StopLineTracker
from src.tests.navigation_checks import packet

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
    assert STOP_DELAY_MS == 1500 and NEAR_BOTTOM_ROWS == 25.0


@pytest.mark.software
def test_a_line_passing_under_the_view_is_reached_after_the_delay_once():
    t = StopLineTracker()
    out = feed(t, [None, 60.0, 30.0, 5.0] + [None] * 40)
    assert out[0] == (IDLE, False) and out[1][0] == APPROACH and out[3][0] == APPROACH
    assert out[4] == (CROSSING, False) and t.lost_ms is None
    reached = [i for i, (_, r) in enumerate(out) if r]
    assert reached == [4 + STOP_DELAY_MS // MS]
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
    assert [i for i, (_, r) in enumerate(out) if r] == [1 + STOP_DELAY_MS // MS]


@pytest.mark.software
def test_the_delay_and_near_bottom_are_configurable():
    out = feed(StopLineTracker(delay_ms=200, near_bottom_rows=50.0), [40.0, None, None, None, None, None])
    assert [i for i, (_, r) in enumerate(out) if r] == [1 + 200 // MS]


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
