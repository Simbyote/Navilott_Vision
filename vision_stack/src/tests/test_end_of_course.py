"""
test_end_of_course.py  --  src/navigation/end_of_course.py

The end-of-course rule: nothing to say while the lane is fine; on a stale
lane, creep at SLOW_DUTY steering by heading; still stale after
END_STALE_MS, brake and finish for good; the lane coming back resets the
count; held frames (a crossing, a stop) don't count; reset.
"""
import pytest

from src.navigation.end_of_course import (
    END_STALE_MS, MAX_DT_MS, REASON_FINISHED, REASON_SLOW, SLOW_DUTY, SLOW_MAX_STEERING_ADJ, EndOfCourseRule,
)
from src.navigation.lane_keeping import KP_HEADING, LaneKeepingNavigator
from src.navigation.navigation_contract import BRAKE, STALL_DUTY, Command
from src.tests.navigation_checks import packet

MS = 50
END = END_STALE_MS // MS


def drive(seq, held=(), ms=MS, rule=None):
    """seq: per-frame packet fields; held: frame indexes a higher rule holds. Returns (rule, [(cmd, record)])."""
    rule = rule or EndOfCourseRule(LaneKeepingNavigator())
    out = []
    for i, fields in enumerate(seq):
        cmd = rule.update(packet(frame_id=i, timestamp_ms=i * ms, **fields), held=i in held)
        out.append((cmd, dict(rule.record)))
    return rule, out


STALE = {"lane_status": "stale"}


@pytest.mark.software
def test_the_constants():
    assert (SLOW_DUTY, SLOW_MAX_STEERING_ADJ, END_STALE_MS, MAX_DT_MS) == (0.30, 0.30, 1000, 500)
    assert SLOW_DUTY > STALL_DUTY                       # creeping, not stalled


@pytest.mark.software
@pytest.mark.parametrize("status", ["vision", "hold"])
def test_nothing_to_say_while_the_lane_is_fine(status):
    _, out = drive([{"lane_status": status}] * 60)
    assert all(cmd is None and rec == {} for cmd, rec in out)


@pytest.mark.software
def test_a_stale_lane_creeps_then_finishes():
    rule, out = drive([{}] + [STALE] * (END + 5))
    assert out[1][0] == Command(SLOW_DUTY, SLOW_DUTY)
    assert out[1][1]["reason"] == REASON_SLOW and out[1][1]["stale_ms"] == MS
    finished = [i for i, (cmd, _) in enumerate(out) if cmd == BRAKE]
    assert finished[0] == END and rule.finished
    assert out[END][1] == {"reason": REASON_FINISHED}


@pytest.mark.software
def test_finished_is_for_good():
    rule, out = drive([{}] + [STALE] * END + [{}] * 20)
    assert rule.finished and all(cmd == BRAKE for cmd, _ in out[END:])


@pytest.mark.software
def test_creeping_steers_against_the_heading_turned():
    _, out = drive([{}, {"lane_status": "stale", "heading_error": 5.0}])
    cmd, rec = out[1]
    assert cmd == Command(SLOW_DUTY - 5.0 * KP_HEADING, SLOW_DUTY + 5.0 * KP_HEADING)
    assert rec["source"] == "heading" and rec["steer"] == pytest.approx(5.0 * KP_HEADING)


@pytest.mark.software
def test_creeping_steering_is_clamped_so_no_wheel_reverses():
    _, out = drive([{}, {"lane_status": "stale", "heading_error": 90.0}])
    cmd = out[1][0]
    assert cmd.left == 0.0 and cmd.right == pytest.approx(SLOW_DUTY + SLOW_MAX_STEERING_ADJ)


@pytest.mark.software
def test_the_lane_coming_back_resets_the_count():
    seq = [{}] + [STALE] * (END - 2) + [{"lane_status": "hold"}] + [STALE] * (END - 1)
    rule, out = drive(seq)
    assert not rule.finished and all(cmd != BRAKE for cmd, _ in out)


@pytest.mark.software
def test_held_frames_do_not_count():
    seq = [{}] + [STALE] * (END + 40)
    rule, out = drive(seq, held=set(range(5, 45)))
    assert [i for i, (cmd, _) in enumerate(out) if cmd == BRAKE][0] == END + 40


@pytest.mark.software
def test_a_long_gap_counts_as_at_most_max_dt():
    rule, out = drive([{}, STALE], ms=2000)
    assert out[1][1]["stale_ms"] == MAX_DT_MS and not rule.finished


@pytest.mark.software
def test_the_end_time_is_configurable():
    rule, out = drive([{}] + [STALE] * 5, rule=EndOfCourseRule(LaneKeepingNavigator(), end_stale_ms=3 * MS))
    assert rule.finished and out[3][0] == BRAKE


@pytest.mark.software
def test_reset_starts_over():
    rule, _ = drive([{}] + [STALE] * (END + 1))
    rule.reset()
    assert (rule.finished, rule.record) == (False, {})
    assert rule.update(packet(lane_status="stale", timestamp_ms=99_000)).brake is False
