"""
test_stop_sign.py  --  src/navigation/stop_sign.py

The stop-sign rule against a real tracker: no sign, no stop; a sign seen
within SIGN_MEMORY_MS before the line is reached brakes until the wheels
stop, holds STOP_SIGN_HOLD_TIME_MS, then lets go; jittery or missing
encoders still start the hold; the sign is used up by one line; reset.
"""
import pytest

from src.navigation.navigation_contract import BRAKE
from src.navigation.stop_line import StopLineTracker
from src.navigation.stop_sign import (
    REASON_HOLD, REASON_STOPPING, SIGN_MEMORY_MS, STOP_SETTLE_MAX_MS, STOP_SIGN_HOLD_TIME_MS, STOPPED_CPS,
    StopSignRule,
)
from src.tests.navigation_checks import packet, reach_frames

MS = 50
LINE = [30.0, 15.0, 5.0]            # a stop line coming down the view, then gone
REACHED = len(LINE) + reach_frames(frame_ms=MS)


def drive(seq, rule=None):
    """seq: per-frame packet fields. Returns (rule, [(command, record)])."""
    tracker = StopLineTracker()
    rule = rule or StopSignRule(tracker)
    rule.tracker = tracker
    out = []
    for i, fields in enumerate(seq):
        p = packet(frame_id=i, timestamp_ms=i * MS, **fields)
        tracker.update(p)
        out.append((rule.update(p), dict(rule.record)))
    return rule, out


def line(sign=True, cps=0.0, after=120, sign_frames=None):
    """Packets: the line (with a sign on sign_frames, default all of them), then after frames, wheels at cps."""
    sign_frames = range(len(LINE)) if sign_frames is None else sign_frames
    seq = [{"stop_line_detected": True, "stop_line_distance_px": r, "stop_sign_detected": sign and i in sign_frames}
           for i, r in enumerate(LINE)]
    return seq + [{"left_wheel_cps": cps, "right_wheel_cps": cps}] * after


def braked(out):
    return [i for i, (cmd, _) in enumerate(out) if cmd == BRAKE]


@pytest.mark.software
def test_the_constants():
    assert (STOP_SIGN_HOLD_TIME_MS, STOPPED_CPS, STOP_SETTLE_MAX_MS, SIGN_MEMORY_MS) == (2000, 20.0, 1000, 5000)


@pytest.mark.software
def test_a_line_without_a_sign_says_nothing():
    _, out = drive(line(sign=False))
    assert all(cmd is None and rec == {} for cmd, rec in out)


@pytest.mark.software
def test_a_sign_with_no_line_says_nothing():
    _, out = drive([{"stop_sign_detected": True}] * 60)
    assert not braked(out)


@pytest.mark.software
def test_stopped_wheels_hold_from_the_line_for_the_hold_time_then_release():
    _, out = drive(line())
    assert braked(out) == list(range(REACHED, REACHED + STOP_SIGN_HOLD_TIME_MS // MS))
    assert out[REACHED][1] == {"reason": REASON_HOLD}
    assert out[braked(out)[-1] + 1] == (None, {})


@pytest.mark.software
def test_moving_wheels_brake_until_stopped_then_hold():
    seq = line(cps=900.0)
    for i in range(REACHED, REACHED + 4):         # still rolling for 4 frames after reaching it
        seq[i] = {"left_wheel_cps": 900.0, "right_wheel_cps": 15.0}
    for i in range(REACHED + 4, len(seq)):
        seq[i] = {"left_wheel_cps": 5.0, "right_wheel_cps": -5.0}
    _, out = drive(seq)
    assert [rec["reason"] for _, rec in out[REACHED:REACHED + 4]] == [REASON_STOPPING] * 4
    assert braked(out) == list(range(REACHED, REACHED + 4 + STOP_SIGN_HOLD_TIME_MS // MS))
    assert out[REACHED + 4][1] == {"reason": REASON_HOLD}


@pytest.mark.software
def test_wheels_that_never_read_stopped_still_hold_after_the_settle_time():
    _, out = drive(line(cps=STOPPED_CPS + 1.0))
    settle = STOP_SETTLE_MAX_MS // MS
    assert braked(out) == list(range(REACHED, REACHED + settle + STOP_SIGN_HOLD_TIME_MS // MS))


@pytest.mark.software
def test_a_sign_seen_before_the_line_counts_within_the_memory():
    early = [{"stop_sign_detected": True}] + [{}] * 20     # the sign, then 1 s before the line shows
    _, out = drive(early + line(sign=False))
    assert braked(out)


@pytest.mark.software
def test_a_sign_seen_too_long_before_does_not():
    gap = SIGN_MEMORY_MS // MS
    _, out = drive([{"stop_sign_detected": True}] + [{}] * gap + line(sign=False))
    assert not braked(out)


@pytest.mark.software
def test_one_sign_is_used_up_by_one_line():
    first = line(after=reach_frames(frame_ms=MS) + STOP_SIGN_HOLD_TIME_MS // MS + 5)
    long_memory = StopSignRule(StopLineTracker(), sign_memory_ms=60_000)       # the sign hasn't expired
    _, out = drive(first + line(sign=False), rule=long_memory)
    stops = braked(out)
    assert stops and stops[-1] < len(first)               # the second line passes without stopping


@pytest.mark.software
def test_reset_forgets_the_sign_and_any_stop():
    rule, out = drive(line(after=reach_frames(frame_ms=MS) + 3))
    assert braked(out)
    rule.reset()
    assert (rule._sign_ms, rule._braking_ms, rule._hold_ms, rule.record) == (None, None, None, {})


@pytest.mark.software
@pytest.mark.parametrize("left, right", [(5.0, 900.0), (900.0, 5.0)])
def test_both_wheels_must_read_stopped(left, right):
    seq = line()
    for i in range(REACHED, REACHED + 4):
        seq[i] = {"left_wheel_cps": left, "right_wheel_cps": right}
    _, out = drive(seq)
    assert [rec["reason"] for _, rec in out[REACHED:REACHED + 4]] == [REASON_STOPPING] * 4
