"""
test_wheel_stall.py  --  src/navigation/wheel_stall.py and its place in Navigation

The guard trips when a wheel driven at MIN_DUTY or more turns under
MOVING_SHARE of what its duty gives for STALL_MS: not before the encoders
read, not at a low duty, not on a start from rest or a slow wheel; a brake
clears the count; once tripped it stays tripped until reset(). Navigation
then brakes on every frame and ends the run with OUTCOME_STALLED; the
2026-10-28 dying turn trips it.

--software  Hand-built packets on a fake clock. No robot.
"""
from types import SimpleNamespace

import pytest

from src.navigation.navigation import RULE_WHEEL_STALL, Navigation
from src.navigation.navigation_contract import BRAKE, Command
from src.navigation.wheel_stall import (
    MIN_DUTY, MOVING_SHARE, OUTCOME_STALLED, REASON_STALLED, STALL_MS, WheelStallGuard,
)
from src.tests.navigation_checks import packet

MS = 50
GAIN = 3000.0
HEALTHY = SimpleNamespace(encoders=True, gains=(GAIN, GAIN))


def drive(guard, cmd, speeds, wheels=HEALTHY, start=0):
    """Each frame answers cmd with speeds(i) -> (left, right) counts/s; the frame it tripped on, else None."""
    for i in range(start, start + 60):
        left, right = speeds(i)
        if guard.observe(cmd, packet(frame_id=i, timestamp_ms=i * MS, left_wheel_cps=left, right_wheel_cps=right),
                         wheels):
            return i
    return None


pytestmark = pytest.mark.software


def test_the_constants():
    assert (MIN_DUTY, MOVING_SHARE, STALL_MS) == (0.30, 0.15, 500)


def test_a_dead_wheel_trips_it_after_stall_ms_and_names_the_wheel():
    g = WheelStallGuard()
    tripped = drive(g, Command(0.4, 0.8), lambda i: (0.4 * GAIN, 0.0))
    assert tripped == STALL_MS // MS                    # the first frame has no dt to count
    assert g.tripped and g.record == {"reason": REASON_STALLED, "wheel": "right", "duty": 0.8, "cps": 0.0,
                                      "expected_cps": 2400, "stalled_ms": STALL_MS}


def test_healthy_wheels_a_start_from_rest_and_a_slow_wheel_dont_trip_it():
    assert drive(WheelStallGuard(), Command(1.0, 1.0), lambda i: (GAIN, GAIN)) is None
    assert drive(WheelStallGuard(), Command(0.4, 0.4), lambda i: (0.0, 0.0) if i < 4 else (1200, 1200)) is None
    # a dying motor that still turns at 20% of its speed is the PI's to hold up, not a stall
    assert drive(WheelStallGuard(), Command(1.0, 1.0), lambda i: (GAIN, 0.2 * GAIN)) is None


def test_a_low_duty_or_no_encoders_never_trips():
    assert drive(WheelStallGuard(), Command(0.29, 0.29), lambda i: (0.0, 0.0)) is None
    unarmed = SimpleNamespace(encoders=False, gains=(GAIN, GAIN))
    assert drive(WheelStallGuard(), Command(1.0, 1.0), lambda i: (0.0, 0.0), unarmed) is None


def test_a_wheel_turning_the_wrong_way_counts_as_stalled():
    assert drive(WheelStallGuard(), Command(-0.6, 0.6), lambda i: (0.6 * GAIN, 0.6 * GAIN)) is not None


def test_a_brake_clears_the_count_and_a_trip_lasts_until_reset():
    g = WheelStallGuard()
    for i in range(STALL_MS // MS - 1):
        g.observe(Command(0.8, 0.8), packet(frame_id=i, timestamp_ms=i * MS), HEALTHY)
    g.observe(BRAKE, packet(frame_id=20, timestamp_ms=20 * MS), HEALTHY)
    assert not g.tripped and drive(g, Command(0.8, 0.8), lambda i: (0.0, 0.0), start=21) == 20 + STALL_MS // MS   # the brake frame set the clock
    assert g.observe(Command(0.8, 0.8), packet(frame_id=99, timestamp_ms=99 * MS, left_wheel_cps=GAIN), HEALTHY)
    g.reset()
    assert not g.tripped and g.record == {}


def test_navigation_brakes_for_good_and_ends_the_run_on_a_stall():
    nav = Navigation()
    out = []
    for i in range(40):
        right = 1200.0 if i < 10 else 0.0               # turning, then the right motor dies
        p = packet(frame_id=i, timestamp_ms=i * MS, left_wheel_cps=1200.0, right_wheel_cps=right)
        out.append((nav.update(p), dict(nav.record)))
    first = next(i for i, (_, rec) in enumerate(out) if rec["rule"] == RULE_WHEEL_STALL)
    assert first == 9 + STALL_MS // MS                  # frames 10-19 read the dead wheel
    assert all(cmd == BRAKE and rec["rule"] == RULE_WHEEL_STALL for cmd, rec in out[first:])
    assert out[first][1]["wheel"] == "right" and out[first][1]["reason"] == REASON_STALLED
    assert nav.finished and nav.outcome == OUTCOME_STALLED and nav.end_step == 0
    nav.reset()
    assert not nav.finished and nav.outcome is None
