"""
test_traffic_light.py  --  src/navigation/traffic_light.py

The traffic-light rule against a real tracker: red at the line waits until
the light changes; green and caution drive on; red with no line, or a light
that turns red after the line, says nothing; reset.
"""
import pytest

from src.navigation.navigation_contract import BRAKE
from src.navigation.stop_line import STOP_DELAY_MS, StopLineTracker
from src.tests.navigation_checks import packet
from src.navigation.traffic_light import RED_STATE, REASON_RED, TrafficLightRule

MS = 50
LINE = [30.0, 15.0, 5.0]
REACHED = len(LINE) + STOP_DELAY_MS // MS


def drive(states, lines=LINE):
    """One drive_state per frame; a stop line on the first frames. Returns (rule, [(command, record)])."""
    tracker = StopLineTracker()
    rule = TrafficLightRule(tracker)
    out = []
    for i, state in enumerate(states):
        rows = lines[i] if i < len(lines) else None
        p = packet(frame_id=i, timestamp_ms=i * MS, drive_state=state, stop_line_detected=rows is not None,
                   stop_line_distance_px=rows)
        tracker.update(p)
        out.append((rule.update(p), dict(rule.record)))
    return rule, out


def braked(out):
    return [i for i, (cmd, _) in enumerate(out) if cmd == BRAKE]


@pytest.mark.software
def test_red_is_phase_3s_stop_state():
    assert RED_STATE == "stop"


@pytest.mark.software
def test_red_at_the_line_waits_until_it_changes():
    green_at = REACHED + 30
    _, out = drive(["stop"] * green_at + ["go"] * 20)
    assert braked(out) == list(range(REACHED, green_at))
    assert out[REACHED][1] == {"reason": REASON_RED} and out[green_at] == (None, {})


@pytest.mark.software
@pytest.mark.parametrize("state", ["go", "caution"])
def test_green_and_caution_drive_on(state):
    _, out = drive([state] * (REACHED + 30))
    assert not braked(out)


@pytest.mark.software
def test_caution_after_red_releases_too():
    _, out = drive(["stop"] * (REACHED + 5) + ["caution"] * 10)
    assert braked(out) == list(range(REACHED, REACHED + 5))


@pytest.mark.software
def test_red_without_a_stop_line_says_nothing():
    _, out = drive(["stop"] * 80, lines=[])
    assert not braked(out)


@pytest.mark.software
def test_a_light_turning_red_after_the_line_is_reached_says_nothing():
    _, out = drive(["go"] * (REACHED + 1) + ["stop"] * 20)
    assert not braked(out)


@pytest.mark.software
def test_red_is_watched_while_a_higher_rule_holds():
    tracker = StopLineTracker()
    rule = TrafficLightRule(tracker)
    tracker.reached = True
    assert rule.update(packet(drive_state="stop"), held=True) == BRAKE


@pytest.mark.software
def test_reset_stops_waiting():
    rule, out = drive(["stop"] * (REACHED + 2))
    assert braked(out)
    rule.reset()
    assert rule.update(packet(drive_state="stop")) is None and rule.record == {}
