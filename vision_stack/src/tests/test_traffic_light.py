"""
test_traffic_light.py  --  src/navigation/traffic_light.py

The traffic-light rule against a real tracker: red or yellow at the line
waits until the light has been green or nothing for RELEASE_MS; green drives
on; red misread as yellow keeps the wait; red with no line, or a light that turns red after the line, says
nothing; red seen within RED_MEMORY_MS before the line counts, older red
doesn't; a dropped frame neither runs the light nor releases the wait; reset.
"""
import pytest

from src.navigation.navigation_contract import BRAKE
from src.navigation.stop_line import StopLineTracker
from src.tests.navigation_checks import packet, reach_frames
from src.navigation.traffic_light import (RED_MEMORY_MS, RED_STATE, REASON_RED, RELEASE_MS, STOP_STATES,
                                          TrafficLightRule)

MS = 50
LINE = [30.0, 15.0, 5.0]
REACHED = len(LINE) + reach_frames(frame_ms=MS)
RELEASE = RELEASE_MS // MS          # frames after the first non-red one that still brake


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
def test_the_memory_and_release_cover_a_few_frames_at_20_fps():
    assert RED_MEMORY_MS == 500 and RELEASE_MS == 250


@pytest.mark.software
def test_red_at_the_line_waits_until_it_has_changed_for_release_ms():
    green_at = REACHED + 30
    _, out = drive(["stop"] * green_at + ["go"] * 20)
    assert braked(out) == list(range(REACHED, green_at + RELEASE))
    assert out[REACHED][1] == {"reason": REASON_RED} and out[green_at + RELEASE] == (None, {})


@pytest.mark.software
def test_green_drives_on():
    _, out = drive(["go"] * (REACHED + 30))
    assert not braked(out)


@pytest.mark.software
def test_yellow_at_the_line_waits_too_and_green_releases_it():
    green_at = REACHED + 10
    _, out = drive(["caution"] * green_at + ["go"] * 20)
    assert braked(out) == list(range(REACHED, green_at + RELEASE))
    assert out[REACHED][1] == {"reason": REASON_RED}


@pytest.mark.software
def test_red_misread_as_yellow_keeps_the_wait():
    # The overexposed red LED reads yellow from some angles (2026-10-07)
    states = ["stop", "caution"] * ((REACHED + 40) // 2)
    _, out = drive(states)
    assert braked(out) == list(range(REACHED, len(states)))
    assert STOP_STATES == (RED_STATE, "caution")


@pytest.mark.software
def test_red_dropped_on_the_reached_frame_still_stops():
    states = ["stop"] * (REACHED + 30)
    states[REACHED] = "go"
    _, out = drive(states)
    assert braked(out) == list(range(REACHED, REACHED + 30))


@pytest.mark.software
@pytest.mark.parametrize("gap_ms, waits", [(RED_MEMORY_MS, True), (RED_MEMORY_MS + MS, False)])
def test_red_counts_only_within_red_memory_ms_of_the_line(gap_ms, waits):
    lines = [LINE[0]] * (RED_MEMORY_MS // MS + 2) + list(LINE)          # a long approach: room for the memory
    reached = len(lines) + reach_frames(frame_ms=MS)
    last_red = reached - gap_ms // MS
    assert last_red >= 0
    _, out = drive(["stop"] * (last_red + 1) + ["go"] * (reached + 30 - last_red - 1), lines=lines)
    assert bool(braked(out)) == waits
    if waits:
        assert braked(out) == list(range(reached, reached + RELEASE))      # then clear for RELEASE_MS


@pytest.mark.software
def test_a_dropped_frame_while_waiting_does_not_release():
    states = ["stop"] * (REACHED + 40)
    for k in (10, 11, 20):                          # dropouts shorter than RELEASE_MS
        states[REACHED + k] = "go"
    _, out = drive(states)
    assert braked(out) == list(range(REACHED, REACHED + 40))


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
def test_reset_stops_waiting_and_forgets_the_red():
    rule, out = drive(["stop"] * (REACHED + 2))
    assert braked(out)
    rule.reset()
    assert rule.update(packet(drive_state="go")) is None and rule.record == {}
    rule.tracker.reached = True
    assert rule.update(packet(drive_state="go")) is None        # no red remembered from before the reset
