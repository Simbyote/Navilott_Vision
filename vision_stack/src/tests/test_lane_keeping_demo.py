"""
test_lane_keeping_demo.py  --  src/scripts/lane_keeping_demo.py

The scripted run against a fake motor: every frame is driven, Navigation
keeps the lane, crosses the stop-sign intersection (straight past the line,
the stop 1.5 s after it leaves the view, the 2 s hold, on again), and the
motor stops at the end even after an error.
"""
import pytest

from src.navigation import Navigation
from src.navigation_contract import BRAKE
from src.scripts.lane_keeping_demo import DEMO_FRAMES, STEP_S, mock_packet, run
from src.stop_line import STOP_DELAY_MS
from src.stop_sign import STOP_SIGN_HOLD_TIME_MS


class FakeMotor:
    def __init__(self):
        self.calls = []

    def drive(self, left, right):
        self.calls.append(("drive", left, right))

    def brake(self):
        self.calls.append(("brake",))

    def stop(self):
        self.calls.append(("stop",))


@pytest.mark.software
def test_every_frame_is_driven_then_the_motor_stops():
    motor, sleeps = FakeMotor(), []
    cmds = run(DEMO_FRAMES, Navigation(), motor, sleep=sleeps.append)
    assert len(cmds) == len(DEMO_FRAMES) and sleeps == [s for _, _, s in DEMO_FRAMES]
    assert motor.calls[-1] == ("stop",) and len(motor.calls) == len(DEMO_FRAMES) + 1
    for cmd, call in zip(cmds, motor.calls):
        assert call == (("brake",) if cmd.brake else ("drive", cmd.left, cmd.right))


@pytest.mark.software
def test_the_demo_stops_at_the_stop_sign_line_holds_and_drives_on():
    nav = Navigation()
    rules = []

    class Recording:
        def update(self, packet):
            cmd = nav.update(packet)
            rules.append(nav.record["rule"])
            return cmd

        def reset(self):
            nav.reset()
    cmds = run(DEMO_FRAMES, Recording(), FakeMotor(), sleep=lambda s: None)
    left, right = cmds[1], cmds[2]                   # offset left, then right
    assert left.left > left.right and right.left < right.right
    braked = [i for i, c in enumerate(cmds) if c == BRAKE]
    lost = next(i for i, (d, _, _) in enumerate(DEMO_FRAMES) if d == "past the line")
    frame = lambda ms: round(ms / (STEP_S * 1000))
    assert braked == list(range(lost + frame(STOP_DELAY_MS), lost + frame(STOP_DELAY_MS + STOP_SIGN_HOLD_TIME_MS)))
    assert set(rules[lost:braked[0]]) == {"intersection"} and rules[braked[-1] + 1] == "intersection"
    assert rules[-1] == "lane_keeping"               # both boundaries back


@pytest.mark.software
def test_the_motor_stops_even_if_the_navigator_fails():
    class Broken:
        def update(self, packet):
            raise RuntimeError("boom")
    motor = FakeMotor()
    with pytest.raises(RuntimeError):
        run(DEMO_FRAMES, Broken(), motor, sleep=lambda s: None)
    assert motor.calls == [("stop",)]


@pytest.mark.software
def test_mock_packets_carry_the_offset_both_ways_and_a_stop_line_only_when_given():
    p = mock_packet(3, offset_cm=6.0)
    assert p.lane_offset_cm == 6.0 and p.lane_offset > 0 and not p.stop_line_detected
    q = mock_packet(4, offset_cm=None, stop_line_rows=2.0)
    assert q.lane_offset == 0.0 and q.stop_line_detected and q.stop_line_distance_px == 2.0
    assert mock_packet(5).timestamp_ms == 5 * STEP_S * 1000 and mock_packet(5).lane_mode == "two_boundary"
