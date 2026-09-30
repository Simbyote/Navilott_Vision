"""
test_lane_keeping_demo.py  --  src/scripts/lane_keeping_demo.py

The scripted run against a fake motor: every frame is driven, the stop
triggers brake, and the motor stops at the end even after an error.
"""
import pytest

from src.lane_keeping import LaneKeepingNavigator
from src.navigation import BRAKE
from src.scripts.lane_keeping_demo import DEMO_FRAMES, mock_packet, run


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
    cmds = run(DEMO_FRAMES, LaneKeepingNavigator(), motor, sleep=sleeps.append)
    assert len(cmds) == len(DEMO_FRAMES) and sleeps == [s for _, _, s in DEMO_FRAMES]
    assert motor.calls[-1] == ("stop",) and len(motor.calls) == len(DEMO_FRAMES) + 1
    for cmd, call in zip(cmds, motor.calls):
        assert call == (("brake",) if cmd.brake else ("drive", cmd.left, cmd.right))


@pytest.mark.software
def test_the_demo_drives_until_the_stop_line_threshold_then_brakes():
    cmds = run(DEMO_FRAMES, LaneKeepingNavigator(), FakeMotor(), sleep=lambda s: None)
    braked = [d for (d, _, _), c in zip(DEMO_FRAMES, cmds) if c == BRAKE]
    assert braked == ["stop line 2.0 cm", "stop line 1.0 cm", "stop line 0.5 cm",
                      "stop line 0.2 cm", "stop sign"]
    left, right = cmds[1], cmds[2]                   # offset left, then right
    assert left.left > left.right and right.left < right.right


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
    q = mock_packet(4, offset_cm=None, stop_line_dist=2.0)
    assert q.lane_offset == 0.0 and q.stop_line_detected and q.stop_line_distance_cm == 2.0
