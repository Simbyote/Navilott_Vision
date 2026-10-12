"""
test_wheel_speed.py  --  src/navigation/wheel_speed.py

Speed control for the wheels: without encoders it is the feedforward at the
configured gain (the tuned duties, unchanged); the gain is learned from
steady frames only and kept within GAIN_LIMITS; the PI brings a weak or a
strong motor (a low or a full pack) onto the target speed; the integral
waits for the INTEGRATE_BAND and is cleared by start(); a driven wheel gets
at least STALL_DUTY, a zero target nothing.

--software  Against a simulated motor (cps = gain x duty, answering the
            command a frame late, as the encoder window does). No robot.
"""
import pytest

from src.navigation.navigation_contract import STALL_DUTY, Command, command_problems
from src.navigation.wheel_speed import (
    CPS_PER_DUTY, GAIN_LIMITS, INTEGRAL_LIMIT, LEARN_ALPHA, WheelSpeed,
)
from src.tests.navigation_checks import packet

MS = 50


def p(i, left=0.0, right=0.0):
    return packet(frame_id=i, timestamp_ms=i * MS, left_wheel_cps=left, right_wheel_cps=right)


def run(ws, target, motor_gain, frames, observe=True):
    """
    Drive target (left, right) counts/s for frames against a motor of
    motor_gain counts/s per duty per wheel, the packet answering last frame's
    command. Returns the commands and the measured speeds.
    """
    cmd, cmds, speeds = None, [], []
    ws.start()
    for i in range(frames):
        cps = (0.0, 0.0) if cmd is None else (motor_gain[0] * cmd.left, motor_gain[1] * cmd.right)
        pk = p(i, *cps)
        if observe:
            ws.observe(cmd, pk)
        cmd = ws.drive(*target, pk)
        cmds.append(cmd)
        speeds.append(cps)
    return cmds, speeds


@pytest.mark.software
def test_without_encoders_it_is_the_tuned_duties():
    ws = WheelSpeed()
    for i in range(20):                                     # every reading 0.0: no encoders
        ws.observe(Command(0.5, 0.73), p(i))
        cmd = ws.drive(0.50 * CPS_PER_DUTY, 0.73 * CPS_PER_DUTY, p(i))
        assert cmd == Command(0.5, 0.73)
    assert not ws.encoders and ws.gains == (CPS_PER_DUTY, CPS_PER_DUTY)
    assert ws.record["closed"] is False and ws.record["duty"] == (0.5, 0.73)


@pytest.mark.software
def test_the_gain_is_learned_from_steady_frames_only():
    ws = WheelSpeed()
    ws.observe(Command(0.5, 0.5), p(0, 1200, 1200))       # no frame before: not steady
    assert ws.encoders and ws.gains == (CPS_PER_DUTY, CPS_PER_DUTY)
    ws.observe(Command(0.6, 0.5), p(1, 1200, 1200))       # left changed 0.1: only the right learns
    left, right = ws.gains
    assert left == CPS_PER_DUTY
    assert right == pytest.approx(CPS_PER_DUTY + LEARN_ALPHA * (2400 - CPS_PER_DUTY))
    for i in range(2, 200):
        ws.observe(Command(0.5, 0.5), p(i, 1200, 1200))   # a tired pack: 2400 per duty
    assert ws.gains == pytest.approx((2400, 2400), rel=0.01)


@pytest.mark.software
@pytest.mark.parametrize("cmd, cps", [
    (Command(0.2, 0.2), (100, 100)),            # below LEARN_MIN_DUTY
    (Command(0.0, 0.0, brake=True), (0, 0)),    # a brake teaches nothing
    (Command(0.5, 0.5), (-800, -800)),          # wheel spinning against the duty
])
def test_frames_that_teach_nothing(cmd, cps):
    ws = WheelSpeed()
    for i in range(50):
        ws.observe(cmd, p(i, *cps))
    assert ws.gains == (CPS_PER_DUTY, CPS_PER_DUTY)


@pytest.mark.software
def test_the_gain_stays_within_its_limits():
    ws = WheelSpeed()
    for i in range(400):
        ws.observe(Command(0.5, 0.5), p(i, 4000, 50))       # lifted wheel, dragged wheel
    lo, hi = (f * CPS_PER_DUTY for f in GAIN_LIMITS)
    assert ws.gains == pytest.approx((hi, lo))


@pytest.mark.software
@pytest.mark.parametrize("factor", [0.75, 1.2])          # a low pack, a full one
def test_the_pi_puts_a_weak_or_strong_motor_on_target(factor):
    ws = WheelSpeed()
    ws.encoders = True                                    # learning off: the PI alone does it
    target = (0.50 * CPS_PER_DUTY, 0.73 * CPS_PER_DUTY)
    motor = (factor * CPS_PER_DUTY, factor * CPS_PER_DUTY)
    cmds, speeds = run(ws, target, motor, 40, observe=False)
    assert speeds[-1] == pytest.approx(target, rel=0.03)  # 2 s at 20 FPS
    assert speeds[20] == pytest.approx(target, rel=0.06)  # most of it inside 1 s
    duty = cmds[-1].left
    assert duty == pytest.approx(0.50 / factor, rel=0.03)
    assert all(command_problems(c) == [] for c in cmds)


@pytest.mark.software
def test_learning_and_the_pi_together_hold_the_speed_as_the_pack_drains():
    ws = WheelSpeed()
    target = (1500.0, 2190.0)
    fresh, _ = run(ws, target, (3300, 3300), 60)
    tired, speeds = run(ws, target, (2300, 2300), 60)
    assert speeds[-1] == pytest.approx(target, rel=0.03)
    assert tired[-1].left > fresh[-1].left               # more duty on the tired pack
    assert ws.gains[0] < 2600                            # and the feedforward learned it


@pytest.mark.software
def test_the_integral_waits_for_the_band_and_start_clears_it():
    ws = WheelSpeed()
    ws.encoders = True
    ws.start()
    ws.drive(1500, 1500, p(0, 0, 0))
    for i in range(1, 10):                               # stalled: error the whole target, outside the band
        ws.drive(1500, 1500, p(i, 0, 0))
    assert all(w.integral == 0.0 for w in ws._wheels)
    for i in range(10, 200):                             # just short of target: inside it, winds to its limit
        ws.drive(1500, 1500, p(i, 1300, 1300))
    assert all(w.integral == pytest.approx(INTEGRAL_LIMIT) for w in ws._wheels)
    ws.start()
    assert all(w.integral == 0.0 for w in ws._wheels)


@pytest.mark.software
def test_a_driven_wheel_gets_the_stall_duty_and_a_zero_target_nothing():
    ws = WheelSpeed()
    cmd = ws.drive(100.0, 0.0, p(0))
    assert cmd == Command(STALL_DUTY, 0.0)
    cmd = ws.drive(-100.0, 5000.0, p(1))
    assert cmd == Command(-STALL_DUTY, 1.0)               # and never past full duty


@pytest.mark.software
def test_reset_forgets_the_gains_and_the_encoders():
    ws = WheelSpeed()
    for i in range(50):
        ws.observe(Command(0.5, 0.5), p(i, 1200, 1200))
    ws.reset()
    assert ws.gains == (CPS_PER_DUTY, CPS_PER_DUTY) and not ws.encoders and ws.record == {}


@pytest.mark.software
def test_nav_csv_carries_the_turn_targets_and_the_learned_gains():
    import src.navigation_linker as nl
    assert {"target_left_cps", "target_right_cps", "gain_left", "gain_right"} <= set(nl.NAV_FIELDS)
    ws = WheelSpeed()
    rec = {"wheels": {"target": (1500.0, 2190.0)}}
    assert nl.wheel_columns(rec, ws) == {"target_left_cps": 1500.0, "target_right_cps": 2190.0,
                                         "gain_left": CPS_PER_DUTY, "gain_right": CPS_PER_DUTY}
    assert nl.wheel_columns({}, None) == dict.fromkeys(("target_left_cps", "target_right_cps",
                                                        "gain_left", "gain_right"))
