"""
test_drive.py  --  src/peripherals/drive.py

drive.py imports pigpio at module load, so software tests swap in a fake
before importing it; a fake clock steps a simple wheel model that injects
the motion as encoder edges, so nothing waits.

--software  Quadrature decoding, cps, TB6612 direction/PWM commands, brake
            and stop, against a fake pigpio and a simple wheel model. No GPIO.
--hardware  Skips unless pigpio is up and a short nudge makes an encoder
            count, so the motorless chassis skips here. Then spins each
            wheel alone, both forward, reverse and in place, checks the
            counts hold when stopped, and drives 3 s straight open loop,
            recording counts per leg and the motors' natural mismatch.
            WHEELS OFF THE GROUND: about 10 s of motor time.
"""
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.tests.presence import pigpio_or_skip

DRIVE_MODULE = "src.peripherals.drive"
FULL_DUTY = 1_000_000


class FakeCallback:
    def __init__(self, gpio, edge, func):
        self.gpio, self.edge, self.func = gpio, edge, func
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakePi:
    """pigpio.pi stand-in: records pin setup, writes and PWM; delivers encoder edges on request."""
    def __init__(self, connected=True):
        self.connected = connected
        self.modes, self.pulls = {}, {}
        self.levels = {}                        # last write() per pin
        self.pwm = {}                           # pin -> (freq, duty)
        self.callbacks = []
        self._tick = 0

    def set_mode(self, pin, mode):
        self.modes[pin] = mode

    def set_pull_up_down(self, pin, pud):
        self.pulls[pin] = pud

    def callback(self, gpio, edge, func):
        cb = FakeCallback(gpio, edge, func)
        self.callbacks.append(cb)
        return cb

    def write(self, pin, level):
        self.levels[pin] = level

    def hardware_PWM(self, pin, freq, duty):
        self.pwm[pin] = (freq, duty)

    def stop(self):
        self.connected = False

    def edge(self, gpio, level):
        self._tick += 1
        for cb in self.callbacks:
            if cb.gpio == gpio and not cb.cancelled:
                cb.func(gpio, level, self._tick)

    def quad(self, c1, c2, cycles, c1_leads=True):
        """Full quadrature cycles; c1_leads means C1 rises while C2 is low."""
        lead, lag = (c1, c2) if c1_leads else (c2, c1)
        for _ in range(cycles):
            self.edge(lead, 1); self.edge(lag, 1); self.edge(lead, 0); self.edge(lag, 0)


class FakeClock:
    """time stand-in: sleep() advances perf_counter() and is recorded; on_sleep lets a wheel model run."""
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []
        self.on_sleep = None

    def perf_counter(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s
        if self.on_sleep:
            self.on_sleep(self)


@pytest.fixture
def env(monkeypatch):
    """Imports drive.py against a fake pigpio; returns (module, pi, clock), the clock driving the wheel model."""
    pi = FakePi()
    pigpio = types.ModuleType("pigpio")
    pigpio.INPUT, pigpio.OUTPUT = "INPUT", "OUTPUT"
    pigpio.PUD_UP, pigpio.EITHER_EDGE = "PUD_UP", "EITHER_EDGE"
    pigpio.pi = lambda: pi
    monkeypatch.setitem(sys.modules, "pigpio", pigpio)
    monkeypatch.delitem(sys.modules, DRIVE_MODULE, raising=False)
    mod = importlib.import_module(DRIVE_MODULE)
    clock = FakeClock()
    yield mod, pi, clock
    sys.modules.pop(DRIVE_MODULE, None)         # the next import gets the real (or fresh fake) pigpio


def _pins(mod):
    e = mod.EncoderReader
    return e.LEFT_C1, e.LEFT_C2, e.RIGHT_C1, e.RIGHT_C2


def _counted(enc):
    """enc.counts() as named fields: left_count, right_count."""
    left, right = enc.counts()
    return types.SimpleNamespace(left_count=left, right_count=right)


def _assert_stopped(pi, m):
    assert pi.pwm[m.pwma][1] == 0 and pi.pwm[m.pwmb][1] == 0
    for pin in (m.ain1, m.ain2, m.bin1, m.bin2, m.stby):
        assert pi.levels[pin] == 0, f"GPIO {pin} left high after stop"


def _wheel_model(mod, pi, motor, right_gain=1.0, cps_at_full=400):
    """on_sleep hook: turns each wheel duty x gain x cps_at_full x dt counts, as encoder edges."""
    l1, l2, r1, r2 = _pins(mod)
    acc = {"l": 0.0, "r": 0.0, "t": None}

    def step(clock):
        dt = clock.sleeps[-1] if acc["t"] is None else clock.now - acc["t"]
        acc["t"] = clock.now
        if pi.levels.get(motor.stby) != 1:
            return
        for side, pwm, fwd, gain, c1, c2, fwd_leads in (
                # The pin each motor raises for forward, after bc78064's polarity fix
                ("l", motor.pwma, motor.ain2, 1.0, l1, l2, True),
                ("r", motor.pwmb, motor.bin1, right_gain, r1, r2, False)):   # right is mirrored
            other = {motor.ain2: motor.ain1, motor.bin1: motor.bin2}[fwd]
            if pi.levels.get(fwd) and pi.levels.get(other):
                continue                        # short brake: both pins high, the wheel stops
            acc[side] += pi.pwm.get(pwm, (0, 0))[1] / FULL_DUTY * gain * cps_at_full * dt
            n = int(acc[side])
            acc[side] -= n
            pi.quad(c1, c2, n, c1_leads=fwd_leads if pi.levels.get(fwd) else not fwd_leads)
    return step


# -----------------------------------------------------------------------------
# EncoderReader
# -----------------------------------------------------------------------------
@pytest.mark.software
def test_encoder_missing_pigpio_daemon_raises_with_the_fix(env):
    mod, pi, _ = env
    pi.connected = False
    with pytest.raises(RuntimeError, match="sudo pigpiod"):
        mod.EncoderReader(pi)


@pytest.mark.software
def test_encoder_pins_are_pulled_up_inputs_with_either_edge_callbacks(env):
    mod, pi, _ = env
    mod.EncoderReader(pi)
    pins = _pins(mod)
    assert all(pi.modes[p] == "INPUT" and pi.pulls[p] == "PUD_UP" for p in pins)
    assert sorted(cb.gpio for cb in pi.callbacks) == sorted(pins)
    assert all(cb.edge == "EITHER_EDGE" for cb in pi.callbacks)


@pytest.mark.software
def test_left_counts_up_when_c1_leads_and_down_when_c2_leads(env):
    # + = forward, per turning the wheel forward by hand (2026-09-29)
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    pi.quad(l1, l2, 10, c1_leads=True)
    assert enc.counts()[0] == 10
    pi.quad(l1, l2, 4, c1_leads=False)
    assert enc.counts()[0] == 6


@pytest.mark.software
def test_right_count_is_mirrored(env):
    # the right motor faces the other way: the phase order that counts the left up counts the right down
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    _, _, r1, r2 = _pins(mod)
    pi.quad(r1, r2, 7, c1_leads=False)
    assert enc.counts()[1] == 7
    pi.quad(r1, r2, 7, c1_leads=True)
    assert enc.counts()[1] == 0


@pytest.mark.software
def test_only_c1_rising_edges_count(env):
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    pi.edge(l2, 1); pi.edge(l2, 0)              # C2 alone: nothing
    pi.edge(l1, 1); pi.edge(l1, 0)              # C1 rise with C2 low: +1 (left is C1-leads-forward); the fall: nothing
    assert enc.counts()[0] == 1


@pytest.mark.software
def test_counts_are_the_live_totals(env):
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, r1, r2 = _pins(mod)
    pi.quad(l1, l2, 12, c1_leads=True); pi.quad(r1, r2, 5, c1_leads=False)
    assert enc.counts() == (12, 5)
    assert enc.counts() == (12, 5)                  # reading changes nothing


@pytest.mark.software
def test_reset_zeroes_and_cancel_stops_counting(env):
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, r1, r2 = _pins(mod)
    pi.quad(l1, l2, 4); pi.quad(r1, r2, 4)
    enc.reset()
    f = _counted(enc)
    assert (f.left_count, f.right_count) == (0, 0)
    enc.cancel()
    assert all(cb.cancelled for cb in pi.callbacks)
    pi.quad(l1, l2, 5)
    assert enc.counts()[0] == 0


@pytest.mark.software
def test_motor_missing_pigpio_daemon_raises_with_the_fix(env):
    mod, pi, _ = env
    pi.connected = False
    with pytest.raises(RuntimeError, match="sudo pigpiod"):
        mod.MotorController(pi)


@pytest.mark.software
def test_motor_pins_are_outputs_on_separate_pwm_channels_clear_of_the_encoders(env):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    control = (m.ain1, m.ain2, m.bin1, m.bin2, m.stby)
    assert all(pi.modes[p] == "OUTPUT" for p in control)
    channel = {12: 0, 18: 0, 13: 1, 19: 1}      # the Pi's only hardware-PWM pins, two channels
    assert channel[m.pwma] != channel[m.pwmb]
    assert not set(_pins(mod)) & {m.pwma, m.pwmb, *control}


@pytest.mark.software
@pytest.mark.parametrize("left, right, a_in, b_in", [
    # Pin levels per bc78064's polarity fix, from the robot driving backward before it
    (0.5, 0.5, (0, 1), (1, 0)),                 # forward; B is mirrored, so forward is BIN1
    (-0.5, -0.5, (1, 0), (0, 1)),               # reverse
    (0.5, -0.5, (0, 1), (0, 1)),                # spin in place
    (0.0, 0.0, (0, 0), (0, 0)),                 # coast, never short-brake
])
def test_drive_sets_standby_and_direction_pins(env, left, right, a_in, b_in):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    m.drive(left, right)
    assert pi.levels[m.stby] == 1
    assert (pi.levels[m.ain1], pi.levels[m.ain2]) == a_in
    assert (pi.levels[m.bin1], pi.levels[m.bin2]) == b_in


@pytest.mark.software
@pytest.mark.parametrize("cmd, duty", [(0.5, 500_000), (0.333, 333_000), (1.0, FULL_DUTY),
                                       (1.7, FULL_DUTY), (-3.0, FULL_DUTY), (0.0, 0)])
def test_drive_scales_and_clamps_duty(env, cmd, duty):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    m.drive(cmd, cmd)
    assert pi.pwm[m.pwma] == (m.pwm_freq, duty) and pi.pwm[m.pwmb] == (m.pwm_freq, duty)


@pytest.mark.software
def test_brake_shorts_both_motors_with_standby_up(env):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    m.drive(0.5, -0.3)
    m.brake()
    # TB6612 short brake: IN1 = IN2 = H with PWM full; standby low would float the outputs (coast)
    assert all(pi.levels[p] == 1 for p in (m.ain1, m.ain2, m.bin1, m.bin2, m.stby))
    assert pi.pwm[m.pwma][1] == pi.pwm[m.pwmb][1] == FULL_DUTY
    m.stop()
    _assert_stopped(pi, m)


@pytest.mark.software
def test_brake_stops_the_wheels_turning_in_the_wheel_model(env):
    mod, pi, clock = env
    enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    clock.on_sleep = _wheel_model(mod, pi, m)
    m.drive(0.5, 0.5)
    clock.sleep(0.5)
    moving = enc.counts()[0]
    m.brake()
    clock.sleep(0.5)
    assert moving > 0 and enc.counts()[0] == moving


@pytest.mark.software
def test_drive_loads_without_the_display_driver():
    # importing the motor driver mustn't pull in the start button / display
    # module (tm1637) or anything outside src/
    code = ("import sys, types; sys.modules['tm1637'] = None; "
            "p = types.ModuleType('pigpio'); p.pi = object; sys.modules['pigpio'] = p; import " + DRIVE_MODULE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[2])
    assert r.returncode == 0, r.stderr


@pytest.mark.software
def test_stop_zeroes_pwm_and_drops_every_control_pin(env):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    m.drive(0.8, -0.8)
    m.stop()
    _assert_stopped(pi, m)


@pytest.mark.software
@pytest.mark.parametrize("left, right, sign", [(0.5, 0.5, (1, 1)), (-0.5, -0.5, (-1, -1)),
                                               (0.5, -0.5, (1, -1))])
def test_encoder_signs_follow_the_commanded_direction(env, left, right, sign):
    # software model of what the hardware test checks for real
    mod, pi, clock = env
    enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    clock.on_sleep = _wheel_model(mod, pi, m)
    m.drive(left, right)
    clock.sleep(1.0)
    m.stop()
    f = _counted(enc)
    assert (f.left_count * sign[0] > 150) and (f.right_count * sign[1] > 150)


def _motors_respond(enc, motor, sleep, duty=0.4, seconds=0.3):
    """
    Presence probe: a short nudge forward. True if either encoder counted.

    Either side counting means motors are fitted, so a one-sided result runs
    the full test and fails there as a fault, rather than hiding as a skip.
    """
    enc.reset()
    motor.drive(duty, duty)
    sleep(seconds)
    motor.stop()
    f = _counted(enc)
    sleep(seconds)                              # spin down before the real legs
    return f.left_count != 0 or f.right_count != 0


@pytest.mark.software
@pytest.mark.parametrize("fitted", [True, False])
def test_motor_probe_tells_a_bare_chassis_from_a_fitted_one(env, fitted):
    mod, pi, clock = env
    enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    if fitted:
        clock.on_sleep = _wheel_model(mod, pi, m)
    assert _motors_respond(enc, m, clock.sleep) is fitted
    _assert_stopped(pi, m)                      # the probe never leaves the motors on


@pytest.mark.software
def test_motor_probe_counts_a_chassis_with_one_dead_encoder_as_fitted(env):
    # a one-sided fault must reach the real test and fail there, not skip
    mod, pi, clock = env
    enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    model = _wheel_model(mod, pi, m)
    _, _, r1, r2 = _pins(mod)
    right_cbs = [cb for cb in pi.callbacks if cb.gpio in (r1, r2)]
    for cb in right_cbs:
        cb.cancel()                             # right encoder unplugged
    clock.on_sleep = model
    assert _motors_respond(enc, m, clock.sleep)


# -----------------------------------------------------------------------------
# Hardware
# -----------------------------------------------------------------------------
@pytest.mark.hardware
def test_drive_characterization(artifacts):
    import time
    pi = pigpio_or_skip("drive")
    try:
        mod = importlib.import_module(DRIVE_MODULE)
        enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    except BaseException:
        pi.stop()                               # pigpio is up, so a failure here is a fault: raise it
        raise

    def leg(left, right, seconds=1.0):
        enc.reset()
        m.drive(left, right)
        time.sleep(seconds)
        m.stop()
        f = _counted(enc)
        time.sleep(0.5)                         # let the wheels spin down before the next leg
        return {"cmd": [left, right], "s": seconds, "left": f.left_count, "right": f.right_count}

    try:
        if not _motors_respond(enc, m, time.sleep):     # skip raised here still runs the finally
            pytest.skip("drive not found on this chassis: a 0.3 s nudge at 40% duty gave no "
                        "encoder counts on either side (no motors, or both encoders unplugged)")
        left_only = leg(0.5, 0.0)               # catches a motor/encoder cross-pairing
        right_only = leg(0.0, 0.5)
        forward = leg(0.5, 0.5)
        reverse = leg(-0.5, -0.5)
        spin = leg(0.5, -0.5)
        enc.reset()
        time.sleep(0.5)
        coast = _counted(enc)                  # motors stopped: counts should stay put
        enc.reset()
        m.drive(0.5, 0.5)                       # 3 s straight, open loop: the motors' natural mismatch
        time.sleep(3.0)
        m.stop()
        straight = _counted(enc)
    finally:
        m.stop()
        enc.cancel()
        pi.stop()

    total = max(abs(straight.left_count), abs(straight.right_count)) or 1
    artifacts.json("summary.json", {
        "pins": {"pwma": m.pwma, "ain1": m.ain1, "ain2": m.ain2, "pwmb": m.pwmb,
                 "bin1": m.bin1, "bin2": m.bin2, "stby": m.stby,
                 "left_c1_c2": [enc.LEFT_C1, enc.LEFT_C2], "right_c1_c2": [enc.RIGHT_C1, enc.RIGHT_C2]},
        "left_only": left_only, "right_only": right_only,
        "forward": forward, "reverse": reverse, "spin": spin,
        "stopped_0.5s": {"left": coast.left_count, "right": coast.right_count},
        "straight_3s": {"left": straight.left_count, "right": straight.right_count,
                        "mismatch_pct": 100 * abs(straight.left_count - straight.right_count) / total},
        "cps_at_half_duty": {"left": forward["left"] / forward["s"], "right": forward["right"] / forward["s"]},
    })
    # wiring and sign convention first: every later number depends on them
    assert left_only["left"] > 20 and abs(left_only["right"]) <= 2, "left motor moved the right encoder: pairing crossed"
    assert right_only["right"] > 20 and abs(right_only["left"]) <= 2, "right motor moved the left encoder: pairing crossed"
    assert forward["left"] > 20 and forward["right"] > 20, "forward: a wheel has no counts or counts backward"
    assert reverse["left"] < -20 and reverse["right"] < -20, "reverse: a wheel has no counts or counts forward"
    assert spin["left"] > 20 and spin["right"] < -20, "spin: sides don't turn opposite ways"
    assert abs(coast.left_count) <= 1 and abs(coast.right_count) <= 1, "counts while stopped: noisy encoder line"
    # straight_3s's mismatch is recorded, not asserted: open loop, the motors
    # never match exactly; lane keeping and the gyro heading hold correct for it
    assert straight.left_count > 20 and straight.right_count > 20, "straight: a wheel stopped counting"