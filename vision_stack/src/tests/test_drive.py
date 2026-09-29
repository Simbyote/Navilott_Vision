"""
test_drive.py  --  src/peripherals/drive.py

drive.py imports pigpio at module load, so software tests swap in a fake
before importing it and patch time, so the closed-loop and differential
routines run instantly and wheel motion can be injected as encoder edges.

--software  Quadrature decoding, cps, TB6612 direction/PWM commands, stop,
            and the closed-loop / differential routines, against a fake
            pigpio and a simple wheel model. No GPIO.
--hardware  Skips unless pigpio is up and a short nudge makes an encoder
            count, so the motorless chassis skips here. Then spins each
            wheel alone, both forward, reverse and in place, checks the
            counts hold when stopped, and runs the closed loop, recording
            counts per leg. WHEELS OFF THE GROUND: about 10 s of motor time.
"""
import importlib
import sys
import types

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


class ScriptedEncoders:
    """EncoderReader stand-in: snapshot() returns fixed counts, or raises on a chosen call."""
    def __init__(self, mod, left=0, right=0, raise_on=None):
        self.frame = mod.EncoderFrame(left_count=left, right_count=right)
        self.raise_on = raise_on
        self.calls = self.resets = 0

    def reset(self):
        self.resets += 1

    def snapshot(self):
        if self.calls == self.raise_on:
            raise OSError("simulated encoder read error")
        self.calls += 1
        return self.frame


@pytest.fixture
def env(monkeypatch):
    """Imports drive.py against fake pigpio / time; returns (module, pi, clock)."""
    pi = FakePi()
    pigpio = types.ModuleType("pigpio")
    pigpio.INPUT, pigpio.OUTPUT = "INPUT", "OUTPUT"
    pigpio.PUD_UP, pigpio.EITHER_EDGE = "PUD_UP", "EITHER_EDGE"
    pigpio.pi = lambda: pi
    monkeypatch.setitem(sys.modules, "pigpio", pigpio)
    monkeypatch.delitem(sys.modules, DRIVE_MODULE, raising=False)
    mod = importlib.import_module(DRIVE_MODULE)
    clock = FakeClock()
    monkeypatch.setattr(mod, "time", clock)
    yield mod, pi, clock
    sys.modules.pop(DRIVE_MODULE, None)         # the next import gets the real (or fresh fake) pigpio


def _pins(mod):
    e = mod.EncoderReader
    return e.LEFT_C1, e.LEFT_C2, e.RIGHT_C1, e.RIGHT_C2


def _record_drive(motor):
    """Wraps motor.drive so each (left, right) command is kept; returns the list."""
    cmds, real = [], motor.drive
    motor.drive = lambda l, r: (cmds.append((l, r)), real(l, r))
    return cmds


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
                ("l", motor.pwma, motor.ain1, 1.0, l1, l2, True),
                ("r", motor.pwmb, motor.bin2, right_gain, r1, r2, False)):   # right is mirrored
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
    assert enc.snapshot().left_count == 10
    pi.quad(l1, l2, 4, c1_leads=False)
    assert enc.snapshot().left_count == 6


@pytest.mark.software
def test_right_count_is_mirrored(env):
    # the right motor faces the other way: the phase order that counts the left up counts the right down
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    _, _, r1, r2 = _pins(mod)
    pi.quad(r1, r2, 7, c1_leads=False)
    assert enc.snapshot().right_count == 7
    pi.quad(r1, r2, 7, c1_leads=True)
    assert enc.snapshot().right_count == 0


@pytest.mark.software
def test_only_c1_rising_edges_count(env):
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    pi.edge(l2, 1); pi.edge(l2, 0)              # C2 alone: nothing
    pi.edge(l1, 1); pi.edge(l1, 0)              # C1 rise with C2 low: +1 (left is C1-leads-forward); the fall: nothing
    assert enc.snapshot().left_count == 1


@pytest.mark.software
def test_reset_zeroes_and_cancel_stops_counting(env):
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, r1, r2 = _pins(mod)
    pi.quad(l1, l2, 4); pi.quad(r1, r2, 4)
    enc.reset()
    f = enc.snapshot()
    assert (f.left_count, f.right_count) == (0, 0)
    enc.cancel()
    assert all(cb.cancelled for cb in pi.callbacks)
    pi.quad(l1, l2, 5)
    assert enc.snapshot().left_count == 0


@pytest.mark.software
def test_cps_over_the_first_window(env):
    mod, pi, clock = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    pi.quad(l1, l2, 10, c1_leads=True)
    clock.now += 0.5
    assert enc.snapshot().left_cps == pytest.approx(20.0)


@pytest.mark.software
def test_cps_holds_steady_at_constant_speed(env):
    mod, pi, clock = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    speeds = []
    for _ in range(3):
        pi.quad(l1, l2, 10, c1_leads=True)
        clock.now += 0.5
        speeds.append(enc.snapshot().left_cps)
    assert speeds == pytest.approx([20.0] * 3)


@pytest.mark.software
def test_cps_after_a_reset_counts_only_the_new_window(env):
    mod, pi, clock = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    pi.quad(l1, l2, 30, c1_leads=True)
    clock.now += 0.5
    enc.snapshot()
    enc.reset()
    pi.quad(l1, l2, 5, c1_leads=True)
    clock.now += 0.5
    # 5 counts in 0.5 s; without resetting the previous count too it reads (5 - 30) / 0.5
    assert enc.snapshot().left_cps == pytest.approx(10.0)


@pytest.mark.software
def test_a_stopped_wheel_reads_zero_cps_after_moving(env):
    mod, pi, clock = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    pi.quad(l1, l2, 10, c1_leads=True)
    clock.now += 0.5
    enc.snapshot()
    clock.now += 0.5
    f = enc.snapshot()
    assert (f.left_count, f.left_cps) == (10, 0.0)


@pytest.mark.software
def test_cps_is_zero_rather_than_a_crash_when_no_time_passed(env):
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    l1, l2, _, _ = _pins(mod)
    enc.snapshot()
    pi.quad(l1, l2, 3)
    assert enc.snapshot().left_cps == 0.0


# -----------------------------------------------------------------------------
# MotorController: raw commands
# -----------------------------------------------------------------------------
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
    (0.5, 0.5, (1, 0), (0, 1)),                 # forward; B is mirrored, so forward is BIN2
    (-0.5, -0.5, (0, 1), (1, 0)),               # reverse
    (0.5, -0.5, (1, 0), (1, 0)),                # spin in place
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
def test_stop_zeroes_pwm_and_drops_every_control_pin(env):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    m.drive(0.8, -0.8)
    m.stop()
    _assert_stopped(pi, m)


# -----------------------------------------------------------------------------
# MotorController: movements
# -----------------------------------------------------------------------------
@pytest.mark.software
def test_closed_loop_holds_base_speed_when_wheels_match(env):
    mod, pi, clock = env
    m = mod.MotorController(pi)
    cmds = _record_drive(m)
    enc = ScriptedEncoders(mod, 100, 100)
    m.drive_straight_closed_loop(enc, 0.5, 1.0)
    assert enc.resets == 1
    assert cmds and all(c == pytest.approx((0.5, 0.5)) for c in cmds)
    assert set(clock.sleeps) == {0.02} and len(cmds) == pytest.approx(50, abs=1)   # 50 Hz for 1 s
    _assert_stopped(pi, m)


@pytest.mark.software
def test_closed_loop_correction_is_proportional_then_clamped(env):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    cmds = _record_drive(m)
    m.drive_straight_closed_loop(ScriptedEncoders(mod, 40, 0), 0.5, 0.02, kp=0.001)
    l1, r1 = cmds[0]
    cmds.clear()
    m.drive_straight_closed_loop(ScriptedEncoders(mod, 10_000, 0), 0.5, 0.02, max_corr=0.15)
    l2, r2 = cmds[0]
    assert abs(l1 - r1) == pytest.approx(2 * 40 * 0.001)
    assert abs(l2 - r2) == pytest.approx(2 * 0.15)


@pytest.mark.software
def test_closed_loop_never_commands_below_min_speed(env):
    # base 0.3 less a clamped 0.15 correction would be 0.15; the floor holds it at 0.25
    mod, pi, _ = env
    m = mod.MotorController(pi)
    cmds = _record_drive(m)
    m.drive_straight_closed_loop(ScriptedEncoders(mod, 10_000, 0), 0.3, 0.1, min_speed=0.25)
    assert all(min(c) >= 0.25 for c in cmds)


@pytest.mark.software
def test_closed_loop_slows_the_wheel_that_is_ahead(env):
    mod, pi, _ = env
    m = mod.MotorController(pi)
    cmds = _record_drive(m)
    m.drive_straight_closed_loop(ScriptedEncoders(mod, 60, 40), 0.5, 0.02)
    left, right = cmds[0]
    assert left < right


@pytest.mark.software
def test_closed_loop_keeps_a_weak_right_motor_in_step(env):
    mod, pi, clock = env
    enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    clock.on_sleep = _wheel_model(mod, pi, m, right_gain=0.85)
    m.drive_straight_closed_loop(enc, 0.5, 3.0)
    f = enc.snapshot()
    assert abs(f.left_count - f.right_count) <= 0.05 * max(f.left_count, f.right_count)


@pytest.mark.software
def test_differential_commands_once_logs_at_20hz_and_stops(env):
    mod, pi, clock = env
    m = mod.MotorController(pi)
    cmds = _record_drive(m)
    enc = ScriptedEncoders(mod)
    m.drive_differential_for_duration(enc, 0.3, 0.6, 1.0)
    assert cmds == [(0.3, 0.6)] and enc.resets == 1
    assert set(clock.sleeps) == {0.05} and enc.calls == pytest.approx(20, abs=1)
    _assert_stopped(pi, m)


@pytest.mark.software
@pytest.mark.parametrize("routine", ["straight", "differential"])
def test_motors_stop_even_if_the_encoder_read_fails(env, routine):
    # both routines stop in a finally; a dead encoder must not leave the wheels running
    mod, pi, _ = env
    m = mod.MotorController(pi)
    m.drive(0.5, 0.5)
    enc = ScriptedEncoders(mod, raise_on=3)
    with pytest.raises(OSError):
        if routine == "straight":
            m.drive_straight_closed_loop(enc, 0.5, 1.0)
        else:
            m.drive_differential_for_duration(enc, 0.5, -0.5, 1.0)
    _assert_stopped(pi, m)


@pytest.mark.software
@pytest.mark.parametrize("left, right, sign", [(0.5, 0.5, (1, 1)), (-0.5, -0.5, (-1, -1)),
                                               (0.5, -0.5, (1, -1))])
def test_encoder_signs_follow_the_commanded_direction(env, left, right, sign):
    # software model of what the hardware test checks for real
    mod, pi, clock = env
    enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    clock.on_sleep = _wheel_model(mod, pi, m)
    m.drive_differential_for_duration(enc, left, right, 1.0)
    f = enc.snapshot()
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
    f = enc.snapshot()
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
        f = enc.snapshot()
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
        coast = enc.snapshot()                  # motors stopped: counts should stay put
        m.drive_straight_closed_loop(enc, 0.5, 3.0)
        straight = enc.snapshot()
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
        "closed_loop_3s": {"left": straight.left_count, "right": straight.right_count,
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
    assert abs(straight.left_count - straight.right_count) <= 0.05 * total, \
        "closed loop let the wheels drift apart by more than 5%"