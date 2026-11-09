"""test_drive.py  --  src/peripherals/drive.py

drive.py imports pigpio at module load, so software tests swap in a fake
before importing it; a fake clock steps a wheel model that injects the
motion as encoder edges, so nothing waits.

--software  Quadrature decoding, cps, TB6612 commands, brake and stop; the
            watchdog on a fake clock and on its thread; the pigpiod
            watchdog's script, run by a small interpreter of pigpio's script
            language, and its heartbeat and fallback.
--hardware  Skips unless a short nudge makes an encoder count. Each wheel
            alone, both, in place; counts held when stopped; 3 s straight
            open loop. Then drive once and go quiet: the watchdog must stop
            the wheels, and pigpiod's script alone must too. WHEELS OFF THE
            GROUND: about 14 s of motor time.
"""
import importlib
import subprocess
import sys
import time
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

    # Scripts: off unless a test turns them on (scripts_on), as a pigpiod that refuses them
    scripts_on, script_fails = False, False

    def store_script(self, text):
        if not self.scripts_on:
            raise RuntimeError("scripts not supported")
        self.scripts = getattr(self, "scripts", {})
        sid = len(self.scripts)
        self.scripts[sid] = {"text": text.decode(), "status": 0, "params": None, "deleted": False, "updates": []}
        return sid

    def script_status(self, sid):
        sc = self.scripts[sid]
        if sc["status"] == 0:                                   # INITING once, then ready (or failed)
            sc["status"] = 4 if self.script_fails else 1
            return 0, []
        return sc["status"], sc["params"]

    def run_script(self, sid, params):
        self.scripts[sid].update(status=2, params=list(params))

    def update_script(self, sid, params):
        self.scripts[sid]["params"] = list(params)
        self.scripts[sid]["updates"].append(list(params))

    def stop_script(self, sid):
        self.scripts[sid]["status"] = 1

    def delete_script(self, sid):
        self.scripts[sid]["deleted"] = True

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
    (pigpio.PI_SCRIPT_INITING, pigpio.PI_SCRIPT_HALTED, pigpio.PI_SCRIPT_RUNNING,
     pigpio.PI_SCRIPT_WAITING, pigpio.PI_SCRIPT_FAILED) = range(5)
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


def _hold(motor, left, right, seconds, sleep, period=0.05):
    """Command left/right every period for seconds, as a run loop does each frame (the watchdog wants it)."""
    for _ in range(max(1, round(seconds / period))):
        motor.drive(left, right)
        sleep(period)


CROSSTALK = 0.05    # idle side may count up to 5% of the driven side in a one-wheel leg


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


@pytest.mark.hardware
def test_drive_characterization(artifacts):
    pi = pigpio_or_skip("drive")
    try:
        mod = importlib.import_module(DRIVE_MODULE)
        enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    except BaseException:
        pi.stop()                               # pigpio is up, so a failure here is a fault: raise it
        raise

    def leg(left, right, seconds=1.0):
        enc.reset()
        _hold(m, left, right, seconds, time.sleep)
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
        _hold(m, 0.5, 0.5, 3.0, time.sleep)     # 3 s straight, open loop: the motors' natural mismatch
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
    # wiring and sign convention first: every later number depends on them.
    # A crossed pairing puts the driven motor's counts on the OTHER encoder, so
    # compare the two sides rather than demand zero: driving one wheel pivots
    # the chassis and nudges the idle one (2026-10-12: 24 counts against 1770)
    # A dead side says so first: with no counts of its own, any nudge on the other reads as "crossed"
    for side, other, leg_ in (("left", "right", left_only), ("right", "left", right_only)):
        assert leg_[side] > 20, (f"{side} motor alone: {leg_[side]} {side} counts, {leg_[other]} {other} "
                                 f"(motor, its driver channel or its encoder isn't working)")
        assert abs(leg_[other]) <= CROSSTALK * leg_[side], f"{side} motor moved the {other} encoder: pairing crossed"
    assert forward["left"] > 20 and forward["right"] > 20, "forward: a wheel has no counts or counts backward"
    assert reverse["left"] < -20 and reverse["right"] < -20, "reverse: a wheel has no counts or counts forward"
    assert spin["left"] > 20 and spin["right"] < -20, "spin: sides don't turn opposite ways"
    assert abs(coast.left_count) <= 1 and abs(coast.right_count) <= 1, "counts while stopped: noisy encoder line"
    # straight_3s's mismatch is recorded, not asserted: open loop, the motors
    # never match exactly; lane keeping and the gyro heading hold correct for it
    assert straight.left_count > 20 and straight.right_count > 20, "straight: a wheel stopped counting"


@pytest.mark.software
def test_opening_the_encoders_or_motors_names_pigpios_callback_threads(env, monkeypatch):
    mod, pi, _ = env
    calls = []
    monkeypatch.setattr(mod, "name_pigpio_threads", lambda: calls.append(1) or 0)
    mod.EncoderReader(pi)
    mod.MotorController(pi)
    assert len(calls) == 2


@pytest.mark.software
def test_each_encoder_belongs_to_the_wheel_its_motor_command_turns(env):
    # The physical wiring, by raw pin (test_drive_characterization proves it on the robot):
    # the left command's wheel carries the encoder on 19/16, the right's the one on 21/20,
    # and driving forward 19 leads 16 and 21 leads 20 (signs measured by hand 2026-09-29).
    mod, pi, _ = env
    enc = mod.EncoderReader(pi)
    pi.quad(19, 16, 25, c1_leads=True)          # the left wheel forward
    pi.quad(21, 20, 10, c1_leads=True)          # the right wheel forward
    assert enc.counts() == (25, 10)
    pi.quad(19, 16, 5, c1_leads=False)          # the left wheel backward
    assert enc.counts() == (20, 10)


def _braked(pi, m):
    return all(pi.levels[p] == 1 for p in (m.ain1, m.ain2, m.bin1, m.bin2, m.stby)) and \
        pi.pwm[m.pwma][1] == pi.pwm[m.pwmb][1] == FULL_DUTY


def _watched(env, watchdog_s=0.5):
    """A MotorController on the fake pigpio whose watchdog reads the fake clock (its thread can't trip it early)."""
    mod, pi, clock = env
    return mod.MotorController(pi, watchdog_s=watchdog_s, clock=lambda: clock.now), pi, clock


@pytest.mark.software
def test_the_default_watchdog_is_the_params_timeout(env):
    from src.params import MOTOR_WATCHDOG_S
    mod, pi, _ = env
    assert mod.MotorController(pi).watchdog_s == MOTOR_WATCHDOG_S


@pytest.mark.software
def test_a_quiet_loop_is_braked_once_drive_is_overdue(env):
    m, pi, clock = _watched(env)
    m.drive(0.5, 0.5)
    clock.now += 0.49
    assert not m.check() and not _braked(pi, m), "braked before the timeout"
    clock.now += 0.02
    m.check()                                   # or the thread did: either way, one trip
    assert _braked(pi, m) and m.watchdog_trips == 1
    clock.now += 5.0
    assert not m.check() and m.watchdog_trips == 1, "a trip disarms: it brakes once"
    m.stop()


@pytest.mark.software
def test_a_steady_loop_never_trips_it(env):
    m, pi, clock = _watched(env)
    for _ in range(200):                        # 10 s at 20 FPS
        m.drive(0.4, 0.4)
        clock.now += 0.05
        assert not m.check()
    assert m.watchdog_trips == 0 and pi.levels[m.ain2] == 1 and not _braked(pi, m)
    m.stop()


@pytest.mark.software
@pytest.mark.parametrize("then", ["brake", "stop"])
def test_brake_and_stop_disarm_it_since_both_leave_the_motors_safe(env, then):
    m, pi, clock = _watched(env)
    m.drive(0.5, 0.5)
    getattr(m, then)()
    clock.now += 10.0
    assert not m.check() and m.watchdog_trips == 0
    if then == "stop":
        _assert_stopped(pi, m)                  # left in standby, not turned into a brake


@pytest.mark.software
def test_driving_again_after_a_trip_drives_and_re_arms(env):
    m, pi, clock = _watched(env)
    m.drive(0.5, 0.5)
    clock.now += 0.6
    m.check()
    m.drive(-0.3, 0.3)                          # a slow loop that came back carries on
    assert not _braked(pi, m) and pi.levels[m.ain1] == 1 and pi.levels[m.bin1] == 1
    clock.now += 0.6
    m.check()
    assert _braked(pi, m) and m.watchdog_trips == 2
    m.stop()


@pytest.mark.software
def test_the_watchdog_can_be_turned_off(env):
    mod, pi, clock = env
    for off in (None, 0):
        m = mod.MotorController(pi, watchdog_s=off, clock=lambda: clock.now)
        m.drive(0.5, 0.5)
        clock.now += 60.0
        assert m.watchdog_s is None and not m.check() and m._wd_thread is None


@pytest.mark.software
def test_its_thread_brakes_a_stalled_loop_in_real_time_names_itself_and_ends_with_stop(env, monkeypatch):
    mod, pi, _ = env
    names = []
    monkeypatch.setattr(mod, "name_os_thread", lambda name, thread=None: names.append(name) or True)
    m = mod.MotorController(pi, watchdog_s=0.05)
    m.drive(0.5, 0.5)                           # then nothing: the loop is stuck
    deadline = time.monotonic() + 2.0
    while m.watchdog_trips == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert m.watchdog_trips == 1 and _braked(pi, m)
    assert names == ["motor-watchdog"]
    thread = m._wd_thread
    m.stop()
    assert not thread.is_alive() and m._wd_thread is None
    _assert_stopped(pi, m)


@pytest.mark.software
def test_a_failing_brake_is_logged_and_the_watching_goes_on(env, caplog):
    mod, pi, _ = env
    m = mod.MotorController(pi, watchdog_s=0.05)
    m.drive(0.5, 0.5)
    real_write, fails = pi.write, []

    def write(pin, level):
        if not fails:
            fails.append(pin)
            raise ConnectionError("pigpio socket closed")
        real_write(pin, level)
    pi.write = write
    deadline = time.monotonic() + 2.0
    while m.watchdog_trips == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fails and "check failed" in caplog.text
    assert m.watchdog_trips == 1 and _braked(pi, m), "the next look braked"
    m.stop()


@pytest.mark.hardware
def test_the_watchdog_stops_the_real_wheels_when_drive_goes_quiet():
    # As when the camera stops mid-run: one drive(), then silence. The wheels
    # turn, the watchdog brakes them within watchdog_s (+ one check period),
    # and they stay stopped. WHEELS OFF THE GROUND.
    pi = pigpio_or_skip("drive")
    try:
        mod = importlib.import_module(DRIVE_MODULE)
        enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    except BaseException:
        pi.stop()
        raise
    try:
        if not _motors_respond(enc, m, time.sleep):
            pytest.skip("drive not found on this chassis: a 0.3 s nudge at 40% duty gave no encoder counts")
        enc.reset()
        m.drive(0.4, 0.4)                       # the loop's last command, then it's stuck
        time.sleep(0.3)
        moving = _counted(enc)
        time.sleep(m.watchdog_s * (1 + 1 / mod.WATCHDOG_CHECKS) + 0.3)     # the trip, then spin-down
        trips = m.watchdog_trips
        enc.reset()
        time.sleep(0.5)
        after = _counted(enc)
    finally:
        m.stop()
        enc.cancel()
        pi.stop()
    assert moving.left_count > 10 and moving.right_count > 10, "the wheels never turned: nothing to stop"
    assert trips == 1, f"watchdog tripped {trips} times, want 1"
    assert abs(after.left_count) <= 2 and abs(after.right_count) <= 2, "wheels still turning after the watchdog"


@pytest.mark.hardware
def test_the_pigpiod_watchdog_alone_stops_the_real_wheels():
    # As when the whole process stalls: one drive(), then the Python watchdog
    # can't run (its check is stubbed out), so only the script in pigpiod can
    # brake. It must, within watchdog_s plus one look. WHEELS OFF THE GROUND.
    pi = pigpio_or_skip("drive")
    try:
        mod = importlib.import_module(DRIVE_MODULE)
        enc, m = mod.EncoderReader(pi), mod.MotorController(pi)
    except BaseException:
        pi.stop()
        raise
    try:
        assert m.watchdog_status == "python + pigpiod", m.watchdog_status
        if not _motors_respond(enc, m, time.sleep):
            pytest.skip("drive not found on this chassis: a 0.3 s nudge at 40% duty gave no encoder counts")
        m.check = lambda: False                 # the process is "stalled": its own watchdog never looks
        enc.reset()
        m.drive(0.4, 0.4)
        time.sleep(0.3)
        moving = _counted(enc)
        time.sleep(m.watchdog_s + mod.DAEMON_CHECK_MS / 1000.0 + 0.3)        # the script's trip, then spin-down
        enc.reset()
        time.sleep(0.5)
        after = _counted(enc)
    finally:
        m.stop()
        enc.cancel()
        pi.stop()
    assert moving.left_count > 10 and moving.right_count > 10, "the wheels never turned: nothing to stop"
    assert abs(after.left_count) <= 2 and abs(after.right_count) <= 2, "wheels still turning: pigpiod didn't brake"


class ScriptSim:
    """
    pigpio's script language, the commands the watchdog script uses, run on
    simulated milliseconds against FakePi. Any other command fails the test,
    so the script can't quietly grow one this doesn't model. CMP sets
    F = A - x; JZ / JNZ / JM test F (pigpio's semantics).
    """
    ARITY = {"ld": 2, "lda": 1, "cmp": 1, "jz": 1, "jnz": 1, "jm": 1, "jmp": 1, "tag": 1, "inr": 1,
             "mils": 1, "w": 2, "hp": 3}

    def __init__(self, text, pi, params):
        tokens, self.code = text.split(), []
        while tokens:
            cmd = tokens.pop(0).lower()
            assert cmd in self.ARITY, f"command the simulator doesn't model: {cmd}"
            self.code.append((cmd, [tokens.pop(0) for _ in range(self.ARITY[cmd])]))
        self.tags = {int(a[0]): i for i, (c, a) in enumerate(self.code) if c == "tag"}
        self.pi, self.p, self.v = pi, list(params) + [0] * (10 - len(params)), [0] * 150
        self.a = self.f = self.pc = self.t = 0

    def val(self, x):
        return self.p[int(x[1:])] if x[0] == "p" else self.v[int(x[1:])] if x[0] == "v" else int(x)

    def run_until(self, t_ms):
        steps = 0
        while self.t < t_ms:
            steps += 1
            assert steps < 100_000, "the script loops without waiting"
            cmd, a = self.code[self.pc]
            self.pc += 1
            if cmd == "ld":
                self.v[int(a[0][1:])] = self.val(a[1])
            elif cmd == "lda":
                self.a = self.val(a[0])
            elif cmd == "cmp":
                self.f = self.a - self.val(a[0])
            elif cmd == "inr":
                self.v[int(a[0][1:])] += 1
                self.f = self.v[int(a[0][1:])]
            elif cmd in ("jz", "jnz", "jm", "jmp"):
                go = {"jz": self.f == 0, "jnz": self.f != 0, "jm": self.f < 0, "jmp": True}[cmd]
                if go:
                    self.pc = self.tags[int(a[0])]
            elif cmd == "mils":
                self.t += int(a[0])
            elif cmd == "w":
                self.pi.write(int(a[0]), int(a[1]))
            elif cmd == "hp":
                self.pi.hardware_PWM(int(a[0]), int(a[1]), int(a[2]))


def _daemon_pins(m):
    return {"stby": m.stby, "ain1": m.ain1, "ain2": m.ain2, "bin1": m.bin1, "bin2": m.bin2,
            "pwma": m.pwma, "pwmb": m.pwmb}


def _sim(env, timeout_s=0.5):
    """The script for a default MotorController's pins, armed, on a fresh FakePi; and that controller."""
    mod, pi, clock = env
    m = mod.MotorController(pi, daemon_watchdog=False)
    pi.levels, pi.pwm = {}, {}
    return ScriptSim(mod.daemon_script(_daemon_pins(m), m.pwm_freq, timeout_s), pi, [0, 1]), pi, m


@pytest.mark.software
def test_the_script_never_brakes_while_the_heartbeat_moves(env):
    sim, pi, m = _sim(env)
    for t in range(50, 3000, 50):                 # a drive() every 50 ms, as a frame loop
        sim.p[0] += 1
        sim.run_until(t)
    assert pi.levels == {} and pi.pwm == {}


@pytest.mark.software
def test_the_script_brakes_once_the_heartbeat_stops_for_the_timeout(env):
    sim, pi, m = _sim(env)
    for t in range(50, 1001, 50):
        sim.p[0] += 1
        sim.run_until(t)
    sim.run_until(1500)
    assert pi.levels == {}, "not before the timeout"
    sim.run_until(1601)
    assert _braked(pi, m), "within one look after it"


@pytest.mark.software
def test_a_disarmed_script_never_brakes_however_long_it_is_quiet(env):
    sim, pi, m = _sim(env)
    sim.p[1] = 0                                  # brake() / stop(): safe however long the loop is gone
    sim.run_until(10_000)
    assert pi.levels == {} and pi.pwm == {}


@pytest.mark.software
def test_after_a_brake_a_new_heartbeat_starts_the_watch_again(env):
    sim, pi, m = _sim(env)
    sim.run_until(700)
    assert _braked(pi, m)
    pi.levels, pi.pwm = {}, {}
    sim.run_until(3000)
    assert pi.levels == {}, "it brakes once, then waits for the heartbeat to move"
    for t in range(3050, 4001, 50):
        sim.p[0] += 1
        sim.run_until(t)
    assert pi.levels == {}
    sim.run_until(4700)
    assert _braked(pi, m), "quiet again: braked again"


@pytest.mark.software
def test_the_controller_beats_on_drive_and_disarms_on_brake(env):
    mod, pi, clock = env
    pi.scripts_on = True
    m = mod.MotorController(pi, clock=lambda: clock.now)
    assert m.watchdog_status == "python + pigpiod" and m.daemon.status == "running"
    sc = pi.scripts[0]
    assert sc["status"] == 2 and sc["params"] == [0, 0], "running, disarmed until the first drive()"
    assert sc["text"] == mod.daemon_script(_daemon_pins(m), m.pwm_freq, mod.MOTOR_WATCHDOG_S)
    m.drive(0.5, 0.5)
    m.drive(0.4, 0.6)
    m.brake()
    assert sc["updates"] == [[1, 1], [2, 1], [2, 0]]
    m.stop()
    assert sc["deleted"] and m.daemon is None
    m.drive(0.3, 0.3)                             # stop() ended it; the next drive() starts a fresh one
    assert pi.scripts[1]["updates"] == [[1, 1]] and m.watchdog_status == "python + pigpiod"
    m.stop()


@pytest.mark.software
def test_a_python_watchdog_trip_disarms_the_daemon(env):
    mod, pi, clock = env
    pi.scripts_on = True
    m = mod.MotorController(pi, watchdog_s=0.5, clock=lambda: clock.now)
    m.drive(0.5, 0.5)
    clock.now += 0.6
    assert m.check()
    assert pi.scripts[0]["updates"][-1] == [1, 0]
    m.stop()


@pytest.mark.software
@pytest.mark.parametrize("fails, why", [(False, "scripts not supported"), (True, "pigpiod failed the script")])
def test_a_daemon_that_wont_run_the_script_leaves_the_python_watchdog(env, fails, why):
    mod, pi, clock = env
    pi.scripts_on, pi.script_fails = fails, fails
    m = mod.MotorController(pi, clock=lambda: clock.now)
    assert m.daemon is None and m.watchdog_status == f"python (pigpiod: off ({why}))"
    m.drive(0.5, 0.5)                             # still drives; a refusal isn't retried every frame
    assert m.daemon is None and pi.levels[m.ain2] == 1
    if fails:
        assert pi.scripts[0]["deleted"] and len(pi.scripts) == 1
    m.stop()


@pytest.mark.software
def test_no_daemon_when_asked_not_or_with_the_watchdog_off(env):
    mod, pi, clock = env
    pi.scripts_on = True
    assert mod.MotorController(pi, daemon_watchdog=False).watchdog_status == "python"
    assert mod.MotorController(pi, watchdog_s=None).watchdog_status == "off"
    assert not getattr(pi, "scripts", {})
