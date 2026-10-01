"""
test_sensing.py  --  src/peripherals/sensing.py

SensorHub runs over fake sensors and a fake clock, with tick() called by the
test, so the grouping, the yaw flip and the wheel-speed math are exact. The
threaded tests only check what survives scheduling jitter.
"""
import sys
import threading
import time
import types

import pytest

from src.params import IMU_YAW_SIGN, SENSOR_HISTORY_S, SENSOR_RATE_HZ
from src.peripherals.sensing import SensorBatch, SensorHub, SensorReading


class FakeIMU:
    """IMUReader.read() stand-in: (raw yaw deg/s, accel m/s^2); raises when told to."""
    def __init__(self, yaw=0.0, accel=0.0):
        self.yaw, self.accel, self.fail, self.reads = yaw, accel, False, 0

    def read(self):
        self.reads += 1
        if self.fail:
            raise OSError("simulated I2C error")
        return self.yaw, self.accel


class FakeEncoders:
    """EncoderReader.counts() stand-in; the test sets left and right."""
    def __init__(self):
        self.left = self.right = 0

    def counts(self):
        return self.left, self.right


class Clock:
    def __init__(self):
        self.now = 50.0

    def __call__(self):
        return self.now


def hub(imu=True, encoders=True, yaw_sign=-1, **kw):
    """
    A started-by-hand hub (no thread) on fakes; returns (hub, imu, encoders, clock).
    It flips yaw (yaw_sign -1) unless told otherwise, so the flip is tested
    whatever this robot's IMU_YAW_SIGN is.
    """
    i = FakeIMU() if imu else None
    e = FakeEncoders() if encoders else None
    clock = Clock()
    h = SensorHub(i, e, clock=clock, yaw_sign=yaw_sign, **kw)
    h._base = h._counts_reading()           # what start() does, without the thread
    return h, i, e, clock


def reading(t, yaw=None, accel=None, left=None, right=None):
    return SensorReading(t, yaw, accel, left, right)


# =============================================================================
# Parameters
# =============================================================================

@pytest.mark.software
def test_this_robots_yaw_sign_and_100_hz_sampling():
    # Upside-down IMU: the driver's + = left (Z-up) reads + = right here,
    # Estimation's convention already (measured 2026-10-01, motors fixed)
    assert IMU_YAW_SIGN == 1
    assert SENSOR_RATE_HZ == 100.0 and SENSOR_HISTORY_S == 2.0


# =============================================================================
# SensorBatch
# =============================================================================

@pytest.mark.software
def test_an_empty_batch_has_nothing_to_report():
    b = SensorBatch(())
    assert (b.span_s, b.imu_count, b.mean_yaw_dps, b.peak_lateral_accel) == (0.0, 0, None, None)
    assert (b.left_count, b.right_count, b.left_cps, b.right_cps) == (None, None, None, None)


@pytest.mark.software
def test_yaw_is_the_mean_of_the_imu_readings_only():
    b = SensorBatch((reading(1.0, 10.0), reading(1.01, 20.0), reading(1.02, None, left=5, right=5)))
    assert b.mean_yaw_dps == pytest.approx(15.0) and b.imu_count == 2


@pytest.mark.software
def test_lateral_accel_is_the_signed_reading_with_the_largest_magnitude():
    b = SensorBatch(tuple(reading(1.0, 0.0, a) for a in (0.5, -1.2, 1.1)))
    assert b.peak_lateral_accel == -1.2


@pytest.mark.software
def test_wheel_speed_is_counts_since_base_over_time_since_base():
    base = reading(1.0, left=100, right=200)
    b = SensorBatch((reading(1.02, 0.0, 0.0, 110, 205), reading(1.05, None, None, 120, 190)), base)
    assert b.left_cps == pytest.approx(20 / 0.05) and b.right_cps == pytest.approx(-10 / 0.05)
    assert (b.left_count, b.right_count) == (120, 190)
    assert b.span_s == pytest.approx(0.05)


@pytest.mark.software
def test_counts_come_from_the_last_reading_that_has_them():
    b = SensorBatch((reading(1.0, left=3, right=4), reading(1.01, yaw=1.0)), reading(0.9, left=0, right=0))
    assert (b.left_count, b.right_count) == (3, 4)
    assert b.left_cps == pytest.approx(3 / 0.1)


@pytest.mark.software
@pytest.mark.parametrize("base", [None, reading(1.0, yaw=0.0), reading(1.05, left=0, right=0)])
def test_wheel_speed_is_zero_without_a_window_to_measure_over(base):
    # no base, a base with no counts, or no time passed since it
    b = SensorBatch((reading(1.05, left=10, right=10),), base)
    assert (b.left_cps, b.right_cps) == (0.0, 0.0)


@pytest.mark.software
def test_span_runs_from_the_first_reading_without_a_base():
    assert SensorBatch((reading(2.0, 0.0), reading(2.03, 0.0))).span_s == pytest.approx(0.03)


# =============================================================================
# SensorHub, ticked by hand
# =============================================================================

@pytest.mark.software
def test_a_tick_reads_both_sensors_together_on_the_hub_clock_and_flips_yaw():
    h, imu, enc, clock = hub()
    imu.yaw, imu.accel, enc.left, enc.right = 30.0, -0.9, 7, 8
    r = h.tick()
    assert r == SensorReading(clock.now, -30.0, -0.9, 7, 8)       # a flipping hub: raw + reads -


@pytest.mark.software
def test_the_yaw_sign_is_configurable_per_robot():
    h, imu, _, _ = hub(yaw_sign=1)
    imu.yaw = 30.0
    assert h.tick().yaw_dps == 30.0


@pytest.mark.software
def test_accel_is_not_flipped():
    h, imu, _, _ = hub()
    imu.accel = 0.7
    assert h.tick().lateral_accel == 0.7


@pytest.mark.software
def test_drain_hands_over_every_reading_since_the_last_drain_oldest_first():
    h, imu, _, clock = hub()
    for i in range(5):
        clock.now += 0.01
        imu.yaw = float(i)
        h.tick()
    b = h.drain()
    imu_readings = [r for r in b.readings if r.yaw_dps is not None]
    assert [r.yaw_dps for r in imu_readings] == [-0.0, -1.0, -2.0, -3.0, -4.0]
    assert [r.t for r in b.readings] == sorted(r.t for r in b.readings)
    assert h.drain().imu_count == 0                     # emptied


@pytest.mark.software
def test_drain_closes_the_window_with_counts_but_never_reads_the_imu():
    h, imu, enc, clock = hub()
    clock.now += 0.03
    enc.left = 9
    reads = imu.reads
    b = h.drain()
    assert imu.reads == reads                          # no I2C in the frame loop
    assert b.readings[-1] == SensorReading(clock.now, None, None, 9, 0)
    assert b.left_cps == pytest.approx(9 / 0.03)


@pytest.mark.software
def test_each_window_measures_speed_from_where_the_last_one_ended():
    h, _, enc, clock = hub()
    clock.now += 0.05; enc.left = 50; h.tick()
    clock.now += 0.05; enc.left = 100
    first = h.drain()
    clock.now += 0.10; enc.left = 110
    second = h.drain()
    assert first.left_cps == pytest.approx(100 / 0.10)
    assert second.base == first.readings[-1]
    assert second.left_cps == pytest.approx(10 / 0.10)
    clock.now += 0.05
    assert h.drain().left_cps == 0.0                    # stopped reads zero, not None


@pytest.mark.software
def test_a_failed_imu_read_is_counted_and_still_keeps_the_counts():
    h, imu, enc, _ = hub()
    imu.fail, enc.left = True, 4
    r = h.tick()
    assert (r.yaw_dps, r.lateral_accel, r.left_count) == (None, None, 4)
    assert h.read_errors == 1


@pytest.mark.software
def test_a_full_buffer_drops_the_oldest_and_says_how_many_without_losing_the_speed():
    h, imu, enc, clock = hub(rate_hz=10.0, history_s=0.5)       # holds 5
    for i in range(8):
        clock.now += 0.1
        imu.yaw, enc.left = float(i), 10 * (i + 1)
        h.tick()
    b = h.drain()
    assert len(b.readings) == 5 and b.dropped == 4               # 8 ticks + the closing reading, 5 kept
    assert b.readings[0].yaw_dps == -4.0                         # the oldest four went
    assert b.left_cps == pytest.approx(80 / 0.8)                 # measured from the base, which isn't dropped
    assert h.drain().dropped == 0


@pytest.mark.software
def test_without_encoders_there_are_no_counts_and_no_closing_reading():
    h, imu, _, clock = hub(encoders=False)
    clock.now += 0.01
    imu.yaw = 5.0
    h.tick()
    b = h.drain()
    assert len(b.readings) == 1 and b.mean_yaw_dps == -5.0
    assert (b.left_count, b.left_cps, b.right_cps) == (None, None, None)


@pytest.mark.software
def test_without_an_imu_readings_carry_only_counts():
    h, _, enc, _ = hub(imu=False)
    enc.right = 3
    r = h.tick()
    assert (r.yaw_dps, r.lateral_accel, r.right_count) == (None, None, 3)
    assert h.drain().mean_yaw_dps is None


@pytest.mark.software
def test_has_sensors_needs_at_least_one():
    assert not SensorHub().has_sensors
    assert SensorHub(imu=FakeIMU()).has_sensors and SensorHub(encoders=FakeEncoders()).has_sensors


# =============================================================================
# SensorHub, on its thread
# =============================================================================

@pytest.mark.software
def test_the_thread_samples_at_about_the_rate_until_stopped():
    h = SensorHub(FakeIMU(yaw=2.0), FakeEncoders(), rate_hz=200.0, yaw_sign=-1)
    h.start()
    time.sleep(0.2)
    b = h.drain()
    h.stop()
    assert 10 <= b.imu_count <= 60                     # ~40 at 200 Hz, with room for a slow machine
    assert b.mean_yaw_dps == -2.0
    assert not any(t.name == "sensor-hub" and t.is_alive() for t in threading.enumerate())


@pytest.mark.software
def test_start_takes_the_first_base_and_is_idempotent():
    enc = FakeEncoders()
    enc.left = 40
    h = SensorHub(None, enc, rate_hz=200.0)
    h.start()
    thread = h._thread
    h.start()
    assert h._thread is thread
    enc.left = 60
    b = h.drain()
    h.stop()
    assert b.base.left_count == 40 and b.left_count == 60


@pytest.mark.software
def test_drain_is_safe_while_the_thread_runs():
    enc = FakeEncoders()
    h = SensorHub(FakeIMU(), enc, rate_hz=1000.0)
    h.start()
    total = 0
    for i in range(200):
        enc.left = i
        total += len(h.drain().readings)
    h.stop()
    assert total >= 200                                # every drain got at least its closing reading


@pytest.mark.software
def test_stop_before_start_is_safe_and_closes_once():
    closed = []
    h = SensorHub()
    h._closers = [lambda: closed.append(1)]
    h.stop()
    h.stop()
    assert closed == [1]


# =============================================================================
# SensorHub.open
# =============================================================================

@pytest.mark.software
def test_open_with_nothing_opens_nothing():
    h = SensorHub.open()
    assert not h.has_sensors
    h.stop()


@pytest.mark.software
def test_open_uses_the_real_drivers_and_stop_releases_the_encoders(monkeypatch):
    from src.tests.test_drive import DRIVE_MODULE, FakePi
    import importlib
    pi = FakePi()
    pigpio = types.ModuleType("pigpio")
    pigpio.INPUT, pigpio.OUTPUT = "INPUT", "OUTPUT"
    pigpio.PUD_UP, pigpio.EITHER_EDGE = "PUD_UP", "EITHER_EDGE"
    pigpio.pi = lambda: pi
    monkeypatch.setitem(sys.modules, "pigpio", pigpio)
    monkeypatch.delitem(sys.modules, DRIVE_MODULE, raising=False)
    drive = importlib.import_module(DRIVE_MODULE)
    imu_mod = types.ModuleType("src.peripherals.imu")
    imu_mod.IMUReader = FakeIMU
    monkeypatch.setitem(sys.modules, "src.peripherals.imu", imu_mod)
    try:
        h = SensorHub.open(imu=True, encoders=True)
        assert isinstance(h._imu, FakeIMU) and isinstance(h._encoders, drive.EncoderReader)
        h.stop()
        assert all(cb.cancelled for cb in pi.callbacks) and not pi.connected
    finally:
        sys.modules.pop(DRIVE_MODULE, None)
