"""
test_imu.py  --  src/peripherals/imu.py

Software tests drive IMUReader through a scripted fake sensor (the mpu= hook),
so nothing touches I2C.

--software  Calibration and the single read() the sensor hub samples. No sensor.
--hardware  Skips unless an MPU-6050 answers on I2C at 0x68 or 0x69. Then
            calibrates it and reads it the way the robot does, through
            sensing.SensorHub, grouped per frame at the pipeline frame rate
            for --frames windows; writes per-window CSV, a summary and a
            yaw-noise histogram. Keep the robot still: the numbers are the
            stationary noise floor.
"""
import math
import time
import warnings

import pytest

from src.params import FPS, IMU_RATE_HZ
from src.peripherals.imu import IMUReader
from src.peripherals.sensing import SensorHub
from src.tests.artifacts import summarize
from src.tests.presence import i2c_device_or_skip

FAST_HZ = 1000.0        # keeps calibrate() quick in software tests
MPU6050_ADDRESSES = (0x68, 0x69)    # AD0 low / high


class FakeMPU:
    """Stands in for the Adafruit driver: fixed or scripted gyro/accel readings, in its units (rad/s, m/s^2)."""
    def __init__(self, gz_dps=0.0, ay=0.0, fail_every=0):
        self.gz_rad = math.radians(gz_dps)
        self.ay = ay
        self.fail_every = fail_every      # raise on every Nth read; 0 never fails
        self.reads = 0

    def _tick(self):
        self.reads += 1
        if self.fail_every and self.reads % self.fail_every == 0:
            raise OSError("simulated I2C error")

    @property
    def gyro(self):
        self._tick()
        return (0.0, 0.0, self.gz_rad)

    @property
    def acceleration(self):
        return (0.0, self.ay, 9.81)


def reader(mpu=None, rate_hz=FAST_HZ):
    """IMUReader on a fake sensor."""
    return IMUReader(mpu=mpu or FakeMPU(), rate_hz=rate_hz)


@pytest.mark.software
def test_calibrate_returns_the_bias_in_degrees():
    assert reader(FakeMPU(gz_dps=0.5)).calibrate(samples=5) == pytest.approx(0.5)


@pytest.mark.software
def test_calibrate_skips_failed_reads():
    assert reader(FakeMPU(gz_dps=0.5, fail_every=2)).calibrate(samples=6) == pytest.approx(0.5)


@pytest.mark.software
def test_calibrate_raises_when_every_read_fails():
    with pytest.raises(RuntimeError, match="no successful reads"):
        reader(FakeMPU(fail_every=1)).calibrate(samples=3)


@pytest.mark.software
def test_read_is_one_bias_corrected_reading_in_degrees_in_the_drivers_frame():
    mpu = FakeMPU(gz_dps=0.5, ay=-0.3)
    r = reader(mpu)
    r.calibrate(samples=5)
    mpu.gz_rad = math.radians(10.5)
    yaw, ay = r.read()
    assert yaw == pytest.approx(10.0, abs=1e-6) and ay == -0.3      # no sign flip: that's sensing's job


@pytest.mark.software
def test_read_raises_a_failed_read_for_its_caller_to_count():
    with pytest.raises(OSError):
        reader(FakeMPU(fail_every=1)).read()


@pytest.mark.hardware
def test_imu_characterization(request, artifacts):
    n = request.config.getoption("--frames")
    i2c_device_or_skip("imu", MPU6050_ADDRESSES)
    try:
        r = IMUReader()
    except ImportError as e:                 # the part answered, but its driver isn't installed
        pytest.skip(f"imu unavailable: {e}")

    bias_dps = r.calibrate()
    period_s = 1.0 / FPS
    rows = []
    hub = SensorHub(imu=r)                   # the path the robot reads the IMU through
    hub.start()
    hub.drain()                              # the first window starts now
    try:
        for i in range(n):
            time.sleep(period_s)
            b = hub.drain()
            rows.append((i, b.imu_count, b.mean_yaw_dps, b.peak_lateral_accel))
    finally:
        hub.stop()

    counts = [row[1] for row in rows]
    yaws = [row[2] for row in rows if row[2] is not None]
    accels = [row[3] for row in rows if row[3] is not None]
    expected = IMU_RATE_HZ / FPS
    assert sum(1 for c in counts if c > 0) >= 0.9 * n, "most frame windows got no IMU sample"

    artifacts.csv("imu_windows.csv", ["window", "sample_count", "mean_yaw_rate_dps", "peak_lateral_accel"], rows)
    artifacts.json("summary.json", {
        "bias_dps": bias_dps, "rate_hz": IMU_RATE_HZ, "frame_fps": FPS,
        "expected_samples_per_window": expected, "read_errors": hub.read_errors,
        "samples_per_window": summarize(counts),
        "yaw_rate_dps": summarize(yaws), "lateral_accel_mps2": summarize(accels),
    })
    artifacts.histogram("yaw_noise_hist.png", yaws, "Stationary yaw rate per frame window", "deg/s")

    # Soft flag only: data collection, not a gate. 0.9: sleep jitter costs a sample now and then
    if counts and sum(counts) / len(counts) < 0.9 * expected:
        warnings.warn(f"{sum(counts) / len(counts):.1f} samples per window, expected {expected:.1f}")