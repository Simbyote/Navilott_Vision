"""
test_imu.py  --  src/peripherals/imu.py

Software tests drive IMUReader through a scripted fake sensor (the mpu= hook),
so nothing touches I2C.

--software  Calibration and the single read() the sensor hub samples. No sensor.
--hardware  Skips unless an MPU-6050 answers on I2C at 0x68 or 0x69. Then
            calibrates it and reads it the way the robot does, through
            sensing.SensorHub, grouped per frame at the pipeline frame rate
            for --frames windows; writes per-window CSV, a summary and a
            yaw-noise histogram. And checks the one-transfer block read
            against the Adafruit driver's own reads on the same chip, and
            times both. Keep the robot still: the numbers are the
            stationary noise floor.
"""
import struct
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
# The chip's encoding, written out here from the datasheet rather than
# imported, so the fake is an independent check of the reader's decoding
SAMPLE_REG = 0x3B                                   # ACCEL_XOUT_H (register map 4.17)
GYRO_LSB_PER_DPS = (131.0, 65.5, 32.8, 16.4)        # FS_SEL 0-3 (datasheet 6.1)
ACCEL_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0) # AFS_SEL 0-3 (datasheet 6.2)
STANDARD_GRAVITY = 9.80665


class FakeMPU:
    """
    Stands in for the Adafruit MPU6050 at the register level: its range
    settings, and an i2c_device that answers the 14-byte sample block from
    0x3B (accel X Y Z, temperature, gyro X Y Z) the way the chip encodes it,
    so the reader's decoding is what's tested. Counts the transfers.
    """
    def __init__(self, gz_dps=0.0, ay=0.0, fail_every=0, gyro_range=1, accel_range=0, others=None):
        self.gz_dps, self.ay = gz_dps, ay
        self.gyro_range, self.accelerometer_range = gyro_range, accel_range    # the driver's defaults: 500 dps, 2 g
        self.others = others or {}            # raw values for the other fields, to catch reading the wrong one
        self.fail_every = fail_every          # raise on every Nth transfer; 0 never fails
        self.transfers = 0
        self.i2c_device = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write_then_readinto(self, out, buf, **kw):
        assert bytes(out) == bytes([SAMPLE_REG]) and len(buf) == 14 and not kw, "not the one sample-block read"
        self.transfers += 1
        if self.fail_every and self.transfers % self.fail_every == 0:
            raise OSError("simulated I2C error")
        clamp = lambda v: max(-32768, min(32767, round(v)))                    # noqa: E731
        raw = [self.others.get(i, 0) for i in range(7)]
        raw[1] = clamp(self.ay / STANDARD_GRAVITY * ACCEL_LSB_PER_G[self.accelerometer_range])
        raw[6] = clamp(self.gz_dps * GYRO_LSB_PER_DPS[self.gyro_range])
        buf[:] = struct.pack(">7h", *raw)


def reader(mpu=None, rate_hz=FAST_HZ):
    """IMUReader on a fake sensor."""
    return IMUReader(mpu=mpu or FakeMPU(), rate_hz=rate_hz)


@pytest.mark.software
def test_calibrate_returns_the_bias_in_degrees():
    assert reader(FakeMPU(gz_dps=0.5)).calibrate(samples=5) == pytest.approx(0.5, abs=0.02)     # one LSB at 500 dps


@pytest.mark.software
def test_calibrate_skips_failed_reads():
    assert reader(FakeMPU(gz_dps=0.5, fail_every=2)).calibrate(samples=6) == pytest.approx(0.5, abs=0.02)


@pytest.mark.software
def test_calibrate_raises_when_every_read_fails():
    with pytest.raises(RuntimeError, match="no successful reads"):
        reader(FakeMPU(fail_every=1)).calibrate(samples=3)


@pytest.mark.software
def test_read_is_one_bias_corrected_reading_in_degrees_in_the_drivers_frame():
    mpu = FakeMPU(gz_dps=0.5, ay=-0.3)
    r = reader(mpu)
    r.calibrate(samples=5)
    mpu.gz_dps = 10.5
    yaw, ay = r.read()
    # within one LSB (1/65.5 deg/s, 2 g / 32768); no sign flip: that's sensing's job
    assert yaw == pytest.approx(10.0, abs=0.02) and ay == pytest.approx(-0.3, abs=1e-3)


@pytest.mark.software
def test_a_reading_is_one_i2c_transfer():
    mpu = FakeMPU(gz_dps=3.0)
    r = reader(mpu)
    r.read()
    r.read()
    assert mpu.transfers == 2


@pytest.mark.software
@pytest.mark.parametrize("gyro_range", [0, 1, 2, 3])
@pytest.mark.parametrize("accel_range", [0, 1, 2, 3])
def test_each_full_scale_setting_is_scaled_by_its_datasheet_sensitivity(gyro_range, accel_range):
    # +-250..2000 deg/s and +-2..16 g: a value inside every range, read back to within one LSB
    yaw, ay = reader(FakeMPU(gz_dps=-123.4, ay=7.5, gyro_range=gyro_range, accel_range=accel_range)).read()
    assert yaw == pytest.approx(-123.4, abs=1.0 / GYRO_LSB_PER_DPS[gyro_range])
    assert ay == pytest.approx(7.5, abs=STANDARD_GRAVITY / ACCEL_LSB_PER_G[accel_range])


@pytest.mark.software
def test_gyro_z_and_accel_y_come_from_their_own_place_in_the_block():
    # every other field set to something large: picking the wrong offset shows
    others = {0: 11111, 2: -22222, 3: 3333, 4: -4444, 5: 5555}
    yaw, ay = reader(FakeMPU(gz_dps=-50.0, ay=-2.0, others=others)).read()
    assert yaw == pytest.approx(-50.0, abs=0.02) and ay == pytest.approx(-2.0, abs=1e-3)


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

@pytest.mark.hardware
def test_the_block_read_agrees_with_the_adafruit_driver_and_is_faster(artifacts):
    """
    Robot still. Interleaves read()'s one transfer with the driver's own
    .gyro / .acceleration (7 transfers) on the same chip: the two agree to
    within the stationary noise, and the block read takes well under half
    the time per reading.
    """
    import math
    i2c_device_or_skip("imu", MPU6050_ADDRESSES)
    try:
        r = IMUReader()
    except ImportError as e:
        pytest.skip(f"imu unavailable: {e}")
    mpu, n = r._mpu, 200
    block_s = driver_s = 0.0
    dz, dy = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        gz, ay = r.read()                    # no calibrate(): the bias is 0, both are raw
        t1 = time.perf_counter()
        gz_ref, ay_ref = math.degrees(mpu.gyro[2]), mpu.acceleration[1]
        t2 = time.perf_counter()
        block_s, driver_s = block_s + (t1 - t0), driver_s + (t2 - t1)
        dz.append(gz - gz_ref)
        dy.append(ay - ay_ref)
    block_ms, driver_ms = 1000 * block_s / n, 1000 * driver_s / n
    mean_dz, mean_dy = sum(dz) / n, sum(dy) / n
    artifacts.json("block_read.json", {
        "reads": n, "block_ms_per_read": block_ms, "adafruit_ms_per_read": driver_ms,
        "speedup": driver_ms / block_ms if block_ms else None,
        "mean_gyro_z_diff_dps": mean_dz, "mean_accel_y_diff_mps2": mean_dy,
        "gyro_lsb_per_dps": r._gyro_lsb, "accel_lsb_per_g": r._accel_lsb,
    })
    print(f"\n  block read {block_ms:.2f} ms, Adafruit {driver_ms:.2f} ms per reading "
          f"({driver_ms / block_ms:.1f}x); mean diff gyro Z {mean_dz:+.3f} deg/s, accel Y {mean_dy:+.4f} m/s^2")
    # still: the two read the same signal a few ms apart, so they differ by noise, not scale or offset
    assert abs(mean_dz) < 0.2, "gyro Z disagrees with the driver: wrong offset, scale or sign"
    assert abs(mean_dy) < 0.05, "accel Y disagrees with the driver: wrong offset, scale or sign"
    assert block_ms < driver_ms / 2, "the block read isn't faster: is it still several transfers?"
