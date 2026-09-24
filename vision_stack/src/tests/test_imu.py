"""
test_imu.py  --  src/peripherals/imu.py

Software tests drive IMUReader through a scripted fake sensor (the mpu= hook),
so nothing touches I2C. The accumulator is tested directly where timing would
make a threaded test flaky; the threaded tests only check what survives jitter.

--software  Accumulator, calibration, snapshot and worker contract. No sensor.
--hardware  Calibrates the real MPU-6050, samples it at the pipeline frame rate
            for --frames windows, and writes per-window CSV, a summary and a
            yaw-noise histogram. Keep the robot still: the numbers are the
            stationary noise floor.
"""
import math
import threading
import time
import warnings

import pytest

from src.params import FPS, IMU_RATE_HZ
from src.peripherals.imu import IMUFrame, IMUReader, _Accum
from src.tests.artifacts import summarize

FAST_HZ = 1000.0        # keeps calibrate() and the worker quick in software tests


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


def run_for(r, seconds):
    """Start the worker, let it sample, stop it."""
    r.start()
    time.sleep(seconds)
    r.stop()


@pytest.mark.software
def test_empty_frame_is_invalid_and_carries_no_readings():
    f = IMUFrame()
    assert not f.valid and f.mean_yaw_rate_dps is None and f.peak_lateral_accel is None


@pytest.mark.software
def test_frame_with_samples_is_valid():
    assert IMUFrame(0.0, 0.0, sample_count=1).valid


@pytest.mark.software
def test_accumulator_keeps_the_signed_sample_with_the_largest_magnitude():
    # Phase 3 reads the sign, so the peak must not be folded to |a|
    acc = _Accum()
    for ay in (0.2, -0.5, 0.3):
        acc.add(0.0, ay)
    assert acc.peak == -0.5 and acc.n == 3


@pytest.mark.software
def test_snapshot_reports_the_mean_yaw_and_drains_the_window():
    r = reader()
    for yaw in (1.0, 2.0, 6.0):
        r._acc.add(yaw, 0.1)
    f = r.snapshot()
    assert (f.mean_yaw_rate_dps, f.sample_count) == (3.0, 3) and f.valid
    assert not r.snapshot().valid        # the next window starts empty


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
def test_worker_subtracts_the_calibrated_bias_and_converts_to_degrees():
    mpu = FakeMPU(gz_dps=0.5, ay=-0.3)
    r = reader(mpu)
    r.calibrate(samples=5)
    mpu.gz_rad = math.radians(10.5)          # 10 deg/s of real turning on top of the bias
    run_for(r, 0.05)
    f = r.snapshot()
    assert f.valid
    assert f.mean_yaw_rate_dps == pytest.approx(10.0, abs=1e-6)
    assert f.peak_lateral_accel == -0.3


@pytest.mark.software
def test_read_errors_are_counted_not_fatal():
    r = reader(FakeMPU(fail_every=3))
    run_for(r, 0.05)
    assert r.read_errors > 0 and r.snapshot().valid


@pytest.mark.software
def test_worker_paces_to_the_requested_rate():
    r = reader(rate_hz=200.0)
    run_for(r, 0.5)
    n = r.snapshot().sample_count
    # ~100 expected; wide bounds because a loaded CI box oversleeps, and
    # deadline pacing resyncs rather than bursting to catch up
    assert 50 <= n <= 110, n


@pytest.mark.software
def test_start_is_idempotent_and_stop_ends_the_thread():
    r = reader()
    r.start()
    first = r._thread
    r.start()
    assert r._thread is first
    r.stop()
    assert not first.is_alive()


@pytest.mark.software
def test_stop_before_start_is_safe():
    reader().stop()


@pytest.mark.software
def test_snapshot_is_safe_while_the_worker_runs():
    # Swapping the accumulator under the lock must never lose or double-count a
    # window: total samples across snapshots equals the fake's successful reads
    mpu = FakeMPU()
    r = reader(mpu)
    r.start()
    total, stop = 0, time.perf_counter() + 0.2
    while time.perf_counter() < stop:
        total += r.snapshot().sample_count
        time.sleep(0.005)
    r.stop()
    total += r.snapshot().sample_count
    assert total == mpu.reads


@pytest.mark.hardware
def test_imu_characterization(request, artifacts):
    n = request.config.getoption("--frames")
    try:
        r = IMUReader()
    except Exception as e:                   # no board module, no I2C, no sensor
        pytest.skip(f"IMU unavailable: {e}")

    bias_dps = r.calibrate()
    period_s = 1.0 / FPS
    rows = []
    r.start()
    try:
        for i in range(n):
            time.sleep(period_s)
            f = r.snapshot()
            rows.append((i, f.sample_count, f.mean_yaw_rate_dps, f.peak_lateral_accel))
    finally:
        r.stop()

    counts = [row[1] for row in rows]
    yaws = [row[2] for row in rows if row[2] is not None]
    accels = [row[3] for row in rows if row[3] is not None]
    expected = IMU_RATE_HZ / FPS
    assert sum(1 for c in counts if c > 0) >= 0.9 * n, "most frame windows got no IMU sample"

    artifacts.csv("imu_windows.csv", ["window", "sample_count", "mean_yaw_rate_dps", "peak_lateral_accel"], rows)
    artifacts.json("summary.json", {
        "bias_dps": bias_dps, "rate_hz": IMU_RATE_HZ, "frame_fps": FPS,
        "expected_samples_per_window": expected, "read_errors": r.read_errors,
        "samples_per_window": summarize(counts),
        "yaw_rate_dps": summarize(yaws), "lateral_accel_mps2": summarize(accels),
    })
    artifacts.histogram("yaw_noise_hist.png", yaws, "Stationary yaw rate per frame window", "deg/s")

    # Soft flag only: data collection, not a gate. 0.9: sleep jitter costs a sample now and then
    if counts and sum(counts) / len(counts) < 0.9 * expected:
        warnings.warn(f"{sum(counts) / len(counts):.1f} samples per window, expected {expected:.1f}")