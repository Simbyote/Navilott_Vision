"""IMU background accumulator: MPU-6050 yaw rate and lateral accel per pipeline frame.

Purpose:
    The camera frame rate is too slow to integrate gyro rate cleanly, so a
    daemon thread samples the MPU-6050 (through the Adafruit driver) at
    IMU_RATE_HZ and accumulates into running totals. The main loop drains
    them once per frame, so each frame gets every sample since the last one
    at constant memory and O(1) cost. Phase 3 reads the result by its fields
    (SensorSample.from_imu), so estimation never imports this module or the
    board drivers it pulls in.

Main package:
    IMUFrame: one frame window's bias-corrected mean yaw rate and signed peak
    lateral acceleration, with the sample count; invalid when no sample
    arrived. Axes assume a flat mount, Z up, X forward, Y left: gyro Z is
    + = CCW = turning LEFT (right-hand rule), accel Y is + = accelerating
    left. Verify both signs on the bench for the actual mounting.

Flow:
    1. calibrate(): average gyro Z at standstill to get the zero-rate bias.
    2. start(): sample on a background thread, paced to IMU_RATE_HZ.
    3. snapshot(), once per frame: swap out the accumulator and summarize it.
    4. stop(): end the thread.
"""

import math
import time
import logging
import threading
from dataclasses import dataclass

from src.params import IMU_I2C_ADDRESS, IMU_RATE_HZ

log = logging.getLogger(__name__)


@dataclass
class IMUFrame:
    """Aggregated IMU values for one pipeline frame window. valid is False when no sample arrived."""
    mean_yaw_rate_dps : float | None = None  # bias-corrected gyro-Z mean, deg/s; + = turning left
    peak_lateral_accel: float | None = None  # signed accel-Y sample with the largest |a|, m/s^2; + = left
    sample_count      : int = 0

    @property
    def valid(self) -> bool:
        return self.sample_count > 0


class _Accum:
    """Running yaw sum, sample count and signed peak accel for one frame window."""
    __slots__ = ("yaw_sum", "n", "peak")

    def __init__(self) -> None:
        self.yaw_sum = 0.0
        self.n       = 0
        self.peak    = 0.0

    def add(self, yaw_dps: float, ay: float) -> None:
        self.yaw_sum += yaw_dps
        self.n       += 1
        if abs(ay) > abs(self.peak):
            self.peak = ay


class IMUReader:
    """
    Owns the MPU-6050 and its background sampling thread.

    Inputs:
        address: I2C address; 0x68 with AD0 low, 0x69 with it high.
        rate_hz: Sampling rate. Higher smooths the per-frame mean but costs
            CPU on the Pi and I2C bandwidth.
        mpu: Any object exposing .gyro and .acceleration tuples, to test
            without hardware. None opens the real sensor.

    Side effects:
        With mpu None, opens I2C and sets the sensor's on-chip low-pass filter.
    """

    def __init__(
            self,
            address: int   = IMU_I2C_ADDRESS,
            rate_hz: float = IMU_RATE_HZ,
            mpu            = None,
    ) -> None:
        if mpu is None:
            import board
            import busio
            import adafruit_mpu6050

            i2c = busio.I2C(board.SCL, board.SDA)
            mpu = adafruit_mpu6050.MPU6050(i2c, address=address)
            # Anti-aliasing: band-limit on-chip below the 50 Hz Nyquist limit of 100 Hz sampling
            mpu.filter_bandwidth = adafruit_mpu6050.Bandwidth.BAND_44_HZ

        self._mpu         = mpu
        self._address     = address
        self._rate_hz     = rate_hz
        self._lock        = threading.Lock()
        self._stop_evt    = threading.Event()
        self._acc         = _Accum()
        self._thread      : threading.Thread | None = None
        self._gz_bias_rad = 0.0
        self.read_errors  = 0

    def calibrate(self, samples: int = 200) -> float:
        """
        Estimate the gyro-Z zero-rate bias. Call before start(), with the robot stationary.

        Inputs:
            samples: Reads to average, one per sample period; 200 at 100 Hz
                is 2 s. More averages out noise but lengthens startup.

        Outputs:
            The bias in deg/s. It's subtracted from every later sample, so
            Phase3Config.gyro_bias_dps should stay 0 alongside it.

        Raises:
            RuntimeError: If every read failed.
        """
        interval = 1.0 / self._rate_hz
        total, n = 0.0, 0
        for _ in range(samples):
            try:
                total += self._mpu.gyro[2]
                n     += 1
            except Exception as exc:
                log.debug("IMU calib read error (skipped): %s", exc)
            time.sleep(interval)

        if n == 0:
            raise RuntimeError("IMU calibration failed: no successful reads")

        self._gz_bias_rad = total / n
        bias_dps = math.degrees(self._gz_bias_rad)
        log.info("IMU gyro-Z bias: %.3f deg/s (%d samples)", bias_dps, n)
        return bias_dps

    def start(self) -> None:
        """Start the background sampling thread; a no-op if it's already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target = self._worker,
            daemon = True,
            name   = "imu-accumulator",
        )
        self._thread.start()
        log.info("IMUReader started at %.0f Hz on 0x%02X",
                 self._rate_hz, self._address)

    def stop(self, timeout: float = 0.5) -> None:
        """Signal the worker to stop and wait up to timeout seconds for it to exit."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("IMUReader stopped (%d read errors).", self.read_errors)

    def snapshot(self) -> IMUFrame:
        """
        Drain the accumulator atomically.

        Outputs:
            IMUFrame covering every sample since the previous call (or since
            start() on the first); invalid if none arrived.
        """
        with self._lock:
            acc, self._acc = self._acc, _Accum()   # swap only, so the lock is held for O(1)

        if acc.n == 0:
            return IMUFrame()

        return IMUFrame(
            mean_yaw_rate_dps  = acc.yaw_sum / acc.n,
            peak_lateral_accel = acc.peak,
            sample_count       = acc.n,
        )

    def _worker(self) -> None:
        """Sample at rate_hz until stop(); read errors are counted, not fatal."""
        interval = 1.0 / self._rate_hz
        next_t   = time.perf_counter()

        while not self._stop_evt.is_set():
            try:
                gz      = self._mpu.gyro[2]           # rad/s
                ay      = self._mpu.acceleration[1]   # m/s^2
                yaw_dps = math.degrees(gz - self._gz_bias_rad)
                with self._lock:
                    self._acc.add(yaw_dps, ay)
            except Exception as exc:
                self.read_errors += 1
                if self.read_errors % 100 == 1:       # don't flood the log
                    log.warning("IMU read error #%d: %s", self.read_errors, exc)

            # Deadline pacing: no cumulative drift, and wait() stays responsive to stop()
            next_t += interval
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                self._stop_evt.wait(sleep_t)
            else:
                next_t = time.perf_counter()          # fell behind; resync