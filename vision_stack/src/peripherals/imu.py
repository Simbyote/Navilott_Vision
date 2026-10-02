"""MPU-6050 gyro Z and lateral accel: one reading on request, for the sensor hub.

Purpose:
    The IMU driver and nothing above it. sensing.SensorHub calls read() at
    its own rate (100 Hz) on its own thread, together with the encoders,
    and groups the readings per camera frame; this module never decides
    when to sample. calibrate() measures the gyro's zero-rate bias with the
    robot still (the IMU hardware test's bias figure).

Main package:
    IMUReader: read() -> (gyro Z deg/s, accel Y m/s^2) in the driver's
        frame (Z up: + = turning LEFT; this robot's mount is upside down,
        which sensing's IMU_YAW_SIGN accounts for), calibrate().

Flow:
    1. IMUReader() opens I2C and sets the on-chip low-pass filter.
    2. Optionally calibrate(), robot still: later reads subtract the bias.
    3. read(), as often as the caller samples.
"""

import math
import time
import logging

from src.params import IMU_I2C_ADDRESS, IMU_RATE_HZ

log = logging.getLogger(__name__)


class IMUReader:
    """
    Owns the MPU-6050: one reading per read() call.

    Inputs:
        address: I2C address; 0x68 with AD0 low, 0x69 with it high.
        rate_hz: calibrate()'s sampling rate.
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
        self._rate_hz     = rate_hz
        self._gz_bias_rad = 0.0

    def calibrate(self, samples: int = 200) -> float:
        """
        Estimate the gyro-Z zero-rate bias, with the robot stationary.

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

    def read(self) -> tuple[float, float]:
        """
        One reading, now, in the driver's frame: (bias-corrected gyro Z in
        deg/s, + = left for a Z-up mount; accel Y in m/s^2). The caller
        paces the sampling (src/peripherals/sensing.py's SensorHub).

        Raises:
            Whatever the driver raises on a failed I2C read.
        """
        gz = self._mpu.gyro[2]                  # rad/s
        ay = self._mpu.acceleration[1]          # m/s^2
        return math.degrees(gz - self._gz_bias_rad), ay
