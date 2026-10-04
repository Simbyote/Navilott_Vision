"""MPU-6050 gyro Z and lateral accel: one reading on request, for the sensor hub.

Purpose:
    The IMU driver and nothing above it. sensing.SensorHub calls read() at
    its own rate (100 Hz) on its own thread, together with the encoders,
    and groups the readings per camera frame; this module never decides
    when to sample. calibrate() measures the gyro's zero-rate bias with the
    robot still (the IMU hardware test's bias figure).

    Each reading is one I2C transfer: the 14-byte sample block from
    ACCEL_XOUT_H, scaled with the ranges read once at open. The Adafruit
    driver (adafruit_mpu6050 1.3.9) fetches each axis in its own transfer
    and .gyro re-reads the range register: 7 transfers for the two values
    (strace on the robot, 2026-10-03: 7.3 per tick), 4.6 ms per read at
    100 kHz, and the sensor hub's thread giving up and retaking the GIL for
    every one of them.

Main package:
    IMUReader: read() -> (gyro Z deg/s, accel Y m/s^2) in the driver's
        frame (Z up: + = turning LEFT; this robot's mount is upside down,
        which sensing's IMU_YAW_SIGN accounts for), calibrate().

Flow:
    1. IMUReader() opens I2C, sets the on-chip low-pass filter and reads
       the gyro and accel ranges.
    2. Optionally calibrate(), robot still: later reads subtract the bias.
    3. read(), as often as the caller samples.
"""

import struct
import time
import logging

from src.params import IMU_I2C_ADDRESS, IMU_RATE_HZ

log = logging.getLogger(__name__)

# MPU-6050 register map (rev 4.2, section 4.17-4.19): ACCEL_XOUT_H starts
# the sample block, 7 big-endian int16s: accel X Y Z, temperature, gyro X Y Z
SAMPLE_REG = 0x3B
SAMPLE_FORMAT = ">7h"
SAMPLE_BYTES = struct.calcsize(SAMPLE_FORMAT)       # 14
ACCEL_Y, GYRO_Z = 1, 6                              # indices into the unpacked block
# Sensitivity per full-scale setting (datasheet section 6.1-6.2): FS_SEL 0-3
# is +-250/500/1000/2000 deg/s, AFS_SEL 0-3 is +-2/4/8/16 g; the Adafruit
# driver's GyroRange / Range values are the same 0-3
GYRO_LSB_PER_DPS = (131.0, 65.5, 32.8, 16.4)
ACCEL_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0)
STANDARD_GRAVITY = 9.80665                          # m/s^2 per g, as the Adafruit driver converts


class IMUReader:
    """
    Owns the MPU-6050: one reading per read() call.

    Inputs:
        address: I2C address; 0x68 with AD0 low, 0x69 with it high.
        rate_hz: calibrate()'s sampling rate.
        mpu: An Adafruit MPU6050, or a stand-in with its .i2c_device,
            .gyro_range and .accelerometer_range, to test without
            hardware. None opens the real sensor.

    Side effects:
        With mpu None, opens I2C and sets the sensor's on-chip low-pass
        filter. Reads the two range settings either way.
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

        self._mpu         = mpu                 # the driver itself, kept for the hardware test's comparison
        self._dev         = mpu.i2c_device
        self._gyro_lsb    = GYRO_LSB_PER_DPS[int(mpu.gyro_range)]
        self._accel_lsb   = ACCEL_LSB_PER_G[int(mpu.accelerometer_range)]
        self._reg         = bytes((SAMPLE_REG,))
        self._buf         = bytearray(SAMPLE_BYTES)
        self._rate_hz     = rate_hz
        self._gz_bias_dps = 0.0

    def _sample(self) -> tuple[float, float]:
        """One I2C transfer: (raw gyro Z in deg/s, accel Y in m/s^2), before the bias."""
        with self._dev as dev:
            dev.write_then_readinto(self._reg, self._buf)
        block = struct.unpack(SAMPLE_FORMAT, self._buf)
        return (block[GYRO_Z] / self._gyro_lsb,
                block[ACCEL_Y] / self._accel_lsb * STANDARD_GRAVITY)

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
                total += self._sample()[0]
                n     += 1
            except Exception as exc:
                log.debug("IMU calib read error (skipped): %s", exc)
            time.sleep(interval)

        if n == 0:
            raise RuntimeError("IMU calibration failed: no successful reads")

        self._gz_bias_dps = total / n
        log.info("IMU gyro-Z bias: %.3f deg/s (%d samples)", self._gz_bias_dps, n)
        return self._gz_bias_dps

    def read(self) -> tuple[float, float]:
        """
        One reading, now, in the driver's frame: (bias-corrected gyro Z in
        deg/s, + = left for a Z-up mount; accel Y in m/s^2). The caller
        paces the sampling (src/peripherals/sensing.py's SensorHub).

        Raises:
            Whatever the driver raises on a failed I2C read.
        """
        gz_dps, ay = self._sample()
        return gz_dps - self._gz_bias_dps, ay
