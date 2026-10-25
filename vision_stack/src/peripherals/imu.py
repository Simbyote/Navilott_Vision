"""MPU-6050 gyro Z and lateral accel: one reading on request, or every sample its FIFO holds.

Purpose:
    The driver only; the sensor hub decides when to read. A reading is one
    I2C transfer of the sample block. With the FIFO on, the chip samples on
    its own clock and read_fifo() takes everything since the last call in two
    transfers (phase3_estimation.md, "Sensors").

Main package:
    IMUReader: (gyro Z deg/s, accel Y m/s^2) in the driver's frame, + = left.
"""

import struct
import time
import logging

from src.params import IMU_FIFO, IMU_I2C_ADDRESS, IMU_RATE_HZ

log = logging.getLogger(__name__)

# Register map rev 4.2: 7 big-endian int16s, accel X Y Z, temperature, gyro X Y Z
SAMPLE_REG = 0x3B
SAMPLE_FORMAT = ">7h"
SAMPLE_BYTES = struct.calcsize(SAMPLE_FORMAT)       # 14
ACCEL_Y, GYRO_Z = 1, 6                              # indices into the unpacked block
# Per range setting 0-3 (datasheet 6.1-6.2), as the Adafruit driver numbers them
GYRO_LSB_PER_DPS = (131.0, 65.5, 32.8, 16.4)
ACCEL_LSB_PER_G = (16384.0, 8192.0, 4096.0, 2048.0)
STANDARD_GRAVITY = 9.80665                          # m/s^2 per g, as the Adafruit driver converts

# Sample rate = 1 kHz / (1 + SMPLRT_DIV) with the low-pass filter on
SMPLRT_DIV, CONFIG, FIFO_EN, USER_CTRL = 0x19, 0x1A, 0x23, 0x6A
FIFO_COUNTH, FIFO_R_W = 0x72, 0x74
FIFO_EN_ZG_ACCEL = 0x10 | 0x08                      # gyro Z and the accel only
USER_CTRL_FIFO_EN, USER_CTRL_FIFO_RESET = 0x40, 0x04
FIFO_BYTES = 1024
# accel X Y Z, gyro Z
FIFO_FORMAT = ">4h"
FIFO_SAMPLE_BYTES = struct.calcsize(FIFO_FORMAT)    # 8
FIFO_ACCEL_Y, FIFO_GYRO_Z = 1, 3


class IMUReader:
    """
    mpu: a stand-in for tests; None opens the real sensor and sets its low-pass filter.
    fifo: None means IMU_FIFO on the real sensor; a chip that won't take it leaves it off, warned.
    """

    def __init__(
            self,
            address: int   = IMU_I2C_ADDRESS,
            rate_hz: float = IMU_RATE_HZ,
            mpu            = None,
            fifo: bool | None = None,
    ) -> None:
        opened = mpu is None
        if mpu is None:
            import board
            import busio
            import adafruit_mpu6050

            i2c = busio.I2C(board.SCL, board.SDA)
            mpu = adafruit_mpu6050.MPU6050(i2c, address=address)
            # Under the 50 Hz Nyquist limit of 100 Hz sampling
            mpu.filter_bandwidth = adafruit_mpu6050.Bandwidth.BAND_44_HZ

        self._mpu         = mpu                 # for the hardware test's comparison
        self._dev         = mpu.i2c_device
        self._gyro_lsb    = GYRO_LSB_PER_DPS[int(mpu.gyro_range)]
        self._accel_lsb   = ACCEL_LSB_PER_G[int(mpu.accelerometer_range)]
        self._reg         = bytes((SAMPLE_REG,))
        self._buf         = bytearray(SAMPLE_BYTES)
        self._rate_hz     = rate_hz
        self._gz_bias_dps = 0.0
        self.period_s       = 1.0 / rate_hz
        self.fifo_overflows = 0
        self._count_buf     = bytearray(2)
        want_fifo = (IMU_FIFO and opened) if fifo is None else fifo
        self.fifo = want_fifo and self._fifo_setup()

    def _write(self, reg: int, value: int) -> None:
        with self._dev as dev:
            dev.write(bytes((reg, value)))

    def _read_reg(self, reg: int) -> int:
        buf = bytearray(1)
        with self._dev as dev:
            dev.write_then_readinto(bytes((reg,)), buf)
        return buf[0]

    def _fifo_setup(self) -> bool:
        """False, warned, if the chip won't take it."""
        try:
            dlpf = self._read_reg(CONFIG) & 0x07
            gyro_hz = 8000.0 if dlpf in (0, 7) else 1000.0
            div = max(0, min(255, round(gyro_hz / self._rate_hz) - 1))
            self.period_s = (1 + div) / gyro_hz
            self._write(SMPLRT_DIV, div)
            self._write(FIFO_EN, FIFO_EN_ZG_ACCEL)
            self.fifo_reset()
        except Exception as exc:
            log.warning("IMU FIFO not set up (%s); the sensor hub polls instead", exc)
            return False
        return True

    def fifo_reset(self) -> None:
        """Empty the FIFO and keep it filling."""
        self._write(USER_CTRL, USER_CTRL_FIFO_RESET)
        self._write(USER_CTRL, USER_CTRL_FIFO_EN)

    def read_fifo(self) -> list[tuple[float, float]]:
        """
        Every whole sample since the last call, oldest first, period_s apart, bias-corrected.
        A full FIFO has overflowed: it's reset, counted in fifo_overflows, and [] returned.
        A failed read leaves the samples for the next call.
        """
        with self._dev as dev:
            dev.write_then_readinto(bytes((FIFO_COUNTH,)), self._count_buf)
        count = (self._count_buf[0] << 8) | self._count_buf[1]
        if count >= FIFO_BYTES:
            self.fifo_overflows += 1
            log.warning("IMU FIFO overflowed (%d so far): reset, %.2f s of samples lost",
                        self.fifo_overflows, FIFO_BYTES / FIFO_SAMPLE_BYTES * self.period_s)
            self.fifo_reset()
            return []
        n = count // FIFO_SAMPLE_BYTES
        if n == 0:
            return []
        buf = bytearray(n * FIFO_SAMPLE_BYTES)
        with self._dev as dev:
            dev.write_then_readinto(bytes((FIFO_R_W,)), buf)
        out = []
        for i in range(n):
            s = struct.unpack_from(FIFO_FORMAT, buf, i * FIFO_SAMPLE_BYTES)
            out.append((s[FIFO_GYRO_Z] / self._gyro_lsb - self._gz_bias_dps,
                        s[FIFO_ACCEL_Y] / self._accel_lsb * STANDARD_GRAVITY))
        return out

    def _sample(self) -> tuple[float, float]:
        """Raw, before the bias."""
        with self._dev as dev:
            dev.write_then_readinto(self._reg, self._buf)
        block = struct.unpack(SAMPLE_FORMAT, self._buf)
        return (block[GYRO_Z] / self._gyro_lsb,
                block[ACCEL_Y] / self._accel_lsb * STANDARD_GRAVITY)

    def calibrate(self, samples: int = 200) -> float:
        """
        Gyro-Z bias, robot still, subtracted from every later sample, so keep
        Phase3Config.gyro_bias_dps at 0 alongside it. RuntimeError if every read failed.
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
        """One bias-corrected reading now; the caller paces the sampling."""
        gz_dps, ay = self._sample()
        return gz_dps - self._gz_bias_dps, ay
