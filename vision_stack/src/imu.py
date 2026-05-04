"""
imu.py

IMU background accumulator for the Navilott pipeline.

Wraps the MPU-6050 via the Adafruit driver. Runs a daemon thread that
samples at ~100 Hz and accumulates into a per-frame buffer. The main
loop calls IMUReader.snapshot() once per frame to drain the buffer and
get aggregated values.

Axis convention (flat-mount, Z up):
    gyro Z   -> yaw_rate     (deg/s,  + = turning right)
    accel Y  -> lateral_accel (m/s², + = accelerating left)
"""

import math
import time
import threading
import logging

import board
import busio
import adafruit_mpu6050

log = logging.getLogger(__name__)


# =============================================================================
# Output container
# =============================================================================

from dataclasses import dataclass, field

@dataclass
class IMUFrame:
    """
    Aggregated IMU values for one pipeline frame window.

    Fields
    ------
    mean_yaw_rate_dps   : mean gyro-Z over the frame interval (deg/s).
                          None if no samples were collected.
    peak_lateral_accel  : highest-magnitude accel-Y sample in the frame
                          interval (m/s²). None if no samples were collected.
    sample_count        : number of raw IMU reads that went into this frame.
    valid               : False when sample_count == 0; Phase 3 should treat
                          both float fields as unusable when False.
    """
    mean_yaw_rate_dps  : float | None = None
    peak_lateral_accel : float | None = None
    sample_count       : int          = 0
    valid              : bool         = False


# =============================================================================
# Reader
# =============================================================================

class IMUReader:
    """
    Initializes the MPU-6050 and manages a background sampling thread.

    Usage
    -----
        reader = IMUReader()
        reader.start()

        # inside frame loop:
        imu_frame = reader.snapshot()

        # on shutdown:
        reader.stop()
    """

    def __init__(
            self,
            address   : int   = 0x68,
            rate_hz   : float = 100.0,
    ) -> None:
        i2c       = busio.I2C(board.SCL, board.SDA)
        self._mpu = adafruit_mpu6050.MPU6050(i2c, address=address)

        self._rate_hz  = rate_hz
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()
        self._yaw_buf  : list[float] = []
        self._accel_buf: list[float] = []
        self._thread   : threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread."""
        self._thread = threading.Thread(
            target  = self._worker,
            daemon  = True,
            name    = "imu-accumulator",
        )
        self._thread.start()
        log.info("IMUReader started at %.0f Hz on 0x%02X", self._rate_hz, 0x68)

    def stop(self, timeout: float = 0.5) -> None:
        """Signal the worker to stop and wait for it to exit."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("IMUReader stopped.")

    # ------------------------------------------------------------------
    # Frame interface
    # ------------------------------------------------------------------

    def snapshot(self) -> IMUFrame:
        """
        Atomically drain the accumulation buffers.

        Returns an IMUFrame covering all samples collected since the
        previous call. Returns an invalid IMUFrame if no samples arrived.
        """
        with self._lock:
            yaw_buf   = self._yaw_buf.copy()
            accel_buf = self._accel_buf.copy()
            self._yaw_buf.clear()
            self._accel_buf.clear()

        n = len(yaw_buf)
        if n == 0:
            return IMUFrame()  # valid=False, all None

        return IMUFrame(
            mean_yaw_rate_dps  = sum(yaw_buf) / n,
            peak_lateral_accel = max(accel_buf, key=abs),
            sample_count       = n,
            valid              = True,
        )

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        interval = 1.0 / self._rate_hz
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                gx, gy, gz = self._mpu.gyro          # rad/s
                ax, ay, az = self._mpu.acceleration   # m/s²
                yaw_dps    = gz * (180.0 / math.pi)
                with self._lock:
                    self._yaw_buf.append(yaw_dps)
                    self._accel_buf.append(ay)
            except Exception as exc:
                log.debug("IMU read error (skipped): %s", exc)

            sleep_t = interval - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)