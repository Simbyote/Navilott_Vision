"""Sensor collection: the IMU and wheel encoders read together, grouped per camera frame.

Purpose:
    The sensors run faster than the camera (100 Hz against ~20 FPS), so a
    background thread reads the IMU and both encoder counts together at
    SENSOR_RATE_HZ, stamps each reading on the camera's clock
    (time.monotonic), and keeps them until the frame loop drains them. Each
    frame then gets every reading taken since the previous frame as one
    SensorBatch, and SensorSample.from_batch() turns that into Phase 3's input.
    Yaw is put into Estimation's convention (+ = turning right) here, once,
    with IMU_YAW_SIGN, so nothing downstream knows how the IMU is mounted.

    This module never imports the board drivers or pigpio. SensorHub takes
    any objects with the drivers' read() / counts(); SensorHub.open() imports
    the real ones only when asked for, so replays and tests never load them.

Main package:
    SensorReading: one reading: time, yaw rate, lateral accel, both counts.
    SensorBatch: one frame window's readings, and what Phase 3 needs from them.
    SensorHub: the sampling thread, its buffer, and drain().
    Sensors: what a frame loop uses (src/main.py and every linker): opens and
        starts a hub, then one read() / sample() per frame.

Flow:
    1. Sensors(imu=..., encoders=...): SensorHub.open() and start().
       (Or by hand: SensorHub.open() / SensorHub(imu, encoders), start().)
    2. Once per frame, sample() (or read() for the batch too): the hub's
       drain() closes the window with a fresh encoder reading and hands over
       every reading since the previous one, as a SensorSample.
    3. stop(): end the thread and release whatever open() opened.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from src.estimation.estimation import SensorSample
from src.params import IMU_YAW_SIGN, SENSOR_HISTORY_S, SENSOR_RATE_HZ

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorReading:
    """
    One reading. A field is None when its sensor isn't there, or when the
    read failed (IMU) or wasn't taken (the IMU in a drain's closing reading).
    """
    t: float                        # time.monotonic() seconds
    yaw_dps: float | None           # gyro Z, deg/s, + = turning right (IMU_YAW_SIGN applied)
    lateral_accel: float | None     # accel Y, m/s^2, as the driver reports it
    left_count: int | None          # cumulative since the encoders' reset(), + = forward
    right_count: int | None


@dataclass(frozen=True)
class SensorBatch:
    """
    Every reading in one frame window, oldest first.

    base: The previous window's last reading, where this window's wheel
        speeds are measured from. None only if the hub had no reading to
        start from.
    dropped: Readings lost to a full buffer since the previous drain.
    """
    readings: tuple[SensorReading, ...]
    base: SensorReading | None = None
    dropped: int = 0

    @property
    def span_s(self) -> float:
        """Seconds from base (or the first reading) to the last reading; 0 when empty."""
        if not self.readings:
            return 0.0
        start = self.base if self.base is not None else self.readings[0]
        return self.readings[-1].t - start.t

    @property
    def imu_count(self) -> int:
        """Readings that carry an IMU value."""
        return sum(r.yaw_dps is not None for r in self.readings)

    @property
    def mean_yaw_dps(self) -> float | None:
        """Mean yaw rate over the window's IMU readings; None without any."""
        yaws = [r.yaw_dps for r in self.readings if r.yaw_dps is not None]
        return sum(yaws) / len(yaws) if yaws else None

    @property
    def peak_lateral_accel(self) -> float | None:
        """The signed lateral accel with the largest magnitude; None without any."""
        accels = [r.lateral_accel for r in self.readings if r.lateral_accel is not None]
        return max(accels, key=abs) if accels else None

    def _last_counts(self) -> SensorReading | None:
        return next((r for r in reversed(self.readings) if r.left_count is not None), None)

    @property
    def left_count(self) -> int | None:
        """Cumulative left count at the window's end; None without encoders."""
        last = self._last_counts()
        return None if last is None else last.left_count

    @property
    def right_count(self) -> int | None:
        """Cumulative right count at the window's end; None without encoders."""
        last = self._last_counts()
        return None if last is None else last.right_count

    def _cps(self, side: str) -> float | None:
        last = self._last_counts()
        if last is None:
            return None
        start = self.base if self.base is not None and self.base.left_count is not None else None
        if start is None or last.t <= start.t:
            return 0.0
        return (getattr(last, side) - getattr(start, side)) / (last.t - start.t)

    @property
    def left_cps(self) -> float | None:
        """Left counts per second from base to the window's end; 0.0 stopped, None without encoders."""
        return self._cps("left_count")

    @property
    def right_cps(self) -> float | None:
        """Right counts per second from base to the window's end; 0.0 stopped, None without encoders."""
        return self._cps("right_count")


class SensorHub:
    """
    Reads the IMU and encoders together on one background thread and hands
    each frame its readings.

    Inputs:
        imu: An object with read() -> (raw yaw deg/s, lateral accel m/s^2),
            like peripherals.imu.IMUReader; None without an IMU.
        encoders: An object with counts() -> (left, right), like
            peripherals.drive.EncoderReader; None without encoders.
        rate_hz: Readings per second.
        yaw_sign: Multiplies the IMU's yaw into + = turning right.
        history_s: Seconds of readings kept between drains; older ones are
            dropped and counted.
        clock: Seconds, time.monotonic's clock (the camera stamps with
            time.monotonic_ns, so batches line up with frames).
    """
    def __init__(
            self,
            imu=None,
            encoders=None,
            rate_hz: float = SENSOR_RATE_HZ,
            yaw_sign: int = IMU_YAW_SIGN,
            history_s: float = SENSOR_HISTORY_S,
            clock=time.monotonic,
        ) -> None:
        self._imu, self._encoders = imu, encoders
        self._rate_hz, self._yaw_sign, self._clock = rate_hz, yaw_sign, clock
        self._buf: deque[SensorReading] = deque(maxlen=max(1, int(rate_hz * history_s)))
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._base: SensorReading | None = None
        self._dropped = 0
        self._closers: list = []
        self.read_errors = 0

    @classmethod
    def open(cls, imu: bool = False, encoders: bool = False, **kw) -> "SensorHub":
        """
        A hub over the robot's real sensors, each opened only when asked for.

        Inputs:
            imu: Open the MPU-6050 (IMU_I2C_ADDRESS). It isn't calibrated
                here, so its gyro bias is left to Phase3Config.gyro_bias_dps.
            encoders: Open the encoders; needs the pigpio daemon (sudo pigpiod).
            kw: SensorHub's other arguments.
        Side effects:
            Opens I2C and / or a pigpio connection; stop() releases them.
        """
        imu_reader = enc_reader = None
        closers = []
        if imu:
            from src.peripherals.imu import IMUReader
            imu_reader = IMUReader()
        if encoders:
            import pigpio
            from src.peripherals.drive import EncoderReader
            pi = pigpio.pi()
            enc_reader = EncoderReader(pi)
            closers += [enc_reader.cancel, pi.stop]
        hub = cls(imu_reader, enc_reader, **kw)
        hub._closers = closers
        return hub

    @property
    def has_sensors(self) -> bool:
        return self._imu is not None or self._encoders is not None

    def start(self) -> None:
        """Take the first window's starting reading and start sampling; a no-op if running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._base = self._counts_reading()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="sensor-hub")
        self._thread.start()
        log.info("SensorHub started at %.0f Hz (imu=%s, encoders=%s)",
                 self._rate_hz, self._imu is not None, self._encoders is not None)

    def stop(self, timeout: float = 0.5) -> None:
        """Stop the thread, then release whatever open() opened."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        for close in self._closers:
            close()
        self._closers = []
        log.info("SensorHub stopped (%d IMU read errors).", self.read_errors)

    def tick(self) -> SensorReading:
        """
        Take one full reading now and keep it. The thread calls this every
        period; tests call it directly.

        Outputs:
            The reading. A failed IMU read leaves its fields None and is
            counted in read_errors, never raised.
        """
        yaw = accel = None
        if self._imu is not None:
            try:
                raw_yaw, accel = self._imu.read()
                yaw = self._yaw_sign * raw_yaw
            except Exception as exc:
                self.read_errors += 1
                if self.read_errors % 100 == 1:       # don't flood the log
                    log.warning("IMU read error #%d: %s", self.read_errors, exc)
        left, right = self._counts()
        reading = SensorReading(self._clock(), yaw, accel, left, right)
        self._keep(reading)
        return reading

    def drain(self) -> SensorBatch:
        """
        Every reading since the previous drain, as one batch.

        The window is closed with an encoder reading taken now (the IMU
        isn't read, so the frame loop never waits on I2C), so the wheel
        speeds run right up to this frame.

        Side effects:
            Empties the buffer; the batch's last reading becomes the next
            batch's base.
        """
        if self._encoders is not None:
            self._keep(self._counts_reading())
        with self._lock:
            readings, dropped = tuple(self._buf), self._dropped
            self._buf.clear()
            self._dropped = 0
        batch = SensorBatch(readings, self._base, dropped)
        if readings:
            self._base = readings[-1]
        return batch

    def _counts(self) -> tuple[int | None, int | None]:
        return (None, None) if self._encoders is None else self._encoders.counts()

    def _counts_reading(self) -> SensorReading:
        return SensorReading(self._clock(), None, None, *self._counts())

    def _keep(self, reading: SensorReading) -> None:
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(reading)

    def _worker(self) -> None:
        """tick() every period until stop(), on deadline pacing like the IMU's own thread."""
        interval = 1.0 / self._rate_hz
        next_t = time.perf_counter()
        while not self._stop_evt.is_set():
            self.tick()
            next_t += interval
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                self._stop_evt.wait(sleep_t)
            else:
                next_t = time.perf_counter()          # fell behind; resync


class Sensors:
    """
    A frame loop's sensors: a started SensorHub over the IMU and wheel
    encoders, each opened only when asked for, so replays never load the
    board drivers or pigpio. With neither, every frame runs without sensors.

    The IMU uses IMU_I2C_ADDRESS from params and isn't calibrated here, so
    any gyro bias correction comes from Phase3Config.gyro_bias_dps, in the
    hub's frame (+ = turning right). The encoders need the pigpio daemon
    (sudo pigpiod).

    Inputs:
        imu, encoders: Which sensors SensorHub.open() opens.
        hub: A hub to use instead (tests, or one built by hand); imu and
            encoders are then ignored. Started here if it has sensors.
    """
    def __init__(self, imu: bool = False, encoders: bool = False, hub: SensorHub | None = None) -> None:
        self._hub = hub if hub is not None else SensorHub.open(imu=imu, encoders=encoders)
        if self._hub.has_sensors:
            self._hub.start()

    def read(self) -> tuple[SensorSample | None, SensorBatch | None]:
        """
        This frame window's readings.

        Outputs:
            (sample, batch): Phase 3's SensorSample and the SensorBatch it came
            from, which also carries the cumulative counts the sample leaves
            out. (None, None) with no sensors.
        """
        if not self._hub.has_sensors:
            return None, None
        batch = self._hub.drain()
        return SensorSample.from_batch(batch), batch

    def sample(self) -> SensorSample | None:
        """This frame window's readings; None with no sensors."""
        return self.read()[0]

    def stop(self) -> None:
        """Stop the hub and release its sensors."""
        self._hub.stop()
