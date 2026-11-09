"""Sensor collection: the IMU and wheel encoders read together, grouped per camera frame.

Purpose:
    The sensors run at 100 Hz against the camera's ~20 FPS, so readings are
    collected between frames on the camera's clock and handed to each frame
    as one group, yaw already + = right. Drivers load only when asked for,
    so replays and tests never import them.

Main package:
    SensorBatch: one frame window's readings; SensorSample.from_batch() makes
    it Phase 3's input.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

from src.os_threads import name_os_thread, set_realtime
from src.params import IMU_YAW_SIGN, SENSOR_HISTORY_S, SENSOR_HUB_RT_PRIORITY, SENSOR_RATE_HZ

log = logging.getLogger(__name__)

# A reading stands for the time since the previous one, up to this. A longer
# gap lost readings, and the turn in them isn't guessed at
MAX_YAW_GAP_S = 0.1
# Of the nominal period: room for the chip's clock, not for a stall's lost time
FIFO_STEP = (0.8, 1.2)


@dataclass(frozen=True)
class SensorReading:
    """A field is None when its sensor isn't there, or its read failed or wasn't taken."""
    t: float                        # time.monotonic() seconds
    yaw_dps: float | None           # + = turning right
    lateral_accel: float | None     # m/s^2
    left_count: int | None          # cumulative, + = forward
    right_count: int | None
    imu_ms: float | None = None     # the read's time, failed or not


@dataclass(frozen=True)
class SensorBatch:
    """
    Oldest first. base is the previous window's last reading, which the wheel speeds are
    measured from. dropped counts readings lost to a full buffer.
    """
    readings: tuple[SensorReading, ...]
    base: SensorReading | None = None
    dropped: int = 0
    imu_base_t: float | None = None

    @property
    def span_s(self) -> float:
        if not self.readings:
            return 0.0
        start = self.base if self.base is not None else self.readings[0]
        return self.readings[-1].t - start.t

    @property
    def imu_count(self) -> int:
        return sum(r.yaw_dps is not None for r in self.readings)

    @property
    def imu_attempts(self) -> int:
        """SENSOR_RATE_HZ / FPS (5) while the hub keeps its ticks."""
        return sum(r.imu_ms is not None for r in self.readings)

    @property
    def imu_failed(self) -> int:
        return sum(r.imu_ms is not None and r.yaw_dps is None for r in self.readings)

    @property
    def imu_read_ms_max(self) -> float | None:
        return max((r.imu_ms for r in self.readings if r.imu_ms is not None), default=None)

    @property
    def mean_yaw_dps(self) -> float | None:
        yaws = [r.yaw_dps for r in self.readings if r.yaw_dps is not None]
        return sum(yaws) / len(yaws) if yaws else None

    def _yaw_integral(self) -> tuple[float, float] | None:
        deg = span = 0.0
        prev, seen = self.imu_base_t, False
        for r in self.readings:
            if r.yaw_dps is None:
                continue
            seen = True
            if prev is not None:
                gap = min(max(r.t - prev, 0.0), MAX_YAW_GAP_S)
                deg += r.yaw_dps * gap
                span += gap
            prev = r.t
        return (deg, span) if seen else None

    @property
    def yaw_deg(self) -> float | None:
        """
        Degrees turned, + = right: each reading's rate times the time since the previous one
        (at most MAX_YAW_GAP_S), however long the window. Raw; the bias is the caller's.
        """
        got = self._yaw_integral()
        return None if got is None else got[0]

    @property
    def yaw_span_s(self) -> float | None:
        """The seconds yaw_deg covers, which a bias is subtracted over."""
        got = self._yaw_integral()
        return None if got is None else got[1]

    @property
    def peak_lateral_accel(self) -> float | None:
        """The signed value with the largest magnitude."""
        accels = [r.lateral_accel for r in self.readings if r.lateral_accel is not None]
        return max(accels, key=abs) if accels else None

    def _last_counts(self) -> SensorReading | None:
        return next((r for r in reversed(self.readings) if r.left_count is not None), None)

    @property
    def left_count(self) -> int | None:
        last = self._last_counts()
        return None if last is None else last.left_count

    @property
    def right_count(self) -> int | None:
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
        """From base to the window's end; 0.0 stopped, None without encoders."""
        return self._cps("left_count")

    @property
    def right_cps(self) -> float | None:
        return self._cps("right_count")


@dataclass(frozen=True)
class SensorSample:
    """None means not available."""
    yaw_rate_dps: float | None = None          # mean over the window, + = right
    lateral_accel_mps2: float | None = None    # the largest |a|, signed
    left_wheel_cps: float | None = None
    right_wheel_cps: float | None = None
    left_count: int | None = None              # cumulative, at the window's end
    right_count: int | None = None
    # Raw; None falls back to yaw_rate_dps x the frame step, as replays do
    yaw_deg: float | None = None
    yaw_span_s: float | None = None

    @classmethod
    def from_batch(cls, batch) -> "SensorSample":
        """Duck-typed on the batch's fields, so test fakes need only the ones they set."""
        return cls(
            yaw_rate_dps = batch.mean_yaw_dps,
            lateral_accel_mps2 = batch.peak_lateral_accel,
            left_wheel_cps = batch.left_cps,
            right_wheel_cps = batch.right_cps,
            left_count = getattr(batch, "left_count", None),
            right_count = getattr(batch, "right_count", None),
            yaw_deg = getattr(batch, "yaw_deg", None),
            yaw_span_s = getattr(batch, "yaw_span_s", None),
        )


class SensorHub:
    """
    Reads the IMU and encoders together on one thread. An IMU with fifo True is read per
    drain instead. yaw_sign maps the IMU's yaw to + = right. rt_priority None stays at normal
    priority; open() asks for SENSOR_HUB_RT_PRIORITY. clock is time.monotonic's, as the camera's.
    """
    def __init__(
            self,
            imu=None,
            encoders=None,
            rate_hz: float = SENSOR_RATE_HZ,
            yaw_sign: int = IMU_YAW_SIGN,
            history_s: float = SENSOR_HISTORY_S,
            rt_priority: int | None = None,
            clock=time.monotonic,
            timer=time.perf_counter,
        ) -> None:
        self._imu, self._encoders = imu, encoders
        self._rt_priority = rt_priority
        self.priority = "not started"            # the thread's: "SCHED_FIFO 20", "normal" or "normal (why)"
        self._rate_hz, self._yaw_sign, self._clock, self._timer = rate_hz, yaw_sign, clock, timer
        self._buf: deque[SensorReading] = deque(maxlen=max(1, int(rate_hz * history_s)))
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._base: SensorReading | None = None
        self._imu_t: float | None = None
        self._dropped = 0
        self._closers: list = []
        self.read_errors = 0
        self._fifo = bool(getattr(imu, "fifo", False))
        self._fifo_t: float | None = None       # the last FIFO sample's stamp

    @classmethod
    def open(cls, imu: bool = False, encoders: bool = False, **kw) -> "SensorHub":
        """
        The real sensors, each only when asked for. The IMU isn't calibrated here, so the bias
        is Phase3Config's. Encoders need the pigpio daemon. stop() releases them.
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
        kw.setdefault("rt_priority", SENSOR_HUB_RT_PRIORITY)
        hub = cls(imu_reader, enc_reader, **kw)
        hub._closers = closers
        return hub

    @property
    def has_sensors(self) -> bool:
        return self._imu is not None or self._encoders is not None

    def start(self) -> None:
        """Takes the first window's starting reading; a no-op if running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._base = self._counts_reading()
        if self._fifo:
            self._start_fifo()
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="sensor-hub")
        self._thread.start()
        log.info("SensorHub started at %.0f Hz (imu=%s, encoders=%s)",
                 self._rate_hz, self._imu is not None, self._encoders is not None)

    def stop(self, timeout: float = 0.5) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        for close in self._closers:
            close()
        self._closers = []
        log.info("SensorHub stopped (%d IMU read errors).", self.read_errors)

    def tick(self) -> SensorReading:
        """One full reading, kept. A failed IMU read leaves its fields None and is counted, never raised."""
        yaw = accel = imu_ms = None
        if self._imu is not None:
            r0 = self._timer()
            try:
                raw_yaw, accel = self._imu.read()
                yaw = self._yaw_sign * raw_yaw
            except Exception as exc:
                self.read_errors += 1
                if self.read_errors % 100 == 1:       # don't flood the log
                    log.warning("IMU read error #%d: %s", self.read_errors, exc)
            imu_ms = round((self._timer() - r0) * 1000.0, 3)
        left, right = self._counts()
        reading = SensorReading(self._clock(), yaw, accel, left, right, imu_ms)
        self._keep(reading)
        return reading

    def _start_fifo(self) -> None:
        """Empty what built up since it opened, so the first window starts now."""
        try:
            self._imu.fifo_reset()
        except Exception as exc:
            log.warning("IMU FIFO reset failed: %s", exc)
        self._fifo_t = None
        self.priority = "no thread (the IMU's FIFO)"
        log.info("SensorHub started on the IMU's FIFO (%.0f Hz on the chip's clock, encoders=%s)",
                 1.0 / self._imu.period_s, self._encoders is not None)

    @property
    def fifo_overflows(self) -> int | None:
        return getattr(self._imu, "fifo_overflows", None) if self._fifo else None

    def _drain_fifo(self) -> None:
        """
        Every FIFO sample into the buffer, spread over the time since the last drain, newest
        stamped now, so yaw integrates over real time even if the chip's clock runs a few percent
        off. The step stays within FIFO_STEP of nominal, so lost time isn't spread over survivors.
        """
        now, r0 = self._clock(), self._timer()
        try:
            samples = self._imu.read_fifo()
        except Exception as exc:
            self.read_errors += 1
            if self.read_errors % 100 == 1:
                log.warning("IMU FIFO read error #%d: %s", self.read_errors, exc)
            self._keep(SensorReading(now, None, None, None, None, round((self._timer() - r0) * 1000.0, 3)))
            return
        ms = round((self._timer() - r0) * 1000.0, 3)
        if not samples:
            return
        n, nominal = len(samples), self._imu.period_s
        if self._fifo_t is None:
            step, first = nominal, now - (n - 1) * nominal
        else:
            lo, hi = FIFO_STEP
            step = min(max((now - self._fifo_t) / n, lo * nominal), hi * nominal)
            first = self._fifo_t + step
        for i, (raw_yaw, accel) in enumerate(samples):
            self._keep(SensorReading(first + i * step, self._yaw_sign * raw_yaw, accel, None, None, ms))
        self._fifo_t = first + (n - 1) * step

    def drain(self) -> SensorBatch:
        """
        Every reading since the last drain, closed with an encoder reading taken now (the IMU
        isn't read, so the frame loop never waits on I2C). The last reading becomes the next base.
        """
        if self._fifo:
            self._drain_fifo()
        if self._encoders is not None:
            self._keep(self._counts_reading())
        with self._lock:
            readings, dropped = tuple(self._buf), self._dropped
            self._buf.clear()
            self._dropped = 0
        batch = SensorBatch(readings, self._base, dropped, self._imu_t)
        if readings:
            self._base = readings[-1]
        self._imu_t = next((r.t for r in reversed(readings) if r.yaw_dps is not None), self._imu_t)
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
        name_os_thread("sensor-hub")
        self._take_priority()
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


    def _take_priority(self) -> None:
        """Says once what it got."""
        if self._rt_priority is None:
            self.priority = "normal"
            return
        why = set_realtime(self._rt_priority)
        self.priority = f"SCHED_FIFO {self._rt_priority}" if not why else f"normal ({why})"
        (log.info if not why else log.warning)("SensorHub thread: %s", self.priority)


class Sensors:
    """
    A frame loop's started SensorHub, each sensor opened only when asked for; with
    neither, frames run without sensors. hub replaces imu and encoders (tests).
    """
    def __init__(self, imu: bool = False, encoders: bool = False, hub: SensorHub | None = None) -> None:
        self._hub = hub if hub is not None else SensorHub.open(imu=imu, encoders=encoders)
        if self._hub.has_sensors:
            self._hub.start()

    def read(self) -> tuple[SensorSample | None, SensorBatch | None]:
        """(SensorSample, the SensorBatch it came from), or (None, None) without sensors."""
        if not self._hub.has_sensors:
            return None, None
        batch = self._hub.drain()
        return SensorSample.from_batch(batch), batch

    @property
    def priority(self) -> str:
        return self._hub.priority

    @property
    def fifo_overflows(self) -> int | None:
        return self._hub.fifo_overflows

    def sample(self) -> SensorSample | None:
        return self.read()[0]

    def stop(self) -> None:
        self._hub.stop()
