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
    SensorReading: one reading: time, yaw rate, lateral accel, both counts,
        and how long the IMU read took (imu_ms: I2C plus the GIL, as the
        thread lived it; diagnostics.i2c_trace has the bus's side).
    SensorBatch: one frame window's readings, what Phase 3 needs from them
        (the yaw integrated reading by reading, so a long window loses
        nothing), and the window's IMU reads: attempted, failed, the slowest.
    SensorHub: the sampling thread, its buffer, and drain(). Over the real
        sensors (open()) the thread asks for real-time priority
        (SENSOR_HUB_RT_PRIORITY); priority says what it got. With an IMU
        sampling into its own FIFO (IMUReader.fifo, params.IMU_FIFO) there
        is no thread: drain() reads the FIFO, each sample stamped by the
        chip's own period.
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

from src.diagnostics.threads import name_os_thread, set_realtime
from src.estimation.estimation import SensorSample
from src.params import IMU_YAW_SIGN, SENSOR_HISTORY_S, SENSOR_HUB_RT_PRIORITY, SENSOR_RATE_HZ

log = logging.getLogger(__name__)

# An IMU reading stands for the time since the previous one, up to this long.
# A longer gap means readings were lost (I2C errors, or a buffer that filled
# during a stall): the turn made in them can't be recovered, so it isn't
# guessed at either
MAX_YAW_GAP_S = 0.1


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
    imu_ms: float | None = None     # how long the IMU read took, failed or not; None when not read


@dataclass(frozen=True)
class SensorBatch:
    """
    Every reading in one frame window, oldest first.

    base: The previous window's last reading, where this window's wheel
        speeds are measured from. None only if the hub had no reading to
        start from.
    dropped: Readings lost to a full buffer since the previous drain.
    imu_base_t: When the last IMU reading before this window was taken,
        where its first IMU reading's interval starts. None before any.
    """
    readings: tuple[SensorReading, ...]
    base: SensorReading | None = None
    dropped: int = 0
    imu_base_t: float | None = None

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
    def imu_attempts(self) -> int:
        """IMU reads tried in the window: SENSOR_RATE_HZ / FPS (5) when the hub keeps its ticks."""
        return sum(r.imu_ms is not None for r in self.readings)

    @property
    def imu_failed(self) -> int:
        """IMU reads tried that gave nothing (an I2C error)."""
        return sum(r.imu_ms is not None and r.yaw_dps is None for r in self.readings)

    @property
    def imu_read_ms_max(self) -> float | None:
        """The slowest IMU read in the window, ms; None without any."""
        return max((r.imu_ms for r in self.readings if r.imu_ms is not None), default=None)

    @property
    def mean_yaw_dps(self) -> float | None:
        """Mean yaw rate over the window's IMU readings; None without any."""
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
        Degrees turned over the window, + = right: each IMU reading's rate
        times the time since the previous IMU reading (at most
        MAX_YAW_GAP_S), so the window is integrated however long it is.
        Raw: the gyro bias is the caller's, over yaw_span_s. None without
        an IMU reading.
        """
        got = self._yaw_integral()
        return None if got is None else got[0]

    @property
    def yaw_span_s(self) -> float | None:
        """Seconds yaw_deg covers (what a gyro bias is subtracted over); None without an IMU reading."""
        got = self._yaw_integral()
        return None if got is None else got[1]

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
            like peripherals.imu.IMUReader; None without an IMU. One with
            fifo True also has read_fifo(), fifo_reset(), period_s and
            fifo_overflows, and the hub reads it per drain instead.
        encoders: An object with counts() -> (left, right), like
            peripherals.drive.EncoderReader; None without encoders.
        rate_hz: Readings per second.
        yaw_sign: Multiplies the IMU's yaw into + = turning right.
        history_s: Seconds of readings kept between drains; older ones are
            dropped and counted.
        rt_priority: SCHED_FIFO priority the thread asks for; None (the
            default, for fakes and tests) stays at normal priority. open()
            passes SENSOR_HUB_RT_PRIORITY.
        clock: Seconds, time.monotonic's clock (the camera stamps with
            time.monotonic_ns, so batches line up with frames).
        timer: Seconds, for timing each IMU read (time.perf_counter).
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
        A hub over the robot's real sensors, each opened only when asked for.

        Inputs:
            imu: Open the MPU-6050 (IMU_I2C_ADDRESS). It isn't calibrated
                here, so its gyro bias is left to Phase3Config.gyro_bias_dps.
            encoders: Open the encoders; needs the pigpio daemon (sudo pigpiod).
            kw: SensorHub's other arguments; rt_priority defaults to
                SENSOR_HUB_RT_PRIORITY here.
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
        kw.setdefault("rt_priority", SENSOR_HUB_RT_PRIORITY)
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
        if self._fifo:
            self._start_fifo()
            return
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
            counted in read_errors, never raised. Either way imu_ms holds
            how long the read took.
        """
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
        """No thread: empty the FIFO of what built up since it opened, so the first window starts now."""
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
        """Times the IMU's FIFO filled and was reset (samples lost); None without one."""
        return getattr(self._imu, "fifo_overflows", None) if self._fifo else None

    def _drain_fifo(self) -> None:
        """
        Every sample in the IMU's FIFO into the buffer, newest stamped now and
        each earlier one a period before; never before the previous drain's
        last, so the windows tile. A failed read keeps one reading with no
        yaw (counted as failed); its samples wait in the FIFO for the next.
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
        dt = self._imu.period_s
        start = now - (len(samples) - 1) * dt
        if self._fifo_t is not None:
            start = max(start, self._fifo_t + dt)
        for i, (raw_yaw, accel) in enumerate(samples):
            self._keep(SensorReading(start + i * dt, self._yaw_sign * raw_yaw, accel, None, None, ms))
        self._fifo_t = start + (len(samples) - 1) * dt

    def drain(self) -> SensorBatch:
        """
        Every reading since the previous drain, as one batch.

        The window is closed with an encoder reading taken now (the IMU
        isn't read, so the frame loop never waits on I2C), so the wheel
        speeds run right up to this frame.

        Side effects:
            Empties the buffer; the batch's last reading becomes the next
            batch's base. With the IMU's FIFO, reads it first.
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
        """tick() every period until stop(), on deadline pacing like the IMU's own thread."""
        name_os_thread("sensor-hub")            # top, ps and diagnostics.monitor show it by name
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
        """Ask for SCHED_FIFO for this thread; say once what it got."""
        if self._rt_priority is None:
            self.priority = "normal"
            return
        why = set_realtime(self._rt_priority)
        self.priority = f"SCHED_FIFO {self._rt_priority}" if not why else f"normal ({why})"
        (log.info if not why else log.warning)("SensorHub thread: %s", self.priority)


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

    @property
    def priority(self) -> str:
        """The hub thread's scheduling, as SensorHub.priority says it."""
        return self._hub.priority

    @property
    def fifo_overflows(self) -> int | None:
        """The IMU FIFO's overflows, as SensorHub.fifo_overflows says; None without one."""
        return self._hub.fifo_overflows

    def sample(self) -> SensorSample | None:
        """This frame window's readings; None with no sensors."""
        return self.read()[0]

    def stop(self) -> None:
        """Stop the hub and release its sensors."""
        self._hub.stop()
