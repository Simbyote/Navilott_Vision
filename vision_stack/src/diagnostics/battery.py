"""Battery monitor: pack voltage through the V-Sense divider and the ADS1115, with a latched warning.

Purpose:
    A 3S pack has no protection board (DEC-019), so the software has to
    notice a flat battery. The voltage is sampled on a daemon thread,
    smoothed, and debounced by BatteryStateMachine, so a brief sag under
    motor load doesn't raise a false alarm; callbacks fire once per
    transition. How the ADS1115 is read, and why: docs/architecture.md, "Power".

Main package:
    Power: the monitor. preflight() judges the pack at rest before a run;
    during it, state(), should_stop() (CRITICAL, latched) and the
    on_warning / on_critical / on_fault callbacks.

Flow:
    1. Power() opens the ADS1115 (ContinuousADC, or the library's single shots).
    2. preflight(), motors idle; then start_monitoring().
    3. Once a second: read, smooth, the state machine, any callbacks.
    4. cleanup() stops the thread and releases the bus.
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, List, Tuple

from src.diagnostics.battery_state import BatteryState, BatteryStateMachine
from src.params import ADS1115_CONTINUOUS, ADS1115_I2C_ADDRESS

log = logging.getLogger(__name__)

# ADS1115 registers (datasheet SBAS444, 8.6): the pointer byte picks one
REG_CONVERSION, REG_CONFIG = 0x00, 0x01
# Config: A0 against GND (MUX 100), +-4.096 V full scale (PGA 001; the pin
# reads 12.6 / 4 = 3.15 V at most), continuous (MODE 0), 128 SPS (DR 100),
# comparator off (COMP_QUE 11). OS, bit 15, reads as status, so it's masked
ADC_CONFIG = (0b100 << 12) | (0b001 << 9) | (0 << 8) | (0b100 << 5) | 0b11
CONFIG_MASK = 0x7FFF
FULL_SCALE_V = 4.096
DATA_RATE_SPS = 128


class ContinuousADC:
    """
    The ADS1115 converting A0 on its own; .voltage is the pin's volts from
    the latest conversion. Same interface as the Adafruit AnalogIn.

    Inputs:
        i2c: A busio.I2C (or anything with try_lock / unlock / writeto /
            writeto_then_readfrom).
        address: The ADS1115's.
        sleep: For the wait after (re)configuring, swapped in tests.

    reconfigured: Times a read found the config changed and rewrote it.
    """
    def __init__(self, i2c, address: int = ADS1115_I2C_ADDRESS, sleep=time.sleep) -> None:
        self._i2c, self._addr, self._sleep = i2c, address, sleep
        self._buf = bytearray(2)
        self.reconfigured = 0
        self._configure()

    def _locked(self, fn):
        while not self._i2c.try_lock():
            pass
        try:
            return fn()
        finally:
            self._i2c.unlock()

    def _register(self, reg: int) -> int:
        """One register, the pointer set in the same transfer as the read."""
        self._locked(lambda: self._i2c.writeto_then_readfrom(self._addr, bytes([reg]), self._buf))
        return self._buf[0] << 8 | self._buf[1]

    def _configure(self) -> None:
        """Write the config, then wait two conversions so the result is this config's."""
        out = bytes([REG_CONFIG, ADC_CONFIG >> 8, ADC_CONFIG & 0xFF])
        self._locked(lambda: self._i2c.writeto(self._addr, out))
        self._sleep(2.0 / DATA_RATE_SPS)

    @property
    def voltage(self) -> float:
        if self._register(REG_CONFIG) & CONFIG_MASK != ADC_CONFIG:
            self.reconfigured += 1
            log.warning("ADS1115 config changed under us; rewritten (%d times)", self.reconfigured)
            self._configure()
        raw = self._register(REG_CONVERSION)
        if raw & 0x8000:
            raw -= 1 << 16
        return raw * FULL_SCALE_V / 32768.0


class Power:
    """Battery health monitor.

    Reads battery voltage through a 4:1 voltage divider and ADS1115 ADC
    on the shared I2C bus. Runs a background thread that samples
    periodically and maintains a smoothed voltage estimate.
    """

    # Voltage divider: 30kΩ + 10kΩ → ratio = (30k + 10k) / 10k = 4.0
    DIVIDER_RATIO = 4.0

    # 3S 18650 thresholds (DEC-002)
    VOLTAGE_FULL = 12.6
    VOLTAGE_NOMINAL = 11.1
    VOLTAGE_WARNING = 10.5
    VOLTAGE_CRITICAL = 9.9
    VOLTAGE_EMPTY = 9.0

    # Exponential moving average weight (0–1, higher = more responsive)
    _EMA_ALPHA = 0.3

    # Debounce / hysteresis (see battery_state.py)
    HYSTERESIS_V = 0.3     # WARNING clears at VOLTAGE_WARNING + this
    CONFIRM_SAMPLES = 3    # consecutive samples needed to change state
    FAULT_AFTER = 3        # consecutive failed ADC reads before sensor fault
    READ_HISTORY = 3600    # monitoring read times kept: an hour at 1 Hz

    def __init__(self, channel=None, timer=time.perf_counter):
        """
        channel: anything with .voltage (volts at the ADC pin), to test
        without hardware; None opens the ADS1115 (the Pi's I2C libraries are
        imported only then, so this module imports anywhere).
        timer: seconds, for timing each monitoring read.
        """
        self._timer = timer
        self._read_ms = deque(maxlen=self.READ_HISTORY)
        self._i2c = None
        if channel is None:
            import board
            import busio

            self._i2c = busio.I2C(board.SCL, board.SDA)
            if ADS1115_CONTINUOUS:
                channel = ContinuousADC(self._i2c)
            else:
                from adafruit_ads1x15.ads1115 import ADS1115
                from adafruit_ads1x15.analog_in import AnalogIn
                ads = ADS1115(self._i2c, address=ADS1115_I2C_ADDRESS)
                channel = AnalogIn(ads, 0)  # A0 = V-Sense divider
        self._chan = channel

        self._voltage_ema = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._sm = BatteryStateMachine(
            self.VOLTAGE_WARNING,
            self.VOLTAGE_CRITICAL,
            self.HYSTERESIS_V,
            self.CONFIRM_SAMPLES,
        )
        self._failures = 0
        self._sensor_ok = True
        self._callbacks = {"warning": [], "critical": [], "recovered": [], "fault": []}

    # ── Callbacks (run on the sampling thread; keep them short) ───────

    def on_warning(self, cb: Callable[[], None]) -> None:
        """Called once when the battery enters WARNING."""
        self._callbacks["warning"].append(cb)

    def on_critical(self, cb: Callable[[], None]) -> None:
        """Called once when the battery enters CRITICAL (latched)."""
        self._callbacks["critical"].append(cb)

    def on_recovered(self, cb: Callable[[], None]) -> None:
        """Called when WARNING clears back to OK."""
        self._callbacks["recovered"].append(cb)

    def on_fault(self, cb: Callable[[], None]) -> None:
        """Called once when the ADC can't be read FAULT_AFTER times in a row."""
        self._callbacks["fault"].append(cb)

    def state(self) -> BatteryState:
        with self._lock:
            return self._sm.state

    def sensor_ok(self) -> bool:
        """False if the ADC has been unreadable (voltage() is stale)."""
        with self._lock:
            return self._sensor_ok

    def preflight(self, samples: int = 5, delay_s: float = 0.2) -> Tuple[bool, float]:
        """Resting-voltage check before a run. Returns (ok, volts).

        Averages a few raw reads; ok is False below the warning threshold
        or if the ADC can't be read. Call with motors idle.
        """
        readings = []
        for _ in range(samples):
            try:
                readings.append(self._read_battery_voltage())
            except Exception:
                log.exception("preflight ADC read failed")
            time.sleep(delay_s)
        if not readings:
            return False, 0.0
        v = sum(readings) / len(readings)
        return v > self.VOLTAGE_WARNING, v

    def start_monitoring(self, interval_s: float = 1.0) -> None:
        """Start background voltage sampling."""
        if self._running:
            return
        # Seed the EMA with a real reading
        self._voltage_ema = self._read_battery_voltage()
        self._running = True
        self._thread = threading.Thread(
            target=self._sample_loop, args=(interval_s,), daemon=True
        )
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop the background sampling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def voltage(self) -> float:
        """Current smoothed battery voltage in volts."""
        with self._lock:
            return self._voltage_ema

    def read_ms(self) -> list[float]:
        """How long each monitoring read took, ms, oldest first (failed ones too)."""
        with self._lock:
            return list(self._read_ms)

    def voltage_raw(self) -> float:
        """Single un-smoothed ADC reading (for characterization)."""
        return self._read_battery_voltage()

    def percentage(self) -> float:
        """Estimated remaining charge as 0–100%."""
        v = self.voltage()
        if v <= self.VOLTAGE_EMPTY:
            return 0.0
        if v >= self.VOLTAGE_FULL:
            return 100.0
        return (v - self.VOLTAGE_EMPTY) / (self.VOLTAGE_FULL - self.VOLTAGE_EMPTY) * 100.0

    def is_warning(self) -> bool:
        """True in WARNING or CRITICAL, or if the sensor has faulted."""
        with self._lock:
            return self._sm.state >= BatteryState.WARNING or not self._sensor_ok

    def should_stop(self) -> bool:
        """True once CRITICAL has latched. A sensor fault alone does not stop the bot."""
        with self._lock:
            return self._sm.state == BatteryState.CRITICAL

    def is_healthy(self) -> bool:
        """True if OK and the sensor is readable."""
        return not self.is_warning()

    def cleanup(self) -> None:
        """Stop monitoring thread, release I2C."""
        self.stop_monitoring()
        if self._i2c is not None:
            self._i2c.deinit()

    # ── Internal ───────────────────────────────────────────────────────

    def _read_battery_voltage(self) -> float:
        """Read ADC and convert to battery voltage."""
        return self._chan.voltage * self.DIVIDER_RATIO

    def _fire(self, name: str) -> None:
        for cb in self._callbacks[name]:
            try:
                cb()
            except Exception:
                log.exception("battery %s callback failed", name)

    def _sample_loop(self, interval_s: float) -> None:
        while self._running:
            events: List[str] = []
            r0 = self._timer()
            try:
                v = self._read_battery_voltage()
            except Exception:
                v = None
                self._failures += 1
                log.warning("ADC read failed (%d in a row)", self._failures)
            took_ms = round((self._timer() - r0) * 1000.0, 3)

            with self._lock:
                self._read_ms.append(took_ms)
                if v is None:
                    if self._failures == self.FAULT_AFTER and self._sensor_ok:
                        self._sensor_ok = False
                        events.append("fault")
                else:
                    self._failures = 0
                    self._sensor_ok = True
                    self._voltage_ema = (
                        self._EMA_ALPHA * v + (1 - self._EMA_ALPHA) * self._voltage_ema
                    )
                    changed = self._sm.update(self._voltage_ema)
                    if changed == BatteryState.WARNING:
                        events.append("warning")
                    elif changed == BatteryState.CRITICAL:
                        events.append("critical")
                    elif changed == BatteryState.OK:
                        events.append("recovered")

            for name in events:
                log.warning("battery event: %s (%.2fV)", name, self.voltage())
                self._fire(name)
            time.sleep(interval_s)
