"""
test_battery.py  --  src/diagnostics/battery.py and src/diagnostics/battery_run.py

Power on a fake ADC channel (the pin voltage it reads, scripted): the 4:1
divider, preflight on a good, low and unreadable pack, the sampling thread
taking it to WARNING and a latched CRITICAL with each callback once, a
sensor fault after FAULT_AFTER failed reads that doesn't stop the robot, the
percentage, cleanup without an I2C bus, and every monitoring read timed
(a failed one too) in a bounded history. Then battery_run: a missing ADC
is said and gives None, and preflight's verdicts (refuse only when critical
with the motors on).

--software  A fake channel; no ADS1115, I2C or Pi libraries.
"""
import time

import pytest

from src.diagnostics import battery_run as br
from src.diagnostics.battery import Power
from src.diagnostics.battery_state import BatteryState

pytestmark = pytest.mark.software


class Channel:
    """AnalogIn stand-in: .voltage is the pin's volts (pack / 4); raises while fail is set."""
    def __init__(self, pack_v=12.0):
        self.pack_v, self.fail = pack_v, False

    @property
    def voltage(self):
        if self.fail:
            raise OSError("ADC read failed")
        return self.pack_v / Power.DIVIDER_RATIO


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def pwr():
    ch = Channel()
    p = Power(channel=ch)
    yield p, ch
    p.cleanup()


def test_the_pack_voltage_is_the_pin_times_the_divider(pwr):
    p, ch = pwr
    ch.pack_v = 11.6
    assert p.voltage_raw() == pytest.approx(11.6)
    assert Power.DIVIDER_RATIO == 4.0


def test_the_thresholds_are_the_3s_packs():
    # 3.5 V and 3.3 V a cell: warn, then end the run after CONFIRM_SAMPLES readings
    assert (Power.VOLTAGE_FULL, Power.VOLTAGE_NOMINAL) == (12.6, 11.1)
    assert (Power.VOLTAGE_WARNING, Power.VOLTAGE_CRITICAL, Power.VOLTAGE_EMPTY) == (10.5, 9.9, 9.0)
    assert (Power.HYSTERESIS_V, Power.CONFIRM_SAMPLES, Power._EMA_ALPHA) == (0.3, 3, 0.3)


@pytest.mark.parametrize("pack_v, ok", [(12.2, True), (10.4, False)])
def test_preflight_averages_resting_reads_against_the_warning_threshold(pwr, pack_v, ok):
    p, ch = pwr
    ch.pack_v = pack_v
    assert p.preflight(samples=3, delay_s=0.0) == (ok, pytest.approx(pack_v))


def test_preflight_with_an_unreadable_adc_is_not_ok_and_zero(pwr):
    p, ch = pwr
    ch.fail = True
    assert p.preflight(samples=2, delay_s=0.0) == (False, 0.0)


def test_monitoring_goes_to_warning_then_latches_critical_firing_each_callback_once(pwr):
    p, ch = pwr
    fired = []
    p.on_warning(lambda: fired.append("warning"))
    p.on_critical(lambda: fired.append("critical"))
    ch.pack_v = 11.8
    p.start_monitoring(interval_s=0.001)
    assert p.state() == BatteryState.OK and not p.should_stop()
    ch.pack_v = 10.2
    assert wait_for(lambda: p.state() == BatteryState.WARNING)
    assert p.is_warning() and not p.should_stop()
    ch.pack_v = 9.0
    assert wait_for(p.should_stop)
    ch.pack_v = 12.6                                     # a rested pack mid-run doesn't un-latch it
    time.sleep(0.05)
    assert p.should_stop() and fired == ["warning", "critical"]


def test_a_sensor_fault_is_reported_once_and_does_not_stop_the_robot(pwr):
    p, ch = pwr
    faults = []
    p.on_fault(lambda: faults.append(1))
    p.start_monitoring(interval_s=0.001)
    ch.fail = True
    assert wait_for(lambda: not p.sensor_ok())
    time.sleep(0.03)
    assert faults == [1] and p.is_warning() and not p.should_stop()
    ch.fail = False
    assert wait_for(p.sensor_ok)


@pytest.mark.parametrize("pack_v, pct", [(12.6, 100.0), (13.0, 100.0), (9.0, 0.0), (10.8, 50.0)])
def test_percentage_is_linear_from_empty_to_full(pack_v, pct):
    p = Power(channel=Channel(pack_v))
    p._voltage_ema = pack_v
    assert p.percentage() == pytest.approx(pct)


def test_cleanup_without_a_bus_and_twice_is_fine(pwr):
    p, _ = pwr
    p.start_monitoring(interval_s=0.001)
    p.cleanup()
    p.cleanup()


def test_a_missing_adc_is_said_and_gives_none():
    said = []

    def no_adc():
        raise ImportError("No module named 'board'")
    assert br.open_battery(said.append, factory=no_adc) is None
    assert said and "not monitored" in said[0]
    assert isinstance(br.open_battery(said.append, factory=lambda: "power"), str)


@pytest.mark.parametrize("pack_v, motors_on, verdict", [
    (12.0, True, br.GO), (10.3, True, br.LOW), (9.6, True, br.REFUSE),
    (9.6, False, br.LOW),                                # a dry run: nothing moves, so carry on
])
def test_preflight_refuses_only_a_critical_pack_with_the_motors_on(monkeypatch, pack_v, motors_on, verdict):
    p = Power(channel=Channel(pack_v))
    monkeypatch.setattr(p, "preflight", lambda: (pack_v > p.VOLTAGE_WARNING, pack_v))
    said = []
    assert br.preflight(p, motors_on, said.append) == (verdict, pack_v)
    assert f"{pack_v:.2f} V" in said[0]


def test_an_unreadable_adc_at_preflight_goes_on_unmonitored(monkeypatch):
    p = Power(channel=Channel())
    monkeypatch.setattr(p, "preflight", lambda: (False, 0.0))
    said = []
    assert br.preflight(p, True, said.append) == (br.GO, None) and "unreadable" in said[0]


def test_start_says_and_reports_a_first_read_that_fails():
    ch = Channel()
    ch.fail = True
    p = Power(channel=ch)
    said = []
    assert br.start(p, said.append) is False and "didn't start" in said[0]
    ch.fail = False
    assert br.start(p, said.append) is True
    p.cleanup()


class Timer:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SlowChannel(Channel):
    """A channel whose reads take read_ms on timer: the ADS1115's single-shot conversion."""
    def __init__(self, timer, read_ms, pack_v=12.0):
        super().__init__(pack_v)
        self.timer, self.read_ms = timer, list(read_ms)

    @property
    def voltage(self):
        self.timer.now += (self.read_ms.pop(0) if self.read_ms else 1.0) / 1000.0
        return super().voltage


def loop(monkeypatch, p, n):
    """Run the monitoring loop for n reads on this thread (time.sleep ends it after the nth)."""
    left = [n]

    def sleep(_s):
        left[0] -= 1
        if left[0] == 0:
            p._running = False
    monkeypatch.setattr(time, "sleep", sleep)
    p._running = True
    p._sample_loop(0.0)


@pytest.mark.software
def test_each_monitoring_read_is_timed_failed_ones_too(monkeypatch):
    timer = Timer()
    ch = SlowChannel(timer, [9.0, 8.5, 9.25])
    p = Power(channel=ch, timer=timer)
    p._voltage_ema = 12.0
    assert p.read_ms() == []
    loop(monkeypatch, p, 2)
    ch.fail = True                                                  # a read that times out still took its time
    loop(monkeypatch, p, 1)
    assert p.read_ms() == [9.0, 8.5, 9.25]


@pytest.mark.software
def test_the_read_history_keeps_only_the_newest(monkeypatch):
    monkeypatch.setattr(Power, "READ_HISTORY", 3)
    timer = Timer()
    p = Power(channel=SlowChannel(timer, [1.0, 2.0, 3.0, 4.0, 5.0]), timer=timer)
    p._voltage_ema = 12.0
    loop(monkeypatch, p, 5)
    assert p.read_ms() == [3.0, 4.0, 5.0]


from src.diagnostics import battery as bat  # noqa: E402


class ADS1115Chip:
    """
    An ADS1115 on a busio-like bus: one pointer register shared by every
    client. A write sets the pointer (and, with data, the register); a read
    returns the register the pointer names. Logs each transfer.
    """
    def __init__(self, pin_v=3.0):
        self.regs = {bat.REG_CONVERSION: 0, bat.REG_CONFIG: 0x8583}     # the chip's power-on config
        self.pointer = bat.REG_CONVERSION
        self.pin_v = pin_v
        self.transfers = []
        self.locked = False

    def try_lock(self):
        if self.locked:
            return False
        self.locked = True
        return True

    def unlock(self):
        self.locked = False

    def _convert(self):
        if self.regs[bat.REG_CONFIG] & 0x0100 == 0:                     # continuous: always a fresh result
            self.regs[bat.REG_CONVERSION] = round(self.pin_v / bat.FULL_SCALE_V * 32768) & 0xFFFF

    def writeto(self, addr, data):
        self.transfers.append(("write", bytes(data)))
        self.pointer = data[0]
        if len(data) == 3:
            self.regs[self.pointer] = data[1] << 8 | data[2]

    def readfrom_into(self, addr, buf):
        """A read with no pointer write: the Adafruit library's continuous-mode "fast" read."""
        self.transfers.append(("read", None))
        self._convert()
        v = self.regs[self.pointer]
        buf[0], buf[1] = v >> 8, v & 0xFF

    def writeto_then_readfrom(self, addr, out, buf):
        self.transfers.append(("combined", bytes(out)))
        self.pointer = out[0]
        self._convert()
        v = self.regs[self.pointer]
        buf[0], buf[1] = v >> 8, v & 0xFF


def adc(chip):
    waits = []
    a = bat.ContinuousADC(chip, sleep=waits.append)
    return a, waits


def test_opening_sets_continuous_a0_and_waits_two_conversions():
    chip = ADS1115Chip()
    _, waits = adc(chip)
    assert chip.regs[bat.REG_CONFIG] == bat.ADC_CONFIG and chip.regs[bat.REG_CONFIG] & 0x0100 == 0
    assert waits == [pytest.approx(2 / 128)]


def test_a_read_is_the_pin_voltage_from_two_combined_transfers():
    chip = ADS1115Chip(pin_v=3.0)
    a, _ = adc(chip)
    chip.transfers.clear()
    assert a.voltage == pytest.approx(3.0, abs=1e-3)
    assert [t[0] for t in chip.transfers] == ["combined", "combined"]           # config, then the result
    assert Power(channel=a)._read_battery_voltage() == pytest.approx(12.0, abs=4e-3)


def test_another_client_moving_the_pointer_does_not_change_the_reading():
    # The Adafruit library's fast read trusts the pointer: after another client
    # touches the config register, it reads the config as if it were a voltage
    chip = ADS1115Chip(pin_v=3.0)
    a, _ = adc(chip)
    chip.writeto(0x48, bytes([bat.REG_CONFIG]))                                 # someone else's pointer write
    buf = bytearray(2)
    chip.readfrom_into(0x48, buf)
    fast = (buf[0] << 8 | buf[1]) * bat.FULL_SCALE_V / 32768
    assert abs(fast - 3.0) > 0.1
    chip.writeto(0x48, bytes([bat.REG_CONFIG]))
    assert a.voltage == pytest.approx(3.0, abs=1e-3) and a.reconfigured == 0


def test_a_config_someone_else_changed_is_rewritten_before_reading():
    chip = ADS1115Chip(pin_v=3.0)
    a, waits = adc(chip)
    chip.regs[bat.REG_CONFIG] = 0x8583                                          # single-shot, A0-A1, +-2.048 V
    chip.regs[bat.REG_CONVERSION] = 0x1234                                      # its stale result
    assert a.voltage == pytest.approx(3.0, abs=1e-3)
    assert a.reconfigured == 1 and chip.regs[bat.REG_CONFIG] == bat.ADC_CONFIG and len(waits) == 2


def test_a_negative_result_reads_negative():
    chip = ADS1115Chip(pin_v=-0.5)
    a, _ = adc(chip)
    assert a.voltage == pytest.approx(-0.5, abs=1e-3)


def test_the_power_monitor_opens_the_continuous_adc_when_enabled(monkeypatch):
    import sys
    import types
    chip = ADS1115Chip(pin_v=3.0)
    chip.deinit = lambda: None
    monkeypatch.setitem(sys.modules, "board", types.SimpleNamespace(SCL=3, SDA=2))
    monkeypatch.setitem(sys.modules, "busio", types.SimpleNamespace(I2C=lambda scl, sda: chip))
    monkeypatch.setattr(bat, "ADS1115_CONTINUOUS", True)
    p = Power()
    assert isinstance(p._chan, bat.ContinuousADC)
    assert p._read_battery_voltage() == pytest.approx(12.0, abs=4e-3)
    p.cleanup()
