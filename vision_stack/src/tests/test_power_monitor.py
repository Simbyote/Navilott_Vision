"""Battery state machine: debounce, hysteresis and the CRITICAL latch. No hardware.

Purpose:
    The warning must ignore a brief sag under motor load, not flicker while
    the voltage hovers at the threshold, and never un-latch CRITICAL
    mid-run. Readings are fed straight in, so no ADC is needed.
"""

import pytest

from src.diagnostics.battery_state import BatteryState as S, BatteryStateMachine


def run(volts, **kw):
    sm = BatteryStateMachine(**kw)
    return [sm.update(v) for v in volts], sm


@pytest.mark.software
def test_brief_sag_ignored():
    # 2 low samples (< confirm=3) then recovery: no state change
    ev, sm = run([11.0, 10.4, 10.4, 11.0, 11.0])
    assert sm.state == S.OK and all(e is None for e in ev)


@pytest.mark.software
def test_sustained_low_warns_once():
    ev, sm = run([11.0, 10.4, 10.4, 10.4, 10.4, 10.4])
    assert sm.state == S.WARNING
    assert [e for e in ev if e is not None] == [S.WARNING]


@pytest.mark.software
def test_hysteresis_no_flicker():
    # Hovering at 10.6 (above warn, below warn+0.3) must not clear WARNING
    ev, sm = run([10.4] * 3 + [10.6] * 10)
    assert sm.state == S.WARNING


@pytest.mark.software
def test_recovers_above_hysteresis():
    ev, sm = run([10.4] * 3 + [10.9] * 3)
    assert sm.state == S.OK
    assert [e for e in ev if e is not None] == [S.WARNING, S.OK]


@pytest.mark.software
def test_critical_latches():
    ev, sm = run([10.4] * 3 + [9.8] * 3 + [12.0] * 10)
    assert sm.state == S.CRITICAL
    assert [e for e in ev if e is not None] == [S.WARNING, S.CRITICAL]


@pytest.mark.software
def test_direct_to_critical():
    ev, sm = run([9.5] * 3)
    assert sm.state == S.CRITICAL
