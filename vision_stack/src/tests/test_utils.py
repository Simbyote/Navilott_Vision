"""
test_utils.py  --  src/utils.py

--software  Known-answer tests for the shared helpers. No camera.
"""
import pytest

import time

import numpy as np

from src.utils import Laps, check_frame_size, clamp


@pytest.mark.software
@pytest.mark.parametrize("v, lo, hi, want", [(5, 0, 10, 5), (-3, 0, 10, 0), (99, 0, 10, 10), (0, 0, 10, 0), (10, 0, 10, 10)])
def test_clamp(v, lo, hi, want):
    assert clamp(v, lo, hi) == want


@pytest.mark.software
def test_check_frame_size_passes_the_size_and_any_size_without_one():
    frame = np.zeros((48, 64, 3), np.uint8)
    check_frame_size(frame, (48, 64), "t")
    check_frame_size(frame, None, "t")


@pytest.mark.software
def test_check_frame_size_refuses_another_size_naming_the_stage():
    with pytest.raises(ValueError, match=r"Stage: frame is \(48, 64\).*\(480, 640\)"):
        check_frame_size(np.zeros((48, 64), np.uint8), (480, 640), "Stage")


@pytest.mark.software
def test_laps_time_each_stage_since_the_last_and_start_fresh():
    laps = Laps()
    assert laps.times == {}
    laps.start()
    time.sleep(0.02)
    laps.lap("a")
    laps.lap("b")
    first = laps.times
    assert list(first) == ["a", "b"] and first["a"] >= 15.0 and 0.0 <= first["b"] < first["a"]
    laps.start()
    assert laps.times == {} and first == {"a": first["a"], "b": first["b"]}   # a fresh dict; the old one kept


@pytest.mark.software
def test_laps_mark_restarts_the_clock_without_recording():
    laps = Laps()
    laps.start()
    time.sleep(0.02)
    laps.mark()
    laps.lap("after_mark")
    assert list(laps.times) == ["after_mark"] and laps.times["after_mark"] < 15.0


@pytest.mark.software
def test_gc_watch_freezes_and_times_collections():
    import gc
    from src.utils import GcWatch
    watch = GcWatch().start()
    try:
        assert gc.get_freeze_count() > 0
        assert watch.take() == (0.0, -1)
        gc.collect(2)
        ms, gen = watch.take()
        assert ms >= 0.0 and gen == 2                   # frozen, the collection itself can take ~0 ms
        assert watch.take() == (0.0, -1)
    finally:
        watch.stop()
        gc.unfreeze()
    assert watch._callback not in gc.callbacks


@pytest.mark.software
def test_gc_watch_can_turn_collection_off_and_collect_the_young_itself(monkeypatch):
    import gc
    from src.utils import GcWatch
    collected = []
    real = gc.collect
    monkeypatch.setattr(gc, "collect", lambda gen=2: collected.append(gen) or real(gen))
    watch = GcWatch(disable=True, collect_every=3).start()
    try:
        assert not gc.isenabled()
        collected.clear()
        for braked in (False, False, False, True, False):
            watch.tick(braked)
        assert collected == [0, 0]                       # the 3rd frame, then the braked one
    finally:
        watch.stop()
        gc.unfreeze()
    assert gc.isenabled()
