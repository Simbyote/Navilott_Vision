"""
test_load_compare.py  --  src/analysis/load_compare.py

Two made-up diagnostics recordings, task0 at 35% before and 6% after. Each
run is cut to its busy stretch, the threads compare, the busy cores drop,
and a window given by hand overrides the stretch.

--software  Synthetic CSVs.
"""
import pytest

from src.analysis import load_compare as lc
from src.tests import run_fixtures as rf

pytestmark = pytest.mark.software


def test_the_busy_stretch_and_the_threads_that_moved(tmp_path):
    before = lc.Run(rf.diag(tmp_path / "diag_b", task0=35.0))
    after = lc.Run(rf.diag(tmp_path / "diag_a", task0=6.0, seed=2))
    assert (before.t0, before.t1) == (5.0, 54.5)
    res = lc.compare(before, after)
    t = {x["name"]: x for x in res["threads"]}
    assert t["task0"]["before"] == pytest.approx(35, abs=2) and t["task0"]["after"] == pytest.approx(6, abs=2)
    assert res["after"]["busy_cores"] < res["before"]["busy_cores"]
    text = " ".join(res["findings"])
    assert "task0:" in text and "busy cores" in text and "main" not in text


def test_a_window_by_hand_and_main(tmp_path, capsys):
    b, a = rf.diag(tmp_path / "diag_b"), rf.diag(tmp_path / "diag_a", task0=6.0)
    assert lc.Run(b, lc.window_of("10-20")).t0 == 10.0
    assert lc.main([str(b), str(a)]) == 0
    assert (a / "load_compare.json").is_file()
    assert "task0" in capsys.readouterr().out
