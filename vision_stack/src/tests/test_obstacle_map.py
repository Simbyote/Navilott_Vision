"""
test_obstacle_map.py  --  src/analysis/obstacle_map.py

Three made-up obstacle-check runs: a tall box seen near and too short far
off, a short one, and nothing at all. The grid holds each gap's result, the
saved looks replayed through Phase 2 give the tallest blob and its gate (a
box under the opening's 47 px leaves nothing; 47-60 px is too short), and
--no-replay leaves the gates out.

--software  Synthetic frames through the real chain (SCENE_CONFIG).
"""
import pytest

from src.analysis import obstacle_map as om
from src.phase2_linker import run_chain
from src.tests import run_fixtures as rf
from src.tests.scenes import SCENE_CONFIG

pytestmark = pytest.mark.software


def chain(frame):
    return run_chain(frame, 0, 0, SCENE_CONFIG, parallel=False)


def runs(tmp_path):
    rf.obstacle_run(tmp_path, "tall_box", (100, 90, 54, 0, 0), (True, True, False, False, False))
    rf.obstacle_run(tmp_path, "short_box", (54, 30, 0, 0, 0), (False,) * 5, alarms=1)
    rf.obstacle_run(tmp_path, "nothing", (0,) * 5, (False,) * 5)
    return om.run_folders([str(tmp_path)])


def test_the_grid_the_replayed_gates_and_the_heights(tmp_path):
    res = om.analyze(runs(tmp_path), chain=chain)
    by = {o["object"]: o for o in res["objects"]}
    assert set(by) == {"tall box", "short box", "nothing"}
    tall = by["tall box"]["cells"]
    assert tall[10.0]["seen"] and tall[10.0]["reported_cm"] == pytest.approx(10.8)
    assert tall[20.0]["gate"] == "short" and 0 < tall[20.0]["blob_h"] < 60, "under the height gate"
    assert by["short box"]["cells"][10.0]["gate"] == "short"
    assert by["short box"]["cells"][15.0]["gate"] is None, "30 px: the 47 px opening leaves nothing"
    assert by["nothing"]["cells"][10.0]["gate"] is None and by["nothing"]["cells"][10.0]["blob_h"] == 0
    assert res["farthest_seen_cm"] == 15.0 and res["false_alarms"] == 1
    assert set(res["never_seen"]) == {"short box", "nothing"}
    assert "too short" in " ".join(res["findings"])


def test_no_replay_and_main(tmp_path, capsys):
    folders = runs(tmp_path)
    res = om.analyze(folders, do_replay=False)
    assert all(c["gate"] == "" for o in res["objects"] for c in o["cells"].values())
    assert om.main([str(tmp_path), "--no-replay"]) == 0
    assert (tmp_path / "obstacle_map.json").is_file()
    assert "tall box" in capsys.readouterr().out
    assert om.main([str(tmp_path / "none")]) == 1
