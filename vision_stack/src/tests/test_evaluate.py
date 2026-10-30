"""
test_evaluate.py  --  src/analysis/evaluate.py and src/scripts/label_frames.py

Labels: ranges of frame ids, checked (a bad value or two labels for one frame
names its row); a blank cell isn't scored. The replay runs synthetic frames
through the robot's Pipeline and scores the stop sign, the light and the
obstacle against them: a scene with the thing in it is found, one without
isn't, a wrong label shows as a miss or a false alarm, and the vote is
scored beside the frame. A routine's saved looks label themselves by name.
The labeller's keys turn into the same labels.csv the scorer reads.

--software  Synthetic scenes (src/tests/scenes.py). No camera.
"""
import json
from dataclasses import replace

import cv2
import pytest

from src.analysis import evaluate as ev
from src.config import MEASURED_ESTIMATION
from src.scripts import label_frames as lf
from src.tests.scenes import RED_LAMP, SCENE_CONFIG, scene, with_obstacle

pytestmark = pytest.mark.software

EST = MEASURED_ESTIMATION


def write_frames(folder, frames, start=0):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{start + i:06d}.png"), f)
    return folder


def labels(path, text):
    path.write_text(text)
    return path


def test_labels_are_checked_and_blank_cells_left_unscored(tmp_path):
    p = labels(tmp_path / "labels.csv", "# a comment\nfirst,last,sign,light,line_cm,obstacle_cm,offset_cm,range_cm,"
                                        "condition\n0,9,yes,red,none,12.5,,60,glare\n10,19,no,,,,,,\n")
    a, b = ev.read_labels(p)
    assert a == {"first": 0, "last": 9, "sign": True, "light": "red", "line_cm": "none", "obstacle_cm": 12.5,
                 "range_cm": 60.0, "condition": "glare"}
    assert b == {"first": 10, "last": 19, "sign": False}
    assert ev.truth_for([a, b], 12) is b and ev.truth_for([a, b], 20) is None
    for bad, words in (("0,9,maybe", "sign 'maybe'"), ("0,9,,blue", "light 'blue'"), ("5,2", "before first"),
                       ("0,9,,,far", "line_cm 'far'")):
        labels(p, "first,last,sign,light,line_cm\n" + bad + "\n")
        with pytest.raises(ValueError, match=words):
            ev.read_labels(p)
    labels(p, "first,last,sign\n0,9,yes\n5,12,no\n")
    with pytest.raises(ValueError, match="frames 5-9 are labelled twice"):
        ev.read_labels(p)


def test_the_wilson_interval_is_honest_at_the_edges():
    assert ev.wilson(0, 0) is None
    lo, hi = ev.wilson(10, 10)
    assert hi == 1.0 and 0.70 < lo < 0.75         # 10 of 10 isn't "100%"
    lo, hi = ev.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.2


def test_a_sign_and_a_light_scored_frame_by_frame_and_as_voted(tmp_path):
    run = tmp_path / "run"
    write_frames(run / "frames", [scene()] * 12 + [scene(sign=True, lights=(RED_LAMP,))] * 12)
    labels(run / "labels.csv", "first,last,sign,light,range_cm,condition\n0,11,no,off,,plain\n"
                               "12,23,yes,red,40,posted\n")
    res, what = ev.run_eval(run, config=SCENE_CONFIG, estimation=EST)
    assert res["labelled"] == 24 and "24 frames" in what
    s = res["sign"]["frame"]
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (12, 0, 0, 12)
    assert s["by_range"]["30-45"]["share"] == 1.0 and s["by_condition"]["plain"]["false_alarm"]["k"] == 0
    v = res["sign"]["voted"]
    # the vote's first 2 frames after the sign appears are its latency, not scored: 10 left, all right
    assert res["vote_settle_frames"] == 2 and (v["tp"], v["fn"], v["fp"], v["tn"]) == (10, 0, 0, 12)
    m = res["light"]["frame"]["matrix"]
    assert m["red"]["red"] == 12 and m["off"]["none"] == 12 and res["light"]["frame"]["accuracy"]["share"] == 1.0
    assert res["light"]["voted"]["would_run_it"] == {"k": 0, "n": 10, "share": 0.0, "ci95": ev.wilson(0, 10)}
    assert any("only 12 labelled frames" in f for f in res["findings"])


def test_a_wrong_label_shows_as_misses_and_false_alarms(tmp_path):
    run = tmp_path / "run"
    write_frames(run / "frames", [scene()] * 6 + [scene(sign=True)] * 6)
    labels(run / "labels.csv", "first,last,sign\n0,11,yes\n")         # sign "there" all the way
    s = ev.run_eval(run, config=SCENE_CONFIG, estimation=EST)[0]["sign"]["frame"]
    assert (s["tp"], s["fn"]) == (6, 6) and s["recall"]["share"] == 0.5
    labels(run / "labels.csv", "first,last,sign\n0,11,no\n")
    res = ev.run_eval(run, config=SCENE_CONFIG, estimation=EST)[0]
    assert res["sign"]["frame"]["fp"] == 6 and any("6 false alarms" in f for f in res["findings"])


def test_an_obstacle_found_where_it_is_and_not_where_it_isnt(tmp_path):
    run = tmp_path / "run"
    write_frames(run / "frames", [scene()] * 5 + [with_obstacle(scene(), 200, 280, 150, 250)] * 5)
    labels(run / "labels.csv", "first,last,obstacle_cm\n0,4,none\n5,9,15\n")
    o = ev.run_eval(run, config=SCENE_CONFIG, estimation=EST)[0]["obstacle"]["frame"]
    assert (o["tp"], o["fp"]) == (5, 0)
    assert o["no_cm"] == 5                        # SCENE_CONFIG has no table: found, but no cm to score
    assert o["error_cm"] == {"n": 0}


def test_distances_are_scored_against_the_tape():
    pairs = [(10.0, True, 11.0, {}), (10.0, True, 9.0, {}), (20.0, True, 26.0, {}), ("none", False, None, {}),
             ("none", True, 5.0, {}), (15.0, False, None, {})]
    s = ev.score_distance(pairs, 3.0)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (3, 1, 1, 1)
    assert s["error_cm"]["mean"] == pytest.approx(2.0) and s["error_cm"]["within_tol"]["k"] == 2
    assert s["error_by_distance"]["20-30"]["mean"] == 6.0


def test_lane_offset_without_a_scale_is_fitted_and_with_one_is_scored_in_cm():
    fit = ev.score_offset([(-4.0, -0.4, None), (0.0, 0.0, None), (4.0, 0.4, None), (2.0, None, None)])
    assert fit["seen"]["k"] == 3 and fit["fit"]["cm_per_unit"] == pytest.approx(10.0) and fit["units"] == "normalized"
    cm = ev.score_offset([(2.0, 0.2, 3.0), (-2.0, -0.2, -2.5)])
    assert cm["error_cm"]["mean"] == pytest.approx(0.25) and cm["units"] == "cm"


def test_a_detect_range_routine_labels_itself(tmp_path):
    run = tmp_path / "routine_detect-range_20261029_100000"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({"options": {"target": "sign"}}))
    cv2.imwrite(str(run / "attempt_01_10cm_present.jpg"), scene(sign=True))
    cv2.imwrite(str(run / "attempt_02_10cm_absent.jpg"), scene())
    cv2.imwrite(str(run / "attempt_03_30cm_5deg_present.jpg"), scene(sign=True))
    found = ev.labels_from_routine(run)
    assert [t for _, t in found] == [{"condition": "", "sign": True, "range_cm": 10.0},
                                     {"condition": "", "sign": False, "range_cm": 10.0},
                                     {"condition": "", "sign": True, "range_cm": 30.0}]
    res, what = ev.run_eval(run, config=SCENE_CONFIG, estimation=EST)
    assert "detect-range" in what and "voted" not in res["sign"]           # single looks: no vote to score
    assert res["sign"]["frame"]["recall"]["k"] == 2 and res["sign"]["frame"]["fp"] == 0


def test_an_obstacle_check_routine_labels_its_clear_lane_none(tmp_path):
    run = tmp_path / "routine_obstacle-check_bottle"
    run.mkdir()
    (run / "results.json").write_text(json.dumps({"options": {"object": "bottle"}}))
    cv2.imwrite(str(run / "attempt_01_15cm.jpg"), with_obstacle(scene(), 200, 280, 150, 250))
    cv2.imwrite(str(run / "attempt_01_15cm_3cm.jpg"), with_obstacle(scene(), 200, 280, 150, 250))
    cv2.imwrite(str(run / "attempt_01_clear.jpg"), scene())
    truths = [t for _, t in ev.labels_from_routine(run)]
    assert {(t["obstacle_cm"], t["condition"]) for t in truths} == {(15.0, "bottle"), ("none", "bottle")}


def test_the_command_line_writes_beside_the_run_and_compares_with_a_base(tmp_path, capsys, monkeypatch):
    run = tmp_path / "run"
    write_frames(run / "frames", [scene()] * 4 + [scene(sign=True)] * 4)
    labels(run / "labels.csv", "first,last,sign\n0,3,no\n4,7,yes\n")
    real = ev.replay
    monkeypatch.setattr(ev, "replay", lambda items, config=None, estimation=None, sequence=True:
                        real(items, SCENE_CONFIG, EST, sequence))
    assert ev.main([str(run)]) == 0
    first = json.loads((run / "eval.json").read_text())
    assert first["sign"]["frame"]["recall"]["share"] == 1.0 and "commit" in json.dumps(first["code"])
    if ev.common.pyplot() is not None:
        assert (run / "eval.png").is_file()
    base = tmp_path / "base.json"
    first["sign"]["frame"]["recall"]["share"] = 0.5
    base.write_text(json.dumps(first))
    assert ev.main([str(run), "--base", str(base)]) == 0
    assert "stop sign recall" in capsys.readouterr().out
    assert ev.main([str(tmp_path / "nothing")]) == 1
    (run / "labels.csv").unlink()
    assert ev.main([str(run)]) == 1


def test_the_labeller_s_keys_become_labels_csv_ranges(tmp_path):
    lab = lf.Labeller(list(range(10)), conditions=["glare"])
    for k in "s" + "l" + "c" + "ddd":                 # sign, light off, glare: frames 0-2
        lab.key(k)
    for k in "s" + "l" + "18L" + "60R" + "ddd":       # no sign, red, a line at 18 cm, range 60: frames 3-5
        lab.key(k)
    lab.key("L")                                      # nothing typed: the line's gone
    lab.key("d")
    assert lab.key("q") == "quit"
    rows = lab.rows()
    assert [(r["first"], r["last"]) for r in rows] == [(0, 2), (3, 5), (6, 6)]
    assert rows[1] == {"first": 3, "last": 5, "sign": "no", "light": "red", "line_cm": "18", "obstacle_cm": "",
                       "offset_cm": "", "range_cm": "60", "condition": "glare"}
    assert rows[2]["line_cm"] == "none"
    p = lf.write_labels(tmp_path / "labels.csv", rows)
    back = ev.read_labels(p)
    assert back[1]["line_cm"] == 18.0 and back[2]["line_cm"] == "none" and back[0]["sign"] is True
    again = lf.Labeller(list(range(10)), labels=lf.load_labels(p, list(range(10))))
    assert again.rows() == rows


def test_the_labeller_steps_back_without_painting_and_erases(tmp_path):
    lab = lf.Labeller([5, 6, 7, 8])
    lab.key("s")
    lab.key("D")                                       # 10 on: stops at the last, all painted
    assert sorted(lab.labels) == [5, 6, 7, 8] and lab.fid == 8
    lab.key("A")
    lab.key("e")
    assert lab.fid == 5 and sorted(lab.labels) == [6, 7, 8]
    assert lf.template(tmp_path / "t.csv", [5, 6, 7, 8]) and ev.read_labels(tmp_path / "t.csv") == [{"first": 5, "last": 8}]


def test_with_glow_on_the_measured_config_the_scene_light_is_still_scored(tmp_path):
    # MEASURED's glow mode reads the course's LEDs; on a synthetic lamp it may not: scored, not assumed
    run = tmp_path / "run"
    write_frames(run / "frames", [scene(lights=(RED_LAMP,))] * 3)
    labels(run / "labels.csv", "first,last,light\n0,2,red\n")
    res = ev.run_eval(run, config=replace(SCENE_CONFIG, color=replace(SCENE_CONFIG.color)), estimation=EST)[0]
    assert sum(res["light"]["frame"]["matrix"]["red"].values()) == 3


def test_a_run_s_report_pngs_are_not_taken_for_frames(tmp_path):
    run = tmp_path / "attempt_01"
    run.mkdir()
    cv2.imwrite(str(run / "nav_run.png"), scene())
    (run / "labels.csv").write_text("first,last,sign\n0,0,no\n")
    with pytest.raises(FileNotFoundError, match="no frames in"):
        ev.find_frames(run)
    assert ev.main([str(run)]) == 1
    write_frames(run, [scene()])                      # the recorder's 000000.png beside them: those are frames
    assert ev.find_frames(run) == run and [f for f, _ in ev.frames_in(run)] == [0]


def test_the_labeller_jumps_to_a_frame_id_without_labelling_the_way():
    lab = lf.Labeller([1, 2, 3, 50, 51, 52])
    lab.key("s")
    for k in "50g":
        lab.key(k)
    assert lab.fid == 50 and lab.labels == {}
    for k in "4g":                                      # no frame 4: the next one there is
        lab.key(k)
    assert lab.fid == 50
    lab.key("d")
    assert sorted(lab.labels) == [50] and lab.fid == 51
    lab.key("g")
    assert lab.fid == 1
