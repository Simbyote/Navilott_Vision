"""
test_range_compare.py  --  src/analysis/range_compare.py

Made-up detect-range runs at two spots: a wall, where the sign is a strong
red, and a window, where it is a washed-out pink. The sign is found by what
changed between the present and absent looks; its redness clears the
detector's threshold at the wall and falls under it at the window. Rounds
come from time order, conditions from the CSV, frame_meta runs match by name,
and an empty run (a full disk) is skipped, not fatal.

--software  Synthetic frames, no camera.
"""
import csv
import json

import cv2
import numpy as np
import pytest

from src.analysis import range_compare as rc

pytestmark = pytest.mark.software

FIELDS = ("trial", "t_s", "gap_cm", "angle_deg", "offset_cm", "state", "frames", "seen_pct", "wrong_frames",
          "voted", "expected", "tilt_deg", "result")
WALL_RED, WINDOW_PINK = (40, 40, 210), (150, 140, 160)     # BGR: redness 170 and 10


def background() -> np.ndarray:
    frame = np.full((270, 480, 3), 120, np.uint8)
    frame[200:, :] = 60
    return frame


def with_sign(frame: np.ndarray, gap: float, bgr) -> np.ndarray:
    out = frame.copy()
    r = int(40 - gap * 0.6)
    cx, cy = 400, 90
    pts = np.array([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8],
                   np.int32)
    cv2.fillPoly(out, [pts], bgr)
    return out


def detect_run(root, stamp, bgr, seen, gaps=(10, 20, 30), wrong=0) -> None:
    folder = root / f"routine_detect-range_{stamp}"
    folder.mkdir(parents=True)
    rows, trial = [], 0
    for g, s in zip(gaps, seen):
        for state in ("present", "absent"):
            trial += 1
            frame = with_sign(background(), g, bgr) if state == "present" else background()
            cv2.imwrite(str(folder / f"attempt_{trial:02d}_{g:g}cm_{state}.jpg"), frame)
            present = state == "present"
            rows.append({"trial": trial, "t_s": trial * 5.0, "gap_cm": g, "angle_deg": 0, "offset_cm": 0,
                         "state": state, "frames": 40, "seen_pct": (100 if s else 0) if present else 0,
                         "wrong_frames": 0 if present else wrong, "voted": present and s, "expected": present,
                         "tilt_deg": "", "result": ("ok" if s else "missed") if present else "ok"})
    with open(folder / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    t = f"2026-11-02 {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}"
    (folder / "results.json").write_text(json.dumps({"options": {"target": "sign"},
                                                     "conditions": {"start": {"time": t, "commit": "abc1234"}}}))


def meta_run(root, name, lux, gain) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    with open(folder / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("frame", "t_s", "exposure_us", "analogue_gain", "digital_gain", "lux"))
        for i in range(5):
            w.writerow((i, i * 0.05, 29985, gain, 1.0, lux))


@pytest.fixture
def day(tmp_path):
    runs = tmp_path / "runs"
    detect_run(runs, "20261102_103000", WALL_RED, (False, True, True))
    detect_run(runs, "20261102_110000", WINDOW_PINK, (False, False, False), wrong=2)
    detect_run(runs, "20261102_123000", WALL_RED, (False, True, True))
    empty = runs / "routine_detect-range_20261102_113000"
    empty.mkdir()
    (empty / "trials.csv").write_text("")
    (runs / "conditions.csv").write_text("run,condition\n20261102_103000,Wall\n20261102_110000,Window\n"
                                         "20261102_123000,Wall\n")
    metas = tmp_path / "meta"
    meta_run(metas, "meta_20261102_135618-FACING-WALL", 109.0, 1.74)
    meta_run(metas, "meta_20261102_135911-FACING-WINDOW", 107.0, 1.80)
    return runs, metas


def test_conditions_rounds_and_rates(day):
    runs, metas = day
    skipped = []
    res = rc.analyze(rc.run_folders([str(runs)], skipped), rc.read_conditions(runs / "conditions.csv"),
                     rc.meta_folders([str(metas)]), floor=20.0)
    assert skipped == ["routine_detect-range_20261102_113000"], "the empty run is named, not fatal"
    assert res["conditions"] == ["Wall", "Window"]
    wall, window = res["by_condition"]["Wall"], res["by_condition"]["Window"]
    assert set(wall["rounds"]) == {"1", "2"}, "a condition's runs in time order"
    assert wall["window"] == {"k": 4, "n": 4, "pct": 100.0, "mean_seen_pct": 100.0}
    assert window["window"]["k"] == 0 and window["window"]["n"] == 2
    assert window["absent_wrong_frames"] == 6 and window["absent_wrong_votes"] == 0
    assert wall["meta"]["lux"] == pytest.approx(109.0) and window["meta"]["analogue_gain"] == pytest.approx(1.8)


def test_the_sign_is_found_by_difference_and_measured_against_the_threshold(day):
    runs, metas = day
    res = rc.analyze(rc.run_folders([str(runs)]), rc.read_conditions(runs / "conditions.csv"), floor=20.0)
    looks = {(r["condition"], r["gap_cm"]): r for r in res["looks"] if r["state"] == "present" and r["round"] == "1"}
    wall, window = looks[("Wall", 20.0)], looks[("Window", 20.0)]
    assert wall["target_found"] and window["target_found"]
    assert wall["margin"] > 50, "a strong red clears the threshold easily"
    assert window["margin"] < 0, "washed-out pink stays under the floor"
    assert res["by_condition"]["Window"]["margin_median"] < 0 < res["by_condition"]["Wall"]["margin_median"]
    assert any("under the detector's threshold" in f and "Window" in f for f in res["findings"])


def test_main_writes_the_table_and_the_figure(day, tmp_path):
    runs, metas = day
    out = tmp_path / "out"
    assert rc.main([str(runs), "--conditions", str(runs / "conditions.csv"), "--meta", str(metas),
                    "--out", str(out)]) == 0
    with open(out / "range_compare.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 6 and set(rows[0]) == set(rc.LOOSE_FIELDS)
    assert (out / "range_compare.json").is_file()
    if rc.common.pyplot() is not None:
        assert (out / "range_compare.png").is_file()


def test_meta_matches_by_name_and_a_conditions_file_needs_a_run_column(tmp_path):
    metas = [tmp_path / "meta_20261102_135712-FACING_FIRST-TRAFFIC-LIGHT", tmp_path / "meta_20261102_135402-STARTING-POSITION"]
    assert rc.match_meta("Traffic light", metas) == metas[0]
    assert rc.match_meta("Start", metas) == metas[1]
    assert rc.match_meta("Window", metas) is None
    bad = tmp_path / "c.csv"
    bad.write_text("folder,condition\nx,y\n")
    with pytest.raises(ValueError):
        rc.read_conditions(bad)
