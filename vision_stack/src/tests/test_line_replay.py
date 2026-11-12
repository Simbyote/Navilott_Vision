"""
test_line_replay.py  --  src/analysis/line_replay.py

Made-up detect-range line runs: an older one with only saved pictures,
replayed through a stand-in Phase 2 (a line read 1.5 cm long, clipped near
the robot, unseen far off), and a newer one whose looks logged their own
median and SD. Each look is judged on D4b's tolerance, tighter at 10 cm and
in; clipped and unseen looks are counted apart; a steady offset is called
out; runs aimed at the light or sign are left out.

--software  No camera; a stand-in chain.
"""
import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.analysis import line_replay as lr

pytestmark = pytest.mark.software

FIELDS = ("trial", "t_s", "gap_cm", "angle_deg", "offset_cm", "state", "frames", "seen_pct", "wrong_frames",
          "voted", "expected", "tilt_deg", "result", "reported_cm", "error_cm", "cm_sd", "cm_frames", "clipped_pct")


def run(root, stamp, poses, target="line", live=None):
    folder = root / f"routine_detect-range_{stamp}"
    folder.mkdir(parents=True)
    rows = []
    for i, (gap, angle) in enumerate(poses, start=1):
        tag = f"{gap:g}cm" + (f"_{angle:g}deg" if angle else "")
        img = np.full((270, 480, 3), 60, np.uint8)
        img[0, 0] = (gap, angle + 50, 0)                     # the stand-in chain reads the pose from here
        cv2.imwrite(str(folder / f"attempt_{i:02d}_{tag}_present.png"), img)
        (folder / f"attempt_{i:02d}_{tag}_present.png").rename(folder / f"attempt_{i:02d}_{tag}_present.jpg")
        row = {"trial": i, "gap_cm": gap, "angle_deg": angle, "offset_cm": 0, "state": "present", "frames": 100,
               "seen_pct": 100, "voted": True, "tilt_deg": round(0.5 - 0.25 * angle, 2), "result": "ok"}
        if live:
            row.update(live[i - 1])
        rows.append(row)
    with open(folder / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    (folder / "results.json").write_text(json.dumps({"options": {"target": target},
                                                     "conditions": {"start": {"time": f"2026-11-02 {stamp}"}}}))
    return folder


def chain(frame):
    gap = float(frame[0, 0, 0])
    line = SimpleNamespace(detected=gap <= 18, clipped=gap <= 8, distance_cm=0.0 if gap <= 8 else gap + 1.5,
                           tilt_deg=0.0)
    return SimpleNamespace(stop_line=line)


@pytest.fixture
def folder(tmp_path):
    poses = [(8, 0), (10, 0), (14, 0), (14, 10), (20, 0)]
    run(tmp_path, "161713", poses)
    run(tmp_path, "180000", [(12, 0), (16, 0)], live=[{"reported_cm": 12.3, "cm_sd": 0.1, "clipped_pct": 0},
                                                       {"reported_cm": 16.2, "cm_sd": 0.7, "clipped_pct": 0}])
    run(tmp_path, "190000", [(10, 0)], target="sign")
    return tmp_path


def test_replayed_and_live_looks_judged_on_the_tolerance(folder):
    res = lr.analyze(lr.run_folders([str(folder)]), chain=chain)
    assert len(res["runs"]) == 2, "the sign run is left out"
    by = {(r["run"][-6:], r["gap_cm"], r["angle_deg"]): r for r in res["rows"]}
    assert by[("161713", 8, 0)]["result"] == lr.CLIPPED and by[("161713", 8, 0)]["reported_cm"] is None
    ten = by[("161713", 10, 0)]
    assert (ten["source"], ten["reported_cm"], ten["error_cm"], ten["tolerance_cm"], ten["result"]) == \
        (lr.REPLAY, 11.5, 1.5, 2.0, "pass")                  # D4b since 2026-11-05: +-2 cm
    assert ten["result_original"] == "fail", "the original rule was +-0.5 cm up to 10 cm"
    assert res["all"]["pass_original"] < res["all"]["pass"]
    assert by[("161713", 20, 0)]["result"] == lr.NOT_SEEN
    assert by[("161713", 14, 10)]["tilt_deg"] == -2.0, "the routine's own tilt is kept"
    assert by[("180000", 12, 0)]["source"] == lr.LIVE and by[("180000", 12, 0)]["result"] == "pass"
    assert by[("180000", 16, 0)]["result"] == "fail", "inside 1 cm but its SD is over 0.5"
    assert res["all"]["clipped"] == 1 and res["all"]["not_seen"] == 1 and res["all"]["judged"] == 5
    assert any("steady offset" in f for f in res["findings"])
    assert any("within D4b's +-2 cm" in f and "within the original" in f for f in res["findings"])
    assert res["tilt_fit"]["tilt_to_heading"] == -4.0


def test_main_writes_the_table(folder, tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "default_chain", chain)
    monkeypatch.setattr(lr.analyze, "__defaults__", (chain, None))          # the chain; table_c stays MEASURED's
    out = tmp_path / "out"
    assert lr.main([str(folder), "--out", str(out)]) == 0
    with open(out / "line_replay.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7 and set(rows[0]) == set(lr.FIELDS)


def test_no_line_run_is_an_error(tmp_path):
    run(tmp_path, "190000", [(10, 0)], target="light")
    with pytest.raises(FileNotFoundError):
        lr.run_folders([str(tmp_path)])


def test_the_table_correction_centres_the_median_and_warns_about_the_obstacle(folder):
    res = lr.analyze(lr.run_folders([str(folder)]), chain=chain, table_c=-0.4383)
    m = res["all"]["median_error_cm"]
    fix = res["table_c"]
    assert fix["c"] == -0.4383 and fix["c_centred"] == pytest.approx(-0.4383 - m, abs=1e-4)
    assert fix["shift_cm"] == pytest.approx(-m, abs=0.01)
    text = " ".join(res["findings"])
    assert "curve[2]" in text and "obstacle" in text and "D6" in text


def test_no_correction_without_an_error_or_a_table():
    assert lr.table_correction(None, -0.4) is None and lr.table_correction(1.7, None) is None
    assert lr.table_correction(1.7, -0.4383) == {"c": -0.4383, "c_centred": -2.1383, "shift_cm": -1.7}
