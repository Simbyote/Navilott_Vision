"""
test_light_clips.py  --  src/analysis/light_clips.py

A made-up parked clip: green 40 frames, yellow 20, red 40, green again. The
vote follows each change two frames late; yellow is voted stop (right, by the
2 Nov agreement); two green frames read red (wrong colour per frame, filtered
by the vote) and the release after red waits on the red memory. Per colour,
per frame and as voted; every change timed; the CLI scores a p3.csv log.

--software  No camera.
"""
import csv

import pytest

from src.analysis import light_clips as lc

pytestmark = pytest.mark.software

COLOUR_STATE = {"red": "stop", "yellow": "caution", "green": "go"}


def clip():
    seq = ["green"] * 40 + ["yellow"] * 20 + ["red"] * 40 + ["green"] * 30
    readings, vote = [], "go"
    for fid, truth in enumerate(seq):
        said = truth
        if fid in (10, 11):                     # a red ring on a green lamp, two frames
            said = "red"
        if fid >= 2 and seq[fid - 2] == truth:  # the vote follows two frames late
            vote = "stop" if truth == "yellow" else COLOUR_STATE[truth]
        if truth == "green" and fid >= 100 and fid < 110:
            vote = "stop"                       # the red memory holds the release
        readings.append({"fid": fid, "light": said, "vote_drive": vote})
    edges = [(0, 39, "green"), (40, 59, "yellow"), (60, 99, "red"), (100, 129, "green")]
    labels = [{"first": a, "last": b, "light": c, "range_cm": 10.0, "condition": "room"} for a, b, c in edges]
    return readings, labels


def test_each_colour_per_frame_and_voted():
    res = lc.score_clip(*clip(), settle=2, run="nav_a")
    by = {r["light"]: r for r in res["rows"]}
    assert set(by) == {"green", "yellow", "red"}
    assert by["green"]["frames"] == 70 and by["green"]["wrong_colour_frames"] == 2
    assert by["green"]["frame_right_pct"] == pytest.approx(100 * 68 / 70, abs=0.1)
    assert by["yellow"]["voted_right_pct"] == 100.0             # voted stop counts as right
    assert by["red"]["voted_wrong_frames"] == 0
    assert by["green"]["voted_wrong_frames"] == 8                # the held release, past the settle frames
    assert all(r["gap_cm"] == 10.0 and r["condition"] == "room" for r in res["rows"])


def test_every_change_is_timed_and_kinded():
    ch = lc.score_clip(*clip(), settle=2)["changes"]
    assert [(c["from"], c["to"], c["kind"]) for c in ch] == [
        ("green", "yellow", lc.TO_STOP), ("yellow", "red", lc.STOP_TO_STOP), ("red", "green", lc.RELEASE)]
    assert ch[0]["response_frames"] == 2 and ch[0]["response_ms"] == 100
    assert ch[1]["response_frames"] == 0                         # stop already stands for red
    assert ch[2]["response_frames"] == 10


def test_cli_from_log(tmp_path):
    folder = tmp_path / "nav_20261105_101500"
    folder.mkdir()
    readings, labels = clip()
    with open(folder / "p3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "p2_stop", "p2_traffic", "drive_state", "p3_log", "p2_stop_line_px", "p2_offset"])
        for r in readings:
            w.writerow([r["fid"], 0, 1, r["vote_drive"], "", "", ""])
    with open(folder / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["first", "last", "sign", "light", "line_cm", "obstacle_cm", "offset_cm", "range_cm", "condition"])
        for lb in labels:
            w.writerow([lb["first"], lb["last"], "", lb["light"], "", "", "", 10, "room"])
    assert lc.main([str(tmp_path), "--from-log", "--out", str(tmp_path / "out")]) == 0
    rows = list(csv.DictReader(open(tmp_path / "out" / "light_clips.csv")))
    assert [r["light"] for r in rows] == ["red", "yellow", "green"]
    assert list(rows[0]) == list(lc.CLIP_FIELDS)
    assert lc.main([str(tmp_path / "nothing")]) == 1
