"""
test_sweep_lamps.py  --  src/scripts/sweep_lamps.py

Synthetic recordings shaped like the course's light: a teal board in the
green band but dimmer than a lit LED, and small LEDs, each a white center
with a colored ring and a dimmer glow. Starting from bands that take the board in, the sweep
reads every labelled frame right, and says so before and after. Then the
pieces: labels and frame spans, reading by frame number, floors set on both
red bands, a color scored on its own, the tie-breaks, --write, and the
command line's refusals.

--software  Synthetic frames through the robot's own ROI crop. No hardware.
"""
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.perception.color_branch import BlobFilter, ColorRange, HSVRanges, load_hsv_ranges
from src.scripts import sweep_lamps as sw
from src.tests.scenes import FRAME_H, FRAME_W, SCENE_CONFIG

pytestmark = pytest.mark.software

# Frame px, inside SCENE_CONFIG's traffic ROI (x 144-288, y 0-94)
LED = {"red": (180, 45), "yellow": (210, 45), "green": (240, 45)}
RING = {"red": (0, 0, 255), "yellow": (0, 220, 255), "green": (200, 255, 0)}     # BGR, lit
BOARD = (110, 120, 20)          # teal, H ~85, V ~120: in a wide green band, under a lit LED's V
# The starting bands: green's floors take the board in, as the hand-tuned ones did
START = {"red_low": {"lower": [0, 40, 100], "upper": [10, 255, 255]},
         "red_high": {"lower": [170, 40, 100], "upper": [180, 255, 255]},
         "yellow": {"lower": [20, 40, 100], "upper": [35, 255, 255]},
         "green": {"lower": [76, 40, 100], "upper": [95, 255, 255]}}


def light_frame(lit=None):
    frame = np.full((FRAME_H, FRAME_W, 3), 60, np.uint8)
    cv2.rectangle(frame, (165, 33), (224, 57), BOARD, -1)       # 60 x 25: passes the aspect gate
    if lit:
        cv2.circle(frame, LED[lit], 8, [int(0.7 * c) for c in RING[lit]], -1)      # glow: a low V floor takes it in
        cv2.circle(frame, LED[lit], 5, RING[lit], -1)
        cv2.circle(frame, LED[lit], 2, (255, 255, 255), -1)
    return frame


def recording(tmp_path, spans, first=0):
    """One run folder, frames named by number as navigation_linker writes them: [(lit, count), ...]."""
    folder = tmp_path / "run" / "frames"
    folder.mkdir(parents=True)
    i = first
    for lit, n in spans:
        for _ in range(n):
            cv2.imwrite(str(folder / f"{i:06d}.png"), light_frame(lit))
            i += 1
    return folder.parent


@pytest.fixture
def start_hsv(tmp_path):
    p = tmp_path / "start.json"
    p.write_text(json.dumps(START))
    return p


# Gates for LEDs, as MEASURED's: the board passes them too, so it reads as green
LED_BLOB = BlobFilter(min_area=4.0, max_area=3000.0, ref_area=20.0, min_roundness=0.2, min_core_px=0)
CONFIG = replace(SCENE_CONFIG, color=replace(SCENE_CONFIG.color, blob=LED_BLOB))


def run(args, **kw):
    lines = []
    code = sw.main([str(a) for a in args], config=CONFIG, say=lines.append, **kw)
    return code, "\n".join(lines)


# =============================================================================
# The sweep, end to end
# =============================================================================

def test_from_bands_that_take_the_board_the_sweep_reads_every_frame_right(tmp_path, start_hsv):
    run_dir = recording(tmp_path, [("green", 6), ("yellow", 6), ("red", 6), (None, 4)])
    code, out = run([f"green={run_dir}:0-5", f"yellow={run_dir}:6-11", f"red={run_dir}:12-17",
                     f"off={run_dir}:18-21", "--hsv", start_hsv, "--frames", "6"])
    assert code == 0
    now, swept = out.split("\nswept:")
    assert "wrong color 0" not in now                       # the board read as green somewhere
    assert "right 22, wrong color 0, missed 0" in swept
    assert "BlobFilter(min_area = " in swept and "WARNING: " not in swept


def test_write_saves_bands_the_loader_accepts(tmp_path, start_hsv):
    run_dir = recording(tmp_path, [("green", 4), (None, 4)])
    out_path = tmp_path / "hsv.json"
    code, out = run([f"green={run_dir}:0-3", f"off={run_dir}:4-7", "--hsv", start_hsv, "--frames", "4",
                     "--write", "--out", out_path])
    assert code == 0 and f"wrote {out_path}" in out
    ranges = load_hsv_ranges(str(out_path))
    assert ranges.green.lower[2] > 120                      # V floor above the board's
    assert ranges.green.lower[0] == START["green"]["lower"][0]      # hue kept


# =============================================================================
# The pieces
# =============================================================================

@pytest.mark.parametrize("arg, expect", [
    ("red=runs/nav_x", ("red", "runs/nav_x", None)),
    ("green=runs/nav_x:0-86", ("green", "runs/nav_x", (0, 86))),
    ("off=C:/x/y:3-3", ("off", "C:/x/y", (3, 3))),
])
def test_labels_take_a_path_and_an_optional_frame_span(arg, expect):
    assert sw.parse_label(arg) == expect


@pytest.mark.parametrize("arg", ["blue=x", "red", "red=", "red=x:9-3"])
def test_bad_labels_are_refused(arg):
    with pytest.raises(ValueError):
        sw.parse_label(arg)


def test_a_span_picks_frames_by_the_number_in_their_name(tmp_path):
    run_dir = recording(tmp_path, [("red", 3), ("green", 3)])
    frames = sw.read_labelled(str(run_dir), (3, 5))
    assert len(frames) == 3 and all(f[LED["green"][1], LED["green"][0]].tolist() == [255, 255, 255] for f in frames)
    assert len(sw.read_labelled(str(run_dir), None, limit=2)) == 2
    with pytest.raises(FileNotFoundError, match="frames 50-60"):
        sw.read_labelled(str(run_dir), (50, 60))


def test_the_span_is_frame_numbers_not_places_in_the_folder(tmp_path):
    run_dir = recording(tmp_path, [("red", 3), ("green", 3)], first=50)
    assert len(sw.read_labelled(str(run_dir), (53, 55))) == 3
    with pytest.raises(FileNotFoundError):
        sw.read_labelled(str(run_dir), (0, 5))


def samples_of(tmp_path, spans):
    run_dir = recording(tmp_path, [(lit, n) for lit, n in spans])
    out, i = [], 0
    for lit, n in spans:
        out += [(lit or sw.OFF, np.ascontiguousarray(sw.traffic_roi(f, CONFIG)))
                for f in sw.read_labelled(str(run_dir), (i, i + n - 1))]
        i += n
    return out


def test_tied_floors_go_to_the_one_with_the_strongest_weakest_reading(tmp_path):
    samples = samples_of(tmp_path, [("green", 3), ("red", 3), (None, 2)])
    # ref_area above the ring: with the glow taken in a lamp scores higher
    hsv, tables = sw.sweep_bands(samples, load_hsv_ranges_dict(START), replace(LED_BLOB, ref_area=150.0), 0.4)
    for color, scores in tables.items():
        best = max(r["score"] for r in scores.values())
        margin = max(r["low"] or 0.0 for r in scores.values() if r["score"] == best)
        chosen = tuple(getattr(hsv, "red_low" if color == "red" else color).lower[1:])
        assert scores[chosen]["score"] == best and scores[chosen]["low"] == margin, color
        assert len({r["low"] for r in scores.values() if r["score"] == best}) > 1, "the test needs a tie to break"


def test_the_blob_gates_are_the_best_of_the_whole_grid_on_whole_frames(tmp_path):
    samples = samples_of(tmp_path, [("green", 3), ("red", 3), (None, 2)])
    hsv, _ = sw.sweep_bands(samples, load_hsv_ranges_dict(START), LED_BLOB, 0.4)
    blob, scores = sw.sweep_blob(samples, hsv, 0.4)
    assert [b for b, _ in scores] == sw.blob_grid() and all(b.ref_area > b.min_area for b, _ in scores)
    assert blob == sw._best(scores) and blob != sw.blob_grid()[0]
    chosen = next(r for b, r in scores if b == blob)
    assert chosen["right"] == len(samples) and chosen["wrong"] == 0


def test_a_floor_sets_s_and_v_and_keeps_hue_and_caps():
    hsv = HSVRanges()
    red = sw.with_floor(hsv, "red", 33, 144)
    assert red.red_low == ColorRange((0, 33, 144), (10, 255, 255))
    assert red.red_high == ColorRange((170, 33, 144), (180, 255, 255))
    assert red.green == hsv.green
    assert sw.with_floor(hsv, "green", 1, 2).green == ColorRange((40, 1, 2), (80, 255, 255))


def test_a_color_is_scored_on_its_own_frames_and_everyone_elses():
    roi = lambda lit: np.ascontiguousarray(sw.traffic_roi(light_frame(lit), SCENE_CONFIG))
    samples = [("green", roi("green")), ("red", roi("red")), ("off", roi(None))]
    hsv = load_hsv_ranges_dict(START)
    blob = LED_BLOB
    loose = sw.color_score(samples, "green", hsv, blob, 0.4)
    assert (loose["hits"], loose["n"], loose["false"]) == (1, 1, 2)             # the board, on red and off
    tight = sw.color_score(samples, "green", sw.with_floor(hsv, "green", 40, 200), blob, 0.4)
    assert (tight["hits"], tight["false"]) == (1, 0) and tight["score"] == 1
    assert loose["score"] == 1 - sw.FALSE_WEIGHT * 2


def test_the_best_is_the_top_score_then_the_strongest_weakest_reading_then_the_middle():
    r = lambda score, low: {"score": score, "low": low}
    assert sw._best([("a", r(5, 0.9)), ("b", r(6, 0.5)), ("c", r(4, 1.0))]) == "b"
    assert sw._best([("a", r(6, 0.5)), ("b", r(6, 0.5)), ("c", r(6, 0.9)), ("d", r(6, 0.5))]) == "c"
    assert sw._best([(k, r(6, 0.9)) for k in "abcde"]) == "c"
    assert sw._best([("a", r(0, None)), ("b", r(0, None))]) == "b"


def test_no_lamp_frames_and_bad_paths_are_refused(tmp_path, start_hsv):
    run_dir = recording(tmp_path, [(None, 2)])
    assert run([f"off={run_dir}", "--hsv", start_hsv])[0] == 2
    code, out = run([f"red={tmp_path / 'nope'}", "--hsv", start_hsv])
    assert code == 2 and "no such image or folder" in out


def load_hsv_ranges_dict(d):
    return HSVRanges(**{k: ColorRange(tuple(v["lower"]), tuple(v["upper"])) for k, v in d.items()})
