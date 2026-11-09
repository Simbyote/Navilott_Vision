"""
test_sweep_camera.py  --  src/scripts/sweep_camera.py

A fake camera whose picture depends on the controls it was opened with: under
"awb-mode=daylight" the synthetic stop sign is red, otherwise its blue is
raised to its red, as the backlit sign read on 2026-10-26. The sweep runs the
real Phase 2 chain on both and ranks daylight first, with the baseline run
before it; then the folder it writes, the settle time, a camera that won't
open, and the command line's refusals.

--software  Synthetic frames through the real chain. No hardware.
"""
import csv

import numpy as np
import pytest

from src.capture.camera import FrameData
from src.phase2_linker import run_chain
from src.scripts import sweep_camera as sc
from src.tests.scenes import SCENE_CONFIG, scene

pytestmark = pytest.mark.software

RED_SIGN = scene(sign=True)


def backlit(frame: np.ndarray) -> np.ndarray:
    """Blue raised to red wherever red leads: a red sign reads purple, its redness gone."""
    out = frame.copy()
    out[..., 0] = np.maximum(out[..., 0], out[..., 2])
    return out


class FakeCamera:
    def __init__(self, controls, opened):
        opened.append(dict(controls))
        self.frame = RED_SIGN if controls.get("awb-mode") == "daylight" else backlit(RED_SIGN)
        self.n, self.released = 0, False

    def read(self):
        self.n += 1
        if self.n % 5 == 0:
            return None                         # a dropped frame doesn't count
        return FrameData(self.frame, self.n, self.n * 50)

    def release(self):
        self.released = True


def chain(frame, frame_id, ts):
    return run_chain(frame, frame_id, ts, SCENE_CONFIG, parallel=False)


def run(tmp_path, settings, **kw):
    opened = []
    rows = sc.sweep(settings, tmp_path, settle_s=0.0, frames=4, chain=chain,
                    opener=lambda c: FakeCamera(c, opened), say=lambda *_: None, **kw)
    return rows, opened


def test_the_setting_that_keeps_the_sign_red_ranks_first(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CAMERA_CONTROLS", {"saturation": 1.2})
    rows, opened = run(tmp_path, [{}, {"awb-mode": "daylight"}, {"exposure-value": 1}])
    assert [r["setting"] for r in rows] == ["current", "awb-mode=daylight", "exposure-value=1"]
    assert opened[1] == {"saturation": 1.2, "awb-mode": "daylight"}, "each setting goes over params'"
    by = {r["setting"]: r for r in rows}
    assert by["awb-mode=daylight"]["sign_pass_pct"] == 100.0 and by["current"]["sign_pass_pct"] == 0.0
    assert by["awb-mode=daylight"]["sign_redness_p99"] > 100 > by["current"]["sign_redness_p99"]
    assert all(r["frames"] == 4 for r in rows)
    assert sc.ranked(rows)[0]["setting"] == "awb-mode=daylight"
    summary = (tmp_path / "summary.txt").read_text().splitlines()
    assert "awb-mode=daylight" in summary[3], "the summary is best first"


def test_the_folder_holds_the_table_the_pictures_and_the_sheet(tmp_path):
    rows, _ = run(tmp_path, [{}, {"awb-mode": "daylight"}], rounds=2)
    assert [(r["round"], r["setting"]) for r in rows] == [(1, "current"), (1, "awb-mode=daylight"),
                                                           (2, "current"), (2, "awb-mode=daylight")]
    with open(tmp_path / "sweep.csv") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 4 and list(table[0]) == list(sc.FIELDS)
    assert all((tmp_path / r["picture"]).is_file() for r in rows)
    assert (tmp_path / "sheet.png").is_file()


def test_a_camera_that_wont_open_is_skipped_not_fatal(tmp_path):
    def opener(controls):
        if "exposure-value" in controls:
            raise sc.CaptureError("Failed to open camera pipeline")
        return FakeCamera(controls, [])
    rows = sc.sweep([{}, {"exposure-value": 1}], tmp_path, settle_s=0.0, frames=2, chain=chain,
                    opener=opener, say=lambda *_: None)
    assert [r["setting"] for r in rows] == ["current"]


def test_frames_read_while_settling_are_dropped():
    cam = FakeCamera({}, [])
    t = iter([0.0, 0.5, 1.0, 1.5, 9.0])
    got = sc.grab(cam, settle_s=2.0, n=3, clock=lambda: next(t))
    assert [fd.frame_id for fd in got] == [4, 6, 7], "3 reads dropped while settling, then the drop at 5 skipped"


def test_a_setting_is_pairs_split_on_spaces_or_commas():
    assert sc.parse_setting("awb-mode=daylight saturation=1.5") == {"awb-mode": "daylight", "saturation": 1.5}
    assert sc.parse_setting("exposure-value=1,ae-constraint-mode=shadows") == \
        {"exposure-value": 1, "ae-constraint-mode": "shadows"}
    assert sc.parse_setting("") == {} and sc.label({}) == "current"
    with pytest.raises(ValueError, match="awb-mode"):
        sc.parse_setting("awb-mode=sunny")


def test_every_backlight_setting_is_a_valid_one():
    for text in sc.BACKLIGHT:
        sc.parse_setting(text)


def test_the_command_line_lists_refuses_and_splits(tmp_path, monkeypatch, capsys):
    assert sc.main(["--list"]) == 0
    assert "awb-mode" in capsys.readouterr().out
    assert sc.main(["--try", "awb-mode=sunny"]) == 2
    seen = {}
    monkeypatch.setattr(sc, "sweep", lambda settings, out, *a: seen.setdefault("s", settings) and [])
    sc.main(["--try", "awb-mode=daylight|exposure-value=1 saturation=1.5", "--out", str(tmp_path)])
    assert seen["s"] == [{}, {"awb-mode": "daylight"}, {"exposure-value": 1, "saturation": 1.5}]
    monkeypatch.setattr(sc, "CAMERA_CONTROLS", {"awb-mode": "sunny"})
    assert sc.main(["--out", str(tmp_path)]) == 2, "a bad params.CAMERA_CONTROLS is said before the camera opens"
