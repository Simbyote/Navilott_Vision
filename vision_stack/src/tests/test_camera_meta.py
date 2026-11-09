"""
test_camera_meta.py  --  src/analysis/camera_meta.py

A made-up frame_meta recording: auto exposure falling from a bright start
and settling after a second, gains, colour gains and a colour temperature that
keeps wandering, lux, and the traffic label. analyze() finds when it settled,
the swing after, the shutter-limited share and the findings; the figure and
JSON are written beside the CSV; the newest frame_meta frames.csv is found and
capture's frames.csv of the same name passed over; a CSV without exposure is
refused.

--software  Synthetic CSVs. No camera.
"""
import csv
import os

import numpy as np
import pytest

from src.analysis import camera_meta as cm
from src.analysis.common import Table
from src.diagnostics.frame_meta import FRAME_FIELDS

pytestmark = pytest.mark.software

N = 100         # 5 s at 20 FPS


def recording(path, temp_swing=800.0, settle_frames=20, exposure_after=20000.0, frame_us=50000):
    rows = []
    for i in range(N):
        exp = exposure_after * (1 + 3.0 * max(0, settle_frames - i) / settle_frames)
        label = "red" if 40 <= i < 70 else ""
        rows.append({"frame": i, "t_s": round(i * 0.05, 3), "exposure_us": round(exp), "analogue_gain": 2.0,
                     "digital_gain": 1.0, "colour_gain_r": 1.8, "colour_gain_b": round(1.4 + 0.002 * i, 3),
                     "colour_temp_k": round(4500 + temp_swing * np.sin(i / 10)), "lux": 120.0 + i,
                     "frame_duration_us": frame_us, "ae_locked": int(i >= settle_frames),
                     "sensor_to_python_ms": 60.0, "label": label, "confidence": 1.0 if label else "",
                     "white_px": 4 if label else ""})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FRAME_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_settling_the_swing_after_and_the_findings(tmp_path):
    res = cm.analyze(Table(recording(tmp_path / "meta" / "frames.csv")))
    assert res["frames"] == N and res["time_unit"] == "s"
    assert res["settled_at"] == pytest.approx(1.0), "exposure x gain within 10% of its end from frame 20"
    assert res["light_swing_after_settle"] == pytest.approx(1.0)
    assert res["temp_range_after_settle_k"] > 1500
    assert res["shutter_limited_share"] == pytest.approx(0.12), "frames 0-11 are at 45 ms or more of a 50 ms frame"
    assert res["labels"] == {"none": 70, "red": 30}
    assert res["fields"]["lux"]["min"] == 120.0
    text = " ".join(res["findings"])
    assert "white balance moved" in text and "settle" not in text, "1 s of 5 is under a quarter"


def test_a_steady_recording_has_nothing_to_say(tmp_path):
    res = cm.analyze(Table(recording(tmp_path / "frames.csv", temp_swing=0.0, settle_frames=1)))
    assert res["findings"] == [] and res["settled_at"] == pytest.approx(0.05), "only frame 0 is outside"


def test_a_late_settle_and_a_shutter_limited_scene_are_findings(tmp_path):
    res = cm.analyze(Table(recording(tmp_path / "frames.csv", temp_swing=0.0, settle_frames=60,
                                     exposure_after=48000.0)))
    text = " ".join(res["findings"])
    assert "took until 2.9 s" in text and "exposure was at the frame's length" in text


def test_settle_index():
    assert cm.settle_index(np.array([4.0, 3.0, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0])) == 2
    assert cm.settle_index(np.array([1.0, 1.0, 1.0])) == 0
    assert cm.settle_index(np.array([np.nan, np.nan])) is None


def test_main_writes_the_figure_and_json_beside_the_csv(tmp_path, capsys):
    path = recording(tmp_path / "meta_x" / "frames.csv")
    assert cm.main([str(path.parent)]) == 0
    assert (path.parent / "camera_meta.json").is_file()
    if cm.common.pyplot() is not None:
        assert (path.parent / "camera_meta.png").stat().st_size > 10_000
    out = capsys.readouterr().out
    assert "AE settled at 1.00 s" in out and "white balance moved" in out


def test_the_newest_frame_meta_csv_is_found_and_captures_is_passed_over(tmp_path, monkeypatch):
    old = recording(tmp_path / "runs" / "meta_a" / "frames.csv")
    new = recording(tmp_path / "runs" / "meta_b" / "frames.csv")
    other = tmp_path / "artifacts" / "x" / "test_capture_characterization" / "frames.csv"
    other.parent.mkdir(parents=True)
    other.write_text("frame_id,timestamp_ms,dt_ms\n0,0,\n1,50,50\n")
    os.utime(old, (1, 1))
    monkeypatch.setattr(cm.common, "SEARCH_ROOTS", (tmp_path / "runs", tmp_path / "artifacts"))
    assert cm.find(None) == new
    assert cm.find(str(tmp_path / "runs" / "meta_a")) == old
    with pytest.raises(FileNotFoundError):
        cm.find(str(other.parent))
    assert not cm.is_frame_meta(other)


def test_a_csv_without_exposure_is_refused(tmp_path, capsys):
    p = tmp_path / "frames.csv"
    p.write_text("frame_id,timestamp_ms\n0,0\n1,50\n")
    with pytest.raises(ValueError, match="exposure_us"):
        cm.analyze(Table(p))
    assert cm.main([str(p)]) == 1
