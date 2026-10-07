"""
test_frame_meta.py  --  src/diagnostics/frame_meta.py

libcamerasrc control names as Picamera2's, enums looked up; one frame's row
from libcamera's metadata (the sensor-to-Python time, a clock mismatch
dropped, missing keys blank); the color branch on a real synthetic frame
with a lit red LED in MEASURED's traffic ROI; record() on a fake Picamera2
(the camera set up like the robot's, every frame read and released, frames
saved every Nth, the camera stopped and closed even on Ctrl-C, the files);
the summary, each finding and its threshold; the command line without
Picamera2, with a bad control, and with a busy camera.

--software  A fake camera. No Pi, libcamera or Picamera2.
--hardware  Three seconds of the real camera through Picamera2.
"""
import csv
import json

import cv2
import numpy as np
import pytest

import src.diagnostics.frame_meta as fm
from src.params import FPS, FRAME_H, FRAME_W


def md(n=0, exposure=10000, gain=2.0, ts0=5_000_000_000, period_ns=50_000_000, **kw):
    """libcamera metadata for frame n."""
    d = {"SensorTimestamp": ts0 + n * period_ns, "ExposureTime": exposure, "AnalogueGain": gain,
         "DigitalGain": 1.0, "ColourGains": (1.8, 1.4), "ColourTemperature": 4500, "Lux": 120.0,
         "FrameDuration": 50000, "AeLocked": True}
    d.update(kw)
    return d


# =============================================================================
# Controls and one frame
# =============================================================================

@pytest.mark.software
def test_controls_take_picamera2_names_and_enum_values():
    seen = []

    def lookup(name, value):
        seen.append((name, value))
        return f"<{name}.{value}>"
    out = fm.to_libcamera_controls({"exposure-value": -1.0, "ae-enable": True, "awb-mode": "daylight",
                                    "analogue-gain": 2}, enum_lookup=lookup)
    assert out == {"ExposureValue": -1.0, "AeEnable": True, "AwbMode": "<AwbMode.daylight>", "AnalogueGain": 2}
    assert seen == [("AwbMode", "daylight")]


@pytest.mark.software
def test_the_default_enum_lookup_needs_libcamera(monkeypatch):
    import types
    import sys
    enum = types.SimpleNamespace(Daylight=7, Auto=0)
    fake = types.SimpleNamespace(controls=types.SimpleNamespace(AwbModeEnum=enum))
    monkeypatch.setitem(sys.modules, "libcamera", fake)
    assert fm._libcamera_enum("AwbMode", "day-light") == 7
    with pytest.raises(ValueError, match="no value 'tungsten'"):
        fm._libcamera_enum("AwbMode", "tungsten")
    with pytest.raises(ValueError, match="takes a number"):
        fm._libcamera_enum("ExposureValue", "high")


@pytest.mark.software
def test_a_row_holds_the_metadata_and_the_sensor_to_python_time():
    m = md(n=2)
    row = fm.frame_row(2, m, now_ns=m["SensorTimestamp"] + 38_500_000, t0_ns=md()["SensorTimestamp"],
                       detection={"label": "red", "confidence": 0.8, "white_px": 9})
    assert list(row) == list(fm.FRAME_FIELDS)
    assert (row["t_s"], row["exposure_us"], row["analogue_gain"]) == (0.1, 10000, 2.0)
    assert (row["colour_gain_r"], row["colour_gain_b"], row["ae_locked"]) == (1.8, 1.4, 1)
    assert row["sensor_to_python_ms"] == 38.5
    assert (row["label"], row["confidence"], row["white_px"]) == ("red", 0.8, 9)


@pytest.mark.software
@pytest.mark.parametrize("delta_ms, kept", [(-1.0, None), (0.0, 0.0), (fm.MAX_LATENCY_MS, fm.MAX_LATENCY_MS),
                                            (fm.MAX_LATENCY_MS + 1, None)])
def test_a_sensor_time_off_the_monotonic_clock_is_dropped(delta_ms, kept):
    m = md()
    assert fm.frame_row(0, m, m["SensorTimestamp"] + int(delta_ms * 1e6), None, None)["sensor_to_python_ms"] == kept


@pytest.mark.software
def test_missing_metadata_leaves_blanks():
    row = fm.frame_row(0, {}, 1, None, None)
    assert row["t_s"] is None and row["exposure_us"] is None and row["colour_gain_r"] is None
    assert row["ae_locked"] is None and row["label"] == "" and row["white_px"] is None


def lit_frame(color=(20, 20, 235)):
    """A 480x270 frame with a lit LED (white center, colored ring) in MEASURED's traffic ROI."""
    from src.perception.roi_crop import TRAFFIC
    f = np.full((FRAME_H, FRAME_W, 3), 70, np.uint8)
    c = (int((TRAFFIC.x0 + TRAFFIC.x1) / 2 * FRAME_W), int((TRAFFIC.y0 + TRAFFIC.y1) / 2 * FRAME_H))
    cv2.circle(f, c, 7, color, -1)
    cv2.circle(f, c, 3, (255, 255, 255), -1)
    return f


@pytest.mark.software
def test_detect_runs_the_robots_color_branch_on_the_traffic_roi():
    from src.config import MEASURED
    det = fm.detect(lit_frame(), 0, 0, MEASURED)
    assert det["label"] == "red" and det["confidence"] > 0 and det["white_px"] > 0
    assert fm.detect(np.full((FRAME_H, FRAME_W, 3), 70, np.uint8), 1, 50, MEASURED) == \
        {"label": "", "confidence": None, "white_px": 0}


# =============================================================================
# record()
# =============================================================================

class Request:
    def __init__(self, cam, n):
        self.cam, self.n = cam, n

    def make_array(self, name):
        assert name == "main"
        return np.full((FRAME_H, FRAME_W, 3), self.n % 256, np.uint8)

    def get_metadata(self):
        return self.cam.metas[self.n % len(self.cam.metas)]

    def release(self):
        self.cam.released += 1


class Camera:
    """Picamera2's interface: create_video_configuration, configure, start, capture_request, stop, close."""
    camera_properties = {"Model": "imx290", "PixelArraySize": (1945, 1097)}

    def __init__(self, metas=None, interrupt_at=None):
        self.metas = metas or [md(n) for n in range(40)]
        self.interrupt_at, self.n, self.released = interrupt_at, 0, 0
        self.config = self.configured = None
        self.started = self.stopped = self.closed = False

    def create_video_configuration(self, **kw):
        self.config = kw
        return {"cfg": kw}

    def configure(self, cfg):
        self.configured = cfg

    def start(self):
        self.started = True

    def capture_request(self):
        if self.interrupt_at is not None and self.n == self.interrupt_at:
            raise KeyboardInterrupt
        r = Request(self, self.n)
        self.n += 1
        return r

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Clock:
    """A monotonic_ns that advances step_ns every read: record() reads it twice a frame (the check, then the frame's time)."""
    def __init__(self, step_ns=25_000_000):
        self.t, self.step = 0, step_ns

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.mark.software
def test_the_camera_is_set_up_like_the_robots():
    cam = Camera()
    assert fm.open_camera({"ExposureValue": -1.0}, factory=lambda: cam) is cam
    main, raw, controls = cam.config["main"], cam.config["raw"], cam.config["controls"]
    assert main == {"size": (FRAME_W, FRAME_H), "format": "RGB888"} and raw == {"size": (1920, 1080)}
    assert controls == {"FrameRate": float(FPS), "ExposureValue": -1.0}
    assert cam.configured == {"cfg": cam.config} and cam.started


@pytest.mark.software
def test_record_reads_every_frame_writes_the_files_and_saves_every_nth(tmp_path):
    cam = Camera()
    dets = []
    rows = fm.record(tmp_path / "m", seconds=10.0, controls={"ExposureValue": -1.0}, save_every=4,
                     detect_fn=lambda f, n, ts: dets.append((int(f[0, 0, 0]), n, ts)) or {"label": "red"},
                     camera=cam, clock_ns=Clock(), max_frames=10)
    assert len(rows) == 10 and cam.released == 10 and cam.stopped and cam.closed
    assert [d[1] for d in dets] == list(range(10)) and dets[3][0] == 3          # each frame's own image
    assert rows[0]["t_s"] == 0.0 and rows[9]["t_s"] == 0.45
    assert sorted(p.name for p in (tmp_path / "m" / "frames").iterdir()) == ["000000.png", "000004.png", "000008.png"]
    with open(tmp_path / "m" / "frames.csv") as f:
        assert [r["label"] for r in csv.DictReader(f)] == ["red"] * 10
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["controls"] == {"ExposureValue": "-1.0"} and meta["camera_properties"]["Model"] == "imx290"
    assert meta["summary"]["frames"] == 10 and meta["save_every"] == 4 and meta["detect"]
    assert "10 frames at 20.0 fps" in (tmp_path / "m" / "summary.txt").read_text()


@pytest.mark.software
def test_record_stops_at_the_duration_and_on_ctrl_c_still_closes_and_writes(tmp_path):
    cam = Camera()
    rows = fm.record(tmp_path / "a", seconds=0.2, camera=cam, clock_ns=Clock(50_000_000))
    assert len(rows) == 2 and not (tmp_path / "a" / "frames").exists()          # two 50 ms reads a frame
    cam = Camera(interrupt_at=5)
    rows = fm.record(tmp_path / "b", seconds=10.0, camera=cam, clock_ns=Clock())
    assert len(rows) == 5 and cam.stopped and cam.closed
    assert json.loads((tmp_path / "b" / "meta.json").read_text())["interrupted"]


# =============================================================================
# Summary and findings
# =============================================================================

def rows_of(*groups):
    """Rows from (count, label, exposure, gain) groups, one frame per 50 ms."""
    rows, n = [], 0
    for count, label, exposure, gain in groups:
        for _ in range(count):
            m = md(n, exposure=exposure, gain=gain)
            rows.append(fm.frame_row(n, m, m["SensorTimestamp"] + 30_000_000, md()["SensorTimestamp"],
                                     {"label": label, "confidence": 0.6, "white_px": 5}))
            n += 1
    return rows


@pytest.mark.software
def test_summary_ranges_fps_and_per_label_light():
    s = fm.summarize(rows_of((10, "red", 10000, 2.0), (5, "yellow", 20000, 2.0), (5, "", 10000, 1.0)))
    assert s["frames"] == 20 and s["fps"] == 20.0 and s["ae_locked_share"] == 1.0
    assert s["fields"]["exposure_us"] == {"min": 10000.0, "median": 10000.0, "p95": 20000.0, "max": 20000.0}
    assert s["fields"]["sensor_to_python_ms"]["median"] == 30.0
    assert set(s["labels"]) == {"red", "yellow", "none"}
    assert s["labels"]["red"]["light"]["median"] == 20000.0 and s["labels"]["yellow"]["light"]["median"] == 40000.0
    text = "\n".join(fm.summary_lines(s, {"controls": {"ExposureValue": -1.0}, "started": "t"}))
    assert "exposure us          10000 / 10000 / 20000" in text and "controls: ExposureValue=-1.0" in text
    assert "  red         10      20000     10000    2.00      5" in text


@pytest.mark.software
def test_yellow_with_more_light_than_red_points_at_exposure_and_without_at_the_angle():
    more = fm.summarize(rows_of((10, "red", 10000, 2.0), (10, "yellow", 10000, 2.0 * fm.LIGHT_GAP * 1.1)))
    assert any("yellow frames got 1.3x the light of red ones" in f and "exposure-value=-1" in f
               for f in more["findings"]), more["findings"]
    same = fm.summarize(rows_of((10, "red", 10000, 2.0), (10, "yellow", 10000, 2.0 * fm.LIGHT_GAP)))
    assert any("about the same light" in f and "the angle" in f for f in same["findings"]), same["findings"]
    alone = fm.summarize(rows_of((10, "red", 10000, 2.0)))
    assert alone["findings"] == []


@pytest.mark.software
def test_an_agc_swing_is_a_finding_at_its_threshold():
    swung = fm.summarize(rows_of((5, "", 10000, 1.0), (5, "", 10000, fm.EXPOSURE_SWING * 1.01)))
    assert any("AGC moved the light gathered (exposure x gain) 2.0x" in f for f in swung["findings"])
    held = fm.summarize(rows_of((5, "", 10000, 1.0), (5, "", 10000, fm.EXPOSURE_SWING)))
    assert held["findings"] == []


@pytest.mark.software
def test_late_frames_unsettled_ae_and_a_slow_sensor_are_findings():
    rows = []
    for n in range(10):
        m = md(n, period_ns=100_000_000, AeLocked=n < 3)
        rows.append(fm.frame_row(n, m, m["SensorTimestamp"] + 80_000_000, md()["SensorTimestamp"], None))
    found = fm.summarize(rows)["findings"]
    assert any("reach Python 80 ms" in f for f in found)
    assert any("AE reported settled on only 30%" in f for f in found)
    assert any("delivered 10.0 fps" in f for f in found)


@pytest.mark.software
def test_an_empty_recording_summarizes_to_blanks():
    s = fm.summarize([])
    assert s["frames"] == 0 and s["fps"] is None and s["light"] is None and s["findings"] == []
    assert "exposure us          --" in "\n".join(fm.summary_lines(s))


# =============================================================================
# The command line
# =============================================================================

@pytest.mark.software
def test_cli_without_picamera2_says_how_to_get_it(monkeypatch, capsys):
    def missing(controls):
        raise ImportError("No module named 'picamera2'")
    monkeypatch.setattr(fm, "open_camera", missing)
    assert fm.cli(["--no-detect"]) == 2
    assert "python3-picamera2" in capsys.readouterr().out


@pytest.mark.software
def test_cli_rejects_a_bad_control_and_reports_a_busy_camera(monkeypatch, capsys):
    assert fm.cli(["--camera-control", "nonsense=1"]) == 2
    assert "unknown camera control" in capsys.readouterr().out

    def busy(controls):
        raise RuntimeError("Failed to acquire camera: Device or resource busy")
    monkeypatch.setattr(fm, "open_camera", busy)
    assert fm.cli(["--no-detect"]) == 2
    assert "is the robot's pipeline still running" in capsys.readouterr().out


@pytest.mark.software
def test_cli_records_with_the_configured_controls_plus_the_given_ones(monkeypatch, tmp_path, capsys):
    seen = {}
    monkeypatch.setattr(fm, "CAMERA_CONTROLS", {"exposure-value": -0.5})
    monkeypatch.setattr(fm, "open_camera", lambda controls: seen.update(controls=controls) or Camera())
    monkeypatch.setattr(fm.time, "monotonic_ns", Clock(400_000_000))
    out = tmp_path / "m"
    assert fm.cli(["--seconds", "1", "--camera-control", "ae-enable=true", "--out", str(out)]) == 0
    assert seen["controls"] == {"ExposureValue": -0.5, "AeEnable": True}
    assert (out / "frames.csv").exists() and "frame metadata" in capsys.readouterr().out


# =============================================================================
# On the Pi
# =============================================================================

@pytest.mark.hardware
def test_frame_meta_on_the_pi(artifacts):
    pytest.importorskip("picamera2")
    rows = fm.record(artifacts.path / "frame_meta", seconds=3.0, controls={},
                     camera=fm.open_camera({}), save_every=20)
    assert rows, "no frame"
    assert rows[-1]["exposure_us"] is not None and rows[-1]["sensor_to_python_ms"] is not None
