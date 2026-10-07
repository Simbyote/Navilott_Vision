"""
test_capture_anatomy.py  --  src/diagnostics/capture_anatomy.py

Each parser on output shaped like the tool's own (media-ctl -p for Unicam
and the ISP, gst-launch -v caps lines, the latency tracer's log, vcgencmd,
/proc/meminfo, libcamera's log); the latency statistics; record() end to
end with fake tools that write what the real ones would; the findings and
summary; a laptop with no tools at all; the command line.

--software  Canned tool output and fake runners. No Pi, camera or GStreamer.
--hardware  The real thing for 3 s on the Pi (the camera must be free).
"""
import csv
import json
import shutil
from pathlib import Path

import pytest

import src.diagnostics.capture_anatomy as ca
from src.params import FPS

MEDIA_UNICAM = """\
Media controller API version 6.1.21

Media device information
------------------------
driver          unicam
model           unicam
serial
bus info        platform:3f801000.csi

Device topology
- entity 1: imx290 10-001a (1 pad, 1 link, 0 routes)
            type V4L2 subdev subtype Sensor flags 0
            device node name /dev/v4l-subdev0
        pad0: SOURCE
                [stream:0 fmt:SRGGB10_1X10/1920x1080 field:none colorspace:raw]
                -> "unicam-image":0 [ENABLED,IMMUTABLE]

- entity 2: unicam-image (1 pad, 1 link)
            type Node subtype V4L flags 1
            device node name /dev/video0
        pad0: Sink
                <- "imx290 10-001a":0 [ENABLED,IMMUTABLE]

- entity 3: unicam-embedded (1 pad, 0 link)
            type Node subtype V4L flags 0
            device node name /dev/video1
        pad0: Sink
"""

MEDIA_ISP = """\
Media device information
------------------------
driver          bcm2835-isp

Device topology
- entity 1: bcm2835_isp0 (4 pads, 4 links)
            type Node subtype Unknown flags 0
        pad1: Source
                -> "bcm2835-isp0-capture1":0 [ENABLED,IMMUTABLE]
        pad2: Source
                -> "bcm2835-isp0-capture2":0 []
- entity 6: bcm2835-isp0-output0 (1 pad, 1 link)
            type Node subtype V4L flags 0
            device node name /dev/video13
        pad0: Source
                -> "bcm2835_isp0":0 [ENABLED,IMMUTABLE]
- entity 10: bcm2835-isp0-capture1 (1 pad, 1 link)
            type Node subtype V4L flags 0
            device node name /dev/video14
"""

GST_V = """\
Setting pipeline to PAUSED ...
Pipeline is live and does not need PREROLL ...
/GstPipeline:pipeline0/GstLibcameraSrc:libcamerasrc0.GstLibcameraPad:src: caps = video/x-raw, format=(string)NV12, width=(int)480, height=(int)270, colorimetry=(string)bt709, framerate=(fraction)20/1
/GstPipeline:pipeline0/GstCapsFilter:capsfilter0.GstPad:src: caps = video/x-raw, format=(string)NV12, width=(int)480, height=(int)270, framerate=(fraction)20/1
/GstPipeline:pipeline0/GstVideoConvert:videoconvert0.GstPad:src: caps = video/x-raw, format=(string)YUY2, width=(int)480, height=(int)270, framerate=(fraction)20/1
/GstPipeline:pipeline0/GstVideoConvert:videoconvert0.GstPad:src: caps = video/x-raw, format=(string)BGR, width=(int)480, height=(int)270, framerate=(fraction)20/1
/GstPipeline:pipeline0/GstVideoFlip:videoflip0.GstPad:src: caps = video/x-raw, format=(string)BGR, width=(int)480, height=(int)270, framerate=(fraction)20/1
/GstPipeline:pipeline0/GstFakeSink:fakesink0.GstPad:sink: caps = video/x-raw, format=(string)BGR, width=(int)480, height=(int)270, framerate=(fraction)20/1
Setting pipeline to PLAYING ...
"""


def tracer_log(frames=21, conv_ms=3.0, flip_ms=1.0, pipe_ms=6.0, period_ms=50.0):
    """A latency tracer log: element-latency for videoconvert0 and videoflip0 and the pipeline latency per frame."""
    pre = "0:00:01.000000000  4242 0x55aa TRACE           GST_TRACER :0:: "
    lines = []
    for i in range(frames):
        ts = int((1000 + i * period_ms) * 1e6)
        for el, ms in (("videoconvert0", conv_ms + 0.1 * (i % 3)), ("videoflip0", flip_ms)):
            lines.append(pre + f"element-latency, element-id=(string)0x55ab, element=(string){el}, "
                               f"src=(string)src, time=(guint64){int(ms * 1e6)}, ts=(guint64){ts};")
        lines.append(pre + "latency, src-element-id=(string)0x55ac, src-element=(string)libcamerasrc0, "
                           "src=(string)src, sink-element-id=(string)0x55ad, sink-element=(string)fakesink0, "
                           f"sink=(string)sink, time=(guint64){int(pipe_ms * 1e6)}, ts=(guint64){ts};")
    return "\n".join(lines) + "\n"


LIBCAMERA_LOG = """\
[0:00:00.1] [4242]  INFO Camera camera_manager.cpp:325 libcamera v0.7.2+rpt20250903
[0:00:00.2] [4242]  INFO RPI vc4.cpp:448 Sensor: /base/soc/i2c0mux/i2c@1/imx290@1a - Selected sensor format: 1920x1080-SRGGB10_1X10 - Selected unicam format: 1920x1080-pRAA
[0:00:00.3] [4242]  INFO Camera camera.cpp:1197 configuring streams: (0) 480x270-NV12
[0:00:00.2] [4242]  INFO RPI vc4.cpp:448 Sensor: /base/soc/i2c0mux/i2c@1/imx290@1a - Selected sensor format: 1920x1080-SRGGB10_1X10 - Selected unicam format: 1920x1080-pRAA
"""

VC = {("vcgencmd", "version"): "Aug 23 2026\nversion abc (clean)",
      ("vcgencmd", "measure_clock", "arm"): "frequency(48)=1000000000",
      ("vcgencmd", "measure_clock", "core"): "frequency(1)=400000000",
      ("vcgencmd", "measure_clock", "isp"): "frequency(45)=300000000",
      ("vcgencmd", "measure_clock", "v3d"): "frequency(46)=300000000",
      ("vcgencmd", "measure_clock", "h264"): "frequency(28)=0",
      ("vcgencmd", "measure_volts", "core"): "volt=1.2000V",
      ("vcgencmd", "get_mem", "arm"): "arm=448M",
      ("vcgencmd", "get_mem", "gpu"): "gpu=64M"}

MEMINFO = "MemTotal:  427000 kB\nCmaTotal:  262144 kB\nCmaFree:   200704 kB\n"


# =============================================================================
# Parsers
# =============================================================================

@pytest.mark.software
def test_media_graph_keeps_entities_nodes_formats_and_only_enabled_links():
    m = ca.parse_media(MEDIA_UNICAM)
    assert m["driver"] == "unicam"
    assert [e["name"] for e in m["entities"]] == ["imx290 10-001a", "unicam-image", "unicam-embedded"]
    sensor = m["entities"][0]
    assert sensor["node"] == "/dev/v4l-subdev0" and sensor["fmt"] == "SRGGB10_1X10 1920x1080"
    assert m["links"] == [("imx290 10-001a", "unicam-image")]                       # "<-" is the same link, seen from its end
    isp = ca.parse_media(MEDIA_ISP)
    assert isp["driver"] == "bcm2835-isp"
    assert ("bcm2835_isp0", "bcm2835-isp0-capture2") not in isp["links"]           # not ENABLED
    assert ("bcm2835-isp0-output0", "bcm2835_isp0") in isp["links"]
    assert ca.parse_media("") == {"driver": None, "entities": [], "links": []}
    two = ca.parse_media("- entity 4: scaler (2 pads, 1 link)\n  pad0: Sink\n    [fmt:SRGGB10_1X10/1920x1080]\n"
                         "  pad1: Source\n    [fmt:YUYV8_1X16/480x270]\n")
    assert two["entities"][0]["fmt"] == "SRGGB10_1X10 1920x1080"                    # the first pad's format


@pytest.mark.software
def test_caps_one_row_per_pad_the_last_it_settled_on_in_first_seen_order():
    caps = ca.parse_caps(GST_V)
    assert [(c["element"], c["pad"]) for c in caps] == [
        ("libcamerasrc0", "src"), ("capsfilter0", "src"), ("videoconvert0", "src"), ("videoflip0", "src"),
        ("fakesink0", "sink")]
    src = caps[0]
    assert (src["format"], src["size"], src["framerate"], src["caps"]) == ("NV12", "480x270", "20/1", "video/x-raw")
    assert caps[2]["format"] == "BGR"                                               # renegotiated: the last wins


@pytest.mark.software
def test_tracer_rows_per_element_and_for_the_pipeline_in_ms():
    rows = ca.parse_tracer(tracer_log(frames=2))
    assert rows[0] == {"ts_ms": 1000.0, "kind": "element", "element": "videoconvert0", "ms": 3.0}
    assert rows[2] == {"ts_ms": 1000.0, "kind": "pipeline", "element": "libcamerasrc0 -> fakesink0", "ms": 6.0}
    assert len(rows) == 6
    assert ca.parse_tracer("latency, time=(guint64)abc\nnothing here\n") == []


@pytest.mark.software
def test_latency_stats_median_p95_max_and_fps_over_the_run():
    stats = {s["element"]: s for s in ca.latency_stats(ca.parse_tracer(tracer_log(frames=21)))}
    conv = stats["videoconvert0"]
    assert (conv["frames"], conv["median_ms"], conv["max_ms"]) == (21, 3.1, 3.2)
    assert conv["p95_ms"] == 3.2 and conv["fps"] == 20.0                        # 20 intervals over 1 s
    pipe = stats["libcamerasrc0 -> fakesink0"]
    assert pipe["kind"] == "pipeline" and pipe["median_ms"] == 6.0
    spread = ca.latency_stats([{"ts_ms": i * 50.0, "kind": "element", "element": "x", "ms": float(i)} for i in range(21)])
    assert (spread[0]["median_ms"], spread[0]["p95_ms"], spread[0]["max_ms"]) == (10.0, 19.0, 20.0)
    one = ca.latency_stats([{"ts_ms": 0.0, "kind": "element", "element": "x", "ms": 1.0}])
    assert one[0]["fps"] is None


@pytest.mark.software
@pytest.mark.parametrize("text, value", [("frequency(45)=300000000", 300000000.0), ("volt=1.2000V", 1.2),
                                         ("gpu=64M", 64.0), ("error", None), (None, None)])
def test_vcgencmd_reply_is_the_number_after_the_equals(text, value):
    assert ca.parse_vcgencmd(text) == value


@pytest.mark.software
def test_cma_from_meminfo_in_mb_and_none_without_it(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text(MEMINFO)
    assert ca.read_cma_mb(p) == {"cma_total_mb": 256.0, "cma_free_mb": 196.0}
    p.write_text("MemTotal: 1 kB\n")
    assert ca.read_cma_mb(p) == {"cma_total_mb": None, "cma_free_mb": None}
    assert ca.read_cma_mb(tmp_path / "missing") == {"cma_total_mb": None, "cma_free_mb": None}


@pytest.mark.software
def test_libcamera_keeps_the_mode_selection_once_without_the_log_prefix():
    sel = ca.libcamera_selected(LIBCAMERA_LOG)
    assert sel == ["libcamera v0.7.2+rpt20250903",
                   "Sensor: /base/soc/i2c0mux/i2c@1/imx290@1a - Selected sensor format: 1920x1080-SRGGB10_1X10 "
                   "- Selected unicam format: 1920x1080-pRAA",
                   "configuring streams: (0) 480x270-NV12"]
    assert ca.libcamera_selected(None) == []


@pytest.mark.software
def test_the_pipeline_is_the_robots_with_a_silent_fakesink():
    pipe = ca.anatomy_pipeline({"exposure-value": -1})
    assert pipe.startswith("libcamerasrc ") and "exposure-value=-1" in pipe
    assert pipe.endswith(" ! videoflip method=rotate-180 ! video/x-raw,format=BGR ! fakesink sync=false silent=true")
    assert "appsink" not in pipe


# =============================================================================
# record()
# =============================================================================

class FakePi:
    """The Pi's tools: run() answers from canned output; launch() writes what gst-launch would."""
    def __init__(self, gst_v=GST_V, trace=None, dot=True, graphviz=True):
        self.gst_v, self.trace, self.dot, self.graphviz = gst_v, trace if trace is not None else tracer_log(), dot, graphviz
        self.calls, self.launched = [], None

    def run(self, argv):
        self.calls.append(argv)
        key = tuple(argv)
        if key in VC:
            return 0, VC[key]
        if argv[0] == "media-ctl":
            text = MEDIA_UNICAM if argv[2].endswith("media0") else MEDIA_ISP
            return 0, (f"digraph board {{ \"{argv[2]}\" }}" if argv[3] == "--print-dot" else text)
        if argv[0] == "dot":
            if not self.graphviz:
                return None, "not available: dot is not installed"
            Path(argv[4]).write_bytes(b"png")
            return 0, ""
        return 0, f"{argv[0]} output"

    def launch(self, argv, env, duration_s):
        self.launched = (argv, env, duration_s)
        Path(env["GST_DEBUG_FILE"]).write_text(self.trace)
        Path(env["LIBCAMERA_LOG_FILE"]).write_text(LIBCAMERA_LOG)
        if self.dot:
            d = Path(env["GST_DEBUG_DUMP_DOT_DIR"])
            (d / "0.00.00.1-gst-launch.NULL_READY.dot").write_text("digraph a {}")
            (d / "0.00.00.5-gst-launch.PAUSED_PLAYING.dot").write_text("digraph b {}")
        return 0, self.gst_v


def media_devs(tmp_path, n=2):
    d = tmp_path / "dev"
    d.mkdir()
    for i in range(n):
        (d / f"media{i}").write_text("")
    return str(d / "media*")


def meminfo(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text(MEMINFO)
    return p


@pytest.mark.software
def test_record_runs_every_probe_and_the_pipeline_with_the_tracer_and_logs_on(tmp_path):
    pi = FakePi()
    out = tmp_path / "anatomy"
    res = ca.record(out, 7.0, {"exposure-value": -1}, run=pi.run, launch=pi.launch,
                    media_glob=media_devs(tmp_path), meminfo=meminfo(tmp_path))
    argv, env, dur = pi.launched
    assert argv[:3] == ["gst-launch-1.0", "-v", "-e"] and argv[3] == res["pipeline"] and dur == 7.0
    assert "exposure-value=-1" in argv[3]
    assert env["GST_TRACERS"] == ca.GST_TRACERS and env["GST_DEBUG"] == "GST_TRACER:7"
    assert env["LIBCAMERA_LOG_LEVELS"] == ca.LIBCAMERA_LOG_LEVELS
    assert ["vcgencmd", "measure_clock", "isp"] in pi.calls and ["v4l2-ctl", "--list-devices"] in pi.calls
    assert res["vc"]["clock_isp"] == 3e8 and res["vc"]["mem_gpu"] == 64.0 and res["vc"]["cma_free_mb"] == 196.0
    assert "clock_version" not in res["vc"] and "version" not in res["vc"]
    assert [m["driver"] for m in res["media"]] == ["unicam", "bcm2835-isp"]
    assert res["caps"][0]["format"] == "NV12" and res["libcamera"]
    assert {s["element"] for s in res["latency"]} == {"videoconvert0", "videoflip0", "libcamerasrc0 -> fakesink0"}
    for name in ("summary.txt", "anatomy.json", "latency.csv", "media0.txt", "media1.dot", "gst_launch.txt",
                 "gst_pipeline.dot", "vc_clock_isp.txt", "uname.txt"):
        assert (out / name).exists(), name
    assert (out / "gst_pipeline.dot").read_text() == "digraph b {}"                 # the PAUSED_PLAYING dump
    assert res["pngs"] == ["media0.png", "media1.png", "gst_pipeline.png"]
    assert json.loads((out / "anatomy.json").read_text())["caps"] == res["caps"]
    with open(out / "latency.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["element"] == "videoconvert0" and len(rows) == 63


@pytest.mark.software
def test_videoconvert_and_videoflip_over_the_threshold_earn_a_finding_naming_the_isp_format(tmp_path):
    pi = FakePi(trace=tracer_log(conv_ms=3.0, flip_ms=1.0))
    res = ca.record(tmp_path / "a", 1.0, run=pi.run, launch=pi.launch, media_glob=media_devs(tmp_path),
                    meminfo=meminfo(tmp_path))
    (f,) = [x for x in res["findings"] if "videoconvert" in x]
    assert "take 4.1 ms of every frame from NV12" in f and f"{100 * 4.1 / (1000 / FPS):.0f}%" in f
    quick = FakePi(trace=tracer_log(conv_ms=0.5, flip_ms=0.5))
    res = ca.record(tmp_path / "b", 1.0, run=quick.run, launch=quick.launch, media_glob=str(tmp_path / "none*"),
                    meminfo=meminfo(tmp_path))
    assert not any("videoconvert" in x for x in res["findings"])
    assert res["media"] == []


@pytest.mark.software
def test_a_slow_pipeline_and_a_full_cma_pool_are_findings(tmp_path):
    pi = FakePi(trace=tracer_log(period_ms=100.0, conv_ms=0.1, flip_ms=0.1))
    mi = tmp_path / "meminfo"
    mi.write_text("CmaTotal: 262144 kB\nCmaFree: 10240 kB\n")
    res = ca.record(tmp_path / "a", 1.0, run=pi.run, launch=pi.launch, media_glob=media_devs(tmp_path), meminfo=mi)
    assert any("delivered 10.0 fps" in x for x in res["findings"])
    assert any("CMA nearly full (10 of 256 MB free)" in x for x in res["findings"])


@pytest.mark.software
def test_no_frames_says_the_camera_may_be_busy(tmp_path):
    pi = FakePi(trace="", dot=False)
    res = ca.record(tmp_path / "a", 1.0, run=pi.run, launch=pi.launch, media_glob=media_devs(tmp_path),
                    meminfo=meminfo(tmp_path))
    assert res["findings"] == ["no frame reached the sink: is the camera open in another process (the robot's "
                               "pipeline)? See gst_launch.txt"]
    assert "gst_pipeline.png" not in res["pngs"]


@pytest.mark.software
def test_a_laptop_without_any_tool_records_what_it_can_and_says_so(tmp_path):
    def run(argv):
        return None, f"not available: {argv[0]} is not installed"

    def launch(argv, env, duration_s):
        return None, "not available: gst-launch-1.0 is not installed"
    res = ca.record(tmp_path / "a", 1.0, run=run, launch=launch, media_glob=media_devs(tmp_path),
                    meminfo=tmp_path / "missing")
    assert res["missing"] == ["uname", "v4l2-ctl", "rpicam-hello", "gst-inspect-1.0", "vcgencmd", "media-ctl",
                              "gst-launch-1.0"]
    assert res["findings"] == ["no GStreamer here: only the static probes ran"]
    assert res["vc"] == {"cma_total_mb": None, "cma_free_mb": None} and res["media"] == []
    text = (tmp_path / "a" / "summary.txt").read_text()
    assert "not available here: uname, v4l2-ctl" in text and "nothing recorded" in text


@pytest.mark.software
def test_without_graphviz_the_dot_files_stay_undrawn(tmp_path):
    pi = FakePi(graphviz=False)
    res = ca.record(tmp_path / "a", 1.0, run=pi.run, launch=pi.launch, media_glob=media_devs(tmp_path),
                    meminfo=meminfo(tmp_path))
    assert res["pngs"] == [] and (tmp_path / "a" / "media0.dot").exists()


@pytest.mark.software
def test_summary_shows_the_route_caps_times_videocore_and_libcamera(tmp_path):
    pi = FakePi()
    res = ca.record(tmp_path / "a", 1.0, run=pi.run, launch=pi.launch, media_glob=media_devs(tmp_path),
                    meminfo=meminfo(tmp_path))
    text = "\n".join(ca.summary_lines(res))
    assert "imx290 10-001a [SRGGB10_1X10 1920x1080] [/dev/v4l-subdev0] -> unicam-image [/dev/video0]" in text
    assert "libcamerasrc0.src" in text and "NV12" in text
    assert text.index("pipeline libcamerasrc0 -> fakesink0") < text.index("videoconvert0 ")   # the pipeline first
    assert "clocks MHz: arm 1000  core 400  isp 300  v3d 300" in text and "h264" not in text.split("clocks")[1].split("\n")[0]
    assert "core 1.200 V   memory arm 448 MB, gpu 64 MB   CMA 196 of 256 MB free" in text
    assert "Selected sensor format: 1920x1080-SRGGB10_1X10" in text
    assert "drawn: media0.png, media1.png, gst_pipeline.png" in text


# =============================================================================
# Running tools and the command line
# =============================================================================

@pytest.mark.software
def test_run_tool_and_launch_for_report_a_missing_tool_and_capture_output():
    assert ca.run_tool(["no-such-tool-xyz"])[0] is None
    assert ca.run_tool(["sh", "-c", "echo hi; echo err >&2; exit 3"]) == (3, "hi\nerr\n")
    assert ca.run_tool(["sleep", "5"], timeout=0.1) == (None, "timed out after 0 s")
    assert ca.launch_for(["no-such-tool-xyz"], {}, 0.1)[0] is None
    code, out = ca.launch_for(["sh", "-c", "echo up; exec sleep 5"], {"PATH": "/usr/bin:/bin"}, 0.3)
    assert out == "up\n" and code != 0                                              # Ctrl-C'd
    assert ca.launch_for(["sh", "-c", "echo done"], {"PATH": "/usr/bin:/bin"}, 5.0) == (0, "done\n")


@pytest.mark.software
def test_cli_adds_camera_controls_to_the_configured_ones_and_rejects_unknown(monkeypatch, tmp_path, capsys):
    seen = {}
    monkeypatch.setattr(ca, "CAMERA_CONTROLS", {"awb-mode": "daylight"})
    monkeypatch.setattr(ca, "record", lambda out, s, controls: seen.update(out=out, s=s, c=controls) or
                        {"started": "t", "duration_s": s, "missing": [], "vc": {}, "media": [], "caps": [],
                         "latency": [], "libcamera": [], "pngs": [], "findings": []})
    assert ca.cli(["--seconds", "3", "--camera-control", "exposure-value=-1", "--out", str(tmp_path)]) == 0
    assert seen == {"out": str(tmp_path), "s": 3.0, "c": {"awb-mode": "daylight", "exposure-value": -1}}
    assert ca.cli(["--out", str(tmp_path)]) == 0 and seen["c"] is None
    assert ca.cli(["--camera-control", "nonsense=1"]) == 2
    assert "unknown camera control" in capsys.readouterr().out


# =============================================================================
# On the Pi
# =============================================================================

@pytest.mark.hardware
def test_capture_anatomy_on_the_pi(artifacts):
    if not shutil.which("gst-launch-1.0") or not Path("/dev/media0").exists():
        pytest.skip("no GStreamer or no media device: not the robot")
    res = ca.record(artifacts.path / "anatomy", 3.0)                 # summary.txt and the rest land in there
    assert res["media"], "media-ctl found no media device"
    assert any(s["kind"] == "pipeline" for s in res["latency"]), res["findings"]
