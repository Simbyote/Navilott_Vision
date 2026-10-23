"""Capture anatomy: how a frame gets from the IMX290 to OpenCV on this Pi, recorded once.

Purpose:
    A frame crosses the sensor, Unicam (the CSI-2 receiver), the VideoCore
    ISP and libcamera's IPA before GStreamer hands it to OpenCV, and none of
    it is visible from Python. This records what the OS can show: the
    hardware graph, the caps negotiated at every pad, how long each element
    holds a frame, libcamera's log, the VideoCore's clocks and memory
    (docs/guides/diagnostics.md, section 7). The camera must be free.

Main package:
    The run folder: anatomy.json (everything parsed), latency.csv and
    summary.txt (findings, such as any CPU-side conversion and its cost).

Flow:
    1. Static probes: uname, v4l2-ctl, rpicam-hello, gst-inspect, vcgencmd, media-ctl.
    2. The pipeline under gst-launch -v: the caps (bisected if it stalls).
    3. The pipeline under pad_timing.py: the time per element.
    4. Parse; write the folder.
"""
import argparse
import csv
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.capture.camera import build_gst_pipeline, parse_controls
from src.diagnostics.os_counters import parse_vcgencmd, read_cma_mb
from src.params import CAMERA_CONTROLS, FPS, RUNS_DIR

DURATION_S = 10.0           # long enough for AGC to settle and a few hundred frames at 20 FPS
STOP_WAIT_S = 5.0           # after Ctrl-C, how long gst-launch gets to send EOS and exit
# The sink takes duration_s x FPS buffers and ends the run itself (EOS); this
# is the extra time allowed before it's interrupted. Ending on Ctrl-C lost
# gst-launch's buffered output and left no way to tell a camera that never
# started from one that ran (2026-10-12: by hand, 100 frames in 5.2 s)
STARTUP_S = 10.0
INTERRUPTED = "[capture_anatomy] interrupted:"   # appended to gst_launch.txt when the run had to be stopped
# The probe timer, standalone (stdlib + gi) so the system python3 can run it
PAD_TIMING = Path(__file__).with_name("pad_timing.py")
PAD_TAG = "[pad-timing]"            # pad_timing.TAG: it's kept free of src imports, so it's repeated here
SYSTEM_PYTHON = "/usr/bin/python3"  # has python3-gi on Raspberry Pi OS when the venv doesn't
LIBCAMERA_LOG_LEVELS = "*:INFO"     # INFO holds the mode selection; DEBUG logs every frame
FRAME_BUDGET_MS = 1000.0 / FPS
# videoconvert + videoflip together over this many ms of each frame earns a finding
CONVERT_FINDING_MS = 2.0
VC_CLOCKS = ("arm", "core", "isp", "v3d", "h264")
MEMINFO_PATH = Path("/proc/meminfo")
SINK = "fakesink sync=false silent=true"     # silent: -v would print every buffer
LATENCY_FIELDS = ("ts_ms", "kind", "element", "ms")

_CLI_HELP = """\
Record how the camera pipeline routes a frame on this Pi: the media graph
(sensor -> Unicam -> ISP), the caps negotiated at every GStreamer pad, how
long each element takes per frame, libcamera's mode selection, and the
VideoCore clocks and memory. Stop the robot's own pipeline first: the camera
opens in one process at a time.

Examples (from vision_stack/):
    python3 -m src.diagnostics.capture_anatomy
    python3 -m src.diagnostics.capture_anatomy --seconds 20 --camera-control exposure-value=-1

Output (--out DIR, default <root>/runs/anatomy_<timestamp>):
    summary.txt        the route, the caps, time per element, the VideoCore side, findings
    anatomy.json       everything parsed
    latency.csv        one row per frame per element (and the whole pipeline, and sensor -> first pad)
    media<N>.txt/.dot/.png, gst_*.dot/.png, gst_launch.txt, pad_timing.txt,
    libcamera.log, and each tool's raw output
"""


def run_tool(argv: list[str], timeout: float = 10.0) -> tuple[int | None, str]:
    """
    (exit code, stdout + stderr) of a command; (None, why) when it can't run.
    """
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"not available: {argv[0]} is not installed"
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:.0f} s"
    return p.returncode, p.stdout + p.stderr


def launch_for(argv: list[str], env: dict, duration_s: float) -> tuple[int | None, str]:
    """
    Run argv until it ends or duration_s passes, then Ctrl-C it (then kill
    it); (exit code, stdout + stderr, with an INTERRUPTED line when it had
    to be stopped). (None, why) when it can't start.
    """
    try:
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        return None, f"not available: {argv[0]} is not installed"
    try:
        out, _ = p.communicate(timeout=duration_s)
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGINT)
        how = "Ctrl-C"
        try:
            out, _ = p.communicate(timeout=STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            how = "killed"
        out = f"{out}\n{INTERRUPTED} {how} after {duration_s:.0f} s\n"
    return p.returncode, out


def anatomy_pipeline(controls: dict | None = None, num_buffers: int | None = None) -> str:
    """The robot's capture pipeline string with its appsink swapped for SINK, ending itself after num_buffers."""
    sink = SINK if num_buffers is None else f"{SINK} num-buffers={int(num_buffers)}"
    return build_gst_pipeline(controls=controls).rsplit(" ! ", 1)[0] + " ! " + sink


def gi_python(run) -> str | None:
    """The first of this interpreter and SYSTEM_PYTHON that imports Gst 1.0 through gi; None if neither does."""
    check = "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst"
    for py in dict.fromkeys((sys.executable, SYSTEM_PYTHON)):
        code, _ = run([py, "-c", check])
        if code == 0:
            return py
    return None


_ENTITY = re.compile(r"^- entity \d+: (.+?) \(")
_LINK = re.compile(r'->\s*"(.+?)":(\d+)\s*\[([^\]]*)\]')
_FMT = re.compile(r"fmt:([^/\s]+)/(\d+x\d+)")
_NODE = re.compile(r"device node name (\S+)")
_DRIVER = re.compile(r"^driver\s+(\S+)", re.M)


def parse_media(text: str) -> dict:
    """
    media-ctl -p output: {"driver", "entities": [{name, node, fmt}], "links": [(from, to)]}.

    Only enabled links count (the route a frame takes); fmt is the first
    format on the entity's pads, e.g. "SRGGB10_1X10 1920x1080". It's the
    format at the time media-ctl ran: before libcamera configures the
    sensor, that's the driver's default, not the mode that streams.
    """
    driver = _DRIVER.search(text)
    entities, links, cur = [], [], None
    for line in text.splitlines():
        m = _ENTITY.match(line.strip()) if line.strip().startswith("- entity") else None
        if m:
            cur = {"name": m.group(1), "node": None, "fmt": None}
            entities.append(cur)
            continue
        if cur is None:
            continue
        if (m := _NODE.search(line)) and cur["node"] is None:
            cur["node"] = m.group(1)
        if (m := _FMT.search(line)) and cur["fmt"] is None:
            cur["fmt"] = f"{m.group(1)} {m.group(2)}"
        if (m := _LINK.search(line)) and "ENABLED" in m.group(3):
            links.append((cur["name"], m.group(1)))
    return {"driver": driver.group(1) if driver else None, "entities": entities, "links": links}


_CAPS_LINE = re.compile(r"^/GstPipeline:[^/]+/(?:[^/]+/)*(?:\w+:)?([\w-]+)\.(?:\w+:)?([\w-]+): caps = (.+)$")
_FIELD = re.compile(r"([\w-]+)=\((\w+)\)(\"(?:[^\"\\]|\\.)*\"|[^,;]*)")


def _fields(text: str) -> dict:
    """A GstStructure's fields, key=(type)value, as {key: value string}."""
    return {k: v.strip().strip('"') for k, _, v in _FIELD.findall(text)}


def parse_caps(text: str) -> list[dict]:
    """
    gst-launch -v's caps lines: one {element, pad, format, size, framerate,
    caps} per pad, the last caps each pad settled on, in order of first
    appearance.
    """
    out = {}
    for line in text.splitlines():
        m = _CAPS_LINE.match(line.strip())
        if not m:
            continue
        element, pad, caps = m.groups()
        f = _fields(caps)
        size = f"{f['width']}x{f['height']}" if "width" in f and "height" in f else None
        out[(element, pad)] = {"element": element, "pad": pad, "format": f.get("format"), "size": size,
                               "framerate": f.get("framerate"), "caps": caps.split(",")[0].strip()}
    return list(out.values())


PAD_KINDS = ("element", "pipeline", "camera")


def parse_pad_timing(text: str) -> list[dict]:
    """
    pad_timing.py's lines: one {ts_ms, kind, element, ms} per buffer, the
    same rows parse_tracer gives (plus kind "camera": sensor timestamp to
    the source's first pad).
    """
    rows = []
    for line in text.splitlines():
        p = line.rstrip("\n").split("\t")
        if len(p) != 5 or p[0] != PAD_TAG or p[1] not in PAD_KINDS:
            continue
        try:
            rows.append({"ts_ms": round(float(p[3]), 3), "kind": p[1], "element": p[2], "ms": round(float(p[4]), 4)})
        except ValueError:
            continue
    return rows


def pad_timing_status(text: str) -> tuple[str | None, str]:
    """pad_timing.py's last status line: ("EOS" | "ERROR" | "NOGI" | None, the message)."""
    for line in reversed(text.splitlines()):
        p = line.split("\t", 2)
        if len(p) >= 2 and p[0] == PAD_TAG and p[1] in ("EOS", "ERROR", "NOGI"):
            return p[1], p[2] if len(p) > 2 else ""
    return None, ""


_TRACE = re.compile(r"\b(element-latency|latency), (.*)$")


def parse_tracer(text: str) -> list[dict]:
    """
    GStreamer's latency tracer lines: one {ts_ms, kind, element, ms} per
    buffer. No longer recorded (the tracer stalls libcamerasrc on the Pi);
    kept to read a gst_trace.log taken elsewhere.

    kind "element": the time the element held the buffer (element-latency).
    kind "pipeline": source pad to sink pad ("latency"), element
    "<src> -> <sink>".
    """
    rows = []
    for line in text.splitlines():
        m = _TRACE.search(line)
        if not m:
            continue
        f = _fields(m.group(2))
        try:
            ms, ts = int(f["time"]) / 1e6, int(f.get("ts", 0)) / 1e6
        except (KeyError, ValueError):
            continue
        if m.group(1) == "element-latency":
            if "element" in f:
                rows.append({"ts_ms": round(ts, 3), "kind": "element", "element": f["element"], "ms": round(ms, 4)})
        elif "src-element" in f and "sink-element" in f:
            rows.append({"ts_ms": round(ts, 3), "kind": "pipeline",
                         "element": f"{f['src-element']} -> {f['sink-element']}", "ms": round(ms, 4)})
    return rows


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def latency_stats(rows: list[dict]) -> list[dict]:
    """Per (kind, element): frames, median, p95 and max ms, and frames per second over the run."""
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["element"]), []).append(r)
    out = []
    for (kind, element), rs in groups.items():
        ms = [r["ms"] for r in rs]
        span_s = (max(r["ts_ms"] for r in rs) - min(r["ts_ms"] for r in rs)) / 1000.0
        out.append({"kind": kind, "element": element, "frames": len(rs),
                    "median_ms": round(_pct(ms, 0.5), 3), "p95_ms": round(_pct(ms, 0.95), 3),
                    "max_ms": round(max(ms), 3),
                    "fps": round((len(rs) - 1) / span_s, 1) if span_s > 0 else None})
    return out


# libcamera's log lines worth keeping: its version, the sensor mode and Unicam format it
# picked, and the streams it configured
LIBCAMERA_KEEP = ("libcamera v", "Selected", "configuring streams")
_LIBCAMERA_PREFIX = re.compile(r"^\[[^\]]*\]\s*\[\d+\]\s*[A-Z]+\s+\S+\s+\S+:\d+\s+")


def libcamera_selected(text: str) -> list[str]:
    """libcamera's log lines holding LIBCAMERA_KEEP, once each, without the [time] [pid] LEVEL Category file:line prefix."""
    keep = []
    for line in (text or "").splitlines():
        if any(k in line for k in LIBCAMERA_KEEP):
            msg = _LIBCAMERA_PREFIX.sub("", line.strip())
            if msg not in keep:
                keep.append(msg)
    return keep


def _static_probes() -> list[tuple[str, list[str]]]:
    probes = [("uname.txt", ["uname", "-a"]),
              ("v4l2_devices.txt", ["v4l2-ctl", "--list-devices"]),
              ("cameras.txt", ["rpicam-hello", "--list-cameras"]),
              ("gst_libcamerasrc.txt", ["gst-inspect-1.0", "libcamerasrc"]),
              ("vc_version.txt", ["vcgencmd", "version"])]
    probes += [(f"vc_clock_{c}.txt", ["vcgencmd", "measure_clock", c]) for c in VC_CLOCKS]
    probes += [("vc_volts_core.txt", ["vcgencmd", "measure_volts", "core"]),
               ("vc_mem_arm.txt", ["vcgencmd", "get_mem", "arm"]),
               ("vc_mem_gpu.txt", ["vcgencmd", "get_mem", "gpu"])]
    return probes


def _save(out_dir: Path, name: str, text: str) -> None:
    (out_dir / name).write_text(text if text.endswith("\n") or not text else text + "\n")


def _draw(dot_path: Path, run) -> str | None:
    """Render a .dot to .png with graphviz; the png's name, or None without graphviz."""
    png = dot_path.with_suffix(".png")
    code, _ = run(["dot", "-Tpng", str(dot_path), "-o", str(png)])
    return png.name if code == 0 else None


def _bisect_file(name: str) -> str:
    return f"gst_bisect_{name.replace(' ', '_')}.txt"


def record(out_dir, duration_s: float = DURATION_S, controls: dict | None = None,
           run=run_tool, launch=launch_for, media_glob: str = "/dev/media*",
           meminfo: Path = MEMINFO_PATH) -> dict:
    """
    Every probe, the pipeline for duration_s (twice: gst-launch for the caps,
    pad_timing.py for the time per element), and the folder written.

    Inputs:
        out_dir: The folder; made if missing.
        controls: libcamerasrc controls; None is params.CAMERA_CONTROLS.
        run: (argv) -> (code | None, text), a short command (run_tool).
        launch: (argv, env, duration_s) -> (code | None, text), the
            pipeline (launch_for). Both injectable for tests.
    Outputs:
        The parsed result (anatomy.json); see summary_lines for its use.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": duration_s, "missing": [],
           "vc": {}, "media": [], "caps": [], "caps_from": None, "latency": [], "libcamera": [], "pngs": [],
           "pad_timing": None}

    for name, argv in _static_probes():
        code, text = run(argv)
        _save(out, name, text)
        if code is None:
            if argv[0] not in res["missing"]:
                res["missing"].append(argv[0])
            continue
        if name.startswith("vc_") and name != "vc_version.txt" and code == 0:
            res["vc"][name[3:-4]] = parse_vcgencmd(text)
    res["vc"].update(read_cma_mb(meminfo))

    for dev in sorted(glob.glob(media_glob)):
        n = re.sub(r"\D", "", Path(dev).name) or "0"
        code, text = run(["media-ctl", "-d", dev, "-p"])
        _save(out, f"media{n}.txt", text)
        if code is None:
            if "media-ctl" not in res["missing"]:
                res["missing"].append("media-ctl")
            break
        res["media"].append({"device": dev, **parse_media(text)})
        code, dot = run(["media-ctl", "-d", dev, "--print-dot"])
        if code == 0:
            _save(out, f"media{n}.dot", dot)
            if png := _draw(out / f"media{n}.dot", run):
                res["pngs"].append(png)

    # 1. gst-launch -v: the caps, the pipeline dumps, libcamera's log
    pipeline = anatomy_pipeline(controls, num_buffers=max(1, int(round(duration_s * FPS))))
    res["pipeline"] = pipeline
    dot_dir = out / "gst_dot"
    dot_dir.mkdir(exist_ok=True)
    groups = {
        "dot dumps": {"GST_DEBUG_DUMP_DOT_DIR": str(dot_dir)},
        "libcamera log": {"LIBCAMERA_LOG_FILE": str(out / "libcamera.log"), "LIBCAMERA_LOG_LEVELS": LIBCAMERA_LOG_LEVELS},
    }
    env = {**os.environ, **{k: v for g in groups.values() for k, v in g.items()}}
    # one argv word per token, as a shell would split it: gst-launch escapes the spaces
    # inside a single argument, so the whole pipeline as one word is a syntax error
    code, text = launch(["gst-launch-1.0", "-v", "-e", *shlex.split(pipeline)], env, duration_s + STARTUP_S)
    _save(out, "gst_launch.txt", text)
    res["gst_code"] = code
    res["gst_errors"] = gst_errors(text)
    res["gst_playing"] = "Setting pipeline to PLAYING" in text
    res["gst_interrupted"] = INTERRUPTED in text
    # Ending on its own at EOS means the sink took every buffer it asked for
    res["gst_eos"] = "Got EOS from element" in text and not res["gst_interrupted"]
    res["stall_bisect"] = (bisect_stall(controls, groups, out, launch)
                           if res["gst_interrupted"] and not res["gst_eos"] else {})
    if code is None:
        res["missing"].append("gst-launch-1.0")
    res["caps"] = parse_caps(text)
    if res["caps"]:
        res["caps_from"] = "gst_launch.txt"
    # the main run stalled: a bisect run that reached EOS printed the same caps
    for name, ran in res["stall_bisect"].items():
        if res["caps"]:
            break
        if ran:
            res["caps"] = parse_caps((out / _bisect_file(name)).read_text(errors="replace"))
            res["caps_from"] = _bisect_file(name) if res["caps"] else None

    # 2. pad_timing.py: the time per element, by buffer probes
    rows = []
    py = gi_python(run) if code is not None else None
    if code is not None and py is None:
        res["missing"].append("PyGObject (gi)")
    if py is not None:
        _, ptext = launch([py, str(PAD_TIMING), pipeline], dict(os.environ), duration_s + STARTUP_S)
        _save(out, "pad_timing.txt", ptext)
        status, message = pad_timing_status(ptext)
        res["pad_timing"] = {"python": py, "status": status, "message": message,
                             "interrupted": INTERRUPTED in ptext}
        rows = parse_pad_timing(ptext)
    res["latency"] = latency_stats(rows)
    with open(out / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, LATENCY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    log = out / "libcamera.log"
    res["libcamera"] = libcamera_selected(log.read_text(errors="replace")) if log.exists() else []
    # the dump made as the pipeline started playing shows the caps every link settled on
    for dot in sorted(dot_dir.glob("*.dot")):
        if "PAUSED_PLAYING" in dot.name:
            target = out / "gst_pipeline.dot"
            shutil.copy(dot, target)
            if png := _draw(target, run):
                res["pngs"].append(png)
            break

    res["findings"] = findings(res)
    with open(out / "anatomy.json", "w") as f:
        json.dump(res, f, indent=2)
    lines = summary_lines(res)
    _save(out, "summary.txt", "\n".join(lines))
    return res


def _element_ms(res: dict, factory: str) -> float:
    """The median ms of the elements whose name starts with factory (videoconvert0...), summed."""
    return sum(s["median_ms"] for s in res["latency"] if s["kind"] == "element" and s["element"].startswith(factory))


_BUSY = ("busy", "acquire", "in use")
BISECT_BUFFERS = 40         # 2 s at 20 FPS: enough to show the camera runs


def _ran(text: str) -> bool:
    return "Got EOS from element" in text and INTERRUPTED not in text and not gst_errors(text)


def bisect_stall(controls, groups: dict, out: Path, launch) -> dict:
    """
    When the recording stalled: which of the tool's settings stalls it.

    Runs the pipeline for BISECT_BUFFERS with none of them ("none"), then
    with each group alone. By hand, with none, it ran (2026-10-12), so a
    group that stalls it alone is the cause. Each run's output is saved as
    gst_bisect_<group>.txt.

    Outputs:
        {"none": ran, group: ran, ...}; True = it reached EOS on its own.
    """
    argv = ["gst-launch-1.0", "-v", "-e", *shlex.split(anatomy_pipeline(controls, num_buffers=BISECT_BUFFERS))]
    timeout = BISECT_BUFFERS / FPS + STARTUP_S
    found = {}
    for name, extra in (("none", {}), *groups.items()):
        _, text = launch(argv, {**os.environ, **extra}, timeout)
        _save(out, _bisect_file(name), text)
        found[name] = _ran(text)
        if name == "none" and not found[name]:
            break                           # stalls with nothing added: the settings aren't it
    return found


def gst_errors(text: str) -> list[str]:
    """gst-launch's ERROR lines, each with the debug-info line under it, in order."""
    lines, out = text.splitlines(), []
    for i, line in enumerate(lines):
        if line.startswith("ERROR"):
            more = lines[i + 1].strip() if i + 1 < len(lines) and lines[i + 1].startswith("Additional") else ""
            out.append(f"{line.strip()} {more}".strip())
    return out


def frames_flowed(res: dict) -> bool:
    """
    Buffers reached the sink: the probes saw some, or gst-launch ended on its
    own at EOS (the sink's num-buffers) with no error.
    """
    return (bool(res["latency"]) or bool(res.get("gst_eos") and not res.get("gst_errors"))
            or bool((res.get("stall_bisect") or {}).get("none")))


def _no_timing(res: dict) -> str | None:
    """Why pad_timing.py gave no rows, when the recording can tell; None when it can't."""
    pt = res.get("pad_timing")
    if "PyGObject (gi)" in res["missing"]:
        return (f"no time per element: neither this python nor {SYSTEM_PYTHON} imports Gst through gi "
                "(sudo apt install python3-gi gir1.2-gstreamer-1.0)")
    if not pt:
        return None
    if pt["interrupted"]:
        return ("the pad-timing run stalled (stopped after its time ran out) though gst-launch ran: "
                "see pad_timing.txt")
    if pt["status"] in ("ERROR", "NOGI"):
        return f"the pad-timing run ended with {pt['status']}: {pt['message']}. See pad_timing.txt"
    return None


def _no_pipeline_latency(res: dict) -> str:
    """Why there's no source-to-sink time, as far as the recording can tell."""
    if "gst-launch-1.0" in res["missing"]:
        return "no GStreamer here: only the static probes ran"
    errors = res.get("gst_errors") or []
    if errors:
        busy = any(w in errors[0].lower() for w in _BUSY)
        hint = ": the camera is open in another process (the robot's pipeline, or a test before this one)" if busy else ""
        return f"gst-launch stopped with {errors[0]}{hint}. See gst_launch.txt"
    bisect = res.get("stall_bisect") or {}
    if bisect:
        culprits = [g for g, ok in bisect.items() if g != "none" and not ok]
        if bisect.get("none") and culprits:
            return (f"gst-launch stalled with the tool's {', '.join(culprits)} set, and runs without "
                    f"({BISECT_BUFFERS} frames to EOS): the recording can't use {' or '.join(culprits)} on this "
                    "Pi. See gst_bisect_*.txt")
        if bisect.get("none"):
            return ("gst-launch stalled with all the tool's settings together, but runs with none and with each "
                    "alone. See gst_bisect_*.txt")
        return ("gst-launch stalls in this process even with none of the tool's settings, though it runs by "
                "hand: what differs is the launch itself (output to a pipe, under pytest). See gst_bisect_none.txt")
    if why := _no_timing(res):
        return why
    if frames_flowed(res) and not res["latency"]:
        return ("frames reached the sink (gst-launch ended on its own at EOS), but the pad-timing run logged "
                "nothing. See pad_timing.txt")
    if frames_flowed(res):
        return ("element times recorded but no source-to-sink time: the source or the sink has no static pad "
                "named src / sink for the probes. See pad_timing.txt")
    if res.get("gst_playing") and not res["caps"]:
        return (f"gst-launch reached PLAYING but the camera never negotiated its caps in "
                f"{res['duration_s'] + STARTUP_S:.0f} s: the camera may be held by another process. See libcamera.log")
    if res["caps"]:
        return ("the pipeline negotiated its caps but no buffer was timed: either no frame reached the sink, or "
                "the pad-timing run failed. See gst_launch.txt and pad_timing.txt")
    return ("no frame reached the sink and gst-launch reported no error: is the camera open in another process "
            "(the robot's pipeline)? See gst_launch.txt and libcamera.log")


def findings(res: dict) -> list[str]:
    """What the recording says, in words."""
    out = []
    pipe = [s for s in res["latency"] if s["kind"] == "pipeline"]
    if not pipe:
        return [_no_pipeline_latency(res)]
    if res.get("stall_bisect"):
        out.append(_no_pipeline_latency(res))       # the stall is still worth saying when timing came anyway
    p = pipe[0]
    if p["fps"] is not None and p["fps"] < 0.9 * FPS:
        out.append(f"the pipeline delivered {p['fps']:.1f} fps, under the {FPS} asked for")
    cam = next((s for s in res["latency"] if s["kind"] == "camera"), None)
    if cam and not 0 <= cam["median_ms"] <= 1000:
        out.append(f"sensor -> {cam['element']} came out {cam['median_ms']:.0f} ms: the source's PTS isn't the "
                   "sensor timestamp in running time here, so that row means nothing")
    conv = _element_ms(res, "videoconvert") + _element_ms(res, "videoflip")
    src = next((c for c in res["caps"] if c["element"].startswith("libcamerasrc")), None)
    if conv >= CONVERT_FINDING_MS:
        fmt = f" from {src['format']}" if src and src["format"] else ""
        out.append(f"videoconvert + videoflip take {conv:.1f} ms of every frame{fmt} "
                   f"({100 * conv / FRAME_BUDGET_MS:.0f}% of the {FRAME_BUDGET_MS:.0f} ms budget), on the ARM "
                   "cores: the ISP can output BGR and the sensor can flip, which would make both copies go away")
    cma = res["vc"]
    if cma.get("cma_total_mb") and cma.get("cma_free_mb") is not None and cma["cma_free_mb"] < 0.1 * cma["cma_total_mb"]:
        out.append(f"CMA nearly full ({cma['cma_free_mb']:.0f} of {cma['cma_total_mb']:.0f} MB free): "
                   "camera buffers are allocated from it")
    return out


_KIND_ORDER = {"pipeline": 0, "camera": 1}
_KIND_LABEL = {"pipeline": "pipeline ", "camera": "sensor -> "}


def summary_lines(res: dict) -> list[str]:
    """summary.txt: the route, the caps, time per element, the VideoCore side, libcamera, findings."""
    fmt = lambda v, spec="": "--" if v is None else f"{v:{spec}}"      # noqa: E731
    lines = [f"capture anatomy  {res['started']}  ({res['duration_s']:.0f} s of the pipeline)",
             f"  {res.get('pipeline', '')}"]
    if res["missing"]:
        lines.append(f"  not available here: {', '.join(res['missing'])}")
    lines += ["", "hardware graph (media-ctl): enabled links, the route a frame takes",
              "  (formats as media-ctl found them, before libcamera set the mode: see libcamera below)"]
    for m in res["media"]:
        lines.append(f"  {m['device']}  {m['driver'] or '?'}")
        info = {e["name"]: e for e in m["entities"]}
        for a, b in m["links"]:
            ea, eb = info.get(a, {}), info.get(b, {})
            tag = lambda e: "".join(f" [{x}]" for x in (e.get("fmt"), e.get("node")) if x)   # noqa: E731
            lines.append(f"    {a}{tag(ea)} -> {b}{tag(eb)}")
        if not m["links"]:
            lines.append("    (no enabled links)")
    if not res["media"]:
        lines.append("  none recorded")
    src = res.get("caps_from")
    lines += ["", "negotiated caps (gst-launch -v), per pad"
              + (f"  (from {src}: the main run stalled)" if src and src != "gst_launch.txt" else "")]
    for c in res["caps"]:
        lines.append(f"  {c['element'] + '.' + c['pad']:<26} {fmt(c['format']):<8} {fmt(c['size']):<10} "
                     f"{fmt(c['framerate']):<6} {c['caps']}")
    if not res["caps"]:
        lines.append("  none recorded")
    lines += ["", "time per frame (buffer probes, pad_timing.py)",
              f"  {'':<38} {'frames':>6} {'median ms':>9} {'p95 ms':>7} {'max ms':>7} {'fps':>5}"]
    for s in sorted(res["latency"], key=lambda s: (_KIND_ORDER.get(s["kind"], 2), s["element"])):
        name = _KIND_LABEL.get(s["kind"], "") + s["element"]
        lines.append(f"  {name:<38} {s['frames']:>6} {s['median_ms']:>9.2f} {s['p95_ms']:>7.2f} "
                     f"{s['max_ms']:>7.2f} {fmt(s['fps'], '.1f'):>5}")
    if not res["latency"]:
        lines.append("  none recorded")
    lines.append("  (sensor -> libcamerasrc: the sensor timestamp to the first pad, i.e. readout, Unicam, ISP;")
    lines.append("   frame_meta records the same per frame in the robot)")
    vc = res["vc"]
    clocks = "  ".join(f"{c} {vc[f'clock_{c}'] / 1e6:.0f}" for c in VC_CLOCKS if vc.get(f"clock_{c}"))
    lines += ["", "VideoCore",
              f"  clocks MHz: {clocks or '--'}",
              f"  core {fmt(vc.get('volts_core'), '.3f')} V   memory arm {fmt(vc.get('mem_arm'), '.0f')} MB, "
              f"gpu {fmt(vc.get('mem_gpu'), '.0f')} MB   CMA {fmt(vc.get('cma_free_mb'), '.0f')} of "
              f"{fmt(vc.get('cma_total_mb'), '.0f')} MB free"]
    lines += ["", "libcamera"] + [f"  {x}" for x in res["libcamera"] or ["(nothing recorded)"]]
    lines += ["", "findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]
    if res["pngs"]:
        lines += ["", "drawn: " + ", ".join(res["pngs"])]
    return lines


def cli(argv: list[str] | None = None) -> int:
    """python3 -m src.diagnostics.capture_anatomy [--seconds S] [--camera-control K=V ...] [--out DIR]"""
    ap = argparse.ArgumentParser(prog="python3 -m src.diagnostics.capture_anatomy", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=DURATION_S, metavar="S",
                    help=f"how long the pipeline runs (default {DURATION_S:.0f})")
    ap.add_argument("--camera-control", action="append", default=None, metavar="KEY=VALUE",
                    help="a libcamerasrc control for this recording, as the linkers take it")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        # added to params.CAMERA_CONTROLS for this recording, as the linkers do
        controls = {**CAMERA_CONTROLS, **parse_controls(args.camera_control)} if args.camera_control else None
    except ValueError as exc:
        print(exc)
        return 2
    out_dir = args.out or str(RUNS_DIR / ("anatomy_" + time.strftime("%Y%m%d_%H%M%S")))
    print(f"capture anatomy: {args.seconds:.0f} s of the pipeline, output {out_dir}", flush=True)
    res = record(out_dir, args.seconds, controls)
    print("\n" + "\n".join(summary_lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
