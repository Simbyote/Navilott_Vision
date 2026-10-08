"""Capture anatomy: how a frame gets from the IMX290 to OpenCV on this Pi, recorded once.

Purpose:
    The capture pipeline is one GStreamer string (capture/camera.py), but a
    frame crosses three pieces of hardware and two pieces of software on the
    way: the sensor reads out raw Bayer over CSI-2; Unicam, the Pi's CSI-2
    receiver, DMA-writes it to memory; the VideoCore ISP demosaics,
    corrects and downscales it; libcamera's IPA, running on the ARM cores in
    the robot's process, reads the ISP's statistics and sets the next
    exposure and gain; then videoconvert and videoflip turn it into the BGR
    image OpenCV gets, in software. None of that is visible from Python.
    This records what the OS can show of it, once, into one folder:

    - the hardware graph the kernel built (media-ctl): sensor -> Unicam ->
      memory, and the ISP's input and output nodes, with the format on each
      link;
    - the caps GStreamer negotiated at every pad (gst-launch -v): what the
      ISP hands over (NV12, YUY2, BGR...) decides how much videoconvert
      does;
    - how long every element holds each frame, and the source-to-sink
      latency (GStreamer's latency tracer), so the CPU-side conversion has
      a number;
    - libcamera's log (the sensor mode and Unicam format it picked);
    - the VideoCore side: ARM, core, ISP, 3D and H.264 clocks, core volts,
      the ARM / GPU memory split, the CMA pool camera buffers come from.

    It runs the robot's exact pipeline string with appsink swapped for a
    fakesink, so nothing reads the frames and the timing is the capture's
    alone. The camera can be open in one process at a time: stop the robot's
    pipeline first. Every tool is optional; a missing one is noted and the
    rest is recorded (on a laptop, almost everything is missing).

Main package:
    record(out_dir, duration_s, ...) -> dict: runs every probe, writes the
        folder, returns what it parsed.
    parse_media(text), parse_caps(text), parse_tracer(text),
    libcamera_selected(text): the parsers, one per tool's output
        (vcgencmd's and /proc/meminfo's are os_counters', shared with the
        recorder).
    summary_lines(res): summary.txt.
    cli(): python3 -m src.diagnostics.capture_anatomy [--seconds S] [--out DIR]

Flow:
    1. Static probes: uname, v4l2-ctl, rpicam-hello --list-cameras,
       gst-inspect-1.0 libcamerasrc, vcgencmd, /proc/meminfo, media-ctl
       on every /dev/media* (graph and dot, drawn to PNG when graphviz is
       there).
    2. The pipeline under gst-launch-1.0 -v for --seconds, with the latency
       tracer, libcamera's log and GStreamer's pipeline dumps on; then
       Ctrl-C (-e: a clean EOS).
    3. Parse, write latency.csv, anatomy.json and summary.txt.
"""
import argparse
import csv
import glob
import json
import os
import re
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
# GStreamer's built-in latency tracer (1.18+): per-element time ("element-latency")
# and source-to-sink time ("latency") for every buffer
GST_TRACERS = "latency(flags=pipeline+element)"
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
    latency.csv        one row per frame per element (and the whole pipeline)
    media<N>.txt/.dot/.png, gst_*.dot/.png, gst_launch.txt, gst_trace.log,
    libcamera.log, and each tool's raw output
"""


# =============================================================================
# Running tools
# =============================================================================

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
    Run argv for duration_s, then Ctrl-C it; (exit code, stdout + stderr).
    (None, why) when it can't start. A run that ends early keeps its own code.
    """
    try:
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        return None, f"not available: {argv[0]} is not installed"
    try:
        out, _ = p.communicate(timeout=duration_s)
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGINT)
        try:
            out, _ = p.communicate(timeout=STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
    return p.returncode, out


def anatomy_pipeline(controls: dict | None = None) -> str:
    """The robot's capture pipeline string with its appsink swapped for SINK."""
    return build_gst_pipeline(controls=controls).rsplit(" ! ", 1)[0] + " ! " + SINK


# =============================================================================
# Parsers
# =============================================================================

_ENTITY = re.compile(r"^- entity \d+: (.+?) \(")
_LINK = re.compile(r'->\s*"(.+?)":(\d+)\s*\[([^\]]*)\]')
_FMT = re.compile(r"fmt:([^/\s]+)/(\d+x\d+)")
_NODE = re.compile(r"device node name (\S+)")
_DRIVER = re.compile(r"^driver\s+(\S+)", re.M)


def parse_media(text: str) -> dict:
    """
    media-ctl -p output: {"driver", "entities": [{name, node, fmt}], "links": [(from, to)]}.

    Only enabled links count (the route a frame takes); fmt is the first
    format on the entity's pads, e.g. "SRGGB10_1X10 1920x1080".
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


_TRACE = re.compile(r"\b(element-latency|latency), (.*)$")


def parse_tracer(text: str) -> list[dict]:
    """
    The latency tracer's lines: one {ts_ms, kind, element, ms} per buffer.

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


# =============================================================================
# Recording
# =============================================================================

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


def record(out_dir, duration_s: float = DURATION_S, controls: dict | None = None,
           run=run_tool, launch=launch_for, media_glob: str = "/dev/media*",
           meminfo: Path = MEMINFO_PATH) -> dict:
    """
    Every probe, the pipeline for duration_s, and the folder written.

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
           "vc": {}, "media": [], "caps": [], "latency": [], "libcamera": [], "pngs": []}

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

    pipeline = anatomy_pipeline(controls)
    res["pipeline"] = pipeline
    dot_dir = out / "gst_dot"
    dot_dir.mkdir(exist_ok=True)
    env = {**os.environ, "GST_TRACERS": GST_TRACERS, "GST_DEBUG": "GST_TRACER:7",
           "GST_DEBUG_FILE": str(out / "gst_trace.log"), "GST_DEBUG_NO_COLOR": "1",
           "GST_DEBUG_DUMP_DOT_DIR": str(dot_dir),
           "LIBCAMERA_LOG_FILE": str(out / "libcamera.log"), "LIBCAMERA_LOG_LEVELS": LIBCAMERA_LOG_LEVELS}
    code, text = launch(["gst-launch-1.0", "-v", "-e", pipeline], env, duration_s)
    _save(out, "gst_launch.txt", text)
    res["gst_code"] = code
    if code is None:
        res["missing"].append("gst-launch-1.0")
    res["caps"] = parse_caps(text)
    trace = out / "gst_trace.log"
    rows = parse_tracer(trace.read_text(errors="replace")) if trace.exists() else []
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


# =============================================================================
# Summary
# =============================================================================

def _element_ms(res: dict, factory: str) -> float:
    """The median ms of the elements whose name starts with factory (videoconvert0...), summed."""
    return sum(s["median_ms"] for s in res["latency"] if s["kind"] == "element" and s["element"].startswith(factory))


def findings(res: dict) -> list[str]:
    """What the recording says, in words."""
    out = []
    pipe = [s for s in res["latency"] if s["kind"] == "pipeline"]
    if not pipe:
        if "gst-launch-1.0" in res["missing"]:
            out.append("no GStreamer here: only the static probes ran")
        else:
            out.append("no frame reached the sink: is the camera open in another process (the robot's "
                       "pipeline)? See gst_launch.txt")
        return out
    p = pipe[0]
    if p["fps"] is not None and p["fps"] < 0.9 * FPS:
        out.append(f"the pipeline delivered {p['fps']:.1f} fps, under the {FPS} asked for")
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


def summary_lines(res: dict) -> list[str]:
    """summary.txt: the route, the caps, time per element, the VideoCore side, libcamera, findings."""
    fmt = lambda v, spec="": "--" if v is None else f"{v:{spec}}"      # noqa: E731
    lines = [f"capture anatomy  {res['started']}  ({res['duration_s']:.0f} s of the pipeline)",
             f"  {res.get('pipeline', '')}"]
    if res["missing"]:
        lines.append(f"  not available here: {', '.join(res['missing'])}")
    lines += ["", "hardware graph (media-ctl): enabled links, the route a frame takes"]
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
    lines += ["", "negotiated caps (gst-launch -v), per pad"]
    for c in res["caps"]:
        lines.append(f"  {c['element'] + '.' + c['pad']:<26} {fmt(c['format']):<8} {fmt(c['size']):<10} "
                     f"{fmt(c['framerate']):<6} {c['caps']}")
    if not res["caps"]:
        lines.append("  none recorded")
    lines += ["", "time per frame (GStreamer latency tracer)",
              f"  {'':<34} {'frames':>6} {'median ms':>9} {'p95 ms':>7} {'max ms':>7} {'fps':>5}"]
    for s in sorted(res["latency"], key=lambda s: (s["kind"] != "pipeline", s["element"])):
        name = ("pipeline " if s["kind"] == "pipeline" else "") + s["element"]
        lines.append(f"  {name:<34} {s['frames']:>6} {s['median_ms']:>9.2f} {s['p95_ms']:>7.2f} "
                     f"{s['max_ms']:>7.2f} {fmt(s['fps'], '.1f'):>5}")
    if not res["latency"]:
        lines.append("  none recorded")
    lines.append("  (the ISP's own time is inside libcamerasrc, before the first pad: frame_meta records it)")
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


# =============================================================================
# Command line
# =============================================================================

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
