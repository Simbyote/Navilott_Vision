"""Frame metadata: what libcamera did to every frame, beside what the color branch read in it.

Purpose:
    The camera's auto exposure (AGC) and white balance (AWB) run in
    libcamera's IPA on the ARM cores: from the ISP's statistics of each
    frame they set the next frame's exposure time, analogue and digital
    gain and colour gains. libcamera reports those per frame, with the
    moment the sensor started exposing it, but the robot's GStreamer
    pipeline drops that metadata at the appsink. This opens the camera
    through Picamera2 instead, set up like the robot's (the same sensor
    mode, size, frame rate, 180 degree flip and CAMERA_CONTROLS), and logs
    every frame's metadata next to the color branch's own reading of its
    traffic ROI (MEASURED: the label, confidence and clipped-white pixels).

    That answers what the GStreamer path can't:
    - did a red LED read yellow on frames with longer exposure or more
      gain (hold the light on one color and compare the labels);
    - how far AGC swings as the robot turns, and whether it has settled;
    - how long a frame takes from the sensor starting to expose it to
      Python having it (the ISP, the IPA and the queue included).

    Like capture_anatomy, it needs the camera free: stop the robot's
    pipeline first. --save-every keeps frames as a folder the linkers
    (--frames) and calib-lamps read.

Main package:
    record(out_dir, seconds, ...) -> rows: the camera for that long, frames.csv,
        meta.json, summary.txt and frames/ written.
    to_libcamera_controls(controls): libcamerasrc names (exposure-value)
        as Picamera2's (ExposureValue), enum names as their values.
    frame_row(md, now_ns, t0_ns, detection): one frame's CSV row.
    summarize(rows), findings(summary), summary_lines(summary).
    cli(): python3 -m src.diagnostics.frame_meta [--seconds S] [--save-every N]
        [--camera-control K=V] [--no-detect] [--out DIR]

Flow:
    1. Picamera2 (lazy import: only the Pi has it) configured like the robot.
    2. Each frame: its request's metadata and image; the color branch on
       the traffic ROI; a row; every Nth image saved.
    3. Stop at --seconds (or Ctrl-C); write and summarize.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from src.capture.camera import FrameData, parse_controls
from src.params import CAMERA_CONTROLS, CAMERA_ROTATE_180, FPS, FRAME_H, FRAME_W, RUNS_DIR, SENSOR_CONFIG

SECONDS = 20.0
SENSOR_SIZE = tuple(int(SENSOR_CONFIG.split(f"{k}=")[1].split(",")[0]) for k in ("width", "height"))
FRAME_FIELDS = ("frame", "t_s", "exposure_us", "analogue_gain", "digital_gain", "colour_gain_r", "colour_gain_b",
                "colour_temp_k", "lux", "frame_duration_us", "ae_locked", "sensor_to_python_ms",
                "label", "confidence", "white_px")
NUMERIC = ("exposure_us", "analogue_gain", "digital_gain", "colour_gain_r", "colour_gain_b", "colour_temp_k",
           "lux", "frame_duration_us", "sensor_to_python_ms", "white_px")
MAX_LATENCY_MS = 1000.0         # a sensor-to-Python time past this means the clocks don't match: dropped
EXPOSURE_SWING = 2.0            # exposure x gain moving more than this factor earns a finding
LIGHT_GAP = 1.2                 # yellow frames with this much more light than red ones earn a finding
AE_SETTLED_SHARE = 0.5          # under this share of frames AeLocked: AGC never settled
BUDGET_MS = 1000.0 / FPS
MISSING_PICAMERA2 = ("Picamera2 isn't available: on Raspberry Pi OS `sudo apt install python3-picamera2` (a venv "
                     "needs --system-site-packages to see it). Without it, `rpicam-hello -n -t 10000 --metadata "
                     "meta.json` records the same metadata, without the detection")

_CLI_HELP = """\
Record libcamera's metadata for every frame (exposure, gains, colour gains,
lux, the sensor-to-Python time) next to what the color branch read in the
traffic ROI, with the camera set up like the robot's. Stop the robot's
pipeline first: the camera opens in one process at a time.

Examples (from vision_stack/):
    python3 -m src.diagnostics.frame_meta                       20 s, the configured controls
    python3 -m src.diagnostics.frame_meta --camera-control exposure-value=-1
    python3 -m src.diagnostics.frame_meta --seconds 30 --save-every 5

Point it at the traffic light and hold one color: if a red LED reads yellow on
some frames, the summary compares their exposure and gain.

Output (--out DIR, default <root>/runs/meta_<timestamp>):
    summary.txt   ranges, the sensor-to-Python time, per label the light it got, findings
    frames.csv    one row per frame
    meta.json     the configuration and the camera's properties
    frames/       every Nth frame as PNG (--save-every N), a --frames source
"""


# =============================================================================
# Controls
# =============================================================================

def _libcamera_enum(name: str, value: str):
    """libcamera.controls.<name>Enum's member called value (any case, - and _ ignored)."""
    from libcamera import controls                      # only on the Pi
    enum = getattr(controls, f"{name}Enum", None)
    if enum is None:
        raise ValueError(f"{name} takes a number, not {value!r}")
    want = value.replace("-", "").replace("_", "").lower()
    for member in dir(enum):
        if member.replace("_", "").lower() == want:
            return getattr(enum, member)
    raise ValueError(f"{name}: no value {value!r}")


def to_libcamera_controls(controls: dict, enum_lookup=_libcamera_enum) -> dict:
    """
    libcamerasrc controls (exposure-value=-1, awb-mode=daylight) as Picamera2's
    ({"ExposureValue": -1, "AwbMode": <AwbModeEnum.Daylight>}).

    Raises:
        ValueError: An enum name the control doesn't have.
    """
    out = {}
    for key, value in controls.items():
        name = "".join(w[:1].upper() + w[1:] for w in key.split("-"))
        out[name] = enum_lookup(name, value) if isinstance(value, str) else value
    return out


# =============================================================================
# One frame
# =============================================================================

def frame_row(n: int, md: dict, now_ns: int, t0_ns: int | None, detection: dict | None) -> dict:
    """
    One frame's row.

    Inputs:
        md: The request's metadata (libcamera's names).
        now_ns: time.monotonic_ns() when Python had the frame. The sensor
            timestamp is on the same clock (the V4L2 buffer's), so the
            difference is the sensor-to-Python time; past MAX_LATENCY_MS
            or negative it's dropped as a clock mismatch.
        t0_ns: The first frame's sensor timestamp: t_s counts from it.
        detection: {label, confidence, white_px} or None (--no-detect).
    """
    sensor_ns = md.get("SensorTimestamp")
    lat = None if sensor_ns is None else (now_ns - sensor_ns) / 1e6
    gains = md.get("ColourGains") or (None, None)
    det = detection or {}
    return {"frame": n,
            "t_s": None if sensor_ns is None or t0_ns is None else round((sensor_ns - t0_ns) / 1e9, 4),
            "exposure_us": md.get("ExposureTime"), "analogue_gain": _r(md.get("AnalogueGain")),
            "digital_gain": _r(md.get("DigitalGain")), "colour_gain_r": _r(gains[0]), "colour_gain_b": _r(gains[1]),
            "colour_temp_k": md.get("ColourTemperature"), "lux": _r(md.get("Lux"), 1),
            "frame_duration_us": md.get("FrameDuration"),
            "ae_locked": None if md.get("AeLocked") is None else int(bool(md["AeLocked"])),
            "sensor_to_python_ms": round(lat, 2) if lat is not None and 0 <= lat <= MAX_LATENCY_MS else None,
            "label": det.get("label") or "", "confidence": det.get("confidence"), "white_px": det.get("white_px")}


def _r(v, digits=3):
    return None if v is None else round(float(v), digits)


def detect(frame, n: int, ts_ms: int, config) -> dict:
    """The color branch on the frame's traffic ROI, as the robot runs it: {label, confidence, white_px}."""
    from src.perception.color_branch import run_color_stage
    from src.perception.preprocess import preprocess_frame
    from src.perception.roi_crop import crop_rois
    rois = crop_rois(preprocess_frame(FrameData(frame, n, ts_ms), config.preprocess), config.roi)
    cands, dbg = run_color_stage(rois, config.color)
    white = dbg.get("white")
    best = cands[0] if cands else None
    return {"label": best.label if best else "", "confidence": best.confidence if best else None,
            "white_px": None if white is None else int((white > 0).sum())}


# =============================================================================
# Recording
# =============================================================================

def open_camera(controls: dict, factory=None):
    """Picamera2 configured like the robot's capture, started. factory: Picamera2 (injectable)."""
    if factory is None:
        from picamera2 import Picamera2
        factory = Picamera2
    cam = factory()
    try:
        from libcamera import Transform
        transform = Transform(hflip=1, vflip=1) if CAMERA_ROTATE_180 else Transform()
    except ImportError:                                  # a fake camera in tests
        transform = None
    config = cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},       # RGB888 is BGR in memory: OpenCV's order
        raw={"size": SENSOR_SIZE},                                    # SENSOR_CONFIG's mode, so the field of view matches
        controls={"FrameRate": float(FPS), **controls},
        **({"transform": transform} if transform is not None else {}))
    cam.configure(config)
    cam.start()
    return cam


def record(out_dir, seconds: float = SECONDS, controls: dict | None = None, save_every: int = 0,
           detect_fn=None, camera=None, clock_ns=time.monotonic_ns, max_frames: int | None = None) -> list[dict]:
    """
    The camera for seconds (or max_frames, or Ctrl-C); every frame's row.

    Inputs:
        controls: Picamera2 controls (to_libcamera_controls' output).
        save_every: Keep every Nth frame as frames/<frame>.png; 0 keeps none.
        detect_fn: (frame, n, ts_ms) -> detection dict, or None to skip.
        camera: A started camera (open_camera's); opened here when None.
    Outputs:
        The rows; frames.csv, meta.json and summary.txt written to out_dir.
    """
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save_every:
        (out / "frames").mkdir(exist_ok=True)
    cam = camera if camera is not None else open_camera(controls or {})
    rows, t0_ns, start = [], None, clock_ns()
    interrupted = False
    try:
        n = 0
        while (clock_ns() - start) / 1e9 < seconds and (max_frames is None or n < max_frames):
            req = cam.capture_request()
            try:
                frame = req.make_array("main")
                md = req.get_metadata()
            finally:
                req.release()
            now = clock_ns()
            if t0_ns is None:
                t0_ns = md.get("SensorTimestamp")
            det = detect_fn(frame, n, (now - start) // 1_000_000) if detect_fn else None
            rows.append(frame_row(n, md, now, t0_ns, det))
            if save_every and n % save_every == 0:
                cv2.imwrite(str(out / "frames" / f"{n:06d}.png"), frame)
            n += 1
    except KeyboardInterrupt:
        interrupted = True
    finally:
        props = dict(getattr(cam, "camera_properties", {}) or {})
        cam.stop()
        cam.close()
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FRAME_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": seconds, "interrupted": interrupted,
            "controls": {k: str(v) for k, v in (controls or {}).items()}, "size": [FRAME_W, FRAME_H],
            "sensor_size": list(SENSOR_SIZE), "fps": FPS, "rotate_180": CAMERA_ROTATE_180,
            "save_every": save_every, "detect": detect_fn is not None,
            "camera_properties": {k: str(v) for k, v in props.items()}}
    with open(out / "meta.json", "w") as f:
        json.dump({**meta, "summary": summary}, f, indent=2)
    (out / "summary.txt").write_text("\n".join(summary_lines(summary, meta)) + "\n")
    return rows


# =============================================================================
# Summary
# =============================================================================

def _stats(xs: list) -> dict | None:
    v = sorted(float(x) for x in xs if x not in (None, ""))
    if not v:
        return None
    return {"min": v[0], "median": statistics.median(v), "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
            "max": v[-1]}


def _light(r: dict) -> float | None:
    """Exposure x total gain: how much light the sensor gathered for a frame, in exposure-us units."""
    e, a = r.get("exposure_us"), r.get("analogue_gain")
    if e in (None, "") or a in (None, ""):
        return None
    return float(e) * float(a) * float(r.get("digital_gain") or 1.0)


def summarize(rows: list[dict]) -> dict:
    """Ranges of every numeric field, the frame rate, AE settling, and per label the light it got."""
    s = {"frames": len(rows), "fields": {k: _stats([r.get(k) for r in rows]) for k in NUMERIC},
         "light": _stats([_light(r) for r in rows])}
    ts = [float(r["t_s"]) for r in rows if r.get("t_s") not in (None, "")]
    s["fps"] = round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and ts[-1] > ts[0] else None
    locked = [int(r["ae_locked"]) for r in rows if r.get("ae_locked") not in (None, "")]
    s["ae_locked_share"] = round(sum(locked) / len(locked), 3) if locked else None
    labels = {}
    for r in rows:
        labels.setdefault(r.get("label") or "none", []).append(r)
    s["labels"] = {lab: {"frames": len(rs), "light": _stats([_light(r) for r in rs]),
                         "exposure_us": _stats([r.get("exposure_us") for r in rs]),
                         "analogue_gain": _stats([r.get("analogue_gain") for r in rs]),
                         "white_px": _stats([r.get("white_px") for r in rs])}
                   for lab, rs in sorted(labels.items())}
    s["findings"] = findings(s)
    return s


def findings(s: dict) -> list[str]:
    """What the recording says, in words."""
    out = []
    light = s["light"]
    if light and light["min"] > 0 and light["max"] / light["min"] > EXPOSURE_SWING:
        out.append(f"AGC moved the light gathered (exposure x gain) {light['max'] / light['min']:.1f}x over the "
                   "recording: the lamps look different as it moves; fixing the exposure (exposure-time, "
                   "analogue-gain) or a negative exposure-value narrows it")
    red, yellow = s["labels"].get("red"), s["labels"].get("yellow")
    if red and yellow and red["light"] and yellow["light"]:
        r, y = red["light"]["median"], yellow["light"]["median"]
        if y > LIGHT_GAP * r:
            out.append(f"yellow frames got {y / r:.1f}x the light of red ones (median exposure x gain {y:.0f} vs "
                       f"{r:.0f}): if the light was red throughout, overexposure turns its ring orange; try "
                       "--camera-control exposure-value=-1")
        else:
            out.append(f"red ({red['frames']}) and yellow ({yellow['frames']}) frames got about the same light: "
                       "if the light was red throughout, the angle, not exposure, turns its ring orange")
    lat = s["fields"]["sensor_to_python_ms"]
    if lat and lat["median"] > BUDGET_MS:
        out.append(f"frames reach Python {lat['median']:.0f} ms (median) after the sensor starts exposing them, "
                   f"over a {BUDGET_MS:.0f} ms frame: the detections act on a picture that old")
    if s["ae_locked_share"] is not None and s["ae_locked_share"] < AE_SETTLED_SHARE:
        out.append(f"AE reported settled on only {100 * s['ae_locked_share']:.0f}% of frames: it kept adapting")
    if s["fps"] is not None and s["fps"] < 0.9 * FPS:
        out.append(f"the sensor delivered {s['fps']:.1f} fps, under the {FPS} asked for (exposure longer than a "
                   "frame stretches the frame)")
    return out


def summary_lines(s: dict, meta: dict | None = None) -> list[str]:
    """summary.txt."""
    meta = meta or {}
    f = s["fields"]
    rng = lambda st, spec=".0f": "--" if st is None else (                                  # noqa: E731
        f"{st['min']:{spec}} / {st['median']:{spec}} / {st['max']:{spec}}")
    ctl = ", ".join(f"{k}={v}" for k, v in (meta.get("controls") or {}).items()) or "the camera's own"
    lines = [f"frame metadata  {meta.get('started', '')}  {s['frames']} frames"
             + (f" at {s['fps']:.1f} fps" if s["fps"] else "") + (" (ended by Ctrl-C)" if meta.get("interrupted") else ""),
             f"  controls: {ctl}", "",
             "per frame (min / median / max)",
             f"  exposure us          {rng(f['exposure_us'])}",
             f"  analogue gain        {rng(f['analogue_gain'], '.2f')}",
             f"  digital gain         {rng(f['digital_gain'], '.2f')}",
             f"  colour gains r, b    {rng(f['colour_gain_r'], '.2f')}  |  {rng(f['colour_gain_b'], '.2f')}",
             f"  colour temperature K {rng(f['colour_temp_k'])}",
             f"  lux                  {rng(f['lux'], '.1f')}",
             f"  frame duration us    {rng(f['frame_duration_us'])}",
             f"  sensor -> python ms  {rng(f['sensor_to_python_ms'], '.1f')}"
             + (f"  (p95 {f['sensor_to_python_ms']['p95']:.1f})" if f["sensor_to_python_ms"] else ""),
             "  AE settled           " + ("--" if s["ae_locked_share"] is None else f"{100 * s['ae_locked_share']:.0f}% of frames"),
             "", "per label (the color branch on the traffic ROI): frames, median exposure x gain, exposure us, gain, white px"]
    for lab, d in s["labels"].items():
        med = lambda st, spec: "--" if st is None else f"{st['median']:{spec}}"            # noqa: E731
        lines.append(f"  {lab:<8} {d['frames']:>5}   {med(d['light'], '.0f'):>8}   {med(d['exposure_us'], '.0f'):>7}   "
                     f"{med(d['analogue_gain'], '.2f'):>5}   {med(d['white_px'], '.0f'):>4}")
    lines += ["", "findings"] + [f"  - {x}" for x in s["findings"] or ["nothing stood out"]]
    return lines


# =============================================================================
# Command line
# =============================================================================

def cli(argv: list[str] | None = None) -> int:
    """python3 -m src.diagnostics.frame_meta [--seconds S] [--save-every N] [--camera-control K=V] [--no-detect] [--out DIR]"""
    ap = argparse.ArgumentParser(prog="python3 -m src.diagnostics.frame_meta", description=_CLI_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=SECONDS, metavar="S", help=f"how long (default {SECONDS:.0f})")
    ap.add_argument("--save-every", type=int, default=0, metavar="N", help="keep every Nth frame as PNG (default none)")
    ap.add_argument("--camera-control", action="append", default=None, metavar="KEY=VALUE",
                    help="a libcamerasrc control for this recording, added to CAMERA_CONTROLS")
    ap.add_argument("--no-detect", action="store_true", help="metadata only, no color branch")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        controls = {**CAMERA_CONTROLS, **parse_controls(args.camera_control)}
        pic_controls = to_libcamera_controls(controls)
    except ImportError:
        print(MISSING_PICAMERA2)
        return 2
    except ValueError as exc:
        print(exc)
        return 2
    detect_fn = None
    if not args.no_detect:
        from src.config import MEASURED
        detect_fn = lambda frame, n, ts: detect(frame, n, ts, MEASURED)                    # noqa: E731
    try:
        camera = open_camera(pic_controls)
    except ImportError:
        print(MISSING_PICAMERA2)
        return 2
    except (RuntimeError, IndexError, OSError) as exc:
        print(f"couldn't open the camera ({exc}): is the robot's pipeline still running?")
        return 2
    out_dir = args.out or str(RUNS_DIR / ("meta_" + time.strftime("%Y%m%d_%H%M%S")))
    print(f"frame metadata: {args.seconds:.0f} s, output {out_dir} (Ctrl-C ends it early)", flush=True)
    record(out_dir, args.seconds, pic_controls, args.save_every, detect_fn, camera)
    print("\n" + (Path(out_dir) / "summary.txt").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(cli())
