"""Camera metadata: what auto exposure and white balance did over a recording, as graphs.

Purpose:
    libcamera's auto loops judge the whole frame and settle wherever the scene
    leads them, so a sign can read red one run and purple the next with nothing
    in the room moved (2026-10-26). frame_meta records their choices per frame;
    this draws them on one time axis, with when they settled and how far they
    swung, so two recordings can be compared.

Main package:
    analyze() -> dict; camera_meta.png (exposure, gain, colour gains and
    temperature, lux, sensor-to-Python time, the traffic label) and
    camera_meta.json beside the frames.csv. Reads frame_meta's frames.csv,
    from make frame-meta or pytest --hardware.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table

CSV_NAME = "frames.csv"
# Settled once exposure x gain stays within this of the recording's last-quarter median
SETTLE_BAND = 0.10
# A frame whose exposure is this close to its frame duration has run out of shutter: AE raises gain
SHUTTER_LIMIT = 0.9
# Findings past these
LIGHT_SWING = 2.0               # max / min of exposure x gain after settling
TEMP_SWING_K = 500.0            # colour temperature range after settling
LABEL_COLORS = {"red": "#d62728", "yellow": "#e6b400", "green": "#2ca02c"}

_CLI_HELP = """\
Graph what the camera's auto exposure and white balance chose per frame:
exposure, gains, colour gains and temperature, lux, sensor-to-Python time.
Writes camera_meta.png and camera_meta.json next to the frames.csv.

Run from vision_stack/:
    python3 -m src.analysis.camera_meta                       newest frame_meta recording
    python3 -m src.analysis.camera_meta runs/meta_<time>
    make camera-meta RUN=runs/meta_<time>

Record one with make frame-meta (CAM="..." for other controls); pytest
--hardware records three seconds too, and make hw-report draws it. Compare a
recording where the stop sign failed with one where it worked, from the same
spot: what differs in these graphs is what the camera changed.
"""


def is_frame_meta(path: Path) -> bool:
    """frame_meta's frames.csv, not another tool's file of the same name."""
    try:
        return Table(path, 0).has("exposure_us")
    except (OSError, ValueError):
        return False


def find(arg) -> Path:
    """arg as given, a folder's frame_meta frames.csv, or the newest one under runs/ and artifacts/."""
    if arg and Path(arg).is_file():
        return Path(arg)
    roots = [Path(arg)] if arg else [Path(r) for r in common.SEARCH_ROOTS]
    found = [p for r in roots if r.is_dir() for p in r.rglob(CSV_NAME) if is_frame_meta(p)]
    if not found:
        raise FileNotFoundError(f"no frame_meta {CSV_NAME} in {', '.join(str(r) for r in roots)}")
    return max(found, key=lambda p: p.stat().st_mtime)


def time_axis(table: Table) -> tuple[np.ndarray, str]:
    """Seconds from the first frame, or the frame number when the recording has no sensor times."""
    t = table.numeric("t_s") if table.has("t_s") else np.full(len(table), np.nan)
    if np.isfinite(t).sum() >= 2:
        return t - np.nanmin(t), "s"
    return np.arange(len(table), dtype=np.float64), "frame"


def light(table: Table) -> np.ndarray:
    """Exposure x analogue x digital gain: the light the sensor gathered, per frame."""
    gain = np.ones(len(table))
    for col in ("analogue_gain", "digital_gain"):
        if table.has(col):
            g = table.numeric(col)
            gain = gain * np.where(np.isnan(g), 1.0, g)
    return table.numeric("exposure_us") * gain


def settle_index(values: np.ndarray, band: float = SETTLE_BAND) -> int | None:
    """The first frame from which values stay within band of their last quarter's median; None without values."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return None
    tail = v[ok][-max(1, ok.sum() // 4):]
    final = float(np.median(tail))
    outside = ok & (np.abs(v - final) > band * abs(final))
    idx = np.flatnonzero(outside)
    return 0 if idx.size == 0 else min(int(idx[-1]) + 1, len(v) - 1)


def _swing(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v) & (v > 0)]
    return float(v.max() / v.min()) if v.size else None


def analyze(table: Table) -> dict:
    """The recording's ranges, when AE settled, how far it swung after, and findings. ValueError for another CSV."""
    if not table.has("exposure_us"):
        raise ValueError(f"{table.path} isn't a frame_meta frames.csv (no exposure_us)")
    t, unit = time_axis(table)
    gathered = light(table)
    s = settle_index(gathered)
    after = slice(s or 0, None)
    exp, dur = table.numeric("exposure_us"), (table.numeric("frame_duration_us") if table.has("frame_duration_us")
                                             else np.full(len(table), np.nan))
    limited = np.isfinite(exp) & np.isfinite(dur) & (exp >= SHUTTER_LIMIT * dur)
    temp = table.numeric("colour_temp_k") if table.has("colour_temp_k") else np.full(len(table), np.nan)
    temp_after = temp[after][np.isfinite(temp[after])]
    labels = table.text("label") if table.has("label") else [""] * len(table)
    res = {
        "frames": len(table), "time_unit": unit,
        "duration": float(np.nanmax(t)) if len(t) else 0.0,
        "fields": {k: common.stats(table.numeric(k)) for k in
                   ("exposure_us", "analogue_gain", "digital_gain", "colour_gain_r", "colour_gain_b",
                    "colour_temp_k", "lux", "frame_duration_us", "sensor_to_python_ms") if table.has(k)},
        "light": common.stats(gathered),
        "settled_at": None if s is None else float(t[s]),
        "light_swing_after_settle": _swing(gathered[after]),
        "temp_range_after_settle_k": float(temp_after.max() - temp_after.min()) if temp_after.size else None,
        "shutter_limited_share": float(limited.mean()) if len(table) else None,
        "labels": {lab or "none": labels.count(lab) for lab in sorted(set(labels))},
    }
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    unit = res["time_unit"]
    if res["settled_at"] is not None and res["settled_at"] > 0.25 * res["duration"]:
        out.append(f"auto exposure took until {res['settled_at']:.1f} {unit} of {res['duration']:.1f} to settle: "
                   "frames before then saw a different picture")
    sw = res["light_swing_after_settle"]
    if sw is not None and sw > LIGHT_SWING:
        out.append(f"after settling, the light gathered (exposure x gain) still swung {sw:.1f}x: the scene or the "
                   "light through the windows kept moving it; fixed exposure (exposure-time, analogue-gain) stops it")
    tr = res["temp_range_after_settle_k"]
    if tr is not None and tr > TEMP_SWING_K:
        out.append(f"white balance moved {tr:.0f} K after settling: colours shift with it; an awb-mode preset bounds it")
    if res["shutter_limited_share"] and res["shutter_limited_share"] > 0.5:
        out.append(f"exposure was at the frame's length on {100 * res['shutter_limited_share']:.0f}% of frames: "
                   "the scene is dim for 20 FPS, so AE used gain (noise) instead")
    return out


def figure(table: Table, res: dict, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    t, unit = time_axis(table)
    panels = [("exposure (ms)", [("exposure_us", 1e-3, "#1f77b4", "exposure"),
                                 ("frame_duration_us", 1e-3, "#7f7f7f", "frame length")]),
              ("gain", [("analogue_gain", 1, "#ff7f0e", "analogue"), ("digital_gain", 1, "#8c564b", "digital")]),
              ("colour gains", [("colour_gain_r", 1, "#d62728", "red"), ("colour_gain_b", 1, "#1f77b4", "blue")]),
              ("lux", [("lux", 1, "#2ca02c", "lux")]),
              ("sensor→python (ms)", [("sensor_to_python_ms", 1, "#9467bd", "latency")])]
    panels = [(name, [s for s in series if table.has(s[0])]) for name, series in panels]
    panels = [p for p in panels if p[1]]
    labels = table.text("label") if table.has("label") else []
    strip = any(lab in LABEL_COLORS for lab in labels)
    n = len(panels) + (1 if strip else 0)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.7 * n + 0.6), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3] * len(panels) + ([0.6] if strip else [])})
    axes = axes[:, 0]
    for ax, (name, series) in zip(axes, panels):
        for col, scale, color, lab in series:
            dashed = col == "frame_duration_us"
            ax.plot(t, table.numeric(col) * scale, color=color, lw=0.9, ls="--" if dashed else "-", label=lab)
        if name == "colour gains" and table.has("colour_temp_k"):
            ax2 = ax.twinx()
            ax2.plot(t, table.numeric("colour_temp_k"), color="#bcbd22", lw=0.9, label="colour temp")
            ax2.set_ylabel("K", fontsize=8)
            ax2.tick_params(labelsize=7)
            ax2.legend(fontsize=7, loc="upper right", frameon=False)
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
        if len(series) > 1:
            ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=len(series))
        if res["settled_at"]:
            ax.axvspan(t[0], res["settled_at"], color="#cccccc", alpha=0.35, lw=0)
        ax.grid(alpha=0.25)
    if strip:
        ax = axes[-1]
        step = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        for x, lab in zip(t, labels):
            if lab in LABEL_COLORS:
                ax.axvspan(x - step / 2, x + step / 2, color=LABEL_COLORS[lab], lw=0)
        ax.set_yticks([])
        ax.set_ylabel("light", fontsize=8, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("time (s)" if unit == "s" else "frame")
    settled = "" if not res["settled_at"] else f"   (gray: AE settling, to {res['settled_at']:.1f} {unit})"
    axes[0].set_title(title + settled, fontsize=10, loc="left")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    f = res["fields"]
    rng = lambda k, spec: "--" if k not in f or f[k]["n"] == 0 else (                   # noqa: E731
        f"{f[k]['min']:{spec}} / {f[k]['p50']:{spec}} / {f[k]['max']:{spec}}")
    fmt = lambda v, spec: "--" if v is None else f"{v:{spec}}"                          # noqa: E731
    unit = res["time_unit"]
    lines = [f"{res['frames']} frames over {res['duration']:.1f} {unit}",
             "per frame (min / median / max)",
             f"  exposure us          {rng('exposure_us', '.0f')}",
             f"  analogue gain        {rng('analogue_gain', '.2f')}",
             f"  digital gain         {rng('digital_gain', '.2f')}",
             f"  colour gain r / b    {rng('colour_gain_r', '.2f')}  |  {rng('colour_gain_b', '.2f')}",
             f"  colour temperature K {rng('colour_temp_k', '.0f')}",
             f"  lux                  {rng('lux', '.1f')}",
             f"  sensor -> python ms  {rng('sensor_to_python_ms', '.1f')}",
             f"AE settled at {fmt(res['settled_at'], '.2f')} {unit}; after it, light swung "
             f"{fmt(res['light_swing_after_settle'], '.2f')}x and white balance {fmt(res['temp_range_after_settle_k'], '.0f')} K",
             "traffic label: " + ", ".join(f"{k} {v}" for k, v in res["labels"].items())]
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.camera_meta", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="frame_meta folder or frames.csv (default: the newest)")
    p.add_argument("--out", help="output folder (default: next to the CSV)")
    args = p.parse_args(argv)
    try:
        path = find(args.run)
        table = Table(path)
        res = analyze(table)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else path.parent
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "camera_meta.json", res)
    fig = figure(table, res, f"Camera metadata: {path.parent.name}", out / "camera_meta.png")
    print(f"{path}")
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'camera_meta.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
