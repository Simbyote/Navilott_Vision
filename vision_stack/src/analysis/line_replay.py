"""Line replay: the stop line's reported distance against the tape, over every detect-range line run in a folder.

Purpose:
    D4b asks whether the stop line's cm matches the tape. A detect-range run
    with target=line parks the robot at taped gaps and angles; since
    2026-11-03 each look logs the distance itself (reported_cm, cm_sd, the
    median of its frames). Older runs saved only the last frame of each look,
    so this replays that picture through Phase 2 for one reading per look:
    the error against the tape and its spread between placements, but no
    frame-to-frame SD. Each look is judged on D4b's tolerance (tighter near
    the robot); a line at the view's bottom (clipped) reads 0, "on the line",
    and is counted apart. The tilt fit across angles comes along.

Main package:
    analyze(folders) -> dict; line_replay.csv (one row per look), .json and
    .png in --out.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table

RUN_GLOB = "routine_detect-range_*"
NEAR_CM, TOL_NEAR_CM, TOL_FAR_CM, SD_MAX_CM = 10.0, 0.5, 1.0, 0.5     # requirements.md D4b
LIVE, REPLAY = "live", "replay"
CLIPPED, NOT_SEEN = "clipped", "not seen"
FIELDS = ("run", "time", "gap_cm", "angle_deg", "offset_cm", "seen_pct", "voted", "source", "reported_cm",
          "error_cm", "tolerance_cm", "cm_sd", "clipped", "tilt_deg", "result")
SHADES = {-10.0: "#1E63B5", -5.0: "#2A9A88", 0.0: "#1F2933", 5.0: "#E0A100", 10.0: "#C62828"}

_CLI_HELP = """\
Every detect-range line run (target=line) in the folders given: each look's
reported stop-line cm against its taped gap, judged on D4b's tolerance
(+-0.5 cm up to 10 cm, +-1.0 cm beyond; SD under 0.5 cm where the look logged
one). Runs from before 2026-11-03 have no logged distance: the look's saved
picture is replayed through Phase 2 (one frame, so no SD). Gaps are from the
robot's front edge to the line's near edge, the stop-line table's reference.

Run from vision_stack/ (on the Pi or a laptop with the runs copied over):
    python3 -m src.analysis.line_replay runs/stopline_day
    python3 -m src.analysis.line_replay runs/routine_detect-range_A runs/routine_detect-range_B --out runs/d4b
    make line-replay RUN=runs/stopline_day
"""


def run_folders(args: list[str]) -> list[Path]:
    """detect-range runs with target=line and a non-empty trials.csv, in time order."""
    found = []
    for a in args or [str(common.RUNS_DIR)]:
        p = Path(a)
        found += [p] if (p / "trials.csv").is_file() else sorted(p.rglob(RUN_GLOB)) if p.is_dir() else []
    out = [f for f in found if (f / "trials.csv").is_file() and (f / "trials.csv").stat().st_size
           and options(f).get("target") == "line"]
    if not out:
        raise FileNotFoundError(f"no detect-range line run in {', '.join(args or [str(common.RUNS_DIR)])}")
    return sorted(out, key=lambda f: f.name)


def options(folder: Path) -> dict:
    res = folder / "results.json"
    if not res.is_file() or not res.stat().st_size:
        return {}
    return json.loads(res.read_text()).get("options") or {}


def start_time(folder: Path) -> str:
    res = folder / "results.json"
    try:
        return json.loads(res.read_text())["conditions"]["start"]["time"]
    except (OSError, KeyError, ValueError, TypeError):
        return ""


def pose_tag(angle: float, offset: float) -> str:
    return (f"_{angle:g}deg" if angle else "") + (f"_{offset:g}cm" if offset else "")


def picture(folder: Path, gap: float, angle: float, offset: float) -> Path | None:
    """The last saved look at this pose (a redo saves another; the last is what the row holds)."""
    found = sorted(folder.glob(f"attempt_*_{gap:g}cm{pose_tag(angle, offset)}_present.jpg"))
    return found[-1] if found else None


def default_chain(frame):
    from src.config import MEASURED
    from src.phase2_linker import run_chain
    return run_chain(frame, 0, 0, MEASURED, parallel=False)


def replay(path: Path | None, chain) -> dict:
    """One saved picture's stop line through Phase 2: cm, clipped, tilt."""
    import cv2
    frame = cv2.imread(str(path)) if path else None
    if frame is None:
        return {"reported_cm": None, "clipped": None, "tilt_deg": None, "seen": False}
    sl = chain(frame).stop_line
    if sl is None or not sl.detected:
        return {"reported_cm": None, "clipped": False, "tilt_deg": None, "seen": False}
    return {"reported_cm": None if sl.distance_cm is None else round(float(sl.distance_cm), 2),
            "clipped": bool(sl.clipped), "tilt_deg": None if sl.tilt_deg is None else round(float(sl.tilt_deg), 2),
            "seen": True}


def tolerance(gap: float) -> float:
    return TOL_NEAR_CM if gap <= NEAR_CM else TOL_FAR_CM


def judge(row: dict) -> str:
    if row["clipped"]:
        return CLIPPED
    if row["reported_cm"] is None:
        return NOT_SEEN
    ok = abs(row["error_cm"]) <= row["tolerance_cm"] and (row["cm_sd"] is None or row["cm_sd"] < SD_MAX_CM)
    return "pass" if ok else "fail"


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def looks(folder: Path, chain=default_chain) -> list[dict]:
    t = Table(folder / "trials.csv")
    cols = {c: t.text(c) if t.has(c) else [""] * len(t) for c in
            ("gap_cm", "angle_deg", "offset_cm", "state", "seen_pct", "voted", "tilt_deg", "reported_cm", "cm_sd",
             "clipped_pct")}
    when = start_time(folder)
    out = []
    for i in range(len(t)):
        if cols["state"][i] not in ("present", ""):
            continue
        gap, angle, offset = (_num(cols[k][i]) or 0.0 for k in ("gap_cm", "angle_deg", "offset_cm"))
        row = {"run": folder.name, "time": when, "gap_cm": gap, "angle_deg": angle, "offset_cm": offset,
               "seen_pct": _num(cols["seen_pct"][i]), "voted": str(cols["voted"][i]).strip().lower() == "true",
               "tilt_deg": _num(cols["tilt_deg"][i]), "tolerance_cm": tolerance(gap)}
        live = _num(cols["reported_cm"][i])
        if live is not None or _num(cols["clipped_pct"][i]) is not None:
            row.update(source=LIVE, reported_cm=live, cm_sd=_num(cols["cm_sd"][i]),
                       clipped=live is None and (_num(cols["clipped_pct"][i]) or 0) >= 50)
        else:
            r = replay(picture(folder, gap, angle, offset), chain)
            row.update(source=REPLAY, reported_cm=r["reported_cm"], cm_sd=None, clipped=bool(r["clipped"]))
            if row["tilt_deg"] is None:
                row["tilt_deg"] = r["tilt_deg"]
        if row["clipped"]:
            row["reported_cm"] = None
        row["error_cm"] = None if row["reported_cm"] is None else round(row["reported_cm"] - gap, 2)
        row["result"] = judge(row)
        out.append(row)
    return out


def summarize(rows: list[dict]) -> dict:
    errs = [r["error_cm"] for r in rows if r["error_cm"] is not None]
    judged = [r for r in rows if r["result"] in ("pass", "fail")]
    return {"looks": len(rows), "measured": len(errs), "clipped": sum(r["result"] == CLIPPED for r in rows),
            "not_seen": sum(r["result"] == NOT_SEEN for r in rows),
            "pass": sum(r["result"] == "pass" for r in judged), "judged": len(judged),
            "median_error_cm": round(statistics.median(errs), 2) if errs else None,
            "mean_error_cm": round(statistics.fmean(errs), 2) if errs else None,
            "spread_sd_cm": round(statistics.stdev(errs), 2) if len(errs) > 1 else None,
            "worst_cm": round(max(errs, key=abs), 2) if errs else None}


def analyze(folders: list[Path], chain=default_chain) -> dict:
    from src.routines.detect_range import tilt_fit
    rows = []
    for f in folders:
        rows += looks(f, chain)
    gaps = sorted({r["gap_cm"] for r in rows})
    angles = sorted({r["angle_deg"] for r in rows})
    res = {"runs": [f.name for f in folders], "gaps": gaps, "angles": angles,
           "rule": {"near_cm": NEAR_CM, "tol_near_cm": TOL_NEAR_CM, "tol_far_cm": TOL_FAR_CM, "sd_max_cm": SD_MAX_CM},
           "sources": sorted({r["source"] for r in rows}),
           "all": summarize(rows),
           "by_gap": {g: summarize([r for r in rows if r["gap_cm"] == g]) for g in gaps},
           "by_angle": {a: summarize([r for r in rows if r["angle_deg"] == a]) for a in angles},
           "tilt_fit": tilt_fit(rows), "rows": rows}
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    a, out = res["all"], []
    if a["judged"]:
        out.append(f"{a['pass']} of {a['judged']} measured looks within tolerance; median error "
                   f"{a['median_error_cm']:+.2f} cm, spread (SD between looks) {common.fmt(a['spread_sd_cm'])} cm")
    m = a["median_error_cm"]
    if m is not None and abs(m) > 1.0 and a["spread_sd_cm"] is not None and a["spread_sd_cm"] < abs(m):
        out.append(f"the error is mostly a steady offset ({m:+.1f} cm), larger than its spread: check the tape's "
                   "reference (the robot's front edge, as the stop-line table) before blaming the fit")
    if a["clipped"]:
        near = sorted({r["gap_cm"] for r in res["rows"] if r["result"] == CLIPPED})
        out.append(f"{a['clipped']} looks had the line at the view's bottom (read as on it, 0 cm) at "
                   f"{', '.join(f'{g:g}' for g in near)} cm: the view starts about there, so nearer gaps can't be measured")
    unseen = sorted({r["gap_cm"] for r in res["rows"] if r["result"] == NOT_SEEN})
    if unseen:
        out.append(f"line not measured at {', '.join(f'{g:g}' for g in unseen)} cm in some looks (past the lane ROI's far edge, or missed)")
    if REPLAY in res["sources"]:
        out.append("replayed looks are one saved frame each: error and spread, no frame-to-frame SD; "
                   "a live run (detect-range since 2026-11-03) logs the median of every frame")
    fit = res["tilt_fit"]
    if fit:
        out.append(f"tilt fit over {fit['points']} looks: TILT_ZERO_DEG = {fit['tilt_zero_deg']}, "
                   f"TILT_TO_HEADING = {fit['tilt_to_heading']}")
    return out


def figure(res: dict, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    rows, gaps = res["rows"], res["gaps"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a in res["angles"]:
        pts = [(r["gap_cm"] + a * 0.03, r["reported_cm"]) for r in rows if r["angle_deg"] == a and r["reported_cm"] is not None]
        cl = [(r["gap_cm"] + a * 0.03, 0.0) for r in rows if r["angle_deg"] == a and r["result"] == CLIPPED]
        col = SHADES.get(a, "#5F6B76")
        if pts:
            ax[0].scatter(*zip(*pts), s=18, color=col, label=f"{a:+g} deg")
        if cl:
            ax[0].scatter(*zip(*cl), s=22, color=col, marker="x")
    xs = np.array([min(gaps) - 1, max(gaps) + 1])
    ax[0].plot(xs, xs, color="#5F6B76", lw=1)
    ax[0].fill_between(xs, xs - TOL_FAR_CM, xs + TOL_FAR_CM, color="#E8EDF1")
    ax[0].set_xlabel("tape gap, front edge to the line (cm)", fontsize=8)
    ax[0].set_ylabel("reported cm", fontsize=8)
    ax[0].set_title("Reported vs tape (band +-1 cm; x = at the view's bottom)", loc="left", fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)
    for k, g in enumerate(gaps):
        e = [r["error_cm"] for r in rows if r["gap_cm"] == g and r["error_cm"] is not None]
        if e:
            ax[1].boxplot([e], positions=[k], widths=0.5, showfliers=True)
            ax[1].text(k, min(e) - 0.4, f"n={len(e)}", ha="center", fontsize=7, color="#5F6B76")
    for k, g in enumerate(gaps):
        t = tolerance(g)
        ax[1].plot([k - 0.35, k + 0.35], [t, t], color="#B26A00", lw=1)
        ax[1].plot([k - 0.35, k + 0.35], [-t, -t], color="#B26A00", lw=1)
    ax[1].axhline(0, color="#5F6B76", lw=0.8)
    ax[1].set_xticks(range(len(gaps)), [f"{g:g}" for g in gaps])
    ax[1].set_xlabel("tape gap (cm)", fontsize=8)
    ax[1].set_ylabel("error, reported - tape (cm)", fontsize=8)
    ax[1].set_title("Error by gap (orange: D4b tolerance)", loc="left", fontsize=10)
    pts = [(r["angle_deg"], r["tilt_deg"]) for r in rows if r["tilt_deg"] is not None]
    fit = res["tilt_fit"]
    if pts:
        a_, t_ = np.array(pts).T
        ax[2].scatter(a_, t_, s=14, color="#1E63B5", alpha=0.6)
        if fit:
            xa = np.array([min(a_) - 1, max(a_) + 1])
            ax[2].plot(xa, fit["tilt_zero_deg"] + xa / fit["tilt_to_heading"], color="#C62828",
                       label=f"zero {fit['tilt_zero_deg']}, to heading {fit['tilt_to_heading']}")
            ax[2].legend(fontsize=7, frameon=False)
    ax[2].set_xlabel("robot turned from square (deg)", fontsize=8)
    ax[2].set_ylabel("measured tilt (deg)", fontsize=8)
    ax[2].set_title("Tilt vs angle", loc="left", fontsize=10)
    for x in ax:
        x.tick_params(labelsize=8)
        x.grid(alpha=0.25)
    a = res["all"]
    fig.suptitle(f"Stop-line distance (D4b): {len(res['runs'])} runs, {a['looks']} looks, "
                 f"{a['pass']}/{a['judged']} within tolerance ({', '.join(res['sources'])})", x=0.01, ha="left")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    lines = [f"{len(res['runs'])} runs, sources: {', '.join(res['sources'])}",
             f"  {'gap':>5} {'looks':>5} {'pass':>8} {'clipped':>7} {'unseen':>6} {'median err':>10} "
             f"{'spread sd':>9} {'worst':>6}"]
    for g, s in res["by_gap"].items():
        lines.append(f"  {g:>5g} {s['looks']:>5} {s['pass']:>3}/{s['judged']:<4} {s['clipped']:>7} {s['not_seen']:>6} "
                     f"{common.fmt(s['median_error_cm']):>10} {common.fmt(s['spread_sd_cm']):>9} "
                     f"{common.fmt(s['worst_cm']):>6}")
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def write_rows(path, rows) -> Path:
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in FIELDS})
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.line_replay", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="detect-range line runs, or folders holding them (default: runs/)")
    p.add_argument("--out", help="output folder (default: the first argument's folder)")
    args = p.parse_args(argv)
    try:
        res = analyze(run_folders(args.runs))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    first = Path(args.runs[0]) if args.runs else Path(common.RUNS_DIR)
    out = Path(args.out) if args.out else (first if not (first / "trials.csv").is_file() else first.parent)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "line_replay.csv", res["rows"])
    common.write_json(out / "line_replay.json", {k: v for k, v in res.items() if k != "rows"})
    fig = figure(res, out / "line_replay.png")
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'line_replay.csv'}\nwrote {out / 'line_replay.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
