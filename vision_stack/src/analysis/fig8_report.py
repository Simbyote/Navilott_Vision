"""Figure-eight report: every turn, every lap and where the chassis didn't follow its wheels, as graphs.

Purpose:
    routine_report draws a routine's criteria; the figure eight's story is in
    its run: which turns ended on the gyro and which on the time limit, how
    far each landed from its 90 degrees, what braked the robot in each lap,
    and the spots where the gyro and the wheels disagreed. With --base (an
    earlier run), its turn errors sit beside this run's, by turn position.

Main package:
    analyze(attempt, base) -> dict; fig8_report.png and fig8_report.json in
    the attempt folder (runs/routine_figure-eight_<time>/attempt_NN/).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common
from src.analysis.common import Table
from src.navigation.intersection import LEFT_TURN_MAX_MS, RIGHT_TURN_MAX_MS, TURN_END_GYRO, TURN_END_TIME

TURNS_PER_LAP = 8               # routines.figure_eight.ROUTE_LAP: four lefts, four rights
TURN_LIMIT_DEG = 20.0           # the routine's tolerance on a turn's end
LIMIT_S = {"left": LEFT_TURN_MAX_MS / 1000.0, "right": RIGHT_TURN_MAX_MS / 1000.0}
COLORS = {"left": "#1E63B5", "right": "#2A9A88"}
BRAKES = (("red_light", "red light", "#C62828"), ("stop_sign", "stop sign", "#1baf7a"),
          ("obstacle", "obstacle", "#e87ba4"))
SLIP, LANE, MUTED = "#B26A00", "#EEF2F5", "#5F6B76"

_CLI_HELP = """\
A figure-eight run's turns and laps: heading through every turn, each turn's
end against its 90 degrees, each lap on a strip with its turns, brakes,
touches and slips. Writes fig8_report.png and fig8_report.json in the
attempt folder.

Run from vision_stack/:
    python3 -m src.analysis.fig8_report                       newest figure-eight run
    python3 -m src.analysis.fig8_report runs/routine_figure-eight_<time>
    python3 -m src.analysis.fig8_report NEW --base OLD        an earlier run's turn errors beside
    make fig8-report RUN=runs/routine_figure-eight_<time> [BASE=...]
"""


def find(arg) -> Path:
    """The attempt folder (with intersections.csv): arg itself, the newest attempt in it, or the newest run's."""
    roots = [Path(arg)] if arg else [Path(common.RUNS_DIR)]
    for root in roots:
        if (root / "intersections.csv").is_file():
            return root
    found = [p.parent for r in roots if r.is_dir() for p in r.rglob("intersections.csv")]
    if not found:
        raise FileNotFoundError(f"no figure-eight attempt (intersections.csv) in {', '.join(str(r) for r in roots)}")
    return max(found, key=lambda p: (p / "intersections.csv").stat().st_mtime)


def _opt(path: Path) -> Table | None:
    try:
        return Table(path) if path.is_file() else None
    except ValueError:
        return None


def step_num(label: str) -> float:
    try:
        return float(str(label).split("/")[0])
    except ValueError:
        return float("nan")


def turns_of(nav: Table) -> list[dict]:
    """Each turn's frames: step, maneuver, seconds into the turn, heading."""
    if not all(nav.has(c) for c in ("t", "stage", "step", "heading_deg")):
        return []
    t, stage, step = nav.numeric("t"), nav.text("stage"), [step_num(s) for s in nav.text("step")]
    man = nav.text("maneuver") if nav.has("maneuver") else [""] * len(nav)
    hd = nav.numeric("heading_deg")
    out = {}
    for i in range(len(nav)):
        if stage[i] == "turn" and np.isfinite(step[i]):
            out.setdefault(step[i], {"step": int(step[i]), "maneuver": man[i], "t": [], "heading": []})
            out[step[i]]["t"].append(t[i])
            out[step[i]]["heading"].append(hd[i])
    for d in out.values():
        d["t"] = (np.array(d["t"]) - d["t"][0]).tolist()
    return list(out.values())


def analyze(folder: Path, base: Path | None = None) -> dict:
    inter = Table(folder / "intersections.csv")
    laps = _opt(folder / "laps.csv")
    slips = _opt(folder / "slips.csv")
    touches = _opt(folder / "touches.csv")
    steps, ends, off = inter.numeric("step"), inter.text("turn_end"), inter.numeric("heading_off_deg")
    done = [i for i in range(len(inter)) if ends[i] in (TURN_END_GYRO, TURN_END_TIME)]
    timed = [int(steps[i]) for i in done if ends[i] == TURN_END_TIME]
    pos = lambda s: int((s - 1) % TURNS_PER_LAP + 1)            # noqa: E731
    by_pos = {}
    for i in done:
        if np.isfinite(off[i]):
            by_pos.setdefault(pos(steps[i]), []).append(float(off[i]))
    res = {"turns": len(done), "on_gyro": len(done) - len(timed), "time_limit_steps": timed,
           "time_limit_positions": sorted({pos(s) for s in timed}),
           "worst_off_deg": float(np.nanmax(np.abs(off[done]))) if done else None,
           "over_limit": int(sum(abs(off[i]) > TURN_LIMIT_DEG for i in done if np.isfinite(off[i]))),
           "off_by_position": {k: {"median": float(np.median(v)), "n": len(v)} for k, v in sorted(by_pos.items())},
           "laps": [] if laps is None else [{"lap": int(a), "seconds": float(b)} for a, b in
                                             zip(laps.numeric("lap"), laps.numeric("seconds"))],
           "slips": 0 if slips is None else len(slips), "touches": 0 if touches is None else len(touches)}
    if slips is not None and len(slips):
        spots = {}
        for st, stg in zip(slips.numeric("step"), slips.text("stage")):
            if np.isfinite(st):
                key = f"turn {pos(st)} ({stg or 'lane'})"
                spots[key] = spots.get(key, 0) + 1
        res["slip_spots"] = dict(sorted(spots.items(), key=lambda kv: -kv[1]))
    if base:
        res["base"] = analyze(base)
        res["base"].pop("base", None)
    res["findings"] = findings(res)
    return res


def findings(res: dict) -> list[str]:
    out = []
    if res["time_limit_steps"]:
        same = len(res["time_limit_positions"]) == 1
        out.append(f"{len(res['time_limit_steps'])} of {res['turns']} turns ended on the time limit, at position"
                   f"{'' if same else 's'} {', '.join(map(str, res['time_limit_positions']))} of the lap"
                   + (": the same corner every time, so look at that corner" if same and len(res["time_limit_steps"]) > 1 else ""))
    if res["over_limit"]:
        out.append(f"{res['over_limit']} turns ended more than {TURN_LIMIT_DEG:.0f} deg off (worst {res['worst_off_deg']:.1f})")
    if res.get("slip_spots"):
        top, n = next(iter(res["slip_spots"].items()))
        out.append(f"the chassis didn't follow its wheels {res['slips']} times; most at {top} ({n})")
    if len(res["laps"]) > 2:
        secs = [x["seconds"] for x in res["laps"]]
        if max(secs) > 1.25 * min(secs):
            out.append(f"laps took {min(secs):.0f}-{max(secs):.0f} s: the slow ones held at a light or sign, or stalled")
    b = res.get("base")
    if b and b["turns"]:
        out.append(f"against the base run: time-limit turns {len(b['time_limit_steps'])} -> {len(res['time_limit_steps'])}, "
                   f"worst end {b['worst_off_deg'] or 0:.1f} -> {res['worst_off_deg'] or 0:.1f} deg, "
                   f"touches {b['touches']} -> {res['touches']}")
    return out


def _turn_curves(ax, nav: Table, timed: set):
    for d in turns_of(nav):
        c = COLORS.get(d["maneuver"], MUTED)
        ax.plot(d["t"], d["heading"], color=c, lw=1.2 if d["step"] in timed else 0.7,
                ls="--" if d["step"] in timed else "-", alpha=1 if d["step"] in timed else 0.55)
    for m, s in LIMIT_S.items():
        ax.axvline(s, color=COLORS[m], ls=":", lw=0.9)
        ax.text(s, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5, f"{m} limit", color=COLORS[m], fontsize=7,
                ha="right", va="top")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("seconds into the turn (dashed: ended on the time limit)", fontsize=8)
    ax.set_ylabel("heading (deg)", fontsize=8)


def _turn_errors(ax, inter: Table, base: Path | None):
    steps, ends, man, off = inter.numeric("step"), inter.text("turn_end"), inter.text("maneuver"), \
        inter.numeric("heading_off_deg")
    for m in ("left", "right"):
        sel = [i for i in range(len(inter)) if man[i] == m and ends[i] in (TURN_END_GYRO, TURN_END_TIME)]
        gy = [i for i in sel if ends[i] == TURN_END_GYRO]
        tl = [i for i in sel if ends[i] == TURN_END_TIME]
        ax.plot(steps[gy], off[gy], "o", color=COLORS[m], ms=3.5, label=f"{m} turns")
        ax.plot(steps[tl], off[tl], "o", mfc="white", mec=COLORS[m], ms=4.5, mew=1)
    if base is not None:
        b = Table(base / "intersections.csv")
        bs, be, bo = b.numeric("step"), b.text("turn_end"), b.numeric("heading_off_deg")
        keep = [i for i in range(len(b)) if be[i] in (TURN_END_GYRO, TURN_END_TIME)]
        ax.plot(bs[keep], bo[keep], "x", color=MUTED, ms=3.5, alpha=0.7, label="base run")
    for lap in range(1, int(np.nanmax(steps) // TURNS_PER_LAP) + 2 if len(steps) else 1):
        ax.axvline(lap * TURNS_PER_LAP + 0.5, color=MUTED, ls=":", lw=0.5)
    for s in (TURN_LIMIT_DEG, -TURN_LIMIT_DEG):
        ax.axhline(s, color=SLIP, ls="--", lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("turn, in order (hollow: ended on the time limit; dotted: a new lap)", fontsize=8)
    ax.set_ylabel("off its 90 deg at the end", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))


def _lap_strip(ax, nav: Table, inter: Table, slips: Table | None, touches: Table | None):
    t = nav.numeric("t")
    steps_i, t_enter = inter.numeric("step"), inter.numeric("t_enter")
    firsts = [t_enter[i] for i in range(len(inter)) if (steps_i[i] - 1) % TURNS_PER_LAP == 0 and np.isfinite(t_enter[i])]
    starts = firsts + [float(np.nanmax(t))]
    stage, step, reason = nav.text("stage"), [step_num(s) for s in nav.text("step")], \
        nav.text("reason") if nav.has("reason") else [""] * len(nav)
    man = nav.text("maneuver") if nav.has("maneuver") else [""] * len(nav)
    ends = dict(zip(inter.numeric("step"), inter.text("turn_end")))
    for k, (t0, t1) in enumerate(zip(starts, starts[1:])):
        ax.barh(k, t1 - t0, height=0.62, color=LANE)
        inside = [i for i in range(len(nav)) if t0 <= t[i] < t1]
        turns = {}
        for i in inside:
            if stage[i] == "turn" and np.isfinite(step[i]):
                turns.setdefault(step[i], [t[i], t[i], man[i]])[1] = t[i]
        for s, (a, b, m) in turns.items():
            ax.barh(k, b - a + 0.05, left=a - t0, height=0.62, color=COLORS.get(m, MUTED), alpha=0.6)
            if ends.get(s) == TURN_END_TIME:
                ax.plot(b - t0, k - 0.42, "v", color=SLIP, ms=4)
            if k == 0:
                ax.text(a - t0, -0.55, f"{int((s - 1) % TURNS_PER_LAP + 1)}", fontsize=7, va="bottom")
        for key, _, c in BRAKES:
            br = [t[i] - t0 for i in inside if reason[i].startswith(key)]
            ax.plot(br, [k + 0.4] * len(br), "|", color=c, ms=3, alpha=0.6)
        if slips is not None:
            for a, b in zip(slips.numeric("t0"), slips.numeric("t1")):
                if t0 <= a < t1:
                    ax.barh(k, max(b - a, 0.25), left=a - t0, height=0.62, color=SLIP)
        if touches is not None:
            for x in touches.numeric("t"):
                if t0 <= x < t1:
                    ax.plot(x - t0, k, "x", color="#C62828", ms=6, mew=1.5)
    n = max(len(starts) - 1, 1)
    ax.set_yticks(range(n), [f"lap {k + 1}" for k in range(n)], fontsize=7)
    ax.set_ylim(n - 0.5, -0.9)
    ax.set_xlabel("seconds into the lap (blue: lefts, green: rights, numbered on top; orange: chassis not following "
                  "its wheels; ▼ time limit; ticks: braked; x: a touch)", fontsize=7.5)


def figure(folder: Path, res: dict, base: Path | None, title: str, out_path):
    plt = common.pyplot()
    if plt is None:
        return None
    nav = _opt(folder / "nav.csv")
    inter = Table(folder / "intersections.csv")
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.4, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    if nav is not None:
        _turn_curves(ax, nav, set(res["time_limit_steps"]))
    ax.set_title(title, loc="left", fontsize=10)
    _turn_errors(fig.add_subplot(gs[0, 1]), inter, base)
    ax = fig.add_subplot(gs[1, :])
    if nav is not None:
        _lap_strip(ax, nav, inter, _opt(folder / "slips.csv"), _opt(folder / "touches.csv"))
    for a in fig.axes:
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def report_lines(res: dict) -> list[str]:
    lines = [f"{res['turns']} turns, {res['on_gyro']} on the gyro; worst end {res['worst_off_deg'] or 0:.1f} deg off; "
             f"{len(res['laps'])} laps; {res['touches']} touches; {res['slips']} slips",
             "median off its 90 deg by position in the lap: "
             + ", ".join(f"{k}: {v['median']:+.1f}" for k, v in res["off_by_position"].items())]
    return lines + ["findings"] + [f"  - {x}" for x in res["findings"] or ["nothing stood out"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.fig8_report", description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", help="figure-eight routine folder or attempt (default: the newest)")
    p.add_argument("--base", help="an earlier figure-eight run to compare with")
    p.add_argument("--out", help="output folder (default: the attempt folder)")
    args = p.parse_args(argv)
    try:
        folder = find(args.run)
        base = find(args.base) if args.base else None
        res = analyze(folder, base)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else folder
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "fig8_report.json", res)
    fig = figure(folder, res, base, f"Figure eight: {folder.parent.name}/{folder.name}", out / "fig8_report.png")
    print(folder)
    print("\n".join(report_lines(res)))
    print(f"wrote {out / 'fig8_report.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
