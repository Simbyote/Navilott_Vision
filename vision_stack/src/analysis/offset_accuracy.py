"""Offset accuracy against ground truth: does the reported offset match where the robot really is?

Purpose:
    The other interpreters show offsets are produced and steady; this one
    checks they're right. Park the robot at measured lateral positions (for
    example -4, -2, 0, +2, +4 cm from lane center), record a short still run
    at each, and give the runs with their true positions. The result is the
    error at each position, the fitted scale and zero, and a verdict against
    the +/-2 cm spec.

Main package:
    analyze()             per-position bias, spread and share within spec;
                          overall fit (slope 1 and intercept 0 are perfect),
                          RMS bias, worst bias, verdict
    offset_accuracy.png   measured vs true with +/-1 std bars, the ideal line
                          and the spec band, beside the bias at each position

Input:
    One run per position, read like stability.py (p3.csv's lane_offset_cm on
    vision frames, or a normalized offset). Positions come from a manifest
    CSV with columns true_cm,run (run relative to the manifest's folder), or
    from --at TRUE_CM RUN, repeated. A normalized offset needs --cm-per-unit.

Sign convention: true_cm uses the same sign as lane_offset_cm. A negative
slope means the two disagree, which the verdict reports on its own.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from src.analysis import common, stability
from src.analysis.common import Table, fmt

SPEC_CM = 2.0
WARMUP_FRAMES = 5

_CLI_HELP = """\
Offset error against measured ground-truth positions, and a verdict against
the +/-2 cm spec. Writes offset_accuracy.png and offset_accuracy.json.

Record one still run per position, then either list them in a manifest
(columns true_cm,run) or pass them directly:
    python3 -m src.analysis.offset_accuracy positions.csv
    python3 -m src.analysis.offset_accuracy --at -4 runs/A --at 0 runs/B --at 4 runs/C

Exit status is 0 for PASS, 2 for FAIL and 1 for an input error, so a script
can gate on it.
"""


def analyze(points, spec_cm: float = SPEC_CM) -> dict:
    """
    Inputs:
        points: [(true_cm, measured_cm array)], one per position; NaNs ignored
        spec_cm: allowed absolute error

    Outputs:
        {
          "spec_cm",
          "positions": [{"true_cm", "n", "mean", "median", "std", "bias",
                         "p95_abs_error", "within_spec"}] sorted by true_cm,
          "fit": {"slope", "intercept"} of mean on true, None under 2 positions,
          "rms_bias", "worst_bias", "frames_within_spec",
          "sign_agrees": slope > 0 (None under 2 positions),
          "verdict": "PASS" when every position's bias is within spec and the
                     sign agrees, else "FAIL",
        }
    """
    rows, all_err = [], []
    for true_cm, values in sorted(points, key=lambda p: p[0]):
        v = np.asarray(values, dtype=np.float64)
        v = v[~np.isnan(v)]
        if v.size == 0:
            raise ValueError(f"position {true_cm:+g} cm has no measured frames")
        err = v - true_cm
        all_err.append(err)
        rows.append({
            "true_cm": float(true_cm), "n": int(v.size),
            "mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "bias": float(err.mean()),
            "p95_abs_error": float(np.percentile(np.abs(err), 95)),
            "within_spec": float(np.mean(np.abs(err) <= spec_cm)),
        })
    if not rows:
        raise ValueError("no positions given")

    fit = None
    if len(rows) >= 2:
        slope, intercept = np.polyfit([r["true_cm"] for r in rows], [r["mean"] for r in rows], 1)
        fit = {"slope": float(slope), "intercept": float(intercept)}
    biases = np.array([r["bias"] for r in rows])
    errors = np.concatenate(all_err)
    sign_agrees = None if fit is None else bool(fit["slope"] > 0)
    ok = bool(np.all(np.abs(biases) <= spec_cm)) and sign_agrees is not False

    return {
        "spec_cm": spec_cm,
        "positions": rows,
        "fit": fit,
        "rms_bias": float(np.sqrt(np.mean(biases ** 2))),
        "worst_bias": float(biases[np.argmax(np.abs(biases))]),
        "frames_within_spec": float(np.mean(np.abs(errors) <= spec_cm)),
        "sign_agrees": sign_agrees,
        "verdict": "PASS" if ok else "FAIL",
    }


def figure(result: dict, title: str, out_path) -> Path | None:
    plt = common.pyplot()
    if plt is None:
        return None
    pos = result["positions"]
    spec = result["spec_cm"]
    t = np.array([p["true_cm"] for p in pos])
    m = np.array([p["mean"] for p in pos])
    sd = np.array([p["std"] for p in pos])
    b = np.array([p["bias"] for p in pos])
    lo, hi = min(t.min(), m.min()) - spec - 1, max(t.max(), m.max()) + spec + 1

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    line = np.array([lo, hi])
    ax.fill_between(line, line - spec, line + spec, color="#2ca02c", alpha=0.12,
                    label=f"±{spec:g} cm spec")
    ax.plot(line, line, color="black", linewidth=1, label="ideal")
    if result["fit"]:
        f = result["fit"]
        ax.plot(line, f["slope"] * line + f["intercept"], color="#d62728", linestyle="--",
                linewidth=1, label=f"fit: {f['slope']:.2f}x {f['intercept']:+.2f}")
    ax.errorbar(t, m, yerr=sd, fmt="o", color="#1f77b4", capsize=4, label="measured ±1 std")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("true offset (cm)")
    ax.set_ylabel("measured offset (cm)")
    ax.set_title(f"{title}   {result['verdict']}", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    colors = ["#2ca02c" if abs(v) <= spec else "#d62728" for v in b]
    bx.bar([f"{v:+g}" for v in t], b, color=colors)
    for y in (-spec, spec):
        bx.axhline(y, color="black", linestyle=":", linewidth=1)
    bx.axhline(0, color="black", linewidth=0.8)
    bx.set_xlabel("true offset (cm)")
    bx.set_ylabel("bias (cm)")
    bx.set_title("mean error per position", fontsize=10, loc="left")
    bx.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def measured_cm(run, skip: int, cm_per_unit: float | None) -> np.ndarray:
    """One run's measured offsets in cm, measured frames only."""
    table = Table(common.find_csv(run, stability.CSV_NAMES), skip)
    offset, measured, _, unit = stability.load(table, normalized=cm_per_unit is not None)
    if unit != "cm":
        if cm_per_unit is None:
            raise ValueError(f"{run}: no lane_offset_cm; set the cm scale in Phase 3 "
                             "or pass --cm-per-unit")
        offset = offset * cm_per_unit
    return offset[measured]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m src.analysis.offset_accuracy",
                                description=_CLI_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", nargs="?", help="CSV with columns true_cm,run")
    p.add_argument("--at", nargs=2, action="append", metavar=("TRUE_CM", "RUN"), default=[],
                   help="one position and its run; repeat per position")
    p.add_argument("--spec", type=float, default=SPEC_CM, help=f"allowed error, cm (default {SPEC_CM})")
    p.add_argument("--cm-per-unit", type=float,
                   help="scale for a normalized offset, when runs have no lane_offset_cm")
    p.add_argument("--skip", type=int, default=WARMUP_FRAMES, help="drop the first N frames of each run")
    p.add_argument("--out", help="output folder (default: the manifest's, or the current one)")
    args = p.parse_args(argv)

    try:
        points = common.read_manifest(args.manifest, "true_cm") if args.manifest else []
        points += [(float(cm), Path(run)) for cm, run in args.at]
        if not points:
            raise ValueError("no positions: give a manifest or --at TRUE_CM RUN")
        data = [(cm, measured_cm(run, args.skip, args.cm_per_unit)) for cm, run in points]
        res = analyze(data, args.spec)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else (Path(args.manifest).parent if args.manifest else Path("."))
    common.write_json(out / "offset_accuracy.json", res)
    fig = figure(res, "Offset accuracy", out / "offset_accuracy.png")

    print(f"{'true cm':>8} {'n':>5} {'mean':>7} {'std':>6} {'bias':>7} {'in spec':>8}")
    for r in res["positions"]:
        print(f"{r['true_cm']:>+8.1f} {r['n']:>5} {r['mean']:>+7.2f} {r['std']:>6.2f} "
              f"{r['bias']:>+7.2f} {100 * r['within_spec']:>7.1f}%")
    if res["fit"]:
        print(f"fit: measured = {res['fit']['slope']:.3f} x true {res['fit']['intercept']:+.2f} cm")
    print(f"RMS bias {res['rms_bias']:.2f} cm, worst {res['worst_bias']:+.2f} cm, "
          f"{100 * res['frames_within_spec']:.1f}% of frames within ±{res['spec_cm']:g} cm")
    if res["sign_agrees"] is False:
        print("sign disagrees: measured offset moves opposite to the robot")
    print(f"verdict: {res['verdict']}")
    print(f"wrote {out / 'offset_accuracy.json'}" + (f"\nwrote {fig}" if fig else ""))
    return 0 if res["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
