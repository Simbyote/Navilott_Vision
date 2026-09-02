"""
analyze_run.py

Bench-run triage for the post-fix log format (run_pipeline patch 11).

    python3 analyze_run.py session3.log
    python3 analyze_run.py session3.log --baseline    # print session2 comparison

Purpose:
    Turn a run log into the five numbers that decide "stop and diagnose" vs
    "tune in place". Every metric here has a known pre-fix value from
    session2.log, so a regression is visible without re-reading 3,000 lines.

    The single most important number is SEAM DISAGREEMENT: frames where Phase 2
    reported a lane and Phase 3 refused it. That was 523 frames (81% of all
    dead-reckoning holds) pre-fix. It should now be ~0. If it is not, stop --
    the seam contract is still broken and no amount of gain tuning will help.
"""
import re
import sys
import statistics
from collections import Counter

# Pre-fix reference values, measured from session2.log (2,882 frames, 03:51).
BASELINE = {
    "frames":              2882,
    "hold_rate_pct":       22.4,
    "seam_disagree":       523,
    "seam_disagree_pct":   18.1,
    "longest_hold":        39,
    "mode_none_pct":       4.3,
    "over_budget_pct":     11.2,
    "d_offset_mean":       0.0357,
    "d_offset_p95":        0.1065,
    "d_offset_max":        0.1858,
    "sign_flips_per_100":  None,
}

# Post-fix log line (patch 11).
PAT = re.compile(
    r"f=(?P<f>\d+)\s+t=(?P<t>[\d.]+)ms\s+off=(?P<off>[+-][\d.]+)\s+n=(?P<n>[+-][\d.]+)\s+"
    r"(?P<val>OK|DR)\s+age=(?P<age>\d+)\s*(?P<stale>STALE)?\s+head=(?P<head>[+-][\d.]+)\S*\s+"
    r"src=(?P<src>\S+)\s+drive=(?P<drive>\S+)\s+stop_sign=(?P<ss>\w)\s+"
    r"p2_mode=(?P<p2>\S+)\s+p3_mode=(?P<p3>\S+)\s+imu_n=(?P<imu>\d+)\s+yaw=(?P<yaw>[+-][\d.]+)"
)

# Legacy format, so a pre-fix log can still be summarised for comparison.
PAT_OLD = re.compile(
    r"f=(\d+) t=([\d.]+)ms offset=([+-][\d.]+) head=([+-][\d.]+). drive=(\S+)\s+"
    r"stop_sign=(\w) lane_mode=(\S+) imu_n=(\d+) yaw=([+-][\d.]+)"
)

LOOP_BUDGET_MS = 1000.0 / 15


def parse(path):
    rows, legacy = [], False
    for line in open(path, errors="ignore"):
        m = PAT.search(line)
        if m:
            g = m.groupdict()
            rows.append(dict(
                f=int(g["f"]), t=float(g["t"]), off=float(g["off"]), n=float(g["n"]),
                valid=(g["val"] == "OK"), age=int(g["age"]), stale=bool(g["stale"]),
                head=float(g["head"]), src=g["src"], drive=g["drive"],
                p2=g["p2"], p3=g["p3"], imu=int(g["imu"]), yaw=float(g["yaw"]),
            ))
            continue
        m = PAT_OLD.search(line)
        if m:
            legacy = True
            rows.append(dict(
                f=int(m.group(1)), t=float(m.group(2)), off=float(m.group(3)), n=None,
                valid=None, age=None, stale=None, head=float(m.group(4)),
                src=None, drive=m.group(5), p2=m.group(7), p3=None,
                imu=int(m.group(8)), yaw=float(m.group(9)),
            ))
    return rows, legacy


def hdr(s):
    print(f"\n{s}\n" + "-" * 74)


def verdict(label, value, base, worse_if_higher=True, tol=0.0):
    """Print a metric next to its pre-fix baseline with a direction marker."""
    if base is None:
        print(f"  {label:<38} {value}")
        return
    better = (value < base - tol) if worse_if_higher else (value > base + tol)
    worse = (value > base + tol) if worse_if_higher else (value < base - tol)
    mark = "BETTER" if better else ("WORSE" if worse else "same")
    print(f"  {label:<38} {value:<12} (pre-fix {base:<8}) {mark}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    rows, legacy = parse(path)
    if not rows:
        print(f"No navigation frames parsed from {path}.")
        print("Check that run_pipeline patch 11 (log format) was applied.")
        sys.exit(1)

    n = len(rows)
    print("=" * 74)
    print(f"  {path}   {n} frames" + ("   [LEGACY pre-fix format]" if legacy else ""))
    print("=" * 74)

    # ---------------------------------------------------------------- timing
    hdr("TIMING")
    ts = sorted(r["t"] for r in rows)
    over = sum(1 for t in ts if t > LOOP_BUDGET_MS)
    print(f"  mean {statistics.mean(ts):.1f} ms   median {statistics.median(ts):.1f} ms   "
          f"p95 {ts[int(.95 * n)]:.1f} ms   max {max(ts):.1f} ms")
    verdict("over budget (%)", round(100 * over / n, 1), BASELINE["over_budget_pct"])

    # ------------------------------------------------------- SEAM AGREEMENT
    hdr("SEAM AGREEMENT  <-- the number that decides stop vs tune")
    if legacy:
        print("  legacy log: p3_mode absent, cannot measure. Re-run with patch 11.")
    else:
        # Phase 2 found a lane, Phase 3 refused it.
        disagree = [r for r in rows if r["p2"] != "none" and not r["valid"]]
        pct = round(100 * len(disagree) / n, 1)
        verdict("frames P2 saw a lane, P3 held", len(disagree), BASELINE["seam_disagree"])
        verdict("  as % of run", pct, BASELINE["seam_disagree_pct"])
        if disagree:
            print(f"  modes P2 reported while P3 held: "
                  f"{dict(Counter(r['p2'] for r in disagree).most_common(5))}")
        if len(disagree) > 0.02 * n:
            print("\n  >>> STOP. The seam is still dropping valid measurements.")
            print("  >>> Do not tune gains. Capture this log and diagnose.")
        else:
            print("\n  Seam is clean. Any remaining misbehaviour is perception or tuning.")

    # ------------------------------------------------------- dead-reckoning
    hdr("DEAD-RECKONING / HOLDS")
    if legacy:
        runs, start = [], 0
        for i in range(1, n + 1):
            if i == n or rows[i]["off"] != rows[start]["off"]:
                if i - start >= 2:
                    runs.append(i - start)
                start = i
        held = sum(runs)
        verdict("frames inside a hold (%)", round(100 * held / n, 1), BASELINE["hold_rate_pct"])
        verdict("longest hold (frames)", max(runs) if runs else 0, BASELINE["longest_hold"])
    else:
        dr = [r for r in rows if not r["valid"]]
        stale = [r for r in rows if r["stale"]]
        verdict("frames dead-reckoned (%)", round(100 * len(dr) / n, 1), BASELINE["hold_rate_pct"])
        verdict("longest hold (max age)", max((r["age"] for r in rows), default=0),
                BASELINE["longest_hold"])
        print(f"  frames flagged STALE                   {len(stale)} "
              f"({100 * len(stale) / n:.1f}%)   [pre-fix: field did not exist]")
        print(f"  heading source mix                     "
              f"{dict(Counter(r['src'] for r in rows))}")

    # -------------------------------------------------------- lane coverage
    hdr("LANE COVERAGE (perception health -- commit 5 / 6 sensitive)")
    p2c = Counter(r["p2"] for r in rows)
    for mode, cnt in p2c.most_common():
        print(f"  p2_mode {mode:<16} {cnt:>5}  ({100 * cnt / n:>5.1f}%)")
    verdict("mode=none (%)", round(100 * p2c.get("none", 0) / n, 1), BASELINE["mode_none_pct"])
    print("  NOTE: commit 5 makes three dead filters live. A modest rise in")
    print("  'none' is expected. A large rise means the aspect band is too tight.")

    # ------------------------------------------------------------ smoothness
    hdr("STEERING SMOOTHNESS")
    sig = [r["n"] if r["n"] is not None else r["off"] for r in rows]
    d = [abs(sig[i + 1] - sig[i]) for i in range(n - 1)]
    scale = "normalised" if not legacy else "metres (legacy)"
    print(f"  signal: {scale}")
    print(f"  |d_offset|  mean {statistics.mean(d):.4f}   "
          f"p95 {sorted(d)[int(.95 * len(d))]:.4f}   max {max(d):.4f}")
    if legacy:
        verdict("  mean vs baseline", round(statistics.mean(d), 4),
                BASELINE["d_offset_mean"])
    flips = sum(1 for i in range(len(sig) - 1)
                if sig[i] * sig[i + 1] < 0 and abs(sig[i + 1] - sig[i]) > 0.05)
    print(f"  significant sign flips                 {flips} "
          f"({100 * flips / n:.1f} per 100 frames)")

    # ------------------------------------------------- loop closure polarity
    hdr("LOOP CLOSURE  (must stay NEGATIVE -- positive means inverted steering)")
    dd = [sig[i + 1] - sig[i] for i in range(n - 1)]
    a, b = sig[:-1], dd
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5
    db = sum((y - mb) ** 2 for y in b) ** .5
    c = num / (da * db) if da * db else 0.0
    print(f"  corr(offset, d_offset) = {c:+.3f}      (pre-fix -0.271)")
    if c > -0.05:
        print("  >>> STOP. The loop is not self-correcting. Check sign convention")
        print("  >>> before driving further -- see patch 12 on _drive().")
    else:
        print("  Loop is mean-reverting. Polarity is correct.")

    # ------------------------------------------------- worst recovery events
    hdr("WORST RECOVERY EVENTS (post-gap transients)")
    events = []
    i = 0
    while i < n:
        gap_ok = (rows[i]["p2"] == "none") if legacy else (not rows[i]["valid"])
        if gap_ok:
            j = i
            while j < n and ((rows[j]["p2"] == "none") if legacy else not rows[j]["valid"]):
                j += 1
            if j - i >= 3 and j < n:
                post = sig[j:j + 8]
                if len(post) > 1:
                    jump = max(abs(post[k + 1] - post[k]) for k in range(len(post) - 1))
                    events.append((rows[i]["f"], j - i, max(map(abs, post)), jump))
            i = j
        else:
            i += 1
    if events:
        print(f"  {'frame':>7} {'gap':>5} {'peak |off| after':>18} {'max step':>10}")
        for f, l, peak, jump in sorted(events, key=lambda e: -e[3])[:6]:
            print(f"  {f:>7} {l:>5} {peak:>18.4f} {jump:>10.4f}")
        print("\n  Pre-fix reference: 29-frame gap at f=2268 ramped to 0.2485 "
              "(71% of full scale).")
    else:
        print("  No gaps of 3+ frames. Nothing to recover from.")

    print()


if __name__ == "__main__":
    main()