"""Analyze a SYS-A log from sysa_power_chain.py: figures plus a results summary.

Usage (on the Mac):
    python3 analyze_sysa.py sysa_20261019_141500.csv [--outdir results]

Writes fig_timeline.png, fig_accuracy.png, fig_critical_zoom.png and
summary.json, and prints the numbers the report table needs.

Definitions
    ADC reading at a setpoint: mean of the raw samples in the ACC_WINDOW_S
        before the operator's multimeter marker (supply already settled).
    Detection latency: event time minus the first raw sample that crossed the
        threshold (the monitor cannot see a crossing sooner than that). The
        true crossing is up to one sample period (1 s) earlier.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WARN_V, CRIT_V, CLEAR_V = 10.5, 9.9, 10.8     # CLEAR_V = warning + hysteresis
ACC_WINDOW_S = 8.0
ACC_LIMIT_PCT = 2.0
LATENCY_LIMIT_S = 2.0                          # SYS-A step 6 requirement

STATE_COLOR = {"OK": "#d8f0d8", "WARNING": "#fff2c2", "CRITICAL": "#f6d0d0"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["t"] = float(r["t"])
    return rows


def f(x):
    return float(x) if x not in ("", None) else np.nan


def first_crossing(raws, t_event, thr, below):
    """Time of the first raw sample in the unbroken run (ending at the event)
    that is on the crossing side of thr."""
    prior = [(t, v) for t, v in raws if t <= t_event + 1e-6]
    t0 = None
    for t, v in reversed(prior):
        ok = v <= thr if below else v >= thr
        if ok:
            t0 = t
        else:
            break
    return t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)

    rows = load(args.log)
    raws = [(r["t"], float(r["value"])) for r in rows if r["kind"] == "raw"]
    trace = [r for r in rows if r["kind"] == "trace"]
    meters = [(r["t"], float(r["value"])) for r in rows if r["kind"] == "meter"]
    setpoints = [(r["t"], float(r["value"])) for r in rows if r["kind"] == "setpoint"]
    notes = [(r["t"], r["note"]) for r in rows if r["kind"] == "note"]
    events = [(r["t"], r["note"]) for r in rows if r["kind"] == "event"
              and r["note"] in ("warning", "critical", "recovered", "fault", "motors_stop")]
    # raw samples from the running monitor only (drop preflight/seed reads)
    t_start = next((r["t"] for r in rows if r["note"] == "monitoring_started"), 0.0)
    raws_mon = [(t, v) for t, v in raws if t >= t_start]

    summary = {"log": str(args.log)}

    # ---- accuracy ------------------------------------------------------
    acc = []
    for tm, vm in meters:
        w = [v for t, v in raws_mon if tm - ACC_WINDOW_S <= t <= tm]
        if len(w) < 4:
            continue
        adc = float(np.mean(w))
        acc.append({"t": tm, "meter_v": vm, "adc_v": round(adc, 4),
                    "adc_sd_mv": round(float(np.std(w)) * 1000, 1), "n": len(w),
                    "err_v": round(adc - vm, 4), "err_pct": round((adc - vm) / vm * 100, 3)})
    summary["accuracy"] = acc
    if acc:
        worst = max(acc, key=lambda a: abs(a["err_pct"]))
        summary["accuracy_worst_pct"] = worst["err_pct"]
        summary["accuracy_pass"] = abs(worst["err_pct"]) <= ACC_LIMIT_PCT

    # ---- events and latency -----------------------------------------------
    ev = {}
    for t, name in events:
        ev.setdefault(name, []).append(t)
    summary["events"] = ev
    lat = {}
    for name, thr, below in (("warning", WARN_V, True), ("critical", CRIT_V, True),
                             ("recovered", CLEAR_V, False)):
        for te in ev.get(name, []):
            t0 = first_crossing(raws_mon, te, thr, below)
            if t0 is not None:
                lat.setdefault(name, []).append(
                    {"t_event": te, "t_cross_first_seen": t0, "latency_s": round(te - t0, 2)})
    summary["latency"] = lat
    if ev.get("critical") and ev.get("motors_stop"):
        summary["critical_to_motors_stop_ms"] = round(
            (min(ev["motors_stop"], key=lambda t: abs(t - ev["critical"][0])) - ev["critical"][0]) * 1000, 1)
    if lat.get("critical"):
        summary["critical_latency_pass"] = lat["critical"][0]["latency_s"] <= LATENCY_LIMIT_S

    # ---- sag rejection (brief dip must raise no event) -----------------------
    sb = [t for t, n in notes if "sag-begin" in n]
    se = [t for t, n in notes if "sag-end" in n]
    if sb and se:
        a, b = sb[0], max(se[0], sb[0])
        w = [v for t, v in raws_mon if a <= t <= b + 3]
        in_win = [(t, n) for t, n in events if a <= t <= b + 6]
        summary["sag"] = {"min_raw_v": round(min(w), 3) if w else None,
                          "window_s": round(b - a, 1), "events_in_window": in_win,
                          "pass": len(in_win) == 0}

    # ---- hysteresis (volts stay above WARN_V but below CLEAR_V: no clear) -------
    # reported from events + the setpoint list; read the timeline figure to confirm.

    # ---- encoders around the critical stop ------------------------------------
    if ev.get("motors_stop"):
        ts = np.array([r["t"] for r in trace])
        L = np.array([f(r["left"]) for r in trace])
        R = np.array([f(r["right"]) for r in trace])
        tc = ev["motors_stop"][0]

        def counts_at(t):
            i = int(np.argmin(np.abs(ts - t)))
            return L[i], R[i]

        l0, r0 = counts_at(tc)
        lb, rb = counts_at(tc - 1.0)
        l1, r1 = counts_at(tc + 1.0)
        l3, r3 = counts_at(tc + 3.0)
        summary["encoders"] = {
            "rate_before_cps": [float(l0 - lb), float(r0 - rb)],
            "moved_1s_after": [float(l1 - l0), float(r1 - r0)],
            "moved_3s_after": [float(l3 - l0), float(r3 - r0)]}

    # ---- figures ---------------------------------------------------------------
    ts = np.array([r["t"] for r in trace])
    ema = np.array([f(r["value"]) for r in trace])
    st = [r["state"] for r in trace]
    L = np.array([f(r["left"]) for r in trace])
    R = np.array([f(r["right"]) for r in trace])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.1]})
    i = 0
    while i < len(st):          # shade by state
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        ax.axvspan(ts[i], ts[min(j + 1, len(ts) - 1)], color=STATE_COLOR.get(st[i], "w"), lw=0)
        i = j + 1
    ax.plot(ts, ema, color="#1b4f8a", lw=1.6, label="smoothed (EMA) voltage")
    ax.plot([t for t, _ in raws_mon], [v for _, v in raws_mon], "o", ms=2.5, color="#555",
            label="raw ADC sample (1 Hz)")
    ax.plot([t for t, _ in meters], [v for _, v in meters], "D", ms=5, mfc="none", mec="#c0392b",
            label="multimeter")
    for y, lbl in ((WARN_V, "warn 10.5"), (CLEAR_V, "clear 10.8"), (CRIT_V, "crit 9.9")):
        ax.axhline(y, ls="--", lw=0.8, color="#888")
        ax.text(ts[-1], y, f" {lbl}", va="center", fontsize=7, color="#555")
    mark = {"warning": "#d68910", "critical": "#c0392b", "recovered": "#1e8449"}
    side = {"warning": "right", "critical": "left", "recovered": "right"}
    for name, tl in ev.items():
        if name in mark:
            for t in tl:
                ax.axvline(t, color=mark[name], lw=1.2)
                ax.text(t, ax.get_ylim()[1], name, rotation=90, va="top", ha=side[name],
                        fontsize=7, color=mark[name])
    ax.set_ylabel("battery voltage (V)")
    ax.legend(loc="lower left", fontsize=7, ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    # wheel speed from encoder counts (counts/s over 0.5 s)
    k = 5
    if len(ts) > k:
        ax2.plot(ts[k:], (L[k:] - L[:-k]) / (ts[k:] - ts[:-k]), lw=1, label="left", color="#1b4f8a")
        ax2.plot(ts[k:], (R[k:] - R[:-k]) / (ts[k:] - ts[:-k]), lw=1, label="right", color="#d68910")
    ax2.set_ylabel("wheel (cnt/s)")
    ax2.set_xlabel("time since log start (s)")
    ax2.legend(fontsize=7, frameon=False, ncol=2)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_timeline.png", dpi=200)
    plt.close(fig)

    if acc:
        m = np.array([a["meter_v"] for a in acc])
        e = np.array([a["err_pct"] for a in acc])
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6))
        xs = np.linspace(m.min() - 0.3, m.max() + 0.3, 10)
        a1.fill_between(xs, xs * 0.98, xs * 1.02, color="#d8f0d8", label="±2 % band")
        a1.plot(xs, xs, color="#555", lw=0.8, label="ideal")
        a1.plot(m, [a["adc_v"] for a in acc], "o", color="#1b4f8a", label="ADS1115 reading")
        a1.set_xlabel("multimeter (V)")
        a1.set_ylabel("ADC-reported (V)")
        a1.legend(fontsize=7, frameon=False)
        a1.grid(alpha=0.25)
        a2.axhspan(-ACC_LIMIT_PCT, ACC_LIMIT_PCT, color="#d8f0d8")
        a2.axhline(0, color="#555", lw=0.8)
        a2.plot(m, e, "o-", color="#1b4f8a")
        a2.set_xlabel("multimeter (V)")
        a2.set_ylabel("error (%)")
        a2.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "fig_accuracy.png", dpi=200)
        plt.close(fig)

    if ev.get("critical"):
        tc = ev["critical"][0]
        t0 = lat["critical"][0]["t_cross_first_seen"] if lat.get("critical") else tc - 3
        lo, hi = t0 - 3, tc + 4
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        sel = (ts >= lo) & (ts <= hi)
        ax.plot(ts[sel], ema[sel], color="#1b4f8a", lw=1.6, label="EMA")
        w = [(t, v) for t, v in raws_mon if lo <= t <= hi]
        ax.plot([t for t, _ in w], [v for _, v in w], "o", ms=3, color="#555", label="raw")
        ax.axhline(CRIT_V, ls="--", lw=0.8, color="#888")
        ax.axvline(t0, color="#888", lw=0.8)
        ax.axvline(tc, color="#c0392b", lw=1.2)
        ax.annotate("", xy=(tc, CRIT_V - 0.15), xytext=(t0, CRIT_V - 0.15),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b"))
        ax.text((t0 + tc) / 2, CRIT_V - 0.35, f"{tc - t0:.1f} s", ha="center", fontsize=8, color="#c0392b")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("V")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "fig_critical_zoom.png", dpi=200)
        plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nAccuracy (ADS1115 vs multimeter)")
    print(f"{'meter V':>8} {'ADC V':>8} {'err V':>7} {'err %':>7} {'sd mV':>6}")
    for a in acc:
        print(f"{a['meter_v']:8.2f} {a['adc_v']:8.3f} {a['err_v']:7.3f} {a['err_pct']:7.2f} {a['adc_sd_mv']:6.1f}")
    print("\nEvents (s):", {k: [round(t, 1) for t in v] for k, v in ev.items()})
    print("Latency:", json.dumps(lat))
    for k in ("critical_to_motors_stop_ms", "sag", "encoders", "accuracy_pass", "critical_latency_pass"):
        if k in summary:
            print(f"{k}:", summary[k])


if __name__ == "__main__":
    main()
