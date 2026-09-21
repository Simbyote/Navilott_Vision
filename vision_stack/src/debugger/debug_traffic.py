"""
debug_traffic.py -- Traffic-light view for the debug tooling

Shows what the color branch saw and decided on the traffic ROI, so the HSV
ranges and blob filter can be calibrated on their own. The masks are the
calibration surface, so they sit beside the picture, not behind it.

Inputs (from a chain result, via TrafficView.extract):
    chain.traffic         the accepted TrafficLightCandidates
    chain.traffic_debug   enabled, calibrated, roi (BGR), the red / yellow /
                          green masks, mask_px, reject_counts, and the
                          per-blob "trace" when the chain ran with trace=True
    chain.fusion, chain.fusion_debug
                          the traffic light fusion kept, if any, and how many
                          it suppressed. Absent when the process callable has
                          no fusion

Picture:
    left panel    the traffic ROI with every traced blob
    right column  the red, yellow and green masks, tinted, with pixel counts
    header        frame id, pass / low / rejected counts, the best candidate
                  with its mean HSV, and whether fusion passed a light on
    footer        this frame's blobs seen / accepted per color and rejects

Blob colors (the light's own color is in the label):
    green   accepted, and at or above the confidence threshold if one is set
    amber   accepted by the blob filter but below the confidence threshold
    red     rejected; the label names the gate and the value that failed it

Labels on accepted blobs end in f<fill>: blob area over bounding-box area. A
round light fills about 0.78, a streak or slab fills more. There is no fill
gate today; the number is there to judge whether one would help.

The best candidate's mean HSV shows where the blob sits relative to the
calibrated bands: a blob whose hue is at the edge of a band is one lighting
change from disappearing.

If the color branch is off (no calibrated HSV ranges) the view says so instead
of drawing. Uncalibrated ranges are flagged in the header.

Fusion applies no confidence threshold and keeps the single highest-confidence
light across all three colors. Confidence is area only, so the largest blob
wins. The threshold in this view stands in for whatever Phase 3 applies.

Coordinates are traffic-ROI-relative. This module imports only debug_video.
"""
import cv2
import numpy as np

import src.debugger.debug_video as dv

# =============================================================================
# Configuration
# =============================================================================
COLORS = ("red", "yellow", "green")
LIGHT_COLORS = {"red": (0, 0, 255), "yellow": (0, 200, 255), "green": (0, 200, 0)}
GATE_SHORT = {"area": "area", "aspect": "asp"}
REPORT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
GAP_PX = 6                                # between the panels, before zoom

STATE_COLORS = {"pass": dv.C_USABLE, "low": dv.C_AMBER, "reject": dv.C_RED}

# =============================================================================
# View
# =============================================================================
class TrafficView:
    """
    Traffic-light view. Implements the view interface live_view runs:

        name, CSV_FIELDS
        extract(chain, frame=None) -> data
                                    pull this target's data out of a chain
                                    result; frame is unused, this view draws
                                    on its own ROI
        observe(data)               accumulate run statistics
        render(data, scale) -> img  the annotated picture
        row(data) -> list           one CSV row
        report() -> [str]           lines for summary.txt

    conf_threshold: the confidence a light has to reach downstream. None
                    draws no threshold and no amber state
    zoom: extra magnification on top of the run's scale
    """
    name = "traffic"
    CSV_FIELDS = ("frame_id", "timestamp_ms", "enabled", "seen", "traced",
                  "passed", "low", "best_label", "best_conf", "best_area",
                  "best_h", "best_s", "best_v", "mask_red_px",
                  "mask_yellow_px", "mask_green_px", "rej_area", "rej_aspect",
                  "fused", "fused_label", "fused_conf")

    def __init__(self, conf_threshold=None, zoom=2):
        self.conf_threshold = conf_threshold
        self.zoom = max(1, int(zoom))
        self._frames = 0
        self._enabled_frames = 0
        self._uncalibrated = False
        self._with_candidate = 0
        self._with_pass = 0
        self._fused_frames = 0
        self._fusion_seen = False
        self._fused_colors = {}
        self._best_conf = []
        self._rej = {}
        self._coverage = {c: [] for c in COLORS}

    # -- data -----------------------------------------------------------------
    def extract(self, chain, frame=None):
        dbg = getattr(chain, "traffic_debug", None) or {}
        fusion = getattr(chain, "fusion", None)
        fdbg = getattr(chain, "fusion_debug", None) or {}
        # None means the chain has no fusion; [] means fusion ran and kept none
        fused = (None if fusion is None
                 else [d for d in fusion.detections if d.type == "traffic_light"])
        return {
            "frame_id": chain.geometry.frame_id,
            "timestamp_ms": chain.geometry.timestamp_ms,
            "enabled": bool(dbg.get("enabled", False)),
            "calibrated": dbg.get("calibrated", True),
            "roi": dbg.get("roi"),
            "masks": {c: dbg.get(c) for c in COLORS},
            "mask_px": dict(dbg.get("mask_px", {})),
            "counts": dict(dbg.get("reject_counts", {})),
            "trace": dbg.get("trace"),
            "accepted": list(getattr(chain, "traffic", None) or []),
            "fused": fused,
            "suppressed": sum(1 for e in fdbg.get("log", ())
                              if e.startswith("[SUPPRESSED] traffic_light")),
        }

    def _entries(self, data):
        """The trace, or one entry per accepted candidate if there is none."""
        if data["trace"] is not None:
            return data["trace"]
        return [{"label": c.label, "bbox": c.bbox, "gate": None, "area": None,
                 "aspect": None, "fill": None, "confidence": c.confidence,
                 "hsv": None} for c in data["accepted"]]

    def _state(self, e):
        if e["gate"] is not None:
            return "reject"
        thr = self.conf_threshold
        if thr is not None and (e["confidence"] or 0.0) < thr:
            return "low"
        return "pass"

    def _summary(self, data):
        entries = self._entries(data)
        accepted = [e for e in entries if e["gate"] is None]
        states = [self._state(e) for e in entries]
        best = max(accepted, key=lambda e: e["confidence"] or 0.0, default=None)
        return {
            "entries": entries,
            "best": best,
            "passed": states.count("pass"),
            "low": states.count("low"),
            "rejected": states.count("reject"),
        }

    def _totals(self, data):
        """Blob counts summed over the three colors."""
        tot = {"seen": 0, "area": 0, "aspect": 0, "accepted": 0}
        for rc in data["counts"].values():
            for k in tot:
                tot[k] += rc.get(k, 0)
        return tot

    # -- statistics -----------------------------------------------------------
    def observe(self, data):
        self._frames += 1
        if not data["enabled"]:
            return
        self._enabled_frames += 1
        if not data["calibrated"]:
            self._uncalibrated = True
        sm = self._summary(data)
        if sm["best"] is not None:
            self._with_candidate += 1
            self._best_conf.append(sm["best"]["confidence"] or 0.0)
        if sm["passed"]:
            self._with_pass += 1
        if data["fused"] is not None:
            self._fusion_seen = True
            if data["fused"]:
                self._fused_frames += 1
                lab = data["fused"][0].label_detail
                self._fused_colors[lab] = self._fused_colors.get(lab, 0) + 1
        for k, v in self._totals(data).items():
            self._rej[k] = self._rej.get(k, 0) + v
        roi = data["roi"]
        if roi is not None:
            area = roi.shape[0] * roi.shape[1]
            for c in COLORS:
                self._coverage[c].append(100.0 * data["mask_px"].get(c, 0) / area)

    def row(self, data):
        sm = self._summary(data)
        b, tot = sm["best"], self._totals(data)
        opt = lambda v: "" if v is None else v
        hsv = (b or {}).get("hsv") or ("", "", "")
        px = data["mask_px"]
        fused = data["fused"]
        return [
            data["frame_id"], data["timestamp_ms"], int(data["enabled"]),
            tot["seen"], len(sm["entries"]), sm["passed"], sm["low"],
            opt(b and b["label"]), opt(b and b["confidence"]),
            opt(b and b["area"]), hsv[0], hsv[1], hsv[2],
            px.get("red", ""), px.get("yellow", ""), px.get("green", ""),
            tot["area"], tot["aspect"],
            "" if fused is None else len(fused),
            opt(fused[0].label_detail if fused else None),
            opt(fused[0].confidence if fused else None),
        ]

    def report(self):
        n = max(self._frames, 1)
        out = [f"[TRAFFIC LIGHT] {self._frames} frames"]
        if not self._enabled_frames:
            out.append(" color branch OFF: no calibrated HSV ranges were "
                       "supplied (--hsv PATH)")
            return out
        if self._uncalibrated:
            out.append(" WARNING: HSV ranges were not loaded with "
                       "load_hsv_ranges() (uncalibrated scaffold)")
        out.append(f" frames with a candidate through the blob filter  "
                   f"{self._with_candidate:6}  "
                   f"({100 * self._with_candidate / n:5.1f}%)")
        if self.conf_threshold is not None:
            out.append(f" ...and at or above threshold {self.conf_threshold:.2f}"
                       f"           {self._with_pass:6}  "
                       f"({100 * self._with_pass / n:5.1f}%)")
        if self._fusion_seen:
            out.append(f" frames where fusion passed a light on            "
                       f"{self._fused_frames:6}  "
                       f"({100 * self._fused_frames / n:5.1f}%)")
            if self._fused_colors:
                out.append("  fused light color: " + "  ".join(
                    f"{k} {v}" for k, v in sorted(self._fused_colors.items())))
        if self._best_conf:
            s = sorted(self._best_conf)
            out.append(f" best confidence per frame: min {s[0]:.3f}  "
                       f"med {s[len(s) // 2]:.3f}  max {s[-1]:.3f}")
            out.append(" frames that would pass at a threshold of:")
            out.append("  " + "   ".join(
                f"{t:.2f} -> {100 * sum(1 for c in s if c >= t) / n:.0f}%"
                for t in REPORT_THRESHOLDS))
        cov = {c: sorted(v) for c, v in self._coverage.items() if v}
        if cov:
            out.append(" median mask coverage of the ROI: " + "  ".join(
                f"{c} {v[len(v) // 2]:.1f}%" for c, v in cov.items()))
        if self._rej.get("seen"):
            out.append(f" blobs seen {self._rej['seen']}, accepted "
                       f"{self._rej.get('accepted', 0)}; rejected by gate: "
                       f"area {self._rej.get('area', 0)}, "
                       f"aspect {self._rej.get('aspect', 0)}")
        return out

    # -- picture --------------------------------------------------------------
    def render(self, data, scale=1):
        s = max(1, int(scale)) * self.zoom
        fs = max(0.34, 0.18 * s)
        th = 1 if s < 4 else 2
        lh = int(38 * fs)
        hh, fh = 2 * lh + 8, lh + 6

        roi = data["roi"]
        if not data["enabled"] or roi is None:
            img = np.zeros((hh + 2 * lh, 560, 3), np.uint8)
            dv.draw_text(img, f"#{data['frame_id']}  traffic_light", (6, lh),
                         dv.C_WHITE, fs, th)
            dv.draw_text(img, "color branch OFF: no calibrated HSV ranges "
                         "(pass --hsv PATH)", (6, 2 * lh + 2), dv.C_AMBER, fs, th)
            return img

        H, W = roi.shape[:2]
        pw, ph, gap = W * s, H * s, GAP_PX * s
        mw, mh = pw // 3, ph // 3
        canvas = np.zeros((hh + ph + fh, pw + gap + mw, 3), np.uint8)

        canvas[hh:hh + ph, :pw] = cv2.resize(roi, (pw, ph),
                                             interpolation=cv2.INTER_LINEAR)
        dv.draw_text(canvas, "traffic ROI", (4, hh + lh), dv.C_GRAY, fs, th)

        # masks, tinted in the light's own color
        for i, c in enumerate(COLORS):
            mask = data["masks"][c]
            y0 = hh + i * mh
            if mask is not None:
                tint = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
                tint[mask > 0] = LIGHT_COLORS[c]
                canvas[y0:y0 + mh, pw + gap:pw + gap + mw] = cv2.resize(
                    tint, (mw, mh), interpolation=cv2.INTER_NEAREST)
            px = data["mask_px"].get(c, 0)
            dv.draw_text(canvas, f"{c} {px}px {100.0 * px / (H * W):.1f}%",
                         (pw + gap + 4, y0 + lh), dv.C_WHITE, fs * 0.9, th)
            cv2.rectangle(canvas, (pw + gap, y0), (pw + gap + mw - 1, y0 + mh - 1),
                          dv.C_GRAY, 1)

        sm = self._summary(data)
        thr = self.conf_threshold
        for e in sm["entries"]:
            state = self._state(e)
            color = STATE_COLORS[state]
            x, y, w, h = e["bbox"]
            x0, y0 = x * s, hh + y * s
            cv2.rectangle(canvas, (x0, y0), ((x + w) * s - 1, hh + (y + h) * s - 1),
                          color, 2 if state == "pass" else 1)
            dv.draw_text(canvas, self._label(e, state),
                         (x0, max(y0 - 3, hh + lh)), color, fs, th)

        # header
        b = sm["best"]
        head_color = (dv.C_USABLE if sm["passed"]
                      else dv.C_AMBER if sm["low"] else dv.C_WHITE)
        title = f"#{data['frame_id']}  traffic_light"
        dv.draw_text(canvas, title, (6, lh), head_color, fs * 1.15, th)
        (tw, _), _ = cv2.getTextSize(title, dv.FONT, fs * 1.15, th)
        counts = f"pass {sm['passed']}"
        if thr is not None:
            counts += f"  low {sm['low']}"
        counts += f"  rejected {sm['rejected']}"
        dv.draw_text(canvas, counts, (6 + tw + 14, lh), dv.C_WHITE, fs, th)
        if not data["calibrated"]:
            dv.draw_text(canvas, "UNCALIBRATED HSV",
                         (canvas.shape[1] - int(170 * fs / 0.36), lh),
                         dv.C_RED, fs, th)

        if b is not None:
            best = f"best {b['label']} c{b['confidence']:.2f}"
            if b["area"] is not None:
                best += f"  area {b['area']:.0f}"
            if b["hsv"] is not None:
                best += "  hsv({:.0f},{:.0f},{:.0f})".format(*b["hsv"])
        else:
            best = "no blob cleared the filter"
        if thr is not None:
            best += f"   threshold {thr:.2f}"
        fused = data["fused"]
        if fused is not None:
            if fused:
                best += (f"   fusion: passed {fused[0].label_detail} "
                         f"c{fused[0].confidence:.2f}")
                if data["suppressed"]:
                    best += f" ({data['suppressed']} suppressed)"
            else:
                best += "   fusion: none"
        dv.draw_text(canvas, best, (6, 2 * lh + 2), dv.C_WHITE, fs, th)

        # footer
        rc, tot = data["counts"], self._totals(data)
        per = "  ".join(
            f"{c} {rc.get(c, {}).get('seen', 0)}/{rc.get(c, {}).get('accepted', 0)}"
            for c in COLORS)
        foot = (f"blobs seen/accepted  {per}   rejected: area {tot['area']} "
                f"aspect {tot['aspect']}")
        if data["trace"] is None:
            foot += "   (trace off: rejected blobs not shown)"
        dv.draw_text(canvas, foot, (6, hh + ph + lh), dv.C_GRAY, fs, th)
        return canvas

    def _label(self, e, state):
        if state == "reject":
            if e["gate"] == "aspect":
                return f"{e['label']} asp {e['aspect']:.2f}"
            return f"{e['label']} area {e['area']:.0f}"
        label = f"{e['label']} c{e['confidence']:.2f}"
        if e["fill"] is not None:
            label += f" f{e['fill']:.2f}"
        if state == "low":
            label += f"<{self.conf_threshold:.2f}"
        return label
    