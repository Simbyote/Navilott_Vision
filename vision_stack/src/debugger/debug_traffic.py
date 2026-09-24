"""Traffic-light view: what the color branch saw and decided on the traffic ROI.

Purpose:
    Lets the HSV ranges and blob filter be calibrated on their own. The masks
    are the calibration surface, so they sit beside the picture, not behind
    it, and the best candidate's mean HSV shows how close it sits to a band
    edge: a blob at the edge of a band is one lighting change from vanishing.
    Fusion keeps the single highest-confidence light across all three colors
    at any confidence, and confidence is area only, so the largest blob wins;
    the threshold here stands in for whatever Phase 3 applies. With the color
    branch off the view says so instead of drawing, and uncalibrated ranges
    are flagged. Coordinates are traffic-ROI-relative.

Main package:
    The rendered panel: the traffic ROI with every traced blob, beside the
    red, yellow and green masks, tinted, with pixel counts, under a header
    with pass / low / rejected counts, the best candidate and whether fusion
    passed a light on, over a footer of blobs seen / accepted per color.

Flow:
    1. Pull the color debug, accepted candidates and fused light from the chain.
    2. Classify each traced blob as pass, low or reject.
    3. Draw the ROI panel, the mask column, the header and the footer.
"""
import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.params import GREEN, RED, TRAFFIC_LIGHT, YELLOW

COLORS = (RED, YELLOW, GREEN)
LIGHT_COLORS = {RED: (0, 0, 255), YELLOW: (0, 200, 255), GREEN: (0, 200, 0)}    # BGR mask tints
# color branch gate names, shortened for labels
GATE_SHORT = {"area": "area", "aspect": "asp"}

class TrafficView(dv.CandidateView):
    """
    Traffic-light view; implements the view interface in debug_video, drawing on its own ROI.
    Blob outlines use dv.STATE_COLORS; the light's own color is in the label.

    conf_threshold: The confidence a light has to reach downstream. None
        draws no threshold and no amber state.
    zoom: Extra magnification on top of the run's scale.
    """
    name = "traffic"
    CSV_FIELDS = ("frame_id", "timestamp_ms", "enabled", "seen", "traced",
                  "passed", "low", "best_label", "best_conf", "best_area",
                  "best_h", "best_s", "best_v", "mask_red_px",
                  "mask_yellow_px", "mask_green_px", "rej_area", "rej_aspect",
                  "fused", "fused_label", "fused_conf")

    def __init__(self, conf_threshold=None, zoom=2):
        super().__init__(conf_threshold, zoom)
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

    def extract(self, chain, frame=None) -> dict:
        """
        This frame's color data from a chain result.

        Reads chain.traffic, chain.traffic_debug (enabled, calibrated, roi,
        the three masks, mask_px, reject_counts, and "trace" when the chain
        ran with trace=True), and chain.fusion / fusion_debug when the process
        callable has fusion.
        """
        dbg = getattr(chain, "traffic_debug", None) or {}
        fusion = getattr(chain, "fusion", None)
        fdbg = getattr(chain, "fusion_debug", None) or {}
        # None means the chain has no fusion; [] means fusion ran and kept none
        fused = (None if fusion is None
                 else [d for d in fusion.detections if d.type == TRAFFIC_LIGHT])
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
                              if e.startswith(f"[SUPPRESSED] {TRAFFIC_LIGHT}")),
        }

    def _entries(self, data):
        """The trace, or one entry per accepted candidate if there is none."""
        if data["trace"] is not None:
            return data["trace"]
        return [{"label": c.label, "bbox": c.bbox, "gate": None, "area": None,
                 "aspect": None, "fill": None, "confidence": c.confidence,
                 "hsv": None} for c in data["accepted"]]

    def _totals(self, data):
        """Blob counts summed over the three colors."""
        tot = {"seen": 0, "area": 0, "aspect": 0, "accepted": 0}
        for rc in data["counts"].values():
            for k in tot:
                tot[k] += rc.get(k, 0)
        return tot

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
            px.get(RED, ""), px.get(YELLOW, ""), px.get(GREEN, ""),
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
        out += self._threshold_report(self._best_conf, n)
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

    def render(self, data, scale=1):
        s, fs, th, lh, hh, fh = self._metrics(scale)

        roi = data["roi"]
        if not data["enabled"] or roi is None:
            img = np.zeros((hh + 2 * lh, 560, 3), np.uint8)
            dv.draw_text(img, f"#{data['frame_id']}  traffic_light", (6, lh),
                         dv.C_WHITE, fs, th)
            dv.draw_text(img, "color branch OFF: no calibrated HSV ranges "
                         "(pass --hsv PATH)", (6, 2 * lh + 2), dv.C_AMBER, fs, th)
            return img

        H, W = roi.shape[:2]
        pw, ph, gap = W * s, H * s, dv.GAP_PX * s
        mw, mh = pw // 3, ph // 3
        canvas = np.zeros((hh + ph + fh, pw + gap + mw, 3), np.uint8)

        canvas[hh:hh + ph, :pw] = cv2.resize(roi, (pw, ph),
                                             interpolation=cv2.INTER_LINEAR)
        dv.draw_text(canvas, "traffic ROI", (4, hh + lh), dv.C_GRAY, fs, th)

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
            color = dv.STATE_COLORS[state]
            x, y, w, h = e["bbox"]
            x0, y0 = x * s, hh + y * s
            cv2.rectangle(canvas, (x0, y0), ((x + w) * s - 1, hh + (y + h) * s - 1),
                          color, 2 if state == "pass" else 1)
            dv.draw_text(canvas, self._label(e, state),
                         (x0, max(y0 - 3, hh + lh)), color, fs, th)

        b = sm["best"]
        head_color, counts = self._header(sm)
        title = f"#{data['frame_id']}  traffic_light"
        dv.draw_text(canvas, title, (6, lh), head_color, fs * 1.15, th)
        (tw, _), _ = cv2.getTextSize(title, dv.FONT, fs * 1.15, th)
        dv.draw_text(canvas, counts, (6 + tw + 14, lh), dv.C_WHITE, fs, th)
        if not data["calibrated"]:
            dv.draw_text(canvas, "UNCALIBRATED HSV",
                         (canvas.shape[1] - int(170 * fs / 0.36), lh),   # ~170 px wide at fs 0.36
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

    def _label(self, e: dict, state: str) -> str:
        """Blob label: the failing gate and value when rejected, else confidence and fill."""
        if state == "reject":
            if e["gate"] == "aspect":
                return f"{e['label']} asp {e['aspect']:.2f}"
            return f"{e['label']} area {e['area']:.0f}"
        label = f"{e['label']} c{e['confidence']:.2f}"
        # fill = blob area / bbox area. A round light fills about 0.78 (pi/4); a streak
        # or slab fills more. There's no fill gate yet; this shows whether one would help.
        if e["fill"] is not None:
            label += f" f{e['fill']:.2f}"
        if state == "low":
            label += f"<{self.conf_threshold:.2f}"
        return label