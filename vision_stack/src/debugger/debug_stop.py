"""Stop-sign view: what the sign detector saw and decided on the sign ROI.

Purpose:
    Lets the sign filter be calibrated on its own, apart from the lane view.
    Fusion forwards the best valid stop sign at any confidence; the threshold
    here stands in for whatever Phase 3 applies, so a sign can be amber here
    and still be in Phase2Output.detections. Without a trace (the chain ran
    with trace=False) the view still draws accepted candidates and says so,
    but can't show rejected contours. Coordinates are sign-ROI-relative.

Main package:
    The rendered panel pair: the sign ROI with every traced contour beside
    the Canny edge map it came from, under a header with pass / low /
    rejected counts, the best candidate and whether fusion passed a stop
    sign on, over a footer of this frame's per-gate rejection counts.

Flow:
    1. Pull the sign debug, accepted candidates and fused stop sign from the chain.
    2. Classify each traced contour as pass, low or reject.
    3. Draw both panels, the header and the footer.
"""
import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.params import STOP_SIGN

# geometry sign gate names, shortened for labels
GATE_SHORT = {"area": "area", "vertices": "vert", "hull": "hull",
              "solidity": "sol"}

class StopView(dv.CandidateView):
    """
    Stop-sign view; implements the view interface in debug_video, drawing on its own ROI.

    conf_threshold: The confidence the sign has to reach downstream. None
        draws no threshold and no amber state.
    zoom: Extra magnification on top of the run's scale; the sign ROI is small.
    """
    name = "stop"
    CSV_FIELDS = ("frame_id", "timestamp_ms", "seen", "traced", "passed",
                  "low", "best_conf", "best_vertices", "best_area",
                  "best_solidity", "rej_area", "rej_vertices", "rej_hull",
                  "rej_solidity", "fused", "fused_conf")

    def __init__(self, conf_threshold=None, zoom=2):
        super().__init__(conf_threshold, zoom)
        self._frames = 0
        self._with_candidate = 0
        self._with_pass = 0
        self._fused_frames = 0
        self._fusion_seen = False
        self._best_conf = []
        self._counts = {}

    def extract(self, chain, frame=None) -> dict:
        """
        This frame's sign data from a chain result.

        Reads chain.geometry.sign_candidates, chain.sign_debug (sign_roi,
        edges, reject_counts, and "trace" when the chain ran with trace=True),
        and chain.fusion / fusion_debug when the process callable has fusion.
        """
        dbg = getattr(chain, "sign_debug", None) or {}
        geo = chain.geometry
        fusion = getattr(chain, "fusion", None)
        fdbg = getattr(chain, "fusion_debug", None) or {}
        # None means the chain has no fusion; [] means fusion ran and kept none
        fused = (None if fusion is None
                 else [d for d in fusion.detections if d.type == STOP_SIGN])
        return {
            "fused": fused,
            "suppressed": sum(1 for e in fdbg.get("log", ())
                              if e.startswith(f"[SUPPRESSED] {STOP_SIGN}")),
            "frame_id": geo.frame_id,
            "timestamp_ms": geo.timestamp_ms,
            "roi": dbg.get("sign_roi"),
            "edges": dbg.get("edges"),
            "trace": dbg.get("trace"),
            "accepted": list(geo.sign_candidates),
            "counts": dict(dbg.get("reject_counts", {})),
        }

    def _entries(self, data):
        """The trace, or one entry per accepted candidate if there is none."""
        if data["trace"] is not None:
            return data["trace"]
        return [{"bbox": c.bbox, "gate": None, "area": c.area,
                 "vertices": c.vertex_count, "solidity": c.solidity,
                 "confidence": c.confidence, "poly": c.contour}
                for c in data["accepted"]]

    def observe(self, data):
        sm = self._summary(data)
        self._frames += 1
        if sm["best"] is not None:
            self._with_candidate += 1
            self._best_conf.append(sm["best"]["confidence"] or 0.0)
        if sm["passed"]:
            self._with_pass += 1
        if data["fused"] is not None:
            self._fusion_seen = True
            if data["fused"]:
                self._fused_frames += 1
        for gate, n in data["counts"].items():
            self._counts[gate] = self._counts.get(gate, 0) + n

    def row(self, data):
        sm = self._summary(data)
        b, rc = sm["best"], data["counts"]
        opt = lambda v: "" if v is None else v
        fused = data["fused"]
        return [
            data["frame_id"], data["timestamp_ms"], rc.get("seen", ""),
            len(sm["entries"]), sm["passed"], sm["low"],
            opt(b and b["confidence"]), opt(b and b["vertices"]),
            opt(b and b["area"]), opt(b and b["solidity"]),
            rc.get("area", ""), rc.get("vertices", ""), rc.get("hull", ""),
            rc.get("solidity", ""),
            "" if fused is None else len(fused),
            opt(fused[0].confidence if fused else None),
        ]

    def report(self):
        n = max(self._frames, 1)
        out = [f"[STOP SIGN] {self._frames} frames"]
        out.append(f" frames with a candidate through the gates  "
                   f"{self._with_candidate:6}  "
                   f"({100 * self._with_candidate / n:5.1f}%)")
        if self.conf_threshold is not None:
            out.append(f" ...and at or above threshold {self.conf_threshold:.2f}"
                       f"      {self._with_pass:6}  "
                       f"({100 * self._with_pass / n:5.1f}%)")
        if self._fusion_seen:
            out.append(f" frames where fusion passed a stop sign on   "
                       f"{self._fused_frames:6}  "
                       f"({100 * self._fused_frames / n:5.1f}%)")
        out += self._threshold_report(self._best_conf, n)
        rej = {k: v for k, v in self._counts.items()
               if k in GATE_SHORT and v}
        if rej:
            out.append(" contours rejected, by gate:")
            for gate, v in sorted(rej.items(), key=lambda kv: -kv[1]):
                out.append(f"  {gate:<20}{v:6}")
        return out

    def render(self, data, scale=1):
        s, fs, th, lh, hh, fh = self._metrics(scale)

        roi = data["roi"]
        if roi is None:
            img = np.zeros((hh + 40, 320, 3), np.uint8)
            dv.draw_text(img, f"#{data['frame_id']}  no sign ROI", (6, lh),
                         dv.C_RED, fs, th)
            return img

        H, W = roi.shape[:2]
        pw, ph, gap = W * s, H * s, dv.GAP_PX * s
        canvas = np.zeros((hh + ph + fh, 2 * pw + gap, 3), np.uint8)

        left = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), (pw, ph),
                          interpolation=cv2.INTER_LINEAR)
        canvas[hh:hh + ph, :pw] = left
        edges = data["edges"]
        if edges is not None:
            right = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                               (pw, ph), interpolation=cv2.INTER_NEAREST)
            canvas[hh:hh + ph, pw + gap:] = right
        dv.draw_text(canvas, "sign ROI", (4, hh + lh), dv.C_GRAY, fs, th)
        dv.draw_text(canvas, "edges", (pw + gap + 4, hh + lh), dv.C_GRAY, fs, th)

        sm = self._summary(data)
        thr = self.conf_threshold
        for e in sm["entries"]:
            state = self._state(e)
            color = dv.STATE_COLORS[state]
            x, y, w, h = e["bbox"]
            x0, y0 = x * s, hh + y * s
            cv2.rectangle(canvas, (x0, y0), ((x + w) * s - 1, hh + (y + h) * s - 1),
                          color, 1)
            if e["poly"] is not None:
                pts = np.asarray(e["poly"]).reshape(-1, 2) * s
                pts = (pts + np.array([0, hh])).astype(np.int32)
                cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, color,
                              2 if state == "pass" else 1, cv2.LINE_AA)
            dv.draw_text(canvas, self._label(e, state), (x0, max(y0 - 3, hh + lh)),
                         color, fs, th)

        b = sm["best"]
        head_color, counts = self._header(sm)
        title = f"#{data['frame_id']}  stop_sign"
        dv.draw_text(canvas, title, (6, lh), head_color, fs * 1.15, th)
        (tw, _), _ = cv2.getTextSize(title, dv.FONT, fs * 1.15, th)
        dv.draw_text(canvas, counts, (6 + tw + 14, lh), dv.C_WHITE, fs, th)
        if b is not None:
            best = (f"best conf {b['confidence']:.2f}  v{b['vertices']}  "
                    f"sol {b['solidity']:.2f}  area {b['area']:.0f}")
        else:
            best = "no contour cleared the geometry gates"
        if thr is not None:
            best += f"   threshold {thr:.2f}"
        fused = data["fused"]
        if fused is not None:
            if fused:
                best += f"   fusion: passed c{fused[0].confidence:.2f}"
                if data["suppressed"]:
                    best += f" ({data['suppressed']} suppressed)"
            else:
                best += "   fusion: none"
        dv.draw_text(canvas, best, (6, 2 * lh + 2), dv.C_WHITE, fs, th)

        rc = data["counts"]
        foot = (f"seen {rc.get('seen', 0)}  area {rc.get('area', 0)}  "
                f"vert {rc.get('vertices', 0)}  hull {rc.get('hull', 0)}  "
                f"sol {rc.get('solidity', 0)}  accepted {rc.get('accepted', 0)}")
        if data["trace"] is None:
            foot += "   (trace off: rejected contours not shown)"
        dv.draw_text(canvas, foot, (6, hh + ph + lh), dv.C_GRAY, fs, th)
        return canvas

    def _label(self, e: dict, state: str) -> str:
        """Contour label: the failing gate and value when rejected, else vertices and confidence."""
        if state == "reject":
            g = e["gate"]
            if g == "vertices":
                return f"vert {e['vertices']}"
            if g == "solidity":
                return f"sol {e['solidity']:.2f}"
            if g == "area":
                # A sign-sized box with a tiny area is an open outline: a gap in the
                # Canny edge made the contour double back on itself, enclosing almost
                # nothing. The sign path doesn't close edges the way the lane path does.
                return f"area {e['area']:.0f}"
            return GATE_SHORT.get(g, g)
        label = f"v{e['vertices']} c{e['confidence']:.2f}"
        if state == "low":
            label += f"<{self.conf_threshold:.2f}"
        return label