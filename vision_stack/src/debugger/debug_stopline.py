"""Stop-line view: what the stop-line detector saw, paired and measured on the lane ROI.

Purpose:
    Lets the stop-line gates be checked on real footage, apart from the lane
    view. The top panel is the lane ROI with every top edge the detector
    fitted: accepted lines as the band between their paired edges, rejected
    edges in red with the gate that stopped them. The line stop_line_distance
    measured is outlined in white with its distance to the ROI bottom, and
    lane candidates lane_offset skipped because they lie on a stop line are
    boxed in amber. The bottom panel is the gradient split itself: every
    Canny edge in gray, the kept top edges (brightness rising going down) in
    cyan and bottom edges in magenta, with the fitted lines over them. If a
    line is missed, the bottom panel says whether its edges survived the
    split; the top panel says which gate refused it.

    Coordinates are lane-ROI-relative. The threshold, when given, stands in
    for stop_line_distance's min_confidence, so a line can be amber here and
    still be a geometry candidate.

Main package:
    StopLineView: implements the view interface in debug_video.

Flow:
    1. Pull the stop-line debug, candidates, measurement and lane-offset skips from the chain.
    2. Grade each traced top edge as pass, low or reject.
    3. Draw both panels, the header and the footer.
"""
import re

import cv2
import numpy as np

import src.debugger.debug_video as dv

GATE_SHORT = {"short": "short", "tilt": "tilt", "unpaired": "unpaired", "intensity": "dim"}
GATES = tuple(GATE_SHORT)
C_TOP = (255, 255, 0)           # cyan
C_BOTTOM = (255, 0, 255)        # magenta
C_EDGE = (90, 90, 90)
_SKIP = re.compile(r"candidate at x=(-?\d+) y=(-?\d+) lies on the stop line")


class StopLineView(dv.CandidateView):
    """
    Stop-line view; implements the view interface in debug_video, drawing on the lane ROI.

    conf_threshold: The confidence a line needs downstream (the chain's
        stop_line.min_confidence). None draws no threshold and no amber state.
    zoom: Extra magnification on top of the run's scale.
    """
    name = "stopline"
    CSV_FIELDS = ("frame_id", "timestamp_ms", "top_edges", "bottom_edges", "seen",
                  "passed", "low", "rej_short", "rej_tilt", "rej_unpaired", "rej_intensity",
                  "best_conf", "measured", "distance_px", "y_near_px", "tilt_deg",
                  "thickness_px", "clipped", "lane_skipped", "distance_cm", "proximity")

    def __init__(self, conf_threshold=None, zoom=2):
        super().__init__(conf_threshold, zoom)
        self._frames = 0
        self._with_candidate = 0
        self._measured = 0
        self._distances = []
        self._distances_cm = []
        self._skipped = 0
        self._best_conf = []
        self._counts = {}

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def extract(self, chain, frame=None) -> dict:
        """
        This frame's stop-line data from a chain result.

        Reads chain.lane_debug (lane_roi, edges_raw, and "stop_line": the edge
        maps, trace, bottoms and reject_counts), chain.geometry's stop-line
        candidates, chain.stop_line (the measurement) and chain.offset_debug's
        log for lane candidates skipped as part of a stop line.
        """
        dbg = getattr(chain, "lane_debug", None) or {}
        sdbg = dbg.get("stop_line") or {}
        geo = chain.geometry
        skipped = []
        by_corner = {(c.bbox[0], c.bbox[1]): c.bbox for c in geo.lane_candidates}
        for entry in (getattr(chain, "offset_debug", None) or {}).get("log", ()):
            m = _SKIP.search(entry)
            if m and (int(m[1]), int(m[2])) in by_corner:
                skipped.append(by_corner[(int(m[1]), int(m[2]))])
        return {
            "frame_id": geo.frame_id,
            "timestamp_ms": geo.timestamp_ms,
            "roi": dbg.get("lane_roi"),
            "edges": dbg.get("edges_raw"),
            "top": sdbg.get("edges_top"),
            "bottom": sdbg.get("edges_bottom"),
            "trace": sdbg.get("trace"),
            "bottoms": list(sdbg.get("bottoms", ())),
            "accepted": list(getattr(geo, "stop_line_candidates", [])),
            "counts": dict(sdbg.get("reject_counts", {})),
            "edge_counts": (sdbg.get("top_count", 0), sdbg.get("bottom_count", 0)),
            "measured": getattr(chain, "stop_line", None),
            "lane_skipped": skipped,
        }

    def _entries(self, data):
        """One entry per traced top edge, or per accepted candidate if there's no trace."""
        if data["trace"] is not None:
            return [{**e, "confidence": e["candidate"].confidence if e["candidate"] else None}
                    for e in data["trace"]]
        return [{"ends": ((c.x_left, c.y_top_px), (c.x_right, c.y_top_px)), "gate": None,
                 "length": c.length_px, "tilt": c.tilt_deg, "candidate": c,
                 "confidence": c.confidence} for c in data["accepted"]]

    @staticmethod
    def _measured_line(data):
        m = data["measured"]
        return m if m is not None and m.detected else None

    def observe(self, data):
        sm = self._summary(data)
        self._frames += 1
        if sm["best"] is not None:
            self._with_candidate += 1
            self._best_conf.append(sm["best"]["confidence"] or 0.0)
        m = self._measured_line(data)
        if m is not None:
            self._measured += 1
            self._distances.append(m.distance_px)
            if m.distance_cm is not None:
                self._distances_cm.append(m.distance_cm)
        self._skipped += len(data["lane_skipped"])
        for gate, n in data["counts"].items():
            self._counts[gate] = self._counts.get(gate, 0) + n

    def row(self, data):
        sm, rc, m = self._summary(data), data["counts"], self._measured_line(data)
        opt = lambda v: "" if v is None else v
        cand = next((c for c in data["accepted"]
                     if m is not None and c.y_near_px == m.y_near_px), None)
        return [
            data["frame_id"], data["timestamp_ms"], *data["edge_counts"],
            rc.get("seen", ""), sm["passed"], sm["low"],
            *(rc.get(g, "") for g in GATES),
            opt(sm["best"] and sm["best"]["confidence"]),
            int(m is not None), opt(m and m.distance_px), opt(m and m.y_near_px),
            opt(m and m.tilt_deg), opt(cand and cand.thickness_px),
            "" if m is None else int(m.clipped), len(data["lane_skipped"]),
            opt(m and m.distance_cm), opt(m and m.proximity),
        ]

    def report(self):
        n = max(self._frames, 1)
        out = [f"[STOP LINE] {self._frames} frames",
               f" frames with a line through the gates        {self._with_candidate:6}  "
               f"({100 * self._with_candidate / n:5.1f}%)",
               f" frames with a measured line                 {self._measured:6}  "
               f"({100 * self._measured / n:5.1f}%)"]
        if self._distances:
            d = sorted(self._distances)
            out.append(f" distance to the ROI bottom, px: min {d[0]:.1f}  "
                       f"med {d[len(d) // 2]:.1f}  max {d[-1]:.1f}")
        if self._distances_cm:
            d = sorted(self._distances_cm)
            out.append(f" distance ahead on the floor, cm: min {d[0]:.1f}  "
                       f"med {d[len(d) // 2]:.1f}  max {d[-1]:.1f}")
        out.append(f" lane candidates skipped as part of a stop line  {self._skipped}")
        out += self._threshold_report(self._best_conf, n)
        rej = {k: v for k, v in self._counts.items() if k in GATE_SHORT and v}
        if rej:
            out.append(" top edges rejected, by gate:")
            for gate, v in sorted(rej.items(), key=lambda kv: -kv[1]):
                out.append(f"  {gate:<20}{v:6}")
        return out

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def render(self, data, scale=1):
        s, fs, th, lh, hh, fh = self._metrics(scale)
        roi = data["roi"]
        if roi is None:
            img = np.zeros((hh + 40, 320, 3), np.uint8)
            dv.draw_text(img, f"#{data['frame_id']}  no lane ROI", (6, lh), dv.C_RED, fs, th)
            return img

        H, W = roi.shape[:2]
        pw, ph, gap = W * s, H * s, dv.GAP_PX * s
        top_y, bot_y = hh, hh + ph + gap
        canvas = np.zeros((hh + 2 * ph + gap + fh, pw, 3), np.uint8)
        canvas[top_y:top_y + ph] = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), (pw, ph),
                                              interpolation=cv2.INTER_LINEAR)
        canvas[bot_y:bot_y + ph] = self._split_panel(data, (W, H), (pw, ph))
        dv.draw_text(canvas, "lane ROI", (4, top_y + lh), dv.C_GRAY, fs, th)
        dv.draw_text(canvas, "edges: all / top / bottom", (4, bot_y + lh), dv.C_GRAY, fs, th)

        pt = lambda x, y, y0: (int(round(x * s)), int(round(y0 + y * s)))

        # Fitted lines on the split panel
        for (a, b) in data["bottoms"]:
            cv2.line(canvas, pt(*a, bot_y), pt(*b, bot_y), C_BOTTOM, 1, cv2.LINE_AA)
        sm = self._summary(data)
        for e in sm["entries"]:
            a, b = e["ends"]
            cv2.line(canvas, pt(*a, bot_y), pt(*b, bot_y), C_TOP, 1, cv2.LINE_AA)

        # Lane candidates lane_offset skipped
        for x, y, w, h in data["lane_skipped"]:
            cv2.rectangle(canvas, pt(x, y, top_y), pt(x + w, y + h, top_y), dv.C_AMBER, 1)
            right, top = pt(x + w, y, top_y)
            dv.draw_text(canvas, "lane skip", (min(right + 3, pw - int(60 * fs)), top + lh),
                         dv.C_AMBER, fs * 0.8, th)

        # Top edges: accepted as bands, rejected as red lines with the gate
        for e in sm["entries"]:
            state = self._state(e)
            color = dv.STATE_COLORS[state]
            c = e["candidate"]
            if c is not None:
                cv2.polylines(canvas, [self._band(c, s, top_y)], True, color, 2, cv2.LINE_AA)
                label = f"c{c.confidence:.2f} t{c.thickness_px:.0f}"
                if state == "low":      # three places, so 0.617 doesn't read as 0.62 < 0.62
                    label = f"c{c.confidence:.3f}<{self.conf_threshold:.3f} t{c.thickness_px:.0f}"
                anchor = pt(c.x_left, c.y_top_px, top_y)
            else:
                a, b = e["ends"]
                cv2.line(canvas, pt(*a, top_y), pt(*b, top_y), color, 2, cv2.LINE_AA)
                label = GATE_SHORT.get(e["gate"], e["gate"])
                if e["gate"] == "short":
                    label += f" {e['length']:.0f}"
                elif e["gate"] == "tilt":
                    label += f" {e['tilt']:+.0f}"
                anchor = pt(*a, top_y)
            dv.draw_text(canvas, label, (anchor[0], max(anchor[1] - 3, top_y + lh)), color, fs, th)

        # The measured line and its distance to the ROI bottom
        m = self._measured_line(data)
        measured = next((c for c in data["accepted"]
                         if m is not None and c.y_near_px == m.y_near_px), None)
        if measured is not None:
            cv2.polylines(canvas, [self._band(measured, s, top_y)], True, dv.C_WHITE, 1, cv2.LINE_AA)
            x = (measured.x_left + measured.x_right) / 2.0
            p0, p1 = pt(x, m.y_near_px, top_y), pt(x, H, top_y)
            if p1[1] - p0[1] > 2:
                cv2.arrowedLine(canvas, p0, (p1[0], p1[1] - 1), dv.C_WHITE, 1, cv2.LINE_AA, tipLength=0.2)
            dist = "ON LINE" if m.clipped else f"{m.distance_px:.0f}px"
            if m.distance_cm is not None and not m.clipped:
                dist += f" {m.distance_cm:.1f}cm"
            dv.draw_text(canvas, dist,
                         (p0[0] + 4, min(p0[1] + lh, top_y + ph - 3)), dv.C_WHITE, fs, th)

        # Header
        head_color, counts = self._header(sm)
        title = f"#{data['frame_id']}  stop_line"
        dv.draw_text(canvas, title, (6, lh), head_color, fs * 1.15, th)
        (tw, _), _ = cv2.getTextSize(title, dv.FONT, fs * 1.15, th)
        dv.draw_text(canvas, counts, (6 + tw + 14, lh), dv.C_WHITE, fs, th)
        if m is not None:
            cm = "" if m.distance_cm is None else f" = {m.distance_cm:.1f}cm"
            info = (f"measured {'on the line' if m.clipped else f'{m.distance_px:.1f}px{cm} ahead'}"
                    f"  tilt {m.tilt_deg:+.1f}  prox {m.proximity:.2f}  conf {m.confidence:.2f}")
        elif data["measured"] is not None and data["measured"].candidate_count:
            info = f"not measured: {data['measured'].candidate_count} line(s) below the confidence gate"
        else:
            info = "no stop line measured"
        if data["lane_skipped"]:
            info += f"   lane candidates skipped {len(data['lane_skipped'])}"
        if self.conf_threshold is not None:
            info += f"   threshold {self.conf_threshold:.2f}"
        dv.draw_text(canvas, info, (6, 2 * lh + 2), dv.C_WHITE, fs, th)

        # Footer
        rc, (n_top, n_bottom) = data["counts"], data["edge_counts"]
        foot = (f"edges top {n_top} bottom {n_bottom}   seen {rc.get('seen', 0)}  "
                + "  ".join(f"{GATE_SHORT[g]} {rc.get(g, 0)}" for g in GATES)
                + f"  accepted {rc.get('accepted', 0)}")
        if data["trace"] is None:
            foot += "   (no trace: rejected edges not shown)"
        dv.draw_text(canvas, foot, (6, bot_y + ph + lh), dv.C_GRAY, fs, th)
        return canvas

    @staticmethod
    def _band(c, s: int, y0: int) -> np.ndarray:
        """The candidate's band, top and bottom edges across its span, as a canvas polygon."""
        slope = np.tan(np.radians(c.tilt_deg))
        mid = (c.x_left + c.x_right) / 2.0
        top = lambda x: c.y_top_px + (x - mid) * slope
        bottom = (lambda x: c.y_bottom_px) if c.clipped else (lambda x: c.y_bottom_px + (x - mid) * slope)
        pts = [(c.x_left, top(c.x_left)), (c.x_right, top(c.x_right)),
               (c.x_right, bottom(c.x_right)), (c.x_left, bottom(c.x_left))]
        return np.array([(round(x * s), round(y0 + y * s)) for x, y in pts], np.int32).reshape(-1, 1, 2)

    def _split_panel(self, data, size, out_size) -> np.ndarray:
        """All Canny edges in gray, the kept top edges in cyan and bottom edges in magenta."""
        W, H = size
        panel = np.zeros((H, W, 3), np.uint8)
        if data["edges"] is not None:
            panel[data["edges"] > 0] = C_EDGE
        if data["top"] is not None:
            panel[data["top"] > 0] = C_TOP
        if data["bottom"] is not None:
            panel[data["bottom"] > 0] = C_BOTTOM
        return cv2.resize(panel, out_size, interpolation=cv2.INTER_NEAREST)
