"""Lane-geometry view: every contour the lane detector traced, and who refused it.

Purpose:
    The lane view (debug_lane) shows what lane_offset decided on the frame.
    This one shows the step before, on the lane ROI: every contour the lane
    detector traced, colored by the stage that decided it. Red contours were
    refused by geometry's own gates (area, aspect, span, intensity), which
    the lane view never draws; amber ones were accepted by geometry and
    refused by lane_offset (confidence, proximity, length, width, intensity,
    or lying on a stop line); green ones are usable boundaries, and the two
    chosen as left and right are marked at their foot. If a lane line is
    missing, the color says which stage lost it and the label says why.

    The bottom panel is the edge map the contours came from: every Canny
    edge in gray, the edges the horizontal-line filter took out (a stop line
    across the lane) in red, the edges the lane detector kept in white, and
    what the closing step added between them in blue. Contours that join
    things they shouldn't, like a lane line and a stop line, show here.

    Coordinates are lane-ROI-relative. Rejected contours need the chain's
    trace (trace=True, which live_view uses); without it only the candidates
    geometry accepted are drawn, and the footer says so.

Main package:
    LaneGeometryView: implements the view interface in debug_video.

Flow:
    1. Pull the lane trace, edge maps, candidates and the lane result from
       the chain; classify each candidate with lane_offset's gates.
    2. Draw the ROI panel (red / amber / green contours, anchors) and the edge panel.
    3. Header with the lane result, footer with this frame's gate counts.
"""
import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.debugger.debug_lane import candidate_gates

# geometry's lane gate names and lane_offset's, shortened for labels
GEO_SHORT = {"area": "area", "degenerate": "degen", "too_few_pts": "pts", "aspect": "aspect",
             "w_span": "span", "h_span": "span", "intensity": "int"}
GEO_GATES = ("area", "degenerate", "too_few_pts", "aspect", "w_span", "h_span", "intensity")
OFFSET_SHORT = {"confidence": "conf", "proximity": "prox", "length_px": "len", "width_px": "wid",
                "mean_intensity": "int", "stop_line": "on stop line"}

C_GEO_REJECT = dv.C_RED
C_OFFSET_REJECT = dv.C_AMBER
C_USABLE = dv.C_USABLE
C_LEFT, C_RIGHT, C_CENTER = (255, 255, 0), (255, 0, 255), (0, 255, 255)   # cyan, magenta, yellow
C_EDGE, C_REMOVED, C_KEPT, C_CLOSED = (90, 90, 90), (60, 60, 255), (235, 235, 235), (255, 150, 60)


class LaneGeometryView(dv.CandidateView):
    """
    Lane-geometry view; implements the view interface in debug_video, drawing on the lane ROI.

    conf_threshold: Unused (lane_offset's own gates decide); accepted so
        live_view can build every view the same way.
    zoom: Extra magnification on top of the run's scale.
    lane_config: The LaneOffsetConfig the chain ran with, used to say which
        gate refused each candidate (debug_lane.candidate_gates). live_view.run
        sets it from its own lane_config; it must be set before extract().
    """
    name = "lanegeo"
    CSV_FIELDS = ("frame_id", "timestamp_ms", "contours", "geo_rejected", "candidates", "usable",
                  "offset_rejected", "mode", "left_x", "right_x", "offset", "confidence",
                  *(f"rej_{g}" for g in GEO_GATES), "offset_rejects", "removed_edge_px")

    def __init__(self, conf_threshold=None, zoom=2, lane_config=None):
        super().__init__(None, zoom)
        self.lane_config = lane_config
        self._frames = 0
        self._modes = {}
        self._geo = {}
        self._offset = {}
        self._filtered_frames = 0

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    def extract(self, chain, frame=None) -> dict:
        """
        This frame's lane-geometry data from a chain result.

        Reads chain.lane_debug (lane_roi, edges_raw, edges_lane, edges,
        reject_counts, and "trace" when the chain ran with trace=True),
        chain.geometry's lane and stop-line candidates and chain.offset.
        Each candidate's lane_offset gate is worked out again with
        self.lane_config.

        Raises:
            ValueError: If lane_config was never set.
        """
        if self.lane_config is None:
            raise ValueError("LaneGeometryView needs lane_config (the chain's LaneOffsetConfig)")
        dbg = getattr(chain, "lane_debug", None) or {}
        geo = chain.geometry
        candidates = list(geo.lane_candidates)
        gates = [g for _, g in candidate_gates(geo, self.lane_config)]
        raw, lane = dbg.get("edges_raw"), dbg.get("edges_lane")
        removed = 0 if raw is None or lane is None else int(((raw > 0) & (lane == 0)).sum())
        return {
            "frame_id": geo.frame_id,
            "timestamp_ms": geo.timestamp_ms,
            "roi": dbg.get("lane_roi"),
            "edges_raw": raw,
            "edges_lane": lane,
            "edges": dbg.get("edges"),
            "trace": dbg.get("trace"),
            "counts": dict(dbg.get("reject_counts", {})),
            "candidates": list(zip(candidates, gates)),
            "stop_lines": list(getattr(geo, "stop_line_candidates", [])),
            "offset": getattr(chain, "offset", None),
            "removed_edge_px": removed,
        }

    @staticmethod
    def _geo_rejects(data):
        return [e for e in (data["trace"] or []) if e["gate"] is not None]

    def _tally(self, data):
        usable = sum(g is None for _, g in data["candidates"])
        offset_rej = {}
        for _, g in data["candidates"]:
            if g is not None:
                offset_rej[g] = offset_rej.get(g, 0) + 1
        return usable, offset_rej

    def observe(self, data):
        self._frames += 1
        mode = data["offset"].mode if data["offset"] is not None else "?"
        self._modes[mode] = self._modes.get(mode, 0) + 1
        for g in GEO_GATES:
            self._geo[g] = self._geo.get(g, 0) + data["counts"].get(g, 0)
        for g, n in self._tally(data)[1].items():
            self._offset[g] = self._offset.get(g, 0) + n
        if data["removed_edge_px"]:
            self._filtered_frames += 1

    def row(self, data):
        rc, off = data["counts"], data["offset"]
        usable, offset_rej = self._tally(data)
        opt = lambda v: "" if v is None else v
        return [
            data["frame_id"], data["timestamp_ms"], rc.get("seen", ""),
            sum(rc.get(g, 0) for g in GEO_GATES), len(data["candidates"]), usable,
            sum(offset_rej.values()),
            opt(off and off.mode), opt(off and off.left_x), opt(off and off.right_x),
            opt(off and off.offset), opt(off and off.confidence),
            *(rc.get(g, "") for g in GEO_GATES),
            ";".join(f"{g}:{n}" for g, n in sorted(offset_rej.items())),
            data["removed_edge_px"],
        ]

    def report(self):
        n = max(self._frames, 1)
        out = [f"[LANE GEOMETRY] {self._frames} frames"]
        for mode, c in sorted(self._modes.items(), key=lambda kv: -kv[1]):
            out.append(f"  {mode:<20}{c:6}  ({100 * c / n:5.1f}%)")
        out.append(f" frames where the horizontal-line filter removed edges  {self._filtered_frames}")
        geo = {g: c for g, c in self._geo.items() if c}
        if geo:
            out.append(" contours refused by geometry, by gate:")
            for g, c in sorted(geo.items(), key=lambda kv: -kv[1]):
                out.append(f"  {g:<20}{c:6}")
        if self._offset:
            out.append(" candidates refused by lane_offset, by gate:")
            for g, c in sorted(self._offset.items(), key=lambda kv: -kv[1]):
                out.append(f"  {g:<20}{c:6}")
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
        canvas[bot_y:bot_y + ph] = self._edge_panel(data, (W, H), (pw, ph))
        dv.draw_text(canvas, "lane ROI", (4, top_y + lh), dv.C_GRAY, fs, th)
        dv.draw_text(canvas, "edges: all / removed / kept / closed", (4, bot_y + lh), dv.C_GRAY, fs, th)

        def poly(contour, color, width):
            pts = (np.asarray(contour).reshape(-1, 2) * s + np.array([0, top_y])).astype(np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, color, width, cv2.LINE_AA)

        placed = []                             # label boxes already drawn, so labels don't overprint

        def label(text, bbox, color):
            x, y, w, h = bbox
            (tw, _), _ = cv2.getTextSize(text, dv.FONT, fs, th)
            lx, ly = x * s + 2, max(top_y + y * s - 3, top_y + lh)
            while ly < top_y + ph and any(lx < px + pw_ and px < lx + tw and ly - lh < py and py - lh < ly
                                          for px, py, pw_ in placed):
                ly += lh + 1
            placed.append((lx, ly, tw))
            dv.draw_text(canvas, text, (lx, ly), color, fs, th)

        # Detected stop lines, faint, for context
        for sl in data["stop_lines"]:
            x, y, w, h = sl.bbox
            cv2.rectangle(canvas, (x * s, top_y + y * s), ((x + w) * s, top_y + (y + h) * s), dv.C_GRAY, 1)

        # Refused by geometry
        for e in self._geo_rejects(data):
            poly(e["contour"], C_GEO_REJECT, 1)
            value = "" if e["value"] is None else f" {e['value']:g}"
            label(GEO_SHORT.get(e["gate"], e["gate"]) + value, e["bbox"], C_GEO_REJECT)

        # Accepted by geometry: refused by lane_offset, or usable
        for cand, gate in data["candidates"]:
            if gate is None:
                poly(cand.contour, C_USABLE, 2)
                label(f"c{cand.confidence:.2f} w{cand.width_px:.0f}", cand.bbox, C_USABLE)
            else:
                poly(cand.contour, C_OFFSET_REJECT, 1)
                label(OFFSET_SHORT.get(gate, gate), cand.bbox, C_OFFSET_REJECT)

        # The result: anchors at their foot, lane center, robot
        off = data["offset"]
        bottom = top_y + ph - 1
        cx = W / 2.0
        cv2.line(canvas, (int(cx * s), top_y), (int(cx * s), bottom), dv.C_GRAY, 1)
        if off is not None:
            for x, color in ((off.left_x, C_LEFT), (off.right_x, C_RIGHT)):
                if x is not None:
                    cv2.line(canvas, (int(x * s), bottom - 10 * s), (int(x * s), bottom), color, 3)
            if off.left_x is not None and off.right_x is not None:
                mid = (off.left_x + off.right_x) / 2.0
                cv2.drawMarker(canvas, (int(mid * s), bottom - 4 * s), C_CENTER, cv2.MARKER_TRIANGLE_UP,
                               8 * s, 2)

        # Header
        usable, offset_rej = self._tally(data)
        geo_rej = len(self._geo_rejects(data)) if data["trace"] is not None else sum(
            data["counts"].get(g, 0) for g in GEO_GATES)
        title = f"#{data['frame_id']}  lane geometry"
        dv.draw_text(canvas, title, (6, lh), C_USABLE if usable else dv.C_WHITE, fs * 1.15, th)
        (tw, _), _ = cv2.getTextSize(title, dv.FONT, fs * 1.15, th)
        dv.draw_text(canvas, f"usable {usable}  lane_offset refused {sum(offset_rej.values())}  "
                             f"geometry refused {geo_rej}", (6 + tw + 14, lh), dv.C_WHITE, fs, th)
        if off is not None:
            fmt = lambda v: "--" if v is None else f"{v:.1f}"
            info = (f"{off.mode}  L {fmt(off.left_x)}  R {fmt(off.right_x)}  "
                    f"offset {off.offset:+.3f}  conf {off.confidence:.2f}")
        else:
            info = "no lane result"
        if data["removed_edge_px"]:
            info += f"   horizontal filter removed {data['removed_edge_px']} edge px"
        dv.draw_text(canvas, info, (6, 2 * lh + 2), dv.C_WHITE, fs, th)

        # Footer
        rc = data["counts"]
        foot = (f"contours {rc.get('seen', 0)}  "
                + "  ".join(f"{GEO_SHORT[g]} {rc.get(g, 0)}" for g in ("area", "aspect", "intensity"))
                + f"  span {rc.get('w_span', 0) + rc.get('h_span', 0)}"
                + f"  other {rc.get('degenerate', 0) + rc.get('too_few_pts', 0)}"
                + f"  accepted {rc.get('accepted', 0)} -> merged {len(data['candidates'])}")
        if data["trace"] is None:
            foot += "   (no trace: geometry's rejects not drawn)"
        dv.draw_text(canvas, foot, (6, bot_y + ph + lh), dv.C_GRAY, fs, th)
        return canvas

    def _edge_panel(self, data, size, out_size) -> np.ndarray:
        """All Canny edges gray, the ones the horizontal filter removed red, kept white, closing's additions blue."""
        W, H = size
        panel = np.zeros((H, W, 3), np.uint8)
        raw, lane, closed = data["edges_raw"], data["edges_lane"], data["edges"]
        if closed is not None and lane is not None:
            panel[(closed > 0) & (lane == 0)] = C_CLOSED
        if raw is not None:
            panel[raw > 0] = C_EDGE
        if lane is not None:
            panel[lane > 0] = C_KEPT
            if raw is not None:
                panel[(raw > 0) & (lane == 0)] = C_REMOVED
        return cv2.resize(panel, out_size, interpolation=cv2.INTER_NEAREST)
