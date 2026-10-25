"""Obstacle view: what the detector took for "not floor", what survived, and what it measured.

Purpose:
    Checks the obstacle thresholds on real footage (make phase2 VIEWS=obstacle).
    The top panel boxes each surviving blob by fate with the measured one's
    distance; the bottom shows the not-floor and opened masks, which tell a
    never-seen obstacle from one opened away.

Main package:
    ObstacleView, a debug_video view in obstacle-ROI coordinates.
"""
import cv2
import numpy as np

import src.debugger.debug_video as dv
from src.perception.obstacle import GATES

C_TINT = (255, 120, 0)          # not-floor pixels on the color ROI
C_MASK = (110, 110, 110)        # not floor, removed by the opening
C_KEPT = (235, 235, 235)        # survived the opening
C_LANE = (0, 0, 200)            # the lane ROI's top edge (distances count from its bottom)


class ObstacleView(dv.CandidateView):
    """conf_threshold is unused, accepted so live_view builds every view alike."""
    name = "obstacle"
    CSV_FIELDS = ("frame_id", "timestamp_ms", "floor_level", "threshold", "not_floor_px", "blobs",
                  "passed", *(f"rej_{g}" for g in GATES), "detected", "distance_px", "distance_cm",
                  "clipped", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area_px")

    def __init__(self, conf_threshold=None, zoom=2):
        super().__init__(None, zoom)
        self._frames = 0
        self._detected = 0
        self._clipped = 0
        self._distances = []
        self._distances_cm = []
        self._floor = []
        self._counts = {}

    def extract(self, chain, frame=None) -> dict:
        """This frame's obstacle data from chain.roi, chain.obstacle and chain.obstacle_debug."""
        dbg = getattr(chain, "obstacle_debug", None) or {}
        roi = chain.roi
        return {
            "frame_id": roi.frame_id,
            "timestamp_ms": roi.timestamp_ms,
            "roi": getattr(roi, "obstacle_color_roi", None),
            "rect": getattr(roi, "obstacle_rect", None),
            "lane_rect": roi.lane_rect,
            "mask": dbg.get("mask"),
            "opened": dbg.get("opened"),
            "floor_level": dbg.get("floor_level"),
            "threshold": dbg.get("threshold"),
            "not_floor_px": dbg.get("not_floor_px", 0),
            "blobs": list(dbg.get("blobs", ())),
            "counts": dict(dbg.get("reject_counts", {})),
            "measured": getattr(chain, "obstacle", None),
        }

    @staticmethod
    def _measured(data):
        m = data["measured"]
        return m if m is not None and m.detected else None

    def observe(self, data):
        self._frames += 1
        m = self._measured(data)
        if m is not None:
            self._detected += 1
            self._clipped += int(m.clipped)
            self._distances.append(m.distance_px)
            if m.distance_cm is not None:
                self._distances_cm.append(m.distance_cm)
        if data["floor_level"] is not None:
            self._floor.append(data["floor_level"])
        for gate, n in data["counts"].items():
            self._counts[gate] = self._counts.get(gate, 0) + n

    def row(self, data):
        m, rc = self._measured(data), data["counts"]
        opt = lambda v: "" if v is None else v
        box = m.bbox if m is not None else ("", "", "", "")
        return [
            data["frame_id"], data["timestamp_ms"],
            "" if data["floor_level"] is None else round(data["floor_level"], 1),
            "" if data["threshold"] is None else round(data["threshold"], 1),
            data["not_floor_px"], len(data["blobs"]), rc.get("passed", 0), *(rc.get(g, 0) for g in GATES),
            int(m is not None), opt(m and m.distance_px), opt(m and m.distance_cm),
            "" if m is None else int(m.clipped), *box, opt(m and m.area_px),
        ]

    def report(self):
        n = max(self._frames, 1)
        out = [f"[OBSTACLE] {self._frames} frames",
               f" frames with an obstacle                     {self._detected:6}  "
               f"({100 * self._detected / n:5.1f}%)",
               f"   of them at the view bottom (clipped)      {self._clipped:6}"]
        for label, values, unit in (("distance to the view bottom", self._distances, "px"),
                                    ("distance ahead on the floor", self._distances_cm, "cm")):
            if values:
                d = sorted(values)
                out.append(f" {label}, {unit}: min {d[0]:.1f}  med {d[len(d) // 2]:.1f}  max {d[-1]:.1f}")
        if self._floor:
            f = sorted(self._floor)
            out.append(f" floor level (gray): min {f[0]:.0f}  med {f[len(f) // 2]:.0f}  max {f[-1]:.0f}")
        rej = {g: self._counts.get(g, 0) for g in GATES if self._counts.get(g)}
        if rej:
            out.append(" blobs rejected, by gate:")
            for gate, v in sorted(rej.items(), key=lambda kv: -kv[1]):
                out.append(f"  {gate:<20}{v:6}")
        return out

    def render(self, data, scale=1):
        s, fs, th, lh, hh, fh = self._metrics(scale)
        roi = data["roi"]
        if roi is None or data["mask"] is None:
            img = np.zeros((hh + 40, 320, 3), np.uint8)
            dv.draw_text(img, f"#{data['frame_id']}  no obstacle ROI", (6, lh), dv.C_RED, fs, th)
            return img

        H, W = roi.shape[:2]
        pw, ph, gap = W * s, H * s, dv.GAP_PX * s
        top_y, bot_y = hh, hh + ph + gap
        canvas = np.zeros((hh + 2 * ph + gap + fh, pw, 3), np.uint8)
        canvas[top_y:top_y + ph] = cv2.resize(self._tinted(roi, data["mask"]), (pw, ph),
                                              interpolation=cv2.INTER_NEAREST)
        canvas[bot_y:bot_y + ph] = cv2.resize(self._masks(data), (pw, ph), interpolation=cv2.INTER_NEAREST)
        dv.draw_text(canvas, "obstacle ROI, not floor tinted", (4, top_y + lh), dv.C_GRAY, fs, th)
        dv.draw_text(canvas, "not floor (gray) / after opening (white)", (4, bot_y + lh), dv.C_GRAY, fs, th)

        pt = lambda x, y, y0: (int(round(x * s)), int(round(y0 + y * s)))

        # Lane ROI's top edge, across both panels
        lane_top = data["lane_rect"][1] - data["rect"][1]
        if 0 < lane_top < H:
            for y0 in (top_y, bot_y):
                cv2.line(canvas, pt(0, lane_top, y0), pt(W, lane_top, y0), C_LANE, 1)
            dv.draw_text(canvas, "lane ROI top", (4, top_y + int(lane_top * s) - 3), C_LANE, fs * 0.8, th)

        # Every blob that survived the opening, by fate
        for b in data["blobs"]:
            x, y, w, h = b["bbox"]
            color = dv.C_USABLE if b["gate"] is None else dv.C_RED
            for y0 in (top_y, bot_y):
                cv2.rectangle(canvas, pt(x, y, y0), pt(x + w, y + h, y0), color, th + (y0 == top_y))
            label = f"{w}x{h} {b['rows_ahead']:.0f}px" if b["gate"] is None else f"{b['gate']} {w}x{h}"
            anchor = pt(x, y, top_y)
            dv.draw_text(canvas, label, (anchor[0] + 2, max(anchor[1] - 3, top_y + lh)), color, fs, th)

        # The measured one, and its distance down to the view bottom
        m = self._measured(data)
        if m is not None:
            x, y, w, h = m.bbox
            cv2.rectangle(canvas, pt(x, y, top_y), pt(x + w, y + h, top_y), dv.C_WHITE, 1)
            p0, p1 = pt(x + w / 2.0, y + h, top_y), pt(x + w / 2.0, H, top_y)
            if p1[1] - p0[1] > 2:
                cv2.arrowedLine(canvas, p0, (p1[0], p1[1] - 1), dv.C_WHITE, 1, cv2.LINE_AA, tipLength=0.2)
            dv.draw_text(canvas, self._distance(m), (p0[0] + 4, min(p0[1] + lh, top_y + ph - 3)),
                         dv.C_WHITE, fs, th)

        rc = data["counts"]
        head_color = dv.C_RED if m is not None else (dv.C_AMBER if data["blobs"] else dv.C_WHITE)
        title = f"#{data['frame_id']}  obstacle"
        dv.draw_text(canvas, title, (6, lh), head_color, fs * 1.15, th)
        (tw, _), _ = cv2.getTextSize(title, dv.FONT, fs * 1.15, th)
        dv.draw_text(canvas, f"passed {rc.get('passed', 0)}  rejected {len(data['blobs']) - rc.get('passed', 0)}",
                     (6 + tw + 14, lh), dv.C_WHITE, fs, th)
        info = (f"IN PATH: {self._distance(m)}  {m.bbox[2]}x{m.bbox[3]} area {m.area_px}"
                if m is not None else "path clear")
        info += f"   floor {data['floor_level']:.0f}  bright >= {data['threshold']:.0f}"
        dv.draw_text(canvas, info, (6, 2 * lh + 2), dv.C_WHITE, fs, th)

        foot = (f"not floor {data['not_floor_px']} px   blobs {len(data['blobs'])}   "
                + "  ".join(f"{g} {rc.get(g, 0)}" for g in GATES))
        dv.draw_text(canvas, foot, (6, bot_y + ph + lh), dv.C_GRAY, fs, th)
        return canvas

    @staticmethod
    def _distance(m) -> str:
        if m.clipped:
            text = "AT VIEW BOTTOM"
            return text + ("" if m.distance_cm is None else f" (<= {m.distance_cm:.1f}cm)")
        return f"{m.distance_px:.0f}px" + ("" if m.distance_cm is None else f" {m.distance_cm:.1f}cm")

    @staticmethod
    def _tinted(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Not-floor pixels blended toward C_TINT."""
        out = roi.copy()
        sel = mask > 0
        out[sel] = (0.55 * out[sel] + 0.45 * np.array(C_TINT)).astype(np.uint8)
        return out

    @staticmethod
    def _masks(data) -> np.ndarray:
        """Gray where not floor, white where it survived the opening."""
        mask, opened = data["mask"], data["opened"]
        panel = np.zeros((*mask.shape, 3), np.uint8)
        panel[mask > 0] = C_MASK
        if opened is not None:
            panel[opened > 0] = C_KEPT
        return panel
