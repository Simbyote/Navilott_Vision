"""Stop-sign shapes from the color sign ROI: a redness mask, its blobs gated largest first.

Purpose:
    A red sign on a floor of similar brightness gives Canny almost no edge
    in gray, so the sign is found by color: a redness mask (Otsu, floored),
    closed, then the blobs gated largest first by area, vertex count of the
    hull's polygon (an octagon gives about 8), solidity and the ghost-sign
    gates; the first that passes is the sign. One of geometry's three
    detectors; geometry.detect_geometry() runs it.

Main package:
    find_sign_candidates(): production, candidates only.
    extract_sign_candidates(): the same with every blob's gate trace, for
        the debug views and tests.
"""

import cv2
import numpy as np
from collections.abc import Sequence

from src.params import STOP_SIGN
from src.utils import clamp

from src.perception.geometry_shared import SignCandidate, SignContourFilter, _close_edges, _contours  # noqa: F401


def _sign_confidence(
        area: float,
        vertex_count: int,
        f: SignContourFilter
    ) -> float:
    """Equal blend of closeness to 8 vertices (an octagon) and area up to ref_area, in [0, 1]."""
    vertex_score = clamp(1.0 - abs(vertex_count - 8) / 8.0, 0.0, 1.0)
    denom = max(f.ref_area - f.min_area, 1.0)
    area_score = clamp((area - f.min_area) / denom, 0.0, 1.0)
    return round(0.5 * vertex_score + 0.5 * area_score, 4)


# An area-rejected contour is traced only if its bounding box covers at least
# this fraction of min_area; smaller ones are edge noise and are only counted.
# Judged by bbox, not contour area: a Canny outline with a gap traces as a thin
# sliver with almost no enclosed area but a sign-sized box, and that is exactly
# the failure the trace is there to show
TRACE_MIN_BBOX_FRAC = 1.0

def _trace_entry(
        contour: np.ndarray,
        gate: str | None,
        area: float,
        vertices: int | None = None,
        solidity: float | None = None,
        confidence: float | None = None,
        poly: np.ndarray | None = None,
    ) -> dict:
    """One sign-trace row: bbox plus the gate that decided (None = accepted); unmeasured fields stay None."""
    return {
        "bbox": cv2.boundingRect(contour),
        "gate": gate,
        "area": area,
        "vertices": vertices,
        "solidity": solidity,
        "confidence": confidence,
        "poly": poly,
    }

def _redness(sign_color_roi: np.ndarray) -> np.ndarray:
    """R - max(G, B) per pixel, 0-255: bright on red, near 0 on gray, black and white alike."""
    b, g, r = cv2.split(sign_color_roi)
    # cv2.subtract saturates at 0; numpy '-' on uint8 would wrap gray-green pixels to ~255
    return cv2.subtract(r, cv2.max(g, b))

def _red_mask(redness: np.ndarray, sign_filter: SignContourFilter) -> tuple[np.ndarray, float]:
    """0/255 mask of the red pixels (Otsu, never below min_redness), closed; and the threshold used."""
    t, mask = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if t < sign_filter.min_redness:
        t, mask = cv2.threshold(redness, sign_filter.min_redness, 255, cv2.THRESH_BINARY)
    k = sign_filter.close_kernel
    if k >= 2:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return mask, float(t)

def _sign_polygon(contour: np.ndarray, sign_filter: SignContourFilter):
    """(hull, hull area, approxPolyDP of the hull at epsilon_factor x its perimeter)."""
    hull = cv2.convexHull(contour)
    epsilon = sign_filter.epsilon_factor * cv2.arcLength(hull, closed=True)
    return hull, cv2.contourArea(hull), cv2.approxPolyDP(hull, epsilon, closed=True)

def _letter_share(sign_color_roi: np.ndarray, hull: np.ndarray, f: SignContourFilter) -> float:
    """Share of the hull's pixels that are white (the sign's letters), in [0, 1]."""
    region = np.zeros(sign_color_roi.shape[:2], np.uint8)
    cv2.fillPoly(region, [hull.reshape(-1, 2).astype(np.int32)], 255)
    inside = region > 0
    n = int(inside.sum())
    if n == 0:
        return 0.0
    white = ((sign_color_roi.min(axis=2) >= f.letter_min_v)
             & (_redness(sign_color_roi) <= f.letter_max_redness) & inside)
    return int(white.sum()) / n


def _shape_gate(contour: np.ndarray, area: float, hull: np.ndarray, sign_color_roi: np.ndarray | None,
                f: SignContourFilter) -> tuple[str | None, dict]:
    """
    The ghost-sign gates after solidity, in order: aspect, extent, letters.
    (the gate that failed or None, the measured values). A gate whose
    setting is None is skipped; letters is skipped without the ROI.
    """
    _, _, bw, bh = cv2.boundingRect(contour)
    got = {"aspect": round(bw / max(bh, 1), 3), "extent": round(area / max(bw * bh, 1), 3)}
    if f.aspect_range is not None and not f.aspect_range[0] <= got["aspect"] <= f.aspect_range[1]:
        return "aspect", got
    if f.extent_range is not None and not f.extent_range[0] <= got["extent"] <= f.extent_range[1]:
        return "extent", got
    if f.min_letter_share is not None and sign_color_roi is not None and area >= f.letters_min_area:
        got["letters"] = round(_letter_share(sign_color_roi, hull, f), 4)
        if got["letters"] < f.min_letter_share:
            return "letters", got
    return None, got


def _gate_sign(contour: np.ndarray, area: float, sign_filter: SignContourFilter,
               sign_color_roi: np.ndarray | None) -> tuple[str | None, dict]:
    """
    One sized blob through the sign gates after min_area: max_area, hull,
    vertex count, solidity, then the ghost-sign gates. (the gate that
    refused it or None, what was measured: hull, approx, vertices,
    solidity, confidence as far as it got).
    """
    got = {}
    if area > sign_filter.max_area:
        return "area", got
    hull, hull_area, approx = _sign_polygon(contour, sign_filter)
    if hull_area <= 0:
        return "hull", got
    got.update(hull=hull, approx=approx, vertices=len(approx),
               confidence=_sign_confidence(area, len(approx), sign_filter))
    if len(approx) < sign_filter.min_vertices or len(approx) > sign_filter.max_vertices:
        return "vertices", got
    # Of the raw contour against its hull; the hull alone is always 1.0
    got["solidity"] = round(area / hull_area, 4)
    if got["solidity"] < sign_filter.min_solidity:
        return "solidity", got
    gate, _ = _shape_gate(contour, area, hull, sign_color_roi, sign_filter)
    return gate, got


def _sign_candidate(contour, area, got, frame_id, timestamp_ms) -> SignCandidate:
    return SignCandidate(
        label = STOP_SIGN,
        bbox = cv2.boundingRect(contour),
        contour = got["approx"],
        vertex_count = got["vertices"],
        confidence = got["confidence"],
        frame_id = frame_id,
        timestamp_ms = timestamp_ms,
        area = area,
        solidity = got["solidity"],
    )


def _extract_sign_candidates(
    contours: Sequence[np.ndarray],
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
    reject_counts: dict | None = None,
    trace: list[dict] | None = None,
    sign_color_roi: np.ndarray | None = None,
) -> list[SignCandidate]:
    """
    Gate the red-mask contours: below min_area dropped, then the rest
    largest first through max_area, hull, vertex count, solidity, and the
    ghost-sign gates (aspect, extent, letters; the letters read
    sign_color_roi), until one passes. Only the largest used to be gated, so
    a larger red thing in the ROI, refused, hid the sign behind it
    (sign_check on detect-range looks, 2026-10-20).

    Inputs:
        contours: External contours of the red mask.
        reject_counts: Filled with one count per gate. Every contour lands in
            exactly one bucket ("not_largest" for the smaller ones never
            gated, a larger one having passed), so the buckets sum to "seen".
        trace: A list to receive one _trace_entry() per contour that reached a
            gate, accepted or not, or None to skip tracing. Area rejects are
            traced only when their bbox clears TRACE_MIN_BBOX_FRAC.

    Outputs:
        At most one candidate, holding the hull's approxPolyDP polygon.

    Side effects:
        Mutates reject_counts and trace.
    """
    rc = reject_counts if reject_counts is not None else {}
    for _k in ("seen", "area", "not_largest", "vertices", "hull", "solidity", "aspect", "extent", "letters",
               "accepted"):
        rc.setdefault(_k, 0)

    sized = []
    for contour in contours:
        rc["seen"] += 1
        area = cv2.contourArea(contour)
        if area < sign_filter.min_area:
            rc["area"] += 1
            if trace is not None:
                _, _, bw, bh = cv2.boundingRect(contour)
                if bw * bh >= sign_filter.min_area * TRACE_MIN_BBOX_FRAC:
                    trace.append(_trace_entry(contour, "area", area))
            continue
        sized.append((area, contour))

    # Largest first; sorted() keeps equal areas in contour order, as _filter_sign_contours does
    found = None
    for area, contour in sorted(sized, key=lambda ac: -ac[0]):
        if found is not None:
            rc["not_largest"] += 1
            if trace is not None:
                trace.append(_trace_entry(contour, "not_largest", area))
            continue
        gate, got = _gate_sign(contour, area, sign_filter, sign_color_roi)
        rc[gate or "accepted"] += 1
        if trace is not None:
            trace.append(_trace_entry(contour, gate, area, got.get("vertices"), got.get("solidity"),
                                      got.get("confidence"), got.get("approx")))
        if gate is None:
            found = _sign_candidate(contour, area, got, frame_id, timestamp_ms)
    return [] if found is None else [found]

def _filter_sign_contours(
    contours: Sequence[np.ndarray],
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
    sign_color_roi: np.ndarray | None = None,
) -> list[SignCandidate]:
    """
    Production twin of _extract_sign_candidates(): same gates in the same
    order, without reject counting or tracing.

    Outputs:
        At most one candidate, holding the hull's approxPolyDP polygon.
    """
    sized = [(cv2.contourArea(c), c) for c in contours]
    for area, contour in sorted((ac for ac in sized if ac[0] >= sign_filter.min_area), key=lambda ac: -ac[0]):
        gate, got = _gate_sign(contour, area, sign_filter, sign_color_roi)
        if gate is None:
            return [_sign_candidate(contour, area, got, frame_id, timestamp_ms)]
    return []

def extract_sign_candidates(
    sign_color_roi: np.ndarray,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
    draw_overlays: bool = True,
    trace: bool = False,
) -> tuple[list[SignCandidate], dict]:
    """
    Find the stop-sign-shaped red blob in the sign ROI.

    Inputs:
        sign_color_roi: (h, w, 3) uint8 BGR.
        draw_overlays: Also build contour_overlay and accepted_overlay. False
            skips two full-ROI allocations and a contour rasterization per frame.
        trace: Record every contour that reached a gate, and the gate that
            decided it, under debug["trace"]. Off by default; the live loop
            doesn't read it.

    Outputs:
        (candidates, debug); at most one candidate. debug always holds
        sign_roi (the BGR ROI), redness (0-255), mask (0/255, closed),
        threshold (the redness threshold used) and reject_counts. With
        trace, debug["trace"] is a list of dicts (bbox, gate, area,
        vertices, solidity, confidence, poly), ROI-relative; gate is None
        for the accepted candidate.
    """
    redness = _redness(sign_color_roi)
    mask, threshold = _red_mask(redness, sign_filter)
    contours = _contours(mask)

    reject_counts = {}
    trace_log = [] if trace else None

    candidates = _extract_sign_candidates(
        contours,
        sign_filter,
        frame_id,
        timestamp_ms,
        reject_counts,
        trace_log,
        sign_color_roi,
    )

    debug_images = {
        "sign_roi": sign_color_roi,
        "redness": redness,
        "mask": mask,
        "threshold": threshold,
        "reject_counts": reject_counts,
    }
    if trace:
        debug_images["trace"] = trace_log

    if draw_overlays:
        contour_overlay = sign_color_roi.copy()
        accepted_overlay = sign_color_roi.copy()

        cv2.drawContours(contour_overlay, contours, -1, (200, 200, 200), 1)
        for c in candidates:
            cv2.drawContours(accepted_overlay, [c.contour], -1, (0, 255, 0), 2)
            x, y, w, h = c.bbox
            cv2.rectangle(accepted_overlay, (x, y), (x + w - 1, y + h - 1), (0, 200, 0), 1)
            cv2.putText(accepted_overlay, f"v={c.vertex_count} {c.confidence:.2f}",
                        (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        debug_images["contour_overlay"] = contour_overlay
        debug_images["accepted_overlay"] = accepted_overlay

    return candidates, debug_images


def find_sign_candidates(
    sign_color_roi: np.ndarray,
    sign_filter: SignContourFilter,
    frame_id: int,
    timestamp_ms: int,
) -> list[SignCandidate]:
    """
    Production twin of extract_sign_candidates(): no debug dict, overlays or trace.

    Outputs:
        ROI-relative candidates, at most one.
    """
    mask, _ = _red_mask(_redness(sign_color_roi), sign_filter)

    return _filter_sign_contours(
        _contours(mask),
        sign_filter,
        frame_id,
        timestamp_ms,
        sign_color_roi,
    )
