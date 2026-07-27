# AutoBot Vision Pipeline — Summer Work Summary

**Branch:** `feature/vision-pipeline-rework` (off current working state)

---

## Problems Identified

### 1. ROI Too Wide

`lane_roi` is full-width lower half (480×180). Course walls and floor edges at the lateral extremes pass the elongation filter and corrupt anchor selection.

**Fix:**

- Horizontal inset margin (~12% each side): `lane_roi = frame[H//2:H, margin:W-margin]`
- Trim top of lane_roi to `H*0.6` — excludes far-field vanishing zone noise

### 2. Center Dividing Lines Not Detected

Two sub-problems:

- Dashed lines fail elongation filter (each dash too short individually) — fix with vertical `cv2.dilate` on Canny output before contour extraction to bridge gaps
- Even when detected, Stage 5 anchor logic picks absolute leftmost/rightmost candidates — this places the robot's center reference on the full road midpoint, not its lane. Fix by constraining: left anchor must fall in the left half of ROI, right anchor in the right half.

### 3. Robot Drives Into Poles

Vertical PVC poles produce tall, thin, near-vertical contours that pass the elongation filter. Geometry branch has no orientation check.

**Fix — orientation filter via `cv2.minAreaRect`:**

```python
(cx, cy), (w, h), angle = cv2.minAreaRect(contour)
long_axis_angle = angle if w < h else (90 + angle)
if abs(long_axis_angle) > 65:   # degrees from horizontal — tune on course
    continue
```

Secondary: reject contours whose centroid falls in the top 25% of lane_roi (far-field, near-vertical objects).

---

## Architecture Changes

### Trapezoid Lane Mask (replaces full-width lane_roi)

A fixed perspective trapezoid applied as a pre-Canny mask in `geometry_branch.py`. Calibrated once from a representative debug frame on the actual course. Encodes the lane's perspective projection directly — walls, poles, and adjacent lanes fall outside it.

```
bottom-wide ──────────────── (near field, robot position)
     \                      /
      \                    /
       top-narrow ─────── (vanishing point)
```

Four corner points become calibration constants. Applied as:

```python
trap_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
cv2.fillPoly(trap_mask, [LANE_TRAP], 255)
masked_roi = cv2.bitwise_and(roi, roi, mask=trap_mask)
```

Pre-compute mask once at startup. Per-frame cost <0.5ms.

**Turn behavior:** On sharp turns the lane exits the trapezoid, detection drops out, `mode` → `"none"`, Phase 3 falls back to IMU dead-reckoning. This is expected and handled.

### Why Flood Fill Was Rejected

`cv2.floodFill` from a seed at the bottom-center was considered. Rejected because: sensitive to lighting-induced false Canny edges causing fill leakage into adjacent regions, unpredictable worst-case compute spikes, and harder to debug on-device. Trapezoid is deterministic and calibration-stable.

---

## Pipeline Mode State Machine

Replaces the implicit mode inference from `lane_offset.mode` alone. The linker holds current `PipelineMode` as persistent state between frames.

```
PipelineMode (enum)
    LANE_FOLLOW          Normal operation — trapezoid + contours + offset
    INTERSECTION_HOLD    Signature detected — accumulating confirmation frames
    TURN_EXECUTING       Protocol active — vision suspended, IMU carries
    TURN_RECOVERY        Turn complete — re-acquiring lane
```

### Dispatch (in linker)

```
frame in → check PipelineMode
    LANE_FOLLOW      → trapezoid mask → Canny → contours → offset
    INTERSECTION_HOLD → edge ratio accumulation only
    TURN_EXECUTING   → IMU yaw integration only (skip vision entirely)
    TURN_RECOVERY    → full lane pass, watch for TWO_BOUNDS return
```

### Transition Logic

```
LANE_FOLLOW
    intersection_conf crosses threshold → INTERSECTION_HOLD

INTERSECTION_HOLD
    N consecutive frames above threshold → TURN_EXECUTING
    signature drops before N → back to LANE_FOLLOW

TURN_EXECUTING
    IMU yaw ≥ target delta → TURN_RECOVERY

TURN_RECOVERY
    Stage 5 returns TWO_BOUNDS → LANE_FOLLOW
    timeout without TWO_BOUNDS → fallback (stop or continue on IMU)
```

### Intersection Detection

Derived from the existing Canny output — no new stage. Edge pixel density ratio between upper and lower halves of the trapezoid:

```python
upper_half = canny_output[:H//2, :]
lower_half = canny_output[H//2:, :]
edge_ratio = np.sum(upper_half) / (np.sum(lower_half) + 1e-6)
# High ratio → far-field opened up → intersection signature
```

Accumulate across N frames in `INTERSECTION_HOLD` before committing to a turn — prevents single-frame false positives.

### Confidence Scoring Per Mode

| Mode | What confidence measures |
|---|---|
| `LANE_FOLLOW` | Contour elongation + area of left/right boundary candidates |
| `INTERSECTION_HOLD` | Edge ratio upper/lower trapezoid + frame-over-frame area change |
| `TURN_EXECUTING` | IMU yaw completion toward target angle (0.0 → 1.0) |
| `TURN_RECOVERY` | First frame returning `TWO_BOUNDS` from Stage 5 |

### Phase 2 Output Extension

```
Phase2Output
    detections
    lane_offset          populated in LANE_FOLLOW / TURN_RECOVERY
    pipeline_mode        NEW — current state machine mode
    intersection_conf    NEW — accumulated intersection score
    frame_id
    timestamp_ms
```

Phase 3 dispatches on `pipeline_mode` directly rather than inferring state from offset validity alone.

---

## Computational Cost

| Addition | Cost | Notes |
|---|---|---|
| Trapezoid bitwise_and | <0.5ms | Mask pre-computed at startup |
| Edge ratio (np.sum ×2) | <0.1ms | Reuses existing Canny output |
| State machine dispatch | negligible | Python if/elif on enum |
| `TURN_EXECUTING` mode | −8–10ms | Vision fully suspended |

Net effect in `LANE_FOLLOW` is +~0.5ms. System spends the majority of frames there. Headroom against the 50ms budget is not materially affected.

---

## Priority Order

### Do First — Lane Following (Summer)

1. Apply horizontal ROI inset margin
2. Implement trapezoid mask — calibrate four points from a debug frame on the actual course
3. Add orientation filter for poles in `geometry_branch.py`
4. Fix anchor spatial constraint in `lane_offset.py` (left anchor left-half, right anchor right-half)
5. Test on a single straight — confirm the robot stays centered and recovers from a manual nudge
6. Extend to full course straights before touching turns

### Do Second — Intersection + Turns (D2)

1. Implement `PipelineMode` state machine in the linker
2. Add intersection edge ratio detection
3. Wire `TURN_EXECUTING` to IMU yaw integration
4. Tune `INTERSECTION_HOLD` frame count threshold on the actual course
5. Validate `TURN_RECOVERY` TWO_BOUNDS re-acquisition

### Do Last — Traffic Rules Scaffold (D2)

1. Lower confidence thresholds for stop sign and traffic light from above 1.0 to operational values
2. Wire stop behavior as a mode transition (same pattern as intersection handling)
3. Traffic light state gates motor enable — straightforward once lane following and intersection are stable

---

## Branch Strategy

All of the above lives in `feature/vision-pipeline-rework`, branched off the current working state. The existing functional pipeline remains untouched in `main` (or current working branch). Once lane following is validated on-device and stable, merge in stages — lane fixes first, then state machine, then traffic rules.

**Key constraint:** All testing from this point forward is on the assembled robot on the actual course. Dataset-based offline harness testing is no longer applicable for these changes — the trapezoid calibration and turn thresholds are course-geometry-dependent and must be tuned on the device.
