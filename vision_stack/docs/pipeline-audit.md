# Phase 2 → Phase 3 Seam Audit — Navilott Vision Pipeline

**Scope:** `run_pipeline.py`, `geometry.py`, `estimation.py`, `lane_offset.py`, with supporting reads of `roi_crop.py`, `feature_fusion.py`, `preprocess.py`, `imu.py`.
**Evidence:** `session2.log` (2,882 navigation frames, 03:51 runtime), plus executable replays of the actual `Phase3Processor` and `compute_lane_offset` code against reconstructed scenes.

---

## TLDR

The two anomalies are not two bugs. They are two symptoms of one structural fault:

> **`compute_lane_offset()` — the only function in the codebase that implements the documented leftmost/rightmost midpoint contract — has its result discarded. It never reaches the motor loop.** The value that actually steers the robot is computed independently inside `estimation._filter_lane()` as the *unweighted arithmetic mean of every surviving lane-boundary centroid*, after passing through a motion-consistency gate that structurally cannot work on multi-boundary frames.

Two divergent implementations of "lane offset" exist. Phase 2's is correct and unused. Phase 3's is used and wrong. In the log, `lane_mode=two_boundary` appears on 2,563 frames — that field is the *unused* estimator narrating what the robot *would* have done.

**Measured impact (replay of a perfectly centred, perfectly detected two-boundary lane):** `compute_lane_offset` returns `+0.0000` every frame; `Phase3Processor` returns values swinging from `−0.1750` to `+0.0304` m — up to **50 % of full scale (±0.35 m) of fabricated error on a stationary, correctly-perceived, centred robot.**

---

# Phase 1 — Dependency Hierarchy and Data-Flow Mapping

## 1.1 Dependency hierarchy

`run_pipeline.py` sits at the root and is the only module that imports across phase boundaries. It has no parent; nothing imports it.

Directly beneath the root sit eight siblings, in execution order. **`system.py`** and **`imu.py`** are peers hanging off the root but outside the per-frame vision chain — `system.py` owns the start button and TM1637 display, `imu.py` owns the MPU-6050 background thread. Neither imports any vision module, so both are leaves.

The vision chain proper begins with **`preprocess.py`**, a leaf that takes a frame and returns a frame. Below it, **`roi_crop.py`** is the first branching node: it produces three sibling outputs (`lane_roi`, `traffic_roi`, `sign_roi`) that fan out to two independent children. `roi_crop.py` has one non-obvious child of its own, `unzip_data`, imported at module scope purely for its test block.

The fan-out's left child is **`color_branch.py`** (traffic lights, HSV). The right child is **`geometry.py`** (lanes + stop signs, Canny/contour). These two are siblings that never reference each other; `geometry.py` also carries the same module-scope `unzip_data` import.

Both branches converge on **`feature_fusion.py`**, the first join node. It is the parent of the canonical Phase 2 `DetectionObject` schema and the only place branch-specific candidate types are erased.

From `feature_fusion.py` the tree forks again into two children that are *not* in series — this is the critical structural detail. **`lane_offset.py`** is one child; it imports `DetectionObject` from `feature_fusion` and produces `LaneOffsetResult`. **`phase2_out.py`** is the sibling child; it packages the same detections for downstream consumption. `lane_offset.py`'s output flows only to the logger and the debug overlay — **it is a terminal leaf with no path to the controller.** `phase2_out.py`'s output flows onward.

**`estimation.py`** is a child of `phase2_out.py` *by way of a hand-written adapter that lives in the root* (`_adapt_detections_for_p3`), not by import. `estimation.py` imports nothing from the vision stack — it re-declares its own mirror copies of `DetectionObject` and `Phase2Output`. It is therefore a structurally *detached* subtree: the two schemas are coupled only by a 12-line function in the root file and by prose in docstrings. Nothing enforces their agreement.

Finally, the motor loop is not a module at all. The discrete PD controller is inlined in the root's `while` loop, consuming `estimation.py`'s output directly. **There is no navigation subsystem node in this tree** — the `EstimationPacket` handoff to Ignacio's subsystem is, today, a handoff to twelve lines of arithmetic in `main()`.

**Two properties fall out of this shape:**

1. The tree has a **dead branch** (`lane_offset.py`) that is fully implemented, documented, and unreachable from the actuator.
2. The `estimation.py` subtree is **detached** — no import edge crosses the P2→P3 boundary, so no type checker, linter, or import-time failure can ever detect a contract drift there.

## 1.2 Data-flow parameter table

| # | Producer → Consumer | Payload | Type | Frame / Units | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | GStreamer → `main()` | `frame_bgr` | `ndarray (360,480,3) uint8` | **BGR**, 180°-rotated | Caps request `colorimetry=bt709` w/o `format=`; negotiated **NV21**, `videoconvert` → BGR |
| 2 | `main()` → `preprocess_frame` | `frame_yuv` | `ndarray uint8` | YUV | Converted BGR→YUV to satisfy stage contract |
| 3 | `preprocess_frame` → `main()` | `preprocessed` | `ndarray uint8` | YUV | Converted back to BGR |
| 4 | `main()` → `crop_rois` | `preprocessed_bgr` | `ndarray uint8` | BGR | Docstring says "uint8 YUV"; **contract mismatch** |
| 5 | `crop_rois` → branches | `lane_roi` | `ndarray (180,480,3)` **view** | ROI-local px, origin `(0, H//2)` | **Full frame width.** No horizontal crop — see §3.2 |
| 6 | " | `traffic_roi` | `ndarray (180,240,3)` view | origin `(W//4, 0)` | |
| 7 | " | `sign_roi` | `ndarray (360,240,3)` view | origin `(W//2, 0)` | |
| 8 | " | `lane_rect`,`traffic_rect`,`sign_rect` | `tuple (x,y,w,h)` | source-frame px | **Produced, never consumed.** The only data that could de-localise ROI coords |
| 9 | `geometry` → `feature_fusion` | `lane_candidates[]` | `List[LaneCandidate]` | `bbox` ROI-local px; `confidence` ∈ [0, **0.745**] | Cap is 0.745, not 1.0 — see §3.1 |
| 10 | `feature_fusion` → `lane_offset` / `phase2_out` | `detections[]` | `List[DetectionObject]` | `position` = **`dict{'x','y'}`**, ROI-local px | **All** lane candidates forwarded, sorted by confidence DESC |
| 11 | `main()` → `compute_lane_offset` | `frame_width` | `int` = `lane_roi.shape[1]` = 480 | px | |
| 12 | `compute_lane_offset` → **logger + overlay only** | `LaneOffsetResult` | dataclass | `offset` **normalised [−1,+1]** | **TERMINATES HERE. Not consumed by control.** |
| 13 | `main()` → `_adapt_detections_for_p3` | `p2_out.detections`, `lane_roi_width` | list, float | | `position['x'] − 240.0` applied to lane types only |
| 14 | adapter → `Phase3Processor` | `P3DetectionObject.position_x` | `float` | **signed px from image centre** | Contract says "lane **centre**", supplied "**each boundary**" — see §3.1 |
| 15 | `imu.snapshot()` → `SensorSample` | `yaw_rate`,`lateral_accel` | `float│None` | deg/s, m/s² | `wheel_speed`, `distance_traveled` **hard-wired `None`** |
| 16 | `Phase3Processor` → `main()` | `EstimationPacket.lane_offset` | `float` | **metres**, ±0.35 | Different unit *and* different scale from #12 |
| 17 | " | `.heading_error` | `float` | degrees | `= lane_offset × 30.0`, double-EMA'd. No independent information |
| 18 | " | `.drive_state`,`.stop_sign_detected` | `str`,`bool` | | Gated off by 1.1 thresholds; `go`/`F` on all 2,882 frames |
| 19 | `main()` → `_drive` | `ramped_base ∓ correction` | `float` | [−1,1] duty | `correction` is **not** ramped — see §3.2 |
| 20 | `main()` ↔ `main()` | `_last_error` | module global | metres | **Not reset** in the `stop` branch |

**Absent from the packet entirely:** `lane_confident`, `mode`, `boundary_count`, `lane_width_px`, `deadreck_frames`, any staleness or validity flag. `_filter_lane` computes `lane_confident` and returns it — `process()` receives it, uses it to pick a heading source, and then **drops it**. This is the single field whose absence causes Anomaly 1.

---

# Phase 2 — Coordinate, Unit, and Type Safety Audits

## 2.1 Coordinate frames

**C-1 — Documented re-projection never happens.** `geometry.py`'s module header states: *"All coordinates are ROI-relative. Feature fusion is responsible for re-projecting into source frame coordinates."* `feature_fusion._centroid()` returns `bbox` centroids verbatim. No re-projection exists anywhere in the codebase. `lane_rect`/`sign_rect`/`traffic_rect` (row 8) are computed for exactly this purpose and never read.

Consequences by branch:

- **lane** — masked by luck. `lane_x = 0`, so x needs no correction; `y` is silently off by `H//2 = 180 px`.
- **sign** — `position_x` is short by `W//2 = 240 px`. A stop sign at true x=400 reports as 160.
- **traffic** — short by `W//4 = 120 px`.

This is currently inert only because sign/traffic are gated off by the 1.1 confidence thresholds. **Un-gating them for D2 will surface it immediately**, and the failure will look like a detection problem rather than a coordinate problem.

**C-2 — Three coordinate frames share one name.** `lane_offset` means: normalised lane-half-widths in `lane_offset.py`; signed pixels in the adapter and `estimation.DetectionObject`; metres in `EstimationPacket`. `pipeline.md` line 190 documents the units as *"Pixels (lane offset normalized to [-1.0, +1.0])"*; `architecture.md` line 94 documents the same wire field as *"lane_offset (meters, signed float)"*. The two governing documents disagree, and the code implements the second.

**C-3 — `px_per_meter` is a fiction with real consequences.** `(FRAME_WIDTH/2)/0.35 = 685.7 px/m` asserts that half the image width equals 35 cm of lateral travel — a flat, fronto-parallel, un-projected mapping on a forward-looking camera. This was an accepted trade-off when homography was dropped, and as a *scaling constant* it is fine. But it is not fine as a *physical* constant, and it is used as one in `_motion_consistency` (see U-2). Anything downstream that treats `EstimationPacket.lane_offset` as true metres — Ignacio's subsystem will, because `architecture.md` says "meters" — is consuming a number with no metric meaning at range.

## 2.2 Units

**U-1 — Heading is not a heading.** `_filter_heading` computes `raw_heading = lane_offset_ema × HEADING_SCALE` with `HEADING_SCALE = 30.0` "deg per metre of lateral offset". Confirmed in the log: frame 0, `offset=+0.1307`, `head=+3.92` (= 0.1307 × 30). A *lateral displacement* is being relabelled as an *angular deviation* by a constant. `pipeline.md` lists `heading_error`'s primary source as "Vision (lane geometry)" — no lane geometry (slope, vanishing point, line angle) is ever computed. Lane contour orientation is available from `cv2.minAreaRect` and discarded.

Measured over the full run: `corr(offset, heading) = +0.41`, ratio median 22.2 rather than 30 — because `heading_error` is EMA'd *on top of* an already-EMA'd offset (double lag) and silently switches to IMU integration during holds. So the two packet fields are neither independent nor consistently related. A consumer fusing both is double-counting one measurement with two different lags.

**U-2 — Dimensional error in the jump threshold.**

```python
speed_scale = 1.0 + sensor_sample.wheel_speed * dt * cfg.px_per_meter
```

`[m/s] × [s] × [px/m] = [px]`. The expression is `1.0 + pixels` — a dimensionless literal added to a dimensioned quantity. At 0.3 m/s and dt=0.055 s this yields 1 + 11.3 = 12.3, immediately saturating the `min(..., 3.0)` clamp, so the "speed-scaled threshold" is really a binary switch between 80 px and 240 px. Currently dormant (`wheel_speed` is `None`), and **it will activate the moment encoders are wired in** — silently tripling the gate and changing lane-keeping behaviour with no code change.

**U-3 — `OFFSET_TRIM` unit is now correct but its provenance is not.** `OFFSET_TRIM = 0.0 # meters` is added to `nav_packet.lane_offset`, which is metres. Consistent. But any previously-tuned non-zero value was tuned against the current `_filter_lane` behaviour, i.e. against the mean-of-survivors bias. Re-tune only after Fix R1.

## 2.3 Sign conventions

**S-1 — `lane_offset.py` contradicts itself internally.** The module header says *"−1.0: robot is at the left boundary / +1.0: robot is at the right boundary."* The `LaneOffsetResult` docstring twelve lines later says *"Negative = robot is right of lane center / Positive = robot is left of lane center."* These are opposite.

**S-2 — and it contradicts itself in code.** Two branches, two polarities:

| Mode | Formula | Polarity |
| --- | --- | --- |
| `two_boundary` | `(lane_center − frame_center) / (lane_width_px/2)` | + when lane appears right |
| `left_only` / `right_only` | `(frame_center − left_x) / frame_center` | **+ when boundary appears left** |
| `width_rejected` | `(frame_center − mid_x) / frame_center` | **inverted** |

**In the analysed run, 195 frames (6.8 %) used an inverted-sign branch** (`left_only` 26, `right_only` 48, `width_rejected` 121) — the exact frames where a lane is partially lost, i.e. at intersections and near walls.

**S-3 — `estimation.py` contradicts the documentation.** `EstimationPacket` docstring: *"Positive = robot is right of center."* `pipeline.md` line 198 and `phase2_pipeline.md` line 300: *"A positive offset means the robot is left of lane center and must steer right."* The code implements the **documentation's** convention (`position_x = centroid_x − 240`; lane markings right of centre ⇒ positive ⇒ robot left of centre). So `estimation.py`'s own docstring is the outlier — and it is the docstring Ignacio will read, because `estimation.py` is the handoff module.

**S-4 — Loop polarity is empirically correct; do not "fix" it blind.** `corr(offset[t], Δoffset[t]) = −0.271` across 2,881 transitions — the loop is mean-reverting, so the composite of camera mount, 180° flip, `_drive`'s double right-motor inversion, and the PD sign closes correctly *today*. **This is a load-bearing accident.** `_drive` applies two independent inversions on the right channel (`right_speed = -right_speed`, plus `bin1/bin2` assigned with the opposite polarity to `ain1/ain2`) that cancel. Anyone tidying either one will invert the robot. Fix the docstrings, add a comment at `_drive`, and change nothing else in the sign chain until the seam bugs below are resolved.

## 2.4 Type safety

| ID | Finding |
| --- | --- |
| T-1 | `run_geometry_branch`, `extract_lane_candidates`, `extract_sign_candidates`, `fuse_detections`, `_adapt_detections_for_p3` all declare bare `-> tuple` / `-> list`. Element types are opaque; no checker can verify the seam. |
| T-2 | Position is `dict{'x','y'}` in Phase 2, flat `position_x/position_y` in Phase 3. Two schemas, one hand-written adapter, zero validation. A silent `KeyError` on a renamed key is the *good* outcome; a silent semantic drift is the likely one. |
| T-3 | `estimation.DetectionObject.timestamp` is declared and populated. `Phase3Processor` reads it **zero** times. Per-detection timestamps are discarded; `dt` derives solely from `phase2_out.timestamp_ms`. |
| T-4 | `timestamp_ms = int(time.time() * 1000)` — wall clock, not monotonic. NTP step or `timedatectl` sync produces a negative `dt`, silently swallowed by `max(0.0, min(dt, 0.5))`. Use `time.perf_counter()`; it is already imported and already used for the budget check three lines away. |
| T-5 | `_last_error` is a module-level global mutated via `global`. Untestable, not reset on stop. |
| T-6 | `geometry.py` and `roi_crop.py` import `unzip_data` (a dataset fetcher) at **module scope**. A test-only dependency is on the production import path; absence is an import-time crash. |
| T-7 | `_filter_heading` returns `self._heading_error_ema.value or 0.0` — truthiness on a float. Benign here (0.0 → 0.0), but it is the pattern that caused the `None`-check inconsistency two lines above, which uses an explicit `is not None`. |
| T-8 | `crop_rois` validates `dtype`/`ndim` and raises on violation, but its docstring declares YUV while `run_pipeline` passes BGR. The one guard that *would* catch the colour-space error is a shape guard, which BGR passes. |

**T-9 — the colour-space contract is broken end-to-end.** `run_pipeline` passes **BGR** into `run_geometry_branch` (its docstring says "uint8 BGR … @TODO: change to YUV"). `geometry._to_grayscale` then runs:

```python
roi_bgr = cv2.cvtColor(roi, cv2.COLOR_YUV2BGR)   # input is already BGR
return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
```

BGR data is reinterpreted as YUV and run through the YUV→BGR matrix. The resulting "grayscale" is a scrambled linear combination of the true channels. Every downstream number — Canny thresholds `(10, 160)`, `min_intensity = 80`, `_mean_contour_intensity` — was empirically tuned against this scrambled image. **They are therefore self-consistent and "working."** Fixing the conversion without simultaneously re-calibrating the three constants will make lane detection worse, not better. Treat T-9 and its constants as one atomic change (R6), not a quick fix.

---

# Phase 3 — Logic and Contract Compliance

## 3.1 Contract violations at the P2→P3 seam

**V-1 (root cause, both anomalies) — the `position_x` contract is inverted in cardinality.**

`estimation.py` declares, three separate times:
> *"For `lane_boundary`: `.position_x` is the signed pixel offset of **the lane center** from the image center column."*

`_adapt_detections_for_p3` supplies the signed offset of **each individual boundary**, one detection per boundary. `feature_fusion` forwards *all* valid lane candidates. `_filter_lane` then does:

```python
avg_px = sum(d.position_x for d in lane_dets) / len(lane_dets)
```

The mean of individual boundary offsets equals the lane-centre offset **only if the surviving detections are laterally symmetric**. They essentially never are: a dashed centre line fragments into 3–5 contours while a solid edge line yields 1, so the mean is pulled toward whichever side fragmented more. Meanwhile `compute_lane_offset` — which implements the documented `(leftmost + rightmost)/2` reduction correctly — has its result thrown away.

**V-2 — the mode contract is unimplementable.** `phase2_pipeline.md` line 313: *"Phase 3 reads `mode` to determine whether to trust `offset` directly or substitute a dead-reckoned estimate."* Phase 3 is never passed `mode`, or `offset`, or `LaneOffsetResult` at all. The documented arbitration mechanism does not exist.

**V-3 — anchor selection contradicts its own documentation.** `phase2_pipeline.md` line 289: *"The **highest-confidence** left and right anchors are selected by x-position."* The code selects `lanes_by_x[0]` and `lanes_by_x[-1]` — **leftmost and rightmost, ignoring confidence entirely**. This is precisely what makes the estimator wall-vulnerable: a single spurious edge contour anywhere becomes an anchor by virtue of position alone.

**V-4 — the motion-consistency gate is structurally incapable of working.** `_CentroidTracker` holds **one** `(last_x, last_y)` slot per detection *class*, but is called once per *detection*:

```python
for det in detections:
    if det.type == "lane_boundary":
        ok = self._lane_tracker.update(det.position_x, det.position_y, jump_thresh)
```

With two boundaries present, the tracker compares the right boundary against the left boundary **within the same frame**. Two lines 240 px apart against an 80 px threshold ⇒ guaranteed rejection. Three compounding defects:

- **Wrong comparand.** Never compares the same physical line at *t−1* vs *t*.
- **State poisoned on reject.** `self.last_x, self.last_y = x, y` executes *before* the `return dist <= threshold`. Rejected outliers become the next frame's reference.
- **Order-dependent.** `feature_fusion` sorts by confidence DESC, so iteration order flips whenever confidence ranking flips — non-deterministic gate behaviour on identical scenes.

**Replay, perfectly centred lane, boundaries at ±120 px, confidence ranking alternating:**

```
f=0  order=[-120,+120]  gate=[PASS, REJECT(d=240)]   -> avg=-120px  offset=-0.175
f=1  order=[+120,-120]  gate=[REJECT(d=180), REJECT(d=240)]  -> NO SURVIVORS
f=2  order=[-120,+120]  gate=[REJECT(d=420), REJECT(d=240)]  -> NO SURVIVORS
f=3+ ... every subsequent frame: BOTH REJECTED
```

Phase 2 reports `two_boundary` throughout. Phase 3 reports a frozen stale value. **This is the silent failure.**

**Log confirmation:** 647 frames (22.4 %) sit inside a frozen-offset run. Of those, **only 124 had `lane_mode=none`**. The other **523 frames — 81 % of all holds — had Phase 2 successfully reporting a lane while Phase 3 discarded it and held a stale number.**

| `lane_mode` reported by Phase 2 *during* a Phase 3 hold | Frames |
| --- | ---: |
| `two_boundary` | 414 |
| `none` | 124 |
| `width_rejected` | 45 |
| `right_only` | 41 |
| `left_only` | 23 |

**V-5 — dead-reckoning never expires and staleness is never signalled.** The two terminal branches of `_filter_lane` are behaviourally identical:

```python
if self._deadreck_frames < cfg.deadreck_max_frames:
    self._deadreck_frames += 1
    return self._last_lane_offset_m, False
return self._last_lane_offset_m, False        # "Stale — hold silently"
```

`deadreck_max_frames = 10` is inert; the counter saturates and the same value is returned forever. Replay confirms frames 11–15 are indistinguishable from frames 1–10. And `lane_confident` is dropped by `process()` — `EstimationPacket` has **no field** that distinguishes a fresh measurement from a 39-frame-old one. The comment *"Navigation must not rely on this value"* describes an obligation the packet gives Navigation no means to honour. **This is the highest-severity item for the `EstimationPacket` handoff**, because Ignacio's subsystem has no way to detect it.

**V-6 — dead filters and a mis-scaled confidence function in `geometry.py`.** Proven by inspection of the shipped defaults:

| Filter | Default | Status |
| --- | --- | --- |
| `min_area` | `0.0` | `area < 0.0` — `contourArea` ≥ 0. **Never fires.** |
| `max_roi_span` | `1.0` | `(w / roi_w) > 1.0` — `w ≤ roi_w` by construction of `boundingRect`. **Never fires.** |
| `max_aspect` | `10.0` | Documented as *"upper bound; rejects extreme aspect ratios."* **No rejection branch exists.** Used only as an `elong_score` normaliser. |

Because `max_area = 300` while `ref_area = 2000`, `area_score ≤ 0.15`, so its 0.30 weight contributes at most **0.045**. Max achievable lane confidence is **0.745**, not 1.0. Confidence is effectively `0.5 × elongation + 0.2 × proximity-to-ROI-bottom` — **a pure shape-and-nearness score with no lateral or lane-likeness term at all.** Scored against the shipped filter:

| Contour | Confidence | vs 0.30 gate |
| --- | ---: | --- |
| **Wall/baseboard seam near ROI bottom** (elong 25) | **0.7068** | PASS |
| Wall seam mid-ROI (elong 12) | 0.6068 | PASS |
| **True lane dash near bottom** (elong 9) | **0.4603** | PASS |
| True lane dash, far (elong 14) | 0.5691 | PASS |

**A wall seam outscores a real lane marking by 54 %.** It must, by construction: the confidence function rewards long unbroken elongation and proximity to the camera, and a continuous floor/wall seam maximises both while a *dashed* lane marking is short and intermittent. Since `feature_fusion` sorts by confidence DESC, the wall is placed **first** in the detection list — exactly the position that wins the `_CentroidTracker` seed on an empty tracker.

## 3.2 Anomaly 2 — Wall-side starting deviation

**Your stated premise is the bug.** The lane ROI does not crop the wall out, because **the lane ROI does not crop horizontally at all**:

```python
lane_x = 0;  lane_w = W          # roi_crop.py:147-149
```

`lane_roi` is `frame[H//2:H, 0:W]` — the **full 480 px width**, lower half only. `traffic_roi` (`W//4`) and `sign_roi` (`W//2`) crop horizontally; the lane ROI does not. Narrow FOV does not help: whatever is beside the robot at floor level lands in the *bottom corner* of the lane ROI, which is the region `proximity_score` rewards most.

So it is not that clipping is mis-ordered or boundaries are mishandled. **There is no lateral clipping to bypass.** Four mechanisms then compound:

1. **Detection** (V-6) — the wall seam scores 0.71, above the 0.30 gate, and above every true lane marking.
2. **Anchor capture** (V-3) — `compute_lane_offset` picks `lanes_by_x[0]`; the wall is the leftmost x in the frame, so it becomes the left anchor unconditionally.
3. **The width gate is inverted in effect.** `min_lane_width_px = 150` was meant to reject implausible pairs. A genuine narrow pair (lines at 300 and 430 → 130 px) is **rejected** into the inverted-sign `width_rejected` branch, while a wall-plus-lane pair (15 and 430 → 415 px) is **accepted** as `two_boundary`. The gate filters out true lanes and passes contaminated ones.
4. **Seeding** — at `t=0` both defences are open by construction: `_CentroidTracker.last_x is None` ⇒ **the first detection is accepted unconditionally**, and `_EMAState.value is None` ⇒ **the first sample is adopted at 100 % weight with no smoothing**. The highest-confidence detection is the wall. It becomes the EMA seed.

**Replay — lane lines at x=300/430 (robot 125 px left of lane centre, truth ≈ +0.182 m):**

| | Phase 2 `offset` | Phase 3 `lane_offset` |
| --- | ---: | ---: |
| without wall contour | −0.5208 (`width_rejected`, inverted branch) | +0.0875 m (52 % under-reported) |
| **with wall contour** | −0.0843 (`two_boundary`) | **−0.3281 m** |

**−0.3281 m out of a ±0.35 m full scale, in the wrong direction.** The robot is left of lane centre and receives a near-saturated "you are far right" command.

**And the actuation makes it a pivot, not a drift.** `RAMP_SECONDS = 0.75` ramps `ramped_base` — but **`correction` is not ramped**:

```python
ramped_base = BASE_SPEED * ramp_frac
correction  = (error * KP) + (derivative * KD)
_drive(ramped_base - correction, ramped_base + correction)
```

While `ramp_frac ≈ 0`, the command is `(−correction, +correction)` — a **pure differential with zero common mode: a spin in place.** Replayed first four frames with the wall present:

```
f  ramp   P3 off    corr    L cmd    R cmd
0  0.00  -0.3281  -0.148   +0.148   -0.148   <- WHEEL REVERSES
1  0.07  -0.3281  -0.131   +0.164   -0.098   <- WHEEL REVERSES
2  0.15  -0.3281  -0.131   +0.197   -0.065   <- WHEEL REVERSES
3  0.22  -0.3281  -0.131   +0.230   -0.032   <- WHEEL REVERSES
```

Four consecutive frames of counter-rotating wheels. **Even without a wall, frames 0–1 still reverse a wheel** — the derivative kick from `_last_error = 0.0` at `t=0` guarantees `|correction| > ramped_base` on the first frame of every segment. The soft-start intended to limit inrush current instead guarantees a pivot at every start and every stop-resume, and does so *at the moment the offset estimate is least trustworthy*.

## 3.3 Anomaly 1 — Post-intersection instability

Your description — *"otherwise performs well at tracking the midpoint between leftmost and rightmost detections"* — describes `compute_lane_offset`. That function is not in the control path. What the robot actually tracks is the mean of whatever survives V-4's gate. On clean, symmetric, well-lit lane segments those two happen to agree closely enough that the loop closes (`corr(offset, Δoffset) = −0.271`). **At an intersection the symmetry assumption breaks, and with it the accidental agreement.**

Failure sequence, corroborated by both replay and the log:

1. **Approach.** Markings thin out; detections become asymmetric. The mean-of-survivors drifts toward the side that still fragments. Note the pre-gap values in the log: `+0.2559` and `+0.1611` — near full scale, on approach.
2. **Crossing.** Vision drops to `mode=none`. `_filter_lane` holds `_last_lane_offset_m` — the *already-biased* approach value. Because of V-5, it holds indefinitely and the packet says nothing.
3. **Reacquisition.** The first frame back is the worst possible input to the tracker: `_lane_tracker.last_x` still holds a centroid from *before* the intersection (nothing resets it), so the reacquisition frame is compared against a position tens of frames stale and is frequently **rejected**, extending the hold.
4. **EMA reseed.** When a sample finally passes, `_lane_offset_ema.value` is still the pre-intersection number. At `α = 0.35` the first post-gap output is 65 % stale, and convergence takes ~5 frames — during which the PD derivative sees every step.

**Log evidence — the 29-frame gap at f=2268:**

```
pre : +0.1066 +0.1066 +0.1066
gap : +0.1066 × 29 frames  (~1.5 s of frozen steering)
post: +0.1713/right_only  +0.2045/right_only  +0.2275/width_rej  +0.2275/none
      +0.2323/two_bnd     +0.2485/two_bnd     +0.1996/two_bnd    +0.1451/two_bnd
```

The offset does not return to truth — it **ramps monotonically to +0.2485 m (71 % of full scale)** over five frames, then decays. That is the EMA climbing out of a stale seed, not a measurement.

**The f=432 gap shows the oscillation directly:**

```
post: +0.1002/left_only  +0.1002/left_only  -0.0102/width_rej  +0.0752/two_bnd
      -0.0059/two_bnd    -0.0421/two_bnd    -0.0272/two_bnd    -0.1250/two_bnd
```

Two frames where Phase 2 found a lane and Phase 3 still held (V-4 rejection), then **five sign reversals in eight frames**, with a peak |Δoffset| of **0.1104 m against a run-wide mean of 0.0357 m — 3.1×**. And each of the first three post-gap frames uses a *different* `lane_offset.py` branch (`left_only`, `width_rejected`, `two_boundary`), which in the unused estimator would have meant **two sign-convention flips inside three frames** (S-2). That is the shape of the instability you are feeling.

**Full-run statistics:** 32 lane-loss events; 193 hold runs; longest hold 39 frames (~2.1 s); mean 3.35.

**One aggravating factor is not a seam bug:** 324 frames (11.2 %) exceeded the 66.7 ms budget (p95 = 71.7 ms, max 358.5 ms). Since `dt` is derived from wall-clock timestamps, over-budget frames stretch `dt`, which perturbs IMU heading integration during exactly the intervals when vision is unavailable. It compounds Anomaly 1; it does not cause it.

---

# Recommendations

Ordered by dependency. **R1–R3 are prerequisites for any meaningful PD or `ema_alpha` calibration** — tuning gains against the current `_filter_lane` is fitting to an artefact.

### R1 — Make `compute_lane_offset` the single source of truth *(fixes V-1, V-2, V-3; primary fix for both anomalies)*

Delete the reduction from `_filter_lane`. Pass the already-correct Phase 2 result across the seam. Extend the Phase 3 contract with one pre-reduced lane detection:

```python
# run_pipeline.py — replace the lane branch of _adapt_detections_for_p3
if lane_offset_result.mode not in ("none",):
    p3_detections.append(P3DetectionObject(
        type="lane_center",                       # NEW type: pre-reduced, singular
        label=lane_offset_result.mode,            # carries the mode contract (V-2)
        position_x=lane_offset_result.offset * (FRAME_WIDTH / 2.0),  # normalised -> signed px
        position_y=lane_roi_h / 2.0,
        confidence=lane_offset_result.confidence,
        timestamp=p2_out.timestamp_ms,
    ))
```

`_filter_lane` then consumes exactly one `lane_center` detection and drops the `avg_px` line entirely. Individual `lane_boundary` detections may still be forwarded for diagnostics but must no longer feed the offset.

**Before this lands, fix S-2** — normalise all four branches of `compute_lane_offset` to the `two_boundary` polarity, since R1 makes those branches load-bearing for the first time:

```python
offset = (left_x  - frame_center) / frame_center   # left_only
offset = (right_x - frame_center) / frame_center   # right_only
offset = (mid_x   - frame_center) / frame_center   # width_rejected
```

Then verify empirically on the bench that `corr(offset, Δoffset)` stays negative before driving (S-4).

### R2 — Rewrite the motion-consistency gate *(fixes V-4)*

With R1 there is one lane detection per frame, so the single-slot tracker becomes *correct by construction* for lanes — the comparand is finally the same entity across frames. Two further repairs are still required:

```python
def update(self, x, y, threshold) -> bool:
    if self.last_x is None or self.last_y is None:
        self.last_x, self.last_y = x, y
        return True
    dist = ((x - self.last_x)**2 + (y - self.last_y)**2) ** 0.5
    if dist <= threshold:
        self.last_x, self.last_y = x, y      # only accepted samples update state
        return True
    return False                              # do NOT poison the reference

def reset(self) -> None:
    self.last_x = self.last_y = None
```

Call `reset()` on every tracker after `deadreck_frames` exceeds a small threshold (2–3), so reacquisition after a gap is never judged against a pre-gap position. Also fix U-2 (`speed_scale` should be `1.0 + wheel_speed * dt * px_per_meter / max_centroid_jump_px`, or simply drop the term until encoders exist).

### R3 — Add validity to `EstimationPacket` *(fixes V-5; highest priority for the Ignacio handoff)*

```python
@dataclass
class EstimationPacket:
    ...
    lane_offset_valid:  bool   # False => dead-reckoned or stale
    lane_offset_age:    int    # frames since last confident vision fix
    lane_mode:          str    # "two_boundary" | "left_only" | ... | "none"
```

Make the two `_filter_lane` terminal branches actually differ: past `deadreck_max_frames`, decay toward 0.0 or flag hard-stale. In the motor loop, when `not lane_offset_valid`, hold the last correction or decay it — do not keep applying a PD correction against a frozen error, which is what produces the post-intersection wind-up. This is the field whose absence makes the failure *silent*; without it Ignacio's subsystem cannot detect the condition at all.

### R4 — Fix the soft-start pivot *(fixes Anomaly 2's actuation)*

Ramp the correction with the base, and zero the derivative state at every segment start:

```python
if nav_packet.drive_state == "stop":
    _drive(0.0, 0.0)
    ramp_start_ts = None
    _last_error = 0.0                       # T-5: clear stale derivative
else:
    if ramp_start_ts is None:
        ramp_start_ts = t_frame_start
        _last_error = nav_packet.lane_offset   # seed: no derivative kick on frame 0
    ramp_frac   = min(1.0, (t_frame_start - ramp_start_ts) / RAMP_SECONDS)
    ramped_base = BASE_SPEED * ramp_frac
    error       = nav_packet.lane_offset + OFFSET_TRIM
    correction  = ((error * KP) + ((error - _last_error) * KD)) * ramp_frac   # ramp it too
    correction  = max(-ramped_base, min(ramped_base, correction))             # never reverse a wheel
    _drive(ramped_base - correction, ramped_base + correction)
    _last_error = error
```

The clamp alone eliminates the counter-rotation; the `* ramp_frac` makes the ramp mean what its comment says.

### R5 — Narrow the lane ROI laterally *(fixes Anomaly 2's detection path)*

The ROI cannot crop the wall out because it spans the full width. Inset it:

```python
LANE_X_INSET = 0.10                        # TODO-CALIBRATE against trackT3/T4/T5
lane_x = int(W * LANE_X_INSET)
lane_w = W - 2 * lane_x
```

**This makes `lane_x != 0`, which activates C-1** — ROI-local x is no longer accidentally frame-correct. Land R5 and R7 together, and re-derive `lane_center_px` in the adapter from `source_shape`, not `lane_roi.shape[1]`.

A trapezoidal mask (already on your fix-toggle list as Fix 2) is the stronger version and subsumes this. Either way, add a lateral-position prior to `_lane_confidence` so the score stops being pure shape-and-nearness — that is the defect (V-6) that lets a wall outscore a lane marking, and no ROI inset fully compensates for it.

### R6 — Repair the colour-space contract, atomically *(fixes T-9)*

Feed `geometry.py` the same colour space its `_to_grayscale` assumes, or change `_to_grayscale` to `cv2.COLOR_BGR2GRAY` and delete the intermediate conversion — which also saves one full-frame `cvtColor` per branch per frame, relevant to the 11.2 % over-budget rate. **Re-calibrate Canny `(10, 160)` and `min_intensity = 80` in the same commit**; they are tuned against the scrambled image and will not transfer.

### R7 — Implement the documented re-projection *(fixes C-1)*

`fuse_detections` already receives `SourceROIInfo`. Extend it to carry `lane_rect`/`traffic_rect`/`sign_rect` (already computed, currently unused) and add the offsets in `_centroid`. Do this **before** un-gating the sign and traffic branches for D2, or their 240 px / 120 px x-errors will present as detection failures.

### R8 — Make the seam mechanically checkable *(fixes T-1…T-8)*

- Parameterise return types: `-> tuple[GeometryBranchResult, dict, dict]`, `-> list[P3DetectionObject]`.
- Move the two mirrored dataclasses into a shared `contracts.py` imported by both phases, so drift becomes an import-time failure instead of a silent one. This is the structural fix for the detached-subtree problem in §1.1, and directly addresses the transcription-drift risk in your workflow.
- Switch `timestamp_ms` to `time.perf_counter()` (T-4).
- Move `unzip_data` imports inside the `__main__` blocks (T-6).
- Encapsulate `_last_error` in a small `PDController` class (T-5).
- Fix the three docstring sign contradictions (S-1, S-3) and add a comment at `_drive` recording that the two right-channel inversions are deliberate and cancel (S-4).

### R9 — Add one regression test at the seam

Boundary-based, matching how you already validate:

```python
def test_centred_lane_yields_zero_offset():
    """Symmetric boundaries at ±120px, 10 frames. Must return ~0.0 every frame."""
    # Today this returns -0.1750 -> -0.0525 -> -0.0954 ... and never settles.
```

It fails today, passes after R1+R2, and would have caught this before it reached the track. Add a second asserting that `lane_offset_valid` goes `False` within one frame of vision loss.

---

## Priority summary

| Rank | Item | Fixes | Blocks |
| --- | --- | --- | --- |
| 1 | **R1** — single source of truth for lane offset | V-1, V-2, V-3; both anomalies | all PD / `ema_alpha` calibration |
| 2 | **R3** — validity + age in `EstimationPacket` | V-5; Anomaly 1 | Ignacio's integration |
| 3 | **R2** — motion-consistency rewrite | V-4; 523 silent-hold frames | Anomaly 1 recovery |
| 4 | **R4** — soft-start pivot | Anomaly 2 actuation | safe bench starts |
| 5 | **R5** — lane ROI lateral inset (+ R7) | Anomaly 2 detection | must land with R7 |
| 6 | **R6** — colour-space + re-calibration | T-9 | all `TODO-CALIBRATE` work |
| 7 | **R7/R8/R9** — re-projection, typing, regression test | C-1, T-1…T-8 | D2 sign/traffic un-gating |

**Do not re-tune `OFFSET_TRIM`, `KP`, `KD`, or `ema_alpha` until R1–R3 are in.** Every gain currently on the robot was fitted against the mean-of-survivors artefact and the 22.4 % hold rate; those values will not survive the fix, and re-tuning first means doing it twice.
