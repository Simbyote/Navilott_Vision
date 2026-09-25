# Phase 2: Vision Perception

> One frame to structured detections and a steering error.

Phase 2 takes one `FrameData` and answers, for that frame alone: where are the lane boundaries, how far is the robot from its lane center, is there a stop sign, and is there a lit traffic light. It has no memory between frames. Anything that needs history or sensors belongs in Phase 3.

**Code:** `src/perception/` · **Order:** `src/phase2_linker.py` · **Tests:** `src/tests/test_<stage>.py`

---

## Design philosophy

**Classical CV, on purpose.** The course is structured: white tape on dark mats, one octagon shape, three lamp colors. Thresholds, edges and contour geometry handle that on a Pi Zero 2 W with predictable latency and low memory. Every decision can be traced to a number that can be tuned, which is the point: most failures are tuning problems, and a tunable pipeline can be fixed on the course.

**Each stage is a function from one frozen result to the next.** A stage takes the previous stage's result dataclass and returns its own. It can't modify its input or reach into another stage's state. That keeps each stage testable alone with hand-built input.

**The stage order is written in one place.** `phase2_linker.run_chain()` is the only code that calls the stages in sequence. `live_view`, `phase3_linker` and the tests all go through it, so no caller can run a different order.

**Frame identity travels with the data.** Every result carries the `frame_id` and `timestamp_ms` capture assigned. Stages that combine inputs check the stamps match and raise if they don't.

**Tuning lives in config, not code.** Each stage has a config dataclass, and `PipelineConfig` bundles them. Re-tuning after a camera change is a config swap and a re-run. `MEASURED` is the tuning currently used on the robot.

**Results and debug are separate.** Each stage returns its result and a debug dict. The result is the contract; the debug dict (edge maps, reject counts, logs, traces) is for inspection and nothing downstream reads it. Overlay drawing and per-contour traces are off in the live loop and switched on by the debug tools.

**Every rejection is counted.** Each gate in the geometry and color branches has a reject counter, and every contour lands in exactly one bucket. "Why did this frame go blind" is answered from the debug dict without a rerun.

---

## Stage order

```
FrameData
   │
   ▼
preprocess_frame()        blurred gray + blurred BGR
   │
   ▼
crop_rois()               lane + sign ROIs (gray), traffic ROI (BGR), their rects
   │
   ├────────────────────────────┐
   ▼                            ▼
run_geometry_stage()        run_color_stage()
lane + sign candidates      traffic light candidates
   │                            │
   ├──────────────────┐         │
   ▼                  ▼         ▼
compute_lane_offset() fuse_detections()      (both also read the ROI crop)
   │                  │
   └────────┬─────────┘
            ▼
     package_phase2()  →  Phase2Output  →  Phase 3
```

Lane offset and fusion are siblings. Lane offset reads the geometry candidates directly, not fusion's output, because fusion keeps only a centroid and a confidence, and the offset needs each contour's foot position, width, length and brightness.

`run_chain()` runs them in this order: preprocess, roi, geometry, color, lane_offset, fusion, package. Each one's wall time goes in `ChainResult.timings_ms` under those names.

---

## Stage 1: Preprocessing

**File:** `preprocess.py` · **Config:** `PreprocessParams` · **Output:** `PreprocessResult`

Conditions the frame before it splits into branches.

```
BGR frame
  │
  ├── undistort()                    only if calibration_path is set
  │
  ├── gray path:  to_grayscale → [equalize] → GaussianBlur (9, 3)   → lane + sign ROIs
  └── color path:                             GaussianBlur (5, 5)   → traffic ROI
```

| Setting | Default | Why |
| --- | --- | --- |
| `gray_kernel` | `(9, 3)` | Wider than tall: smooths more across a lane line than along it, so broken fragments join without the line thinning |
| `color_kernel` | `(5, 5)` | Plain noise suppression before HSV thresholding |
| `equalize` | `False` | Histogram equalization raised sensor noise enough to cost more in false contours than it gained in contrast. Turning it on invalidates the `min_intensity` gates downstream |
| `calibration_path` | `None` | Undistortion is off. See below |
| `undistort_alpha` | `0.0` | If undistortion is on, crop to valid pixels. Higher keeps more field of view but adds a curved black border the geometry branch picks up as an edge |

**Gray conversion happens here and only here.** The geometry branch rejects 3-channel input, so no later stage can quietly convert on its own.

**Undistortion runs before the split** so both branches see the same corrected geometry. The remap tables are built once per calibration and cached. A calibration whose image size doesn't match the frame raises. `PreprocessResult.undistorted` is the unblurred frame that detection coordinates refer to; debug overlays are drawn on it.

Whether undistortion runs on the robot is still open. It was removed to save preprocessing time on the IMX219, whose distortion was mild. The IMX290's M12 lens bends the edges more, so it needs retesting: `pytest --hardware src/tests/test_calibration.py` reports the per-frame cost (`stage_ms`) and how much straighter the boards get.

---

## Stage 2: ROI cropping

**File:** `roi_crop.py` · **Config:** `ROIConfig` · **Output:** `ROICropResult`

Cuts three regions so each branch only processes the part of the frame where its target can appear. Less area means less time and fewer false positives before any threshold runs.

| ROI | Bounds (fraction of frame) | At 480×270 (x, y, w, h) | Source | Why there |
| --- | --- | --- | --- | --- |
| lane | x 0.05–0.95, y 0.70–1.00 | (24, 189, 432, 81) | gray | Near-field road just ahead of the robot |
| traffic | x 0.25–0.75, y 0.00–0.50 | (120, 0, 240, 135) | BGR | Where a light sits when the robot is square to an intersection |
| sign | x 0.50–1.00, y 0.00–0.55 | (240, 0, 240, 148) | gray | Signs are posted right of the lane |

```
x:  0       120      240      360      480
    ┌────────┬────────┬────────┬────────┐  y 0
    │        │traffic │traffic │        │
    │        │        │ + sign │  sign  │
    │        ├────────┴────────┤        │  y 135
    │        │                 └────────┤  y 148
    │                                   │
    ├──┬─────────────────────────────┬──┤  y 189
    │  │            lane             │  │
    └──┴─────────────────────────────┴──┘  y 270
```

Traffic and sign overlap in x 240–360, y 0–135.

- ROIs are read-only views, not copies. A stage that needs to draw on one calls `.copy()`.
- Every rect is in frame pixels. Branch detections are ROI-local, and the rect is what maps them back: `frame_x = roi_x + rect[0]`.
- Bounds are validated on construction: each in [0, 1], and x0 < x1, y0 < y1.

---

## Stage 3A: Geometry branch

**File:** `geometry.py` · **Config:** `GeometryConfig` (`CannyParams`, `LaneContourFilter`, `SignContourFilter`) · **Output:** `GeometryBranchResult`

Finds lane boundaries and stop-sign shapes from intensity edges. Both use Canny and external contours, on separate ROIs.

### Lane boundaries

```
lane ROI → Canny (80, 200) → close (9×3) → contours → gates → confidence → merge fragments
```

White tape on a dark mat gives strong edges. The morphological close bridges gaps along a line so a fragmented line traces as one contour.

Gates, applied in order. Each rejection goes to its own counter:

| Gate | Default | Rejects |
| --- | --- | --- |
| `area` | 1–1000 px² | Specks and large regions |
| `degenerate`, `too_few_pts` | w or h = 0; < 5 points | Contours `minAreaRect` can't measure |
| `aspect` | elongation ≤ 60 | Streaks too thin to be tape |
| `w_span` / `h_span` | ≤ 1.0 of the ROI | Contours spanning more than the ROI along their own axis |
| `intensity` | mean ≥ 120 | Dark blobs such as mat seams |

Confidence, in [0, 1]: 50% length (against a quarter of the ROI extent), 30% brightness above `min_intensity`, 20% thickness (against 30 px). It's a measurement-quality weight, not a probability.

Horizontal fragments of the same line, with endpoints within 40 px, are merged into one candidate. Without that, lane offset could pick two pieces of one line as opposite boundaries.

Each candidate records `foot_x`: the mean x of the contour's lowest 6 rows (`FOOT_BAND_PX`). That's where the marking is closest to the robot, which is what steering should use. For an angled line, the bbox center sits halfway up the ROI instead.

### Stop sign

```
sign ROI → Canny → contours → area → approxPolyDP → vertex count → convex hull → solidity
```

| Gate | Default | Rejects |
| --- | --- | --- |
| `area` | 200–30000 px² | Edge noise and background |
| `vertices` | 8–10 after `approxPolyDP` (ε = 0.03 × perimeter) | Shapes that aren't roughly octagonal |
| `hull` | hull area > 0 | Degenerate outlines |
| `solidity` | area / hull ≥ 0.80 | Fragmented or concave outlines |

Confidence: half closeness to 8 vertices, half area up to 5000 px².

With `trace=True`, every contour that reached a gate is recorded with the gate that decided it. `debug_stop` draws these.

---

## Stage 3B: Color branch

**File:** `color_branch.py` · **Config:** `ColorConfig` · **Output:** `list[TrafficLightCandidate]`

```
traffic ROI (BGR) → HSV → red / yellow / green masks → contours → area + aspect gates → confidence
```

- **It's off until calibrated.** With `ColorConfig.hsv_ranges = None` (the default), the stage returns no candidates and never reads the ROI. Pass calibrated ranges from `calibration/hsv_ranges.json` with `--hsv` or `load_color_config()` to switch it on.
- **Red uses two bands** because hue wraps around: 0–10 and 170–180 in OpenCV units (degrees / 2), combined with OR.
- **The built-in ranges are a scaffold,** not a calibration. `HSVRanges.is_calibrated` is only true for ranges loaded from JSON, and the debug dict reports it.
- **Blob gates** (area 50–5000 px², w/h aspect 0.3–3.0) are placeholders, not tuned.
- **Confidence is area only,** saturating at 800 px², the expected lamp size at detection range. Fusion keeps the highest confidence across all three colors, so the largest blob wins.

The HSV ranges have to be tuned under course lighting. Ranges from a lab or office won't carry over.

---

## Stage 4: Lane offset

**File:** `lane_offset.py` · **Config:** `LaneOffsetConfig` · **Output:** `LaneOffsetResult`

Turns lane candidates into one steering error.

### A second, stricter gate

Geometry decides "is this a lane marking". Lane offset decides "is it trustworthy enough to steer by". A candidate can pass the first and fail the second.

| Gate | Default | `MEASURED` | Rejects |
| --- | --- | --- | --- |
| `conf_threshold` | 0.30 | 0.25 | Weak candidates |
| `min_proximity` | 0.25 | 0.05 | Markings too far up the ROI to describe where the robot is now |
| `min_length_px` | 25 | 25 | Dashed-center-line fragments |
| `width_px` range | 1–25 | 1–45 | Noise below; blobs, glare and merged pairs above |
| `min_intensity` | 90 | 130 | Shadow edges and seams |

`MEASURED` comes from a sweep of 4,827 recorded candidates rather than from course dimensions. The default `min_proximity` sat above the observed median of 0.17 and threw out 1,752 candidates geometry had already scored at 0.30 or higher. The default width cap clipped detections whose 90th percentile was 27 px.

### Picking the lane

Each usable candidate becomes an anchor at its `foot_x`, weighted by confidence × (0.5 + 0.5 × proximity). A strong marking far up the ROI is real, but it says less about where the robot is right now.

The robot sits at the center of the lane ROI. Its lane is bounded by the **nearest** anchor on each side, not the outermost pair. On a two-lane street, the outermost pair would center the robot on the whole street.

1. Anchors on both sides of center: take the nearest on each side.
2. All anchors on one side (the robot has drifted out): take the two nearest center. They still describe the nearest lane, and the offset steers back.
3. One anchor: single-boundary mode.

Then the pair's spacing is checked. Closer than 60 px means one marking seen twice; wider than 400 px means the pair spans more than one lane. Either one falls back to the strongest single anchor.

### The offset

```
lane_center = (left_x + right_x) / 2
offset      = clamp((roi_center − lane_center) / roi_center, −1, +1)
```

**Sign: + means the robot is right of lane center, so steer left.** The lane center appears left of the image center when the robot has drifted right. Older docs said the opposite; the code and this doc are what count.

The offset is normalized by **half the lane ROI width** (216 px at 480×270), not half the lane width. It's a steering error with a fixed scale, which is what the controller needs.

### Modes

| Mode | When | Offset |
| --- | --- | --- |
| `two_boundary` | Two anchors with plausible spacing | Midpoint formula |
| `left_only` / `right_only` | One usable boundary, `expected_half_lane_px` set | Lane center projected from the one boundary |
| `single_uncalibrated` | One usable boundary, `expected_half_lane_px` is `None` | 0.0, confidence 0 |
| `none` | No usable boundaries | 0.0, confidence 0 |

Single-boundary mode projects the lane center `expected_half_lane_px` from the visible line. Without that scale, distance-from-a-line and distance-from-center would feed the same controller on different scales, so the mode reports `single_uncalibrated` instead. The current value (228 px) is hand-set, not calibrated.

`boundary_count` counts usable boundaries, not raw detections, so "saw nothing" and "saw nothing it could steer by" are different.

---

## Stage 5: Feature fusion

**File:** `feature_fusion.py` · **Output:** `FusionResult` of `DetectionObject`

Converts every branch's candidates into one schema and resolves conflicts within each class.

| Class | Rule |
| --- | --- |
| `traffic_light` | Highest confidence wins; the rest are logged as suppressed |
| `lane_boundary` | All valid candidates kept, sorted by descending confidence |
| `stop_sign` | Highest confidence wins; the rest are logged as suppressed |

Any confidence outside [0, 1] (including NaN) is discarded and logged. Output order is fixed: traffic light, lane boundaries, stop sign.

`DetectionObject`:

| Field | Meaning |
| --- | --- |
| `type` | `traffic_light` / `lane_boundary` / `stop_sign` |
| `label_detail` | Branch label: light color, or the type again |
| `confidence` | [0, 1] |
| `position` | `{"x", "y"}` bbox centroid, **ROI-local** |
| `bounding_box` | (x, y, w, h), ROI-local; used by the overlay |
| `source_roi`, `source_rect` | Which ROI, and its rect in frame px |
| `frame_id`, `timestamp` | The frame's stamp |

Positions are ROI-local: a lane boundary at (200, 50) and a stop sign at (200, 50) aren't the same place. Add `source_rect[0]`, `source_rect[1]` for frame coordinates. The rect travels with each detection so it stays self-describing in Phase 3, where the ROI crop is out of scope.

Fusion doesn't import the color branch, so lane and sign fusion work with the color branch off.

---

## Stage 6: Packaging

**File:** `phase2_out.py` · **Output:** `Phase2Output`

No computation. Collects the detections and lane offset, checks that they belong to one frame, and packages them as the Phase 3 contract.

```
Phase2Output
    detections           list[DetectionObject]   in fusion order
    lane_offset_results  list[LaneOffsetResult]  one per frame; [] if none
    frame_id             int
    timestamp_ms         int
    detection_count      int                     always len(detections)
```

Checked on construction, so a malformed handoff fails loudly instead of reaching navigation:

- Both lists must be lists; pass `[]` for none
- Every item has its required fields
- Every item's stamp matches the container's
- `frame_id` and `timestamp_ms` have no default, since a silent 0 would make every frame look like the same instant

**Phase 3 steers from `lane_offset_results`, not from lane-boundary positions.** Those are bbox centroids, which for an angled marking aren't where it meets the robot.

---

## Timing

`ChainResult.timings_ms` holds each stage's wall time. Measure on the Pi with the current camera:

```
pytest --hardware --replay=src/tests/data/frames
```

Each stage's test writes a timing CSV and histogram. The old tables (24–48 ms for the chain) were measured on the IMX219 at 480×360 with equalization on, and no longer apply. The target is the full chain plus Phase 3 inside one frame period: 50 ms at 20 FPS.

---

## Seeing inside it

```
python3 -m src.phase2_linker --camera --views stop,traffic
python3 -m src.phase2_linker --video run.avi --no-display
```

Runs the chain with the debug overlay. Accepts `--camera`, `--video PATH` or `--frames DIR`, and records each view as a video plus CSV under `runs/<timestamp>/`.

| View | Shows |
| --- | --- |
| lane (always on, `debug_lane`) | Every raw candidate (green usable, red with the gate that rejected it), each anchor's foot, the chosen left and right boundaries, robot and lane center, mode and offset gauge |
| `stop` (`debug_stop`) | Sign contours colored by the gate that decided them, with vertex count and confidence |
| `traffic` (`debug_traffic`) | HSV masks and blobs against the bands |

`pytest --hardware -k debug_lane` saves every stage's images for 3 sample frames plus an annotated video, which is the place to start when tuning a detector by eye.

---

## Open items

- **Undistortion** on the IMX290: cost against improvement, measured with the calibration test.
- **HSV calibration** under course lighting, then switch the color branch on.
- **Stop-sign geometry** tuned on real frames. The sign reaches Phase 3's vote now, gated only at 0.45 confidence there.
- **`expected_half_lane_px`** from calibration instead of the hand-set 228 px.
- **The intersection failure:** in 3 of 3 runs the robot drifted right and failed at an intersection. Record those spots and check the lane pair choice and modes there.
- **Offset sign on hardware:** confirm + = robot right of center on the propped-up chassis before tuning anything downstream.
