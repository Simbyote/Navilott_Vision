# Phase 2: Vision Perception

> One frame to structured detections and a steering error.

Phase 2 takes one `FrameData` and answers, for that frame alone: where are the lane boundaries, how far is the robot from its lane center, is there a stop sign, and is there a lit traffic light. It has no memory between frames. Anything that needs history or sensors belongs in Phase 3.

**Code:** `src/perception/` · **Order:** `src/phase2_linker.py` · **Tests:** `src/tests/test_<stage>.py`

---

## Design philosophy

**Classical CV, on purpose.** The course is structured: white tape on dark mats, one octagon shape, three lamp colors. Thresholds, edges and contour geometry handle that on a Pi Zero 2 W with predictable latency and low memory. Every decision can be traced to a number that can be tuned, which is the point: most failures are tuning problems, and a tunable pipeline can be fixed on the course. What a learned model would have cost and what this gives up are in `../concepts.md` ("Classical computer vision"), with the operations used below: HSV, Otsu, Canny, morphological close, contours and the gates.

**Each stage is a function from one frozen result to the next.** A stage takes the previous stage's result dataclass and returns its own. It can't modify its input or reach into another stage's state. That keeps each stage testable alone with hand-built input.

**The stage order is written in one place.** `phase2_linker.run_chain()` is the only code that calls the stages in sequence. `live_view`, `phase3_linker` and the tests all go through it, so no caller can run a different order.

**Frame identity travels with the data.** Every result carries the `frame_id` and `timestamp_ms` capture assigned. Stages that combine inputs check the stamps match and raise if they don't.

**Tuning lives in config, not code.** Each stage has a config dataclass, and `PipelineConfig` bundles them. Re-tuning after a camera change is a config swap and a re-run. `PipelineConfig` and `MEASURED`, the tuning used on the robot, live in `src/config.py`; the pipeline, both linkers and the tests import them from there. `MEASURED` undistorts with `calibration/camera_calibration.json` and runs the color branch with `calibration/hsv_ranges.json`. Tests that feed synthetic frames use `SCENE_CONFIG` (`src/tests/scenes.py`): `MEASURED` with undistortion off, since synthetic frames are drawn already undistorted.

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
crop_rois()               lane + sign ROIs (gray), traffic + sign ROIs (BGR), obstacle ROI (both), their rects
   │
   ├────────────────────────────┐
   ▼                            ▼
run_geometry_stage()        run_color_stage()
lane + stop-line +          traffic light candidates
sign candidates                 │
   │                            │
   ├──────────────────┬──────────────────────────┐   │
   ▼                  ▼                          ▼   ▼
compute_lane_offset() compute_stop_line_distance() fuse_detections()   (all read the ROI crop)
   │                  │                          │
   └──────────────────┴────────────┬─────────────┘
                                   ▼
                     package_phase2()  →  Phase2Output  →  Phase 3
```

compute_obstacle() runs beside them on the obstacle ROI alone (no geometry), between the stop line and fusion.

Lane offset, stop-line distance and fusion are siblings. Lane offset reads the geometry candidates directly, not fusion's output, because fusion keeps only a centroid and a confidence, and the offset needs each contour's foot position, width, length and brightness. Stop-line distance reads geometry's stop-line candidates the same way. Geometry only detects; the two measurement stages turn detections into numbers.

`run_chain()` runs them in this order: preprocess, roi, geometry, color, lane_offset, stop_line, obstacle, fusion, package. Each one's wall time goes in `ChainResult.timings_ms` under those names. The main pipeline (`src/pipeline.py`) runs the same order through the production versions.

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
  └── color path:                             GaussianBlur (5, 5)   → traffic ROI + sign_color_roi
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

Undistortion runs on the robot: `MEASURED` loads `calibration/camera_calibration.json` (`config.py`), since the IMX290's M12 lens bends the edges (k1 = −0.29). The lane gates were swept on distorted frames, and undistortion moves lane marks by up to ~20 px at the ROI edges, so they are due a re-sweep on undistorted captures. `pytest --hardware src/tests/test_calibration.py` reports the per-frame cost (`stage_ms`) and how much straighter the boards get.

---

## Stage 2: ROI cropping

**File:** `roi_crop.py` · **Config:** `ROIConfig` · **Output:** `ROICropResult`

Cuts four regions so each branch only processes the part of the frame where its target can appear. Less area means less time and fewer false positives before any threshold runs.

| ROI | Bounds (fraction of frame) | At 480×270 (x, y, w, h) | Source | Why there |
| --- | --- | --- | --- | --- |
| lane | x 0.05–0.95, y 0.70–1.00 | (24, 189, 432, 81) | gray | Near-field road just ahead of the robot |
| traffic | x 0.23–0.53, y 0.10–0.33 | (110, 27, 144, 62) | BGR | Around the course's light as seen from its stop line (2026-10-06; moved down on the track 2026-10-26); narrow, so a second light to the right and a red box below stay out |
| sign | x 0.55–1.00, y 0.00–0.85 | (264, 0, 216, 230) | gray, and BGR as `sign_color_roi` (the sign detector reads the BGR one) | Signs are posted right of the lane. Tall: near the line the sign rises in the frame, and a lower edge cut it into 5–7 vertices (sign_check, 2026-10-20). Right of the traffic ROI, so the two don't overlap |
| obstacle | x 0.25–0.75, y 0.45–1.00 | (120, 122, 240, 148) | gray and BGR (`obstacle_roi`, `obstacle_color_roi`) | The robot's own path, from the view bottom up past the lane ROI: an upright object rises above its base. Not drawn by `draw_roi_overlay`; the obstacle view draws it |

Traffic and sign sit side by side and don't overlap (x 110–254 and 264–480). The obstacle ROI overlaps the lane ROI (y 189–270) and the bottom of the sign ROI (x 264–360, y 122–230); each branch reads only its own crop, so the overlap costs nothing.

- ROIs are read-only views, not copies. A stage that needs to draw on one calls `.copy()`.
- Every rect is in frame pixels. Branch detections are ROI-local, and the rect is what maps them back: `frame_x = roi_x + rect[0]`.
- Bounds are validated on construction: each in [0, 1], and x0 < x1, y0 < y1.

---

## Stage 3A: Geometry branch

**Files:** `geometry.py` (runs the three detectors), `geometry_lanes.py`, `geometry_stop_line.py`, `geometry_sign.py`, `geometry_shared.py` (configs, candidate types, edge helpers) · **Config:** `GeometryConfig` (`CannyParams`, `LaneContourFilter`, `SignContourFilter`, `StopLineFilter`) · **Output:** `GeometryBranchResult`

Finds lane boundaries and stop lines from intensity edges, and stop-sign shapes from color. Canny runs once on the lane ROI; the lane and stop-line detectors both read that edge map. The sign detector doesn't use Canny: it thresholds a redness image of the color sign ROI.

### Lane boundaries

```
lane ROI → Canny (80, 200) → take lines across the lane out → close (9×3) → contours → gates → confidence → merge fragments
```

White tape on a dark mat gives strong edges. The morphological close bridges gaps along a line so a fragmented line traces as one contour.

**Lines across the lane come out first.** A stop line touching the lane lines would otherwise close into one H-shaped contour with them, too wide for any lane gate, and the lane would be lost for as long as the line is in view. So the lane detector's own copy of the edges loses every horizontal edge run at least `horizontal_min_run_px` (46 px) long within `horizontal_edge_deg` (20°) of horizontal, plus any horizontal edge within `horizontal_band_px` (3 px) of such a run's line (the stubs of a stop line running past the tape). The runs are longer than any tape is wide, so the ends of a piece of tape stay and it still traces as one shape. These are `LaneContourFilter` settings, the lane's own; the stop-line detector reads the full edge map with its own `StopLineFilter`. When the two angles match, the gradient split is computed once for both. The course has no curves the camera steers through, so no lane line lies that flat. `horizontal_edge_deg = None` turns it off.

Checked against the previous version on every test scene and gate-sweep frame under three configs (840 pairs): lane results changed only on frames with a stop line, from `none` or one-sided to `two_boundary`, plus one short line bent by undistortion that the stop-line detector misses. **Known limit:** on wide tape (20–30 px), while a stop line is in view, the anchors can land on the tape's edge instead of its middle, up to half a tape width: the piece of tape below the line has no top edge (tape meets tape there), so its two sides trace apart.

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

### Stop lines

```
lane ROI Canny (shared) → gradient split → top / bottom edge maps → close (15×1) → fitted segments
                        → join pieces broken by lane tape → pair top with bottom → gates → confidence
```

A stop line's edges run across the ROI, so their gradient points up or down; a lane line's sides point left or right. Sobel at each Canny edge pixel keeps only edges of lines within `max_tilt_deg` (20°) of horizontal. That separates a stop line from lane lines it touches before any contour joins them. The course has no curves the camera steers through, so nothing lying across the ROI is a lane line.

The kept edges split by polarity: a **top** edge (brightness rising going down, the upper edge of bright tape) and a **bottom** edge. Each is fitted as a line. Where lane tape crosses the stop line, the edge breaks for the tape's width; pieces at the same height with a gap up to `max_thickness_px` are refit as one. Each top edge is then paired with the nearest bottom edge below it that overlaps it.

| Gate | Default | Rejects |
| --- | --- | --- |
| `short` | length ≥ 60 px | Anything as short as a lane-tape width (up to ~41 px), such as the end of a dash |
| `tilt` | ≤ 20° (`MEASURED`: 5°) | Edges fitted steeper than the split allowed. `MEASURED` uses 5°: the near end of a thick diagonal lane line passed at 20° (2026-10-01 run). Tighter (2° from 2026-10-07, 4° from 2026-10-19) missed real lines approached a little off square; at 5°, parts of the intersection itself have passed for stop lines (2026-10-05), so watch `nav.csv`'s `step` column (`config.py`) |
| `unpaired` | a bottom edge 3–40 px below, overlapping half the shorter edge | Single edges, and bars thicker than tape |
| `intensity` | mean between the edges ≥ 70 | Shadow edges and dim patches |

A top edge within `max_thickness_px` of the ROI bottom with no bottom edge is kept as **clipped**: the robot is on the line, its bottom edge below the ROI.

Each `StopLineCandidate` records its ends, the top and bottom rows at its middle, `y_near_px` (the lowest point of the bottom edge, the part the robot reaches first), signed tilt (+ = right end nearer), length, thickness and brightness. Confidence: 50% length (against 200 px, about a lane), 30% brightness above `min_intensity`, 20% squareness. Candidates come out nearest first.

**Stop lines never change the lane or sign results.** They only read the shared Canny map and have their own config. `test_geometry` checks lanes and signs are identical under any `StopLineFilter`, and the change was checked against a snapshot of every scene and gate-sweep frame under four configs taken before it (1096 of 1096 identical).

Cost on a laptop: about 0.5 ms per frame, most of it the Sobel pass and contour fitting. It classifies only edge pixels (a few hundred of the ROI's ~35 000).

### Stop sign

```
sign_color_roi → redness R − max(G, B) → Otsu (never below min_redness) → close (5×5)
              → external contours → drop < min_area → largest → convex hull
              → approxPolyDP (ε = 0.02 × hull perimeter) → max_area → vertex count → solidity
```

Why color: in gray a stop-sign red (BGR 40, 40, 200) is 88 and a gray floor about 80, so Canny sees almost no edge; on synthetic scenes the old gray-Canny detector found no sign on a gray floor at any aperture or threshold without burying it in noise. Redness is bright on red and near 0 on gray, black and white alike, whatever their brightness. `cv2.subtract` saturates at 0 where numpy's `-` would wrap a green pixel to a large value.

| Setting / gate | Default | Why |
| --- | --- | --- |
| `min_redness` | 20 | Otsu always splits the image, so a sign-free ROI thresholds its own noise (Otsu ≈ 1 there, with false blobs). The dimmest synthetic sign had Otsu at 22. Check on course frames |
| `close_kernel` | 5 | Fills noise pinholes; rejoins a sign cut by a line up to ~2 px wide before blur. Wider cuts split it, and each half fails the vertex gate |
| `min_area` | 100 px² | Noise blobs; the blobs over it are gated largest first, and the first to pass all the gates is the sign |
| `max_area` | 30000 px² | A red area bigger than any sign at range |
| `vertices` | 8–9 after `approxPolyDP` on the hull | Red shapes that aren't octagons. On the hull, so letters and noise notches don't add vertices |
| `hull` | hull area > 0 | Degenerate outlines |
| `solidity` | contour area / hull area ≥ 0.80 | Ragged or concave blobs (measured on the raw contour; the hull alone is always 1.0) |
| `aspect` | off; `MEASURED`: bounding box w / h 0.5–1.6 (0.6 until 2026-10-20) | Ghost signs: a leg's tall outline (~0.3). An octagon square-on is ~1; the course's sign reads 0.55–0.72 |
| `extent` | off; `MEASURED`: contour area / box area 0.6–0.92 (0.68 until 2026-10-20) | Ghost signs: a chair back's boxy outline (a rectangle is 1.0, an octagon 0.83; the course's sign reads 0.63–0.75) |
| `letters` | off; `MEASURED`: ≥ 3% of the hull white (each channel ≥ 150, redness ≤ 60), for contours ≥ 2000 px² | Ghost signs: any red blob without the sign's white letters. A far sign's letters blur into its red, so small ones aren't held to it |

The last three gates came from ghost signs on course runs: now and then a leg or parts of a chair in the background passed as a sign, and the robot stopped at a plain stop line. Their values are first guesses from synthetic signs, not yet checked against the course's own sign; navigation also ties a sign to the stop line it's for (`SIGN_LEAD_MS`, `navigation_contract.md`). At most one sign candidate comes out per frame: the first blob, largest first, that passes every gate. Confidence is unchanged: half closeness to 8 vertices, half area up to 5000 px².

The blobs are gated largest first until one passes: until 2026-10-20 only the largest was gated, so a larger red thing in the sign ROI (a box, a chair), refused, hid the sign behind it (`make sign-check` on the detect-range looks found it). With `trace=True`, every contour that reached a gate is recorded with the gate that decided it, and the smaller blobs never gated, a larger one having passed, as `not_largest`. `debug_stop` draws these beside the redness image, with the mask outlined and the threshold used.

---

## Stage 3B: Color branch

**File:** `color_branch.py` · **Config:** `ColorConfig` · **Output:** `list[TrafficLightCandidate]`

```
traffic ROI (BGR) → HSV → red / yellow / green masks → contours → area + aspect + roundness + clipped-core gates → confidence
```

- **It's off without ranges.** With `ColorConfig.hsv_ranges = None` (the `PipelineConfig` default), the stage returns no candidates and never reads the ROI. `MEASURED` loads `calibration/hsv_ranges.json` at import, tuned or not, so the robot and both linkers run with the branch on; `--hsv` swaps in other ranges.
- **Red uses two bands** because hue wraps around: 0–10 and 170–180 in OpenCV units (degrees / 2), combined with OR.
- **The built-in ranges are a scaffold,** not a calibration. `HSVRanges.is_calibrated` is only true for ranges loaded from JSON, and the debug dict reports it.
- **A lamp, not just a color.** Color and size can't tell a lit lamp from a shirt, a wall or a sign of the same color; the confidence score couldn't either, since it's area only. What can is that a lamp makes light: it's brighter than the camera can record, so its middle clips to near-white inside the colored ring (why every lamp measured had a "missing" center in its color mask), while a reflecting surface stays colored throughout. So a blob passes only with:
  - **a clipped core:** at least `min_core_px` (3) pixels inside its outline with V ≥ `core_min_v` (240) and S ≤ `core_max_s` (60). The outline is the outer contour, so the ring's hole is inside it; clipped pixels beside the blob, in its bounding box but outside the outline, don't count. The lamps' cores read V 254–255, S 4–5 on 2026-10-04; `calibrate_lamps` reports each lamp's core, and warns if a lamp doesn't clip;
  - **a round outline:** area over its enclosing circle's ≥ `min_roundness` (0.5). A disc or ring is ~0.9, a square 0.64, a 2.5:1 bar 0.44: bars and shirts the aspect gate (bounding box only) lets through;
  - **area 30–1200 px², w/h aspect 0.3–3.0,** now only sanity bounds. The lamps measured red 300–400, yellow 400–500 and green 700–800 px² at normal exposure, green glowing most (2026-10-04). Earlier caps: 600 rejected every green lamp, 300 the red too, 70 (8b3944c) every lamp; 5000 passed background patches.

  Setting `min_core_px` to 0 turns the core gate off (some tests do, to test the other gates on solid shapes). The limit: a lamp that doesn't clip at the camera's exposure fails the gate; check the course lights with `calibrate_lamps` first.
- **The course's light is not these lamps (2026-10-06).** It's bare 5 mm LEDs on a teal board, and the values above are `BlobFilter`'s defaults, not what the robot runs. From the stop line a lit LED is a white center with a thin colored ring of 10–35 px², and the red ring and its white center trace as separate blobs, so `MEASURED` (`config.py`) sets `min_area` 4, `max_area` 3000 (the green glow reaches ~700), `ref_area` 30, `min_roundness` 0.2 and turns the core gate off. `roi_crop.TRAFFIC` is cut narrow around the board, to keep out a second traffic light and a red box. It ran to the top of the frame until 2026-10-26, because stopped further forward (at a stop sign) the light rose out of a lower top edge; it now starts at 0.10, re-tuned on the track. The top edge brought the clear panel's reflection of each lit LED into view (red's reads yellow), and the swept bands still read the frames at the stop right with it there. On a nav run's frames at the stop, the real crop and detector got 201 of 201 right (87 green, 21 yellow, 93 red). Two caveats: the lowest red and yellow confidences were 0.46 and 0.44, close to Phase 3's 0.40; and another light, or this one from another stop line, has to land in that box. With the core gate on, 80 of 93 red frames were lost. `ref_area` 20 would lift the low scores to 0.75, but lets 73 frames carry a wrong color over 0.40 (the unlit yellow lens while green is lit), against 4 at 30.
- **Glow mode (what the robot runs since 2026-10-06).** With `ColorConfig.glow` set (`GlowFilter`, set in `config.py`), the blob gates above aren't used. The course's lenses are colored plastic: unlit, the red one reads V ~180 and its color band takes it in, and its colored ring when lit is no brighter (lit 10th percentile V 133 against 178 unlit; `calibrate_lamps` with `off=`). What only a lit LED does is clip the camera to white. So glow mode starts there:
  1. **The white spots:** connected pixels with V ≥ `white_min_v` (230) and S ≤ `white_max_s` (60), the core gate's test. At least `min_white_px` (2), or it's a glint. Its box must be between `min_white_aspect` (0.25) and `max_white_aspect` (3.0) wide for its height: lit LED centers read 0.5–1.0, while the glare along the board's edge is a 2 px strip at 10–11. That glare had more white than any LED, and its ring picked up a red shirt below, so it won as red during green and yellow (2026-10-07).
  2. **Named by the ring:** the pixels within `ring_px` (2) of the spot, counted against each color's HSV band. The band with the most is the spot's color, if it has `min_ring_px` (3) and at least `min_ring_share` (10%) of the ring. A lit LED's ring held 18–20% of its color (the rest is washed-out glow under the bands' S floors); a glare passed on a few stray colored pixels, ~5% of its ring, where the debug view's mask panel showed nothing (2026-10-07). The spot's own pixels don't vote (a clipped center keeps a trace of tint, and yellow's S floor is 20). The board's glare clips too, but its ring is teal, below green's V floor of 190.
  3. **One candidate:** the spot with the most white pixels, since a traffic light shows one lamp at a time. That also picks the LED over its dimmer reflection in the panel (red's reflection reads orange-yellow). Confidence is white pixels over `ref_white_px` (5), to 1.0: 2 px scores Phase 3's 0.40.

  On the 2026-10-06 frames at the stop (196 labelled): every frame right at confidence 1.0. An unlit lens can't be picked however large or bright, because it has no white center. `ring_px` 1 or 3 lost some red frames; the other values didn't move it. The trace keeps its keys: `area` and `core_px` are the spot's white pixels, gate `white`, `ring` (no color, label `none`) or `smaller`. The debug view draws it unchanged. `sweep_lamps` sweeps the bands in this mode, which only name rings now.
- **Confidence is area only:** `(area − min_area) / (ref_area − min_area)`, saturating at `ref_area` = 350 px², the smallest lamp at the stop (red), so every measured lamp scores 1.0; Phase 3 needs 0.40 (about 160 px²). `ref_area` must stay under `max_area`, or no lamp can reach the gate (800 with a 300 cap topped out at 0.35). Fusion keeps the highest confidence across all three colors, so the largest blob wins.

The HSV ranges have to be tuned under course lighting. Ranges from a lab or office won't carry over.

---

## Stage 4: Lane offset

**File:** `lane_offset.py` · **Config:** `LaneOffsetConfig` · **Output:** `LaneOffsetResult`

Turns lane candidates into one steering error.

### Stop lines are not boundaries

Geometry's lane detector doesn't know about stop lines, so a stop line short enough to pass the lane gates arrives as a lane candidate, and without a check lane offset would steer by its middle (on the test scenes, the right boundary moved from 289.5 to ~225 px). Lane offset first skips any candidate that belongs to a detected stop line: it lies across the ROI (wider than tall), at least `stop_line_overlap` (0.5) of its width is within the stop line's span, and it is no further above or below it than the line is thick. That also covers the dark pocket a stop line and two lane lines enclose. A lane line crossing the stop line runs along the ROI, so it is kept. Each skip is logged.

This check is now a backstop: the lane detector takes lines across the lane out of its edges first (see Stage 3A), so a stop line rarely arrives as a lane candidate. It still catches one when that filter is off or tuned narrower. A horizontal blob too short for either (under 46 px) is not skipped and still moves the offset.

### A second, stricter gate

Geometry decides "is this a lane marking". Lane offset decides "is it trustworthy enough to steer by". A candidate can pass the first and fail the second.

| Gate | Default | `MEASURED` | Rejects |
| --- | --- | --- | --- |
| `conf_threshold` | 0.30 | 0.25 | Weak candidates |
| `min_proximity` | 0.25 | 0.05 | Markings too far up the ROI to describe where the robot is now |
| `min_length_px` | 25 | 25 | Dashed-center-line fragments |
| `width_px` range | 1–25 | 1–45 | Noise below; blobs, glare and merged pairs above |
| `min_intensity` | 90 | 90 | Shadow edges and seams. `MEASURED` was 130 from the candidate sweep until a darker stretch of the course dimmed the tape below it (2026-10-05) |

`MEASURED` comes from a sweep of 4,827 recorded candidates rather than from course dimensions. The sweep ran on distorted frames; `MEASURED` now undistorts, which moves marks near the ROI edges by up to ~20 px, so the px gates need a re-sweep on undistorted captures. The default `min_proximity` sat above the observed median of 0.17 and threw out 1,752 candidates geometry had already scored at 0.30 or higher. The default width cap clipped detections whose 90th percentile was 27 px.

### Picking the lane

Each usable candidate becomes an anchor at its `foot_x`, weighted by confidence × (0.5 + 0.5 × proximity). A strong marking far up the ROI is real, but it says less about where the robot is right now.

The robot sits at the center of the lane ROI. Its lane is bounded by the **nearest** anchor on each side, not the outermost pair. On a two-lane street, the outermost pair would center the robot on the whole street.

1. Anchors on both sides of center: take the nearest on each side.
2. All anchors on one side (the robot has drifted out): take the two nearest center. They still describe the nearest lane, and the offset steers back.
3. One anchor: single-boundary mode.

Then the pair's spacing is checked. Closer than `min_lane_width_px` means one marking seen twice, or a line and something beside it; wider than 400 px means the pair spans more than one lane. Either one falls back to the strongest single anchor. The default floor is 60 px; MEASURED uses 300 px (~0.84 of the 357 px lane). On 2026-10-20, pairs 61-65 px apart were taken as a lane as the robot neared its right line, and one flipped the estimate's sign; pairs 250-300 px apart put the robot twice as far left as the right line alone did.

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

Single-boundary mode projects the lane center `expected_half_lane_px` from the visible line. Its side comes from which half of the ROI the line is in, which is wrong once the robot has drifted past a boundary: the lane's right line is then left of center. So the result also carries `other_side_offset`, the offset were the line the lane's other side, and Phase 3 takes it when the guessed side breaks continuity (`phase3_estimation.md`, Lane filter). Without that scale, distance-from-a-line and distance-from-center would feed the same controller on different scales, so the mode reports `single_uncalibrated` instead. MEASURED uses 178 px: half the lane's median two-boundary width over the 2026-10-12 runs (350 / 357 px). The hand-set 228 put every single-line frame ~0.25 off the two-line frames around it; 178 puts it within ~0.04-0.08.

`boundary_count` counts usable boundaries, not raw detections, so "saw nothing" and "saw nothing it could steer by" are different.

---

## Stage 4B: Stop-line distance

**File:** `stop_line_distance.py` · **Config:** `StopLineDistanceConfig` · **Output:** `StopLineResult`

Turns stop-line candidates into one measurement per frame: how far ahead the nearest stop line is.

```
candidates → confidence gate (≥ 0.4) → nearest (largest y_near_px) → distance_px = lane ROI height − y_near_px
```

The reference is the bottom of the lane ROI, which is the bottom of the frame: the nearest floor the camera sees, about 3 cm ahead of the robot. Both distances fall to 0 as the robot reaches the line (a clipped line is at 0).

- **`distance_px`** is in lane-ROI rows and needs no calibration.
- **`distance_cm`** is the floor distance forward from the reference point to where the line's near edge crosses the robot's centerline (X = 0), through `PipelineConfig.ground` (`perception/ground.py`). The near edge's two ends go from lane-ROI to frame coordinates (adding the ROI origin), onto the floor, and the crossing is taken there; a straight line stays straight through a homography, so this is exact even for a line seen at an angle or off to one side. Without a usable ground homography (none, or fit at another frame size), it comes from the stop-line table instead (below). `None` with neither. `cm_per_px` isn't used: it only holds at the bottom row.
- **`proximity`** is `y_near_px / ROI height` in [0, 1], the same closeness measure lane candidates carry (1 = at the ROI bottom).

`StopLineResult` also has `y_near_px`, the line's ends and tilt, `clipped`, the confidence and how many candidates geometry found. With nothing confident, `detected` is False and the numbers are None. Phase 3 votes on it and holds the distances (see `phase3_estimation.md`).

### Ground homography

**File:** `ground.py` · **Config:** `PipelineConfig.ground` (`GroundHomography` or `None`) · **Calibration:** `calibration/ground_homography.json`, from `scripts/calibrate_ground.py` (`make calib-ground`)

One 3×3 homography maps undistorted frame px to floor cm (X right+, Y forward+, origin at the reference point). It is fit on frames from `preprocess_frame` with `MEASURED`'s settings and records the lens calibration (a SHA-256 of its `image_size`, `camera_matrix` and `dist_coeffs`), `undistort_alpha` and image size it was fit under. `config.py` loads it once into `MEASURED.ground`; if it's missing, or any of those three differ from what preprocess uses, or undistortion is off, it warns and leaves `ground` None, which only turns the cm outputs off. `SCENE_CONFIG` has none, since synthetic frames aren't undistorted. Nothing reads it per frame.

### Stop-line table

**File:** `stop_line_table.py` · **Config:** `PipelineConfig.stop_line_table` (`StopLineTable` or `None`) · **Calibration:** `calibration/stop_line_table.json`, from `scripts/calibrate_stop_line.py` (`make calib-stopline`)

The board-free alternative for the stop line. The homography needs a checkerboard found flat and square on the floor, which is hard to get right; the stop line needs only one number, and it depends almost entirely on how many rows above the lane ROI's bottom the line appears. Tape strips laid at measured distances are measured by the robot's own detector (`distance_px`). A flat floor's curve, `cm = A / (B − rows) + C` (B is the horizon's height above the ROI bottom, in rows), is then fit through them by linear least squares. Measured by the detector the robot drives with, it needs no board, no corner finding and no squareness. It gives distance ahead only, from whatever point the marks were measured from (the front of the robot, by default); a clipped line reads 0. It's tied to the same lens calibration, `undistort_alpha` and size as the homography, through the same check (`ground.fit_conditions_problem`). `config.py` loads it into `MEASURED.stop_line_table`; a missing file is silent, since it's optional. The homography wins when both are present.

**Later, lane offset in cm.** `lane_offset_cm` still uses `offset × half ROI width × cm_per_px`, valid only at the bottom row. With the homography it would project both anchors (`foot_x` at their foot rows) to the floor and take the lane center's X there: `lane_offset_cm = −X_center` (+ = robot right of center). `cm_per_px` and `lane_roi_width_px` would retire, and the measured `expected_half_lane_px` (178) would become a course fact, `expected_half_lane_cm` (about 7 cm), projected per frame.

---

## Stage 4C: Obstacle

**File:** `obstacle.py` · **Config:** `ObstacleConfig` (`PipelineConfig.obstacle`) · **Output:** `ObstacleResult`

Is something solid standing in the robot's path, and how far ahead is its base? The course is dark mats and white tape, and neither has color, so in the obstacle ROI a pixel much brighter than the floor, or clearly colored, is "not floor". Tape is bright too, but thin: an opening with a square wider than any tape removes lane lines, dashes and stop lines, and what survives is something large and solid.

```
every 2nd px → floor level (25th percentile of the lane ROI) → not floor: gray ≥ floor + 60, or Lab color ≥ 15 from the frame's neutral with max(B,G,R) ≥ 50
             → opening, 47 px square → blobs → gates: area ≥ 2500, width ≥ 60, height ≥ 60, base ≤ 90 rows ahead
             → nearest (lowest base) → distance_px, distance_cm
```

- **Every second pixel** each way (`step`): the loop already runs at its budget on the Pi; an obstacle is tens of pixels across. Sizes and rows stay in full-resolution px.
- **Color is relative too:** |a − a_ref| + |b − b_ref| in Lab against the whole frame's median (a, b), the scene's neutral. A saturation threshold read the whole floor as colored under the warm light by the windows; the lane ROI's own median was the can itself once it filled the view. Over the 2026-10-12 runs (1001 clear frames, 199 with the can within ~20 cm), 15 gave no false alarm and every can frame.
- **The floor level** is measured each frame on the lane ROI, which is wider than the path, so a box in the path leaves floor beside it. The course's floor runs from dark mats to light gray (95–130 at the 2026-10-12 run's intersections): a fixed level (a cap at 70, the first version) called the light floor an obstacle, 376 of 537 frames on that run against 131 once measured (the can's, by eye).
- **`open_px` (47)** is wider than any lane tape (MEASURED's lane gate allows 45 px, the widest seen 41.7) and taller than any stop line (`max_thickness_px` 40; near the view bottom 1 cm of floor is ~10 rows). Tape wider than it would survive.
- **`distance_px`** is the rows from the blob's lowest row (where an upright object meets the floor) to the lane ROI bottom, as for the stop line. **`distance_cm`** is the floor there: through the ground homography at the base's center, else the stop-line table, whose curve maps any floor row to cm. The table's reference is the robot's front edge, and its row 0 is ~10.2 cm ahead of it, so a blob running off the ROI bottom is **`clipped`**: its base isn't in view, "this near or nearer".
- **Blind to** an object no brighter than the mats and with no color, and to one wholly outside the middle of the frame. By brightness alone, also to a light object so wide it covers the whole lane ROI: it becomes the floor level (by color it is still seen). **May mistake** a bright or colored patch on the floor at least 47 px across (glare, a light floor past the mat's edge, a stop sign low in the corridor) for one. The gates are a first guess; tune them on the course with the obstacle view (`make phase2 VIEWS=obstacle`).

Phase 3 votes on it and holds the measurement through a missed frame, as for the stop line; Navigation's `ObstacleRule` brakes for one within 12 cm (or 24 rows with no cm, or clipped) and waits until it has been gone 1 s.

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

No computation. Collects the detections, lane offset and stop-line distance, checks that they belong to one frame, and packages them as the Phase 3 contract.

```
Phase2Output
    detections           list[DetectionObject]   in fusion order
    lane_offset_results  list[LaneOffsetResult]  one per frame; [] if none
    frame_id             int
    timestamp_ms         int
    detection_count      int                     always len(detections)
    stop_line_results    list[StopLineResult]    one per frame (detected False when none); [] if none
    obstacle_results     list[ObstacleResult]    one per frame (detected False when none); [] if none
```

Stop lines and the obstacle travel beside the detections, like the lane offset, because they are a measurement rather than a detection to fuse. The same packaging rules apply to them.

Checked on construction, so a malformed handoff fails loudly instead of reaching navigation:

- All four lists must be lists; pass `[]` for none
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

Each stage's test writes a timing CSV and histogram. Geometry, the longest stage, is also timed per part: `geometry.edges` (Canny and the gradient split every lane-ROI detector reads), `geometry.lanes`, `geometry.stop_line` and `geometry.sign` (`GEOMETRY_PARTS`). The parts print indented under `geometry` in `summary.txt`'s `[TIMING]` and aren't added to the total, since geometry's own time holds them.

The target is the full chain plus Phase 3 inside one frame period: 50 ms at 20 FPS (P1). On a run, `nav.csv`'s `phase2_ms` and `src.analysis.stage_timing` show where the time goes, and `make routine-frame-budget` judges it.

---

## Parallel perception

**File:** `parallel.py` · **Switch:** `params.PARALLEL_PERCEPTION` (off), or `--parallel` / `--serial` on a linker for one run

Phase 2 ran every stage on the main thread, one after another: on the 2026-10-14 diag run the main thread sat at 70% of a core on average (99.7% peak) while the Pi as a whole was at 43%. Three stages read only the cropped ROIs and nothing of each other's: geometry (~20.6 ms on the course), color (~5.2) and obstacle (~5.9). With parallel perception on, color and obstacle run on one side thread (`p2-side`) while the main thread runs geometry, then the lane offset and the stop line (which need geometry), and only then waits for the side thread.

```
main:  preprocess → roi ─┬─ geometry → lane_offset → stop_line ─┬─ branch_wait → fusion → package
side:                    └─ color → obstacle ────────────────────┘
```

It's a thread, not a process: the heavy work is inside OpenCV calls, which release the GIL, so the two run on two cores without copying a frame. The Python between those calls still takes turns on the GIL, which is why the gain is measured rather than assumed. Stage timing reports `branch_wait`, the main thread's time waiting for the side (0 when the side finished first), and the side thread's own color and obstacle times (`side_timings_ms`). The frame waits on the longer path, which is geometry: splitting color and obstacle onto separate threads would not shorten it. `make compare-parallel` times Phase 2 alone on the same frames both ways; `make ab-parallel` alternates whole runs.

The results are the same either way: the branches are pure functions of the ROIs and the config, and `test_parallel` holds them equal frame by frame.

---

## Seeing inside it

```
python3 -m src.phase2_linker --camera --views stop,traffic,stopline,lanegeo,obstacle
python3 -m src.phase2_linker --video run.avi --no-display
```

Runs the chain with the debug overlay. Accepts `--camera`, `--video PATH` or `--frames DIR`, and records each view as a video plus CSV under `runs/<timestamp>/`.

| View | Shows |
| --- | --- |
| lane (always on, `debug_lane`) | Every raw candidate (green usable, red with the gate that rejected it), each anchor's foot, the chosen left and right boundaries, robot and lane center, mode and offset gauge |
| `stop` (`debug_stop`) | The color sign ROI with its red blobs colored by the gate that decided them ("smaller" for those never gated because a larger one passed), with vertex count and confidence; beside it the redness image, the red mask outlined, and the threshold |
| `traffic` (`debug_traffic`) | HSV masks and blobs against the bands. In glow mode, also the white mask, and each white spot boxed with the glow gate that decided it (white, shape, ring, smaller) and its ring's winning share |
| `lanegeo` (`debug_lanegeo`) | On the lane ROI: every contour the lane detector traced, red if geometry refused it (gate and measured value), amber if lane offset did, green if usable, with the chosen anchors; below it, the edges with what the horizontal-line filter removed and what closing added. Needs the chain's trace (`trace=True`, which the linker sets), which adds `lane_debug["trace"]` |
| `stopline` (`debug_stopline`) | On the lane ROI: accepted stop lines as bands, rejected top edges with their gate, the measured distance, lane candidates skipped as part of a stop line; below it, the gradient split (all edges, kept top and bottom edges, fitted lines) |
| `obstacle` (`debug_obstacle`) | On the obstacle ROI: not-floor pixels tinted, every blob that survived the opening boxed green (passed) or red (with its gate), the measured one in white with its distance to the view bottom; below it, the masks (gray not floor, white after the opening) |

`pytest --hardware -k debug_lane` saves every stage's images for 3 sample frames plus an annotated video, which is the place to start when tuning a detector by eye.

---

## Open items

- **Traffic lights under the day's lighting.** The color branch is on, in glow mode, tuned on the course's LEDs (2026-10-06, above); re-check under the venue's light (`make routine-detect-range`).
- **Traffic and sign ROI bounds** need a small re-tune on the track: both were set from one day's frames (2026-10-06, 2026-10-20). `make phase2 VIEWS=traffic,stop` shows them on a run.
- **Stop-sign range.** The shape gates are tuned on the course's sign (2026-10-20); the range, and Phase 3's 0.45 vote gate, are measured by `make routine-detect-range ARGS="--set target=sign"`.
- **`expected_half_lane_px`** re-measured whenever the mount moves (178 px from the 2026-10-12 runs' two-boundary widths).
- **The intersection failure:** in 3 of 3 runs the robot drifted right and failed at an intersection. On synthetic frames a stop line did exactly this: a stop line touching one lane line removes that line and the single-sided projection reports a large positive (robot right of center) offset, and a short one was taken as the right boundary. Stop lines are now detected and skipped; record those spots again and check what the lane does as the stop line comes into the ROI.
- **Obstacle gates on real frames:** the not-floor thresholds against the mats under the venue's light, `open_px` against the widest tape, the false-alarm rate on a clear course, and what a stop sign low in the corridor does (`make phase2 VIEWS=obstacle`, then `test_obstacle --hardware --obstacle-cm`).
- **Stop-line gates on real frames:** tape thickness in px, the 60 px minimum length against real dash ends, and `MEASURED`'s 5° tilt against real approach angles. A lighter patch of mat next to the line can pair with it into a thick false candidate (seen with looser Canny thresholds on a synthetic frame).
- **`distance_px` to cm:** needs a ground homography or a stop-line table, not `cm_per_px`.
- **Offset accuracy on hardware:** the sign (+ = robot right of center) is what lane keeping steers by on every run; the accuracy against a ruler (P3) is `make routine-lane-offset`.
