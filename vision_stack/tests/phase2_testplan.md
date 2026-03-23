# AutoBot — Perception System

## Phase 2 Unit Tests — Vision Perception

Tests covering preprocessing, ROI cropping, color/geometry branching, feature fusion, and perspective transform. Maps to benchmark stages 1–5.

---

## Phase 2 — Vision Perception

### Test 2.1 — Preprocessing Stage — Undistortion & Equalization

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 2 — Vision Perception (Preprocessing) |
| **Equipment:** | Raspberry Pi Zero 2 W, IMX219 camera module, calibration checkerboard pattern, `calibration/camera_matrix.npz` |
| **Description:** | Verify that lens undistortion and histogram equalization are applied correctly. Straight lines in the scene must remain straight after undistortion. Equalization must improve contrast under simulated low-light input. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Place a printed checkerboard flat in front of the camera at 30 cm. Capture a raw frame and save it. | Checkerboard lines appear curved near frame edges in the raw frame. | | |
| 2 | Apply undistortion using the loaded `camera_matrix.npz`. Save undistorted frame. | Checkerboard lines appear straight to visual inspection in the undistorted output. | | |
| 3 | Feed a uniformly dark test image (all pixels = 30) through histogram equalization stage. | Output pixel range spans 0–255. Mean pixel value increases significantly. | | |
| 4 | Confirm output dtype and shape are unchanged from input. | Output: `dtype=uint8`, `shape=(480, 640, 3)`. No shape or type change. | | |
| 5 | Measure preprocessing time on Pi with both undistortion and equalization active. | Mean preprocessing time < 5 ms per frame at 640×480. | | |

---

### Test 2.2 — ROI Cropping — Region Extraction Accuracy

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 2 — Vision Perception (ROI Cropping) |
| **Equipment:** | Raspberry Pi Zero 2 W, static test image (640×480), SSH terminal |
| **Description:** | Verify that the three ROI regions (`lane_roi`, `traffic_roi`, `sign_roi`) are extracted at the correct pixel coordinates and are views of the original frame with no memory copy. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Pass a known 640×480 test image through `roi_crop.py`. Print `lane_rect`, `traffic_rect`, `sign_rect`. | `lane_roi` covers lower half: y=[240,480], x=[0,640]. `traffic_roi` covers top-center region. `sign_roi` covers right half, full height. | | |
| 2 | Modify a pixel in `lane_roi`. Check if the same pixel in the source frame also changed. | Pixel change in `lane_roi` is reflected in source frame — confirms zero-copy view. | | |
| 3 | Confirm `ROICropResult.source_shape` matches the input frame shape. | `source_shape = (480, 640, 3)`. | | |
| 4 | Pass a frame with distinctive colored squares in each ROI region. Confirm each ROI extract contains only the expected square. | Each ROI contains its corresponding color patch and no bleed from adjacent regions. | | |

---

### Test 2.3 — Color Branch — Traffic Light State Detection

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 2 — Vision Perception (Color Branch) |
| **Equipment:** | Raspberry Pi Zero 2 W, synthetic test images with isolated red / yellow / green blobs on black background, `calibration/hsv_ranges.json` |
| **Description:** | Verify that the HSV color branch correctly identifies traffic light states from canonical test images and rejects off-color false positives. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Feed a synthetic image containing a single red blob (H=0°, S=255, V=200, 30×30 px) in the `traffic_roi` region. | `TrafficLightCandidate` returned with `label='red'`. Confidence > 0.5. | | |
| 2 | Feed a synthetic image with a yellow blob (H=30°, S=255, V=200). | `TrafficLightCandidate` returned with `label='yellow'`. Confidence > 0.5. | | |
| 3 | Feed a synthetic image with a green blob (H=60°, S=255, V=200). | `TrafficLightCandidate` returned with `label='green'`. Confidence > 0.5. | | |
| 4 | Feed a white blob (H=0, S=0, V=255). Verify no candidate is returned. | No `TrafficLightCandidate` returned. Empty list. | | |
| 5 | Feed a frame with both red and green blobs. Verify conflict resolution returns only the highest-confidence candidate. | Only one `TrafficLightCandidate` returned per frame (highest confidence wins). | | |
| 6 | Measure color branch execution time on Pi with a real `traffic_roi` frame. | Color branch time < 6 ms per frame. | | |

---

### Test 2.4 — Geometry Branch — Lane Boundary & Stop Sign Detection

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 2 — Vision Perception (Geometry Branch) |
| **Equipment:** | Raspberry Pi Zero 2 W, synthetic lane images (white lines on dark background), synthetic octagon image, SSH terminal |
| **Description:** | Verify that the geometry branch correctly detects lane boundaries via Canny + contours, and identifies stop sign octagons via polygon approximation. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Feed a synthetic `lane_roi` with two white vertical lines on black. Run `geometry_branch.py`. | Two `LaneCandidate`s returned with `label='lane_boundary'`. Contours align with the white lines. | | |
| 2 | Feed a `lane_roi` with no lines (all black). Verify output is empty. | Empty `LaneCandidate` list returned. | | |
| 3 | Feed a `sign_roi` with a white octagon (8-sided polygon) on black background. | `SignCandidate` returned with `vertex_count ≈ 8`, confidence > 0.5. | | |
| 4 | Feed a `sign_roi` with a white square (4-sided polygon). Verify it is not classified as a stop sign. | No `SignCandidate` returned for 4-sided shapes (`vertex_count ≠ 8` rejects it). | | |
| 5 | Measure geometry branch execution time on Pi using a real 640×480 `lane_roi`. | Geometry branch time < 12 ms per frame. | | |

---

### Test 2.5 — Feature Fusion — Conflict Resolution & Output Schema

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 2 — Vision Perception (Feature Fusion) |
| **Equipment:** | Raspberry Pi Zero 2 W, SSH terminal, synthetic candidate inputs |
| **Description:** | Verify that `feature_fusion.py` correctly merges candidates from both branches, applies conflict resolution rules, and outputs detection objects conforming to the defined schema. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Inject two `TrafficLightCandidate`s with `label='red'`, confidence=0.9 and `label='green'`, confidence=0.6. Run fusion. | One `DetectionObject` returned with `type='traffic_light'`. Position is centroid of the 0.9-confidence candidate. | | |
| 2 | Inject three `LaneCandidate`s. Verify all three are present in output. | Three `DetectionObject`s returned with `type='lane_boundary'`. | | |
| 3 | Inject two `SignCandidate`s (different confidence values). Verify only the highest-confidence one is kept. | One `DetectionObject` returned with `type='stop_sign'`. | | |
| 4 | Verify each output `DetectionObject` has: `type`, `position.x`, `position.y`, `confidence`, `timestamp` fields. | All four fields present and of correct type. Position values are floats (or int per implementation note). | | |
| 5 | Feed a frame with no candidates from either branch. Verify output is an empty list. | Empty list returned. No crash. | | |

---

### Test 2.6 — Perspective Transform — Homography Accuracy

| | |
|---|---|
| **Written by:** | Mike Orduna 03/22/2026 |
| **Subsystem:** | Phase 2 — Vision Perception (Perspective Transform) |
| **Equipment:** | Raspberry Pi Zero 2 W, flat calibration grid on course surface, `calibration/homography_matrix.npz`, SSH terminal |
| **Description:** | Verify that the homography transform correctly maps camera-perspective lane coordinates to a bird's-eye plane. Parallel lines in the real world must appear parallel after transformation. |
| **Results:** | PASS / FAIL / Could not be completed |

| # | Procedure | Expected Result | Actual Result | Comments |
|---|---|---|---|---|
| 1 | Place two parallel tape strips on the course surface 20 cm apart. Capture a frame. Apply `warpPerspective` using the saved homography. | Tape strips appear parallel in the warped output image. No convergence. | | |
| 2 | Use Mode B (point transform). Map four known ground-truth points. Confirm pixel distances scale linearly with physical distances. | Pixel distance between two points at 20 cm physical separation equals twice the pixel distance of two points 10 cm apart. | | |
| 3 | Apply transform to a known image coordinate that corresponds to the lane center directly in front of the robot. Expect transformed x ≈ 0 (center). | Transformed x coordinate < 5 pixels from 0 when camera is centered over lane. | | |
| 4 | Re-run after deliberately tilting the camera 5°. Verify transformed coordinates shift measurably — confirming calibration sensitivity. | Parallel lines no longer appear parallel after tilt, confirming the matrix is camera-mount-sensitive. | | |

---
