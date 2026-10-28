# Course and Camera Geometry

> The physical world the pipeline is tuned for.

Several pipeline settings are really facts about the course and the camera mount: where the ROIs sit, what lane width counts as plausible, and how far the lane center is from one visible line. This doc records those facts and which settings depend on them. If the course, the tape or the mount changes, check the "Depends on it" column.

Values marked **TBD** haven't been measured or written down yet.

---

## Road markings

| Feature | Size | Depends on it |
| --- | --- | --- |
| Lane width | about 14 cm | Ground scale (cm per px); single-line lane center projection |
| Lane line (tape) width | about 1 cm | Lane-line width gates in geometry and lane offset |
| Full street width (two lanes) | about 29–30 cm | Maximum plausible lane width; choosing the robot's own lane over the whole street |
| Intersection square | about 28–30 cm per side | How far a turn drives in before turning and the longest crossing: `ADVANCE_CM` (22.3) and `MAX_CROSS_CM` (49.7) in `navigation/intersection.py` |
| Center line | Dashed: dash and gap about 4 cm each | Minimum line length for steering (so dashes don't anchor a boundary) |
| Surface / marking | White tape on dark mats | Edge thresholds and the minimum brightness gates |
| Stop line | Width and distance from the intersection **TBD** | `StopLineFilter` in `perception/geometry_shared.py`: minimum length 60 px (longer than any lane-tape width), thickness 3–40 px, tilt ≤ 20° (5° in `MEASURED`). Check all three against real captures |

**How lane width is measured:** from the center of one line to the center of the other.

---

## Signs and lights

| Item | Specification | Depends on it |
| --- | --- | --- |
| Stop sign | Size, mounting height and distance from the lane **TBD**. Posted right of the lane | Sign ROI (upper-right of the frame); sign area gates |
| Traffic light | Lamp size, mounting height and position **TBD** | Traffic ROI (top-center of the frame); blob area gates and the expected lamp size |
| Lighting | Venue and lighting type **TBD** | HSV ranges |

---

## Courses

| Course | Size | Layout | Used for |
| --- | --- | --- | --- |
| Practice arena | 250 × 300 cm, 18 mats | Straights and intersections | Full runs |
| Bench course | **TBD** | Two streets and an intersection | Hardware tests without the full arena |
| Demo course | **TBD** | **TBD** | Final evaluation |

A map of the practice arena, with each segment labeled, would make course recordings easier to refer to ("failed at intersection B"). **TBD**

---

## Camera mount

| Quantity | Value | Depends on it |
| --- | --- | --- |
| Lens height above ground | about 4.5 cm | Everything below; chosen from field-of-view testing |
| Forward view at that height | about 3 cm ahead | How much road the lane ROI covers |
| Tilt | **TBD** | ROI placement |
| Lateral position | centered on the robot | The robot is assumed to sit at the horizontal center of the image |
| Orientation | Upside down, corrected in capture | Capture flip setting |

**The camera should sit on the robot's centerline, pointing straight ahead.** Lane offset reads the robot as sitting at `robot_x_px` of the lane ROI, its middle when unset. A camera shifted or turned off the centerline makes lane keeping hold the robot to one side of its lane, and raising `expected_half_lane_px` can't fix that (it only moves one-line frames). After moving the mount, park the robot centered in a straight lane and run `make calib-lane`: it measures where the lane's middle falls (`robot_x_px`) and the lane's width in px (`expected_half_lane_px`, `min_lane_width_px`).

Moving the mount changes the height, tilt or position, which invalidates the lens calibration and the ground scale, and shifts where the lane appears in the frame. Re-measure the ground scale after any mount change (`requirements.md`, P3 procedure, step 1).

---

## Values that come from these measurements

| Setting | Current value | Where it should come from |
| --- | --- | --- |
| Lane ROI | Bottom 30% of the frame, 5–95% across | The rows showing the road just ahead of the robot at this height and tilt |
| Minimum plausible lane width | 300 px (MEASURED; 60 the default) | ~0.84 of the measured 357 px lane: pairs 61–65 px apart (a line and something beside it) were taken for a lane and flipped the estimate's sign (2026-10-20) |
| Maximum plausible lane width | 400 px | Between one lane and a full street, in px |
| Lane center from one visible line | 178 px (MEASURED; 228 hand-set until 2026-10-12) | Half the lane width in px: the median two-boundary width over the 2026-10-12 runs, 350–357 px |
| Ground scale | Not set | 14 cm ÷ lane width in px, measured at the bottom of the lane ROI |

Once the ground scale is measured, the three px values above can be checked against it. For example, at a scale of 0.05 cm/px, one lane (14 cm) is about 280 px and a full street (29–30 cm) is about 590 px. The 400 px maximum would sit correctly between them, and a 228 px half-lane value would be too high (140 px expected). The numbers are only an example; use the measured scale.

Perspective makes the scale depend on the row: the road near the bottom of the frame is closer and spans more pixels per cm than the road near the top of the lane ROI. Measure the scale at the rows where the lane lines are read, which is the bottom of the lane ROI.
