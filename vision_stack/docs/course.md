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
| Intersection square | about 28–30 cm per side | Future intersection detection |
| Center line | Dashed. Dash and gap length; 4 cm ech | Minimum line length for steering (so dashes don't anchor a boundary) |
| Surface / marking | White tape on dark mats | Edge thresholds and the minimum brightness gates |
| Stop line | Width and distance from the intersection; stop line on intersection gates | Future stop-line detection |

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

**The camera must sit on the robot's centerline.** Lane offset treats the image center as the robot's position.

Moving the mount changes the height, tilt or position, which invalidates the lens calibration and the ground scale, and shifts where the lane appears in the frame. Re-measure the ground scale after any mount change (`requirements.md`, P3 procedure, step 1).

---

## Values that come from these measurements

| Setting | Current value | Where it should come from |
| --- | --- | --- |
| Lane ROI | Bottom 30% of the frame, 5–95% across | The rows showing the road just ahead of the robot at this height and tilt |
| Minimum plausible lane width | 60 px | Well under one lane width in px |
| Maximum plausible lane width | 400 px | Between one lane and a full street, in px |
| Lane center from one visible line | 228 px, hand-set | Half the lane width in px, measured |
| Ground scale | Not set | 14 cm ÷ lane width in px, measured at the bottom of the lane ROI |

Once the ground scale is measured, the three px values above can be checked against it. For example, at a scale of 0.05 cm/px, one lane (14 cm) is about 280 px and a full street (29–30 cm) is about 590 px. The 400 px maximum would sit correctly between them, and the 228 px half-lane value would be too high (140 px expected). The numbers are only an example; use the measured scale.

Perspective makes the scale depend on the row: the road near the bottom of the frame is closer and spans more pixels per cm than the road near the top of the lane ROI. Measure the scale at the rows where the lane lines are read, which is the bottom of the lane ROI.
