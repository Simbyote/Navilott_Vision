# Traffic-Light Lamp Calibration

> The HSV bands the robot finds each traffic-light lamp with, measured from frames of the real lamps.

The color branch (`perception/color_branch.py`) finds a lamp as the pixels inside its color's HSV band (`calibration/hsv_ranges.json`), then measures the blob. A lamp on camera isn't one color:
- a **core** so bright it's nearly white (low S, V at 255);
- a **ring** of the lamp's real color around it: the lamp;
- a **glow** past the lamp, still tinted but dimmer and paler.

A band loose enough to take the glow measures the glow, not the lamp: the blob comes out too big, and `BlobFilter.max_area` rejects it. Tuning the band by hand is slow because the channel that separates lamp from glow isn't obvious: on 2026-10-04 the green lamp's hue sat at 98-101 (cyan on camera, outside the old 40-85 green band), and only V and a low S threshold separated it from its glow. This script measures each lamp against its own glow and puts the band between them.

It writes `calibration/hsv_ranges.json` with `--write`, changing only the colors measured.

**Code:** `src/scripts/calibrate_lamps.py` · **Tests:** `test_calibrate_lamps.py`

---

## 1. Recording the lamps

Light the lamps **exactly as they're lit on the course**. Their brightness is what's being calibrated: if the LEDs share one current (one resistor, or a constant-current driver), unhooking two makes the third brighter than it ever is in a run. If the course lights one color at a time by switching the others off, measure it that way.

Put the robot where it stops at the light, square to it. For each color, light that lamp and record a few seconds:

```
python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 5 --no-render --out runs/lamp_red
python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 5 --no-render --out runs/lamp_yellow
python3 -m src.navigation_linker --camera --no-motors --no-button --max-run-s 5 --no-render --out runs/lamp_green
```

A run that ends early because no lane is in view is fine: a few frames are enough. The lamp must be the brightest thing in the traffic ROI (top-center of the view, `roi_crop.TRAFFIC`).

## 2. Measuring

```
python3 -m src.scripts.calibrate_lamps red=runs/lamp_red yellow=runs/lamp_yellow green=runs/lamp_green
```

Any colors, each an image or a run folder (up to 15 frames, spread across it). Each frame goes through the robot's own preprocess and traffic-ROI crop, so the numbers are what the color branch sees. Per color it prints (here a drawn cyan-green lamp, like the course's):

```
green: 15 frame(s); lamp at (48.0, 50.0) in the traffic ROI (first frame)
          H p5/p50/p95      S p5/p50/p95      V p5/p50/p95
  lamp     99   100   100     33    52    52    243   245   254
  glow     96    98    98     24    29    29    210   212   212
  green     lower [95, 32, 228]  upper [104, 255, 255]
  blob area under it: 228 px^2 (median over the frames)
```

- **lamp / glow:** H, S and V percentiles of a disc around the brightest blob (radius 8 px) and of a ring outside it (12-24 px).
- **The band:** hue spans the lamp's (5th-95th percentile, 4 either side); the S and V minimums sit halfway between the lamp's 10th percentile and the glow's 90th. Red gets both halves of its band (`red_low`, `red_high`), split at the hue wrap.
- **blob area:** the largest blob under the new band, measured as the color branch measures it. Set `BlobFilter` from these: `ref_area` (confidence 1.0) about the typical lamp, `max_area` with some headroom above the largest.

Then, if it looks right:

```
python3 -m src.scripts.calibrate_lamps red=runs/lamp_red yellow=runs/lamp_yellow green=runs/lamp_green --write
```

`--write` merges the measured colors into `calibration/hsv_ranges.json`, keeping the others, and only after the color branch's own loader accepts the result.

## 3. Warnings

| Warning | What it means | What to do |
| --- | --- | --- |
| `the lamp and its glow overlap in both S and V` | At this exposure no band takes the lamp without its glow | Darken the exposure and measure again. Preview it with `rpicam-still --ev -1 -o test.jpg` |
| `only V separates the lamp from its glow` (or S) | One channel carries the whole separation | Works, but marginal: a change in lighting can tip it |
| `red and yellow share hues` | An overexposed red goes orange; a red lamp could pass for yellow | Darken the exposure and measure again. Never drive with this one standing |
| `no band: the spot found is white, not colored` | The brightest spot has no color (saturation under 25): the lamp is blown out to white, or the brightest thing isn't the lamp. No band is suggested or written for that color | Darken the exposure; check what's brightest in the traffic ROI (top-center of the view) |
| `no band: the spot's hue spans ...` | The spot's pixels aren't one color | As above |
| `the red and yellow lamps were found at the same spot` | Two colors' brightest spots coincide, so at least one isn't its lamp: a reflection or a light behind the signal, or one diffuser covering both LEDs | Block the other light, or aim so only the signal is in the traffic ROI |

## Room lighting

The lamps make their own light, so the room's lighting barely changes their color directly. It changes everything around them, and the camera's automatic adjustments carry that into the lamps:
- **Exposure:** the camera sets its brightness for the whole scene. A dim room makes it brighten, and the lamps blow out to white (the `no band: ... white` message). A bright room darkens them.
- **White balance:** the camera shifts all colors to make the room's light look neutral. Under a warm, yellowish bulb it pushes the whole image toward blue, lamps included: green toward cyan, yellow toward green.
- **The surroundings:** a warm bulb tints white and gray surfaces orange-yellow, with low saturation. Those can pass a loose yellow or red band and show up as false lamps.

So calibrate **where and under the light the robot will race in**. Bands measured under a ceiling-fan bulb at home will be off under a venue's LED or fluorescent lighting. If you can't calibrate there, keep the room's main light off and light the scene with something neutral (daylight or a white LED), and measure again on site before the run.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The lamp position printed isn't the lamp | Something brighter is in the traffic ROI (a window, a reflection): block it or move the robot |
| `ERROR: ... no readable .jpg or .png frames` | The path holds no frames: point at the run folder `navigation_linker` wrote (its frames are under it) |
| A detected lamp still too large after `--write` | The glow passes the band: check the glow's S and V against the band's minimums; darker exposure separates them |
