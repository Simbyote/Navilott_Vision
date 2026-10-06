# Traffic-Light Sweep

> Try many HSV floors and blob gates on frames where you know which lamp was lit, and keep the settings that read the most of them right.

`calibrate_lamps` measures a lamp as one bright disc with a dimmer glow around it, and puts each HSV band between the two. That works for diffused lamps. The course's light isn't one (2026-10-06): it's bare 5 mm LEDs on a teal board. From the stop line a lit LED is a white center with a thin colored ring. The clear panel above the board reflects each lit LED (red's reflection reads yellow), and there's a second traffic light nearby. Measured bands and hand-tuned ones both passed nothing.

This script doesn't measure the lamp. It runs the color branch itself on your labelled frames with every setting in a grid and counts what Phase 3 would see:
- a lamp of the lit color, at or above Phase 3's confidence gate (0.40), is a **hit**;
- any other color there is a **false reading**.

The best settings are the ones that read the most frames right. Hue stays as it is in `hsv_ranges.json`, because a lamp's hue is rarely the problem. What's swept is what took hand-tuning: each color's S and V floors, and the blob gates (`min_area`, `ref_area`, roundness, the core gate).

**Code:** `src/scripts/sweep_lamps.py` · **Tests:** `test_sweep_lamps.py` · **Make:** `make sweep-lamps`

---

## 1. Record

Park the robot where it stops at the light, motors off, and let the light cycle through every color:

```
make nav-dry MAX_RUN_S=30
```

The frames land in the newest `runs/nav_*/frames/`, named by frame number (`000123.jpg`).

## 2. Label

Find which frames had which lamp lit. Two easy ways:
- open a few frames along the run;
- read `nav_video.csv`'s `traffic_detections` column to see roughly when things change.

Leave out a few frames on each side of a change, where the light is switching. Then:

```
make sweep-lamps LAMPS="green=runs/nav_X:0-86 yellow=runs/nav_X:87-107 red=runs/nav_X:108-200"
```

Each label is `color=path` (a whole folder, or one image) or `color=path:A-B` (frames A to B, by the number in the file name). The labels are `red`, `yellow`, `green`, and `off` for frames with no lamp lit, where anything read counts as false. Labels can come from different runs, for example one recording per color.

Up to 60 frames per label are used, spread across its span (`ARGS="--frames 120"` for more). A sweep takes seconds on a laptop and about a minute on the Pi.

## 3. Read the report

```
now (hsv_ranges.json, MEASURED's blob gates):
  lit \ read as       red  yellow   green     off
  red                  60       0       0       0
  yellow                9       1       9       0
  green                21       0      39       0
  right 100, wrong color 39, missed 0; lowest right confidence 0.77 (gate 0.4)

red: best floors (S, V) on its own, hits / frames, false readings, lowest hit confidence
  S  20  V 140    60 /  60   false   0   1.00   <- chosen
  ...

swept:
  ...
  right 138, wrong color 0, missed 1; lowest right confidence 0.56 (gate 0.4)

bands (hsv_ranges.json):
  red_low   lower [0, 20, 140]  upper [8, 255, 255]
  ...
blob gates (config.py, _TRAFFIC_LIGHT_BLOB):
  BlobFilter(min_area = 4.0, max_area = 3000.0, ref_area = 20.0, min_roundness = 0.2, min_core_px = 0)
```

That's the course's light on 2026-10-06, starting from the hand-tuned bands.

- **The tables** show each lit color (row) against what the frame read as (column). Everything should sit on the diagonal. **Wrong color** is the dangerous one: red read at a green light stops the robot. **Missed** is milder, because the traffic rule remembers red for 0.5 s.
- **The floors list** shows each color's best settings scored on that color alone. Several settings often tie, so the sweep takes the one whose weakest reading is strongest, then the middle of those, so the choice isn't on the edge of what works.
- **Lowest right confidence** near the 0.40 gate is a warning: a dimmer room may drop that lamp.

To keep the result, save the bands with `ARGS=--write` (only `hsv_ranges.json` is written), and paste the `BlobFilter(...)` line into `config.py` as `_TRAFFIC_LIGHT_BLOB`.

## 4. When it can't get there

- **Wrong colors left after the sweep** mean something in the traffic ROI looks like a lamp to every setting: a reflection, another light, a colored object. No band fixes that. Narrow `roi_crop.TRAFFIC` around the light (see `phase2_perception.md`, the color branch), then sweep again.
- **Misses at every setting** mean the lamp is too small or too dim in the frame. Check it's inside the traffic ROI (`make phase2 VIEWS=traffic`), and that the robot stops where it did in the recording.
- **A different room:** record there too and label both runs in one sweep, so the settings fit both.

## How it works

1. Each label's frames are read, and the traffic ROI is cut the way the robot cuts it (undistortion and `roi_crop.TRAFFIC` from `MEASURED`).
2. It starts from `hsv_ranges.json` and `MEASURED`'s blob gates.
3. Then two rounds of:
   - **Floors:** each color's (S, V) floor is scored on its own over `S_FLOORS` × `V_FLOORS`. The score is hits minus 2 × false readings: a false reading costs two hits.
   - **Blob gates:** scored on whole frames over `MIN_AREAS` × `REF_AREAS` × `MIN_ROUNDNESS` × `MIN_CORE_PX`, the frame reading as its highest-confidence lamp at the gate, which is fusion's pick.

The grids are constants at the top of the script; widen them there.

**In glow mode** (`MEASURED.color.glow`, the robot's since 2026-10-06; see `phase2_perception.md`), a lamp is found by its clipped white center and the bands only name the ring around it. The sweep scores that mode, sweeps only the bands, and prints no `BlobFilter`.
