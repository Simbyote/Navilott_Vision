# Bench Testing Guide — `feature/vision-pipeline-rework`

How to run the fix-toggle bench build, what each command is for, and how to read
the results. This is vision-only testing: no Pi hardware, no GPIO/IMU/motors.

---

## One-time setup

Run everything from the **repo root** (`~/Personal_Workspace/Navilott_Vision`),
never from `vision_stack/src/`. Every module hardcodes `vision_stack/frames` as a
relative path, so they all assume root as the working directory. If you see a
`FileNotFoundError` with `cwd=...` in it, this is why.

```bash
cd ~/Personal_Workspace/Navilott_Vision
git switch feature/vision-pipeline-rework      # confirm you're on the right branch
mkdir -p logs                                   # log output lands here
```

### Fetch + extract the dataset (once)

```bash
python3 -c "
import sys; sys.path.insert(0, 'vision_stack/src')
from unzip_data import fetch_dataset
print(fetch_dataset(
    url='https://github.com/Simbyote/Navilott_Vision/releases/download/v1.0-dataset/frame_tracks.zip',
    zip_path='vision_stack/frames/frame_tracks.zip',
    dest_dir='vision_stack/frames',
))
"
ls vision_stack/frames/          # confirm the track dirs exist
```

**Available tracks: `trackT3`, `trackT4`, `trackT5`** (1201 frames each).
There is no `trackT2` — don't reference it in any run.

---

## The commands, and what each is for

### 1. Correctness gate — run before trusting anything

```bash
python3 vision_stack/src/test.py
```

**Point:** proves the five fixes behave correctly on synthetic shapes (pole gets
rejected, trapezoid masks the right region, anchors land in the right half, etc.)
and — critically — that all flags OFF is bit-identical to the pre-fix pipeline.
Must print `ALL PASS`. If it fails, stop: every downstream bench run is
meaningless until the logic is correct. This tests *code correctness*, not
whether the fixes help on real footage.

### 2. Baseline run — the reference every fix is measured against

```bash
python3 vision_stack/src/run_pipeline.py --frames vision_stack/frames/trackT3 \
  > logs/baseline_T3.log 2>&1
```

**Point:** all fixes OFF (`[R0 T0 O0 D0 A0]`). This is the "before" picture —
tag counts, mode distribution, offset stats. You cannot tell whether a fix helps
without this to compare against. Run it once per track:

```bash
for t in trackT3 trackT4 trackT5; do
  python3 vision_stack/src/run_pipeline.py --frames vision_stack/frames/$t \
    > logs/baseline_$t.log 2>&1
done
```

### 3. Single-fix runs — the actual experiment

```bash
python3 vision_stack/src/run_pipeline.py --fix orientation_filt \
  --frames vision_stack/frames/trackT3 > logs/orientation_filt_T3.log 2>&1
```

**Point:** flip exactly one fix on, everything else off, same track as baseline.
The whole reason the fixes are independently toggleable is so you can attribute a
change to one specific fix. The five names (`--fix` is repeatable):

| `--fix` name | Fixes which failure mode | Log flag |
| --- | --- | --- |
| `roi_inset` | wall edges / far-field clutter entering lane ROI | R |
| `trapezoid_mask` | off-lane edges (poles, walls) surviving to contours | T |
| `orientation_filt` | poles misclassified as lane boundaries | O |
| `dashed_dilate` | dashed center line fragmenting into dropped contours | D |
| `anchor_halves` | both lane anchors chosen from the same ROI half | A |

Run each fix across all three tracks:

```bash
for fix in roi_inset trapezoid_mask orientation_filt dashed_dilate anchor_halves; do
  for t in trackT3 trackT4 trackT5; do
    python3 vision_stack/src/run_pipeline.py --fix $fix \
      --frames vision_stack/frames/$t > logs/${fix}_${t}.log 2>&1
  done
done
```

### 4. Combined-fix runs — only after singles

```bash
python3 vision_stack/src/run_pipeline.py --fix roi_inset --fix trapezoid_mask \
  --frames vision_stack/frames/trackT3 > logs/roi_trap_T3.log 2>&1
```

**Point:** test interactions once you know each fix's solo effect. Do singles
first — a combined run can't tell you which of the two did the work.

---

## Reading the logs

Each frame emits up to two lines. The flag prefix `[R0 T0 O0 D0 A0]` is on every
line so you can tell runs apart when diffing.

```
[R0 T0 O0 D0 A0] f=0000 TAGS=pole_misclassified(x13) ... offset=-0.0057 mode=two_boundary conf=0.53
f=0000 t=143.0ms offset=+0.2429 head=+7.29° drive=go stop_sign=F lane_mode=two_boundary imu_n=0 yaw=+0.0°/s
```

- **TAGS line** — only appears when a failure mode fired. `offset` here is the
  RAW Stage 5 value (`compute_lane_offset`).
- **nav line** — every frame. `offset` here is the Phase 3 value that drives the
  motors. **These two offsets should track each other; right now they don't (see
  Known Issues).**

### Useful greps

```bash
# tag totals for a run
grep -oP 'pole_misclassified\(x\K\d+' logs/baseline_T3.log | paste -sd+ | bc

# mode distribution
grep -oP 'lane_mode=\K\w+' logs/baseline_T3.log | sort | uniq -c

# baseline vs fix, side by side
grep -c pole_misclassified logs/baseline_T3.log
grep -c pole_misclassified logs/orientation_filt_T3.log

# did the intersection detector fire, and at what ratio
grep INTERSECTION logs/baseline_T3.log | head

# frames over the timing budget
grep -c 'budget exceeded' logs/baseline_T3.log
```

**What "the fix worked" looks like:** its target tag count drops sharply between
`baseline_T3.log` and `<fix>_T3.log`, without wrecking the mode distribution
(e.g. `orientation_filt` should cut `pole_misclassified` toward zero while
keeping `two_boundary` frames roughly stable). If the tag drops but
`two_boundary` collapses into `none`/`left_only`, the fix is over-rejecting.

---

## Known issues (as of the first baseline_T3 run)

These are outstanding — expect them in the logs until fixed.

1. **`nav_packet.lane_offset` never goes negative (HIGH PRIORITY).**
   `_adapt_detections_for_p3()` passes the absolute ROI centroid x to Phase 3,
   which documents `position_x` as a *signed offset from image center*. Phase 3
   never sees a negative offset, so the correctly-computed Stage 5 offset never
   reaches the motors. **Fix 5 runs will show no effect on the nav offset until
   this is resolved** — the RAW (TAGS-line) offset does change, the nav offset
   doesn't. Fix at the adapter (`position_x = x - roi_width/2`) or pass
   `lane_offset_result.offset` into the packet directly.

2. **Intersection threshold is 0.25, should be ~2.5.** At 0.25, ~28% of frames
   spuriously "would trigger" (`INTERSECTION_EDGE_RATIO_THRESH` in `config.py`).
   Observed ratios run 0.25–1.47; at 2.5, nothing fires — verify the bottom band
   actually contains stop-line edges before trusting either number.

3. **`output.avi` I/O inflates timing.** The `SAVE_VIDEO` writer's `cv2.imwrite`
   dominates the wall-clock max (316 ms spikes) and the ~5 s of runtime. Real
   per-frame *compute* is the ~16 ms median. Set `SAVE_VIDEO = False` in
   `run_pipeline.py` for clean timing numbers; leave it on when you want the
   visual overlay.

4. **Calibration TODOs still open** in `config.py`: trapezoid `corners`,
   `DashedDilateParams.kernel_h`, `OrientationFiltParams` angle, intersection
   band/threshold. Tag *counts* are trustworthy; the fixes' *tuning* is not until
   these are set against real overlays.

---

## Quick reference

```bash
# from repo root, always
cd ~/Personal_Workspace/Navilott_Vision

python3 vision_stack/src/smoke_test.py                      # 1. correctness gate
python3 vision_stack/src/run_pipeline.py \                  # 2. baseline
    --frames vision_stack/frames/trackT3 > logs/baseline_T3.log 2>&1
python3 vision_stack/src/run_pipeline.py --fix <name> \     # 3. one fix
    --frames vision_stack/frames/trackT3 > logs/<name>_T3.log 2>&1

# fix names: roi_inset trapezoid_mask orientation_filt dashed_dilate anchor_halves
# tracks:    trackT3 trackT4 trackT5   (no T2)
# flags in log: R=roi_inset T=trapezoid O=orientation D=dilate A=anchors
```
