# Checking Estimation (phase3_linker)

Runs frames through all three phases headless and reports what estimation decided, frame by frame, next to the Phase 2 input it came from. When a packet looks wrong, this shows whether the input was bad or the filtering was.

It prints to the console and writes a CSV and summary. It doesn't draw or record video; for that, use `phase2_linker.md`. It doesn't drive the motors.

## Requirements

- Camera not in use by `phase2_linker`, `rpicam-hello` or a pytest run (`--camera` only)
- `--imu` only: I²C enabled, MPU-6050 at 0x68, and the Adafruit MPU-6050 libraries installed. Without `--imu` it runs anywhere, including a laptop

## 1. Setup

```
cd ~/Navilott_Vision/vision_stack
source .venv/bin/activate
```

On the Pi, set the clock first. Run folders are named by timestamp and are wrong until it's fixed.

```
timedatectl                                   # check Local time and Time zone
sudo timedatectl set-ntp false                # required before set-time
sudo timedatectl set-time "YYYY-MM-DD HH:MM:SS"
timedatectl                                   # confirm
```

## 2. Run

One source is required:

```
python3 -m src.phase3_linker --camera                   # live
python3 -m src.phase3_linker --video clip.mp4           # a recorded video
python3 -m src.phase3_linker --frames DIR               # images, sorted by filename
```

Stop with Ctrl-C; the CSV and summary are still written.

Common variations:

```
python3 -m src.phase3_linker --camera --imu --limit 600         # 30 s with the IMU
python3 -m src.phase3_linker --frames src/tests/data/frames     # repeatable replay
python3 -m src.phase3_linker --camera --print-every 1           # every frame
python3 -m src.phase3_linker --camera --print-every 0           # events only
```

| Option | Effect |
| --- | --- |
| `--camera` / `--video PATH` / `--frames DIR` | Frame source (pick one) |
| `--limit N` | Stop after N frames |
| `--fps N` | Capture rate for `--camera`; replay rate for `--frames`. Videos default to their own rate |
| `--width`, `--height` | Capture size; defaults to `params.py` (480×270) |
| `--imu` | Start the MPU-6050 and feed it to Phase 3 |
| `--gyro-bias DPS` | Gyro Z at standstill, subtracted before integrating. `--imu` doesn't calibrate, so pass it here |
| `--cm-per-px S` | Hand-measured ground scale; fills `lane_offset_cm` |
| `--hsv PATH` | Calibrated HSV ranges; switches the color branch on |
| `--print-every N` | Status line every N frames (default once a second); 0 prints only events |
| `--verbose` | Also print Phase 3's per-frame log |
| `--out DIR` | Output folder instead of `runs/p3_<timestamp>` |

With no arguments it prints the full help.

Replays are deterministic: the same frames give the same packets. Their timestamps come from the frame rate, not a clock, so compare Phase 3 settings on the same replay rather than across live runs.

## 3. Reading the console

A status line, left to right:

```
f0412 t=20.61s P1=6.1 P2=31.8 P3=0.9ms | two_boundary L=150 R=290 n=2 c=.82 raw=+0.045 | off=+0.031 vision hd=+0.0 | go stop=F
└──── frame and timing ───────────────┘ └──────── Phase 2 lane input ──────────────────┘ └──── packet ─────────────┘ └ votes ┘
```

| Field | Meaning |
| --- | --- |
| `P1`, `P2`, `P3` | ms in capture, Phases 2 and 3. `P1` on a live camera includes waiting for the next frame |
| mode, `L`, `R`, `n` | Phase 2 lane mode, left and right anchor x (lane-ROI px), usable boundaries |
| `c`, `raw` | Phase 2 confidence and unfiltered offset |
| `off` | The packet's filtered offset. **+ = robot right of lane center** |
| status | `vision`, `hold` or `stale`; navigation steers only on the first two |
| `hd` | Degrees turned since the last `vision` frame (IMU only) |
| `go`, `stop=` | Voted traffic state and stop-sign flag |

An event line appears whenever `lane_status`, `drive_state` or `stop_sign_detected` changes. A lane change also shows the Phase 2 mode that caused it:

```
>>> f0413 lane: vision -> hold (p2 mode=none)
```

## 4. Reading the results

```
runs/p3_<YYYYMMDD_HHMMSS>/
    p3.csv          every frame: timings, the Phase 2 lane input, the packet, Phase 3's log
    summary.txt     the run's report, also printed at the end
```

`summary.txt`:

| Section | Look for |
| --- | --- |
| `over budget` | Frames where P2 + P3 exceeded one frame period (50 ms at 20 FPS) |
| Timing | Mean, p50, p95 and max per phase |
| Lane status | Share of frames on `vision`, `hold`, `stale`; longest hold and stale runs |
| Phase 2 lane mode | How often each mode occurred |
| Lane offset while on vision | Mean, standard deviation, min and max |
| Transitions | How many times each packet field changed |

In `p3.csv`, columns starting with `p2_` are the Phase 2 input and the rest are the packet. `p3_log` says why a frame was treated as a dropout (mode, jump, missing yaw).

## 5. Bench checks

Each takes about a minute with the robot on the course.

**Offset sign.** Place the robot in a lane, shifted toward the right line, and run `--camera --limit 100`. `off` should be positive. Shift it left: negative. If it's the other way round, stop and fix that before any tuning.

**Noise floor.** Park the robot centered and still, run `--camera --limit 300`. The `std` under "lane offset while on vision" is the measurement noise. Differences smaller than that between two settings aren't meaningful.

**IMU yaw sign.** Run `--camera --imu --print-every 1`, cover the lens so vision drops to `hold`, and turn the robot right by hand. `hd` should go positive. If it goes negative, the IMU sign doesn't match what `estimation.py` expects; see `vision_stack/phase3_estimation.md`.

**Gyro bias.** With the robot still and the lens covered, `hd` should stay near 0. If it drifts steadily, note how fast (degrees per second) and pass that value as `--gyro-bias`.

## 6. Results

```
tar -czf p3_results.tgz runs/p3_<YYYYMMDD_HHMMSS>
```

Send the archive with the terminal output and which source it ran on.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Help text instead of a run | No source given; add `--camera`, `--video` or `--frames` |
| `source error: ...` | Camera busy or missing, or the file or folder doesn't exist. `rpicam-hello --list-cameras` must list imx290 |
| `No module named board` / `adafruit_mpu6050` with `--imu` | IMU libraries not installed in this environment |
| IMU error at startup with `--imu` | Enable I²C; check wiring and address 0x68 (`i2cdetect -y 1`) |
| `hd` always 0.0 | Normal while on `vision`; without `--imu` it never moves |
| `lane_offset_cm` empty | Set `--cm-per-px` |
| `drive_state` always `go` | Expected: the color branch is off without `--hsv` |
| Mostly `stale` from the start | Phase 2 isn't finding usable boundaries; watch the same stretch with `phase2_linker` |
| `No module named src` or `cv2` | Not in `vision_stack/`, or environment not active |

Report any other error with the full terminal output.
