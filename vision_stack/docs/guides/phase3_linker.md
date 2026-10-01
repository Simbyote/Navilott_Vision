# Checking Estimation (phase3_linker)

Runs frames through all three phases and reports what estimation decided, frame by frame, next to the Phase 2 input it came from. When a packet looks wrong, this shows whether the input was bad or the filtering was.

It is the debug twin of the main pipeline (`src/pipeline.py`): Phase 2 through `run_chain()`, Phase 3 through `TracedPhase3Processor` (`src/debugger/estimation_debug.py`), which runs the production Phase 3 stages, records every decision and times every stage. The tests hold its packets to the pipeline's. It prints to the console, writes a CSV and summary, and records the Phase 3 video (`p3_debug.avi`), shown in a window like `phase2_linker`'s. It doesn't drive the motors.

## Requirements

- Camera not in use by `phase2_linker`, `rpicam-hello` or a pytest run (`--camera` only)
- `--imu` only: I²C enabled, MPU-6050 at 0x68, and the Adafruit MPU-6050 libraries installed
- `--encoders` only: the pigpio daemon running (`sudo pigpiod`) and the encoders wired as in `peripherals/drive.py`
- Without either flag it runs anywhere, including a laptop

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
| `--imu` | Read the MPU-6050 through the sensing hub (`src/peripherals/sensing.py`, 100 Hz, grouped per frame, yaw flipped to + = right) and feed it to Phase 3 |
| `--encoders` | Start the wheel encoders and pass each wheel's counts per second through to the packet |
| `--gyro-bias DPS` | Gyro Z at standstill, subtracted before integrating, in the flipped + = right frame (about +1.0 on this robot). `--imu` doesn't calibrate, so pass it here. Applied on top of `MEASURED_ESTIMATION` (`src/config.py`), like `--cm-per-px` |
| `--cm-per-px S` | Hand-measured ground scale; fills `lane_offset_cm` |
| `--hsv PATH` | HSV ranges to use instead of `calibration/hsv_ranges.json`, which `MEASURED` already loads |
| `--print-every N` | Status line every N frames (default once a second); 0 prints only events |
| `--verbose` | Also print Phase 3's per-frame log |
| `--no-video` | Don't record `p3_debug.avi` / `.csv` (text and timing only) |
| `--no-display` | Force headless; the window also falls back on its own over ssh |
| `--scale N` | Magnify the video and window |
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

An event line appears whenever `lane_status`, `drive_state`, `stop_sign_detected` or `stop_line_detected` changes. A lane change also shows the Phase 2 mode that caused it:

```
>>> f0413 lane: vision -> hold (p2 mode=none)
```

## 4. Reading the results

```
runs/p3_<YYYYMMDD_HHMMSS>/
    p3.csv          every frame: timings, the Phase 2 lane and stop-line input, the packet, Phase 3's log
    p3_debug.avi    the Phase 3 video, every frame (unless --no-video)
    p3_debug.csv    every frame's Phase 3 decisions (see below)
    summary.txt     the run's report, also printed at the end
    still_phase3_NNN.png   stills saved with s
```

`summary.txt`:

| Section | Look for |
| --- | --- |
| `over budget` | Frames where P2 + P3 exceeded one frame period (50 ms at 20 FPS) |
| Timing | Mean, p50, p95 and max per phase, plus `render` when the video is on |
| Lane status | Share of frames on `vision`, `hold`, `stale`; longest hold and stale runs |
| Phase 2 lane mode | How often each mode occurred |
| Lane offset while on vision | Mean, standard deviation, min and max |
| Transitions | How many times each packet field changed |
| `[TIMING]` per stage | Median and p95 of every Phase 2 stage, every Phase 3 stage (`p3_lane`, `p3_traffic`, ...) and `render`, then the total as FPS. `render` is marked `(not in total)` and is never in the P2+P3 budget: the robot doesn't pay it. Phase 3 stages run in microseconds, so they print with three decimals. These are the debug twin's times; the production stages do the same work minus the recording |
| `[PHASE 3]` | Lane status share, lane frames not accepted by reason, frames held while the raw offset was a measurement, detections below the gate, frames with a held stop-line distance |

In `p3.csv`, columns starting with `p2_` are the Phase 2 input and the rest are the packet. `p2_stop_line_px` is Phase 2's distance to the nearest stop line (blank when none); `stop_line_detected` and `stop_line_distance_px` are Phase 3's vote and held distance. `p2_stop_line_cm` and `stop_line_distance_cm` are the same in floor cm, blank without a ground homography (`calibrate_ground.md`). The status line ends with `line=<px>`, or `line=<px>/<cm>cm` with one. `p3_log` says why a frame was treated as a dropout (mode, jump, missing yaw). The last two columns, `left_wheel_cps` and `right_wheel_cps`, are each wheel's encoder counts per second with `--encoders` (+ = forward), and 0.0 without it or while the wheel is stopped.

### The Phase 3 video

Each frame of `p3_debug.avi`, top to bottom:

- **Camera frame** with the lane overlay from `phase2_linker`'s `run.avi`. Each traffic light and stop sign Phase 2 handed over is boxed green if it passed Phase 3's confidence gate and amber if not, labeled `conf >= gate` or `conf < gate`. The header's right corner has the timestamp and P1 / P2 / P3 times, and the previous frame's render time.
- **Lane bar**, colored by status (green `vision`, amber `hold`, red `stale`): the raw offset Phase 2 measured as a hollow ring (red when Phase 3 refused it) and the filtered offset as a yellow dot, on [-1, 1]. The line above says the hold counter (`hold 3/7`) and, when the frame wasn't accepted, why: `no_result`, `unusable_mode`, or `jump_gate` with the jump and the limit.
- **Votes**: traffic, stop sign and stop line. Each vote's buffer is a row of cells, oldest on the left (green / amber / red for go / caution / stop; red for a stop sign; white for a stop line), then this frame's raw vote and the voted state. The stop-line row says `seen <px>` or, in amber, `held <px>` when the vote stands on an earlier frame's distance.
- **Timeline**, the last 5 s: the status band, the raw offset in gray (red dots where it was refused) and the filtered offset in yellow, then drive state, stop sign and stop line tracks. A flat yellow line across an amber band while the gray line moves is the output held while the input moved.

`p3_debug.csv` has the same decisions per frame: lane mode, raw offset, accepted, reason, jump, EMA before and after, missed count, status; heading reset; each detection as `label:conf:pass|gated`; each vote's buffer (`G`/`C`/`S`, `T`/`F`, oldest first), raw vote and state; stop line seen, measured, held and reported distances.

## 5. Bench checks

Each takes about a minute with the robot on the course.

**Offset sign.** Place the robot in a lane, shifted toward the right line, and run `--camera --limit 100`. `off` should be positive. Shift it left: negative. If it's the other way round, stop and fix that before any tuning.

**Noise floor.** Park the robot centered and still, run `--camera --limit 300`. The `std` under "lane offset while on vision" is the measurement noise. Differences smaller than that between two settings aren't meaningful.

**IMU yaw sign.** Run `--camera --imu --print-every 1`, cover the lens so vision drops to `hold`, and turn the robot right by hand. `hd` should go positive. If it goes negative, `IMU_YAW_SIGN` in `params.py` is wrong for this robot's mount; see `vision_stack/phase3_estimation.md`, "Sign conventions".

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
| `pigpio daemon not reachable` with `--encoders` | Start it: `sudo pigpiod` |
| Wheel columns stay 0.0 with `--encoders` while driving | Check the encoder wiring against the pins in `peripherals/drive.py`; `pytest --hardware -k drive` spins each wheel and checks it counts |
| `lane_offset_cm` empty | Set `--cm-per-px` |
| `drive_state` always `go` | No light passed the HSV ranges or Phase 3's confidence gate; check `p2_traffic` in `p3.csv`, then the ranges |
| Mostly `stale` from the start | Phase 2 isn't finding usable boundaries; watch the same stretch with `phase2_linker` |
| `No module named src` or `cv2` | Not in `vision_stack/`, or environment not active |

Report any other error with the full terminal output.
