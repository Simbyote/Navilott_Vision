# Concepts

> The ideas the other docs use, explained once. Each entry says what the idea is, why Navilott needs it, and where it lives in the code, with the setting the robot runs.

The guides assume these; read an entry when a term stops you. Values are the robot's (`MEASURED` in `src/config.py`, `src/params.py`) unless a line says otherwise.

**Contents:** [Design choices](#design-choices) · [Images](#images) · [Finding shapes](#finding-shapes) · [Geometry](#geometry) · [Across frames](#across-frames) · [Motors and sensors](#motors-and-sensors) · [The camera path](#the-camera-path) · [Python and the Pi](#python-and-the-pi)

---

## Design choices

### Classical computer vision, not a learned model

There are two ways to make a camera find a lane line or a stop sign:

- **Classical computer vision:** write down what the thing looks like, as rules on pixels. "A lane line is a long bright stripe whose edges run up the image." "A lit lamp is a small round blob, bright enough to clip the camera to white." Each rule is a few image operations (a threshold, an edge detector, a shape test) with numbers chosen by measuring the course.
- **A learned model** (a neural network, such as an object detector): show it thousands of labelled pictures and let training find the rules.

Navilott uses classical vision, on purpose:

| | Classical | Learned model |
|---|---|---|
| **What it needs first** | Measurements of the course: tape width, lane width, lamp sizes | Thousands of labelled course images, and a training setup |
| **Speed on a Pi Zero 2 W** | A median frame of about 34 ms on the lane, with OpenCV on two threads (`params.py`, 2026-10-03) | The Pi has no accelerator; even small detectors take far longer per frame on its CPU |
| **When it's wrong** | Every decision traces to a number; fix the number on the course | The cause is inside the trained weights; the fix is more or different data |
| **Predictability** | Same frame, same answer, same time | Same answer, but the time and the failure cases are harder to bound |
| **What it gives up** | Robustness to things the rules didn't foresee: other lighting, other signs, a worn line | Learned models can generalize across lighting and wear, given enough varied training data |

It's a bet that pays because the course is **structured**: white tape on dark mats, one octagon shape, three lamp colors, a known lane width. Classical rules handle a world that regular. The cost is that rules tuned in one room can fail in another; see [Lighting](#lighting).

### Lighting

Every vision rule is a comparison against light: how bright, which color. Change the room's light and the same tape, sign or lamp reads differently. Nothing makes vision immune to that; a design picks the range of lighting it must handle and verifies at its edges, the way a chip is signed off at its process, voltage and temperature corners.

What helps, cheapest first:

1. **Control the camera.** `params.CAMERA_CONTROLS` is empty, so libcamera's auto exposure and auto white balance chase the room. In a dim room the auto exposure blew the lamps out to white (2026-10-04). Locking both at the venue makes "the lighting" a constant the thresholds were tuned for.
2. **Compare instead of measuring absolutes.** A fixed "brighter than 200" breaks when the room dims; "redder than the other two channels" ([Otsu](#thresholding-and-otsus-method) on redness, for the stop sign) or "clipped to white" (glow mode, for the lamps) holds up better.
3. **Calibrate against a known reference.** The white tape is on every frame; measuring it at the start would set the thresholds per run. Not built yet.
4. **Shade the camera.** A small hood cuts overhead glare off the mat.

`make routine-detect-range` and `make routine-lane-offset`, run under two or three lighting conditions with `--notes`, measure the range the robot actually handles.

---

## Images

### Pixels, BGR and grayscale

A frame is a grid of pixels, 480 wide and 270 high on this robot. Each pixel holds three numbers from 0 to 255: blue, green and red (OpenCV stores them in that order, **BGR**). **Grayscale** keeps one number per pixel, brightness only: the lane and stop-line detectors work on gray, since tape is about brightness, not color.

### Region of interest (ROI)

A rectangle cut out of the frame so a detector only looks where its target can be. Three on this robot (`perception/roi_crop.py`): the **lane ROI** (the bottom 30% of the frame, 5–95% across: the road just ahead), the **traffic ROI** (narrow, around where the light stands) and the **sign ROI** (upper right, where stop signs are posted). Cropping makes each detector faster and stops it seeing things in the wrong place, such as a red shirt in the sky mistaken for a stop sign.

### HSV

Another way to write a color: **hue** (which color, as an angle around the color wheel; 0–179 in OpenCV), **saturation** (how vivid, 0–255) and **value** (how bright, 0–255). It suits color detection because "red" is a range of hue whatever the brightness, where in BGR red's three numbers all shift as the light changes. Red sits at both ends of the hue wheel, so it takes two bands: `red_low` and `red_high` in `calibration/hsv_ranges.json`. Calibrated from the real lamps by `guides/calibrate_lamps.md`.

### Blur

Replacing each pixel with a weighted average of its neighbors (a **Gaussian blur**: nearer neighbors weigh more). It smooths out sensor noise and mat texture so edges and thresholds respond to real shapes, not specks. The kernel is the size of the neighborhood: `PreprocessParams.gray_kernel` and `color_kernel` (`perception/preprocess.py`).

### Histogram equalization

Stretches a gray image's brightness values to use the whole 0–255 range, raising contrast in a dull image. It's in `preprocess.py` but off (`PreprocessParams.equalize = False`): it also amplifies glare and changes every threshold downstream.

### Lens distortion and camera calibration

A wide lens bends straight lines near the edges of the image (**barrel distortion**). **Camera calibration** photographs a checkerboard from many angles and solves for the lens's **intrinsics** (focal length, image center) and **distortion coefficients**; the robot's strongest is k1 = −0.29. **Undistortion** then remaps each frame so straight lines are straight again, which the lane geometry and the ground homography rely on. `calibration/camera_calibration.json`, from `guides/calibrate_camera.md`.

---

## Finding shapes

### Thresholding and Otsu's method

**Thresholding** turns an image into a black-and-white **mask**: pixels above a value become 255 (part of the thing), the rest 0. The hard part is choosing the value.

**Otsu's method** chooses it automatically. It looks at the histogram of the image's values and picks the threshold that best splits them into two groups (formally, the one that maximizes the variance between the groups). It works well when the image really holds two populations, such as a red sign against a not-red background.

Its catch: **it always splits**, even an image with only one population. On a sign ROI with no sign, Otsu still finds a threshold in the noise, often around 1, and outlines specks as "red". That's why the stop sign's redness threshold is floored: `min_redness` (20) is the lowest Otsu is allowed to go (`perception/geometry.py`). The dimmest synthetic sign measured needed 22.

### Edges: Canny and Sobel

An **edge** is where brightness changes sharply, such as the side of a tape line. **Sobel** measures that change, the **gradient**, at each pixel, and its direction: a lane line's sides change left to right, a stop line's change top to bottom (`geometry.py` uses that to tell them apart).

**Canny** turns gradients into thin, clean edge lines. Its two thresholds, `Canny (80, 200)` on the lane ROI, work as **hysteresis**: a gradient above the high value (200) is surely an edge; one between the two (80–200) counts only if it connects to a sure edge; below the low value (80) it's discarded. That keeps a faint stretch of a real line while dropping isolated noise.

### Morphological close

An operation on a mask: grow the white areas (**dilate**), then shrink them back (**erode**). Small gaps and holes inside a shape fill in, while the shape's outline stays about where it was. The kernel's shape sets which gaps close: `close (9×3)` on lane edges bridges gaps along a line, `close (15×1)` joins a stop line's broken edge horizontally, `close (5×5)` fills a sign's mask.

### Contours and the gates

A **contour** is the outline of one white blob in a mask, as a list of points (OpenCV's `findContours`). Each candidate is then **gated**: measured and kept only if every measure is in range.

- **Area:** too small is noise, too big is something else.
- **Aspect ratio:** width over height; a lamp is about 1, a lane line much taller than wide.
- **Roundness:** the blob's area over the area of the smallest circle around it; 1.0 is a perfect disc (`color_branch._roundness`).
- **Vertices and solidity**, for the stop sign: the outline simplified to a polygon must have 8–9 corners (an octagon), and **solidity**, the blob's area over its convex hull's, rejects ragged shapes.

A blob that passes every gate gets a **confidence** from 0 to 1, and Phase 3 only counts it above a gate (0.40 for a light, 0.45 for a sign).

---

## Geometry

### Normalized lane offset

How far the robot sits from the lane's center, as a number from −1 to +1 of half the lane ROI's width; **+ means the robot is right of center**, so it steers left. It's in pixels of the image, not centimeters; `lane_offset_cm` converts it once the ground scale (`cm_per_px`) is set, which `make routine-lane-offset` measures.

### Homography

A 3×3 matrix that maps points on one flat plane to another, here **from the image to the floor**. A camera looking down at an angle sees the floor in perspective: near things big, far things small. For points that lie on the floor, a homography undoes that exactly, turning an image pixel into centimeters on the mat. It's fit once from known marks on the floor (`guides/calibrate_ground.md`) and gives the stop line's distance in cm (`perception/ground.py`). It's only right for things on the floor plane, and only for the camera pose it was fit at: move the mount and refit it.

---

## Across frames

### Exponential moving average (EMA)

A smoothing filter that keeps one number and blends each new measurement in:

> estimate = α · measurement + (1 − α) · previous estimate

α between 0 and 1 sets the trade-off: higher follows changes faster, lower smooths more. The lane offset uses α = 0.35 (`Phase3Config`); the battery voltage 0.3. The first measurement seeds the estimate directly. Navilott resets it when the lane goes stale, so the next measurement starts it fresh.

### Jump gate, hold and stale

A new lane offset more than `max_offset_jump` (0.5) from the estimate is treated as a dropout, not a move: a robot can't jump half a lane in one frame. During a dropout the last good offset is **held** for up to `hold_max_frames` (7 frames, about 350 ms at 20 FPS), then the lane goes **stale** and Navigation must not steer by it. Detail in `vision_pipeline/phase3_estimation.md`.

### Majority vote

The traffic light, the stop sign and the stop line each change state only when one value holds a **strict majority of the last 3 frames** (2 of 3); otherwise the previous state stands. One wrong frame can't flip a decision, at the cost of about one frame of delay (`estimation._Vote`).

### Gyro bias and heading

A gyro reports turn rate in degrees per second. At rest it should read 0 but doesn't: the reading at rest is its **bias**, −1.1 °/s on this robot (`GYRO_BIAS_DPS`). **Heading** is turn rate integrated over time, net of the bias:

> heading += (yaw rate − bias) × dt

An error in the bias integrates too: 0.2 °/s off is 12° a minute of drift. That's why heading is only trusted for seconds at a time (an intersection turn, a lane dropout) and why `make routine-imu-check` re-measures the bias.

---

## Motors and sensors

### PWM and duty cycle

A motor's speed is set by switching its supply fully on and off very fast, **pulse-width modulation**. The **duty cycle** is the fraction of each cycle that's on: 0.4 is on 40% of the time, and the motor turns as if it had 40% of the voltage. Navilott's PWM is hardware PWM on GPIO12 and GPIO13 at 1 kHz, through pigpio, so the waveform stays steady even when Python is late.

### H-bridge, short brake, coast and standby

An **H-bridge** is four switches around a motor that can connect either terminal to + or ground, so it can drive the motor either direction. The TB6612FNG holds two, one per wheel. It has three ways to stop:

- **Short brake** (`MotorController.brake()`): both terminals tied to the same rail. A spinning motor acts as a generator, and shorting it drives current through its own winding, which stops it quickly.
- **Coast:** the terminals left open; the wheel spins down on friction.
- **Standby:** the whole driver off. A run ends with a short brake for 0.3 s, then standby, so the robot stops before the outputs float (`main.halt`).

### Quadrature encoders

Each N20 motor carries an encoder with two outputs, C1 and C2, that pulse as the shaft turns, a quarter cycle apart (**in quadrature**). Which one leads gives the direction; counting the edges gives distance; counts per second give speed (`left_wheel_cps`, `right_wheel_cps`). pigpio counts the edges in callbacks (`drive.EncoderReader`).

### I²C

A two-wire bus (data SDA, clock SCL) that lets the Pi talk to several chips, each at its own address: the MPU-6050 IMU at 0x68 and the ADS1115 ADC at 0x48. One transfer at a time, so a slow read on one device delays the others; `make i2c-trace` records the bus.

### ADC and the voltage divider

The Pi can't measure a voltage itself, so an **ADC** (analog-to-digital converter, the ADS1115) does. The pack's up-to-12.6 V is too high for the ADC's input, so a **voltage divider** of two resistors (30 kΩ and 10 kΩ) feeds it a quarter: the code multiplies the reading by 4 (`DIVIDER_RATIO`, `diagnostics/battery.py`).

### pigpio and its daemon

A library for the Pi's GPIO pins that works through a background service, the **pigpio daemon** (`sudo pigpiod`, or `make pigpiod`). One daemon owns all the pins, so the motors, encoders, button and display can't conflict, and it provides the hardware PWM and the encoder callbacks.

### Watchdog

A timer that does something safe if a program goes quiet. If `drive()` hasn't been called for `MOTOR_WATCHDOG_S` (0.5 s) while the motors are driving, the watchdog brakes them: a frozen program can't leave the robot driving (`drive.py`; requirement R5).

---

## The camera path

From light to a frame in Python (`guides/diagnostics.md` has the measured times):

| Step | What it is |
|---|---|
| **IMX290** | The camera's image sensor, run in its 1920×1080 10-bit mode (`params.SENSOR_CONFIG`) |
| **CSI-2** | The fast serial link from the camera to the Pi |
| **Unicam** | The Pi's receiver for that link |
| **DMA** | Direct memory access: the receiver writes the image into memory without the CPU copying it |
| **CMA** | The contiguous memory pool camera buffers come from; running low stalls capture |
| **Bayer, debayer** | The sensor sees one color per pixel through a checkerboard of red, green and blue filters (a **Bayer** pattern); **debayering** fills in the other two colors at every pixel |
| **ISP** | The Pi's image signal processor: debayers, applies the white balance and color correction, and scales to 480×270; libcamera's algorithms set exposure and white balance from its statistics |
| **libcamera** | The Linux camera stack that drives the sensor and ISP |
| **GStreamer** | A media pipeline framework: `libcamerasrc` brings the frames in, `videoflip` turns them right way up, `videoconvert` makes BGR |
| **appsink** | GStreamer's handoff to the program; set to keep only the newest frame (`drop=true max-buffers=1`), so a slow frame skips stale ones instead of queueing them |

The frame's id and **timestamp** are made once, when Python pulls the frame from the appsink, and carried unchanged through every stage after.

---

## Python and the Pi

### The GIL

The **global interpreter lock** is a lock inside CPython, the standard Python interpreter: a thread must hold it to run Python code, so **only one thread runs Python at a time**, however many cores the Pi has. When several threads want it, the holder hands it over every 5 ms (`sys.getswitchinterval()`).

What that means here:

- **Threads that mostly wait are fine.** The sensor hub sleeps between its 10 ms reads and holds the GIL only briefly.
- **Python code doesn't get faster with threads.** Two threads both running Python take turns on one core's worth of Python.
- **C code that releases the GIL does run in parallel.** OpenCV and NumPy release it inside their heavy calls, and OpenCV runs its own worker threads across the cores (`OPENCV_THREADS`, 2 on the robot): a run on the lane measured 165% of a core for the process, 189% with four threads (`params.py`, 2026-10-03). GStreamer's capture thread is pure C and never needs the GIL.

So the main loop's Python is single-core, while most of the per-frame pixel work isn't. `guides/diagnostics.md` shows how to see the GIL's hand-offs in a run, and `py-spy --gil` shows who holds it.

### Threads and processes

A **thread** runs inside a program and shares its memory and its GIL. A **process** is a separate program with its own memory and its own GIL. Navilott is one process with a few threads (the main loop, the sensor hub, the motor watchdog, pigpio's callbacks, GStreamer's); the diagnostics recorder runs as a separate process so it can't slow the robot.

### Context switches

Each time a core stops running one thread and starts another. **Voluntary**: the thread blocked (a sleep, an I²C wait, waiting for the GIL). **Involuntary**: the scheduler took the core away. `guides/diagnostics.md` reads them per thread.

### Monotonic clock

A clock that only ever moves forward at a steady rate, unlike the wall clock, which jumps when the time is set. Frame timestamps, `dt` and every timeout use it (`time.monotonic()` or `time.perf_counter()`), so a clock correction mid-run can't fake a long or negative frame.

### Throttling and under-voltage

The Pi protects itself by slowing its CPU when it's too hot (**throttling**, from about 80 °C) or when its supply sags (**under-voltage**). Either slows every frame. `vcgencmd get_throttled` reports both, now and since boot; `make routine-power-profile` and the diagnostics recorder watch them.
