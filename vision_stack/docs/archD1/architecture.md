# Architecture & Design Decisions

> Output interface options, CPU contention mitigation, perception fallback hierarchy, and control loop timing.

---

## Table of Contents

- [Output Interface to Navigation Controller](#output-interface-to-navigation-controller)
- [Control Loop Timing](#control-loop-timing)
- [CPU Contention Mitigation](#cpu-contention-mitigation)
- [Perception Implementation Fallback Hierarchy](#perception-implementation-fallback-hierarchy)

---

## Output Interface to Navigation Controller

Two architecture options are under consideration for motor control. The tradeoff is cost and complexity vs. timing reliability.

---

### Option A: Pi Zero 2 W Only Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Raspberry Pi Zero 2 W               │
│                                                     │
│  Perception (Phases 1–3)                            │
│       │                                             │
│       ▼                                             │
│  Decision Logic                                     │
│       │                                             │
│       ▼                                             │
│  Motor Control (Python PID loop)                    │
│       │                                             │
│       ▼                                             │
│  GPIO / PWM  ──────────────────▶  Motor Drivers    │
└─────────────────────────────────────────────────────┘
```

**How it works:** The Pi handles the full stack — perception, decision, and motor output — in a single Python application. Motor commands are issued directly via GPIO
PWM.

**Implementation notes:**

- Use `pigpio` daemon instead of `RPi.GPIO`. It drives PWM via DMA, which is significantly more timing-stable than software PWM from Python.
- Run the PID loop in a dedicated high-priority Python thread, separated from the perception loop.
- Set the perception thread and motor thread to different `os.nice()` priorities to reduce scheduler contention.
- Accept that motor timing will not be deterministic — Linux scheduler preemption can introduce 5–20 ms hiccups under CPU load.

**Known risks:**

| Risk                    | Description                                                                                                                          |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| CPU contention          | Perception pipeline consumes significant CPU. Adding a PID loop on the same core increases the risk of frame drops and motor stalls. |
| Timing jitter           | Python + Linux cannot guarantee sub-millisecond loop timing. Steering may be noticeably less smooth, especially at higher speeds.    |
| Single point of failure | A perception crash takes down motor control. No hardware fallback.                                                                   |

---

### Option B: Pi Zero 2 W + Dedicated MCU Architecture

```
┌─────────────────────────────────────┐
│         Raspberry Pi Zero 2 W       │
│                                     │
│  Perception  →  Decision Logic      │
│  (Phases 1–3)   (lane offset,       │
│                  heading error,     │
│                  traffic state)     │
└──────────────────┬──────────────────┘
                   │  UART (9600–115200 baud)
                   │  Packet: [START | type | value | checksum | END]
                   ▼
┌─────────────────────────────────────┐
│         MCU (e.g. STM32 / Arduino)  │
│                                     │
│  PID Steering Loop  (e.g. 1 kHz)    │
│  PWM Motor Output                   │
│  Stop / Go Logic                    │
└─────────────────────────────────────┘
```

**How it works:** The Pi runs perception and decision logic, then transmits a compact navigation packet over UART. The MCU receives the packet and executes motor
control independently at a fixed hardware-timed rate.

**UART packet schema:**

```
[ 0xAA | type (1B) | value (4B, float) | checksum (1B) | 0xFF ]

Types:
  0x01 — lane_offset     (meters, signed float)
  0x02 — heading_error   (degrees, signed float)
  0x03 — traffic_state   (0=go, 1=caution, 2=stop)
  0x04 — speed_target    (0.0–1.0 normalized)
```

**Implementation notes:**

| Factor               | Impact                                                                                                                                |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Linux is not an RTOS | The Pi OS scheduler can preempt any Python thread with no timing guarantee. A PID loop in Python can jitter by 5–20 ms unpredictably. |
| Hardware PWM on MCU  | MCU generates PWM signals in dedicated hardware timers — zero CPU cost, zero jitter.                                                  |
| Perception isolation | Motor control failures cannot crash the perception process, and vice versa.                                                           |
| Cost                 | STM32F103 ("Blue Pill") ~$2–3. Arduino Nano ~$3–5. Negligible addition to BOM.                                                        |

---

### Comparison Summary

| Criterion              | Option A: Pi Only          | Option B: Pi + MCU         |
|------------------------|----------------------------|----------------------------|
| Motor timing accuracy  | Best-effort (OS scheduler) | Hardware-guaranteed        |
| PID loop rate          | ~100–200 Hz realistic (Pi) | Up to 1 kHz (MCU)          |
| Fault isolation        | No                         | Yes                        |
| Additional cost        | None                       | ~$2–5                      |
| Wiring complexity      | None                       | +1 UART connection         |
| Recommended for        | Prototype / demo only      | Competition / reliability  |

**Note:** Option B is recommended for real-world applications.

---

## Control Loop Timing

End-to-end timing from frame capture to motor command output. All stages must complete within the 33.3 ms frame budget.

### Option A: Pi + MCU

```
Frame capture (GStreamer appsink)
        │  ~1–2 ms
        ▼
Preprocessing (undistortion, equalization, ROI crop)
        │  ~3–5 ms
        ▼
Feature Extraction (HSV threshold + Canny + contours)
        │  ~8–12 ms
        ▼
Detection & Feature Fusion
        │  ~3–5 ms
        ▼
Phase 3 Filtering & State Estimation
        │  ~2–3 ms
        ▼
UART packet transmission (~20 bytes @ 115200 baud)
        │  ~2 ms
        ▼
MCU receives packet → PID update → PWM output
        │  < 1 ms (hardware-timed)
        ▼
Motor response
─────────────────────────────────────
Total Pi budget:   ~19–27 ms   (target ≤ 33.3 ms)
MCU PID loop:      1 kHz independent of Pi cycle
```

### Option B: Pi Only

```
Frame capture → Preprocessing → Feature Extraction → Detection → Fusion → State Estimation
        │  ~19–27 ms (same as above)
        ▼
Python PID loop (same process or high-priority thread)
        │  ~1–2 ms
        ▼
pigpio DMA PWM output
        │  ~1 ms
        ▼
Motor response
─────────────────────────────────────
Total budget:   ~21–30 ms   on average
Jitter risk:    OS scheduler can add 5–20 ms unpredictably
```

**Key difference:** In Option A the MCU PID loop runs at 1 kHz continuously regardless of Pi frame rate. In Option B, motor updates only happen when the Pi completes a
full perception cycle.

---

## CPU Contention Mitigation

> *If timing budget violations are observed during integration testing, this section is a good place to start*

The perception pipeline (Phases 1–3) is CPU-heavy. If navigation decision logic runs on the same Python process, it competes for
CPU time and can cause perception cycles to spike unpredictable. A consistent 20 ms frame time becoming an occasional 45 ms
spike is worse than averaging 25 ms, because Phase 3 temporal filtering receives unevenly spaced detections.

An alternative that is not viable is the Python GIL (Global Interpreter Lock). The GIL prevents CPU-bound threads from running
truly in parallel, and for compute-heavy work like OpenCV, threading provides concurrency in structure only, with no real
performance benefit.

Three approaches are available if contention becomes a measured problem:

---

### Option 1: Separate Processes Pinned to Dedicated Cores (First approach, complex)

```
+-----------------------------+        +-----------------------------+
|  Process A                  |        |  Process B                  |
|  Perception (Phases 1-3)    | -----> |  Navigation Decision Logic  |
|  Pinned to Core 0/1         |  IPC   |  Pinned to Core 2/3         |
+-----------------------------+        +-----------------------------+
        |                                          |
        v                                          v
  Dedicated core,                          Dedicated core,
  no GIL contention                        no GIL contention
```

- True parallel execution across Pi's 4 cores
- IPC overhead (Unix socket or `multiprocessing.Queue`) is ~0.1–0.5 ms per packet. This is negligible against 33.3 ms budget
- **Caveat:** Two Python + OpenCV interpreter instances may consume 160–240 MB RAM combined. On 512 MB this needs to be profiled before committing to this approach.

---

#### Core Allocation

Core 0 — OS, drivers, libcamera, GStreamer
Core 1 — Vision pipeline (Phase 2 heavy work)
Core 2 — Phase 3 filtering + navigation signal output
Core 3 — Motor control loop

---

### Option 2: Time-Slicing (Second approach, simple)

```

+------------------+     +------------------+     +------------------+
|  Perception      | --> |  Navigation      | --> |  Perception      | --> ...
|  (one frame)     |     |  (one decision)  |     |  (next frame)    |
+------------------+     +------------------+     +------------------+

```

- Single process, fully sequential, zero concurrency overhead
- Navigation update rate is tied directly to frame rate. At 30 FPS this will get 30 navigation decisions/sec
- Perception timing is fully predictable since nothing else runs during a frame cycle
- Simplest to implement and debug

---

### Option 3: Separate Processes + Pico MCU (Third approach, best)

If Option A hardware (Pi + Pico) is adopted, much of the navigation decision pressure moves off the Pi entirely. The Pi outputs a
validated packet and the Pico handles PID and motor timing. This reduces the Pi's post-perception workload to packet formatting
and transmission rather than juggle both tasks.

In this configuration CPU contention becomes largely a non-issue and no software mitigation is needed.

---

### When should mitigation be applied?

Profile before restructuring. Run the full pipeline under the CPU & Memory Observation benchmark (see [Benchmark & Testing Reference](development.md#benchmark--testing-reference)) with both perception and navigation active. If average CPU stays below 70% and per-frame timing stays under 33.3 ms, no mitigation is needed.

| Observation                         | Action                                         |
|-------------------------------------|------------------------------------------------|
| Avg CPU < 70%, no timing violations | No change needed                               |
| Occasional spikes > 33.3 ms         | Try Option 2 (time-slicing) first              |
| Consistent timing violations        | Try Option 1 (separate processes), profile RAM |
| RAM pressure + timing violations    | Adopt Pi + Pico (Option 3)                     |

---

## Perception Implementation Fallback Hierarchy

> *If the classical CV pipeline proves insufficient, escalate through these alternatives.*

The primary design goal is a fully self-contained embedded pipeline. External dependencies introduce latency, failure modes, and complexity that are difficult to
manage on a constrained autonomous system.

---

### Alternative 1: Classical CV (Current Target Approach)

The pipeline as documented — HSV thresholding, Canny edge detection, contour extraction, homography. Runs entirely on-device with no external dependencies.

**Note:** Most failure modes in classical CV are tuning problems, not fundamental capability limits:

- Poor detection under varied lighting → tune HSV ranges per lighting condition, add histogram equalization
- Fragile contour detection → adjust Canny thresholds, add morphological operations (dilation/erosion) to clean up edges
- High false positive rate → tighten ROI crop, add blob size filtering, increase confidence threshold in Phase 3

RAM usage is low and latency is predictable. This is the correct solution for this hardware class if it can be made to work.

---

### Alternative 2: On-Device Lightweight Inference (TensorFlow Lite Approach)

If classical CV proves too fragile for reliable stop sign or traffic light detection under real course conditions, a lightweight neural model can be added for those
specific detections only — not as a full replacement of the pipeline.

**TensorFlow Lite** is designed for exactly this hardware class. A MobileNetV2-based detector quantized to INT8 can run on the Pi
Zero 2 W within the latency budget. TensorFlow lite would replace the object detection segment of the pipeline:

```

Classical CV pipeline (lanes, edges)        ← keep as-is
        +
TFLite model (stop signs, traffic lights)   ← targeted replacement only
        |
        v
Feature Fusion (Phase 2) — unchanged

```

**Constraints to be aware of:**

- A MobileNetV2 INT8 model at 640×480 input will consume ~50–80 MB RAM — affordable, but needs to be profiled alongside the rest
of the pipeline
- Inference time on Pi Zero 2 W is roughly 80–150 ms at full resolution. The input must be downscaled to the ROI crop only to
stay within budget
- No network dependency — runs fully on-device, latency profile remains predictable
- Model must be trained or fine-tuned on representative data. A generic COCO-trained model may not generalize well to the course
environment

---

### Alternative 3: External AI Offload (Last Resort Approach)

Offloading inference to an external server (local network or cloud API) introduces dependencies that are fundamentally incompatible with reliable real-time navigation.

**Known high risks:**

| Factor                          | Impact                                                              |
|---------------------------------|---------------------------------------------------------------------|
| Network round-trip (local WiFi) | +5–20 ms baseline, spikes unpredictable                             |
| Pi Zero 2 W WiFi antenna        | *Weak*. real-world interference causes latency variance             |
| 33.3 ms budget                  | *Already tight*. network overhead alone may consume 30–60% of it    |
| Single point of failure         | Network drop = no perception = robot stops or acts on stale data    |
| Cloud API                       | Completely inappropriate for real-time control. Latency 100–500+ ms |

A local inference server (e.g. a laptop on the same network running a full model) is the least-bad version of this approach, but
still introduces network jitter that will show up as navigation instability.

**Only consider if:**

- Alternatives 1 and 2 have both been exhausted and failed
- The course environment is controlled enough that network reliability can be verified
- A fallback behavior is defined for when inference packets are late or missing

---

### Escalation Summary

```

Starting Point
    |
    v
[Alternative 1] Classical CV  ──── works? ──── DONE
    |
    | fails (fragile detections, tuning exhausted)
    v
[Alternative 2] TFLite on-device  ──── works? ──── DONE
    |
    | fails (RAM pressure, inference too slow)
    v
[Alternative 3] External AI offload  ──── high risk, document failure modes,
                                    define fallback behavior before deploying

```
