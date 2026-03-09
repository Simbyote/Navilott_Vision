# AutoBot — Perception System

Autonomous street navigation using a resource-constrained computer vision pipeline on Raspberry Pi Zero 2 W.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware Constraints](#hardware-constraints)
- [Estimated Memory Budget](#estimated-memory-budget)
- [Global Performance Budgets](#global-performance-budgets)
- [Data Flow Through the System](#data-flow-through-the-system)
- [Documentation Index](#documentation-index)

---

## Project Overview

Navilott is a fully autonomous robot that navigates a street map course without human intervention. The perception system is responsible for detecting:

- Lane boundaries
- Traffic lights (red / yellow / green state)
- Stop signs
- (Object detection may be an additional feature in the future)

The architecture is undecided between **high-level perception** on the Pi and **low-level motor control**
on an MCU or that everything runs on the Pi:

### Raspberry Pi Zero 2 W + MCU Architecture

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│  High-Level Computer            │     │  Low-Level Controller            │
│  Raspberry Pi Zero 2 W          │────▶│  Microcontroller (MCU)           │
│  Perception + Decision Logic    │     │  Motor Drivers + PID Steering    │
└─────────────────────────────────┘     └──────────────────────────────────┘
```

### Raspberry Pi Zero 2 W only Architecture

```
┌─────────────────────────────────┐
│  High-Level Computer            │
│  Raspberry Pi Zero 2 W          │
│  Perception + Decision Logic    │
│ Motor Drivers + PID Steering    │
└─────────────────────────────────┘
```

Reasons to split the architecture between an MCU and the Pi would be to ensure that real-time motor control
is never stalled by the perception workload made on the Pi. However, alternatives suggest a workaround may be
possible. More details are discussed in [Architecture & Design Decisions](docs/architecture.md).

---

## Hardware Constraints

| Component        | Specification                          |
|------------------|----------------------------------------|
| SBC              | Raspberry Pi Zero 2 W                  |
| RAM              | 512 MB                                 |
| OS               | Debian Linux (Raspberry Pi OS Lite)    |
| Camera Sensor    | IMX219 (Pi Camera Module v2)           |
| Camera Interface | CSI-2                                  |
| Runtime Stack    | Python 3, OpenCV, GStreamer, libcamera |

The 512 MB RAM constraint is the primary design driver. The pipeline must operate as a **sequential streaming pipeline** — no
unnecessary frame copies, no bulk buffering. Digital frames are organized in a ring buffer with unique timestamps for real-time
processing.

---

## Estimated Memory Budget

> *All figures are approximate.*

### Single Process Footprint

| Component                          | Estimated Usage  |
|------------------------------------|------------------|
| Camera frame (640×480 BGR)         | ≈ 0.92 MB        |
| Working frame buffers              | ≈ 2.8 MB         |
| Edge detection buffers             | ≈ 1.0 MB         |
| HSV masks                          | ≈ 1.0 MB         |
| Detection structures               | < 1 MB           |
| OpenCV runtime                     | ≈ 50 MB          |
| Python interpreter                 | ≈ 30 MB          |
| GStreamer pipeline                 | ≈ 20 MB          |
| **Estimated pipeline footprint**   | **≈ 110–150 MB** |

**Note:** The 3 working frame buffer estimate assumes worst-case simultaneous copies across pipeline stages. Disciplined use of in-place OpenCV operations may reduce
this to 1–2 live buffers at any time.

### System-Level Budget

Linux OS services consume approximately 80–100 MB at idle on Raspberry Pi OS Lite. Combined with the pipeline footprint:

```
512 MB total RAM
 - 80–100 MB  OS + system services
 - 110–150 MB pipeline (single process)
─────────────────────────────────────
≈ 260–320 MB remaining headroom
```

### Two-Process Budget (if CPU contention mitigation Option 1 is adopted)

Running perception and navigation as separate processes doubles the Python + OpenCV interpreter overhead:

```
                        Single Process    Two Processes
OS + system services    ~80–100 MB        ~80–100 MB
Python + OpenCV + GStreamer ~100 MB       ~200 MB
Frame buffers + working ~6 MB             ~6 MB
──────────────────────────────────────────────────────
Estimated total         ~190 MB           ~290 MB
Headroom (from 512 MB)  ~320 MB           ~220 MB
```

220 MB headroom with two processes is still workable, but leaves less margin for TFLite (≈50–80 MB) if Alternative 2 is later adopted. Profile before committing.

---

## Global Performance Budgets

| Metric                   | Target                                        |
|--------------------------|-----------------------------------------------|
| Target frame rate        | 30 FPS                                        |
| Max end-to-end latency   | ≤ 33.3 ms/frame                               |
| Per-stage processing     | < 10–15 ms ideal                              |
| CPU usage (light stages) | < 40%                                         |
| CPU usage (heavy stages) | 40–70% moderate, 70–90% heavy, >90% dangerous |

**Processing budget classification (per frame at 30 FPS):**

```
0–10 ms   → Very good
10–20 ms  → Usable
20–30 ms  → Borderline
>33.3 ms  → Inconsistent / budget exceeded
```

**Buffering policy:** `appsink drop=true max-buffers=1 sync=false`
Old frames are dropped rather than queued. This prevents latency accumulation and ensures the pipeline always operates on the
most recent frame.

---

## Data Flow Through the System

```
                    +------------------------------------------------------+
                    |           Raspberry Pi Zero 2 W                     |
  +----------+ CSI  |  +----------+  BGR  +----------+  detections        |
  |  IMX219  |-----▶|  | Phase 1  |------▶| Phase 2  |-------------+      |
  |  Sensor  |      |  | Camera   | frames| Vision   |             |      |
  +----------+      |  | Acquis.  |       | Percept. |             v      |
                    |  +----------+       +----------+       +----------+  |
                    |                                        | Phase 3  |  |
                    |  Ring buffer (1 frame max)             | Nav Sig. |  |
                    |  Drop policy: drop=true                | Process. |  |
                    |  No frame copies                       +----+-----+  |
                    +----------------------------------------------------+-+
                                                                 |
                    +--------------------------------------------+---------------------+
                    |                                                                  |
                    v  [Option A]                               [Option B]             v
         +------------------------+                          +------------------------------+
         |  UART -> MCU           |                          |  Pi GPIO / pigpio daemon     |
         |  PID loop @ ~1 kHz     |                          |  Python PID loop             |
         |  HW PWM, no jitter     |                          |  Motor drivers (direct GPIO) |
         |  Fault isolated        |                          |  Timing: OS best-effort      |
         +------------------------+                          +------------------------------+
```

### Image Processing Pipeline (High-Level)

```
Frame Capture Loop
      │
      ▼
Image Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
Detection Algorithms
      │
      ▼
Feature Fusion
      │
      ▼
Navigation Output
```

---

## Documentation Index

| Document | Contents |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | Pipeline Phases 1–3 in full detail (camera acquisition, vision perception, navigation signal processing) |
| [docs/architecture.md](docs/architecture.md) | Pi-only vs Pi+MCU output interface, CPU contention mitigation options, perception fallback hierarchy, control loop timing |
| [docs/operations.md](docs/operations.md) | Calibration procedure, telemetry & logging format, failure mode analysis |
| [docs/development.md](docs/development.md) | Development roadmap and benchmark & testing reference |
