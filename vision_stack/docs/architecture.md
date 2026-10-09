# System Architecture

> What Navilott runs on, how it's wired, and why.

Navilott runs everything on one Raspberry Pi Zero 2 W: camera capture, perception, estimation, navigation and motor control. There's no second microcontroller. This doc records that decision, the hardware around it, and the risks that come with it.

Values come from the code, which is cited. The few still unmeasured are marked **TBD**.

---

## Decision: Pi Zero 2 W only

Two layouts were considered:

- **Pi only:** the Pi runs perception and drives the motors directly.
- **Pi + MCU:** the Pi sends a packet over UART to an STM32 or Arduino that runs a fast PID loop and the PWM.

**We went with Pi only.** The reasons:

- **The camera sets the control rate either way.** Steering corrections come from lane offset, which updates once per frame (20 Hz). An MCU running a 1 kHz loop would mostly repeat the last command between frames.
- **PWM is in hardware already.** The motor PWM runs on GPIO12 and GPIO13, the Pi's two hardware PWM channels, through the pigpio daemon. The duty cycle holds steady even when Python is late, so scheduler jitter delays a command change but doesn't disturb the waveform.
- **One board, one codebase.** No second firmware, no UART packet format, no checksums, and no link to debug between two processors.

The MCU option and its packet format are in `archD1/architecture.md` (Option B).

---

## Block diagram

```
              3×18650 pack (3S)
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
   MP1584EN buck            TB6612FNG VM
   → 5 V                    (motor supply)
          │                      │
          ▼                      │
┌──────────────────────────────┐ │
│    Raspberry Pi Zero 2 W     │ │
│                              │ │
│  CSI ◄──────── IMX290 camera │ │
│                              │ │
│  I²C ◄───────► MPU-6050 0x68 │ │
│      ◄───────► ADS1115  0x48 │ │
│                              │ │
│  GPIO12/13 PWM ──────────────┼─┼──► TB6612FNG ──► N20 motors (L, R)
│  GPIO dir+STBY (see pins) ───┼─┘                      │
│                              │                        │
│  GPIO16/19/20/21 enc ◄───────┼────────────────────────┘
│                              │
│  GPIO5/6  ──────────────────►│ TM1637 display
│  GPIO17   ◄──────────────────│ start button
└──────────────────────────────┘
```

---

## Parts

| Part | Role | Interface |
| --- | --- | --- |
| Raspberry Pi Zero 2 W | All computation: capture, Phases 1–3, navigation, motor control | Quad-core Cortex-A53, 512 MB RAM |
| IMX290 camera, M12 lens | Forward view of the course | CSI-2; 1920×1080 sensor mode scaled to 480×270 (see `vision_pipeline/phase1_capture.md`) |
| MPU-6050 | Gyro yaw rate for heading while vision is lost; lateral acceleration | I²C, 0x68 (AD0 low), sampled at 100 Hz |
| ADS1115 | 16-bit ADC: the pack's voltage, on A0 through a 4:1 divider (30 kΩ / 10 kΩ; `diagnostics/battery.py`) | I²C, 0x48 |
| TB6612FNG | Dual H-bridge for the two drive motors | PWM on GPIO12 (left) and GPIO13 (right); direction GPIO22/27 (left) and 25/24 (right); standby GPIO23 |
| N20 gearmotors with encoders (×2) | Differential drive; encoders for wheel speed and distance | Quadrature on GPIO19/16 (left) and GPIO20/21 (right), counted by pigpio callbacks |
| MP1584EN | Buck converter: battery to the Pi's supply | The pack (9.0–12.6 V) in, 5 V out |
| 3×18650 pack | Power for everything: 3S, 11.1 V nominal, 12.6 V full | Warning at 10.5 V, critical (the run ends) at 9.9 V; `diagnostics/battery.py` |
| TM1637 | 4-digit display: ready, countdown, run time | GPIO5 (CLK), GPIO6 (DIO) |
| Tact switch | Start button | GPIO17, active high, pull-down |

Steering is differential: the robot turns by driving the two wheels at different speeds. There's no steering servo, so the servo-feedback fields in older docs no longer apply.

---

## Pin and bus map

BCM numbering, following Product Spec GPIO Table 7. The display, button and I²C addresses are in `src/params.py`; the motor and encoder pins are defaults in `src/peripherals/drive.py` (`MotorController`, `EncoderReader`). Moving those into `params.py` too would keep every pin in one place, so no two modules can claim the same one.

| BCM | Header | Use | In `params.py` |
| --- | --- | --- | --- |
| GPIO2 / GPIO3 | 3 / 5 | I²C SDA / SCL: MPU-6050, ADS1115 | Addresses only |
| GPIO5 | 29 | TM1637 CLK | `GPIO_DISPLAY_CLK` |
| GPIO6 | 31 | TM1637 DIO | `GPIO_DISPLAY_DIO` |
| GPIO12 | 32 | Left motor PWM (hardware PWM0) | `drive.py` |
| GPIO13 | 33 | Right motor PWM (hardware PWM1) | `drive.py` |
| GPIO17 | 11 | Start button | `GPIO_START_BUTTON` |
| GPIO22 / GPIO27 | 15 / 13 | Left motor direction (AIN1 reverse, AIN2 forward) | `drive.py` |
| GPIO25 / GPIO24 | 22 / 18 | Right motor direction (BIN1 forward, BIN2 reverse) | `drive.py` |
| GPIO23 | 16 | TB6612 standby | `drive.py` |
| GPIO19 / GPIO16 | 35 / 36 | Left encoder, C1 / C2 | `drive.py` |
| GPIO20 / GPIO21 | 38 / 40 | Right encoder, C1 / C2 | `drive.py` |

| I²C address | Device | In `params.py` |
| --- | --- | --- |
| 0x68 | MPU-6050 | `IMU_I2C_ADDRESS` |
| 0x48 | ADS1115 | `ADS1115_I2C_ADDRESS` |

---

## Power

```
3×18650 (3S) ───┬── MP1584EN ── 5 V ── Pi Zero 2 W ── 3.3 V ── MPU-6050, ADS1115, TM1637, TB6612 logic
                └── TB6612FNG VM ── N20 motors
```

The Pi and the motors share a battery but not a regulator. Motor current spikes, at startup or when a wheel stalls, sag the battery; the buck converter keeps the Pi's rail steady through that as long as the battery stays above its minimum input. If the Pi resets when the motors start, check the buck's input headroom first.

Still to measure: the current draw of the Pi and of each motor at stall. The robot measures voltage only (`make routine-power-profile` records the sag under each load); current needs a sensor such as an INA219 on the I²C bus. **TBD**

---

## Software on the Pi

```
pigpiod (daemon)                     hardware PWM, GPIO, encoder callbacks
│
python process
├── main thread, once per frame:
│     capture → Phase 2 → Phase 3 → navigation → motor command
│     display update (throttled to 1 Hz)
├── sensor hub thread: MPU-6050 and encoder counts together at 100 Hz, drained once per frame (src/peripherals/sensing.py)
└── encoder counting: pigpio callbacks
```

| Component | Owner | Code |
| --- | --- | --- |
| Capture, Phases 2–3 | Vision | `src/capture/`, `src/perception/`, `src/estimation/estimation.py` |
| IMU reader | Vision | `src/peripherals/imu.py` |
| Sensor collection (IMU + encoders per frame) | Vision | `src/peripherals/sensing.py` |
| Start button, display | Vision | `src/peripherals/system.py` |
| Navigation: the rules, the route, intersections | Navigation | `src/navigation/` |
| Motor control and encoders | Navigation | `src/peripherals/drive.py` |
| The main loop tying them together | Shared | `src/main.py` (the run), `src/pipeline.py` (one frame's stages) |

- **pigpio for all GPIO.** One daemon owns the pins, so the display, button and motor driver can't conflict. Start it with `sudo pigpiod` or the systemd service.
- **The IMU runs on its own thread** because the frame rate is too slow to sample the gyro well. See `vision_pipeline/phase3_estimation.md`.
- **The handoff to navigation is the `EstimationPacket`,** documented in `vision_pipeline/phase3_estimation.md`; what Navigation does with it is in `vision_pipeline/navigation_contract.md`.

---

## Timing budget

At 20 FPS each frame has 50 ms for Phase 2, Phase 3, navigation and the motor update. Capture overlaps with processing, since the appsink keeps only the newest frame.

Measured with navigation in the same process: the frame rate is about 15 FPS on the IMX290 at 480×270, short of the 20 FPS target, with frames still over budget (`requirements.md`, P1). `make routine-frame-budget` judges the rate, the budget, latency, CPU and memory together, and `guides/diagnostics.md` shows where the time goes.

---

## Accepted risks

| Risk | What it looks like | Mitigation |
| --- | --- | --- |
| Scheduler jitter | Linux isn't real-time; a frame can arrive 5–20 ms late under load | PWM is in hardware, so only command updates are late. Phase 3 clamps large `dt` |
| CPU contention | Navigation or logging slows perception; frame times spike | Profile first; see CPU contention below |
| Single point of failure | A crash in the Python process leaves the motors at their last command | A watchdog in `drive.py` brakes the motors when no command arrives for `MOTOR_WATCHDOG_S` (0.5 s), and the run loop halts them on any exception (`main.py`) |
| Brownout | Motor current sags the battery and the Pi resets | Buck headroom; measure stall current |
| Memory | 512 MB shared by the OS, libcamera, OpenCV and Python | Stay in one process; check peak memory during a full run |

The watchdog was the one to do before running at speed, since a frozen Python process with the motors still driving is the likeliest way to damage the robot. It's in place.

---

## CPU contention

The Pi Zero 2 W has 4 cores, but Python's GIL (`concepts.md`) lets only one thread run Python code at a time. Threads help with waiting (the IMU thread spends most of its time asleep), not with running Python in parallel. Heavy OpenCV calls are the exception: they release the GIL and run their own worker threads across the cores, which is why a run on the lane measured 165% of a core for the process with `OPENCV_THREADS = 2` (`params.py`, 2026-10-03).

Profile before changing anything. Run the full pipeline with navigation active and look at per-frame time and CPU:

| Observation | Action |
| --- | --- |
| Frame time under 50 ms, CPU under 70% | No change |
| Occasional spikes over 50 ms | Time-slicing: run perception and navigation strictly one after the other in the main loop, so nothing competes during a frame. `phase3_linker` already runs this way |
| Consistent overruns | Split perception and navigation into separate processes pinned to different cores, with a queue between them. Two Python + OpenCV processes may need 160–240 MB combined, so measure memory first |
| Still over budget | Lower the load: smaller lane ROI, 15 FPS, or fewer stages per frame |

---
