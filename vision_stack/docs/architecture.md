# System Architecture

> What Navilott runs on, how it's wired, and why.

Navilott runs everything on one Raspberry Pi Zero 2 W: camera capture, perception, estimation, navigation and motor control. There's no second microcontroller. This doc records that decision, the hardware around it, and the risks that come with it.

Items marked **TBD** aren't in the code or earlier docs. Fill them in from the wiring and the Product Spec.

---

## Decision: Pi Zero 2 W only

Two layouts were considered:

- **Pi only:** the Pi runs perception and drives the motors directly.
- **Pi + MCU:** the Pi sends a packet over UART to an STM32 or Arduino that runs a fast PID loop and the PWM.

**We went with Pi only.** The reasons:

- **The camera sets the control rate either way.** Steering corrections come from lane offset, which updates once per frame (20 Hz). An MCU running a 1 kHz loop would mostly repeat the last command between frames.
- **PWM is in hardware already.** The motor PWM runs on GPIO12 and GPIO13, the Pi's two hardware PWM channels, through the pigpio daemon. The duty cycle holds steady even when Python is late, so scheduler jitter delays a command change but doesn't disturb the waveform.
- **One board, one codebase.** No second firmware, no UART packet format, no checksums, and no link to debug between two processors.

The MCU option and its packet format are in `archives/`.

---

## Block diagram

```
                  Battery (TBD)
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
   MP1584EN buck            TB6612FNG VM
   → 5 V (TBD)              (motor supply)
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
│  GPIO (dir, STBY) TBD ───────┼─┘                      │
│                              │                        │
│  GPIO (encoders) TBD ◄───────┼────────────────────────┘
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
| IMX290 camera, M12 lens | Forward view of the course | CSI-2; 1920×1080 sensor mode scaled to 480×270 (see `vision_stack/phase1_capture.md`) |
| MPU-6050 | Gyro yaw rate for heading while vision is lost; lateral acceleration | I²C, 0x68 (AD0 low), sampled at 100 Hz |
| ADS1115 | 16-bit ADC. What it measures: **TBD** | I²C, 0x48 |
| TB6612FNG | Dual H-bridge for the two drive motors | PWM on GPIO12/13; direction and standby pins **TBD** |
| N20 gearmotors with encoders (×2) | Differential drive; encoders for wheel speed and distance | Encoder pins **TBD** |
| MP1584EN | Buck converter: battery to the Pi's supply | Input and output voltage **TBD** |
| TM1637 | 4-digit display: ready, countdown, run time | GPIO5 (CLK), GPIO6 (DIO) |
| Tact switch | Start button | GPIO17, active high, pull-down |

Steering is differential: the robot turns by driving the two wheels at different speeds. There's no steering servo, so the servo-feedback fields in older docs no longer apply.

---

## Pin and bus map

BCM numbering, following Product Spec GPIO Table 7. Pins used in code come from `src/params.py`; add the rest there too, so no two modules can claim the same pin.

| BCM | Header | Use | In `params.py` |
| --- | --- | --- | --- |
| GPIO2 / GPIO3 | 3 / 5 | I²C SDA / SCL: MPU-6050, ADS1115 | Addresses only |
| GPIO5 | 29 | TM1637 CLK | `GPIO_DISPLAY_CLK` |
| GPIO6 | 31 | TM1637 DIO | `GPIO_DISPLAY_DIO` |
| GPIO12 | 32 | Motor PWM (hardware PWM0) | Not yet |
| GPIO13 | 33 | Motor PWM (hardware PWM1) | Not yet |
| GPIO17 | 11 | Start button | `GPIO_START_BUTTON` |
| TBD | | TB6612 direction ×4, STBY | Not yet |
| TBD | | Encoder A/B, left and right | Not yet |

| I²C address | Device | In `params.py` |
| --- | --- | --- |
| 0x68 | MPU-6050 | `IMU_I2C_ADDRESS` |
| 0x48 | ADS1115 | Not yet |

---

## Power

```
Battery (TBD) ──┬── MP1584EN ── 5 V (TBD) ── Pi Zero 2 W ── 3.3 V ── MPU-6050, ADS1115, TM1637, TB6612 logic
                └── TB6612FNG VM ── N20 motors
```

The Pi and the motors share a battery but not a regulator. Motor current spikes, at startup or when a wheel stalls, sag the battery; the buck converter keeps the Pi's rail steady through that as long as the battery stays above its minimum input. If the Pi resets when the motors start, check the buck's input headroom first.

Values to record: battery chemistry and voltage, buck output setting, and the measured current draw of the Pi and of each motor at stall. **TBD**

---

## Software on the Pi

```
pigpiod (daemon)                     hardware PWM, GPIO, encoder callbacks
│
python process
├── main thread, once per frame:
│     capture → Phase 2 → Phase 3 → navigation → motor command
│     display update (throttled to 1 Hz)
├── IMU thread: MPU-6050 at 100 Hz, drained once per frame
└── encoder counting: pigpio callbacks (navigation)
```

| Component | Owner | Code |
| --- | --- | --- |
| Capture, Phases 2–3 | Vision | `src/capture/`, `src/perception/`, `src/estimation.py` |
| IMU reader | Vision | `src/peripherals/imu.py` |
| Start button, display | Vision | `src/system.py` |
| Navigation, motor control, encoders | Navigation | **TBD** |
| The main loop tying them together | **TBD**, to agree between vision and navigation | |

- **pigpio for all GPIO.** One daemon owns the pins, so the display, button and motor driver can't conflict. Start it with `sudo pigpiod` or the systemd service.
- **The IMU runs on its own thread** because the frame rate is too slow to sample the gyro well. See `vision_stack/phase3_estimation.md`.
- **The handoff to navigation is the `EstimationPacket`,** documented in `vision_stack/phase3_estimation.md`.

---

## Timing budget

At 20 FPS each frame has 50 ms for Phase 2, Phase 3, navigation and the motor update. Capture overlaps with processing, since the appsink keeps only the newest frame.

Stage timings for the current camera are measured, not estimated. Run `pytest --hardware` (see `guides/pytest.md`) and record the results here once navigation is running in the same process. **TBD**

---

## Accepted risks

| Risk | What it looks like | Mitigation |
| --- | --- | --- |
| Scheduler jitter | Linux isn't real-time; a frame can arrive 5–20 ms late under load | PWM is in hardware, so only command updates are late. Phase 3 clamps large `dt` |
| CPU contention | Navigation or logging slows perception; frame times spike | Profile first; see CPU contention below |
| Single point of failure | A crash in the Python process leaves the motors at their last command | **Needs a watchdog:** stop the motors if no command arrives within a set time, and on any exception in the main loop |
| Brownout | Motor current sags the battery and the Pi resets | Buck headroom; measure stall current |
| Memory | 512 MB shared by the OS, libcamera, OpenCV and Python | Stay in one process; check peak memory during a full run |

The watchdog is the one to do before running at speed. A frozen Python process with the motors still driving is the likeliest way to damage the robot.

---

## CPU contention

The Pi Zero 2 W has 4 cores, but Python's GIL keeps CPU-bound threads from running in parallel. Threads help with waiting (the IMU thread spends most of its time asleep), not with sharing heavy computation.

Profile before changing anything. Run the full pipeline with navigation active and look at per-frame time and CPU:

| Observation | Action |
| --- | --- |
| Frame time under 50 ms, CPU under 70% | No change |
| Occasional spikes over 50 ms | Time-slicing: run perception and navigation strictly one after the other in the main loop, so nothing competes during a frame. `phase3_linker` already runs this way |
| Consistent overruns | Split perception and navigation into separate processes pinned to different cores, with a queue between them. Two Python + OpenCV processes may need 160–240 MB combined, so measure memory first |
| Still over budget | Lower the load: smaller lane ROI, 15 FPS, or fewer stages per frame |

---

## Archived

Moved to `archives/`, kept for reference:

- Pi + MCU architecture, UART packet schema and the Pico variant
- TensorFlow Lite and external inference fallbacks
- Homography calibration (replaced by pixel-based lane offset)
- The 480×360 IMX219 timing tables
