# Brief: Architecture

**This brief covers only the Architecture section: how the robot's work is
divided, why everything runs on one board, what hardware sits around it, and
how the software is shaped. It does not explain how any single detector
works; later sections do.**

## The section's main idea

Every autonomous robot divides into capture, perception, estimation,
navigation, control and actuation. Navilott runs all of them on one
Raspberry Pi Zero 2 W, with no second microcontroller, and accepts what
that costs.

## One board: the decision

Three options were weighed: a microcontroller alone, the Pi alone, or the Pi
plus a microcontroller (an STM32 or Arduino) running a fast PID loop and
receiving packets over UART. The Pi alone won:

- **The camera sets the control rate anyway.** Steering updates once per
  frame, 20 times a second; a 1 kHz microcontroller loop would mostly repeat
  the last command.
- **PWM is already hardware.** The motor PWM comes from the Pi's two
  hardware PWM channels (GPIO12, GPIO13) through the pigpio daemon, so a
  late Python thread delays a change but doesn't disturb the waveform.
- **One codebase.** No second firmware, packet format or clock to
  synchronize.
- **A microcontroller can't hold the frames.** One 480x270 color frame is
  about 389 KB.

The cost: no dedicated speed control loop, and Linux's scheduler sits in the
timing.

## Hardware around the Pi

IMX290 camera (wide lens, replaced the IMX219, whose cropped view left a
22 cm blind spot; now about 10 cm); MPU-6050 gyro and accelerometer;
ADS1115 for battery voltage (the Pi has no analog input); TB6612FNG motor
driver; two N20 gearmotors with encoders; TM1637 display and a start button;
a 3S 18650 pack (12.6 V full, 11.1 V nominal); an MP1584EN buck converter
to 5 V for the Pi. The battery warns at 10.5 V and ends the run at 9.9 V.

## Software shape

- Four phases in a fixed order: capture, perception, estimation,
  navigation.
- One frame at a time, in step with the camera. Each frame has a 50 ms
  budget at 20 FPS.
- A few helpers run on their own threads: the camera's GStreamer thread,
  the sensor hub reading the IMU and encoders at 100 Hz, and the motor
  watchdog.
- **First version and today:** the first integration script
  (archive/run_pipeline.py, 640 lines, archived 2026-09-25) already chose a
  single sequential loop tied to the camera. Today's src/pipeline.py is
  251 lines. Tuning moved into one config file, the stage order is written
  in one place, the hand-off to navigation became a written contract, and
  every stage can run without the robot.
- **Language and tools:** Python over OpenCV's C++ core; Make for setup
  and commands; pytest; git with pull requests.

## Questions the author has to answer here

1. What were the two or three turning points between the first script and
   today, and what forced each?
2. What does Python cost on a Pi Zero 2 W, and does it show in the frame
   rate?
3. Which hardware choices were Ana's or Ignacio's?
