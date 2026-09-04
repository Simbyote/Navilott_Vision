"""
imu_test.py

Bench test for the IMU (yaw rate / gyro) reading pipeline

Purpose:
    This script isolates the IMU reading and integration logic from the rest of
    the vision pipeline, so you can confirm that the yaw rate readings are
    reasonable and that the integration over time matches a known rotation.

    Phase A -- Stationary bias check
        Robot sits completely still
    Phase B -- Known-rotation scale check.
        YManually rotate the robot by a known angle while a script integrates yaw_rate 
        over time. Comparing the integrated result to the actual angle rotated allows to tell
        if there's a scale-factor error
"""

import sys
import time
import statistics

sys.path.insert(0, "vision_stack/src")
from imu import IMUReader, IMUFrame  # noqa: E402

# =============================================================================
# Config
# =============================================================================
IMU_ADDRESS = 0x68
IMU_RATE_HZ = 100.0

# Sample at the same rate as frame capture in run_pipeline.py (15 Hz) 
POLL_INTERVAL_S = 1.0 / 15.0

PHASE_A_DURATION_S = 15.0
BIAS_WARN_THRESHOLD_DPS = 2.0   # mean yaw rate while stationary that triggers a warning


def _collect(duration_s: float, imu: IMUReader, label: str) -> list[IMUFrame]:
    """
    Collects IMU frames for a given duration, printing the mean yaw rate and peak lateral acceleration
    every POLL_INTERVAL_S seconds. Returns the list of collected frames
    """
    print(f"\nCollecting for {duration_s:.0f}s ({label})...")
    frames: list[IMUFrame] = []
    t_start = time.perf_counter()
    next_sample = t_start
    while (time.perf_counter() - t_start) < duration_s:
        now = time.perf_counter()
        if now >= next_sample:
            frame = imu.snapshot()
            frames.append(frame)
            elapsed = now - t_start
            yaw = frame.mean_yaw_rate_dps if frame.valid else float("nan")
            accel = frame.peak_lateral_accel if frame.valid else float("nan")
            print(
                f"  t={elapsed:5.1f}s  valid={frame.valid}  "
                f"n={frame.sample_count}  yaw={yaw:+7.2f} deg/s  "
                f"lat_accel={accel:+7.3f}"
            )
            next_sample += POLL_INTERVAL_S
        else:
            time.sleep(0.001)
    return frames


def _summarize_yaw(frames: list[IMUFrame], label: str) -> None:
    """
    Summarizes the yaw rate readings from a list of IMUFrames, printing mean, stdev, min, and max.
    Returns the mean yaw rate, or None if no valid frames were collected
    """
    yaw_vals = [f.mean_yaw_rate_dps for f in frames if f.valid]
    if not yaw_vals:
        print(f"\n[{label}] No valid frames collected -- check IMU connection.")
        return

    mean_yaw = statistics.mean(yaw_vals)
    stdev_yaw = statistics.stdev(yaw_vals) if len(yaw_vals) > 1 else 0.0
    min_yaw = min(yaw_vals)
    max_yaw = max(yaw_vals)

    print(f"\n[{label}] yaw_rate stats over {len(yaw_vals)} valid frames:")
    print(f"  mean   = {mean_yaw:+.3f} deg/s")
    print(f"  stdev  = {stdev_yaw:.3f} deg/s")
    print(f"  min    = {min_yaw:+.3f} deg/s")
    print(f"  max    = {max_yaw:+.3f} deg/s")
    return mean_yaw


def phase_a_stationary_bias(imu: IMUReader) -> float:
    """
    Phase A: Stationary bias check.
    Robot sits completely still, and we collect IMU frames to estimate the mean yaw rate
    If the mean yaw rate exceeds BIAS_WARN_THRESHOLD_DPS, we print a warning
    Returns the mean yaw rate (bias estimate) for use in Phase B
    """
    print("=" * 70)
    print("PHASE A -- Stationary bias check")
    print("=" * 70)
    input("Set the robot down completely still on the ground. Press Enter to start...")

    frames = _collect(PHASE_A_DURATION_S, imu, "stationary")
    mean_yaw = _summarize_yaw(frames, "Phase A")

    if mean_yaw is not None:
        if abs(mean_yaw) > BIAS_WARN_THRESHOLD_DPS:
            print(
                f"\n  *** FLAG: mean yaw rate {mean_yaw:+.2f} deg/s while stationary "
                f"exceeds the {BIAS_WARN_THRESHOLD_DPS} deg/s threshold\n"
            )
        else:
            print(
                f"\n  OK: mean yaw rate {mean_yaw:+.2f} deg/s is within the "
                f"{BIAS_WARN_THRESHOLD_DPS} deg/s bias threshold while stationary"
            )
    return mean_yaw or 0.0


def phase_b_known_rotation(imu: IMUReader, bias_estimate: float) -> None:
    """
    Phase B: Known-rotation scale check.
    The user manually rotates the robot by a known angle while this function integrates
    the bias-corrected yaw_rate over time. It then compares the integrated result to the
    actual angle rotated, printing the error percentage. If the error exceeds 15%,
    it flags a potential scale-factor issue in the IMU readings
    """
    print("\n" + "=" * 70)
    print("PHASE B -- Known-rotation scale check")
    print("=" * 70)
    print(
        "Manually rotate the robot by a known angle while this script integrates "
        "the bias-corrected yaw_rate over time. After you finish rotating, press Enter to stop the recording"
    )
    target_deg = input("Enter the angle you plan to rotate it by, in degrees (e.g. 360): ").strip()
    try:
        target_deg = float(target_deg)
    except ValueError:
        print("Not a number, skipping Phase B")
        return

    input(f"Position the robot at its 0-degree start mark. Press Enter, then rotate it {target_deg:.0f} deg smoothly and press Enter again when done...")

    integrated_deg = 0.0
    frames: list[IMUFrame] = []
    t_start = time.perf_counter()
    last_t = t_start

    print("Recording -- rotate the robot now, then press Enter when finished")

    import threading
    stop_flag = {"stop": False}

    def _wait_for_enter():
        input()
        stop_flag["stop"] = True

    listener = threading.Thread(target=_wait_for_enter, daemon=True)
    listener.start()

    while not stop_flag["stop"]:
        now = time.perf_counter()
        frame = imu.snapshot()
        dt = now - last_t
        if frame.valid:
            # Subtract the stationary bias estimate from Phase A
            corrected_rate = frame.mean_yaw_rate_dps - bias_estimate
            integrated_deg += corrected_rate * dt
            frames.append(frame)
        last_t = now
        time.sleep(0.02)

    elapsed = time.perf_counter() - t_start
    print(f"\nRotation recorded over {elapsed:.1f}s.")
    print(f"  Target rotation:     {target_deg:+.1f} deg")
    print(f"  Integrated estimate: {integrated_deg:+.1f} deg  (bias-corrected)")

    if target_deg != 0:
        error_pct = ((integrated_deg - target_deg) / target_deg) * 100.0
        print(f"  Error:               {integrated_deg - target_deg:+.1f} deg  ({error_pct:+.1f}%)")
        if abs(error_pct) > 15.0:
            print(
                "\n  *** FLAG: integration error exceeds 15%\n"
            )
        else:
            print("\n  OK: integration tracks the known rotation within 15%\n")


def main() -> None:
    imu = IMUReader(address=IMU_ADDRESS, rate_hz=IMU_RATE_HZ)
    imu.start()
    time.sleep(0.5) # Allow for initialization and first readings

    try:
        bias = phase_a_stationary_bias(imu)
        run_b = input("\nRun Phase B (known-rotation scale check)? (y/n): ").strip().lower()
        if run_b == "y":
            phase_b_known_rotation(imu, bias)
    finally:
        imu.stop()
        print("\nDone.")


if __name__ == "__main__":
    main()