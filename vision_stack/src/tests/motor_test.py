"""
motor_polarity_test.py

Bench test for TB6612 motor polarity

Purpose:
    Isolates each motor and drives it in each commanded direction one at a
    time
"""

import time
import pigpio

# =============================================================================
# Pin Setup
# =============================================================================
# Motor A (Left) — TB6612 AIN side
_ain1 = 24
_ain2 = 25
_pwma = 13
# Motor B (Right) — TB6612 BIN side
_bin1 = 27
_bin2 = 22
_pwmb = 12
_stby = 23

TEST_SPEED = 0.30      # low duty cycle
STEP_DURATION_S = 2.0  # how long each test step runs
PAUSE_S = 1.5          # pause between steps so motors fully stop

pi = pigpio.pi()
if not pi.connected:
    raise SystemExit("Could not connect to pigpiod -- is it running? (sudo pigpiod)")

pi.set_mode(_ain1, pigpio.OUTPUT)
pi.set_mode(_ain2, pigpio.OUTPUT)
pi.set_mode(_bin1, pigpio.OUTPUT)
pi.set_mode(_bin2, pigpio.OUTPUT)
pi.set_mode(_stby, pigpio.OUTPUT)


def _stop_all() -> None:
    pi.hardware_PWM(_pwma, 1000, 0)
    pi.hardware_PWM(_pwmb, 1000, 0)
    pi.write(_ain1, 0)
    pi.write(_ain2, 0)
    pi.write(_bin1, 0)
    pi.write(_bin2, 0)


def _drive_motor_a_raw(direction: str, speed: float) -> None:
    """
    Directly drives Motor A (left) without the inversion logic --
    raw AIN1/AIN2/PWMA behavior, tested in isolation
    """
    duty = int(max(0.0, min(1.0, speed)) * 1_000_000)
    pi.hardware_PWM(_pwma, 1000, duty)
    if direction == "forward":
        pi.write(_ain1, 1)
        pi.write(_ain2, 0)
    else:  # reverse
        pi.write(_ain1, 0)
        pi.write(_ain2, 1)


def _drive_motor_b_raw(direction: str, speed: float) -> None:
    """
    Directly drives Motor B (right) without the inversion logic --
    raw BIN1/BIN2/PWMB behavior, tested in isolation
    """
    duty = int(max(0.0, min(1.0, speed)) * 1_000_000)
    pi.hardware_PWM(_pwmb, 1000, duty)
    if direction == "forward":
        pi.write(_bin1, 1)
        pi.write(_bin2, 0)
    else:  # reverse
        pi.write(_bin1, 0)
        pi.write(_bin2, 1)


def _run_step(label: str, expect: str, fn, *fn_args) -> None:
    """
    Runs a single test step, prompting the user to observe the wheel rotation
    and report whether it matches the expected direction
    """
    print(f"\n[{label}]")
    print(f"  Expect: {expect}")
    pi.write(_stby, 1)
    fn(*fn_args)
    time.sleep(STEP_DURATION_S)
    _stop_all()
    time.sleep(PAUSE_S)
    result = input("  Observed forward / reverse / no-spin? (f/r/n): ").strip().lower()
    verdict = "PASS" if result == "f" else ("FAIL - reversed" if result == "r" else "FAIL - no spin")
    print(f"  -> {verdict}")


def main() -> None:
    print("=" * 70)
    print("MOTOR POLARITY TEST")
    print("=" * 70)
    input("Press Enter when the chassis is propped up and clear to spin freely...")

    _run_step(
        "Motor A (Left) -- AIN1=1, AIN2=0",
        "Left wheel spins in the FORWARD direction",
        _drive_motor_a_raw, "forward", TEST_SPEED,
    )
    _run_step(
        "Motor A (Left) -- AIN1=0, AIN2=1",
        "Left wheel spins in the REVERSE direction",
        _drive_motor_a_raw, "reverse", TEST_SPEED,
    )
    _run_step(
        "Motor B (Right) -- BIN1=1, BIN2=0",
        "Right wheel spins in the FORWARD direction",
        _drive_motor_b_raw, "forward", TEST_SPEED,
    )
    _run_step(
        "Motor B (Right) -- BIN1=0, BIN2=1",
        "Right wheel spins in the REVERSE direction",
        _drive_motor_b_raw, "reverse", TEST_SPEED,
    )

    print("\n" + "=" * 70)
    print("Done. Check the results above to verify that each wheel spins in the expected direction.")
    print("=" * 70)

    _stop_all()
    pi.write(_stby, 0)
    pi.stop()


if __name__ == "__main__":
    main()