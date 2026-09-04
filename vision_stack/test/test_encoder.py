"""
test_encoder.py

Wheel Encoder Discovery and Reader

Purpose:
    Find which GPIO pins the N20 encoders are wired to, verify them, and then
    read distance per frame so the scene pipeline's distance triggers run on
    measured travel instead of integrated commanded speed.

    Mirrors imu.py: a background accumulator that the main loop drains once per
    frame with snapshot().

HARDWARE NOTE -- READ BEFORE WIRING

    Pi GPIO is 3.3 V only and is NOT 5 V tolerant. An N20 encoder board outputs
    at whatever you feed its VCC, so power the ENCODER side from 3.3 V (pin 1
    or 17), not from the 5 V rail. The motor supply is separate and unaffected.
    Getting this wrong damages the pin, usually permanently.

WHY SINGLE-CHANNEL COUNTING IS THE DEFAULT

    Quadrature decoding exists to recover direction of rotation. This robot
    drives forward, and direction is already known from the commanded wheel
    sign, so counting rising edges on ONE channel gives the same distance at a
    quarter of the interrupt rate. At a 100:1 N20 that is roughly 3.5k edges/s
    per wheel instead of 14k, against a 66.7 ms frame budget that already gets
    exceeded.

    Set quadrature=True if you later need to detect rollback on a slope or
    verify that a wheel is actually turning the commanded way.

USAGE

    python encoders.py scan
        Watches every unused GPIO. Spin each wheel by hand; the pins that
        count edges are your encoder channels. This is the "find them" step --
        there is nothing to ping, encoders are not addressable.

    python encoders.py verify --left 20 --right 21 --turns 10
        Counts edges over a known number of wheel revolutions so you can
        compute counts_per_output_rev empirically instead of trusting a
        datasheet gear ratio.

    In the pipeline:
        enc = EncoderReader(left_a=20, right_a=21,
                            counts_per_output_rev=700.0,
                            wheel_diameter_cm=4.3)
        enc.start()
        frame = enc.snapshot()      # once per frame
        enc.stop()
"""

import sys
import time
import math
import argparse
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

import pigpio

log = logging.getLogger(__name__)

# GPIO already committed elsewhere in this robot. Scan skips these so a stray
# motor-direction line is not mistaken for an encoder channel.
#   24,25,13 motor A      22,27,12 motor B      23 STBY
#   5,6 TM1637            17 start button       2,3 I2C (MPU-6050)
RESERVED_GPIO = {2, 3, 5, 6, 12, 13, 17, 22, 23, 24, 25, 27}

# Broadcom-numbered pins available on the 40-pin header.
HEADER_GPIO = [4, 7, 8, 9, 10, 11, 14, 15, 16, 18, 19, 20, 21, 26]


# =============================================================================
# Output container
# =============================================================================
@dataclass
class EncoderFrame:
    """
    Travel over one pipeline frame window.

    distance_cm: mean of the two wheels this frame, signed by commanded
        direction. Mean rather than either wheel alone: during a correction the
        wheels differ, and the chassis centre travels the average.
    total_distance_cm: cumulative since reset
    left_counts / right_counts: raw edges this frame, for imbalance checks
    speed_cm_s: distance_cm / dt
    valid: False when no edges arrived on EITHER wheel while the robot was
        commanded to move -- a stall, a disconnected channel, or a wheel off
        the ground. Phase 3 should not integrate an invalid frame.
    """
    distance_cm:       float = 0.0
    total_distance_cm: float = 0.0
    left_counts:       int = 0
    right_counts:      int = 0
    speed_cm_s:        float = 0.0
    valid:             bool = False


# =============================================================================
# Reader
# =============================================================================
class EncoderReader:
    """
    Counts encoder edges in pigpio's callback thread and accumulates them for
    the main loop.
    """

    def __init__(
            self,
            left_a: int,
            right_a: int,
            counts_per_output_rev: float,
            wheel_diameter_cm: float,
            left_b: Optional[int] = None,
            right_b: Optional[int] = None,
            quadrature: bool = False,
            glitch_us: int = 100,
            pi: Optional[pigpio.pi] = None,
        ):
        """
        Inputs:
            left_a / right_a: GPIO for the C1 channel of each encoder
            counts_per_output_rev: edges counted per revolution of the WHEEL,
                not the motor shaft. Measure it with `verify`; a datasheet gear
                ratio is nominal and N20 gearboxes vary.
            wheel_diameter_cm: for the distance conversion
            left_b / right_b: C2 channels, only needed when quadrature=True
            quadrature: see the module docstring; leave False for forward-only
            glitch_us: pigpio glitch filter. Brushed motors are electrically
                noisy and a few microseconds of filtering removes phantom
                counts without losing real ones at these speeds.
            pi: an existing pigpio.pi(), or None to open one
        """
        self._pi = pi or pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")
        self._owns_pi = pi is None

        self._cm_per_count = (math.pi * wheel_diameter_cm) / max(counts_per_output_rev, 1.0)
        self._quadrature = quadrature

        self._lock = threading.Lock()
        self._left = 0
        self._right = 0
        self._total_cm = 0.0
        self._direction = 1.0
        self._last_ts = time.perf_counter()

        self._callbacks = []
        pins = [(left_a, "left"), (right_a, "right")]
        if quadrature:
            if left_b is None or right_b is None:
                raise ValueError("quadrature=True requires left_b and right_b")
            pins += [(left_b, "left"), (right_b, "right")]

        for gpio, side in pins:
            self._pi.set_mode(gpio, pigpio.INPUT)
            self._pi.set_pull_up_down(gpio, pigpio.PUD_UP)
            self._pi.set_glitch_filter(gpio, glitch_us)
            edge = pigpio.EITHER_EDGE if quadrature else pigpio.RISING_EDGE
            self._callbacks.append(
                self._pi.callback(gpio, edge, self._make_cb(side)))

        log.info("EncoderReader on GPIO left=%d right=%d  %.5f cm/count  %s",
                 left_a, right_a, self._cm_per_count,
                 "quadrature" if quadrature else "single-channel")

    def _make_cb(self, side: str):
        def _cb(gpio, level, tick):
            with self._lock:
                if side == "left":
                    self._left += 1
                else:
                    self._right += 1
        return _cb

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Callbacks are live from construction; kept for symmetry with imu.py."""
        self._last_ts = time.perf_counter()

    def stop(self) -> None:
        for cb in self._callbacks:
            cb.cancel()
        self._callbacks.clear()
        if self._owns_pi:
            self._pi.stop()
        log.info("EncoderReader stopped.")

    # ------------------------------------------------------------------
    # Frame interface
    # ------------------------------------------------------------------
    def set_direction(self, commanded_speed: float) -> None:
        """
        Purpose:
            Supply the sign that single-channel counting cannot recover.

        Notes:
            Call this with the commanded base speed BEFORE snapshot(). With
            quadrature=False the counters are unsigned, so reversing without
            telling the reader makes the robot appear to keep moving forward --
            which would drive the map the wrong way through an intersection.
        """
        if not self._quadrature and commanded_speed < 0:
            self._direction = -1.0
        else:
            self._direction = 1.0

    def snapshot(self, moving: bool = True) -> EncoderFrame:
        """
        Purpose:
            Drain the counters and convert to distance.

        Inputs:
            moving: whether the robot was commanded to move this frame. Used
                only to decide validity -- zero counts while stopped is
                correct, zero counts while driving is a fault.
        """
        now = time.perf_counter()
        with self._lock:
            left, right = self._left, self._right
            self._left = self._right = 0

        dt = max(now - self._last_ts, 1e-6)
        self._last_ts = now

        counts = 0.5 * (left + right)
        distance = counts * self._cm_per_count * self._direction
        self._total_cm += distance

        return EncoderFrame(
            distance_cm=distance,
            total_distance_cm=self._total_cm,
            left_counts=left, right_counts=right,
            speed_cm_s=distance / dt,
            valid=(not moving) or (left > 0 or right > 0),
        )

    @property
    def total_distance_m(self) -> float:
        """Cumulative distance in metres, for SensorSample.distance_traveled."""
        return self._total_cm / 100.0


# =============================================================================
# Discovery
# =============================================================================
def scan(duration_s: float = 30.0, glitch_us: int = 100) -> None:
    """
    Purpose:
        Watch every unused GPIO and report which ones see edges. Spin each
        wheel by hand during the window; the pins that count are the encoder
        channels.

    Notes:
        Expect FOUR active pins for two encoders -- C1 and C2 per motor. The
        two belonging to one wheel will have near-identical counts, since both
        channels see the same shaft. That pairing is how you tell which two go
        together; which pair is left vs right you get by spinning one wheel at
        a time.
    """
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

    candidates = [g for g in HEADER_GPIO if g not in RESERVED_GPIO]
    counts: Dict[int, int] = {g: 0 for g in candidates}
    lock = threading.Lock()

    def _make(g):
        def _cb(gpio, level, tick):
            with lock:
                counts[g] += 1
        return _cb

    cbs = []
    for g in candidates:
        pi.set_mode(g, pigpio.INPUT)
        pi.set_pull_up_down(g, pigpio.PUD_UP)
        pi.set_glitch_filter(g, glitch_us)
        cbs.append(pi.callback(g, pigpio.EITHER_EDGE, _make(g)))

    print(f"Watching GPIO {candidates}")
    print(f"Skipping reserved {sorted(RESERVED_GPIO)}")
    print(f"\nSpin ONE wheel by hand for {duration_s:.0f} s. Ctrl-C to stop early.\n")

    try:
        t0 = time.time()
        while time.time() - t0 < duration_s:
            time.sleep(1.0)
            with lock:
                live = {g: c for g, c in counts.items() if c > 0}
            print(f"  {time.time() - t0:5.1f}s  active: "
                  f"{live if live else '(nothing yet)'}")
    except KeyboardInterrupt:
        pass
    finally:
        for cb in cbs:
            cb.cancel()

    with lock:
        active = sorted(((c, g) for g, c in counts.items() if c > 0), reverse=True)

    print("\n" + "=" * 56)
    if not active:
        print("No edges on any pin. Check that the encoder VCC is powered")
        print("(3.3 V, not 5 V) and that GND is shared with the Pi.")
    else:
        print("Pins that saw edges, most active first:")
        for c, g in active:
            print(f"  GPIO {g:<3} {c:6d} edges")
        print("\nTwo pins with near-equal counts are the two channels of one")
        print("encoder. Spin the other wheel to identify the second pair.")
    print("=" * 56)
    pi.stop()


def verify(gpio_left: int, gpio_right: int, turns: float,
           wheel_diameter_cm: float, glitch_us: int = 100) -> None:
    """
    Purpose:
        Measure counts_per_output_rev empirically.

    Notes:
        Turn each wheel by hand exactly `turns` full revolutions. Mark the tyre
        with tape so the count is unambiguous. More turns is better -- 10 turns
        divides any single-revolution error by ten, and the gearbox backlash
        that makes one turn unreliable averages out.

        Use the measured number, not the datasheet gear ratio. N20 gearboxes
        are sold at nominal ratios and the actual tooth counts differ.
    """
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon not reachable. Run: sudo pigpiod")

    counts = {"left": 0, "right": 0}
    lock = threading.Lock()

    def _make(side):
        def _cb(gpio, level, tick):
            with lock:
                counts[side] += 1
        return _cb

    cbs = []
    for gpio, side in ((gpio_left, "left"), (gpio_right, "right")):
        pi.set_mode(gpio, pigpio.INPUT)
        pi.set_pull_up_down(gpio, pigpio.PUD_UP)
        pi.set_glitch_filter(gpio, glitch_us)
        cbs.append(pi.callback(gpio, pigpio.RISING_EDGE, _make(side)))

    print(f"Turn each wheel exactly {turns:g} full revolutions, then press Enter.")
    input()

    for cb in cbs:
        cb.cancel()
    pi.stop()

    print("\n" + "=" * 56)
    for side in ("left", "right"):
        cpr = counts[side] / max(turns, 1e-6)
        cm_per_count = (math.pi * wheel_diameter_cm) / max(cpr, 1.0)
        print(f"  {side:<6} {counts[side]:6d} edges -> "
              f"{cpr:8.1f} counts/rev -> {cm_per_count:.5f} cm/count")

    l, r = counts["left"], counts["right"]
    if l and r:
        skew = abs(l - r) / max(l, r)
        print(f"\n  left/right skew {100 * skew:.1f}%")
        if skew > 0.05:
            print("  Over 5% on equal hand-turns means one channel is missing")
            print("  edges -- noise, a loose wire, or a glitch filter set too")
            print("  wide. Fix it before trusting distance.")
    print("=" * 56)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Find and verify wheel encoders")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="find which GPIO the encoders use")
    s.add_argument("--seconds", type=float, default=30.0)

    v = sub.add_parser("verify", help="measure counts per wheel revolution")
    v.add_argument("--left", type=int, required=True)
    v.add_argument("--right", type=int, required=True)
    v.add_argument("--turns", type=float, default=10.0)
    v.add_argument("--wheel-cm", type=float, default=4.3,
                   help="wheel DIAMETER in cm")

    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.seconds)
    else:
        verify(a.left, a.right, a.turns, a.wheel_cm)