"""
sim_robot.py  --  a simulated robot for maneuver and maneuver_linker tests

Wheels count at duty x cps_at_full (times a per-wheel gain, so one wheel can
run slow), and the body turns with the difference between them. The IMU
reports that turn with a chosen sign convention and a constant bias, so the
yaw-sign check and the bias measurement have something real to find.
Time is a FakeClock the tests advance; nothing sleeps.
"""
from types import SimpleNamespace

from src.estimation import SensorSample


class FakeClock:
    """perf_counter stand-in, advanced by the test."""
    def __init__(self, t=100.0):
        self.now = t

    def __call__(self):
        return self.now


class SimRobot:
    """
    Motor, encoders and IMU in one. drive()/stop() set the wheels; read()
    returns (SensorSample, imu_frame, encoder_frame) for the time since the
    previous read, as phase3_linker.Sensors does.

    imu_plus_is: "left" or "right", the IMU's sign convention for yaw.
    left_gain / right_gain: Scale each wheel's speed; below 1 it drags.
    deg_per_count: Body turn per count of (right - left) wheel travel.
    stalled: Wheels never turn, whatever the command (a disconnected encoder).
    lag_s: Each wheel follows its command with this time constant, so it
        coasts after a stop; 0 responds at once.
    brake_lag_s: The same while braking (brake()); a short brake stops the
        wheels far faster than coasting. 0 stops them at once.
    """
    def __init__(self, clock, cps_at_full=1000.0, left_gain=1.0, right_gain=1.0,
                 deg_per_count=0.12, imu_plus_is="left", bias_dps=-1.1, accel_baseline=-0.9,
                 stalled=False, imu=True, lag_s=0.0, brake_lag_s=0.0):
        self.clock = clock
        self.cps_at_full, self.gains = cps_at_full, (left_gain, right_gain)
        self.deg_per_count, self.bias, self.accel = deg_per_count, bias_dps, accel_baseline
        self.imu_sign = 1 if imu_plus_is == "left" else -1
        self.stalled, self.imu, self.lag_s, self.brake_lag_s = stalled, imu, lag_s, brake_lag_s
        self.braking = False
        self.brakes = 0
        self.cmd = (0.0, 0.0)
        self.duty = [0.0, 0.0]                  # what each wheel is actually doing
        self.counts = [0.0, 0.0]
        self.heading_deg = 0.0                  # true body heading, + = left
        self.commands = []
        self.stops = 0
        self._last = clock()

    def drive(self, left, right):
        self.cmd, self.braking = (left, right), False
        self.commands.append(self.cmd)

    def brake(self):
        self.cmd, self.braking = (0.0, 0.0), True
        self.commands.append(self.cmd)
        self.brakes += 1

    def stop(self):
        self.cmd = (0.0, 0.0)
        self.stops += 1

    def read(self):
        now = self.clock()
        dt, self._last = now - self._last, now
        lag = self.brake_lag_s if self.braking else self.lag_s
        k = 1.0 if lag <= 0 else min(1.0, dt / lag)
        self.duty = [d + (c - d) * k for d, c in zip(self.duty, self.cmd)]
        rates = [0.0, 0.0] if self.stalled else [
            d * self.cps_at_full * g for d, g in zip(self.duty, self.gains)]
        for i in (0, 1):
            self.counts[i] += rates[i] * dt
        turn_dps = (rates[1] - rates[0]) * self.deg_per_count
        self.heading_deg += turn_dps * dt
        imu = SimpleNamespace(valid=self.imu, mean_yaw_rate_dps=self.imu_sign * turn_dps + self.bias,
                              peak_lateral_accel=self.accel)
        enc = SimpleNamespace(left_count=int(self.counts[0]), right_count=int(self.counts[1]),
                              left_cps=rates[0], right_cps=rates[1])
        return SensorSample.from_frames(imu, enc), imu, enc
