"""
sim_robot.py  --  a simulated robot for maneuver and maneuver_linker tests

Wheels count at duty x cps_at_full (times a per-wheel gain, so one wheel can
run slow), and the body turns with the difference between them. The IMU
reports that turn with a chosen sign convention and a constant bias, as the
sensing hub delivers it, so the yaw-sign check and the bias measurement have
something real to find. Readings go through sensing.SensorBatch, so the wheel
speeds are the hub's own count-delta math.
Time is a FakeClock the tests advance; nothing sleeps. FakeBattery stands in
for diagnostics.battery.Power in the runs that watch the pack.
"""
from src.diagnostics.battery_state import BatteryState
from src.estimation.estimation import SensorSample
from src.peripherals.sensing import SensorBatch, SensorReading


class FakeClock:
    """perf_counter stand-in, advanced by the test."""
    def __init__(self, t=100.0):
        self.now = t

    def __call__(self):
        return self.now


class FakeBattery:
    """
    diagnostics.battery.Power stand-in: a pack draining volts_step per
    should_stop() check (once per frame in the runs) from start_v, CRITICAL
    once it reaches critical_at (None: never). log gets "battery started" /
    "battery released".
    """
    VOLTAGE_WARNING, VOLTAGE_CRITICAL = 10.5, 9.9

    def __init__(self, start_v=12.0, volts_step=0.0, critical_at=None, log=None, start_fails=False):
        self.v, self.step, self.critical_at = start_v, volts_step, critical_at
        self.log = log if log is not None else []
        self.start_fails, self.checks = start_fails, 0

    def start_monitoring(self, interval_s=1.0):
        if self.start_fails:
            raise OSError("ADC read failed")
        self.log.append("battery started")

    def should_stop(self):
        self.checks += 1
        self.v = round(self.v - self.step, 3)
        return self.critical_at is not None and self.checks >= self.critical_at

    def voltage(self):
        return self.v

    def state(self):
        return BatteryState.CRITICAL if self.critical_at is not None and self.checks >= self.critical_at \
            else BatteryState.OK

    def sensor_ok(self):
        return True

    def cleanup(self):
        self.log.append("battery released")


class SimRobot:
    """
    Motor, encoders and IMU in one. drive()/stop() set the wheels; read()
    returns (SensorSample, SensorBatch) for the time since the previous
    read, as phase3_linker.Sensors does.

    imu_plus_is: "left" or "right", the yaw sign as delivered. The hub flips
        this robot's IMU to "right" (IMU_YAW_SIGN), the default.
    left_gain / right_gain: Scale each wheel's speed; below 1 it drags.
    deg_per_count: Body turn per count of (right - left) wheel travel.
    stalled: Wheels never turn, whatever the command (a disconnected encoder).
    lag_s: Each wheel follows its command with this time constant, so it
        coasts after a stop; 0 responds at once.
    brake_lag_s: The same while braking (brake()); a short brake stops the
        wheels far faster than coasting. 0 stops them at once.
    """
    def __init__(self, clock, cps_at_full=1000.0, left_gain=1.0, right_gain=1.0,
                 deg_per_count=0.12, imu_plus_is="right", bias_dps=1.1, accel_baseline=-0.9,
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
        self._base = SensorReading(self._last, None, None, 0.0, 0.0)

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
        reading = SensorReading(now, self.imu_sign * turn_dps + self.bias if self.imu else None,
                                self.accel if self.imu else None, self.counts[0], self.counts[1])
        batch = SensorBatch((reading,), self._base)
        self._base = reading
        return SensorSample.from_batch(batch), batch
