"""test_settings.py  --  the run-wide settings in src/params.py, and the values that must agree across modules

Each module's own tests pin its tuned constants. What's here belongs to no
single module: the run-wide switches in params.py, and copies of one value
that live in several modules (a frame-gap limit, a duty, a battery
threshold) and only mean what they say while they match. A tuning change
that updates one copy and not the others fails here instead of on the course.

--software  Imports and arithmetic only. No camera, Pi or files written.
"""
import pytest

from src import params
from src.diagnostics.battery import Power
from src.diagnostics.battery_state import BatteryStateMachine
from src.estimation.estimation import Phase3Config
from src.navigation import end_of_course, intersection, wheel_speed
from src.navigation.lane_keeping import BASE_SPEED
from src.scripts import calibrate_odometry
from src.tests.sim_robot import FakeBattery

pytestmark = pytest.mark.software


def test_the_camera_runs_at_20_fps_480x270_upside_down():
    assert (params.FPS, params.FRAME_W, params.FRAME_H) == (20, 480, 270)
    assert params.CAMERA_ROTATE_180 and params.MIN_FPS <= params.FPS <= params.MAX_FPS
    assert params.SENSOR_CONFIG == "sensor/config,width=1920,height=1080,depth=10"


def test_threads_perception_and_the_collector_as_the_robot_runs_them():
    assert params.OPENCV_THREADS == 2
    assert params.PARALLEL_PERCEPTION is False          # serial until compare-parallel shows a gain on the Pi
    # No automatic collections while driving; the youngest generation on braked
    # frames and every 200 frames (~10 s at 20 FPS)
    assert params.GC_DISABLE_IN_RUN is True and params.GC_COLLECT_EVERY == 200


def test_the_nominal_odometry_scale_is_the_wheel_over_the_counts_per_turn():
    # 3.3 cm wheels; 7-pulse encoders behind 200:1 N20s (12 V, 140 RPM no load: the seller's listing)
    assert (params.WHEEL_DIAMETER_CM, params.ENCODER_COUNTS_PER_REV) == (3.3, 1400)
    assert params.NOMINAL_CM_PER_COUNT == pytest.approx(0.007405, abs=1e-6)


def test_the_bus_addresses_and_rates_are_the_boards():
    assert (params.IMU_I2C_ADDRESS, params.ADS1115_I2C_ADDRESS) == (0x68, 0x48)    # AD0 low; ADDR to GND
    assert params.SENSOR_RATE_HZ == params.IMU_RATE_HZ == 100.0
    assert params.MOTOR_WATCHDOG_S == 0.5


def test_every_frame_gap_limit_is_phase_3s():
    # A stall is clamped to the same step everywhere, or heading and the navigation timers disagree about it.
    ms = Phase3Config().max_dt_s * 1000
    assert intersection.MAX_DT_MS == end_of_course.MAX_DT_MS == wheel_speed.MAX_DT_MS == ms


def test_the_odometry_drive_trial_runs_at_lane_keepings_duty():
    assert calibrate_odometry.DEFAULT_DUTY == BASE_SPEED


def test_the_battery_thresholds_are_one_pair():
    assert (BatteryStateMachine().warning_v, BatteryStateMachine().critical_v) == \
        (Power.VOLTAGE_WARNING, Power.VOLTAGE_CRITICAL)
    assert (BatteryStateMachine().hysteresis_v, BatteryStateMachine().confirm) == \
        (Power.HYSTERESIS_V, Power.CONFIRM_SAMPLES)
    # the linkers' fake battery reports against the same thresholds
    assert (FakeBattery.VOLTAGE_WARNING, FakeBattery.VOLTAGE_CRITICAL) == (Power.VOLTAGE_WARNING, Power.VOLTAGE_CRITICAL)
