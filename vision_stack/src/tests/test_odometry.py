"""
test_odometry.py  --  src/estimation/odometry.py, and odometry through Phase 3

Odometer: distance from cumulative counts times cm_per_count, zeroed at the
first frame with counts; the axle's midpoint the wheels' mean (a spin adds
nothing); a frame without counts holds and the next catches up; no scale or
no encoders leaves every distance None; backwards is negative. load_odometry:
the file's scale, None when it's missing, unreadable or not positive.
SensorSample.from_batch carries the batch's counts; Phase3Processor puts the
distances in the packet; MEASURED_ESTIMATION reads the calibration file.

--software  No hardware.
"""
import json
from types import SimpleNamespace

import pytest

from src.estimation.estimation import Phase3Config, Phase3Processor
from src.peripherals.sensing import SensorSample
from src.estimation.odometry import NO_ODOMETRY, Odometer, Odometry, load_odometry
from src.perception.phase2_out import Phase2Output
from src.tests.test_estimation import lane


@pytest.mark.software
def test_distance_is_counts_since_the_first_frame_times_the_scale():
    odo, log = Odometer(0.05), []
    assert odo.update(1000, 2000, log) == Odometry(0.0, 0.0, 0.0)
    assert log == ["[ODOM] zero at counts 1000, 2000"]
    assert odo.update(1200, 2220, log) == Odometry(10.5, 10.0, 11.0)
    assert odo.update(1200, 2220, log).distance_cm == 10.5                  # stopped: unchanged
    assert len(log) == 1


@pytest.mark.software
def test_a_spin_in_place_moves_the_axle_nowhere_and_backwards_is_negative():
    odo = Odometer(0.1)
    odo.update(0, 0, [])
    assert odo.update(300, -300, []) == Odometry(0.0, 30.0, -30.0)
    assert odo.update(100, -500, []).distance_cm == -20.0


@pytest.mark.software
def test_a_frame_without_counts_holds_and_the_next_catches_up():
    odo = Odometer(0.1)
    odo.update(0, 0, [])
    assert odo.update(100, 100, []).distance_cm == 10.0
    assert odo.update(None, None, []).distance_cm == 10.0
    assert odo.record["held"] is True
    assert odo.update(400, 400, []).distance_cm == 40.0                      # cumulative: nothing lost


@pytest.mark.software
@pytest.mark.parametrize("scale, counts", [(None, (10, 10)), (0.1, (None, None)), (0.1, (5, None))])
def test_without_a_scale_or_encoders_there_is_no_distance(scale, counts):
    odo = Odometer(scale)
    assert odo.update(*counts, []) == NO_ODOMETRY
    assert odo.update(*counts, []).distance_cm is None


@pytest.mark.software
def test_the_calibration_file_gives_the_scale(tmp_path):
    p = tmp_path / "odometry.json"
    assert load_odometry(p) is None
    p.write_text(json.dumps({"cm_per_count": 0.0213, "counts_per_cm": 46.9}))
    assert load_odometry(p) == 0.0213
    for bad in ("{", json.dumps({"counts_per_cm": 46.9}), json.dumps({"cm_per_count": 0}),
                json.dumps({"cm_per_count": "x"})):
        p.write_text(bad)
        assert load_odometry(p) is None
        assert load_odometry(p, nominal=0.0148) == 0.0148               # unusable: the nominal scale
    assert load_odometry(tmp_path / "missing.json", nominal=0.0148) == 0.0148
    p.write_text(json.dumps({"cm_per_count": 0.0213}))
    assert load_odometry(p, nominal=0.0148) == 0.0213                     # measured beats nominal


@pytest.mark.software
def test_a_batch_carries_its_cumulative_counts_into_the_sample():
    batch = SimpleNamespace(mean_yaw_dps=1.0, peak_lateral_accel=0.1, left_cps=100.0, right_cps=90.0,
                            left_count=5000, right_count=4900)
    s = SensorSample.from_batch(batch)
    assert (s.left_count, s.right_count) == (5000, 4900)
    del batch.left_count, batch.right_count                                   # an older batch: no counts
    assert SensorSample.from_batch(batch).left_count is None


@pytest.mark.software
def test_phase_3_puts_the_distances_in_the_packet():
    def out(i):
        return Phase2Output([], [lane(frame_id=i, ts=i * 50)], i, i * 50)
    proc = Phase3Processor(Phase3Config(cm_per_count=0.02))
    proc.process(out(0), SensorSample(left_count=100, right_count=200))
    pkt, dbg = proc.process(out(1), SensorSample(left_count=600, right_count=800))
    assert (pkt.distance_cm, pkt.left_distance_cm, pkt.right_distance_cm) == (11.0, 10.0, 12.0)
    assert Phase3Processor().process(out(0), SensorSample(left_count=1, right_count=1))[0].distance_cm is None
    assert Phase3Processor(Phase3Config(cm_per_count=0.02)).process(out(0))[0].distance_cm is None


@pytest.mark.software
def test_the_robots_tuning_reads_the_calibration_file():
    from src import config
    from src.params import NOMINAL_CM_PER_COUNT, ODOMETRY_PATH
    assert config.MEASURED_ESTIMATION.cm_per_count == load_odometry(ODOMETRY_PATH, NOMINAL_CM_PER_COUNT)
