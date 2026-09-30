"""
test_maneuver.py  --  src/maneuver.py

The drive trial's state machine against a simulated robot (sim_robot): the
full sequence, the gyro bias and yaw sign it measures, straightness with a
dragging wheel, the 180 turn's pass / fail, and every safety stop. The sim
steps at 25 FPS on a fake clock.

--software  Pure logic, no hardware.
"""
from dataclasses import replace

import pytest

from src.maneuver import (
    ABORTED, DONE, FORWARD_1, FORWARD_2, STOP_1, TURN, TURN_SETTLE, Maneuver, ManeuverConfig, Tick,
)
from src.tests.sim_robot import FakeClock, SimRobot

DT = 0.04                                  # 25 FPS, the pipeline's rate on the Pi without rendering
CFG = ManeuverConfig(leg_counts=400, settle_s=0.4)


def trial(cfg=CFG, max_ticks=4000, dts=None, each=None, **robot):
    """Run a Maneuver against a SimRobot to the end; returns (maneuver, robot, records). each(m, bot) runs before every tick."""
    clock = FakeClock()
    bot = SimRobot(clock, **robot)
    m = Maneuver(cfg)
    t0, records = clock(), []
    for i in range(max_ticks):
        if m.done:
            break
        if each is not None:
            each(m, bot)
        clock.now += DT if dts is None else dts(i)
        sample, _, enc = bot.read()
        cmd = m.step(Tick(clock() - t0, 0.0 if i == 0 else (DT if dts is None else dts(i)),
                          sample.yaw_rate_dps, sample.lateral_accel_mps2,
                          enc.left_count, enc.right_count, enc.left_cps, enc.right_cps))
        bot.brake() if cmd.brake else bot.drive(cmd.left, cmd.right)
        records.append(dict(m.record))
    return m, bot, records


# =============================================================================
# The whole trial
# =============================================================================

@pytest.mark.software
def test_the_trial_runs_every_step_in_order_and_turns_the_robot_around():
    m, bot, recs = trial()
    steps = [r["step"] for r in recs]
    order = [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]]
    assert order == ["settle", "pulse_left", "pulse_left_rest", "pulse_right", "pulse_right_rest",
                     FORWARD_1, STOP_1, TURN, TURN_SETTLE, FORWARD_2, "stop_2", DONE]
    rep = m.report()
    assert rep["completed"] and rep["abort_reason"] is None
    assert bot.heading_deg == pytest.approx(180, abs=CFG.turn_tolerance_deg)   # the real body turned
    assert rep["turn"]["success"] and rep["turn"]["reached"]
    assert rep["forward_1"]["ended_by"] == rep["forward_2"]["ended_by"] == "counts"


@pytest.mark.software
def test_done_commands_stop_forever():
    m, _, _ = trial()
    cmd = m.step(Tick(99.0, DT, 0.0, 0.0, 0, 0, 0.0, 0.0))
    assert m.step_name == DONE and (cmd.left, cmd.right) == (0.0, 0.0)


# =============================================================================
# Settle and the yaw sign
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("bias", [-1.1, 0.0, 2.5])
def test_settle_measures_the_gyro_bias_and_the_accel_baseline(bias):
    m, _, _ = trial(bias_dps=bias, accel_baseline=-0.85)
    s = m.report()["settle"]
    assert s["gyro_bias_measured_dps"] == pytest.approx(bias)
    assert s["gyro_bias_configured_dps"] == CFG.gyro_bias_dps
    assert s["lateral_accel_baseline"] == pytest.approx(-0.85)
    assert s["frames"] == pytest.approx(CFG.settle_s / DT, abs=1)


@pytest.mark.software
def test_a_wrong_configured_bias_is_replaced_by_the_measured_one():
    # Configured far off: the turn would be wrong by bias x time if it were used
    m, bot, _ = trial(cfg=replace(CFG, gyro_bias_dps=+8.0), bias_dps=-1.1)
    assert m.report()["turn"]["success"] and bot.heading_deg == pytest.approx(180, abs=5)


@pytest.mark.software
@pytest.mark.parametrize("plus", ["left", "right"])
def test_the_pulses_find_the_yaw_sign_either_way_and_the_turn_still_goes_left(plus):
    m, bot, _ = trial(imu_plus_is=plus)
    y = m.report()["yaw_sign"]
    assert y["plus_yaw_is"] == plus
    assert (y["left_deg"] > 0) == (plus == "left") and (y["right_deg"] > 0) == (plus == "right")
    assert bot.heading_deg == pytest.approx(180, abs=5)        # + = left on the body, whatever the IMU says


@pytest.mark.software
def test_pulses_that_barely_turn_stop_the_trial_before_it_drives():
    m, bot, recs = trial(deg_per_count=0.001)                 # the gyro sees almost nothing
    rep = m.report()
    assert m.step_name == ABORTED and "yaw sign unclear" in rep["abort_reason"]
    assert FORWARD_1 not in {r["step"] for r in recs}


@pytest.mark.software
def test_pulses_that_read_the_same_sign_both_ways_stop_the_trial():
    # The gyro's drift jumps once settling is over, so both pulses read as + turns
    def drift(m, bot):
        if m.step_name != "settle":
            bot.bias = 120.0
    m, _, recs = trial(each=drift)
    y = m.report()["yaw_sign"]
    assert y["left_deg"] > 0 and y["right_deg"] > 0
    assert m.step_name == ABORTED and "opposite signs" in m.report()["abort_reason"]
    assert FORWARD_1 not in {r["step"] for r in recs}


@pytest.mark.software
def test_a_dead_drivetrain_is_named_in_the_yaw_sign_stop():
    m, _, _ = trial(stalled=True)
    assert "encoders didn't move either" in m.report()["abort_reason"]


# =============================================================================
# Legs
# =============================================================================

@pytest.mark.software
def test_a_dragging_wheel_is_corrected_back_to_straight():
    loose, bot_loose, _ = trial(cfg=replace(CFG, kp_counts=0.0, kp_heading=0.0, leg_counts=800),
                                right_gain=0.85)
    held, bot_held, _ = trial(cfg=replace(CFG, leg_counts=800), right_gain=0.85)
    drift = lambda m: abs(m.report()["forward_1"]["heading_end_deg"])
    assert drift(held) < drift(loose) / 3
    leg = held.report()["forward_1"]
    assert leg["max_c_counts"] > 0 and leg["max_c_heading"] > 0          # both terms worked
    assert abs(leg["imbalance_pct"]) < abs(loose.report()["forward_1"]["imbalance_pct"])


@pytest.mark.software
def test_each_correction_steers_back_on_its_own():
    for kp in ({"kp_heading": 0.0}, {"kp_counts": 0.0}):
        m, _, _ = trial(cfg=replace(CFG, leg_counts=800, **kp), right_gain=0.85)
        loose, _, _ = trial(cfg=replace(CFG, leg_counts=800, kp_counts=0.0, kp_heading=0.0), right_gain=0.85)
        assert abs(m.report()["forward_1"]["heading_end_deg"]) < abs(loose.report()["forward_1"]["heading_end_deg"])


@pytest.mark.software
def test_corrections_never_command_below_min_speed_or_past_max_correction():
    _, _, recs = trial(cfg=replace(CFG, kp_counts=1.0), right_gain=0.5)
    # The frame that enters a leg still commands stop; the rest drive
    legs = [r for r in recs if r["step"] in (FORWARD_1, FORWARD_2) and r["cmd_left"] != 0]
    assert all(CFG.min_speed <= r["cmd_left"] <= 1.0 and CFG.min_speed <= r["cmd_right"] <= 1.0 for r in legs)
    assert all(abs(r["cmd_left"] - r["cmd_right"]) <= 2 * CFG.max_correction + 1e-9 for r in legs)


@pytest.mark.software
def test_a_slow_leg_speed_is_held_at_min_speed_while_correcting():
    _, _, recs = trial(cfg=replace(CFG, speed=0.3, leg_counts=800), right_gain=0.6)
    legs = [r for r in recs if r["step"] in (FORWARD_1, FORWARD_2) and r["cmd_left"] != 0]
    assert min(min(r["cmd_left"], r["cmd_right"]) for r in legs) == CFG.min_speed   # the floor was hit, not passed


@pytest.mark.software
def test_a_leg_that_cannot_reach_its_counts_is_capped_in_time():
    m, _, _ = trial(cfg=replace(CFG, leg_counts=10**6, leg_max_s=1.0))
    leg = m.report()["forward_1"]
    assert leg["ended_by"] == "time cap" and leg["duration_s"] == pytest.approx(1.0, abs=DT)


# =============================================================================
# The 180
# =============================================================================

@pytest.mark.software
def test_the_turn_slows_over_the_last_band():
    _, _, recs = trial()
    turn = [r for r in recs if r["step"] == TURN and r["cmd_right"] != 0]      # past the entry frame
    early = [r for r in turn if r["turn_deg"] < CFG.turn_target_deg - CFG.turn_slow_band_deg - 10]
    late = [r for r in turn if r["turn_deg"] > CFG.turn_target_deg - CFG.turn_slow_band_deg + 5]
    assert early[0]["cmd_right"] == CFG.turn_speed and late[-1]["cmd_right"] == CFG.turn_slow_speed


@pytest.mark.software
def test_overshoot_past_the_tolerance_fails_the_turn_but_the_trial_goes_on():
    fast = replace(CFG, turn_slow_speed=0.9, turn_speed=0.9, turn_tolerance_deg=1.0)
    m, _, _ = trial(cfg=fast)
    t = m.report()["turn"]
    assert t["reached"] and not t["success"] and "outside" in t["reason"]
    assert t["overshoot_deg"] > 1.0 and m.step_name == DONE


@pytest.mark.software
def test_coasting_after_the_turn_is_counted_as_overshoot():
    m, bot, _ = trial(lag_s=0.15, brake_lag_s=0.15)            # a brake no better than coasting
    t = m.report()["turn"]
    assert t["final_deg"] > t["deg_at_stop"] + 1.0           # it kept turning after the stop
    assert t["overshoot_deg"] == pytest.approx(t["final_deg"] - CFG.turn_target_deg)
    assert bot.heading_deg == pytest.approx(t["final_deg"], abs=2.0)


@pytest.mark.software
def test_every_still_step_brakes_and_every_driving_step_drives():
    _, _, recs = trial()
    for r in recs:
        still = r["cmd_left"] == 0 and r["cmd_right"] == 0
        assert r["brake"] == int(still), r


@pytest.mark.software
def test_braking_holds_the_turn_inside_the_tolerance_that_coasting_misses():
    # Coasting as on the 2026-09-30 trial: ~0.15 s to spin down, 7.9 deg past the 180
    coasting, _, _ = trial(lag_s=0.15, brake_lag_s=0.15)
    braked, bot, _ = trial(lag_s=0.15, brake_lag_s=0.02)
    assert coasting.report()["turn"]["overshoot_deg"] > CFG.turn_tolerance_deg
    assert not coasting.report()["turn"]["success"]
    t = braked.report()["turn"]
    assert t["success"] and abs(t["overshoot_deg"]) <= CFG.turn_tolerance_deg
    assert bot.heading_deg == pytest.approx(180, abs=CFG.turn_tolerance_deg)


@pytest.mark.software
def test_a_turn_too_slow_for_the_timeout_stops_the_trial():
    m, _, _ = trial(cfg=replace(CFG, turn_speed=0.26, turn_slow_speed=0.26, turn_timeout_s=1.0))
    rep = m.report()
    assert m.step_name == ABORTED and not rep["turn"]["success"] and "turn_timeout_s" in rep["abort_reason"]


@pytest.mark.software
def test_the_turn_reports_encoder_counts_per_degree():
    m, _, _ = trial()
    t = m.report()["turn"]
    assert t["left_counts"] < 0 < t["right_counts"]            # spun left in place
    assert t["counts_per_deg"] == pytest.approx((abs(t["left_counts"]) + t["right_counts"]) / 2 / t["final_deg"], abs=1e-3)


# =============================================================================
# Safety
# =============================================================================

@pytest.mark.software
def test_a_late_frame_stops_the_trial():
    m, _, recs = trial(dts=lambda i: 0.8 if i == 60 else DT)
    assert m.step_name == ABORTED and "frame gap" in m.report()["abort_reason"]
    assert (recs[-1]["cmd_left"], recs[-1]["cmd_right"]) == (0.0, 0.0)


@pytest.mark.software
def test_a_stall_while_driving_stops_the_trial():
    clock = FakeClock()
    bot = SimRobot(clock)
    m = Maneuver(CFG)
    t0 = clock()
    while not m.done:
        clock.now += DT
        if m.step_name == FORWARD_1:
            bot.stalled = True                               # the wheels jam once the leg begins
        s, _, e = bot.read()
        bot.drive(*(lambda c: (c.left, c.right))(m.step(Tick(clock() - t0, DT, s.yaw_rate_dps, 0.0,
                                                               e.left_count, e.right_count,
                                                               e.left_cps, e.right_cps))))
    assert "stall" in m.report()["abort_reason"] and "forward_1" in m.report()["abort_reason"]


@pytest.mark.software
def test_a_turn_past_the_abort_angle_is_stopped():
    m, _, _ = trial(cfg=replace(CFG, turn_target_deg=300.0, turn_abort_deg=270.0))
    assert m.step_name == ABORTED and "turn_abort_deg" in m.report()["abort_reason"]


@pytest.mark.software
def test_the_run_time_limit_and_an_outside_abort_both_stop_it():
    m, _, _ = trial(cfg=replace(CFG, max_run_s=1.0))
    assert "max_run_s" in m.report()["abort_reason"]
    m = Maneuver(CFG)
    m.step(Tick(0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0))
    m.abort("interrupted (Ctrl-C)")
    assert m.done and m.report()["abort_reason"] == "interrupted (Ctrl-C) (during settle)"
    assert (m.step(Tick(0.1, DT, 0.0, 0.0, 0, 0, 0.0, 0.0)).left, m.record["step"]) == (0.0, ABORTED)


@pytest.mark.software
def test_no_imu_while_settling_stops_before_moving():
    m, bot, _ = trial(imu=False)
    assert "no IMU readings" in m.report()["abort_reason"] and not any(any(c) for c in bot.commands)
