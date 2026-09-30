"""Drive trial: settle, check the yaw sign, drive a leg, turn 180, drive back, as a pure state machine.

Purpose:
    maneuver_linker's control logic, kept apart from the camera, the motors
    and the sensors so it runs and tests anywhere. Each call to step() gets
    one frame's sensor readings and returns one motor command; nothing here
    sleeps or blocks, so the camera keeps running through every maneuver.
    Straightness and the turn rely on the encoders and the IMU only; vision
    is recorded by the linker and never steers. This is a test of the drive,
    IMU and encoder integration, not the Navigation state machine.

Main package:
    Maneuver: step(Tick) -> Command, the per-frame record, and report(), the
    trial's findings (gyro bias at rest, yaw sign, each leg, the 180 turn).

Flow:
    1. SETTLE: still; measure the gyro bias and the lateral-accel baseline.
    2. PULSE_LEFT / PULSE_RIGHT: short spins each way; their yaw sets the sign.
    3. FORWARD: straight on encoder and gyro corrections, to leg_counts (capped by leg_max_s).
    4. STOP, TURN (gyro angle to 180, slowing near it), TURN_SETTLE (overshoot).
    5. FORWARD back, STOP, DONE. Any safety check ends in ABORTED, motors off.
    With hold on, the robot brakes at HOLD after the pulses, each leg and the
    turn, until resume(), so each step can be measured by hand.
"""
import math
from dataclasses import dataclass, field

from src.navigation import BRAKE, Command

# Steps, in order. ABORTED can follow any of them
SETTLE, PULSE_LEFT, PULSE_LEFT_REST, PULSE_RIGHT, PULSE_RIGHT_REST = (
    "settle", "pulse_left", "pulse_left_rest", "pulse_right", "pulse_right_rest")
FORWARD_1, STOP_1, TURN, TURN_SETTLE, FORWARD_2, STOP_2 = (
    "forward_1", "stop_1", "turn", "turn_settle", "forward_2", "stop_2")
DONE, ABORTED = "done", "aborted"
FORWARD_STEPS = (FORWARD_1, FORWARD_2)
HOLD = "hold"

# With hold on, the step each hold comes before: (hold point, what to measure by hand)
HOLD_BEFORE = {
    FORWARD_1: ("after_pulses", "check the heading is back near the start"),
    TURN: ("after_leg_1", "measure leg 1's distance"),
    FORWARD_2: ("after_turn", "measure the turn against the start heading"),
    DONE: ("after_leg_2", "measure leg 2's distance and how far it stopped from the start"),
}


@dataclass(frozen=True)
class ManeuverConfig:
    """
    The drive trial's tuning. Placeholders until course runs set them; every
    field can be overridden per run from maneuver_linker's command line.

    Motor values are TB6612 duty in [0, 1] as MotorController.drive() takes
    them. A correction is added to the left wheel and taken from the right,
    so + steers right.
    """
    speed: float = 0.40                 # forward duty on the legs
    min_speed: float = 0.25             # drive_straight_closed_loop's floor: slower and the N20s stall
    # Leg length in encoder counts, the mean of both wheels. Placeholder:
    # set it on the mat, since counts per meter aren't measured yet
    leg_counts: int = 1500
    leg_max_s: float = 8.0              # backstop: a leg ends here even short of leg_counts
    kp_counts: float = 0.0015           # per count of (right - left) since the leg began; drive.py's kp
    kp_heading: float = 0.01            # per degree turned left since the leg began; placeholder
    max_correction: float = 0.15        # drive.py's max_corr
    turn_speed: float = 0.45
    turn_slow_speed: float = 0.30       # over the last turn_slow_band_deg, to limit overshoot
    turn_slow_band_deg: float = 20.0
    turn_target_deg: float = 171.0
    turn_tolerance_deg: float = 5.0     # pass: final angle within target +/- this
    turn_timeout_s: float = 6.0         # pass: target reached within this
    turn_abort_deg: float = 270.0       # safety: a turn past this is stopped
    # Starting gyro bias, deg/s: the mean yaw at rest from the 2026-09-29
    # phase3_linker run. SETTLE measures it again and uses the measurement
    gyro_bias_dps: float = -1.1
    settle_s: float = 2.0
    pulse_speed: float = 0.40
    pulse_s: float = 0.25
    pulse_rest_s: float = 0.5           # still after each pulse, so its coast is counted
    # A pulse turning less than this can't show the yaw sign: the IMU or motors aren't responding
    pulse_min_deg: float = 3.0
    stop_s: float = 1.0
    max_run_s: float = 60.0
    max_frame_gap_s: float = 0.5        # a longer gap between frames stops the run
    stall_s: float = 0.5                # motors commanded but no encoder counts for this long stops the run


@dataclass(frozen=True)
class Tick:
    """One frame's sensor readings. yaw_dps and lateral_accel are None when the IMU gave nothing."""
    t: float                    # s since the trial started
    dt: float                   # s since the previous tick; 0 on the first
    yaw_dps: float | None       # raw gyro Z mean over the window, before the bias
    lateral_accel: float | None
    left_count: int             # cumulative encoder counts, + = forward
    right_count: int
    left_cps: float             # counts per second over the window
    right_cps: float




@dataclass
class _Leg:
    start_t: float
    start_left: int
    start_right: int
    heading_deg: float = 0.0            # turned left since the leg began, from the gyro
    max_c_counts: float = 0.0
    max_c_heading: float = 0.0
    result: dict = field(default_factory=dict)


class Maneuver:
    """
    The trial's steps. Create one per run and call step() once per frame, in
    order; done is True once it reaches DONE or ABORTED.
    """
    RECORD_FIELDS = ("t", "dt", "step", "cmd_left", "cmd_right", "brake", "yaw_dps", "yaw_corrected",
                     "heading_deg", "turn_deg", "leg_progress", "c_counts", "c_heading",
                     "left_count", "right_count", "left_cps", "right_cps", "lateral_accel", "event",
                     "hold_point")

    def __init__(self, cfg: ManeuverConfig = ManeuverConfig(), hold: bool = False) -> None:
        """hold: brake at each HOLD_BEFORE point until resume(), for measuring by hand."""
        self.cfg = cfg
        self.hold = hold
        self._held_s = 0.0                      # time spent holding; not counted against max_run_s
        self._hold_next = None                  # the step a hold resumes into
        self._resume = False
        self._holds: list[dict] = []
        self.step_name = SETTLE
        self.record: dict = {}
        self._step_t = 0.0
        self._bias = cfg.gyro_bias_dps
        self._yaw_sign = None                   # +1 when + yaw is a left turn
        self._settle_yaw, self._settle_accel = [], []
        self._pulse_deg = 0.0
        self._pulses = {}
        self._pulse_counts0 = (0, 0)
        self._legs: dict[str, _Leg] = {}
        self._turn = {}
        self._turn_deg = 0.0
        self._stalled_s = 0.0
        self._abort_reason = None
        self._last_cmd = Command()

    @property
    def done(self) -> bool:
        return self.step_name in (DONE, ABORTED)

    # -------------------------------------------------------------------------
    # One frame
    # -------------------------------------------------------------------------

    def step(self, tick: Tick) -> Command:
        """
        Advance one frame.

        Inputs:
            tick: This frame's readings. Its dt drives every integral and
                timer, so a missing IMU reading (yaw None) adds nothing.

        Outputs:
            The motor command for this frame; BRAKE whenever the robot should be still.

        Side effects:
            Sets self.record, this frame's row for maneuver.csv.
        """
        event = ""
        yaw = None if tick.yaw_dps is None else tick.yaw_dps - self._bias
        self.record = {"c_counts": 0.0, "c_heading": 0.0, "leg_progress": ""}

        cmd = BRAKE
        if not self.done:
            event = self._safety(tick)
        if not self.done:
            # Every path that ends the trial returns BRAKE, so a done trial never drives
            cmd, event = self._advance(tick, yaw, tick.t - self._step_t)
        self._last_cmd = cmd

        heading = self._legs[self.step_name].heading_deg if self.step_name in self._legs else ""
        self.record.update({
            "t": round(tick.t, 4), "dt": round(tick.dt, 4), "step": self.step_name,
            "cmd_left": round(cmd.left, 4), "cmd_right": round(cmd.right, 4), "brake": int(cmd.brake),
            "yaw_dps": "" if tick.yaw_dps is None else round(tick.yaw_dps, 3),
            "yaw_corrected": "" if yaw is None else round(yaw, 3),
            "heading_deg": heading if heading == "" else round(heading, 2),
            "turn_deg": round(self._turn_deg, 2) if self.step_name in (TURN, TURN_SETTLE) else "",
            "left_count": tick.left_count, "right_count": tick.right_count,
            "left_cps": tick.left_cps, "right_cps": tick.right_cps,
            "lateral_accel": "" if tick.lateral_accel is None else round(tick.lateral_accel, 3),
            "event": event,
            "hold_point": self._holds[-1]["point"] if self.holding else "",
        })
        return cmd

    @property
    def holding(self) -> bool:
        return self.step_name == HOLD

    def resume(self) -> None:
        """Continue from a hold on the next step(). A call before a hold begins is dropped by it, so an early press can't skip one."""
        self._resume = True

    def abort(self, reason: str) -> None:
        """Stop the trial from outside (Ctrl-C, a camera failure); the next command is stopped."""
        if not self.done:
            self._abort_reason = f"{reason} (during {self.step_name})"
            self.step_name = ABORTED

    def _safety(self, tick: Tick) -> str:
        """The checks every step shares; returns the abort event, or ''."""
        cfg = self.cfg
        reason = None
        if self.holding:
            self._held_s += tick.dt
        if tick.t - self._held_s > cfg.max_run_s:
            reason = f"run passed max_run_s {cfg.max_run_s:.0f} s of moving time"
        elif tick.dt > cfg.max_frame_gap_s:
            reason = f"frame gap {tick.dt:.2f} s over max_frame_gap_s {cfg.max_frame_gap_s}"
        else:
            moving = abs(self._last_cmd.left) > 0 or abs(self._last_cmd.right) > 0
            if moving and tick.left_cps == 0 and tick.right_cps == 0:
                self._stalled_s += tick.dt
                if self._stalled_s >= cfg.stall_s:
                    reason = (f"stall: motors commanded for {self._stalled_s:.2f} s "
                              "with no encoder counts")
            else:
                self._stalled_s = 0.0
        if reason:
            self.abort(reason)
            return f"ABORT {reason}"
        return ""

    def _go(self, step: str, tick: Tick) -> str:
        if self.hold and step in HOLD_BEFORE and self._hold_next != step:
            point, measure = HOLD_BEFORE[step]
            self._hold_next, self._resume = step, False
            self.step_name, self._step_t = HOLD, tick.t
            self._holds.append({"point": point, "measure": measure, "t": round(tick.t, 3),
                                "left_count": tick.left_count, "right_count": tick.right_count})
            return f"-> {HOLD} {point}: {measure}, then press the start button (or Enter)"
        self._hold_next = None
        self.step_name = step
        self._step_t = tick.t
        if step in FORWARD_STEPS:
            self._legs[step] = _Leg(tick.t, tick.left_count, tick.right_count)
        elif step == TURN:
            self._turn_deg = 0.0
            self._turn = {"start_t": tick.t, "start_left": tick.left_count,
                          "start_right": tick.right_count}
        return f"-> {step}"

    def _advance(self, tick: Tick, yaw: float | None, elapsed: float) -> tuple[Command, str]:
        """The current step's command, moving to the next step when it's finished."""
        cfg, step = self.cfg, self.step_name
        still = BRAKE

        if step == HOLD:
            if not self._resume:
                return still, ""
            self._holds[-1]["held_s"] = round(elapsed, 3)
            return still, self._go(self._hold_next, tick)

        if step == SETTLE:
            if tick.yaw_dps is not None:
                self._settle_yaw.append(tick.yaw_dps)
            if tick.lateral_accel is not None:
                self._settle_accel.append(tick.lateral_accel)
            if elapsed < cfg.settle_s:
                return still, ""
            if not self._settle_yaw:
                self.abort("no IMU readings while settling")
                return still, "ABORT no IMU readings while settling"
            self._bias = sum(self._settle_yaw) / len(self._settle_yaw)
            self._pulse_counts0 = (tick.left_count, tick.right_count)
            return still, self._go(PULSE_LEFT, tick) + f" (bias {self._bias:+.3f} deg/s)"

        if step in (PULSE_LEFT, PULSE_LEFT_REST, PULSE_RIGHT, PULSE_RIGHT_REST):
            if yaw is not None:
                self._pulse_deg += yaw * tick.dt
            p = cfg.pulse_speed
            if step == PULSE_LEFT:
                return (Command(-p, p), "") if elapsed < cfg.pulse_s else (still, self._go(PULSE_LEFT_REST, tick))
            if step == PULSE_RIGHT:
                return (Command(p, -p), "") if elapsed < cfg.pulse_s else (still, self._go(PULSE_RIGHT_REST, tick))
            if elapsed < cfg.pulse_rest_s:
                return still, ""
            if step == PULSE_LEFT_REST:
                self._pulses["left_deg"] = self._pulse_deg
                self._pulse_deg = 0.0
                return still, self._go(PULSE_RIGHT, tick)
            self._pulses["right_deg"] = self._pulse_deg
            return still, self._decide_sign(tick)

        if step in FORWARD_STEPS:
            return self._forward(tick, yaw, elapsed)

        if step in (STOP_1, STOP_2):
            if elapsed < cfg.stop_s:
                return still, ""
            return still, self._go(TURN if step == STOP_1 else DONE, tick)

        if step == TURN:
            return self._turning(tick, yaw, elapsed)

        if step == TURN_SETTLE:
            if yaw is not None:
                self._turn_deg += self._yaw_sign * yaw * tick.dt
            if elapsed < cfg.stop_s:
                return still, ""
            return still, self._finish_turn(tick)
        return still, ""

    # -------------------------------------------------------------------------
    # Steps
    # -------------------------------------------------------------------------

    def _decide_sign(self, tick: Tick) -> str:
        """+1 when a commanded left spin reads + yaw. Both pulses must move and disagree in sign."""
        left, right = self._pulses["left_deg"], self._pulses["right_deg"]
        moved = abs(tick.left_count - self._pulse_counts0[0]) + abs(tick.right_count - self._pulse_counts0[1])
        self._pulses["encoder_counts_moved"] = moved
        m = self.cfg.pulse_min_deg
        if abs(left) < m or abs(right) < m or (left > 0) == (right > 0):
            reason = (f"yaw sign unclear: left pulse {left:+.1f} deg, right pulse {right:+.1f} deg "
                      f"(need opposite signs, each past {m} deg)")
            if moved == 0:
                reason += "; the encoders didn't move either, so the motors or encoders aren't responding"
            self.abort(reason)
            return f"ABORT {reason}"
        self._yaw_sign = 1 if left > 0 else -1
        return self._go(FORWARD_1, tick) + f" (+ yaw = {'left' if self._yaw_sign > 0 else 'right'})"

    def _forward(self, tick: Tick, yaw: float | None, elapsed: float) -> tuple[Command, str]:
        cfg, leg = self.cfg, self._legs[self.step_name]
        if yaw is not None:
            leg.heading_deg += self._yaw_sign * yaw * tick.dt
        dl, dr = tick.left_count - leg.start_left, tick.right_count - leg.start_right
        progress = (dl + dr) / 2.0
        self.record["leg_progress"] = round(progress, 1)
        if progress >= cfg.leg_counts or elapsed >= cfg.leg_max_s:
            leg.result = {"left_counts": dl, "right_counts": dr, "duration_s": round(elapsed, 3),
                          "ended_by": "counts" if progress >= cfg.leg_counts else "time cap",
                          "heading_end_deg": round(leg.heading_deg, 2),
                          "max_c_counts": round(leg.max_c_counts, 4),
                          "max_c_heading": round(leg.max_c_heading, 4)}
            return BRAKE, self._go(STOP_1 if self.step_name == FORWARD_1 else STOP_2, tick)
        # Left ahead of right, or a drift left, both steer back: + = steer right
        c_counts = cfg.kp_counts * (dr - dl)
        c_heading = cfg.kp_heading * leg.heading_deg
        c = max(-cfg.max_correction, min(cfg.max_correction, c_counts + c_heading))
        leg.max_c_counts = max(leg.max_c_counts, abs(c_counts))
        leg.max_c_heading = max(leg.max_c_heading, abs(c_heading))
        self.record.update(c_counts=round(c_counts, 4), c_heading=round(c_heading, 4))
        clamp = lambda v: max(cfg.min_speed, min(1.0, v))
        return Command(clamp(cfg.speed + c), clamp(cfg.speed - c)), ""

    def _turning(self, tick: Tick, yaw: float | None, elapsed: float) -> tuple[Command, str]:
        cfg = self.cfg
        if yaw is not None:
            self._turn_deg += self._yaw_sign * yaw * tick.dt
        if self._turn_deg >= cfg.turn_abort_deg:
            reason = f"turn passed turn_abort_deg {cfg.turn_abort_deg:.0f}"
            self._turn.update(reached=False, reason=reason)
            self.abort(reason)
            return BRAKE, f"ABORT {reason}"
        if self._turn_deg >= cfg.turn_target_deg:
            self._turn.update(reached=True, time_to_target_s=round(elapsed, 3),
                              deg_at_stop=round(self._turn_deg, 2))
            return BRAKE, self._go(TURN_SETTLE, tick)
        if elapsed >= cfg.turn_timeout_s:
            reason = f"turn reached {self._turn_deg:.1f} deg of {cfg.turn_target_deg:.0f} in turn_timeout_s {cfg.turn_timeout_s:.0f}"
            self._turn.update(reached=False, reason=reason, final_deg=round(self._turn_deg, 2),
                              success=False)
            self.abort(reason)
            return BRAKE, f"ABORT {reason}"
        s = cfg.turn_speed if cfg.turn_target_deg - self._turn_deg > cfg.turn_slow_band_deg else cfg.turn_slow_speed
        return Command(-s, s), ""                  # spin left in place

    def _finish_turn(self, tick: Tick) -> str:
        cfg, t = self.cfg, self._turn
        final = self._turn_deg
        dl, dr = tick.left_count - t["start_left"], tick.right_count - t["start_right"]
        t.update(final_deg=round(final, 2), overshoot_deg=round(final - cfg.turn_target_deg, 2),
                 left_counts=dl, right_counts=dr,
                 counts_per_deg=round((abs(dl) + abs(dr)) / 2.0 / final, 3) if final > 0 else None,
                 success=abs(final - cfg.turn_target_deg) <= cfg.turn_tolerance_deg)
        if not t["success"]:
            t["reason"] = (f"final {final:.1f} deg outside {cfg.turn_target_deg:.0f} "
                           f"+/- {cfg.turn_tolerance_deg:.0f}")
        return self._go(FORWARD_2, tick) + f" (turn {'PASS' if t['success'] else 'FAIL'} {final:.1f} deg)"

    # -------------------------------------------------------------------------
    # Findings
    # -------------------------------------------------------------------------

    def report(self) -> dict:
        """What the trial found; sections it never reached are missing."""
        cfg = self.cfg
        out = {"completed": self.step_name == DONE, "abort_reason": self._abort_reason}
        if self._settle_yaw:
            n = len(self._settle_yaw)
            mean = sum(self._settle_yaw) / n
            out["settle"] = {
                "gyro_bias_measured_dps": round(mean, 3),
                "gyro_bias_configured_dps": cfg.gyro_bias_dps,
                "gyro_noise_sd_dps": round(math.sqrt(sum((v - mean) ** 2 for v in self._settle_yaw) / n), 3),
                "frames": n,
            }
            if self._settle_accel:
                a = self._settle_accel
                out["settle"]["lateral_accel_baseline"] = round(sum(a) / len(a), 3)
        if self._pulses:
            out["yaw_sign"] = {**{k: round(v, 2) for k, v in self._pulses.items()},
                               "plus_yaw_is": None if self._yaw_sign is None
                               else ("left" if self._yaw_sign > 0 else "right")}
        for name, leg in self._legs.items():
            if leg.result:
                l, r = leg.result["left_counts"], leg.result["right_counts"]
                out[name] = {**leg.result,
                             "imbalance_pct": round(100.0 * (l - r) / max(abs(l), abs(r), 1), 2)}
        if self._turn:
            out["turn"] = {k: v for k, v in self._turn.items() if not k.startswith("start_")}
        if self._holds:
            out["holds"] = [dict(h) for h in self._holds]
        return out
