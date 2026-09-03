"""
pd_control.py

Steering PD Controller

Purpose:
    Convert a validated lane offset into a differential wheel command.

    Extracted from run_pipeline.py's main loop, where _last_error lived as a
    module-level global mutated through `global`. That made the controller
    untestable, and the global was never cleared when drive_state went to
    "stop", so the derivative term carried a stale error across every stop.

CHANGES FROM THE INLINE VERSION

  1. The correction is ramped with the base speed.
     RAMP_SECONDS ramped ramped_base but NOT correction, so while ramp_frac was
     near zero the command was (-correction, +correction): a pure differential
     with no common mode, i.e. a spin in place. Replaying a wall-side start gave
     four consecutive frames of counter-rotating wheels, and frames 0-1 reversed
     a wheel even with no wall present.

  2. The correction is clamped to the ramped base speed, so a wheel can never
     be commanded to reverse while the robot is supposed to be driving forward.

  3. The derivative is seeded on the first frame of a segment instead of
     differencing against 0.0, which removed the guaranteed first-frame
     derivative kick.

  4. Correction is held, not recomputed, when the offset is not valid. Applying
     a PD correction against a frozen error is what produced the post-
     intersection wind-up: the error never changes, so the proportional term
     keeps commanding the same turn for the whole gap.
"""

from dataclasses import dataclass


@dataclass
class PDConfig:
    """
    kp: proportional gain - compares the measured offset to the desired offset (0.0)
    kd: derivative gain - compares the change in measured offset to the change in desired offset (0.0)
    base_speed: constant forward speed, [0.0, 1.0]
    ramp_seconds: soft-start ramp from 0 -> base_speed on every resume
    offset_trim: constant added to the measured offset, same units as offset
    hold_decay: per-frame decay applied to the held correction while the offset
                is invalid. 1.0 = hold indefinitely, 0.0 = drop immediately.
                0.90 bleeds a stale turn off over roughly 10 frames.
    """
    kp:           float = 0.40
    kd:           float = 0.05
    base_speed:   float = 0.45
    ramp_seconds: float = 0.75
    offset_trim:  float = 0.0
    hold_decay:   float = 0.90


class PDController:
    """
    One instance per run. Call update() every frame, stop() when halted.
    """

    def __init__(self, config: PDConfig | None = None):
        self._cfg = config or PDConfig()
        self._last_error: float | None = None    # None => start of a segment
        self._last_correction: float = 0.0
        self._ramp_start_ts: float | None = None

        # Last update() term breakdown, for the tracer. Read-only; nothing in
        # the control path consumes it.
        self.last_diag: dict = {}

    def stop(self) -> tuple:
        """
        Purpose:
            Halt and arm the soft start for the next resume.

        Output:
            (left_speed, right_speed) == (0.0, 0.0)
        """
        self._ramp_start_ts = None
        self._last_error = None       # clear the derivative across the stop
        self._last_correction = 0.0
        self.last_diag = {
            "ramp_frac": 0.0, "ramped_base": 0.0, "error": None,
            "derivative": None, "correction_raw": 0.0, "correction": 0.0,
            "clamped": False, "held": False, "stopped": True,
        }
        return 0.0, 0.0

    def update(
            self,
            offset: float,
            offset_valid: bool,
            now_s: float,
        ) -> tuple:
        """
        Purpose:
            Produce one differential wheel command.

        Inputs:
            offset: lane offset, canonical sign convention
                    positive = robot LEFT of lane centre, must steer RIGHT
            offset_valid: EstimationPacket.lane_offset_valid. When False the
                    value is dead-reckoned and carries no new information.
            now_s: monotonic timestamp, seconds (time.perf_counter())

        Output:
            (left_speed, right_speed), each in [-1.0, 1.0]
        """
        cfg = self._cfg

        if self._ramp_start_ts is None:
            self._ramp_start_ts = now_s

        ramp_frac = min(1.0, (now_s - self._ramp_start_ts) / max(cfg.ramp_seconds, 1e-6))
        ramped_base = cfg.base_speed * ramp_frac

        error = None
        derivative = None
        seeded = False
        if offset_valid:
            error = offset + cfg.offset_trim
            if self._last_error is None:
                self._last_error = error          # seed: no first-frame kick
                seeded = True
            derivative = error - self._last_error
            correction = (error * cfg.kp) + (derivative * cfg.kd)
            self._last_error = error
        else:
            # No fresh measurement. Decay the last correction rather than
            # re-applying a proportional term against a frozen error.
            correction = self._last_correction * cfg.hold_decay

        correction_raw = correction

        # Ramp the correction alongside the base, then clamp so neither wheel
        # can be commanded to reverse while driving forward.
        correction *= ramp_frac
        pre_clamp = correction
        correction = max(-ramped_base, min(ramped_base, correction))
        self._last_correction = correction

        self.last_diag = {
            "ramp_frac":      ramp_frac,
            "ramped_base":    ramped_base,
            "error":          error,
            "derivative":     derivative,
            "correction_raw": correction_raw,
            "correction":     correction,
            "clamped":        abs(pre_clamp - correction) > 1e-9,
            "held":           not offset_valid,
            "seeded":         seeded,
            "stopped":        False,
        }

        # Positive correction => left wheel slows, right wheel speeds up.
        return ramped_base - correction, ramped_base + correction