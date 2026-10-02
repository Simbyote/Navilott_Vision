"""Intersection linker: one intersection per run, straight, left or right, with the whole chain driving, recorded and judged.

Purpose:
    Proves the intersection sequences (intersection.py's to the line, turn,
    exit) on the mat, one maneuver at a time, before a full course. Each
    sequence is a navigation_linker run (camera -> Phase 2 -> traced Phase 3
    -> Navigation -> motors, recorded and rendered afterwards) with a
    one-step route, so the robot drives exactly as in a course run: it
    approaches a real stop line, crosses or turns, and the run ends once
    lane keeping has held the lane on vision for SETTLE_S after the
    crossing. SequenceWatch follows the run frame by frame and judge() gives
    each sequence PASS or CHECK: one intersection counted, a turn ended on
    the gyro (not its time limit), the lane found again, the heading turned
    within HEADING_TOLERANCE_DEG of the maneuver's (left -90, right +90,
    straight 0; + = turned right), no command braked for the contract.

Main package:
    SequenceWatch: navigation_linker.run's stop_when; ends the run and
        gathers the findings.
    judge(): PASS / CHECK and why.
    run_sequence(): one maneuver, one run folder.
    cli(): python3 -m src.intersection_linker straight|left|right|all --camera

Flow:
    1. Per maneuver: place the robot in its lane, the stop line ahead in
       view; start button (or Enter with --no-button).
    2. navigation_linker.run with Route((maneuver,)) and a SequenceWatch.
    3. The run ends (lane held after the crossing, the cap, Ctrl-C, ...);
       motors stop first. Findings and verdict into the sequence's report.
    4. runs/intersection_<time>/<maneuver>/ holds each run; summary.txt and
       report.json beside them sum up every sequence.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

from src.capture.camera import CaptureError
from src.config import MEASURED, MEASURED_ESTIMATION, PipelineConfig
from src.debugger.live_view import CameraFrameSource, DirectoryFrameSource, VideoFrameSource
from src.estimation.estimation import LANE_VISION, Phase3Config
from src.maneuver_linker import _NoMotors
from src.navigation.intersection import STAGE_EXIT, STAGE_TO_LINE, STAGE_TURN, TURN_END_GYRO
from src.navigation.navigation import RULE_INTERSECTION, RULE_LANE_KEEPING, Navigation
from src.navigation.route import LEFT, MANEUVERS, RIGHT, STRAIGHT, Route
from src.navigation_linker import run as navigation_run
from src.params import FPS, FRAME_H, FRAME_W, RUNS_DIR
from src.peripherals.sensing import Sensors

# Lane keeping on vision this long after the crossing ends the sequence:
# the robot found its lane again and kept it
SETTLE_S = 1.0
# A sequence's backstop: approach, the line, a turn's limit, the exit, settling
SEQUENCE_MAX_S = 20.0
# Heading each sequence should end on, deg; + = turned right
EXPECTED_DEG = {STRAIGHT: 0.0, LEFT: -90.0, RIGHT: 90.0}
# A sequence within this of its expected heading passes. Wide on purpose:
# lane keeping straightens the robot out after the turn, and the trial
# checks the sequence works; TURN_TARGET_DEG is tuned from the numbers
HEADING_TOLERANCE_DEG = 20.0
SEQUENCE_DONE = "sequence done (lane held after the crossing)"


class SequenceWatch:
    """
    navigation_linker.run's stop_when for one sequence: follows its nav.csv
    rows, ends the run SETTLE_S into lane keeping on vision after the
    crossing, and keeps the findings.

    Inputs:
        gyro_bias_dps: Subtracted from each row's yaw_rate (raw) for the heading.
        settle_s: See SETTLE_S.
    """
    def __init__(self, gyro_bias_dps: float = 0.0, settle_s: float = SETTLE_S) -> None:
        self.gyro_bias_dps, self.settle_s = gyro_bias_dps, settle_s
        self.stage_s = {STAGE_TO_LINE: 0.0, STAGE_TURN: 0.0, STAGE_EXIT: 0.0}
        self.steps = 0                      # intersections the route counted
        self.turn_end = None
        self.heading = 0.0                  # deg turned since the crossing started, from the IMU
        self.heading_at_turn_end = None
        self.started = self.crossed = False
        self._settled = 0.0
        self._last_t = None

    def __call__(self, n: dict) -> str | None:
        t = float(n["t"])
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        rule, stage = n.get("rule"), n.get("stage") if n.get("rule") == RULE_INTERSECTION else ""
        step = str(n.get("step") or "0/").split("/")[0]
        self.steps = max(self.steps, int(step) if step.isdigit() else 0)
        if stage:
            self.started = True
            self.stage_s[stage] += dt
        if self.started:
            self.heading += (float(n.get("yaw_rate") or 0.0) - self.gyro_bias_dps) * dt
        if n.get("turn_end") and self.turn_end is None:
            self.turn_end, self.heading_at_turn_end = n["turn_end"], self.heading
        if self.started and rule == RULE_LANE_KEEPING:
            self.crossed = True
            self._settled = self._settled + dt if n.get("lane_status") == LANE_VISION else 0.0
            if self._settled >= self.settle_s:
                return SEQUENCE_DONE
        return None

    def findings(self, maneuver: str) -> dict:
        """What the sequence did, for judge() and the report."""
        return {"maneuver": maneuver, "intersections": self.steps,
                "stage_s": {k: round(v, 2) for k, v in self.stage_s.items()},
                "turn_end": self.turn_end,
                "heading_at_turn_end_deg": None if self.heading_at_turn_end is None
                else round(self.heading_at_turn_end, 1),
                "heading_deg": round(self.heading, 1), "expected_deg": EXPECTED_DEG[maneuver],
                "lane_back": self.crossed}


def judge(findings: dict) -> tuple[str, list[str]]:
    """
    PASS or CHECK for one sequence, with every reason for a CHECK.

    Inputs:
        findings: SequenceWatch.findings() plus "ended_by" and "rejected"
            (commands braked for the contract) from the run's report.
    """
    problems = []
    if findings["intersections"] != 1:
        problems.append(f"{findings['intersections']} intersections counted, not 1: "
                        "the stop line was missed or something else passed for one")
    if findings["maneuver"] != STRAIGHT and findings["turn_end"] != TURN_END_GYRO:
        problems.append(f"the turn ended on {findings['turn_end'] or 'nothing'}, not the gyro target: "
                        "check the IMU and IMU_YAW_SIGN")
    if findings["ended_by"] != SEQUENCE_DONE:
        problems.append(f"the lane wasn't held after the crossing (ended by {findings['ended_by']})")
    off = findings["heading_deg"] - findings["expected_deg"]
    if abs(off) > HEADING_TOLERANCE_DEG:
        problems.append(f"turned {findings['heading_deg']:+.1f} deg, {off:+.1f} from {findings['expected_deg']:+.0f} "
                        f"(tolerance {HEADING_TOLERANCE_DEG:.0f})")
    if findings.get("rejected"):
        problems.append(f"{findings['rejected']} commands broke the contract and were braked")
    return ("PASS" if not problems else "CHECK"), problems


def summary_lines(findings: dict) -> list[str]:
    """summary.txt's lines for one sequence."""
    verdict, problems = judge(findings)
    s = findings["stage_s"]
    lines = [f"[{findings['maneuver'].upper()}] {verdict}   ended by {findings['ended_by']}",
             f"  stages        to the line {s[STAGE_TO_LINE]:.2f} s, turn {s[STAGE_TURN]:.2f} s, exit {s[STAGE_EXIT]:.2f} s",
             f"  intersections {findings['intersections']}"]
    if findings["maneuver"] != STRAIGHT:
        lines.append(f"  turn          ended on {findings['turn_end']}, at {findings['heading_at_turn_end_deg']} deg")
    lines.append(f"  heading       {findings['heading_deg']:+.1f} deg turned, expected {findings['expected_deg']:+.0f}")
    return lines + [f"  CHECK: {p}" for p in problems]


def run_sequence(maneuver: str, source, sensors, motor, config: PipelineConfig = MEASURED,
                 p3_config: Phase3Config = MEASURED_ESTIMATION, out_dir: str = str(RUNS_DIR / "intersection"),
                 system=None, clock=time.perf_counter, max_run_s: float = SEQUENCE_MAX_S,
                 motors_on: bool = True, render: bool = True, settle_s: float = SETTLE_S) -> dict:
    """
    One maneuver: a navigation_linker run with a one-step route, watched.

    Inputs:
        maneuver: STRAIGHT, LEFT or RIGHT.
        source, sensors, motor, system, clock, motors_on, render: As for
            navigation_linker.run; it closes the source and stops the
            sensors and the motors when it ends.
        config, p3_config: Phase 2 and 3 tuning; p3_config's gyro bias is
            Navigation's and the heading's too.
        out_dir: This sequence's run folder.
        max_run_s: The sequence's backstop.
        settle_s: See SETTLE_S.

    Outputs:
        The findings (SequenceWatch's, the run's ended_by and contract
        count) with "verdict", also written to the folder's sequence.json.
    """
    watch = SequenceWatch(p3_config.gyro_bias_dps, settle_s)
    nav = Navigation(gyro_bias_dps=p3_config.gyro_bias_dps, route=Route((maneuver,)))
    report = navigation_run(source, sensors, motor, nav, config, p3_config, out_dir, system, clock=clock,
                            max_run_s=max_run_s, motors_on=motors_on, render=render, stop_when=watch)
    findings = {**watch.findings(maneuver), "ended_by": report["ended_by"],
                "rejected": report["nav"]["rejected"]}
    findings["verdict"] = judge(findings)[0]
    with open(os.path.join(out_dir, "sequence.json"), "w") as f:
        json.dump(findings, f, indent=2)
    return findings


def _open(args):
    """The source, sensors, motor and start button for one sequence, as navigation_linker opens them."""
    source = sensors = motor = system = None
    try:
        if args.camera:
            source = CameraFrameSource(FRAME_W, FRAME_H, args.fps or FPS)
            sensors = Sensors(imu=True, encoders=True)
        elif args.video:
            source = VideoFrameSource(args.video, args.fps)
        else:
            source = DirectoryFrameSource(args.frames, args.fps or FPS)
        if args.camera and not args.no_motors:
            import pigpio
            from src.peripherals.drive import MotorController
            motor = MotorController(pigpio.pi())
        else:
            motor = _NoMotors()
        if args.camera and not args.no_button:
            from src.peripherals.system import System
            system = System()
    except (CaptureError, OSError, RuntimeError, ImportError):
        for thing in (motor, sensors, source):
            if thing is not None:
                (thing.stop if hasattr(thing, "stop") else thing.close)()
        raise
    return source, sensors, motor, system


def cli(argv: list[str] | None = None) -> int:
    """
    Command line: python3 -m src.intersection_linker straight|left|right|all --camera

    Outputs:
        0 when every sequence passed, 1 when one needs a check, 2 when the
        source or hardware wouldn't open or a replay was asked for "all".
    """
    ap = argparse.ArgumentParser(description="One intersection per run, straight, left or right, judged.")
    ap.add_argument("maneuver", choices=list(MANEUVERS) + ["all"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", action="store_true", help="the robot drives (with the motors)")
    src.add_argument("--video", metavar="PATH", help="replay a recording: what it would have done; nothing moves")
    src.add_argument("--frames", metavar="DIR", help="replay a frame folder; nothing moves")
    ap.add_argument("--no-motors", action="store_true", help="with --camera: everything but the motors")
    ap.add_argument("--no-button", action="store_true", help="with --camera: Enter starts each sequence")
    ap.add_argument("--max-run-s", type=float, default=SEQUENCE_MAX_S,
                    help=f"each sequence's backstop (default {SEQUENCE_MAX_S:.0f})")
    ap.add_argument("--settle-s", type=float, default=SETTLE_S,
                    help=f"lane kept on vision this long after the crossing ends a sequence (default {SETTLE_S})")
    ap.add_argument("--gyro-bias", type=float, default=MEASURED_ESTIMATION.gyro_bias_dps, metavar="DPS",
                    help=f"gyro Z at rest, + = right (default {MEASURED_ESTIMATION.gyro_bias_dps}, config.GYRO_BIAS_DPS)")
    ap.add_argument("--cm-per-px", type=float, default=None, metavar="S")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--no-render", action="store_true", help="skip each sequence's video")
    ap.add_argument("--out", default=None, metavar="DIR")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    maneuvers = list(MANEUVERS) if args.maneuver == "all" else [args.maneuver]
    if len(maneuvers) > 1 and not args.camera:
        print("a replay holds one intersection: name its maneuver instead of all")
        return 2
    p3_config = replace(MEASURED_ESTIMATION, gyro_bias_dps=args.gyro_bias, cm_per_px=args.cm_per_px)
    motors_on = bool(args.camera and not args.no_motors)
    out_dir = args.out or str(RUNS_DIR / ("intersection_" + time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out_dir, exist_ok=True)
    print(f"output   {out_dir}\nmotors   {'ON' if motors_on else 'OFF (dry run)'}   "
          f"cap {args.max_run_s:.0f} s per sequence   gyro bias {args.gyro_bias:+.2f} deg/s")

    results, lines = [], []
    try:
        for m in maneuvers:
            print(f"\n{m.upper()}: robot in its lane, pointing along it, the stop line ahead in view.")
            try:
                source, sensors, motor, system = _open(args)
            except (CaptureError, OSError, RuntimeError, ImportError) as exc:
                print(f"source / hardware error: {exc!r}")
                return 2
            if args.camera and system is None:
                input("  Enter to go ")
            elif system is not None:
                print("  press the start button")
            findings = run_sequence(m, source, sensors, motor, MEASURED, p3_config, os.path.join(out_dir, m),
                                    system, max_run_s=args.max_run_s, motors_on=motors_on,
                                    render=not args.no_render, settle_s=args.settle_s)
            results.append(findings)
            lines += summary_lines(findings)
            print("\n".join(summary_lines(findings)))
    except KeyboardInterrupt:
        print("\nstopped")

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"gyro_bias_dps": args.gyro_bias, "motors": motors_on, "sequences": results}, f, indent=2)
    return 0 if results and all(r["verdict"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(cli())
