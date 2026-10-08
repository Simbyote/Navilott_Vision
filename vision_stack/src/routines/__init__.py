"""Hardware test routines a teammate runs by following prompts (docs/guides/routines.md).

Purpose:
    Each routine answers one question on the real robot, judged against
    pass criteria agreed on a test request card (docs/routines/request_card.md).
    The shared flow (prompts, trials, conditions, verdict, folder) is in
    harness.py; a routine is a module here with a Routine subclass, listed
    in ROUTINES.

Main package:
    ROUTINES: {name: Routine class}, what make routines lists and
        make routine-<name> runs.
"""
from src.routines.detect_range import DetectRange
from src.routines.figure_eight import FigureEight
from src.routines.imu_check import ImuCheck
from src.routines.lane_offset import LaneOffset
from src.routines.power_profile import PowerProfile
from src.routines.recover_offset import RecoverOffset
from src.routines.stop_distance import StopDistance
from src.routines.tape_check import TapeCheck

ROUTINES = {r.name: r for r in (TapeCheck, StopDistance, PowerProfile, FigureEight, ImuCheck, DetectRange, LaneOffset, RecoverOffset)}
