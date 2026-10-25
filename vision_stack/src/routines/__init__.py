"""Hardware test routines a teammate runs by following prompts.

Purpose:
    Each routine answers one question on the real robot, judged against
    criteria agreed up front; each routine's judge() holds them.

Main package:
    ROUTINES: {name: Routine class}, what make routines lists and
    make routine-<name> runs.
"""
from src.routines.course_run import CourseRun
from src.routines.detect_range import DetectRange
from src.routines.figure_eight import FigureEight
from src.routines.frame_budget import FrameBudget
from src.routines.imu_check import ImuCheck
from src.routines.lane_offset import LaneOffset
from src.routines.obstacle_check import ObstacleCheck
from src.routines.obstacle_stop import ObstacleStop
from src.routines.power_profile import PowerProfile
from src.routines.recover_offset import RecoverOffset
from src.routines.stop_distance import StopDistance
from src.routines.stop_sign import StopSign
from src.routines.tape_check import TapeCheck

ROUTINES = {r.name: r for r in (TapeCheck, StopDistance, PowerProfile, FigureEight, ImuCheck, DetectRange, LaneOffset, RecoverOffset, FrameBudget, CourseRun,
                                       ObstacleCheck, ObstacleStop, StopSign)}
