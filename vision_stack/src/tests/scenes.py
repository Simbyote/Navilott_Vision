"""Shared test scenes and the config synthetic frames run under.

Purpose:
    Synthetic frames are drawn already undistorted: a mark drawn straight is
    straight. Running them through MEASURED would warp them with the real
    lens model and move every mark, so tests that feed synthetic frames use
    SCENE_CONFIG instead. Tests on real captures use MEASURED.

Main package:
    SCENE_CONFIG: MEASURED with undistortion off. Every other stage's tuning,
        the color branch included, is MEASURED's, so a synthetic test
        exercises the robot's gates.
"""
from dataclasses import replace

from src.config import MEASURED

SCENE_CONFIG = replace(
    MEASURED,
    preprocess = replace(MEASURED.preprocess, calibration_path = None),
)
