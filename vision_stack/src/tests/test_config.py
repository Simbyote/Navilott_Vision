"""
test_config.py  --  src/config.py and src/tests/scenes.py

MEASURED is the robot's tuning and SCENE_CONFIG is what synthetic frames run
under. These pin what each must hold, that SCENE_CONFIG differs from MEASURED
only by undistortion, and that every consumer reads the one config rather
than a copy of its own.

--software  Config contents, their effect on real stages, the import
            boundary, and the defaults each linker runs with. No camera.
"""
import inspect
import subprocess
import sys
from dataclasses import replace

import cv2
import numpy as np
import pytest

import src.config as config
import src.phase2_linker as p2
import src.phase3_linker as p3
from src.capture.camera import FrameData
from src.config import MEASURED, PipelineConfig
from src.params import CAMERA_CALIB_PATH, FRAME_H, FRAME_W, HSV_RANGES_PATH, PIPELINE_ROOT
from src.perception.color_branch import load_hsv_ranges, run_color_stage
from src.perception.preprocess import preprocess_frame
from src.perception.roi_crop import crop_rois
from src.phase2_linker import run_chain
from src.tests.scenes import SCENE_CONFIG, synthetic_frame


# =============================================================================
# MEASURED
# =============================================================================

@pytest.mark.software
def test_measured_undistorts_with_the_lens_calibration():
    assert MEASURED.preprocess.calibration_path == str(CAMERA_CALIB_PATH)
    # The path has to reach the stage: a grid comes out moved
    grid = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    grid[::20, :] = 255
    grid[:, ::20] = 255
    pre = preprocess_frame(FrameData(grid, 0, 0), MEASURED.preprocess)
    assert not np.array_equal(pre.undistorted, grid)


@pytest.mark.software
def test_measured_runs_the_color_branch_with_the_calibrated_ranges():
    ranges = MEASURED.color.hsv_ranges
    assert ranges is not None and ranges.is_calibrated
    assert ranges == load_hsv_ranges(str(HSV_RANGES_PATH))
    roi = crop_rois(preprocess_frame(FrameData(synthetic_frame([150, 290]), 0, 0),
                                     SCENE_CONFIG.preprocess), SCENE_CONFIG.roi)
    _, debug = run_color_stage(roi, MEASURED.color)
    assert debug["enabled"]


@pytest.mark.software
def test_measured_keeps_the_swept_lane_gates():
    lo = MEASURED.lane_offset
    assert (lo.conf_threshold, lo.min_proximity, lo.max_width_px, lo.min_intensity) \
        == (0.25, 0.05, 45.0, 130.0)


@pytest.mark.software
def test_bare_pipeline_config_is_each_stages_defaults():
    # run_chain's default; tests that mean "defaults" rely on it staying bare
    c = PipelineConfig()
    assert c.preprocess.calibration_path is None
    assert c.color.hsv_ranges is None


# =============================================================================
# SCENE_CONFIG
# =============================================================================

@pytest.mark.software
def test_scene_config_is_measured_without_undistortion():
    assert SCENE_CONFIG.preprocess.calibration_path is None
    # Everything else, the color branch included, is MEASURED's
    assert replace(SCENE_CONFIG, preprocess=MEASURED.preprocess) == MEASURED
    assert replace(SCENE_CONFIG.preprocess, calibration_path=MEASURED.preprocess.calibration_path) \
        == MEASURED.preprocess


@pytest.mark.software
def test_synthetic_marks_come_back_exactly_only_without_undistortion():
    frame = synthetic_frame([150, 290])
    scene = run_chain(frame, 0, 0, SCENE_CONFIG).offset
    assert (scene.left_x, scene.right_x) == (149.5, 289.5)
    # The reason SCENE_CONFIG exists: the lens model moves synthetic marks
    warped = run_chain(frame, 0, 0, MEASURED).offset
    assert (warped.left_x, warped.right_x) != (149.5, 289.5)


# =============================================================================
# One config for every consumer
# =============================================================================

def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


@pytest.mark.software
def test_linkers_run_the_config_module_objects():
    assert p2.MEASURED is config.MEASURED and p2.PipelineConfig is config.PipelineConfig
    assert p3.MEASURED is config.MEASURED
    assert _default(p2.run_live_view, "config") is config.MEASURED
    assert _default(p3.run_phase3_chain, "config") is config.MEASURED
    assert _default(p3.run, "config") is config.MEASURED
    assert _default(p3.run, "p3_config") is config.MEASURED_ESTIMATION


@pytest.mark.software
def test_phase3_linker_flags_apply_on_top_of_measured_estimation(tmp_path, monkeypatch):
    frames = tmp_path / "f"
    frames.mkdir()
    cv2.imwrite(str(frames / "000000.png"), synthetic_frame([150, 290]))
    # MEASURED_ESTIMATION still equals Phase3Config()'s defaults, so a CLI that
    # built a bare Phase3Config would pass unnoticed; swap in one that differs
    tuned = replace(config.MEASURED_ESTIMATION, ema_alpha=0.6, vote_window=5)
    monkeypatch.setattr(p3, "MEASURED_ESTIMATION", tuned)
    got = {}
    monkeypatch.setattr(p3, "run", lambda source, cfg, p3_config, *a, **k: got.update(p3=p3_config))
    p3.cli(["--frames", str(frames), "--gyro-bias", "1.25", "--cm-per-px", "0.07",
            "--out", str(tmp_path / "out")])
    assert got["p3"] == replace(tuned, gyro_bias_dps=1.25, cm_per_px=0.07)


@pytest.mark.software
def test_config_module_imports_no_pipeline_or_debugger_code():
    probe = ("import sys, src.config; "
             "print(sorted(m for m in sys.modules if m.startswith('src.')))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=PIPELINE_ROOT,
                         capture_output=True, text=True, check=True).stdout
    loaded = eval(out)
    # capture.camera is allowed: preprocess takes its FrameData, and importing it opens nothing
    assert not [m for m in loaded if m.startswith(("src.debugger", "src.phase2_linker",
                                                    "src.phase3_linker"))]


@pytest.mark.software
def test_phase3_linker_reports_the_color_branch_from_the_config(tmp_path, capsys):
    frames = tmp_path / "f"
    frames.mkdir()
    cv2.imwrite(str(frames / "000000.png"), synthetic_frame([150, 290]))
    assert p3.cli(["--frames", str(frames), "--out", str(tmp_path / "out")]) == 0
    assert "color branch on" in capsys.readouterr().out
