"""
test_ground.py  --  src/perception/ground.py and src/scripts/calibrate_ground.py

The ground plane is tested against SYNTHETIC_GROUND, a homography with known
answers: mapping both ways, the centerline crossing, fitting it back from
its own corners, and loading refusing every mismatch. The calibration script
is tested on a checkerboard rendered onto that floor and distorted by the
real lens model, so the fit runs through MEASURED's actual undistortion.

--software  Mapping, fit, loading and the script end to end. No camera.
"""
import json
import shutil
import warnings
from dataclasses import replace

import cv2
import numpy as np
import pytest

import src.scripts.calibrate_ground as cg
from src.config import MEASURED, PipelineConfig
from src.params import CAMERA_CALIB_PATH, FRAME_H, FRAME_W
from src.perception.ground import (
    GroundHomography, fit_ground_homography, lens_id, load_ground_homography,
)
from src.perception.preprocess import PreprocessParams
from src.tests.scenes import SCENE_CONFIG, SYNTHETIC_GROUND as G, floor_board, lens_distort

PATTERN = (9, 6)
PX = np.array([[240, 270], [240, 200], [60, 250], [400, 190], [10, 265]], float)


# =============================================================================
# Mapping
# =============================================================================

@pytest.mark.software
def test_px_to_cm_to_px_round_trips():
    np.testing.assert_allclose(G.to_image(G.to_floor(PX)), PX, atol=1e-9)
    cm = np.array([[0.0, 0.0], [-12.0, 7.5], [9.0, 28.0]])
    np.testing.assert_allclose(G.to_floor(G.to_image(cm)), cm, atol=1e-9)


@pytest.mark.software
def test_known_points_land_where_the_synthetic_floor_puts_them():
    np.testing.assert_allclose(G.to_floor([[240, 270], [0, 270], [480, 270], [90, 180]]),
                               [[0, 0], [-15, 0], [15, 0], [-15, 30]], atol=1e-9)


@pytest.mark.software
def test_centerline_crossing_is_exact_for_straight_floor_lines():
    across = G.to_image([[-10.0, 12.0], [14.0, 12.0]])              # square to the robot
    assert G.forward_at_centerline(*across) == pytest.approx(12.0)
    slanted = G.to_image([[-10.0, 10.0], [10.0, 14.0]])             # 12 cm where it crosses X = 0
    assert G.forward_at_centerline(*slanted) == pytest.approx(12.0)
    off_to_one_side = G.to_image([[4.0, 20.0], [10.0, 23.0]])       # extended to X = 0: 18 cm
    assert G.forward_at_centerline(*off_to_one_side) == pytest.approx(18.0)
    along = G.to_image([[3.0, 5.0], [3.0, 20.0]])                   # parallel to the robot's axis
    assert G.forward_at_centerline(*along) is None


@pytest.mark.software
def test_homography_is_frozen_hashable_and_refuses_a_singular_matrix():
    assert hash(G) == hash(GroundHomography.from_matrix(np.asarray(G.H), G.image_size, 0.0, None))
    assert replace(SCENE_CONFIG, ground=G) == replace(SCENE_CONFIG, ground=G)
    with pytest.raises(ValueError, match="singular"):
        GroundHomography(((1, 0, 0), (0, 0, 0), (0, 0, 1)), (FRAME_W, FRAME_H), 0.0, None)
    scaled = GroundHomography.from_matrix(5.0 * np.asarray(G.H), G.image_size, 0.0, None)
    assert scaled.H[2][2] == 1.0
    np.testing.assert_allclose(np.asarray(scaled.H), np.asarray(G.H), rtol=1e-12, atol=1e-12)


# =============================================================================
# Fit
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("noise_px", [0.0, 0.2])
def test_fit_recovers_a_known_homography_from_its_own_corners(noise_px):
    board = cg.board_points_cm(PATTERN, 2.5, -10.0, 22.0)
    px = G.to_image(board) + np.random.default_rng(3).normal(0, noise_px, board.shape)
    H, errors = fit_ground_homography(px, board)
    fitted = GroundHomography.from_matrix(H, G.image_size, 0.0, None)
    tolerance = 1e-3 if noise_px == 0 else 0.3                    # findHomography's own precision, then noise
    np.testing.assert_allclose(fitted.to_floor(PX), G.to_floor(PX), atol=tolerance)
    # At the board's far row one pixel is ~0.33 cm of floor, so 0.2 px of noise costs up to ~0.3 cm there
    assert errors.shape == (54,) and errors.max() < (1e-3 if noise_px == 0 else 0.5)


@pytest.mark.software
def test_fit_refuses_too_few_or_mismatched_points():
    with pytest.raises(ValueError, match=">= 4"):
        fit_ground_homography(PX[:3], PX[:3])
    with pytest.raises(ValueError):
        fit_ground_homography(PX, PX[:4])


# =============================================================================
# Loading
# =============================================================================

def write_ground(tmp_path, **overrides):
    """A ground JSON that matches MEASURED's preprocess, with fields replaced by overrides."""
    rec = {"model": "floor_homography", "H": np.asarray(G.H).tolist(),
           "image_size": [FRAME_W, FRAME_H], "undistort_alpha": 0.0,
           "lens_calibration": {"file": "camera_calibration.json", "created": None,
                                "sha256": lens_id(CAMERA_CALIB_PATH)},
           "reprojection_cm": {"mean": 0.1, "max": 0.2}}
    rec.update(overrides)
    path = tmp_path / "ground.json"
    path.write_text(json.dumps(rec))
    return path

def load(path, preprocess=MEASURED.preprocess, frame_size=(FRAME_H, FRAME_W)):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = load_ground_homography(path, preprocess, frame_size)
    return g, [str(w.message) for w in caught]


@pytest.mark.software
def test_a_matching_file_loads_with_its_fit_quality(tmp_path):
    g, warned = load(write_ground(tmp_path))
    assert warned == [] and g == GroundHomography.from_matrix(np.asarray(G.H), (FRAME_W, FRAME_H), 0.0,
                                                           lens_id(CAMERA_CALIB_PATH), 0.1, 0.2)


@pytest.mark.software
@pytest.mark.parametrize("case, why", [
    ("size", "fit at 640x360"),
    ("alpha", "undistort_alpha"),
    ("lens", "different lens calibration"),
    ("malformed", "malformed"),
])
def test_a_mismatched_file_is_refused_with_a_warning(tmp_path, case, why):
    overrides = {"size": {"image_size": [640, 360]},
                 "alpha": {"undistort_alpha": 0.5},
                 "lens": {"lens_calibration": {"sha256": "0" * 64}},
                 "malformed": {"H": "not a matrix"}}[case]
    g, warned = load(write_ground(tmp_path, **overrides))
    assert g is None and any(why in w for w in warned)


@pytest.mark.software
def test_undistortion_off_or_a_missing_lens_file_refuses_it(tmp_path):
    path = write_ground(tmp_path)
    g, warned = load(path, replace(MEASURED.preprocess, calibration_path=None))
    assert g is None and any("undistortion is off" in w for w in warned)
    g, warned = load(path, replace(MEASURED.preprocess, calibration_path=str(tmp_path / "gone.json")))
    assert g is None and any("not found" in w for w in warned)


@pytest.mark.software
def test_a_missing_file_is_optional_none_without_a_warning(tmp_path):
    g, warned = load(tmp_path / "nope.json")
    assert g is None and warned == []


@pytest.mark.software
def test_lens_id_follows_the_values_preprocess_uses(tmp_path):
    assert lens_id(None) is None and lens_id(tmp_path / "gone.json") is None
    copy = tmp_path / "lens.json"
    shutil.copy(CAMERA_CALIB_PATH, copy)
    assert lens_id(copy) == lens_id(CAMERA_CALIB_PATH)
    calib = json.loads(copy.read_text())
    calib["created"] = "another day"                               # not a value preprocess uses
    copy.write_text(json.dumps(calib))
    assert lens_id(copy) == lens_id(CAMERA_CALIB_PATH)
    calib["dist_coeffs"][0] += 1e-6
    copy.write_text(json.dumps(calib))
    assert lens_id(copy) != lens_id(CAMERA_CALIB_PATH)


@pytest.mark.software
def test_synthetic_scenes_run_without_a_ground_plane():
    assert SCENE_CONFIG.ground is None and PipelineConfig().ground is None


# =============================================================================
# Calibration script
# =============================================================================

@pytest.mark.software
def test_board_points_run_right_across_and_toward_the_robot_down_the_board():
    pts = cg.board_points_cm(PATTERN, 2.5, -10.0, 22.0).reshape(6, 9, 2)
    assert tuple(pts[0, 0]) == (-10.0, 22.0)
    assert tuple(pts[0, 1]) == (-7.5, 22.0) and tuple(pts[1, 0]) == (-10.0, 19.5)


@pytest.mark.software
def test_corners_are_ordered_from_the_top_left_whichever_end_opencv_starts():
    truth = G.to_image(cg.board_points_cm(PATTERN, 2.5, -10.0, 22.0))
    np.testing.assert_allclose(cg.order_corners(truth[::-1].copy(), PATTERN), truth)
    np.testing.assert_allclose(cg.order_corners(truth, PATTERN), truth)


@pytest.mark.software
def test_a_board_turned_a_quarter_turn_is_refused():
    grid = G.to_image(cg.board_points_cm((6, 9), 2.5, -6.0, 26.0)).reshape(9, 6, 2)
    turned = grid.transpose(1, 0, 2).reshape(-1, 2)                 # 9-corner side runs up the image
    with pytest.raises(ValueError, match="quarter turn"):
        cg.order_corners(turned, PATTERN)


@pytest.mark.software
def test_the_reference_moves_to_the_floor_at_the_bottom_center_of_the_view():
    shifted = replace(G)                                            # G already has it there
    moved = GroundHomography.from_matrix(np.array([[1, 0, 0], [0, 1, 7.0], [0, 0, 1]]) @ np.asarray(G.H),
                                         G.image_size, 0.0, None)
    H, shift = cg.reference_at_view_bottom(np.asarray(moved.H), FRAME_W, FRAME_H)
    back = GroundHomography.from_matrix(H, G.image_size, 0.0, None)
    assert shift == pytest.approx(-7.0)
    np.testing.assert_allclose(back.to_floor(PX), shifted.to_floor(PX), atol=1e-9)


@pytest.mark.software
def test_floor_board_corners_are_found_to_a_fraction_of_a_pixel():
    """The window sized to the squares; a fixed 11x11 one is visibly worse on a squeezed floor board."""
    board = cv2.cvtColor(floor_board(G, -10.0, 22.0, 2.5), cv2.COLOR_BGR2GRAY)
    truth = G.to_image(cg.board_points_cm(PATTERN, 2.5, -10.0, 22.0))
    found = cg.find_ground_corners(board, PATTERN)
    assert np.linalg.norm(found - truth, axis=1).mean() < 0.8
    from src.scripts.calibrate_camera import find_corners
    fixed = cg.order_corners(find_corners(board, PATTERN, fast=False), PATTERN)
    assert np.linalg.norm(fixed - truth, axis=1).mean() > 1.5


@pytest.mark.software
@pytest.mark.parametrize("origin_y", [None, 22.0])
def test_fit_from_a_rendered_board_recovers_the_floor(origin_y):
    frames = [floor_board(G, -10.0, 22.0, 2.5)]
    corners, used, _ = cg.average_corners(frames, PATTERN, PreprocessParams())
    result = cg.fit(corners, PATTERN, 2.5, -10.0, origin_y, (FRAME_W, FRAME_H))
    fitted = GroundHomography.from_matrix(result["H"], G.image_size, 0.0, None)
    np.testing.assert_allclose(fitted.to_floor(PX[:4]), G.to_floor(PX[:4]), atol=0.3)
    assert used == 1 and result["errors_cm"].max() < 0.3
    assert result["corner0_cm"] == pytest.approx((-10.0, 22.0), abs=0.3)


@pytest.mark.software
def test_the_script_end_to_end_through_the_real_lens_writes_a_file_measured_loads(tmp_path):
    raw = tmp_path / "raw.png"
    distorted = lens_distort(floor_board(G, -10.0, 22.0, 2.5), str(CAMERA_CALIB_PATH))
    # The synthetic board, pushed through the lens and back, is at the edge of what OpenCV's
    # chessboard detector takes: 4.14 finds it, the Pi's 4.6 (apt) doesn't (2026-10-03). That is
    # this test image, not the script: on the Pi, fit from real frames, or run --image on a desktop
    if cg.average_corners([distorted], PATTERN, MEASURED.preprocess)[1] == 0:
        pytest.skip(f"OpenCV {cv2.__version__} finds no {PATTERN[0]}x{PATTERN[1]} board in the synthetic "
                    "lens-distorted image; run this test, or calibrate_ground --image, on a desktop")
    cv2.imwrite(str(raw), distorted)
    out, dbg = tmp_path / "ground.json", tmp_path / "debug.png"
    assert cg.main([f"--image={raw}", "--square-cm=2.5", "--origin-x-cm=-10",
                    f"--out={out}", f"--debug-image={dbg}"]) == 0
    rec = json.loads(out.read_text())
    assert rec["lens_calibration"]["sha256"] == lens_id(CAMERA_CALIB_PATH)
    assert rec["image_size"] == [FRAME_W, FRAME_H] and rec["undistort_alpha"] == 0.0
    assert rec["pattern"] == [9, 6] and rec["origin_cm"] == [-10.0, None] and len(rec["corners_px"]) == 54
    assert rec["reprojection_cm"]["max"] < 1.0
    g, warned = load(out)
    assert warned == [] and g is not None
    roi_points = np.array([[240, 250], [140, 230], [380, 205]], float)
    np.testing.assert_allclose(g.to_floor(roi_points), G.to_floor(roi_points), atol=0.4)
    assert cv2.imread(str(dbg)).shape[:2] == (3 * FRAME_H, 3 * FRAME_W)


@pytest.mark.software
def test_the_script_refuses_without_a_lens_calibration_or_a_board(tmp_path, monkeypatch):
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((FRAME_H, FRAME_W, 3), 90, np.uint8))
    with pytest.raises(SystemExit, match="no 9x6 board"):
        cg.main([f"--image={blank}", "--square-cm=2.5", "--origin-x-cm=0", f"--out={tmp_path / 'g.json'}"])
    monkeypatch.setattr(cg, "MEASURED", replace(MEASURED, preprocess=replace(MEASURED.preprocess,
                                                                              calibration_path=None)))
    with pytest.raises(SystemExit, match="lens calibration"):
        cg.main([f"--image={blank}", "--square-cm=2.5", "--origin-x-cm=0", f"--out={tmp_path / 'g.json'}"])
    assert not (tmp_path / "g.json").exists()
