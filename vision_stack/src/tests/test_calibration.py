"""
test_calibration.py  --  src/scripts/calibrate_camera.py + its consumer, preprocess.undistort

Capture is interactive and stays a script. What is testable is everything
around it: the math helpers, solve() on frames with a KNOWN lens, the
calibration JSON the team produced, and the undistortion preprocess applies.

Synthetic frames are rendered through an exact distortion model (K_TRUE,
D_TRUE): every output pixel is un-distorted to a ray, intersected with the
board plane and shaded by square parity, at 2x supersampling so corner
subpixel refinement has real gradients. solve() then has a right answer.

--software  Helper known-answers, solve() round trip on a known lens, error
            paths, preprocess.undistort against a reference map build,
            and the contract of calibration/camera_calib.json if it exists
            (including 3x3 coverage of the frames it was solved from).
--hardware  Times preprocess.undistort per frame (live or --replay), and on
            frames where the board is visible scores straightness raw vs
            undistorted. Point --replay at frames the solver never saw.
"""
import argparse
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.params import CAMERA_CALIB_PATH, CAMERA_ROTATE_180, FRAME_H, FRAME_W, SENSOR_CONFIG
from src.perception import preprocess as pp
from src.perception.preprocess import PreprocessParams, undistort
from src.scripts import calibrate_camera as cc
from src.tests.artifacts import summarize

PATTERN = (9, 6)                      # inner corners, matches calibrate_camera's default
SQUARE_MM = 25.0
CALIB_FILE = CAMERA_CALIB_PATH
ROWS, COLS = ("top", "middle", "bottom"), ("left", "center", "right")

# A plausible wide M12 lens at 480x270: strong barrel, slightly off-center
K_TRUE = np.array([[330.0, 0.0, 244.0], [0.0, 330.0, 131.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([-0.32, 0.12, 0.0, 0.0, 0.0])
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)


def render_board(rvec, tvec, K=K_TRUE, D=D_TRUE, size=(FRAME_W, FRAME_H), ss=2):
    """BGR frame of a PATTERN checkerboard at pose (rvec, tvec) seen through (K, D)."""
    cols, rows = PATTERN
    w, h = size[0] * ss, size[1] * ss
    u, v = np.meshgrid((np.arange(w) + 0.5) / ss - 0.5, (np.arange(h) + 0.5) / ss - 0.5)
    px = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    rays = cv2.undistortPointsIter(px, K, D, None, None, UNDISTORT_CRITERIA).reshape(-1, 2)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    d = np.column_stack([rays, np.ones(len(rays))])
    A = np.empty((len(d), 3, 3))
    A[:, :, 0], A[:, :, 1], A[:, :, 2] = R[:, 0], R[:, 1], -d
    bx, by, s = np.linalg.solve(A, np.broadcast_to(-np.asarray(tvec, float), (len(d), 3))[..., None])[..., 0].T
    # Inner corners sit at 0..(cols-1)*sq, so squares run one past each side
    ix, iy = np.floor(bx / SQUARE_MM) + 1, np.floor(by / SQUARE_MM) + 1
    on_board = (ix >= 0) & (ix <= cols) & (iy >= 0) & (iy <= rows) & (s > 0)
    on_paper = (bx > -2 * SQUARE_MM) & (bx < (cols + 1) * SQUARE_MM) & \
               (by > -2 * SQUARE_MM) & (by < (rows + 1) * SQUARE_MM) & (s > 0)
    img = np.full(len(d), 110, np.uint8)                   # grey background
    img[on_paper] = 235
    img[on_board & ((ix + iy) % 2 == 0)] = 20
    img = cv2.resize(img.reshape(h, w), size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def board_poses(centers_y=(-150, 0, 150), centers_x=(-230, 0, 230), z=560.0):
    """Two tilts per 3x3 cell; tvec places the board's middle at (x, y, z) mm."""
    cols, rows = PATTERN
    mid = np.array([(cols - 1) * SQUARE_MM / 2, (rows - 1) * SQUARE_MM / 2, 0.0])
    out = []
    for cy, cx in itertools.product(centers_y, centers_x):
        for tilt in ((0.25, -0.2, 0.02), (-0.2, 0.3, -0.03)):
            R, _ = cv2.Rodrigues(np.array(tilt))
            out.append((np.array(tilt), np.array([cx, cy, z]) - R @ mid))
    return out


def board_corners(rvec, tvec, K=K_TRUE, D=D_TRUE):
    """Exact distorted corner positions, (N, 1, 2) float32, in findChessboardCorners order."""
    cols, rows = PATTERN
    obj = np.zeros((cols * rows, 3), np.float64)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_MM
    img, _ = cv2.projectPoints(obj, np.asarray(rvec, float), np.asarray(tvec, float), K, D)
    return img.astype(np.float32)


def solve_args(frames_dir, out):
    """The argparse namespace solve() reads, for PATTERN boards in frames_dir."""
    return argparse.Namespace(pattern="x".join(map(str, PATTERN)), frames=str(frames_dir),
                              square_mm=SQUARE_MM, out=str(out))


def write_frames(directory, poses):
    """Render one calib_NNN.png per pose into directory, as capture() would name them."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, (r, t) in enumerate(poses):
        cv2.imwrite(str(directory / f"calib_{i:03d}.png"), render_board(r, t))
    return directory


def region_of(corners, w, h):
    """(row, col) of the 3x3 cell holding the board centroid -- same rule capture() uses."""
    cx, cy = corners.reshape(-1, 2).mean(axis=0)
    return min(int(3 * cy / h), 2), min(int(3 * cx / w), 2)


def reference_maps(calib, alpha=0.0):
    """Independent map construction straight from OpenCV, the oracle for preprocess."""
    size = tuple(calib["image_size"])
    K, dist = np.array(calib["camera_matrix"]), np.array(calib["dist_coeffs"])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    return cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)


@pytest.fixture(scope="session")
def solved(tmp_path_factory):
    """solve() run once on 18 synthetic frames covering all 9 cells; returns the JSON dict."""
    root = tmp_path_factory.mktemp("calib")
    frames_dir = write_frames(root / "frames", board_poses())
    cc.solve(solve_args(frames_dir, root / "camera_calib.json"))
    return json.loads((root / "camera_calib.json").read_text())


@pytest.mark.software
def test_straightness_of_an_undistorted_grid_is_zero():
    corners = board_corners(np.zeros(3), [-100.0, -60.0, 500.0], D=np.zeros(5))
    assert cc.straightness(corners, PATTERN) < 1e-3


@pytest.mark.software
def test_straightness_sees_barrel_bow_near_the_frame_edge():
    # Bottom-right cell, board filling a quarter of the frame. Bow is sub-pixel even
    # with strong barrel at 480x270, which is why the thresholds below are tight
    r, t = board_poses(centers_y=(120,), centers_x=(185,), z=450.0)[0]
    bowed = cc.straightness(board_corners(r, t), PATTERN)
    flat = cc.straightness(board_corners(r, t, D=np.zeros(5)), PATTERN)
    assert bowed > 0.3 and flat < 1e-3


@pytest.mark.software
def test_undistort_with_zero_distortion_leaves_geometry_in_place(tmp_path):
    # Compared by corner position, not pixel values: with zero distortion some
    # OpenCV builds (4.6, Ubuntu's apt package) still return a new camera matrix
    # a fraction of a pixel off K, which a pixel-exact comparison can't tolerate
    path = tmp_path / "zero.json"
    path.write_text(json.dumps({"image_size": [FRAME_W, FRAME_H], "camera_matrix": K_TRUE.tolist(),
                                "dist_coeffs": [0.0] * 5}))
    r, t = board_poses(centers_y=(-100,), centers_x=(-160,))[0]      # off-center, where a scale error shows
    frame = render_board(r, t, D=np.zeros(5))
    out = undistort(frame, PreprocessParams(calibration_path=str(path)))
    before = cc.find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), PATTERN, fast=False)
    after = cc.find_corners(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), PATTERN, fast=False)
    assert before is not None and after is not None
    assert np.linalg.norm((after - before).reshape(-1, 2), axis=1).max() < 1.5


@pytest.mark.software
@pytest.mark.parametrize("text, expect", [("9x6", (9, 6)), ("7X5", (7, 5))])
def test_parse_pattern(text, expect):
    assert cc.parse_pattern(text) == expect


@pytest.mark.software
def test_parse_pattern_rejects_garbage():
    with pytest.raises(SystemExit, match="--pattern"):
        cc.parse_pattern("nine by six")


@pytest.mark.software
def test_synthetic_frames_are_all_detectable():
    # Guards the renderer: if boards stop being found, the round trip below tests nothing
    found = 0
    for r, t in board_poses():
        gray = cv2.cvtColor(render_board(r, t), cv2.COLOR_BGR2GRAY)
        found += cc.find_corners(gray, PATTERN, fast=False) is not None
    assert found >= 15


@pytest.mark.software
def test_solve_recovers_the_known_intrinsics(solved):
    K = np.array(solved["camera_matrix"])
    assert solved["rms_px"] < 0.5
    assert abs(K[0, 0] / K_TRUE[0, 0] - 1) < 0.02 and abs(K[1, 1] / K_TRUE[1, 1] - 1) < 0.02
    assert abs(K[0, 2] - K_TRUE[0, 2]) < 4 and abs(K[1, 2] - K_TRUE[1, 2]) < 4
    assert abs(solved["dist_coeffs"][0] - D_TRUE[0]) < 0.03
    assert solved["image_size"] == [FRAME_W, FRAME_H]


@pytest.mark.software
@pytest.mark.parametrize("cell", [(-150, -230), (-150, 230), (150, -230), (150, 230), (0, 0)])
def test_solved_model_straightens_boards_it_never_saw(solved, cell):
    # Held-out pose (different tilt and depth) in each corner: the property the pipeline needs
    K, D = np.array(solved["camera_matrix"]), np.array(solved["dist_coeffs"])
    r, t = board_poses(centers_y=(cell[0],), centers_x=(cell[1],), z=620.0)[0]
    r = r * np.array([-0.6, 0.8, 1.0])
    raw = board_corners(r, t)
    fixed = cv2.undistortPointsIter(raw, K, D, None, K, UNDISTORT_CRITERIA).astype(np.float32)
    assert cc.straightness(fixed, PATTERN) < 0.3
    if cell != (0, 0):
        assert cc.straightness(raw, PATTERN) > 3 * cc.straightness(fixed, PATTERN)


@pytest.mark.software
def test_solve_with_no_frames_exits(tmp_path):
    with pytest.raises(SystemExit, match="no calib_"):
        cc.solve(solve_args(tmp_path, tmp_path / "c.json"))


@pytest.mark.software
def test_solve_with_too_few_usable_frames_exits(tmp_path):
    write_frames(tmp_path, board_poses()[:5])
    with pytest.raises(SystemExit, match="usable frames"):
        cc.solve(solve_args(tmp_path, tmp_path / "c.json"))


@pytest.mark.software
def test_solve_refuses_mixed_resolutions(tmp_path):
    write_frames(tmp_path, board_poses()[:2])
    cv2.imwrite(str(tmp_path / "calib_999.png"), np.zeros((FRAME_H * 2, FRAME_W * 2, 3), np.uint8))
    with pytest.raises(SystemExit, match="different resolutions"):
        cc.solve(solve_args(tmp_path, tmp_path / "c.json"))


@pytest.fixture
def calib_path(tmp_path, solved):
    """The solved calibration written to a path unique to this test."""
    path = tmp_path / "camera_calib.json"                  # fresh path per test: preprocess caches by path
    path.write_text(json.dumps(solved))
    return path


@pytest.mark.software
def test_preprocess_undistort_matches_a_reference_map_build(calib_path, solved):
    r, t = board_poses()[0]
    frame = render_board(r, t)
    m1, m2 = reference_maps(solved, alpha=0.0)
    expect = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    got = undistort(frame, PreprocessParams(calibration_path=str(calib_path)))
    assert got.shape == frame.shape and np.array_equal(got, expect)


@pytest.mark.software
def test_preprocess_refuses_a_calibration_from_another_resolution(calib_path):
    frame = np.zeros((FRAME_H * 2, FRAME_W * 2, 3), np.uint8)
    with pytest.raises(ValueError, match="Recalibrate"):
        undistort(frame, PreprocessParams(calibration_path=str(calib_path)))


@pytest.mark.software
def test_preprocess_passes_frames_through_when_the_file_is_missing(tmp_path):
    frame = np.full((FRAME_H, FRAME_W, 3), 7, np.uint8)
    with pytest.warns(UserWarning, match="not found"):
        out = undistort(frame, PreprocessParams(calibration_path=str(tmp_path / "nope.json")))
    assert out is frame


def _load_team_calibration():
    """The team's calibration JSON, or skip the test if it hasn't been made yet."""
    if not CALIB_FILE.is_file():
        pytest.skip(f"no {CALIB_FILE.name} yet (run: python3 -m src.scripts.calibrate_camera)")
    return json.loads(CALIB_FILE.read_text())


@pytest.mark.software
def test_calibration_file_matches_this_camera_mode():
    c = _load_team_calibration()
    assert c["image_size"] == [FRAME_W, FRAME_H]
    assert c["sensor_config"] == SENSOR_CONFIG and c["rotate_180"] == CAMERA_ROTATE_180


@pytest.mark.software
def test_calibration_file_intrinsics_are_plausible():
    c = _load_team_calibration()
    K = np.array(c["camera_matrix"])
    assert c["rms_px"] < 1.0, f"rms {c['rms_px']:.3f} px: recapture"
    assert c["n_frames"] >= 10
    assert abs(K[0, 0] / K[1, 1] - 1) < 0.03, "fx and fy should agree on square pixels"
    assert abs(K[0, 2] - FRAME_W / 2) < 0.15 * FRAME_W
    assert abs(K[1, 2] - FRAME_H / 2) < 0.15 * FRAME_H


@pytest.mark.software
def test_calibration_file_was_solved_from_frames_covering_the_whole_image():
    # A good rms on top-row-only frames still extrapolates the bottom of the lens
    c = _load_team_calibration()
    frames_dir = cc.DEFAULT_FRAMES
    paths = [frames_dir / name for name in c["frames"] if (frames_dir / name).is_file()]
    if len(paths) < len(c["frames"]):
        pytest.skip(f"{frames_dir} no longer holds the frames this calibration used")
    cells = np.zeros((3, 3), int)
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        corners = cc.find_corners(img, tuple(c["pattern"]), fast=False)
        if corners is not None:
            cells[region_of(corners, img.shape[1], img.shape[0])] += 1
    table = "\n".join(f"  {ROWS[r]:6s} " + " ".join(f"{n:2d}" for n in cells[r]) for r in range(3))
    assert (cells.sum(axis=1) > 0).all(), f"a row of the image has no frames:\n{table}"
    assert (cells.sum(axis=0) > 0).all(), f"a column of the image has no frames:\n{table}"
    assert (cells > 0).sum() >= 7, f"only {(cells > 0).sum()}/9 cells covered:\n{table}"


def _side_by_side(raw, und):
    """Raw and undistorted frames with a white gap between, for eyeballing the correction."""
    gap = np.full((raw.shape[0], 6, 3), 255, np.uint8)
    return np.hstack([raw, gap, und])


@pytest.mark.hardware
def test_undistortion_characterization(request, frames, artifacts):
    calib = _load_team_calibration()
    n = request.config.getoption("--frames")
    params = PreprocessParams(calibration_path=str(CALIB_FILE))
    pp._undistort_maps.cache_clear()

    rows, samples, cells = [], {}, np.zeros((3, 3), int)
    warmed = False
    for i, fd in enumerate(frames(n)):
        if not warmed:                                     # map build happens once, not per frame
            undistort(fd.frame, params)
            warmed = True
        t0 = time.perf_counter_ns()
        und = undistort(fd.frame, params)
        stage_ms = (time.perf_counter_ns() - t0) / 1e6

        # Board scoring is outside the timing window
        raw_c = cc.find_corners(cv2.cvtColor(fd.frame, cv2.COLOR_BGR2GRAY), PATTERN)
        s_raw = s_und = float("nan")
        region = ""
        if raw_c is not None:
            und_c = cc.find_corners(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), PATTERN)
            s_raw = cc.straightness(raw_c, PATTERN)
            s_und = cc.straightness(und_c, PATTERN) if und_c is not None else float("nan")
            r, c = region_of(raw_c, fd.frame.shape[1], fd.frame.shape[0])
            cells[r, c] += 1
            region = f"{ROWS[r]}-{COLS[c]}"
        rows.append((fd.frame_id, fd.timestamp_ms, round(stage_ms, 4), region,
                     round(s_raw, 3), round(s_und, 3)))
        if i in (0, n // 2, n - 1) or (raw_c is not None and fd.frame_id not in samples and len(samples) < 8):
            samples[fd.frame_id] = (fd.frame, und)

    if not rows:
        pytest.skip("no frames delivered")

    stage = [r[2] for r in rows]
    scored = [r for r in rows if r[3]]
    artifacts.json("config.json", {k: calib[k] for k in
                                   ("image_size", "camera_matrix", "dist_coeffs", "rms_px", "n_frames", "created")})
    artifacts.csv("undistort.csv", ["frame_id", "timestamp_ms", "stage_ms", "region",
                                    "straight_raw_px", "straight_und_px"], rows)
    per_region = {}
    for r in scored:
        per_region.setdefault(r[3], []).append((r[4], r[5]))
    artifacts.json("summary.json", {
        "stage_ms": summarize(stage),
        "frames_with_board": len(scored),
        "coverage": {ROWS[r]: cells[r].tolist() for r in range(3)},
        "straightness_by_region": {
            k: {"raw": summarize(a for a, _ in v), "undistorted": summarize(b for _, b in v if not math.isnan(b))}
            for k, v in sorted(per_region.items())},
    })
    artifacts.histogram("stage_ms_hist.png", stage, "preprocess.undistort latency", "ms")
    for fid, (raw, und) in samples.items():
        artifacts.image(f"{fid:06d}_raw_vs_undistorted.png", _side_by_side(raw, und))

    # Characterization, but one hard line: where a board was seen, correction must not make
    # it worse. 0.5 px is the corner-detection noise floor, the same figure calibrate_camera uses
    for r in scored:
        if not math.isnan(r[5]):
            assert r[5] <= r[4] + 0.5, f"frame {r[0]} ({r[3]}): undistorted {r[5]} px vs raw {r[4]} px"