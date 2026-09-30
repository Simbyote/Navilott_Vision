"""
test_stop_line_table.py  --  src/perception/stop_line_table.py

The curve fit on marks with a known answer (exact, and with measurement
noise), every way a set of marks is refused, the table's conversion, and
loading: silent without a file, refused with a warning when malformed or fit
under another camera setup.
"""
import json
import warnings
from dataclasses import replace

import numpy as np
import pytest

from src.config import MEASURED
from src.params import FRAME_H, FRAME_W
from src.perception.ground import lens_id
from src.perception.stop_line_table import (
    MIN_MARKS, fit_stop_line_table, load_stop_line_table,
)
from src.tests.scenes import SYNTHETIC_STOP_LINE_TABLE as T

ROWS = np.array([0.0, 15.0, 30.0, 50.0, 70.0])


def true_cm(rows, a=800.0, b=160.0, c=-2.0):
    return a / (b - np.asarray(rows)) + c


# =============================================================================
# Fit
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("n", [3, 5])
def test_the_fit_recovers_a_flat_floor_curve_exactly(n):
    (a, b, c), err = fit_stop_line_table(ROWS[:n], true_cm(ROWS[:n]))
    assert (a, b, c) == pytest.approx((800.0, 160.0, -2.0), rel=1e-6)
    assert err.max() < 1e-9 and len(err) == n


@pytest.mark.software
def test_measurement_noise_shows_up_as_each_marks_error():
    noise = np.array([0.1, -0.1, 0.15, 0.0, -0.1])
    (a, b, c), err = fit_stop_line_table(ROWS, true_cm(ROWS) + noise)
    assert 0.0 < err.max() < 0.3 and len(err) == len(ROWS)
    assert err == pytest.approx(np.abs(a / (b - ROWS) + c - (true_cm(ROWS) + noise)))     # sizes, not signs


@pytest.mark.software
def test_the_order_of_the_marks_does_not_matter():
    idx = [3, 0, 4, 1, 2]
    p1, _ = fit_stop_line_table(ROWS, true_cm(ROWS))
    p2, e2 = fit_stop_line_table(ROWS[idx], true_cm(ROWS)[idx])
    assert p1 == pytest.approx(p2) and e2.max() < 1e-9


@pytest.mark.software
@pytest.mark.parametrize("rows, cm, words", [
    ([0.0, 20.0], [3.0, 5.0], "at least"),
    ([0.0, 20.0, 40.0], [3.0, 5.0], "at least"),
    ([0.0, 20.0, 20.0], [3.0, 5.0, 6.0], "same row"),
    ([0.0, 20.0, 40.0], [3.0, 6.0, 5.0], "farther away"),
    ([0.0, 20.0, 40.0], [3.0, 3.0, 5.0], "farther away"),
    ([0.0, 20.0, 40.0], [3.0, 8.0, 9.0], "flat floor"),        # growth slows: not a floor seen in perspective
])
def test_marks_that_cannot_be_right_are_refused(rows, cm, words):
    with pytest.raises(ValueError, match=words):
        fit_stop_line_table(rows, cm)


@pytest.mark.software
def test_three_marks_is_the_minimum():
    assert MIN_MARKS == 3


# =============================================================================
# The table
# =============================================================================

@pytest.mark.software
def test_to_cm_follows_the_curve_and_never_goes_negative():
    assert T.to_cm(0.0) == pytest.approx(3.0)
    assert T.to_cm(60.0) == pytest.approx(true_cm(60.0))
    assert T.to_cm(0.0) < T.to_cm(30.0) < T.to_cm(60.0)
    assert replace(T, c=-10.0).to_cm(0.0) == 0.0


@pytest.mark.software
def test_tables_are_frozen_and_compare_by_value():
    assert replace(T) == T and hash(replace(T)) == hash(T)
    with pytest.raises(Exception):
        T.a = 1.0


# =============================================================================
# Loading
# =============================================================================

def table_file(tmp_path, **changes):
    rec = {"curve": [800.0, 160.0, -2.0], "image_size": [FRAME_W, FRAME_H],
           "undistort_alpha": MEASURED.preprocess.undistort_alpha,
           "lens_calibration": {"sha256": lens_id(MEASURED.preprocess.calibration_path)},
           "marks": [{"cm": 3.0, "rows": 0.0}, {"cm": 10.0, "rows": 55.0}], "error_cm": {"max": 0.2}}
    rec.update(changes)
    path = tmp_path / "stop_line_table.json"
    path.write_text(json.dumps(rec))
    return path


def load(path):
    return load_stop_line_table(path, MEASURED.preprocess, (FRAME_H, FRAME_W))


@pytest.mark.software
def test_a_matching_file_loads_with_its_curve_reach_and_error(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = load(table_file(tmp_path))
    assert (t.a, t.b, t.c, t.max_rows, t.error_max_cm) == (800.0, 160.0, -2.0, 55.0, 0.2)
    assert t.image_size == (FRAME_W, FRAME_H)


@pytest.mark.software
def test_no_file_is_silent(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load(tmp_path / "missing.json") is None


@pytest.mark.software
@pytest.mark.parametrize("changes, words", [
    ({"curve": [1.0, 2.0]}, "malformed"),
    ({"marks": []}, "malformed"),
    ({"image_size": [640, 360]}, "640x360"),
    ({"undistort_alpha": 0.5}, "undistort_alpha"),
    ({"lens_calibration": {"sha256": "0" * 64}}, "different lens"),
])
def test_a_file_that_does_not_match_is_refused_with_the_reason(tmp_path, changes, words):
    with pytest.warns(UserWarning, match=words):
        assert load(table_file(tmp_path, **changes)) is None


@pytest.mark.software
def test_a_table_is_refused_when_undistortion_is_off(tmp_path):
    with pytest.warns(UserWarning, match="undistortion is off"):
        assert load_stop_line_table(table_file(tmp_path), replace(MEASURED.preprocess, calibration_path=None),
                                    (FRAME_H, FRAME_W)) is None
