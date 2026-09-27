"""
test_common.py  --  src/analysis/common.py

--software  Table parsing (blank and text cells, short rows, skip), interval
            source order, find_csv over files, folders and roots, stats and
            runs_of known answers, JSON output of numpy values.
"""
import csv
import json
import math
import os

import numpy as np
import pytest

from src.analysis import common
from src.analysis.common import Table


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.mark.software
def test_blank_and_text_cells_read_as_nan_and_text_is_kept(tmp_path):
    t = Table(write_csv(tmp_path / "a.csv", ["x", "mode"], [["1.5", "none"], ["", "left_only"], ["None", ""]]))
    assert t.numeric("x")[0] == 1.5 and math.isnan(t.numeric("x")[1]) and math.isnan(t.numeric("x")[2])
    assert t.text("mode") == ["none", "left_only", ""]


@pytest.mark.software
def test_short_rows_are_padded_with_blanks(tmp_path):
    t = Table(write_csv(tmp_path / "a.csv", ["a", "b"], [["1"], ["2", "3"]]))
    assert t.text("b") == ["", "3"]


@pytest.mark.software
def test_skip_drops_leading_rows_and_rejects_too_few(tmp_path):
    path = write_csv(tmp_path / "a.csv", ["a"], [[i] for i in range(4)])
    assert Table(path, skip=3).numeric("a").tolist() == [3.0]
    with pytest.raises(ValueError, match="skip=4"):
        Table(path, skip=4)


@pytest.mark.software
def test_header_names_are_stripped(tmp_path):
    assert Table(write_csv(tmp_path / "a.csv", [" a ", "b\r"], [[1, 2]])).columns == ["a", "b"]


@pytest.mark.software
def test_has_values_needs_at_least_one_number(tmp_path):
    t = Table(write_csv(tmp_path / "a.csv", ["cm", "off"], [["None", "0.1"], ["None", "0.2"]]))
    assert not t.has_values("cm") and t.has_values("off")
    assert t.first_with_values("cm", "off") == "off"


@pytest.mark.software
@pytest.mark.parametrize("header, row, want", [
    (["interval_ms", "dt_ms", "timestamp_ms"], [12, 50, 0], 12.0),
    (["dt_ms", "timestamp_ms"], [50, 0], 50.0),
    (["dt_s", "timestamp_ms"], [0.083, 0], 83.0),
])
def test_interval_source_order(tmp_path, header, row, want):
    t = Table(write_csv(tmp_path / "a.csv", header, [row, row]))
    assert common.interval_ms(t)[1] == pytest.approx(want)


@pytest.mark.software
def test_interval_from_timestamps_has_no_first_value(tmp_path):
    t = Table(write_csv(tmp_path / "a.csv", ["timestamp_ms"], [[0], [83], [166]]))
    iv = common.interval_ms(t)
    assert math.isnan(iv[0]) and iv[1:].tolist() == [83.0, 83.0]


@pytest.mark.software
def test_no_interval_source_gives_none(tmp_path):
    assert common.interval_ms(Table(write_csv(tmp_path / "a.csv", ["x"], [[1]]))) is None


@pytest.mark.software
def test_find_csv_takes_a_file_as_given(tmp_path):
    f = write_csv(tmp_path / "anything.csv", ["x"], [[1]])
    assert common.find_csv(f, "stages.csv") == f


@pytest.mark.software
def test_find_csv_prefers_files_directly_in_the_folder_then_searches_below(tmp_path):
    write_csv(tmp_path / "stages.csv", ["x"], [[1]])
    deep = write_csv(tmp_path / "t" / "p3.csv", ["x"], [[1]])
    assert common.find_csv(tmp_path, ("p3.csv", "stages.csv")).name == "stages.csv"
    assert common.find_csv(tmp_path / "t", ("stages.csv", "p3.csv")) == deep
    assert common.find_csv(tmp_path, "p3.csv") == deep


@pytest.mark.software
def test_find_csv_without_arg_takes_newest_under_roots(tmp_path):
    old = write_csv(tmp_path / "r1" / "a" / "p3.csv", ["x"], [[1]])
    new = write_csv(tmp_path / "r2" / "b" / "p3.csv", ["x"], [[1]])
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert common.find_csv(None, "p3.csv", roots=(tmp_path / "r1", tmp_path / "r2")) == new


@pytest.mark.software
def test_find_csv_error_names_the_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="p3.csv"):
        common.find_csv(tmp_path, "p3.csv")


@pytest.mark.software
def test_stats_ignore_nan_and_report_empty_as_none():
    s = common.stats([1, 2, np.nan, 3])
    assert s["n"] == 3 and s["p50"] == 2.0 and s["max"] == 3.0
    assert common.stats([np.nan])["mean"] is None


@pytest.mark.software
def test_runs_of_groups_consecutive_labels():
    assert common.runs_of(list("aabccc")) == [("a", 0, 2), ("b", 2, 1), ("c", 3, 3)]
    assert common.runs_of([]) == []


@pytest.mark.software
def test_write_json_handles_numpy_values(tmp_path):
    path = common.write_json(tmp_path / "x" / "o.json", {"a": np.float64(1.5), "b": np.arange(2)})
    assert json.loads(path.read_text()) == {"a": 1.5, "b": [0, 1]}


@pytest.mark.software
@pytest.mark.parametrize("v, want", [(None, ""), (float("nan"), ""), (1.23456, "1.23")])
def test_fmt(v, want):
    assert common.fmt(v) == want


@pytest.mark.software
def test_read_manifest_resolves_runs_against_its_folder(tmp_path):
    (tmp_path / "m.csv").write_text(" distance_cm , run \n20, runs/a\n-4,b\n")
    assert common.read_manifest(tmp_path / "m.csv", "distance_cm") == [
        (20.0, tmp_path / "runs/a"), (-4.0, tmp_path / "b")]


@pytest.mark.software
def test_read_manifest_names_the_columns_it_needs(tmp_path):
    (tmp_path / "m.csv").write_text("x,run\n1,a\n")
    with pytest.raises(ValueError, match="true_cm,run"):
        common.read_manifest(tmp_path / "m.csv", "true_cm")
