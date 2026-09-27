"""
test_offset_accuracy.py  --  src/analysis/offset_accuracy.py

--software  Bias, spread, fit and verdict on positions with known errors;
            sign disagreement; manifest and --at inputs; the normalized
            fallback; the command line and its exit status.
"""
import csv

import numpy as np
import pytest

from src.analysis import offset_accuracy as oa


def write_p3(path, cm_values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "p2_mode", "p2_offset", "lane_offset_cm", "lane_status"])
        for i, v in enumerate(cm_values):
            w.writerow([i, "two_boundary", v / 10, v, "vision"])
    return path


@pytest.mark.software
def test_perfect_measurements_pass_with_unit_slope():
    r = oa.analyze([(t, [t] * 10) for t in (-4, -2, 0, 2, 4)])
    assert r["verdict"] == "PASS" and r["rms_bias"] == pytest.approx(0)
    assert r["fit"]["slope"] == pytest.approx(1) and r["fit"]["intercept"] == pytest.approx(0)


@pytest.mark.software
def test_a_position_biased_past_the_spec_fails():
    r = oa.analyze([(-2, [-2.0] * 5), (0, [0.5] * 5), (2, [4.5] * 5)])
    assert r["verdict"] == "FAIL" and r["worst_bias"] == pytest.approx(2.5)


@pytest.mark.software
def test_bias_and_within_spec_share_per_position():
    r = oa.analyze([(0, [1.0, 1.0, 3.0, 3.0])], spec_cm=2.0)
    pos = r["positions"][0]
    assert pos["bias"] == pytest.approx(2.0) and pos["within_spec"] == pytest.approx(0.5)


@pytest.mark.software
def test_a_flipped_sign_fails_even_when_every_bias_is_small():
    r = oa.analyze([(-1, [0.8] * 5), (1, [-0.8] * 5)], spec_cm=2.0)
    assert r["sign_agrees"] is False and r["verdict"] == "FAIL"


@pytest.mark.software
def test_positions_are_sorted_and_nans_ignored():
    r = oa.analyze([(2, [2.0, np.nan]), (-2, [-2.0])])
    assert [p["true_cm"] for p in r["positions"]] == [-2, 2] and r["positions"][1]["n"] == 1


@pytest.mark.software
def test_a_single_position_has_no_fit():
    r = oa.analyze([(0, [0.1] * 5)])
    assert r["fit"] is None and r["sign_agrees"] is None and r["verdict"] == "PASS"


@pytest.mark.software
def test_a_position_with_no_measured_frames_is_an_error():
    with pytest.raises(ValueError, match="no measured frames"):
        oa.analyze([(0, [np.nan])])


@pytest.mark.software
def test_manifest_runs_resolve_against_its_folder(tmp_path):
    write_p3(tmp_path / "A" / "p3.csv", [-2.1] * 10)
    write_p3(tmp_path / "B" / "p3.csv", [2.2] * 10)
    (tmp_path / "positions.csv").write_text("true_cm,run\n-2,A\n2,B\n")
    assert oa.main([str(tmp_path / "positions.csv"), "--skip", "0"]) == 0
    assert (tmp_path / "offset_accuracy.json").exists()


@pytest.mark.software
def test_at_arguments_accept_negative_positions(tmp_path, capsys):
    write_p3(tmp_path / "A" / "p3.csv", [-4.0] * 10)
    write_p3(tmp_path / "B" / "p3.csv", [4.0] * 10)
    code = oa.main(["--at", "-4", str(tmp_path / "A"), "--at", "4", str(tmp_path / "B"),
                    "--skip", "0", "--out", str(tmp_path)])
    assert code == 0 and "verdict: PASS" in capsys.readouterr().out


@pytest.mark.software
def test_fail_exits_2(tmp_path):
    write_p3(tmp_path / "A" / "p3.csv", [5.0] * 10)
    assert oa.main(["--at", "0", str(tmp_path / "A"), "--skip", "0", "--out", str(tmp_path)]) == 2


@pytest.mark.software
def test_normalized_runs_need_a_scale(tmp_path, capsys):
    path = tmp_path / "A" / "lane_offset_timing.csv"
    path.parent.mkdir()
    path.write_text("frame_id,mode,offset\n0,two_boundary,0.1\n1,two_boundary,0.1\n")
    assert oa.main(["--at", "1", str(path.parent), "--skip", "0", "--out", str(tmp_path)]) == 1
    assert "--cm-per-unit" in capsys.readouterr().err
    assert oa.main(["--at", "1", str(path.parent), "--skip", "0", "--cm-per-unit", "10",
                    "--out", str(tmp_path)]) == 0


@pytest.mark.software
def test_a_manifest_without_the_columns_is_rejected(tmp_path, capsys):
    (tmp_path / "m.csv").write_text("cm,path\n0,A\n")
    assert oa.main([str(tmp_path / "m.csv")]) == 1
    assert "true_cm,run" in capsys.readouterr().err


@pytest.mark.software
def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    r = oa.analyze([(t, [t + 0.3] * 5) for t in (-4, 0, 4)])
    assert oa.figure(r, "t", tmp_path / "f.png").stat().st_size > 0
