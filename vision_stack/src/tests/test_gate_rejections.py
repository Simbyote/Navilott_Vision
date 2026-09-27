"""
test_gate_rejections.py  --  src/analysis/gate_rejections.py

--software  Group and gate discovery from the geometry and color CSV
            headers the hardware tests write, totals and ordering,
            the bucket consistency check, input resolution and the CLI.
"""
import csv

import pytest

from src.analysis import gate_rejections as gr
from src.analysis.common import Table

GEOMETRY_HEADER = (["frame_id", "timestamp_ms", "stage_ms", "lane_ms", "sign_ms", "n_lane", "n_sign"]
                   + [f"lane_{k}" for k in ("seen", "area", "degenerate", "too_few_pts", "aspect",
                                            "w_span", "h_span", "intensity", "accepted")]
                   + [f"sign_{k}" for k in ("seen", "area", "vertices", "hull", "solidity", "accepted")])
COLOR_HEADER = (["frame_id", "timestamp_ms", "stage_ms", "n_candidates"]
                + [f"n_{c}" for c in ("red", "yellow", "green")]
                + [f"mask_px_{c}" for c in ("red", "yellow", "green")]
                + [f"{c}_{g}" for c in ("red", "yellow", "green") for g in ("seen", "area", "aspect", "accepted")])


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def geometry_row(lane, sign):
    """lane, sign: dicts of bucket counts; seen is their sum."""
    lane_keys = GEOMETRY_HEADER[8:16]
    sign_keys = GEOMETRY_HEADER[17:]
    lv = [lane.get(k[5:], 0) for k in lane_keys]
    sv = [sign.get(k[5:], 0) for k in sign_keys]
    return [0, 0, 5.0, 3.0, 2.0, 0, 0, sum(lv), *lv, sum(sv), *sv]


@pytest.mark.software
def test_groups_are_found_from_the_real_headers():
    g = gr.groups(GEOMETRY_HEADER)
    assert set(g) == {"lane", "sign"}
    assert "lane_ms" not in g["lane"] and "lane_area" in g["lane"]
    assert set(gr.groups(COLOR_HEADER)) == {"red", "yellow", "green"}
    assert "mask_px_red" not in gr.groups(COLOR_HEADER)["red"]


@pytest.mark.software
def test_totals_shares_and_order(tmp_path):
    rows = [geometry_row({"area": 6, "aspect": 2, "accepted": 2}, {"vertices": 1}),
            geometry_row({"area": 4, "intensity": 3, "accepted": 3}, {})]
    r = gr.analyze(Table(write_csv(tmp_path / "geometry_timing.csv", GEOMETRY_HEADER, rows)))
    lane = r["lane"]
    assert lane["seen"] == 20 and lane["unbucketed"] == 0
    assert [x["gate"] for x in lane["gates"]][:3] == ["area", "intensity", "aspect"]
    assert lane["gates"][-1]["gate"] == "accepted" and lane["gates"][-1]["count"] == 5
    assert lane["gates"][0]["share"] == pytest.approx(0.5) and lane["gates"][0]["frames"] == 2
    assert r["sign"]["frames_with_candidates"] == 1


@pytest.mark.software
def test_inconsistent_buckets_are_reported(tmp_path):
    row = geometry_row({"area": 2}, {})
    row[7] = 5                                          # lane_seen larger than its buckets
    r = gr.analyze(Table(write_csv(tmp_path / "g.csv", GEOMETRY_HEADER, [row])))
    assert r["lane"]["unbucketed"] == 3


@pytest.mark.software
def test_a_csv_without_groups_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="_seen"):
        gr.analyze(Table(write_csv(tmp_path / "x.csv", ["frame_id", "stage_ms"], [[0, 1]])))


@pytest.mark.software
def test_a_run_folder_yields_both_csvs(tmp_path):
    write_csv(tmp_path / "run" / "test_geometry" / "geometry_timing.csv", GEOMETRY_HEADER,
              [geometry_row({"area": 1}, {})])
    write_csv(tmp_path / "run" / "test_color" / "color_timing.csv", COLOR_HEADER, [[0] * len(COLOR_HEADER)])
    assert sorted(p.name for p in gr.resolve([tmp_path / "run"])) == ["color_timing.csv", "geometry_timing.csv"]


@pytest.mark.software
def test_cli(tmp_path, capsys):
    write_csv(tmp_path / "geometry_timing.csv", GEOMETRY_HEADER, [geometry_row({"area": 3, "accepted": 1}, {})])
    assert gr.main([str(tmp_path / "geometry_timing.csv")]) == 0
    out = capsys.readouterr().out
    assert "lane: 4 candidates" in out and (tmp_path / "gate_rejections.json").exists()


@pytest.mark.software
def test_figure(tmp_path):
    pytest.importorskip("matplotlib")
    path = write_csv(tmp_path / "geometry_timing.csv", GEOMETRY_HEADER,
                     [geometry_row({"area": 3, "accepted": 1}, {"hull": 2})])
    assert gr.figure(gr.analyze(Table(path)), "t", tmp_path / "g.png").stat().st_size > 0
