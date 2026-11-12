"""test_nav_replay.py  --  src/analysis/nav_replay.py

A hand-built run: a stop line comes down the view with a stop sign voted for
a long sighting at the first intersection and a short one (a ghost) at the
second. The robot's log (nav.csv) is what the rule before 2026-11-12 decided;
today's rule keeps the first stop and removes the second. Packets parse from
p3.csv with a column missing; the make target runs it.

--software  CSVs written to a temp folder. No camera or Pi.
"""
import csv

import pytest

from src.analysis import nav_replay as nr
from src.navigation.route import Route
from src.navigation.stop_sign import SIGN_MIN_FRAMES
from src.tests.navigation_checks import reach_frames

MS = 50
LINE = [48.0, 40.0, 32.0, 24.0, 15.0, 5.0]


def approach(sign_votes: int) -> list[dict]:
    """Lane frames with the sign voted sign_votes times, the line down the view, then 3 s of lane."""
    lane = [{} for _ in range(40)]
    for k in range(sign_votes):
        lane[20 + k]["stop_sign_detected"] = True
    line = [{"stop_line_detected": True, "stop_line_distance_px": r} for r in LINE]
    return lane + line + [{} for _ in range(reach_frames(frame_ms=MS) + 60)]


def write_run(folder, frames):
    cols = ["frame_id", "timestamp_ms", "lane_offset", "lane_status", "p2_mode", "drive_state",
            "stop_sign_detected", "stop_line_detected", "stop_line_distance_px"]     # no heading or wheels: defaults
    with open(folder / "p3.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for i, f in enumerate(frames):
            w.writerow({"frame_id": i, "timestamp_ms": i * MS, "lane_offset": 0.0, "lane_status": "vision",
                        "p2_mode": "two_boundary", "drive_state": "go",
                        "stop_sign_detected": f.get("stop_sign_detected", False),
                        "stop_line_detected": f.get("stop_line_detected", False),
                        "stop_line_distance_px": f.get("stop_line_distance_px", "")})
    # The robot's log: the old rule (any one vote counts) over the same packets
    old = nr.replay(folder / "p3.csv", Route(("straight", "straight")), 0.0, sign_min_frames=1)
    with open(folder / "nav.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["frame_id", "rule", "reason", "step", "maneuver"])
        w.writeheader()
        for fid, rule, reason, step in old:
            w.writerow({"frame_id": fid, "rule": rule, "reason": reason, "step": step,
                        "maneuver": step.split()[-1] if " " in step else ""})


@pytest.fixture
def run(tmp_path):
    write_run(tmp_path, approach(SIGN_MIN_FRAMES + 4) + approach(2))
    return tmp_path


@pytest.mark.software
def test_packets_parse_with_missing_columns_as_defaults(run):
    ps = list(nr.packets(run / "p3.csv"))
    assert ps[0].heading_error == 0.0 and ps[0].distance_cm is None and ps[0].lane_mode == "two_boundary"
    assert sum(p.stop_sign_detected for p in ps) == SIGN_MIN_FRAMES + 4 + 2


@pytest.mark.software
def test_todays_rule_keeps_the_real_sign_and_removes_the_ghost(run):
    res = nr.analyze(run, gyro_bias_dps=0.0)
    signs = {r["step"]: r["status"] for r in res["stops"] if r["rule"] == "stop_sign"}
    assert signs == {"1": "kept", "2": "removed"}
    assert res["counts"] == {"kept": 1, "removed": 1, "added": 0}
    assert res["off"] == ["wheel stall guard", "end of course"]


@pytest.mark.software
def test_the_old_rule_reproduces_the_log(run):
    res = nr.analyze(run, gyro_bias_dps=0.0, sign_min_frames=1)
    assert res["counts"] == {"kept": 2, "removed": 0, "added": 0}


@pytest.mark.software
def test_the_command_line_writes_json_and_says_what_is_off(run, capsys):
    assert nr.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "1 kept, 1 removed, 0 added" in out and "off: wheel stall guard, end of course" in out
    assert (run / "nav_replay.json").is_file()
    assert nr.main([str(run / "missing")]) == 1
