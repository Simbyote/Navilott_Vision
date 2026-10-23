"""
test_parallel_ab.py  --  src/analysis/parallel_ab.py

The runs alternate serial and parallel with a label each, every run's
numbers come from its own nav.csv and report.json, and the verdict says
whether the modes separate or overlap. No camera: the launcher is a fake
that writes a run folder.
"""
import csv
import json

import pytest

from src.analysis import parallel_ab as ab
from src.navigation_linker import NAV_FIELDS

pytestmark = pytest.mark.software


def write_run(folder, parallel, interval_ms, phase2_ms, frames=40, label=""):
    """A run folder: nav.csv with frames every interval_ms, Phase 2 at phase2_ms (frame 0 slow), report.json."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "nav.csv", "w", newline="") as f:
        w = csv.DictWriter(f, NAV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for i in range(frames):
            w.writerow({"frame_id": i, "t": round(i * interval_ms / 1000.0, 4),
                        "phase2_ms": 1300.0 if i == 0 else phase2_ms + (i % 2), "phase3_ms": 0.7,
                        "latency_ms": phase2_ms + 2.0})
    (folder / "report.json").write_text(json.dumps({"parallel_perception": parallel, "label": label}))
    return folder


def fake_launch(times):
    """navigation_linker.cli stand-in: writes the run its arguments ask for; times[mode] = (interval, phase2)."""
    calls = []

    def launch(args):
        calls.append(args)
        mode = "parallel" if "--parallel" in args else "serial"
        out = args[args.index("--out") + 1]
        label = args[args.index("--label") + 1]
        from pathlib import Path
        write_run(Path(out), mode == "parallel", *times[mode], label=label)
        return 0
    return launch, calls


def test_a_runs_numbers_come_from_its_own_files_leaving_out_the_start(tmp_path):
    s = ab.run_stats(write_run(tmp_path / "r", True, 55.0, 44.0, label="x"))
    assert (s["mode"], s["label"], s["frames"]) == ("parallel", "x", 39)
    assert s["interval_p50"] == pytest.approx(55.0) and s["fps"] == pytest.approx(1000 / 55, abs=0.01)
    assert s["phase2_p50"] == pytest.approx(44.5, abs=0.5) and s["over_budget_pct"] == 0.0


def test_runs_alternate_from_serial_each_labelled_and_dry_unless_asked(tmp_path):
    launch, calls = fake_launch({"serial": (55.0, 45.0), "parallel": (50.0, 38.0)})
    said = []
    assert ab.main(["--runs", "4", "--seconds", "30", "--out", str(tmp_path / "ab")], launch=launch,
                   say=said.append) == 0
    modes = ["--parallel" if "--parallel" in c else "--serial" for c in calls]
    assert modes == ["--serial", "--parallel", "--serial", "--parallel"]
    assert all("--no-motors" in c and c[c.index("--max-run-s") + 1] == "30" for c in calls)
    assert calls[1][calls[1].index("--label") + 1] == "A/B parallel 2"
    result = json.loads((tmp_path / "ab" / "ab.json").read_text())
    assert result["separated"] and result["change"]["interval_p50"] == pytest.approx(-5.0)
    assert "parallel faster in every run" in "\n".join(said)


def test_overlapping_runs_are_no_clear_difference():
    stats = [{"mode": m, **{k: v for k in ab.METRICS}} for m, v in
             (("serial", 54.0), ("parallel", 53.0), ("serial", 52.0), ("parallel", 55.0))]
    r = ab.compare(stats)
    assert not r["separated"] and r["verdict"].startswith("no clear difference")


def test_parallel_slower_everywhere_says_so():
    stats = [{"mode": m, **{k: v for k in ab.METRICS}} for m, v in
             (("serial", 50.0), ("parallel", 56.0), ("serial", 51.0), ("parallel", 57.0))]
    assert ab.compare(stats)["verdict"].startswith("parallel slower in every run")


def test_motors_wait_for_enter_before_each_run_and_extra_arguments_pass_through(tmp_path):
    launch, calls = fake_launch({"serial": (55.0, 45.0), "parallel": (55.0, 45.0)})
    asked = []
    ab.main(["--runs", "2", "--motors", "--out", str(tmp_path / "ab"), "--", "--no-record"], launch=launch,
            ask=asked.append, say=lambda s: None)
    assert len(asked) == 2 and all("--no-motors" not in c and c[-1] == "--no-record" for c in calls)


def test_compare_only_reads_finished_runs(tmp_path):
    a = write_run(tmp_path / "a", False, 55.0, 45.0)
    b = write_run(tmp_path / "b", True, 51.0, 40.0)
    said = []
    assert ab.main(["--compare", str(a), str(b), "--out", str(tmp_path)], say=said.append) == 0
    assert json.loads((tmp_path / "ab.json").read_text())["separated"]


def test_a_failed_run_stops_the_ab_and_too_few_runs_are_refused(tmp_path):
    said = []
    assert ab.main(["--runs", "1"], launch=lambda a: 0, say=said.append) == 2
    assert ab.main(["--runs", "4", "--out", str(tmp_path)], launch=lambda a: 3, say=said.append) == 3
    assert any("ended with code 3" in s for s in said)
