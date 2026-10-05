"""
test_makefile.py  --  vision_stack/Makefile

The Makefile only spells out commands, so what can go wrong is drift: a
target naming a module that moved, or passing a flag the command no longer
takes. Every target is expanded with `make -n` (nothing runs) and each
python command in it checked against the real command's --help: the module
runs, and every --flag the Makefile passes is one it lists. Then the
variables: a replay drops the camera, pigpiod and the dry flags; the options
reach their commands; an empty option adds nothing; and the targets that
need an argument refuse to run without it.

--software  Runs make and each module's --help. No hardware.
"""
import re
import shutil
import subprocess
import sys
from functools import lru_cache

import pytest

from src.params import PIPELINE_ROOT

pytestmark = [pytest.mark.software,
              pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")]

MAKEFILE = PIPELINE_ROOT / "Makefile"
# Targets that only print or tidy, with no python command to check
NO_COMMAND = {"help", "setup", "pigpiod", "newest", "clean", "compare", "calib-lamps", "render"}
# Arguments the guarded targets need before they run anything
NEEDS = {"compare": "BASE=a NEW=b", "calib-lamps": "LAMPS=red=x", "render": "RUN=runs/nav_x"}


def make_n(target, *assigns):
    """The commands `make -n target VAR=value ...` would run, continuation lines joined."""
    out = subprocess.run(["make", "-s", "-n", "-C", str(PIPELINE_ROOT), target, "PY=python3", *assigns],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [" ".join(line.split()) for line in out.stdout.replace("\\\n", " ").splitlines() if line.strip()]


def targets():
    """Every documented target (a '## ' comment), in file order."""
    return re.findall(r"^([a-z0-9-]+):[^\n]*## ", MAKEFILE.read_text(), re.M)


def python_commands(lines):
    """(module, args) for each `python3 -m module args` in the lines, wrapped commands included."""
    found = []
    for line in lines:
        for m in re.finditer(r"python3 -m (\S+)((?:(?!python3 -m ).)*)", line):
            found.append((m.group(1), m.group(2).split()))
    return found


@lru_cache(maxsize=None)
def help_text(module):
    out = subprocess.run([sys.executable, "-m", module, "--help"], cwd=PIPELINE_ROOT, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, f"python3 -m {module} --help failed: {out.stderr[-400:]}"
    return out.stdout


# =============================================================================
# Every target
# =============================================================================

def test_the_makefile_documents_every_target_it_declares_phony():
    text = MAKEFILE.read_text()
    phony = set(re.search(r"\.PHONY:(.*?)\n\n", text, re.S).group(1).replace("\\", " ").split())
    assert phony == set(targets())


@pytest.mark.parametrize("target", [t for t in sorted(set(targets()) - NO_COMMAND)])
def test_each_target_runs_a_real_module_with_flags_it_takes(target):
    cmds = python_commands(make_n(target))
    assert cmds, f"{target} runs no python command"
    for module, args in cmds:
        text = help_text(module)
        for flag in (a.split("=")[0] for a in args if a.startswith("--")):
            if flag == "--":
                continue
            assert re.search(rf"(^|\s){re.escape(flag)}([\s,=]|$)", text, re.M), \
                f"{target}: python3 -m {module} doesn't take {flag}"


@pytest.mark.parametrize("target", sorted(NEEDS))
def test_guarded_targets_refuse_to_run_without_their_argument(target, tmp_path):
    shutil.copy(MAKEFILE, tmp_path / "Makefile")       # no runs/ here, so render has no newest run to fall back on
    out = subprocess.run(["make", "-s", "-C", str(tmp_path), target, "PY=python3", "RUN=", "LAMPS=",
                          "BASE=", "NEW="], capture_output=True, text=True)
    assert out.returncode != 0
    assert "usage" in out.stdout or "make render RUN=" in out.stdout
    assert "python3 -m" not in out.stdout
    cmds = python_commands(make_n(target, *NEEDS[target].split()))
    assert cmds and all(help_text(m) for m, _ in cmds)


def test_help_lists_every_target():
    out = subprocess.run(["make", "-s", "-C", str(PIPELINE_ROOT), "help"], capture_output=True, text=True)
    assert out.returncode == 0
    listed = re.findall(r"^  ([a-z0-9-]+) ", re.sub(r"\x1b\[[0-9;]*m", "", out.stdout), re.M)
    assert listed == targets()


# =============================================================================
# The variables
# =============================================================================

def test_the_camera_is_the_default_source_and_a_replay_replaces_it():
    assert make_n("phase2")[-1] == "python3 -m src.phase2_linker --camera"
    assert make_n("phase2", "VIDEO=clip.mp4")[-1] == "python3 -m src.phase2_linker --video clip.mp4"
    assert make_n("phase2", "FRAMES=dir")[-1] == "python3 -m src.phase2_linker --frames dir"


def test_camera_targets_start_pigpiod_and_replays_dont():
    assert any("pigpiod" in line for line in make_n("navigate"))
    assert not any("pigpiod" in line for line in make_n("navigate", "VIDEO=clip.mp4"))
    assert not any("pigpiod" in line for line in make_n("phase2"))       # no encoders or motors


def test_dry_runs_turn_the_motors_off_on_the_camera_only():
    assert make_n("nav-dry")[-1].endswith("--camera --no-motors --no-button --route route.json")
    replay = make_n("nav-dry", "VIDEO=clip.mp4")[-1]
    assert "--video clip.mp4" in replay and "--no-motors" not in replay       # replays never drive
    assert "--imu --encoders" in make_n("phase3")[-1]
    assert "--imu" not in make_n("phase3", "FRAMES=d")[-1]


def test_options_reach_their_commands_and_empty_ones_add_nothing():
    line = make_n("navigate", "MAX_RUN_S=60", "ROUTE=my.json", "ARGS=--no-render")[-1]
    assert line.endswith("--camera --route my.json --max-run-s 60 --no-render")
    assert "--max-run-s" not in make_n("navigate")[-1]
    assert make_n("phase2", "VIEWS=traffic,stop")[-1].endswith("--views traffic,stop")
    assert "--views" not in make_n("phase2")[-1]
    assert make_n("pi-load", "RUN=runs/diag_x", "NAV=runs/nav_y")[-1].endswith("pi_load runs/diag_x --run runs/nav_y")
    assert make_n("intersection", "MANEUVER=left")[-1].endswith("--camera left")
    assert make_n("soak", "MINUTES=30")[-1].endswith("--soak-minutes=30 src/tests/test_soak.py")
    assert make_n("test-hw", "HW_FRAMES=600")[-1].endswith("--frames=600")


def test_the_venv_is_used_when_there_is_one(tmp_path):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    out = subprocess.run(["make", "-s", "-n", "-C", str(tmp_path), "lint"], capture_output=True, text=True)
    assert out.stdout.strip() == ".venv/bin/python -m pyflakes src"
    (tmp_path / ".venv" / "bin" / "python").unlink()
    out = subprocess.run(["make", "-s", "-n", "-C", str(tmp_path), "lint"], capture_output=True, text=True)
    assert out.stdout.strip() == "python3 -m pyflakes src"


def test_no_option_carries_a_comment_after_its_value():
    # make keeps the spaces before a trailing '#' in the value: VIDEO ?=   # comment
    # would make VIDEO non-empty, and every linker would replay "--video "
    for line in MAKEFILE.read_text().splitlines():
        if "?=" in line:
            assert "#" not in line, line


def test_compare_needs_both_runs(tmp_path):
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    out = subprocess.run(["make", "-s", "-C", str(tmp_path), "compare", "PY=python3", "BASE=a"],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "usage" in out.stdout and "python3 -m" not in out.stdout
