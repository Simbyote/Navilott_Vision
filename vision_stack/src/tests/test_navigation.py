"""
test_navigation.py  --  the Navigation contract: Command rules, the Navigator interface, and the checks

A navigator that keeps the contract passes every check in navigation_checks;
each broken navigator here breaks exactly one rule, and the checks must name it.
"""
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from src.navigation import BRAKE, STALL_DUTY, Command, Navigator, command_problems
from src.tests.navigation_checks import (
    CASE_FRAMES, FRAME_MS, WARMUP_FRAMES, check_commands, check_no_forward_on_stale,
    check_no_forward_on_stop, check_steers_toward_center, contract_problems, frames, forward, packet,
)

BASE_DUTY = 0.40      # maneuver_linker's leg duty
GAIN = 0.10           # duty per unit of lane_offset


class GoodNavigator:
    """Brakes on stop or a stale lane; otherwise drives forward and steers against the offset."""
    def __init__(self):
        self.calls = 0

    def update(self, packet):
        self.calls += 1
        if packet.drive_state == "stop" or packet.lane_status == "stale":
            return BRAKE
        return Command(BASE_DUTY - GAIN * packet.lane_offset, BASE_DUTY + GAIN * packet.lane_offset)

    def reset(self):
        self.calls = 0


class Overdrives(GoodNavigator):
    def update(self, packet):
        cmd = super().update(packet)
        return cmd if cmd.brake else Command(1.5, 1.5)


class Stalls(GoodNavigator):
    def update(self, packet):
        cmd = super().update(packet)
        return cmd if cmd.brake else Command(0.1, 0.1)


class BrakesWithDuty(GoodNavigator):
    def update(self, packet):
        cmd = super().update(packet)
        return Command(0.4, 0.4, brake=True) if cmd.brake else cmd


class ReturnsNone(GoodNavigator):
    def update(self, packet):
        return None


class DrivesOnStale(GoodNavigator):
    def update(self, packet):
        return super().update(replace(packet, lane_status="vision"))


class DrivesOnStop(GoodNavigator):
    def update(self, packet):
        return super().update(replace(packet, drive_state="go"))


class SteersAway(GoodNavigator):
    def update(self, packet):
        return super().update(replace(packet, lane_offset=-packet.lane_offset))


class NeverDrives(GoodNavigator):
    def update(self, packet):
        return BRAKE


# =============================================================================
# Command
# =============================================================================

@pytest.mark.software
@pytest.mark.parametrize("module", ["src.navigation", "src.config", "src.maneuver", "src.lane_keeping",
                                    "src.scripts.lane_keeping_demo"])
def test_the_contract_and_its_users_load_without_motor_hardware(module):
    # pigpio blocked, as on a laptop: nothing above the drivers may import it at load
    code = f"import sys; sys.modules['pigpio'] = None; import {module}"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[2])
    assert r.returncode == 0, r.stderr


@pytest.mark.software
def test_brake_is_a_zero_duty_brake_and_the_default_command_coasts():
    assert (BRAKE.left, BRAKE.right, BRAKE.brake) == (0.0, 0.0, True)
    assert (Command().left, Command().right, Command().brake) == (0.0, 0.0, False)


@pytest.mark.software
def test_commands_are_frozen():
    with pytest.raises(FrozenInstanceError):
        BRAKE.left = 0.5


@pytest.mark.software
@pytest.mark.parametrize("cmd", [BRAKE, Command(), Command(1.0, -1.0), Command(-STALL_DUTY, STALL_DUTY),
                                 Command(0.0, 0.4), Command(-0.45, 0.45)])
def test_valid_commands_have_no_problems(cmd):
    assert command_problems(cmd) == []


@pytest.mark.software
@pytest.mark.parametrize("cmd, words", [
    (Command(1.01, 0.5), ["left", "outside"]),
    (Command(0.5, -1.01), ["right", "outside"]),
    (Command(0.24, 0.5), ["left", "stall"]),
    (Command(0.5, -0.1), ["right", "stall"]),
    (Command(0.0, 0.4, brake=True), ["brake"]),
    (Command(0.4, 0.0, brake=True), ["brake"]),
])
def test_each_broken_rule_is_named(cmd, words):
    problems = command_problems(cmd)
    assert len(problems) == 1 and all(w in problems[0] for w in words), problems


@pytest.mark.software
def test_every_broken_rule_is_listed_not_just_the_first():
    assert len(command_problems(Command(2.0, 0.1, brake=True))) == 3


@pytest.mark.software
@pytest.mark.parametrize("bad", [None, (0.4, 0.4), 0.4])
def test_anything_but_a_command_is_a_problem(bad):
    assert command_problems(bad) == [f"not a Command: {bad!r}"]


# =============================================================================
# Navigator
# =============================================================================

@pytest.mark.software
def test_the_navigator_interface_is_update_and_reset():
    assert isinstance(GoodNavigator(), Navigator)

    class NoReset:
        def update(self, packet):
            return BRAKE
    assert not isinstance(NoReset(), Navigator)


# =============================================================================
# Packet builders
# =============================================================================

@pytest.mark.software
def test_built_packets_are_numbered_and_spaced_a_frame_apart():
    ps = frames([{}, {"lane_offset": 0.3}, {}], start=5)
    assert [p.frame_id for p in ps] == [5, 6, 7]
    assert [p.timestamp_ms for p in ps] == [5 * FRAME_MS, 6 * FRAME_MS, 7 * FRAME_MS]
    assert ps[1].lane_offset == 0.3 and ps[0].lane_offset == 0.0


@pytest.mark.software
def test_the_default_packet_is_a_centered_go_on_vision():
    p = packet()
    assert (p.lane_offset, p.lane_status, p.drive_state) == (0.0, "vision", "go")


@pytest.mark.software
@pytest.mark.parametrize("cmd, fwd", [(Command(0.4, 0.4), True), (Command(0.3, 0.0), True),
                                      (Command(-0.45, 0.45), False), (Command(-0.4, -0.4), False),
                                      (Command(), False), (BRAKE, False),
                                      (Command(0.4, 0.4, brake=True), False)])
def test_forward_means_a_positive_mean_duty_without_a_brake(cmd, fwd):
    assert forward(cmd) == fwd


# =============================================================================
# The checks
# =============================================================================

@pytest.mark.software
def test_a_navigator_that_keeps_the_contract_passes_every_check():
    assert contract_problems(GoodNavigator()) == []


@pytest.mark.software
def test_every_check_resets_warms_up_and_feeds_the_whole_case():
    nav = GoodNavigator()
    nav.calls = 99                   # left over from an earlier run; reset() clears it
    check_no_forward_on_stop(nav)
    assert nav.calls == WARMUP_FRAMES + CASE_FRAMES


@pytest.mark.software
def test_the_command_check_resets_first_and_feeds_every_packet():
    nav = GoodNavigator()
    nav.calls = 99
    check_commands(nav, frames([{}] * 7))
    assert nav.calls == 7


@pytest.mark.software
@pytest.mark.parametrize("nav, check, word", [
    (Overdrives(), lambda n: check_commands(n, frames([{}])), "outside"),
    (Stalls(), lambda n: check_commands(n, frames([{}])), "stall"),
    (BrakesWithDuty(), lambda n: check_commands(n, frames([{"drive_state": "stop"}])), "brake"),
    (ReturnsNone(), lambda n: check_commands(n, frames([{}])), "not a Command"),
    (DrivesOnStale(), check_no_forward_on_stale, "stale"),
    (DrivesOnStop(), check_no_forward_on_stop, "stop"),
    (SteersAway(), check_steers_toward_center, "toward center"),
    (NeverDrives(), check_steers_toward_center, "never drove forward"),
])
def test_each_broken_navigator_is_caught_by_its_check(nav, check, word):
    problems = check(nav)
    assert problems and all(word in p for p in problems), problems


@pytest.mark.software
@pytest.mark.parametrize("nav", [Overdrives(), Stalls(), BrakesWithDuty(), DrivesOnStale(),
                                 DrivesOnStop(), SteersAway(), NeverDrives()])
def test_the_full_contract_catches_every_broken_navigator(nav):
    assert contract_problems(nav)


@pytest.mark.software
def test_steering_is_checked_both_ways():
    class OnlyRightOffsets(GoodNavigator):
        """Steers correctly for + offsets, the wrong way for - ones."""
        def update(self, packet):
            return super().update(replace(packet, lane_offset=abs(packet.lane_offset)))
    problems = check_steers_toward_center(OnlyRightOffsets())
    assert problems and all("offset -0.5" in p for p in problems)


@pytest.mark.software
def test_stale_is_checked_at_every_old_offset():
    class DrivesOnStaleRight(GoodNavigator):
        def update(self, packet):
            if packet.lane_status == "stale" and packet.lane_offset > 0:
                return Command(0.4, 0.4)
            return super().update(packet)
    assert check_no_forward_on_stale(DrivesOnStaleRight())
