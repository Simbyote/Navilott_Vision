"""
test_navigation_contract.py  --  the Navigation contract: Command rules, the Navigator interface, and the checks

A navigator that keeps the contract passes every check in navigation_checks;
each broken navigator here breaks exactly one rule, and the checks must name it.
"""
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from src.navigation.navigation import Navigation
from src.navigation.navigation_contract import (
    BRAKE, STALL_DUTY, Command, Navigator, command_problems, enforce,
)
from src.tests.navigation_checks import (
    APPROACH_ROWS, CASE_FRAMES, FRAME_MS, WARMUP_FRAMES, check_commands, check_crosses_a_green_line,
    check_goes_when_the_light_turns_green, check_ignores_a_red_light_without_a_line,
    check_stale_lane_slows_then_stops, check_steers_toward_center, check_stops_at_a_red_line,
    check_stops_at_a_stop_sign_line_then_goes, contract_problems, frames, forward, intersection, packet,
)

class GoodNavigator:
    """navigation.Navigation, counting its calls: a navigator that keeps the whole contract."""
    def __init__(self):
        self.nav, self.calls = Navigation(), 0

    def update(self, packet):
        self.calls += 1
        return self.nav.update(packet)

    def reset(self):
        self.calls = 0
        self.nav.reset()


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


class SpeedsOnStale(GoodNavigator):
    """Full speed on a stale lane, though it still stops in the end."""
    def update(self, packet):
        cmd = super().update(packet)
        return Command(0.4, 0.4) if packet.lane_status == "stale" and not cmd.brake else cmd


class NeverEndsOnStale(GoodNavigator):
    """Creeps on a stale lane forever."""
    def update(self, packet):
        return Command(0.3, 0.3) if packet.lane_status == "stale" else super().update(packet)


class StopsTooLate(GoodNavigator):
    """Creeps on a stale lane and stops, but only after 50 frames."""
    def __init__(self):
        super().__init__()
        self.stale = 0

    def update(self, packet):
        self.stale = self.stale + 1 if packet.lane_status == "stale" else 0
        if packet.lane_status != "stale":
            return super().update(packet)
        return BRAKE if self.stale > 50 else Command(0.3, 0.3)

    def reset(self):
        super().reset()
        self.stale = 0


class ResumesOnStale(GoodNavigator):
    """Stops on a stale lane, then creeps on again."""
    def __init__(self):
        super().__init__()
        self.stale = 0

    def update(self, packet):
        self.stale = self.stale + 1 if packet.lane_status == "stale" else 0
        if packet.lane_status != "stale":
            return super().update(packet)
        return BRAKE if 10 <= self.stale < 15 else Command(0.3, 0.3)

    def reset(self):
        super().reset()
        self.stale = 0


class RunsRedLines(GoodNavigator):
    """Never sees a red light."""
    def update(self, packet):
        return super().update(replace(packet, drive_state="go"))


class StopsForRedAnywhere(GoodNavigator):
    """Brakes on any red light, line or no line."""
    def update(self, packet):
        cmd = super().update(packet)
        return BRAKE if packet.drive_state == "stop" else cmd


class IgnoresStopSigns(GoodNavigator):
    def update(self, packet):
        return super().update(replace(packet, stop_sign_detected=False))


class StopsAtEveryLine(GoodNavigator):
    """Treats every stop line as having a stop sign."""
    def update(self, packet):
        return super().update(replace(packet, stop_sign_detected=packet.stop_sign_detected
                                      or packet.stop_line_detected))


class BrakesOnSight(GoodNavigator):
    """Brakes as soon as it sees a stop line with a sign or a red light, short of the line."""
    def update(self, packet):
        cmd = super().update(packet)
        stop = packet.stop_sign_detected or packet.drive_state == "stop"
        return BRAKE if packet.stop_line_detected and stop else cmd


class NeverGoesAgain(GoodNavigator):
    """Once it has braked, it stays braked."""
    def __init__(self):
        super().__init__()
        self.stuck = False

    def update(self, packet):
        cmd = super().update(packet)
        self.stuck = self.stuck or cmd.brake
        return BRAKE if self.stuck else cmd

    def reset(self):
        super().reset()
        self.stuck = False


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
@pytest.mark.parametrize("module", ["src.navigation.navigation_contract", "src.navigation.navigation", "src.config",
                                    "src.maneuver", "src.navigation.lane_keeping", "src.navigation.stop_line",
                                    "src.navigation.stop_sign", "src.navigation.traffic_light",
                                    "src.navigation.intersection", "src.scripts.lane_keeping_demo"])
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
    check_ignores_a_red_light_without_a_line(nav)
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
    (BrakesWithDuty(), lambda n: check_commands(n, frames([{"drive_state": "stop"}] * 7 + [{"lane_status": "stale"}] * 30)), "brake"),
    (ReturnsNone(), lambda n: check_commands(n, frames([{}])), "not a Command"),
    (DrivesOnStale(), check_stale_lane_slows_then_stops, "stale"),
    (SpeedsOnStale(), check_stale_lane_slows_then_stops, "faster than"),
    (NeverEndsOnStale(), check_stale_lane_slows_then_stops, "never stopped"),
    (StopsTooLate(), check_stale_lane_slows_then_stops, "never stopped within"),
    (ResumesOnStale(), check_stale_lane_slows_then_stops, "drives again"),
    (RunsRedLines(), check_stops_at_a_red_line, "never braked at a red light"),
    (RunsRedLines(), check_goes_when_the_light_turns_green, "never stopped at the red light"),
    (StopsForRedAnywhere(), check_ignores_a_red_light_without_a_line, "no stop line"),
    (IgnoresStopSigns(), check_stops_at_a_stop_sign_line_then_goes, "never stopped"),
    (StopsAtEveryLine(), check_crosses_a_green_line, "green stop line"),
    (BrakesOnSight(), check_stops_at_a_stop_sign_line_then_goes, "still in view"),
    (BrakesOnSight(), check_stops_at_a_red_line, "still in view"),
    (NeverGoesAgain(), check_stops_at_a_stop_sign_line_then_goes, "never drove on"),
    (NeverGoesAgain(), check_goes_when_the_light_turns_green, "never drove on"),
    (SteersAway(), check_steers_toward_center, "toward center"),
    (NeverDrives(), check_steers_toward_center, "never drove forward"),
])
def test_each_broken_navigator_is_caught_by_its_check(nav, check, word):
    problems = check(nav)
    assert problems and all(word in p for p in problems), problems


@pytest.mark.software
@pytest.mark.parametrize("nav", [Overdrives(), Stalls(), BrakesWithDuty(), DrivesOnStale(), SpeedsOnStale(),
                                 NeverEndsOnStale(), StopsTooLate(), ResumesOnStale(), RunsRedLines(),
                                 StopsForRedAnywhere(), IgnoresStopSigns(), StopsAtEveryLine(),
                                 BrakesOnSight(), NeverGoesAgain(), SteersAway(), NeverDrives()])
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
                return Command(0.6, 0.6)
            return super().update(packet)
    problems = check_stale_lane_slows_then_stops(DrivesOnStaleRight())
    assert any("faster than" in p for p in problems)


@pytest.mark.software
def test_an_intersection_is_the_line_coming_down_the_image_then_the_time_after():
    case = intersection({"stop_sign_detected": True}, {"drive_state": "stop"}, after_frames=3)
    assert len(case) == len(APPROACH_ROWS) + 3
    assert [c["stop_line_distance_px"] for c in case[:len(APPROACH_ROWS)]] == list(APPROACH_ROWS)
    assert all(c["stop_line_detected"] and c["stop_sign_detected"] for c in case[:len(APPROACH_ROWS)])
    assert case[-1] == {"drive_state": "stop"} and case[-1] is not case[-2]
    assert APPROACH_ROWS[-1] < APPROACH_ROWS[0]           # coming nearer



@pytest.mark.software
def test_enforce_passes_a_valid_command_through():
    for cmd in (Command(0.4, 0.4), Command(-0.5, 0.5), Command(0.0, 0.3), BRAKE):
        assert enforce(cmd) == (cmd, [])


@pytest.mark.software
@pytest.mark.parametrize("cmd", [Command(0.1, 0.4), Command(1.5, 0.4), Command(0.4, 0.4, brake=True), "go"])
def test_enforce_brakes_a_command_that_breaks_the_contract_with_its_problems(cmd):
    assert enforce(cmd) == (BRAKE, command_problems(cmd)) and command_problems(cmd)
