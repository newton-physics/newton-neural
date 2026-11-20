# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for recording command generators: bounds enforcement, per-world resets, and simulation binding."""

from __future__ import annotations

import math
import unittest

import torch
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.data.commands import (
    CommandBase,
    CommandHeld,
    CommandNearState,
    CommandSmooth,
    CommandSweep,
    CommandSwitch,
    start_commands,
)
from newton_neural._src.data.simulation import SimulationBase
from newton_neural.tests.unittest_utils import add_function_test

FRAME_DT = 1.0 / 64.0


class TestCommands(unittest.TestCase):
    pass


def _start(
    command: CommandBase,
    device: torch.device,
    *,
    world_count: int = 64,
    seed: int = 0,
    sizes: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Start a command of three channels on the device and return its bounds ``[3]``.

    Args:
        command: The command to start.
        device: Device of the bounds and of the random generator.
        world_count: Number of worlds.
        seed: Seed of the random generator.
        sizes: The CONTROL signals the three channels belong to. One signal of joint position targets by default.
    """
    low = torch.tensor([-1.0, 0.0, 2.0], device=device)
    high = torch.tensor([1.0, 0.5, 6.0], device=device)
    generator = torch.Generator(device).manual_seed(seed)
    command.start(world_count, low, high, FRAME_DT, generator, sizes=sizes or {"joint_target_q": 3})
    return low, high


class _CommandRamp(CommandBase):
    """A command with state of its own, which rises by one per frame from the lower bound of each world."""

    def allocate(self) -> None:
        self.allocated_for = (self.world_count, dict(self.sizes), self.frame_dt, self.generator)
        self._frame = torch.zeros(self.world_count, 1, device=self.low.device)

    def next(self, frames: Frames | None = None) -> Tensor:
        self._frame += 1.0
        return self.low + self._frame


def test_start_allocates(test: TestCommands, device: torch.device) -> None:
    """Give a command what it is started for as attributes, then let it create its state in ``allocate``."""
    command = _CommandRamp(high=9.0)
    generator = torch.Generator(device).manual_seed(0)
    low = torch.tensor([-1.0, 0.0, 2.0], device=device)
    command.start(5, low, low + 1.0, FRAME_DT, generator, sizes={"joint_target_q": 2, "joint_f": 1})
    test.assertEqual(command.allocated_for, (5, {"joint_target_q": 2, "joint_f": 1}, FRAME_DT, generator))
    test.assertIs(command.generator, generator)
    test.assertTrue(torch.equal(command.low, low))
    test.assertTrue(torch.equal(command.high, torch.full((3,), 9.0, device=device)))
    test.assertTrue(command.bounded)
    command.next()
    test.assertTrue(torch.equal(command.next(), (low + 2.0).expand(5, 3)))
    # A second start allocates anew, and a command without state needs no allocate of its own.
    command.start(2, low, low + 1.0, FRAME_DT, sizes={"joint_f": 3})
    test.assertTrue(torch.equal(command.next(), (low + 1.0).expand(2, 3)))

    class CommandLowest(CommandBase):
        def next(self, frames: Frames | None = None) -> Tensor:
            return self.low.expand(self.world_count, -1)

    plain = CommandLowest()
    plain.start(2, low, low + float("inf"), FRAME_DT, sizes={"joint_f": 3})
    test.assertEqual((plain.world_count, plain.sizes, plain.bounded), (2, {"joint_f": 3}, False))
    test.assertTrue(torch.equal(plain.next(), low.expand(2, 3)))
    # A command without a next is refused where it is built.
    with test.assertRaisesRegex(TypeError, "abstract class CommandBase .*next"):
        CommandBase()
    with test.assertRaisesRegex(TypeError, "sizes"):
        command.start(2, low, low + 1.0, FRAME_DT)


def test_commands_stay_within_bounds(test: TestCommands, device: torch.device) -> None:
    """Keep every command between its bounds through episodes and resets, with given bounds taking precedence."""
    narrow_low = torch.tensor([-0.5, 0.1, 3.0], device=device)
    narrow_high = torch.tensor([0.5, 0.2, 4.0], device=device)
    cases = [
        (CommandHeld(), None),
        (CommandSmooth(timescale=0.05), None),
        (CommandSweep(frequency_high=20.0, duration=0.5), None),
        (CommandSwitch({CommandHeld(): 1.0, CommandSmooth(): 2.0, CommandSweep(): 1.0}), None),
        (CommandHeld(low=0.1, high=0.4), (torch.full((3,), 0.1), torch.full((3,), 0.4))),
        (CommandSmooth(low=narrow_low, high=narrow_high), (narrow_low, narrow_high)),
        (CommandSweep(low=narrow_low.cpu(), high=4.5), (narrow_low, torch.full((3,), 4.5))),
    ]
    resets = torch.Generator(device).manual_seed(1)
    for command, bounds in cases:
        low, high = _start(command, device)
        if bounds is not None:
            low, high = bounds[0].to(device), bounds[1].to(device)
        command.reset(world_mask=torch.ones(64, dtype=torch.bool, device=device))
        values = []
        for frame in range(120):
            if frame % 20 == 19:
                command.reset(torch.rand(64, device=device, generator=resets) < 0.3)
            values.append(command.next())
        values = torch.stack(values)
        name = type(command).__name__
        test.assertEqual(values.shape, (120, 64, 3), name)
        test.assertEqual(values.dtype, torch.float32, name)
        test.assertEqual(values.device, device, name)
        test.assertTrue(((values >= low) & (values <= high)).all(), name)
        # The command uses its range, in every channel.
        span = values.amax(dim=(0, 1)) - values.amin(dim=(0, 1))
        test.assertTrue((span > 0.5 * (high - low)).all(), name)
    with test.assertRaisesRegex(ValueError, "bound"):
        _start(CommandHeld(low=torch.zeros(2)), device)


def test_command_held(test: TestCommands, device: torch.device) -> None:
    """Hold each value for the set number of frames, and draw a new one at once in reset worlds."""
    command = CommandHeld(hold_frames=(3, 3))
    _start(command, device)
    values = [command.next() for _ in range(4)]
    test.assertTrue(torch.equal(values[0], values[1]) and torch.equal(values[1], values[2]))
    test.assertTrue((values[3] != values[2]).all())
    mask = torch.arange(64, device=device) < 16
    command.reset(mask)
    after = command.next()
    test.assertTrue((after[mask] != values[3][mask]).all())
    test.assertTrue(torch.equal(after[~mask], values[3][~mask]))
    # The reset worlds now hold for three frames from the reset, the others end their hold one frame earlier.
    following = [command.next(), command.next()]
    test.assertTrue(torch.equal(following[0], after))
    test.assertTrue(torch.equal(following[1][mask], after[mask]))
    test.assertTrue((following[1][~mask] != after[~mask]).all())

    lengths = CommandHeld(hold_frames=(2, 5))
    _start(lengths, device, world_count=256)
    values = torch.stack([lengths.next() for _ in range(200)])
    change = torch.nonzero((values[1:] != values[:-1]).any(dim=-1)[:, 0]).flatten()
    holds = change[1:] - change[:-1]
    test.assertEqual(holds.unique().tolist(), [2, 3, 4, 5])


def test_command_smooth(test: TestCommands, device: torch.device) -> None:
    """Change slowly for a long timescale and fast for a short one, and jump only in reset worlds."""
    slow, fast = CommandSmooth(timescale=10.0), CommandSmooth(timescale=0.01)
    steps = []
    for command in (slow, fast):
        _start(command, device)
        values = torch.stack([command.next() for _ in range(50)])
        steps.append((values[1:] - values[:-1]).abs().mean().item())
    test.assertLess(steps[0], 0.2 * steps[1])

    before = slow.next()
    mask = torch.arange(64, device=device) < 32
    slow.reset(mask)
    change = (slow.next() - before).abs().mean(dim=-1)
    test.assertGreater(change[mask].mean().item(), 4.0 * change[~mask].mean().item())


def test_command_sweep(test: TestCommands, device: torch.device) -> None:
    """Follow a sine of rising frequency that repeats, and restart it with a new phase and amplitude at a reset."""
    period = 32  # frames in one sweep of half a second

    def sweep(seed: int = 0) -> tuple[CommandSweep, Tensor, Tensor]:
        command = CommandSweep(frequency_low=1.0, frequency_high=5.0, duration=period * FRAME_DT)
        return command, *_start(command, device, world_count=4, seed=seed)

    command, low, high = sweep()
    values = torch.stack([command.next() for _ in range(2 * period)])
    t = torch.arange(period, device=device) * FRAME_DT
    angle = 2 * math.pi * (1.0 * t + 0.5 * (5.0 - 1.0) / (period * FRAME_DT) * t.square())
    expected = (high + low) / 2 + (high - low) / 2 * torch.sin(angle)[:, None, None]
    test.assertTrue(torch.allclose(values[:period], expected.expand(period, 4, 3), atol=1e-5))
    test.assertTrue(torch.allclose(values[period:], values[:period], atol=1e-5))

    # A sweep that is reset after a few frames equals one that is reset at its start, with the same random draws.
    everywhere = torch.ones(4, dtype=torch.bool, device=device)
    late, _, _ = sweep()
    for _ in range(5):
        late.next()
    late.reset(everywhere)
    early, _, _ = sweep()
    early.reset(everywhere)
    restarted = torch.stack([late.next() for _ in range(period)])
    test.assertTrue(torch.equal(restarted, torch.stack([early.next() for _ in range(period)])))
    test.assertFalse(torch.allclose(restarted, values[:period], atol=1e-3))
    test.assertTrue(((restarted - (high + low) / 2).abs() <= (high - low) / 2).all())

    # A reset of one world leaves the others on their sweep.
    command.reset(torch.arange(4, device=device) == 0)
    after = torch.stack([command.next() for _ in range(period)])
    test.assertTrue(torch.allclose(after[:, 1:], values[:period, 1:], atol=1e-5))
    test.assertFalse(torch.allclose(after[:, 0], values[:period, 0], atol=1e-3))


def test_command_switch(test: TestCommands, device: torch.device) -> None:
    """Choose one command per episode by weight, and keep the choice of worlds that are not reset."""
    world_count = 4096
    up, down = CommandHeld(low=1.0, high=1.0), CommandHeld(low=-1.0, high=-1.0)
    command = CommandSwitch({up: 3.0, down: 1.0})
    _start(command, device, world_count=world_count)
    test.assertTrue((command.next() == 1.0).all())
    command.reset(torch.ones(world_count, dtype=torch.bool, device=device))
    first = command.next()
    test.assertTrue((first == first[:, :1]).all())
    test.assertAlmostEqual((first[:, 0] == 1.0).float().mean().item(), 0.75, delta=0.04)
    test.assertTrue(torch.equal(command.next(), first))

    mask = torch.arange(world_count, device=device) % 2 == 0
    command.reset(mask)
    second = command.next()
    test.assertTrue(torch.equal(second[~mask], first[~mask]))
    test.assertFalse(torch.equal(second[mask], first[mask]))
    test.assertAlmostEqual((second[mask][:, 0] == 1.0).float().mean().item(), 0.75, delta=0.05)

    only = CommandSwitch({up: 0.0, down: 1.0})
    _start(only, device)
    only.reset(torch.ones(64, dtype=torch.bool, device=device))
    test.assertTrue((only.next() == -1.0).all())

    # The bounds of a switch are those of its commands that bring none, in place of the simulation's.
    inherited, own = CommandHeld(), CommandHeld(low=5.0, high=5.5)
    shared = CommandSwitch({inherited: 1.0, own: 1.0}, low=3.0, high=3.5)
    _start(shared, device, world_count=world_count)
    shared.reset(torch.ones(world_count, dtype=torch.bool, device=device))
    values = shared.next()
    follows_own = values[:, 0] >= 5.0
    test.assertTrue(((values >= 3.0) & (values <= 3.5))[~follows_own].all())
    test.assertTrue(((values >= 5.0) & (values <= 5.5))[follows_own].all())
    test.assertTrue(follows_own.any() and not follows_own.all())
    test.assertTrue(torch.equal(inherited.low, torch.full((3,), 3.0, device=device)))


def test_command_near_state(test: TestCommands, device: torch.device) -> None:
    """Draw targets within the reach of the actuated joint coordinates, inside the bounds, anew every frame."""
    reach = torch.tensor([0.1, 0.2, 0.3], device=device)
    command = CommandNearState(reach)
    low, high = _start(command, device, world_count=4096)
    # a floating base's seven coordinates come before the three actuated ones
    joint_q = torch.cat(
        (torch.randn(4096, 7, device=device), torch.tensor([[0.0, 0.25, 4.0]], device=device).expand(4096, 3)), dim=1
    )
    first, second = command.next({"joint_q": joint_q}), command.next({"joint_q": joint_q})
    offset = first - joint_q[:, 7:]
    test.assertTrue((offset.abs() <= reach + 1e-6).all())
    test.assertTrue(((first >= low) & (first <= high)).all())
    # the first and the last joint stay inside the bounds, so their offsets fill the reach on both sides
    test.assertGreater(offset[:, 0].max().item(), 0.09)
    test.assertLess(offset[:, 2].min().item(), -0.27)
    test.assertFalse(torch.equal(first, second))

    # Targets near the joint coordinates are no joint forces, alone or beside the targets.
    for sizes in ({"joint_f": 3}, {"joint_target_q": 2, "joint_f": 1}, {"joint_target_qd": 3}):
        with test.assertRaisesRegex(ValueError, rf"driven by \['{next(iter(sizes))}'.*only control is joint_target_q"):
            _start(CommandNearState(reach), device, sizes=sizes)
    with test.assertRaisesRegex(ValueError, "only control is joint_target_q"):
        _start(CommandSwitch({CommandHeld(): 1.0, CommandNearState(reach): 1.0}), device, sizes={"joint_f": 3})


class _SimulationOfBounds(SimulationBase):
    """A simulation that holds command bounds and nothing to step, which is all that starting a command reads."""

    def __init__(self, bounds: dict[str, tuple[Tensor, Tensor]], device: torch.device) -> None:
        self.world_count, self.frame_dt, self.device = 4, FRAME_DT, device
        self._bounds = bounds

    def step(self, command: Frames | None = None) -> None:
        pass

    def restart(self, frames: Frames, world_mask: Tensor | None = None) -> None:
        pass

    def state_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        return {}

    def command_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        return self._bounds


def test_start_commands(test: TestCommands, device: torch.device) -> None:
    """Start commands for a simulation with two CONTROL signals and for one without any."""
    one = torch.ones(2, 3, device=device)
    driven = _SimulationOfBounds({"force": (-one, one), "target": (2.0 * one[0], 3.0 * one[0])}, device)
    held = CommandHeld()
    command = start_commands(held, driven)()
    test.assertEqual(held.sizes, {"force": 6, "target": 3})
    test.assertEqual(list(held.sizes), ["force", "target"])  # the order of the command vector
    shapes = {name: tuple(value.shape) for name, value in command.items()}
    test.assertEqual(shapes, {"force": (4, 2, 3), "target": (4, 3)})
    test.assertTrue((command["force"].abs() <= 1.0).all())
    test.assertTrue(((command["target"] >= 2.0) & (command["target"] <= 3.0)).all())

    # A command says itself whether it is bounded. A switch is when each of its commands is, by its own
    # bounds or by those the switch hands on, whatever the simulation knows.
    endless = _SimulationOfBounds({"force": (-one / 0.0, one / 0.0)}, device)
    with test.assertRaisesRegex(ValueError, r"no finite bounds for all of \['force'\]"):
        start_commands(CommandHeld(), endless)
    with test.assertRaisesRegex(ValueError, "no finite bounds"):
        start_commands(CommandSwitch({CommandHeld(low=-1.0, high=1.0): 1.0, CommandSmooth(): 1.0}), endless)
    with test.assertRaisesRegex(ValueError, "no finite bounds"):
        start_commands(CommandSwitch({CommandHeld(): 1.0}, low=-1.0), endless)
    switches = (
        CommandSwitch({CommandHeld(low=-1.0, high=1.0): 1.0, CommandSmooth(low=-2.0, high=2.0): 1.0}),
        CommandSwitch({CommandHeld(): 1.0, CommandSmooth(low=-2.0, high=2.0): 1.0}, low=-1.0, high=1.0),
    )
    for switch in switches:
        next_command = start_commands(switch, endless)
        switch.reset(torch.ones(4, dtype=torch.bool, device=device))
        forces = torch.stack([next_command()["force"] for _ in range(50)])
        test.assertEqual(forces.shape, (50, 4, 2, 3))
        test.assertLessEqual(forces.abs().max().item(), 2.0)

    # Without CONTROL signals a command has no channels. It starts and resets, and every frame it is empty.
    passive = _SimulationOfBounds({}, device)
    members = {CommandHeld(): 1.0, CommandSmooth(): 2.0}
    for commands in (CommandHeld(), CommandSmooth(), CommandSweep(), CommandSwitch(members)):
        next_command = start_commands(commands, passive)
        commands.reset(torch.ones(4, dtype=torch.bool, device=device))
        test.assertEqual([next_command() for _ in range(3)], [{}, {}, {}], type(commands).__name__)
        test.assertEqual(commands.low.device.type, device.type)


add_function_test(TestCommands, "test_start_allocates", test_start_allocates)
add_function_test(TestCommands, "test_commands_stay_within_bounds", test_commands_stay_within_bounds)
add_function_test(TestCommands, "test_start_commands", test_start_commands)
add_function_test(TestCommands, "test_command_near_state", test_command_near_state)
add_function_test(TestCommands, "test_command_held", test_command_held)
add_function_test(TestCommands, "test_command_smooth", test_command_smooth)
add_function_test(TestCommands, "test_command_sweep", test_command_sweep)
add_function_test(TestCommands, "test_command_switch", test_command_switch)


if __name__ == "__main__":
    unittest.main()
