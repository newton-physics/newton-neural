# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Task-agnostic command generators driving parallel worlds during recording and live audits."""

from __future__ import annotations

import abc
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..core.signal import Frames

if TYPE_CHECKING:
    from .simulation import SimulationBase

Bound = Tensor | float | None
"""Optional command bound override: ``[command_count]``, scalar broadcast to all channels, or ``None`` for default."""


class CommandBase(abc.ABC):
    """Stateful per-world command generator.

    Subclasses implement :meth:`next`, plus :meth:`allocate` and :meth:`reset` if maintaining per-world state.

    Args:
        low: Optional lower bound override (defaults to simulation command bounds).
        high: Optional upper bound override (defaults to simulation command bounds).

    Attributes:
        world_count: Number of worlds (set by :meth:`start`).
        low: Active lower bound ``[command_count]`` (set by :meth:`start`).
        high: Active upper bound ``[command_count]`` (set by :meth:`start`).
        frame_dt: Frame duration [s] (set by :meth:`start`).
        generator: Random generator on the bounds device (set by :meth:`start`).
        sizes: Flattened element count per ``CONTROL`` signal in concatenated order (set by :meth:`start`).
    """

    world_count: int
    low: Tensor
    high: Tensor
    frame_dt: float
    generator: torch.Generator | None
    sizes: dict[str, int]

    def __init__(self, *, low: Bound = None, high: Bound = None) -> None:
        self._bounds = (low, high)

    def start(
        self,
        world_count: int,
        low: Tensor,
        high: Tensor,
        frame_dt: float,
        generator: torch.Generator | None = None,
        *,
        sizes: dict[str, int],
    ) -> None:
        """Bind simulation metadata and default bounds, then call :meth:`allocate`."""
        self.low, self.high = _bound(self._bounds[0], low), _bound(self._bounds[1], high)
        self.world_count, self.frame_dt, self.generator, self.sizes = world_count, frame_dt, generator, sizes
        self.allocate()

    def allocate(self) -> None:  # noqa: B027
        """Allocate per-world internal state and validate ``sizes`` (called by :meth:`start`)."""

    @property
    def bounded(self) -> bool:
        """``True`` if all channels have finite lower and upper bounds."""
        return bool(self.low.isfinite().all() and self.high.isfinite().all())

    def reset(self, world_mask: Tensor) -> None:  # noqa: B027
        """Reset internal generator state for worlds where ``world_mask`` ``[world_count]`` (bool) is ``True``."""

    @abc.abstractmethod
    def next(self, frames: Frames | None = None) -> Tensor:
        """Return commands ``[world_count, command_count]`` within ``[low, high]`` for the current frame."""

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, device=self.low.device, generator=self.generator)

    def _randn(self, *shape: int) -> Tensor:
        return torch.randn(*shape, device=self.low.device, generator=self.generator)

    def _scale(self, unit: Tensor) -> Tensor:
        """Affine-map ``unit`` in ``[-1, 1]`` into ``[low, high]``."""
        scaled = (self.high + self.low) / 2 + (self.high - self.low) / 2 * unit
        return scaled.clamp(self.low, self.high)  # Guard against 1-ULP overshoot past bounds


def _bound(given: Bound, default: Tensor) -> Tensor:
    if given is None:
        return default
    if not isinstance(given, Tensor):
        return torch.full_like(default, float(given))
    if given.shape != default.shape:
        raise ValueError(f"a bound must be {tuple(default.shape)} as the command, got {tuple(given.shape)}")
    return given.to(default)


class CommandHeld(CommandBase):
    """Piecewise-constant uniform random commands held for a random number of frames.

    Resets end the current hold immediately so each episode begins with a fresh target.

    Args:
        hold_frames: Inclusive ``(min, max)`` hold duration [frames].
        low: Optional lower bound override.
        high: Optional upper bound override.
    """

    def __init__(self, hold_frames: tuple[int, int] = (1, 25), *, low: Bound = None, high: Bound = None) -> None:
        super().__init__(low=low, high=high)
        self.hold_frames = hold_frames

    def allocate(self) -> None:
        self._value = self.low.new_zeros(self.world_count, *self.low.shape)
        self._remaining = torch.zeros(self.world_count, dtype=torch.long, device=self.low.device)

    def reset(self, world_mask: Tensor) -> None:
        self._remaining.masked_fill_(world_mask, 0)

    def next(self, frames: Frames | None = None) -> Tensor:
        renew = self._remaining <= 0
        fresh = self.low + (self.high - self.low) * self._rand(*self._value.shape)
        self._value = torch.where(renew[:, None], fresh, self._value)
        hold = torch.randint(
            self.hold_frames[0],
            self.hold_frames[1] + 1,
            renew.shape,
            device=renew.device,
            generator=self.generator,
        )
        self._remaining = torch.where(renew, hold, self._remaining) - 1
        return self._value


class CommandNearState(CommandBase):
    """Resamples ``joint_target_q`` every frame uniformly within ``+-reach`` of current ``joint_q``.

    Requires ``sizes == {"joint_target_q": ...}``. Suited to single-frame free-motion datasets where targets should
    remain close to sampled joint configurations.

    Args:
        reach: Max offset from ``joint_q`` [m or rad], scalar or ``[actuated_count]``.
        low: Optional lower bound override.
        high: Optional upper bound override.
    """

    def __init__(self, reach: Tensor | float, *, low: Bound = None, high: Bound = None) -> None:
        super().__init__(low=low, high=high)
        self.reach = reach

    def allocate(self) -> None:
        if list(self.sizes) != ["joint_target_q"]:
            raise ValueError(
                f"CommandNearState draws joint position targets, and this simulation is driven by "
                f"{list(self.sizes)}. Use it for a system whose only control is joint_target_q, or another command"
            )
        self._reach = torch.as_tensor(self.reach, dtype=self.low.dtype, device=self.low.device)

    def next(self, frames: Frames | None = None) -> Tensor:
        # Actuated joint coordinates occupy the trailing slice of joint_q (after floating-base coords, if any).
        joints = frames["joint_q"][:, -self.low.shape[0] :]
        offset = self._reach * (2.0 * self._rand(*joints.shape) - 1.0)
        return (joints + offset).clamp(self.low, self.high)


class CommandSmooth(CommandBase):
    """Temporally correlated bounded noise (unit-variance Ornstein-Uhlenbeck process squashed by ``tanh``).

    Args:
        timescale: Autocorrelation decay constant [s].
        low: Optional lower bound override.
        high: Optional upper bound override.
    """

    def __init__(self, timescale: float = 0.2, *, low: Bound = None, high: Bound = None) -> None:
        super().__init__(low=low, high=high)
        self.timescale = timescale

    def allocate(self) -> None:
        self._decay = math.exp(-self.frame_dt / self.timescale)
        self._z = self._randn(self.world_count, *self.low.shape)

    def reset(self, world_mask: Tensor) -> None:
        self._z = torch.where(world_mask[:, None], self._randn(*self._z.shape), self._z)

    def next(self, frames: Frames | None = None) -> Tensor:
        self._z = self._decay * self._z + math.sqrt(1 - self._decay**2) * self._randn(*self._z.shape)
        return self._scale(torch.tanh(self._z))


class CommandSweep(CommandBase):
    """Repeating linear frequency chirp with randomized per-channel phase and amplitude on reset.

    Args:
        frequency_low: Starting sweep frequency [Hz].
        frequency_high: Ending sweep frequency [Hz].
        duration: Sweep period [s].
        low: Optional lower bound override.
        high: Optional upper bound override.
    """

    def __init__(
        self,
        frequency_low: float = 0.2,
        frequency_high: float = 5.0,
        duration: float = 4.0,
        *,
        low: Bound = None,
        high: Bound = None,
    ) -> None:
        super().__init__(low=low, high=high)
        self.frequency_low, self.frequency_high, self.duration = frequency_low, frequency_high, duration

    def allocate(self) -> None:
        self._time = torch.zeros(self.world_count, device=self.low.device)
        self._phase = self.low.new_zeros(self.world_count, *self.low.shape)
        self._amplitude = self.low.new_ones(self.world_count, *self.low.shape)

    def reset(self, world_mask: Tensor) -> None:
        self._time = self._time.masked_fill(world_mask, 0.0)
        self._phase = torch.where(world_mask[:, None], 2 * math.pi * self._rand(*self._phase.shape), self._phase)
        self._amplitude = torch.where(world_mask[:, None], self._rand(*self._phase.shape), self._amplitude)

    def next(self, frames: Frames | None = None) -> Tensor:
        t = self._time.remainder(self.duration)
        rate = (self.frequency_high - self.frequency_low) / self.duration
        angle = 2 * math.pi * (self.frequency_low * t + 0.5 * rate * t.square())
        self._time = self._time + self.frame_dt
        return self._scale(self._amplitude * torch.sin(angle[:, None] + self._phase))


class CommandSwitch(CommandBase):
    """Randomly selects one of several child command generators per world at each episode reset.

    Args:
        commands: Mapping from child :class:`CommandBase` to positive selection weight.
        low: Default lower bound passed to child commands without their own override.
        high: Default upper bound passed to child commands without their own override.
    """

    def __init__(self, commands: dict[CommandBase, float], *, low: Bound = None, high: Bound = None) -> None:
        super().__init__(low=low, high=high)
        self.commands = list(commands)
        self.weights = torch.tensor(list(commands.values()), dtype=torch.float32)

    def allocate(self) -> None:
        for command in self.commands:
            command.start(self.world_count, self.low, self.high, self.frame_dt, self.generator, sizes=self.sizes)
        self.weights = self.weights.to(self.low.device)
        self._choice = torch.zeros(self.world_count, dtype=torch.long, device=self.low.device)

    @property
    def bounded(self) -> bool:
        """``True`` if every child command has finite bounds."""
        return all(command.bounded for command in self.commands)

    def reset(self, world_mask: Tensor) -> None:
        fresh = torch.multinomial(self.weights, self.world_count, replacement=True, generator=self.generator)
        self._choice = torch.where(world_mask, fresh, self._choice)
        for command in self.commands:
            command.reset(world_mask)

    def next(self, frames: Frames | None = None) -> Tensor:
        values = torch.stack([command.next(frames) for command in self.commands])  # [commands, worlds, channels]
        index = self._choice[None, :, None].expand(1, *values.shape[1:])
        return values.gather(0, index)[0]


def start_commands(
    commands: CommandBase, simulation: SimulationBase, generator: torch.Generator | None = None
) -> Callable[[Frames | None], Frames]:
    """Initialize ``commands`` against ``simulation.command_bounds()`` and return a frame-to-control callable.

    Flattens all ``CONTROL`` signals into one vector for ``commands`` and splits outputs back into a
    :type:`Frames` dict (returns ``{}`` for unactuated systems).

    Raises:
        ValueError: If any control channel lacks finite bounds or ``commands`` cannot drive ``simulation``.
    """
    bounds = simulation.command_bounds()
    shapes = {name: low.shape for name, (low, _) in bounds.items()}
    sizes = {name: shape.numel() for name, shape in shapes.items()}
    # Seed empty tensor so concatenation works for unactuated systems (0 control channels).
    empty = torch.zeros(0, device=simulation.device)
    low = torch.cat([empty, *(low.flatten() for low, _ in bounds.values())])
    high = torch.cat([empty, *(high.flatten() for _, high in bounds.values())])
    commands.start(simulation.world_count, low, high, simulation.frame_dt, generator, sizes=sizes)
    if not commands.bounded:
        raise ValueError(
            f"the simulation has no finite bounds for all of {list(bounds)}. Give the command its own low and high"
        )

    def next_command(frames: Frames | None = None) -> Frames:
        parts = commands.next(frames).split(list(sizes.values()), dim=1)
        return {name: part.unflatten(1, shapes[name]) for name, part in zip(shapes, parts, strict=True)}

    return next_command
