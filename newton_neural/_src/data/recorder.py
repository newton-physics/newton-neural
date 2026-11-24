# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Episode-driven trajectory recorder (:class:`TrajectoryRecorder`) writing sharded datasets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..core.signal import Frames, SignalSpec
from .commands import CommandBase, CommandHeld, start_commands
from .dataset import TrajectoryDataset, TrajectoryWriter
from .samplers import SamplerBase, SamplerSobol
from .simulation import SimulationBase
from .source import SignalSource

if TYPE_CHECKING:
    from ..models.nerd import NeRDBase

_STATE_LIMIT = 1.0e3
"""Max allowed magnitude [SI units] for any state coordinate before a world is treated as diverged."""

_STREAM = 0x5EED
"""Offset added to ``seed`` for command/episode RNG so it does not correlate with a same-seeded state sampler."""


class TrajectoryRecorder:
    """Drives a batched simulation through random episodes and writes sharded :class:`TrajectoryDataset` data.

    Each world resets to a sampled state, follows ``commands`` for a random length in ``episode_frames``, and resets
    again. Worlds whose state becomes non-finite or exceeds ``1000`` in SI units reset immediately with
    ``valid=False`` on the failing frame; ``settle_frame_count`` initial frames per episode are also marked
    ``valid=False``.

    Args:
        simulation: Batched simulation to step.
        nerd: Model whose :meth:`~newton_neural.models.NeRDBase.sources` declare signals to record, or an explicit
            sequence of :class:`SignalSource` objects.
        states: Initial state sampler (defaults to :class:`SamplerSobol` over simulation state bounds).
        commands: Command generator (defaults to :class:`CommandHeld` over simulation command bounds).
        episode_frames: Inclusive ``(min, max)`` valid episode length [frames]. Use ``(1, 1)`` for independent
            single-step samples (free motion) or at least ``(2, 2)`` for state-increment models.
        settle_frame_count: Leading frames per episode marked invalid to let initial penetrations resolve.
        seed: RNG seed for default state sampler, commands, and episode lengths.
    """

    def __init__(
        self,
        simulation: SimulationBase,
        nerd: NeRDBase | Sequence[SignalSource],
        *,
        states: SamplerBase | None = None,
        commands: CommandBase | None = None,
        episode_frames: tuple[int, int] = (200, 400),
        settle_frame_count: int = 0,
        seed: int = 0,
    ) -> None:
        shortest, longest = episode_frames
        if not 1 <= shortest <= longest:
            raise ValueError(
                f"episode_frames is a shortest and a longest length of 1 frame or more, got {episode_frames}"
            )
        if settle_frame_count < 0:
            raise ValueError(f"settle_frame_count cannot be negative, got {settle_frame_count}")
        self.simulation = simulation
        # Accept either a model (duck-typed via sources()) or explicit SignalSources to keep data independent of models.
        sources = nerd if isinstance(nerd, Sequence) else nerd.sources(simulation)
        self.signals: dict[str, SignalSpec] = {}
        for source in sources:
            repeated = self.signals.keys() & source.signals.keys()
            if repeated:
                raise ValueError(f"signals {sorted(repeated)} come from more than one source")
            _check_source(source, simulation.world_count)
            self.signals |= source.signals
        self._before = [source for source in sources if not source.after_step]
        self._after = [source for source in sources if source.after_step]
        self._episode_frames, self._settle_frame_count = episode_frames, settle_frame_count

        device, world_count = simulation.device, simulation.world_count
        self._generator = torch.Generator(device=device).manual_seed(seed + _STREAM)
        self.states = SamplerSobol(seed=seed) if states is None else states
        self.states.start(simulation.state_bounds())
        self.commands = CommandHeld() if commands is None else commands
        self._next_command = start_commands(self.commands, simulation, self._generator)
        self._generation = {
            "states": type(self.states).__name__,
            "commands": type(self.commands).__name__,
            "episode_frames": list(episode_frames),
            "settle_frame_count": settle_frame_count,
            "seed": seed,
            "world_count": world_count,
        }

        # Per-world frames elapsed since last reset and target valid episode length.
        self._age = torch.zeros(world_count, dtype=torch.long, device=device)
        self._length = torch.zeros(world_count, dtype=torch.long, device=device)
        self._restart(torch.ones(world_count, dtype=torch.bool, device=device))
        # Fail immediately if sampled starting states already exceed _STATE_LIMIT (would reset every frame).
        diverged = int(self._diverged().sum())
        if diverged:
            raise ValueError(
                f"{diverged} of {world_count} worlds start from a state that is not finite or exceeds "
                f"{_STATE_LIMIT:g} in the SI unit of a state signal, which counts as diverged. Sample starting "
                "states within that limit, and keep the worlds within a kilometre of the origin"
            )

    def record(self, path: str | Path, frame_count: int, *, shard_frame_count: int = 256) -> TrajectoryDataset:
        """Simulate ``frame_count`` frames per world and write a sharded dataset to ``path``.

        Subsequent calls continue from the current simulation state into a new output directory. If interrupted,
        completed shards remain readable on disk with ``generation["complete"] == False`` before re-raising.

        Args:
            path: Output directory (must not already contain a dataset).
            frame_count: Frames to record per world.
            shard_frame_count: Frames per shard per world.

        Returns:
            Opened :class:`TrajectoryDataset`.
        """
        with TrajectoryWriter(path, self.signals, self.simulation.frame_dt, generation=self._generation) as writer:
            for first in range(0, frame_count, shard_frame_count):
                steps = [self._advance() for _ in range(min(shard_frame_count, frame_count - first))]
                frames, episode_start, valid = zip(*steps, strict=True)
                signals = {name: torch.stack([frame[name] for frame in frames], dim=1) for name in self.signals}
                writer.write(signals, torch.stack(episode_start, dim=1), torch.stack(valid, dim=1), continues=first > 0)
        return TrajectoryDataset(path)

    def _advance(self) -> tuple[Frames, Tensor, Tensor]:
        """Step one frame across all worlds and return ``(signals, episode_start, valid)``."""
        simulation = self.simulation
        frame = _read(self._before)
        simulation.step(self._next_command(frame))
        frame |= _read(self._after)

        diverged = self._diverged()
        settled = self._age >= self._settle_frame_count
        episode_start = self._age == self._settle_frame_count
        self._age += 1
        self._restart((self._age >= self._settle_frame_count + self._length) | diverged)
        return frame, episode_start, settled & ~diverged

    def _diverged(self) -> Tensor:
        """Return per-world boolean mask ``[world_count]`` where any state value is non-finite or ``> _STATE_LIMIT``."""
        state = self.simulation.binding.read_state(self.simulation.state)
        # NaN fails <= comparison too, catching both overflow and non-finite states in one check.
        sound = [(value.abs() <= _STATE_LIMIT).flatten(1).all(dim=1) for value in state.values()]
        return ~torch.stack(sound).all(dim=0)

    def _restart(self, world_mask: Tensor) -> None:
        """Sample new initial states, reset commands, and draw new episode lengths for masked worlds."""
        worlds = torch.nonzero(world_mask).flatten()
        if len(worlds) == 0:
            return
        states = {}
        # A sampler draws the states of one world. The binding carries them to the worlds they start.
        for name, value in self.simulation.binding.place(self.states.sample(len(worlds)), worlds).items():
            states[name] = value.new_zeros(len(world_mask), *value.shape[1:])
            states[name][worlds] = value
        self.simulation.restart(states, world_mask)
        self.commands.reset(world_mask)
        shortest, longest = self._episode_frames
        lengths = torch.randint(
            shortest, longest + 1, world_mask.shape, device=world_mask.device, generator=self._generator
        )
        self._age = self._age.masked_fill(world_mask, 0)
        self._length = torch.where(world_mask, lengths, self._length)


def _check_source(source: SignalSource, world_count: int) -> None:
    """Call ``source.read()`` once and validate returned keys, shapes, and dtypes against ``source.signals``."""
    frames = source.read()
    if frames.keys() != source.signals.keys():
        raise ValueError(f"a source declares the signals {sorted(source.signals)} and returns {sorted(frames)}")
    for name, spec in source.signals.items():
        found, expected = frames[name], (world_count, *spec.shape)
        if found.dtype != spec.torch_dtype or found.shape != expected:
            raise ValueError(
                f"the source of {name!r} returns {found.dtype} {list(found.shape)}, and its spec declares "
                f"{spec.torch_dtype} {list(expected)}, one row per world"
            )


def _read(sources: list[SignalSource]) -> Frames:
    """Merge current signal dictionaries from all ``sources``."""
    frame: Frames = {}
    for source in sources:
        frame |= source.read()
    return frame
