# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-resident episode window loader (:class:`TrajectoryWindows`) and batch type (:class:`TrajectoryBatch`)."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..core.signal import Frames
from .dataset import TrajectoryDataset, array_layout


@dataclass
class TrajectoryBatch:
    """Batch of contiguous single-episode trajectory windows."""

    frames: Frames
    """All recorded signals, each shaped ``[batch_size, length + 1, *shape]``."""

    episode_start: Tensor
    """Boolean flag ``[batch_size]`` indicating whether each window starts at frame 0 of an episode."""

    frame_dt: float
    """Frame duration [s]."""

    def to(self, device: torch.device | str) -> TrajectoryBatch:
        """Return a copy of this batch with all tensors moved to ``device``."""
        frames = {name: value.to(device) for name, value in self.frames.items()}
        return TrajectoryBatch(frames, self.episode_start.to(device), self.frame_dt)


class TrajectoryWindows:
    """Device-resident index of valid single-episode windows of ``length`` steps (``length + 1`` frames).

    Keeps loaded worlds on ``device`` and gathers batches by direct tensor indexing with no worker processes.

    Args:
        dataset: Dataset directory path or :class:`TrajectoryDataset`.
        length: Steps per window (``length + 1`` frames). Use ``>= model.receptive_field`` for state-increment
            models, or ``0`` for single-frame recorded-label models (e.g., free motion).
        device: Device storing the loaded tensors and gathered batches (defaults to CPU).
        worlds: Optional slice over global world IDs (e.g., ``slice(rank, None, rank_count)`` for one process of
            several that train through ``DistributedDataParallel``).
    """

    def __init__(
        self,
        dataset: str | Path | TrajectoryDataset,
        length: int,
        *,
        device: torch.device | str | None = None,
        worlds: slice | None = None,
    ) -> None:
        self.dataset = dataset if isinstance(dataset, TrajectoryDataset) else TrajectoryDataset(dataset)
        self.device = torch.device("cpu" if device is None else device)
        self.frame_dt = self.dataset.frame_dt
        shards = self.dataset.shards
        # Group contiguous shard chains into separate recordings.
        recordings: list[list[int]] = []
        for i, shard in enumerate(shards):
            if not shard["continues"]:
                recordings.append([])
            recordings[-1].append(i)
        frame_count = self.dataset.frame_count
        # Sort selected global IDs so rows stay ordered by recording regardless of slice step sign.
        world_ids = np.sort(np.arange(self.dataset.world_count)[slice(None) if worlds is None else worlds])
        # Preallocate destination tensors once and populate shard by shard to avoid doubling peak memory.
        data = {
            name: torch.empty(len(world_ids), frame_count, *shape, dtype=dtype, device=self.device)
            for name, (shape, dtype) in array_layout(self.dataset.signals).items()
        }
        row = first_id = 0
        for recording in recordings:
            world_count = shards[recording[0]]["world_count"]
            local = world_ids[(world_ids >= first_id) & (world_ids < first_id + world_count)] - first_id
            frame = 0
            for i in recording:
                arrays = self.dataset.arrays(i)
                stop = frame + shards[i]["frame_count"]
                for name, tensor in data.items():
                    tensor[row : row + len(local), frame:stop] = torch.from_numpy(arrays[name][local])
                frame = stop
            row += len(local)
            first_id += world_count
        self._valid = data.pop("valid")
        self._episode_start = data.pop("episode_start")
        self._frames = data
        self._world_ids = torch.from_numpy(world_ids).to(self.device)
        self._select(length, torch.ones(len(world_ids), dtype=torch.bool, device=self.device))

    def _select(self, length: int, keep: Tensor) -> None:
        """Build the ``(world, start_frame)`` index table for ``length`` steps across worlds where ``keep`` is true."""
        if length < 0:
            raise ValueError(f"a window cannot hold fewer than zero steps, got length {length}")
        self.length = length
        self._keep = keep
        self._index = _window_starts(self._episode_start, self._valid & keep[:, None], length)
        self._offsets = torch.arange(length + 1, device=self.device)

    def _view(self, length: int, keep: Tensor) -> TrajectoryWindows:
        other = copy.copy(self)  # Shares underlying signal tensors
        other._select(length, keep)
        return other

    def __len__(self) -> int:
        """Total number of valid windows."""
        return self._index.shape[0]

    @property
    def world_ids(self) -> Tensor:
        """Ascending global world IDs ``[world_count]`` int64 included in this view."""
        return self._world_ids[self._keep]

    @property
    def episode_start(self) -> Tensor:
        """Boolean mask ``[len(self)]`` marking windows that begin at an episode start."""
        return self._episode_start[self._index[:, 0], self._index[:, 1]]

    def with_length(self, length: int) -> TrajectoryWindows:
        """Return a zero-copy view re-indexed for a different window ``length`` over the same worlds."""
        return self._view(length, self._keep)

    def split(self, fraction: float = 0.05, seed: int = 0) -> tuple[TrajectoryWindows, TrajectoryWindows]:
        """Split deterministically by global world ID into ``(train, held_out)`` views sharing the same tensors.

        Uses :meth:`TrajectoryDataset.held_out_worlds` so all distributed ranks and evaluation scripts agree on the
        exact split (a local slice may contain zero held-out worlds).
        """
        held_out = self.dataset.held_out_worlds(fraction, seed).to(self.device)[self._world_ids]
        return self._view(self.length, self._keep & ~held_out), self._view(self.length, self._keep & held_out)

    def batch(self, indices: Tensor) -> TrajectoryBatch:
        """Gather a :class:`TrajectoryBatch` at window ``indices`` ``[batch_size]`` int64."""
        rows = self._index[torch.as_tensor(indices, device=self.device)]
        world, frame = rows[:, :1], rows[:, 1:] + self._offsets
        frames = {name: value[world, frame] for name, value in self._frames.items()}
        return TrajectoryBatch(frames, self._episode_start[rows[:, 0], rows[:, 1]], self.frame_dt)

    def loader(
        self,
        batch_size: int,
        *,
        epochs: int | None = 1,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        weights: Tensor | None = None,
    ) -> Iterator[TrajectoryBatch]:
        """Yield ``TrajectoryBatch`` objects of fixed ``batch_size`` (drops trailing partial batch per epoch).

        Args:
            batch_size: Windows per batch.
            epochs: Number of epochs, or ``None`` for an infinite stream.
            shuffle: Shuffle window order each epoch (must be ``True`` if ``weights`` is provided).
            generator: Optional random generator on :attr:`device`.
            weights: Optional non-negative sampling weights ``[len(self)]`` (samples with replacement via CDF
                inversion).
        """
        count = len(self)
        if count == 0:
            world_count = int(self._keep.sum())
            if world_count == 0:
                raise ValueError("these windows hold no world, so there is nothing to draw from")
            raise ValueError(
                f"there is no window of {self.length} steps, since none of the {world_count} worlds holds an "
                f"episode of {self.length + 1} valid frames. Use a shorter length, or record longer episodes "
                "with episode_frames of the recorder"
            )
        if not 0 < batch_size <= count:
            raise ValueError(f"batch_size must lie between 1 and the number of windows {count}, got {batch_size}")
        cumulative = None
        if weights is not None:
            if not shuffle:
                raise ValueError("weighted batches are random draws and cannot follow the order of the windows")
            if weights.shape != (count,):
                raise ValueError(f"weights must be [{count}], one per window, got {tuple(weights.shape)}")
            if bool((weights < 0).any()) or not bool(weights.sum() > 0):
                raise ValueError("weights must be non-negative with a positive sum")
            cumulative = weights.to(self.device, torch.float64).cumsum(0)
        batch_count = count // batch_size

        def batches() -> Iterator[TrajectoryBatch]:
            for _ in itertools.count() if epochs is None else range(epochs):
                if cumulative is not None:
                    # CDF inversion avoids torch.multinomial's 2**24 category limit.
                    draw = torch.rand(count, dtype=torch.float64, device=self.device, generator=generator)
                    order = torch.searchsorted(cumulative, draw * cumulative[-1], right=True).clamp_(max=count - 1)
                elif shuffle:
                    order = torch.randperm(count, device=self.device, generator=generator)
                else:
                    order = torch.arange(count, device=self.device)
                for indices in order[: batch_count * batch_size].view(batch_count, batch_size):
                    yield self.batch(indices)

        return batches()  # Return generator after eager argument validation above

    def measure(self, fn: Callable[[TrajectoryBatch], Tensor], batch_size: int = 4096) -> Tensor:
        """Apply ``fn(batch) -> [batch_size, ...]`` sequentially across all windows and concatenate results."""
        count = len(self)
        values = [
            fn(self.batch(torch.arange(i, min(i + batch_size, count), device=self.device)))
            for i in range(0, count, batch_size)
        ]
        return torch.cat(values) if values else torch.zeros(0, device=self.device)


def _window_starts(episode_start: Tensor, valid: Tensor, length: int) -> Tensor:
    """Return ``[window_count, 2]`` ``(world, start_frame)`` pairs for every valid single-episode window."""
    world_count, frame_count = valid.shape
    if frame_count <= length:
        return torch.zeros(0, 2, dtype=torch.long, device=valid.device)
    # Prefix sums check in O(1) per start that [t, t + length] has no invalid frames and no interior episode starts.
    pad = torch.zeros(world_count, 1, dtype=torch.int32, device=valid.device)
    invalid = torch.cat((pad, (~valid).cumsum(1, dtype=torch.int32)), dim=1)
    starts = torch.cat((pad, episode_start.cumsum(1, dtype=torch.int32)), dim=1)
    first_count = frame_count - length
    clean = invalid[:, length + 1 :] == invalid[:, :first_count]
    one_episode = starts[:, length + 1 :] == starts[:, 1 : first_count + 1]
    return torch.nonzero(clean & one_episode)
