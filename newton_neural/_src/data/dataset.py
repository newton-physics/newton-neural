# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sharded memory-mapped datasets (:class:`TrajectoryDataset`) and async writer (:class:`TrajectoryWriter`).

A dataset directory contains ``meta.json`` (format version, ``frame_dt``, signal specs, generation metadata, and
shard index) plus ``shard-XXXXX/`` subdirectories. Each shard holds one ``.npy`` file per signal shaped
``[world_count, frame_count, *shape]`` alongside boolean ``episode_start`` and ``valid`` masks
``[world_count, frame_count]``. Shards either continue the preceding shard's worlds (``continues=True``) or begin a
new batch of global world IDs.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

from ..core.signal import Frames, SignalSpec

_FORMAT = 1
_FLAGS = ("episode_start", "valid")
# Backpressure cap on queued shards waiting for disk I/O to bound host RAM usage.
_PENDING_LIMIT = 4


def array_layout(signals: dict[str, SignalSpec]) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return ``name -> (per_frame_shape, torch_dtype)`` for all signals plus ``episode_start`` and ``valid``."""
    layout = {name: (spec.shape, spec.torch_dtype) for name, spec in signals.items()}
    return layout | dict.fromkeys(_FLAGS, ((), torch.bool))


class TrajectoryDataset:
    """Read-only handle to a sharded trajectory dataset directory written by :class:`TrajectoryWriter`.

    Args:
        path: Dataset directory containing ``meta.json`` and shard subdirectories.

    Attributes:
        path: Dataset directory path.
        frame_dt: Frame duration [s].
        signals: Recorded :class:`SignalSpec` objects keyed by name.
        generation: Generation metadata dict (includes ``complete: bool``).
        shards: Shard metadata list (``path``, ``world_count``, ``frame_count``, ``continues``).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        if meta["format"] != _FORMAT:
            raise ValueError(f"{self.path} has dataset format {meta['format']}. This library reads format {_FORMAT}")
        self.frame_dt = float(meta["frame_dt"])
        self.signals: dict[str, SignalSpec] = {s["name"]: SignalSpec.from_dict(s) for s in meta["signals"]}
        self.generation: dict[str, Any] = meta["generation"]
        self.shards: list[dict[str, Any]] = meta["shards"]

    @property
    def world_count(self) -> int:
        """Total number of worlds across all recordings (global IDs ``0 .. world_count - 1``)."""
        return sum(shard["world_count"] for shard in self.shards if not shard["continues"])

    @property
    def frame_count(self) -> int:
        """Total frames per world (0 if empty). Raises ``ValueError`` if recordings have unequal lengths."""
        # Sum frames across contiguous shard chains (one chain per recording).
        frame_counts: list[int] = []
        for shard in self.shards:
            if shard["continues"]:
                frame_counts[-1] += shard["frame_count"]
            else:
                frame_counts.append(shard["frame_count"])
        if len(set(frame_counts)) > 1:
            raise ValueError(
                f"every recording of a dataset must hold the same number of frames per world, "
                f"{self.path} holds {sorted(set(frame_counts))}"
            )
        return frame_counts[0] if frame_counts else 0

    def arrays(self, shard: int) -> dict[str, np.ndarray]:
        """Open read-only memory-mapped NumPy arrays for ``shards[shard]`` (all signals plus flags)."""
        folder = self.path / self.shards[shard]["path"]
        return {name: np.load(folder / f"{name}.npy", mmap_mode="r") for name in (*self.signals, *_FLAGS)}

    def held_out_worlds(self, fraction: float = 0.05, seed: int = 0) -> Tensor:
        """Deterministic CPU boolean mask ``[world_count]`` selecting held-out global world IDs (at least 1).

        Depends solely on ``world_count``, ``fraction``, and ``seed`` so distributed ranks agree identically.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must lie between 0 and 1, got {fraction}")
        world_count = self.world_count
        order = torch.randperm(world_count, generator=torch.Generator().manual_seed(seed))
        held_out = torch.zeros(world_count, dtype=torch.bool)
        held_out[order[: max(1, round(world_count * fraction))]] = True
        return held_out

    def __repr__(self) -> str:
        # Compute total frames directly so repr works even if recordings have unequal lengths.
        frame_count = sum(shard["world_count"] * shard["frame_count"] for shard in self.shards)
        lines = [
            f"TrajectoryDataset({self.path}), {frame_count:,} frames of {self.world_count:,} worlds, "
            f"frame_dt={self.frame_dt}, {len(self.shards)} shards"
        ]
        lines += [f"  {s.name} {s.shape} {s.dtype} ({s.role.value})" for s in self.signals.values()]
        return "\n".join(lines)


class TrajectoryWriter:
    """Asynchronous background writer for sharded :class:`TrajectoryDataset` directories.

    Copies each shard into recycled host buffers synchronously in :meth:`write` (so caller tensors can be reused
    immediately) and writes ``.npy`` files on a background thread pool (at most 4 shards queued). Use as a context
    manager; exiting flushes completed shards and writes ``meta.json`` even on exception.

    .. code-block:: python

        with TrajectoryWriter(path, signals, frame_dt) as writer:
            writer.write(frames, episode_start, valid)
        dataset = TrajectoryDataset(path)

    Args:
        path: Output directory (must not already contain ``meta.json`` or ``shard-*``; unrelated files allowed).
        signals: Signal specifications keyed by name.
        frame_dt: Frame duration [s].
        generation: Optional JSON-serializable metadata stored in ``meta.json`` (writer adds ``complete``).
    """

    def __init__(
        self,
        path: str | Path,
        signals: dict[str, SignalSpec],
        frame_dt: float,
        *,
        generation: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.signals = {spec.name: spec for spec in signals.values()}
        reserved = self.signals.keys() & set(_FLAGS)
        if reserved:
            raise ValueError(f"signal names {sorted(reserved)} are taken by the per-frame flags of a dataset")
        self._meta: dict[str, Any] = {
            "format": _FORMAT,
            "frame_dt": float(frame_dt),
            "signals": [spec.to_dict() for spec in self.signals.values()],
            "generation": dict(generation or {}),
            "shards": [],
        }
        json.dumps(self._meta)  # Fail fast on non-JSON-serializable metadata before recording
        self._layout = array_layout(self.signals)
        # Check only dataset files so extra files (e.g., an untrained checkpoint) can coexist in the folder.
        if (self.path / "meta.json").exists() or any(self.path.glob("shard-*")):
            raise FileExistsError(f"{self.path} holds a dataset. Record into a new folder, or delete this one")
        self.path.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._saves: list[Future[None]] = []  # Futures in shard order
        # Recycled pre-faulted host buffers avoid slow GPU->unfaulted-page copies on every shard.
        self._buffers: list[dict[str, Tensor]] = []

    def write(self, frames: Frames, episode_start: Tensor, valid: Tensor, *, continues: bool = False) -> None:
        """Copy tensors into host staging buffers and queue an asynchronous shard write.

        Args:
            frames: All dataset signals ``[world_count, frame_count, *shape]``.
            episode_start: Boolean mask ``[world_count, frame_count]`` marking episode starts.
            valid: Boolean mask ``[world_count, frame_count]`` marking usable frames.
            continues: ``True`` if these frames immediately follow the previous shard for the same worlds.
        """
        shards = self._meta["shards"]
        if episode_start.dim() != 2 or valid.shape != episode_start.shape:
            raise ValueError(
                f"episode_start and valid must share one shape [world_count, frame_count], got "
                f"{tuple(episode_start.shape)} and {tuple(valid.shape)}"
            )
        world_count, frame_count = episode_start.shape
        if continues and (not shards or shards[-1]["world_count"] != world_count):
            raise ValueError("a shard can only continue a previous shard of the same worlds")
        if frames.keys() != self.signals.keys():
            raise ValueError(f"the frames hold {sorted(frames)}, the dataset records {sorted(self.signals)}")
        values = {**frames, "episode_start": episode_start, "valid": valid}
        for name, value in values.items():
            shape, dtype = self._layout[name]
            shape = (world_count, frame_count, *shape)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must be {shape} {dtype}, got {tuple(value.shape)} {value.dtype}")
        # Apply backpressure so at most _PENDING_LIMIT shards (and buffer sets) are in flight at once.
        if len(self._saves) >= _PENDING_LIMIT:
            self._saves[-_PENDING_LIMIT].result()
        buffers = self._buffers.pop() if self._buffers else {}
        arrays = {}
        for name, value in values.items():
            if name not in buffers or buffers[name].numel() < value.numel():
                # Pre-fault host pages with zeros before copying from CUDA.
                buffers[name] = torch.zeros(value.numel(), dtype=value.dtype)
            # Synchronous copy into host staging buffer so caller may immediately overwrite inputs.
            arrays[name] = buffers[name][: value.numel()].view(value.shape).copy_(value.detach()).numpy()
        folder = f"shard-{len(shards):05d}"
        shards.append({"path": folder, "world_count": world_count, "frame_count": frame_count, "continues": continues})
        self._saves.append(self._pool.submit(self._save, self.path / folder, arrays, buffers))

    def close(self, *, complete: bool = True) -> TrajectoryDataset:
        """Flush all pending shard writes, write ``meta.json``, and return the opened :class:`TrajectoryDataset`.

        Truncates the shard list before the first failed write (if any), sets ``generation["complete"]``, and
        re-raises the write exception if ``complete=True``.
        """
        self._pool.shutdown()  # Wait for all queued writes to finish
        self._buffers.clear()  # Release host staging buffers
        failures = [save.exception() for save in self._saves]
        written = next((i for i, failure in enumerate(failures) if failure is not None), len(failures))
        self._meta["shards"] = self._meta["shards"][:written]
        self._meta["generation"]["complete"] = complete and written == len(failures)
        (self.path / "meta.json").write_text(json.dumps(self._meta, indent=2))
        if complete and written < len(failures):
            raise failures[written]
        return TrajectoryDataset(self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        # Flush completed shards with complete=False if an exception occurred, then let the exception propagate.
        self.close(complete=error_type is None)

    def _save(self, folder: Path, arrays: dict[str, np.ndarray], buffers: dict[str, Tensor]) -> None:
        """Write one shard's ``.npy`` files and return its staging buffers to the pool."""
        folder.mkdir()
        for name, array in arrays.items():
            np.save(folder / f"{name}.npy", array)
        self._buffers.append(buffers)
