# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph capture and replay for mixed PyTorch and Warp steps.

Executes PyTorch operations on Warp's current CUDA stream with a private PyTorch memory pool so combined Warp/PyTorch
steps can be captured into and replayed from a single CUDA graph. Captured callables must perform no host syncs and
no data-dependent control flow or allocations after their warm-up call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import warp as wp

_GRAPH_LIMIT = 16
"""Max distinct argument-identity tuples cached per instance (guard against per-step object recreation)."""


class GraphedWork:
    """Executes mixed PyTorch/Warp work on Warp's CUDA stream and replays it from cached CUDA graphs.

    On CUDA, the first call executes eagerly to trigger kernel loading and ``torch.compile``. Subsequent calls
    capture and replay one CUDA graph per unique tuple of argument object identities (keeping those arguments alive
    to preserve buffer addresses). Exceeding 16 distinct tuples raises ``RuntimeError`` to catch callers allocating
    fresh state/control objects every step.

    When called inside an external Warp capture (``wp.ScopedCapture`` in ``THREAD_LOCAL`` or ``RELAXED`` mode),
    work is recorded directly into the caller's graph (requires one prior warm-up call outside capture, and this
    instance must remain alive while the caller's graph replays so the private memory pool stays valid).

    On CPU, executes eagerly without graphs.

    Args:
        work: Callable invoked with the arguments passed to :meth:`run`. Must be graph-safe from its second call
            onward (no host syncs, no data-dependent launches).
        device: Target Warp device.
        use_cuda_graph: Enable internal CUDA graph caching on CUDA devices (external capture works regardless).
    """

    def __init__(self, work: Callable[..., None], device: wp.Device, *, use_cuda_graph: bool = True) -> None:
        self._work = work
        self._device = device
        self._use_cuda_graph = use_cuda_graph
        self._warm = False
        self._graphs: dict[tuple[int, ...], tuple[wp.Graph, tuple[Any, ...]]] = {}
        if device.is_cuda:
            self._torch_device = torch.device(wp.device_to_torch(device))
            # Dedicated pool prevents other PyTorch allocations from reusing buffers baked into captured graphs.
            self._pool = torch.cuda.MemPool()

    def run(self, *arguments: Any) -> None:
        """Execute ``work(*arguments)`` eagerly, via cached CUDA graph replay, or into an active Warp capture.

        Args:
            *arguments: Arguments forwarded to ``work``; object identities key the graph cache.

        Raises:
            RuntimeError: If called inside a CPU capture, inside a CUDA capture before warm-up, or after exceeding
                the 16-graph cache limit.
        """
        device = self._device
        if device.is_cpu:
            if device.is_capturing:
                raise RuntimeError(
                    "Warp's graph capture on the CPU records Warp launches only, so the torch part of this "
                    "step would be missing from every replay. Step outside the capture"
                )
            self._work(*arguments)
            return

        stream = wp.get_stream(device)
        if stream.is_capturing:
            if not self._warm:
                raise RuntimeError(
                    "the first step compiles and allocates, which a CUDA graph capture cannot record. "
                    "Take one step before the capture begins"
                )
            self._record(stream, arguments)
        elif not (self._use_cuda_graph and self._warm):
            with self._torch_on(stream):
                self._work(*arguments)
            self._warm = True
        else:
            key = tuple(id(argument) for argument in arguments)
            if key not in self._graphs:
                if len(self._graphs) == _GRAPH_LIMIT:
                    raise RuntimeError(
                        f"this step was recorded for {_GRAPH_LIMIT} combinations of argument objects, such as "
                        "states, controls and contacts, and is called with another. A graph replays the memory "
                        "it was recorded on and is kept with its objects, so new objects at every step record "
                        "and keep a graph at every step. Step with the same few objects, or turn the replay "
                        "off with use_cuda_graph=False"
                    )
                with wp.ScopedCapture(stream=stream) as capture:
                    self._record(stream, arguments)
                self._graphs[key] = (capture.graph, arguments)
            wp.capture_launch(self._graphs[key][0], stream=stream)

    def _record(self, stream: wp.Stream, arguments: tuple[Any, ...]) -> None:
        """Execute ``work`` on the capturing ``stream`` using the private PyTorch memory pool."""
        with self._torch_on(stream), self._allocating_from_pool():
            self._work(*arguments)

    def _torch_on(self, stream: wp.Stream) -> torch.cuda.StreamContext:
        """Bind PyTorch's current CUDA stream to Warp's ``stream`` for correct ordering and capture detection."""
        return torch.cuda.stream(torch.cuda.ExternalStream(stream.cuda_stream, device=self._torch_device))

    @contextmanager
    def _allocating_from_pool(self) -> Iterator[None]:
        """Route current-thread PyTorch CUDA allocations to ``self._pool``.

        Mirrors ``torch.cuda.use_mem_pool`` without querying PyTorch's active graph handle, which fails when capture
        was started by Warp rather than PyTorch. Pool lifetime is tied to ``self``.
        """
        index, pool = self._torch_device.index, self._pool.id
        torch._C._cuda_beginAllocateCurrentThreadToPool(index, pool)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateToPool(index, pool)
            torch._C._cuda_releasePool(index, pool)
