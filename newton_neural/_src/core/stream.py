# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One CUDA stream for work that hands data between PyTorch and Warp.

PyTorch and Warp each keep a current stream per CUDA device. CUDA orders the work within a stream, and it orders
PyTorch's default stream with Warp's. A caller that works on a PyTorch stream of its own has neither, so a function
that takes tensors, computes with Warp, and returns tensors would read a state that PyTorch has not written yet,
or return a result that Warp has not computed yet. :func:`torch_on_warp_stream` puts the PyTorch part of such a
function on Warp's stream and keeps that stream in order with the caller's. Solvers reach the same order through
:class:`~newton_neural._src.core.graph.GraphedWork`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import warp as wp


@contextmanager
def torch_on_warp_stream(device: torch.device) -> Iterator[None]:
    """Run the PyTorch work of the block on Warp's current CUDA stream of ``device``.

    On entry Warp's stream waits for the caller's stream, so the tensors the block reads are complete. On exit the
    caller's stream waits for Warp's, so what the block wrote is complete for the caller. A tensor that the block
    hands back is allocated before the block and filled inside it, so that PyTorch's allocator keeps it with the
    stream that goes on using it.

    Does nothing on the CPU, where PyTorch already works on Warp's stream, and on PyTorch's default stream, which
    CUDA keeps in order with every blocking stream, Warp's among them.
    """
    if device.type == "cuda":
        caller = torch.cuda.current_stream(device)
        if caller.cuda_stream != 0:
            warp_stream = wp.get_stream(wp.device_from_torch(device))
            if caller.cuda_stream != warp_stream.cuda_stream:
                stream = torch.cuda.ExternalStream(warp_stream.cuda_stream, device=device)
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    yield
                caller.wait_stream(stream)
                return
    yield
