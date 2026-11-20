# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract interface for streaming temporal backbones."""

from __future__ import annotations

import abc
from typing import ClassVar

import torch
from torch import Tensor, nn

from ..core.registry import Part


def full_precision(x: Tensor) -> Tensor:
    """Promote float16/bfloat16 tensors to float32 while preserving float32/float64.

    Used to keep normalization, attention scores, and residual streams in float32 under ``torch.autocast``.
    """
    return x if x.dtype == torch.float64 else x.float()


class BackboneBase(nn.Module, Part, abc.ABC):
    """Streaming temporal trunk mapping embedded frames to hidden states.

    Subclasses implement both windowed training (:meth:`forward`) and single-frame deployment (:meth:`step`)
    using fixed-shape in-place history buffers compatible with ``torch.compile`` and CUDA graphs. Verify streaming
    equivalence across mid-episode resets with :func:`newton_neural.testing.check_backbone_streaming`.
    """

    registry: ClassVar[dict[str, type]] = {}

    width: int
    """Feature width of input frame embeddings and output hidden states."""

    receptive_field: int | None
    """Number of frames (including current) visible to one hidden state, or ``None`` for unbounded/recurrent history."""

    @abc.abstractmethod
    def forward(self, h: Tensor) -> Tensor:
        """Process full training windows causally (frame 0 treated as an episode start).

        Args:
            h: Embedded frames ``[B, T, width]``.

        Returns:
            Hidden states ``[B, T, width]``.
        """

    @abc.abstractmethod
    def step(self, h: Tensor, history: dict[str, Tensor]) -> Tensor:
        """Process one newest frame and update ``history`` in place (no data-dependent control flow or allocations).

        Args:
            h: Embedded current frame ``[B, width]``.
            history: Per-world history buffers allocated by :meth:`history`.

        Returns:
            Hidden state ``[B, width]``.
        """

    @abc.abstractmethod
    def history(self, world_count: int, *, device: torch.device | str, dtype: torch.dtype) -> dict[str, Tensor]:
        """Allocate zero-initialized history buffers for ``world_count`` freshly reset worlds.

        Args:
            world_count: Batch/world count ``B``.
            device: Target tensor device.
            dtype: Floating-point dtype for stored activations (counters remain ``int64``).

        Returns:
            Named fixed-shape tensors ``[B, ...]`` initialized to zero.
        """

    def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
        """Zero out history rows in place for worlds where ``world_mask`` ``[B]`` (bool) is ``True``."""
        for value in history.values():
            value.masked_fill_(world_mask.view(-1, *(1,) * (value.dim() - 1)), 0)

    def hidden_weights(self) -> list[nn.Parameter]:
        """Return hidden weight matrices updated by spectral optimizers (defaults to all ``nn.Linear`` weights)."""
        return [module.weight for module in self.modules() if isinstance(module, nn.Linear)]
