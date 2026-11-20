# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""``BackboneTransformer``: standard serial multi-head attention + SwiGLU FFN baseline backbone."""

from __future__ import annotations

import math
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from .attention import AttentionStack, StreamingAttention
from .backbone import BackboneBase


class BackboneTransformer(AttentionStack, BackboneBase):
    """Standard pre-norm transformer backbone (serial multi-head attention + SwiGLU FFN).

    Uses separate K/V heads per attention head and 4 linear layers per block (slower than
    :class:`~newton_neural.backbones.BackboneFlash`). Receptive field is ``1 + depth * (span - 1)``.

    Args:
        width: Hidden dimension (must be divisible by ``head_count``).
        head_count: Number of attention heads.
        depth: Number of transformer blocks.
        span: Causal attention window per block (including current frame).
        ffn_dim: SwiGLU hidden dimension.
    """

    def __init__(
        self, *, width: int = 512, head_count: int = 8, depth: int = 4, span: int = 8, ffn_dim: int = 2048
    ) -> None:
        super().__init__()
        if width % head_count:
            raise ValueError(f"width {width} is not a multiple of head_count {head_count}")
        self.width = width
        self.head_count = head_count
        self.depth = depth
        self.span = span
        self.ffn_dim = ffn_dim
        self.receptive_field = 1 + depth * (span - 1)
        self.blocks = nn.ModuleList(_Block(width, head_count, depth, span, ffn_dim) for _ in range(depth))

    def config(self) -> dict[str, Any]:
        """Return constructor keyword arguments as plain data."""
        return {
            "width": self.width,
            "head_count": self.head_count,
            "depth": self.depth,
            "span": self.span,
            "ffn_dim": self.ffn_dim,
        }


class _Block(nn.Module):
    def __init__(self, width: int, head_count: int, depth: int, span: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention = StreamingAttention(head_count, head_count, width // head_count, span)
        self.norm1 = nn.RMSNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width, eps=1e-6)
        self.up = nn.Linear(width, 2 * ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, width, bias=False)
        for layer in (self.qkv, self.up):
            nn.init.normal_(layer.weight, std=0.02)
        for layer in (self.proj, self.down):
            nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * depth))

    def forward(self, h: Tensor) -> Tensor:
        return self._feedforward(h + self.proj(self.attention(self.qkv(self.norm1(h)))))

    def step(self, h: Tensor, ring: Tensor, step: Tensor) -> Tensor:
        return self._feedforward(h + self.proj(self.attention.step(self.qkv(self.norm1(h)), ring, step)))

    def _feedforward(self, h: Tensor) -> Tensor:
        gate, up = self.up(self.norm2(h)).chunk(2, dim=-1)
        return h + self.down(F.silu(gate) * up)
