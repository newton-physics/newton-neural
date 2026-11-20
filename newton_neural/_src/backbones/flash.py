# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""``BackboneFlash``: parallel multi-query attention and SwiGLU FFN backbone (default for ``FlashNeRD``)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import AttentionStack, StreamingAttention
from .backbone import BackboneBase


class BackboneFlash(AttentionStack, BackboneBase):
    """Parallel attention + SwiGLU FFN transformer backbone optimized for low per-step latency.

    Each block applies RMSNorm, projects once into ``[Q, K, V, gate, up]``, runs multi-query sliding-window
    attention (1 shared KV head) in parallel with SwiGLU, and merges both branches via one output projection
    (2 linear layers per block instead of 4). Receptive field is ``1 + sum(span - 1 for span in spans)``.

    Args:
        head_count: Number of query heads (model width is ``head_count * head_dim``).
        head_dim: Per-head width.
        ffn_dim: SwiGLU hidden width.
        spans: Causal attention window per block (length sets block depth).
    """

    def __init__(
        self, *, head_count: int = 8, head_dim: int = 64, ffn_dim: int = 1024, spans: Sequence[int] = (4, 4, 4, 4)
    ) -> None:
        super().__init__()
        self.head_count = head_count
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.spans = tuple(spans)
        self.width = head_count * head_dim
        self.receptive_field = 1 + sum(span - 1 for span in self.spans)
        self.blocks = nn.ModuleList(_Block(head_count, head_dim, ffn_dim, span, len(self.spans)) for span in self.spans)

    def config(self) -> dict[str, Any]:
        """Return constructor keyword arguments as plain data."""
        return {
            "head_count": self.head_count,
            "head_dim": self.head_dim,
            "ffn_dim": self.ffn_dim,
            "spans": list(self.spans),
        }


class _Block(nn.Module):
    def __init__(self, head_count: int, head_dim: int, ffn_dim: int, span: int, depth: int) -> None:
        super().__init__()
        width = head_count * head_dim
        self.attention = StreamingAttention(head_count, 1, head_dim, span)
        self.widths = (width + 2 * head_dim, ffn_dim, ffn_dim)  # [QKV, gate, up]
        self.norm = nn.RMSNorm(width, eps=1e-6)
        self.inp = nn.Linear(width, sum(self.widths), bias=False)
        self.out = nn.Linear(width + ffn_dim, width, bias=False)
        nn.init.normal_(self.inp.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02 / math.sqrt(2 * depth))

    def forward(self, h: Tensor) -> Tensor:
        qkv, gate, up = self.inp(self.norm(h)).split(self.widths, dim=-1)
        return self._merge(h, self.attention(qkv), gate, up)

    def step(self, h: Tensor, ring: Tensor, step: Tensor) -> Tensor:
        qkv, gate, up = self.inp(self.norm(h)).split(self.widths, dim=-1)
        return self._merge(h, self.attention.step(qkv, ring, step), gate, up)

    def _merge(self, h: Tensor, attended: Tensor, gate: Tensor, up: Tensor) -> Tensor:
        return h + self.out(torch.cat((attended, F.silu(gate) * up), dim=-1))
