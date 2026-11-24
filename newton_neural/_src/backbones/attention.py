# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sliding-window causal attention with zero-copy ring-buffer streaming.

Each layer attends to the current frame and ``span - 1`` previous frames using a learned relative lag bias. At
inference time, keys and values are written into a fixed ring buffer at ``step % span`` without shifting memory,
matching windowed training exactly. :class:`AttentionStack` shares the loop and history layout across
:class:`BackboneFlash` and :class:`BackboneTransformer`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import full_precision


class StreamingAttention(nn.Module):
    """Grouped/multi-query causal sliding-window attention over the last ``span`` frames.

    Scores and softmax always execute in float32 (or higher) even under ``torch.autocast``. For ``span == 1``,
    softmax is identically 1 and the layer returns the frame's own value directly (zero gradient to Q, K, and
    :attr:`lag_bias`).

    Args:
        head_count: Number of query heads (must be a multiple of ``kv_head_count``).
        kv_head_count: Number of key/value heads (``1`` for multi-query attention).
        head_dim: Per-head channel width.
        span: Causal window length including the current frame.
    """

    def __init__(self, head_count: int, kv_head_count: int, head_dim: int, span: int) -> None:
        super().__init__()
        if head_count % kv_head_count:
            raise ValueError(f"head_count {head_count} is not a multiple of kv_head_count {kv_head_count}")
        if span < 1:
            raise ValueError(f"span must be at least 1, got {span}")
        self.kv_head_count = kv_head_count
        self.group_size = head_count // kv_head_count  # Query heads per KV head
        self.head_dim = head_dim
        self.span = span
        self.query_width = head_count * head_dim
        self.kv_width = kv_head_count * head_dim
        self.lag_bias = nn.Parameter(torch.zeros(head_count, span))
        """Learned relative attention bias ``[head_count, span]`` indexed by lag ``0 .. span - 1``."""
        self.register_buffer("slots", torch.arange(span), persistent=False)

    def forward(self, qkv: Tensor) -> Tensor:
        """Apply banded causal attention over a full window.

        Args:
            qkv: Packed ``[Q, K, V]`` tensor ``[B, T, query_width + 2 * kv_width]``.

        Returns:
            Attended output ``[B, T, query_width]`` in ``qkv.dtype``.
        """
        if self.span == 1:
            return self._own_value(qkv)
        b, t = qkv.shape[:2]
        kh, g, d = self.kv_head_count, self.group_size, self.head_dim
        q, keys, values = full_precision(qkv).split((self.query_width, self.kv_width, self.kv_width), dim=-1)
        # Fold KV heads into batch and group Q heads into rows so one bmm serves all heads without copying KV.
        q = q.view(b, t, kh, g * d).transpose(1, 2).reshape(b * kh, t * g, d)
        keys = keys.view(b, t, kh, d).transpose(1, 2).reshape(b * kh, t, d)
        values = values.view(b, t, kh, d).transpose(1, 2).reshape(b * kh, t, d)
        frame = torch.arange(t, device=qkv.device)
        lag = frame[:, None] - frame  # [T, T], key lag behind query
        band = ((lag >= 0) & (lag < self.span))[:, None, :]
        bias = F.embedding(lag.clamp(0, self.span - 1), self.lag_bias.T)  # [T, T, head_count]
        bias = bias.view(t, t, kh, g).permute(2, 0, 3, 1)  # [KH, T, G, T]
        with torch.autocast(qkv.device.type, enabled=False):
            scores = torch.bmm(q, keys.transpose(1, 2)).view(b, kh, t, g, t) * d**-0.5 + bias
            weights = torch.softmax(torch.where(band, scores, -torch.inf), dim=-1)
            out = torch.bmm(weights.view(b * kh, t * g, t), values)  # [B * KH, T * G, D]
        return out.view(b, kh, t, g * d).transpose(1, 2).reshape(b, t, self.query_width).to(qkv.dtype)

    def step(self, qkv: Tensor, ring: Tensor, step: Tensor) -> Tensor:
        """Write the newest KV pair into ``ring`` at ``step % span`` and attend over valid ring slots.

        Slots with ``lag > step`` belong to pre-reset history; both their scores and values are masked via
        ``torch.where`` so stale non-finite values cannot leak into the output.

        Args:
            qkv: Packed ``[Q, K, V]`` tensor ``[B, query_width + 2 * kv_width]``.
            ring: Per-world KV ring buffer ``[B, span, 2 * kv_width]`` updated in place.
            step: Per-world frame count since last reset (before current frame) ``[B]`` int64.

        Returns:
            Attended output ``[B, query_width]`` in ``qkv.dtype``.
        """
        b = qkv.shape[0]
        kh, g, d, span = self.kv_head_count, self.group_size, self.head_dim, self.span
        q, pair = full_precision(qkv).split((self.query_width, 2 * self.kv_width), dim=-1)
        ring[torch.arange(b, device=ring.device), step % span] = pair.to(ring.dtype)
        if span == 1:  # Write ring first so history stays consistent with span > 1
            return self._own_value(qkv)
        lag = (step[:, None] - self.slots) % span  # [B, span]
        valid = (lag <= step[:, None])[:, :, None, None]
        bias = F.embedding(lag, self.lag_bias.T).view(b, span, kh, g)
        keys, values = full_precision(ring).view(b, span, 2, kh, 1, d).unbind(2)  # [B, span, KH, 1, D]
        # Elementwise multiply-sum fuses cleanly under torch.compile for small spans instead of tiny bmm launches.
        scores = (keys * q.view(b, 1, kh, g, d)).sum(-1) * d**-0.5 + bias  # [B, span, KH, G]
        weights = torch.softmax(torch.where(valid, scores, -torch.inf), dim=1)
        out = (weights.unsqueeze(-1) * torch.where(valid.unsqueeze(-1), values, 0.0)).sum(1)  # [B, KH, G, D]
        return out.reshape(b, self.query_width).to(qkv.dtype)

    def _own_value(self, qkv: Tensor) -> Tensor:
        """Broadcast each frame's own value vector across its query-head group (used when ``span == 1``)."""
        kh, g, d = self.kv_head_count, self.group_size, self.head_dim
        values = qkv[..., self.query_width + self.kv_width :]
        lead = values.shape[:-1]
        return values.view(*lead, kh, 1, d).expand(*lead, kh, g, d).reshape(*lead, self.query_width)


class AttentionStack:
    """Mixin implementing ``forward``, ``step``, and ring-buffer ``history`` for stacked attention backbones.

    Place before :class:`~newton_neural.backbones.BackboneBase` and assign :attr:`blocks` (each exposing
    ``attention: StreamingAttention``, ``forward(h)``, and ``step(h, ring, step)``). History contains
    ``ring{i}`` ``[B, span, 2 * kv_width]`` per block and a shared ``step`` counter ``[B]`` int64.
    """

    blocks: nn.ModuleList

    def forward(self, h: Tensor) -> Tensor:
        """Run all blocks sequentially over window ``[B, T, width]``."""
        for block in self.blocks:
            h = block(h)
        return h

    def step(self, h: Tensor, history: dict[str, Tensor]) -> Tensor:
        """Run all blocks for one frame ``[B, width]`` and increment ``history["step"]`` in place."""
        step = history["step"]
        for i, block in enumerate(self.blocks):
            h = block.step(h, history[f"ring{i}"], step)
        step.add_(1)
        return h

    def history(self, world_count: int, *, device: torch.device | str, dtype: torch.dtype) -> dict[str, Tensor]:
        """Allocate zeroed per-block KV rings and frame counter for ``world_count`` worlds."""
        history = {}
        for i, block in enumerate(self.blocks):
            shape = (world_count, block.attention.span, 2 * block.attention.kv_width)
            history[f"ring{i}"] = torch.zeros(shape, device=device, dtype=dtype)
        history["step"] = torch.zeros(world_count, device=device, dtype=torch.int64)
        return history
