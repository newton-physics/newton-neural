# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Running per-channel normalization buffers for network inputs and outputs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

FEATURE_LIMIT = 10.0
"""Clamp limit [std devs] applied to normalized network inputs to bound out-of-distribution states."""


class Normalizer(nn.Module):
    """Per-channel running mean and standard deviation stored as module buffers.

    Buffers move, save, and load with the owning module. Constant channels (``std <= 1e-6``) keep ``std = 1.0``.

    Args:
        width: Number of channels.

    Attributes:
        mean: Channel means ``[width]`` (float32, 0 before fitting).
        std: Channel standard deviations ``[width]`` (float32, 1 before fitting or for constant channels).
        count: Total sample count (float64 scalar).
    """

    mean: Tensor
    std: Tensor
    count: Tensor
    _mean64: Tensor
    _squares64: Tensor

    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        # Accumulate moments in float64 for numerical stability; expose float32 copies in mean/std.
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("_mean64", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("_squares64", torch.zeros(width, dtype=torch.float64))
        # Track fitted state on CPU to avoid device-to-host syncs when checking per training step.
        self._fitted = False
        self.register_load_state_dict_post_hook(_follow_count)

    @property
    def fitted(self) -> bool:
        """``True`` if statistics were fitted via :meth:`update` or loaded from a state dict (no device sync)."""
        return self._fitted

    @torch.no_grad()
    def update(self, x: Tensor, mask: Tensor | None = None) -> None:
        """Incorporate a batch of samples into the running statistics.

        Args:
            x: Input tensor ``[..., width]``.
            mask: Optional boolean mask ``[...]`` selecting valid rows (e.g., non-padded contacts).
        """
        x = x.reshape(-1, x.shape[-1]).double()
        if mask is not None:
            x = x[mask.reshape(-1)]
        sample_count = x.shape[0]
        if sample_count == 0:
            return
        mean = x.mean(dim=0)
        total = self.count + sample_count
        delta = mean - self._mean64
        # Chan's parallel algorithm for combining mean and sum-of-squared-deviations.
        self._squares64 += (x - mean).square().sum(dim=0) + delta.square() * self.count * sample_count / total
        self._mean64 += delta * sample_count / total
        self.count.copy_(total)
        self._fitted = True
        std = (self._squares64 / total).sqrt().float()
        self.mean.copy_(self._mean64.float())
        self.std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))

    def normalize(self, x: Tensor, limit: float | None = None) -> Tensor:
        """Standardize ``x`` to zero mean and unit variance per channel.

        Args:
            x: Input tensor ``[..., width]``.
            limit: Optional symmetric clamp bound ``[-limit, limit]`` in standard deviations.

        Returns:
            Normalized tensor ``[..., width]``.
        """
        normalized = (x - self.mean) / self.std
        return normalized if limit is None else normalized.clamp(-limit, limit)

    def denormalize(self, y: Tensor) -> Tensor:
        """Invert :meth:`normalize` on ``y`` ``[..., width]``."""
        return y * self.std + self.mean


def _follow_count(normalizer: Normalizer, incompatible_keys: Any) -> None:
    """Sync host-side ``_fitted`` flag after ``load_state_dict``."""
    normalizer._fitted = bool(normalizer.count > 0)
