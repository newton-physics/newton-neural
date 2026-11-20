# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""``BackboneMLP``: fixed-window MLP backbone without attention (e.g., ``window=1`` for free motion)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .backbone import BackboneBase, full_precision


class BackboneMLP(BackboneBase):
    """MLP over the concatenated last ``window`` embedded frames (oldest first).

    At the start of an episode, unavailable past frames repeat frame 0.

    Args:
        width: Input/hidden/output feature width.
        depth: Number of ``Linear + SiLU`` layers.
        window: Number of frames concatenated (including current frame).
    """

    def __init__(self, *, width: int = 512, depth: int = 4, window: int = 4) -> None:
        super().__init__()
        if depth < 1 or window < 1:
            raise ValueError(f"depth and window must be at least 1, got {depth} and {window}")
        self.width = width
        self.depth = depth
        self.window = window
        self.receptive_field = window
        layers: list[nn.Module] = []
        for i in range(depth):
            layers += [nn.Linear(window * width if i == 0 else width, width), nn.SiLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, h: Tensor) -> Tensor:
        """Process full training windows ``[B, T, width]``."""
        first = h[:, :1].expand(-1, self.window - 1, -1)
        windows = torch.cat((first, h), dim=1).unfold(1, self.window, 1)  # [B, T, width, window]
        return self.layers(windows.transpose(-1, -2).flatten(-2))

    def step(self, h: Tensor, history: dict[str, Tensor]) -> Tensor:
        """Process one frame ``[B, width]`` and shift ``history["frames"]`` in place."""
        frames, step = history["frames"], history["step"]
        newest = h.unsqueeze(1).to(frames.dtype)
        shifted = torch.cat((frames[:, 1:], newest), dim=1)
        # Use torch.where rather than blending so stale NaNs in reset worlds cannot leak into the new episode.
        frames.copy_(torch.where((step == 0).view(-1, 1, 1), newest, shifted))
        step.add_(1)
        return self.layers(full_precision(frames).flatten(1))

    def history(self, world_count: int, *, device: torch.device | str, dtype: torch.dtype) -> dict[str, Tensor]:
        """Allocate zeroed frame window ``[B, window, width]`` and ``step`` counter ``[B]`` int64."""
        return {
            "frames": torch.zeros(world_count, self.window, self.width, device=device, dtype=dtype),
            "step": torch.zeros(world_count, device=device, dtype=torch.int64),
        }

    def config(self) -> dict[str, Any]:
        """Return constructor keyword arguments as plain data."""
        return {"width": self.width, "depth": self.depth, "window": self.window}
