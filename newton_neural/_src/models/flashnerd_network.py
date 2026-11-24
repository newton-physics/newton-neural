# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""``FlashNeRDNetwork``: feature embedding, contact-code injection, and readout shell wrapping a temporal backbone."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

import torch
from torch import Tensor, nn

from ..backbones.backbone import BackboneBase, full_precision

_INITIAL_GATE = 0.1
"""Initial sigmoid value for gated contact injection projections."""


class ContactInjection(StrEnum):
    """Injection sites where :class:`FlashNeRDNetwork` adds the encoded contact vector."""

    EMBEDDING = "embedding"
    """Added to the frame embedding before the backbone so contact history propagates over time."""

    READOUT = "readout"
    """Added to the readout hidden layer so current contacts influence the prediction directly."""

    INCREMENT = "increment"
    """Projected directly onto the output increment via a zero-initialized linear map."""


class FlashNeRDNetwork(nn.Module):
    """Embeds frame features, runs ``backbone``, injects optional contact codes, and projects the prediction.

    Computes::

        h0 = W_e x + b_e + G_in(c)
        h  = backbone(h0)
        p  = W_h [RMSNorm(h), x] + b_h + G_r(c)
        y  = W_o relu(p) + b_o + W_p x + b_p + G_p(c)

    ``W_p``, ``b_p``, and ``G_p`` are zero-initialized so training starts from the nonlinear readout head. Each
    active injection ``G`` is a bias-free linear projection (optionally scaled by ``sigmoid(g)`` initialized to
    ``0.1`` when ``gated=True``), so a zero contact code contributes nothing. When all three sites are active and
    ungated, ``c`` enters identically to concatenating ``c`` onto ``x``.

    Args:
        feature_count: Input feature dimension.
        output_count: Prediction output dimension.
        backbone: Streaming temporal trunk.
        code_count: Contact code width (``0`` if no contact input).
        injections: Active :class:`ContactInjection` sites (must be non-empty iff ``code_count > 0``).
        gated: Wrap each injection projection with a learned scalar gate initialized to ``0.1``.
        readout_hidden_dim: Hidden width of the readout MLP.
    """

    def __init__(
        self,
        feature_count: int,
        output_count: int,
        backbone: BackboneBase,
        *,
        code_count: int = 0,
        injections: Sequence[ContactInjection | str] = (),
        gated: bool = False,
        readout_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        injections = tuple(ContactInjection(point) for point in injections)
        if bool(code_count) != bool(injections):
            raise ValueError(f"a contact code and injection points come together, got {code_count=}, {injections=}")
        self.embed = nn.Linear(feature_count, backbone.width)
        self.backbone = backbone
        self.norm = nn.RMSNorm(backbone.width, eps=1e-6)
        self.readout_hidden = nn.Linear(backbone.width + feature_count, readout_hidden_dim)
        self.readout_out = nn.Linear(readout_hidden_dim, output_count)
        self.direct = nn.Linear(feature_count, output_count)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.embed.bias)
        nn.init.zeros_(self.direct.weight)
        nn.init.zeros_(self.direct.bias)
        widths = {
            ContactInjection.EMBEDDING: backbone.width,
            ContactInjection.READOUT: readout_hidden_dim,
            ContactInjection.INCREMENT: output_count,
        }
        widths = {point: widths[point] for point in injections}
        self.injection = _Injection(code_count, widths, gated) if code_count else None

    @property
    def receptive_field(self) -> int | None:
        """Number of frames (including current) visible to one prediction, or ``None`` if unbounded."""
        return self.backbone.receptive_field

    def forward(self, x: Tensor, code: Tensor | None = None) -> Tensor:
        """Predict normalized outputs ``[B, T, output_count]`` over feature windows ``[B, T, feature_count]``."""
        terms = self._terms(code)
        return self._readout(self.backbone(self._embed(x, terms)), x, terms)

    def step(self, x: Tensor, history: dict[str, Tensor], code: Tensor | None = None) -> Tensor:
        """Predict outputs ``[B, output_count]`` for one frame ``[B, feature_count]``, updating ``history``."""
        terms = self._terms(code)
        return self._readout(self.backbone.step(self._embed(x, terms), history), x, terms)

    def history(self, world_count: int, *, dtype: torch.dtype | None = None) -> dict[str, Tensor]:
        """Allocate zeroed per-world backbone history buffers on the network's device."""
        weight = self.norm.weight  # Stays FP32 even when linear weights are cast to 16-bit
        return self.backbone.history(world_count, device=weight.device, dtype=dtype or weight.dtype)

    def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
        """Zero out history buffers for worlds where ``world_mask`` ``[B]`` (bool) is ``True``."""
        self.backbone.reset_history(history, world_mask)

    def hidden_weights(self) -> list[nn.Parameter]:
        """Return hidden weight matrices updated by Muon (backbone hidden weights + ``readout_hidden.weight``)."""
        return [*self.backbone.hidden_weights(), self.readout_hidden.weight]

    def _terms(self, code: Tensor | None) -> dict[ContactInjection, Tensor]:
        if (code is None) != (self.injection is None):
            raise ValueError("a contact code must be passed exactly when the network was built with code_count > 0")
        return {} if code is None else self.injection(code)

    def _embed(self, x: Tensor, terms: dict[ContactInjection, Tensor]) -> Tensor:
        return full_precision(_add(self.embed(x), terms, ContactInjection.EMBEDDING))

    def _readout(self, h: Tensor, x: Tensor, terms: dict[ContactInjection, Tensor]) -> Tensor:
        p = self.readout_hidden(torch.cat((self.norm(full_precision(h)), x), dim=-1))
        p = _add(p, terms, ContactInjection.READOUT)
        y = self.readout_out(torch.relu(p)) + self.direct(x)
        return full_precision(_add(y, terms, ContactInjection.INCREMENT))


class _Injection(nn.Module):
    """Projects contact code ``c`` to all active injection sites in a single packed linear layer."""

    def __init__(self, code_count: int, widths: dict[ContactInjection, int], gated: bool) -> None:
        super().__init__()
        self.points = tuple(widths)
        self.widths = tuple(widths.values())
        self.project = nn.Linear(code_count, sum(self.widths), bias=False)
        logit = math.log(_INITIAL_GATE / (1 - _INITIAL_GATE))
        self.gate = nn.Parameter(torch.full((len(widths),), logit)) if gated else None
        if ContactInjection.INCREMENT in widths:
            rows = self.project.weight.split(self.widths)[self.points.index(ContactInjection.INCREMENT)]
            nn.init.zeros_(rows)

    def forward(self, code: Tensor) -> dict[ContactInjection, Tensor]:
        terms = self.project(code).split(self.widths, dim=-1)
        if self.gate is not None:
            terms = [gate * term for gate, term in zip(torch.sigmoid(self.gate), terms, strict=True)]
        return dict(zip(self.points, terms, strict=True))


def _add(value: Tensor, terms: dict[ContactInjection, Tensor], point: ContactInjection) -> Tensor:
    return value + terms[point] if point in terms else value
