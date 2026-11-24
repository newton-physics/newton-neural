# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hybrid Muon + AdamW optimizer (:class:`OptimizerMuon`) and factory (:func:`create_optimizer`)."""

from __future__ import annotations

import functools
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

_NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
"""Quintic Newton-Schulz polynomial coefficients for approximate matrix orthogonalization."""


def create_optimizer(
    module: nn.Module, name: str = "muon", *, lr: float = 1e-3, weight_decay: float = 0.01, **options: Any
) -> torch.optim.Optimizer:
    """Build an optimizer for ``module`` using matched ``lr`` and ``weight_decay`` scaling.

    - ``"muon"``: :class:`OptimizerMuon` updating weights returned by child ``hidden_weights()`` methods with Muon
      and all remaining parameters (embeddings, readouts, biases, norms, contact encoders) with AdamW.
    - ``"adamw"``: ``torch.optim.AdamW`` for all parameters (use for models declaring no ``hidden_weights()``).

    Args:
        module: Target PyTorch module (typically a :class:`~newton_neural.models.NeRDBase`).
        name: ``"muon"`` or ``"adamw"``.
        lr: Learning rate.
        weight_decay: Decoupled weight decay.
        **options: Extra optimizer kwargs (e.g., ``betas``, ``momentum``, ``ns_steps``).

    Raises:
        ValueError: If ``name`` is unknown or ``"muon"`` is requested on a module with no ``hidden_weights()``.
    """
    params = [p for p in module.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **options)
    if name != "muon":
        raise ValueError(f"unknown optimizer {name!r}. Use 'muon' or 'adamw'")
    named = [child.hidden_weights() for child in module.modules() if hasattr(child, "hidden_weights")]
    hidden = {id(weight) for weights in named for weight in weights}
    if not hidden:
        raise ValueError(
            "the module names no hidden weights, so Muon would update nothing. Give one of its modules a "
            "hidden_weights() method, or use name='adamw'"
        )
    groups = [
        {"params": [p for p in params if id(p) in hidden], "muon": True},
        {"params": [p for p in params if id(p) not in hidden], "muon": False},
    ]
    return OptimizerMuon([group for group in groups if group["params"]], lr=lr, weight_decay=weight_decay, **options)


class OptimizerMuon(torch.optim.Optimizer):
    """Muon (orthogonalized Nesterov momentum) for 2D+ hidden weights combined with AdamW for other parameters.

    Orthogonalizes Nesterov momentum buffers via ``ns_steps`` quintic Newton-Schulz iterations in bfloat16 (flattening
    conv kernels ``[out, in, *k]`` to ``[out, in * prod(k)]``) and scales steps by ``0.2 * sqrt(max(rows, cols)) * lr``
    to match AdamW step magnitudes so learning rates and weight decays transfer directly.

    Args:
        params: Parameters or parameter groups (set ``"muon": True/False`` per group, or omit to split by
            ``ndim >= 2``).
        lr: Learning rate.
        weight_decay: Decoupled weight decay.
        momentum: Muon Nesterov momentum factor.
        betas: AdamW ``(beta1, beta2)``.
        eps: AdamW numerical stability epsilon.
        ns_steps: Number of Newton-Schulz orthogonalization iterations.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        ns_steps: int = 5,
    ) -> None:
        groups = list(params)
        if groups and not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        split = []
        for group in groups:
            tensors = list(group["params"])
            if "muon" in group:
                if group["muon"] and any(p.ndim < 2 for p in tensors):
                    raise ValueError("Muon updates matrices and convolution kernels only, not vectors")
                split.append({**group, "params": tensors})
                continue
            for muon in (True, False):  # Auto-split by tensor rank
                chosen = [p for p in tensors if (p.ndim >= 2) == muon]
                if chosen:
                    split.append({**group, "params": chosen, "muon": muon})
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "betas": betas,
            "eps": eps,
            "ns_steps": ns_steps,
        }
        super().__init__(split, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """Perform one Muon + AdamW optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["muon"]:
                self._muon(group)
            else:
                self._adamw(group)
        return loss

    def _muon(self, group: dict[str, Any]) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        lr, decay, momentum = group["lr"], group["weight_decay"], group["momentum"]
        grads = [p.grad for p in params]
        for p in params:
            if not self.state[p]:
                self.state[p]["momentum"] = torch.zeros_like(p)
        buffers = [self.state[p]["momentum"] for p in params]
        torch._foreach_lerp_(buffers, grads, 1 - momentum)
        ahead = torch._foreach_lerp(grads, buffers, momentum)  # Nesterov lookahead
        matrices = [a.reshape(len(a), -1) for a in ahead]
        updates = [u.reshape_as(p) for u, p in zip(_orthogonalize(matrices, group["ns_steps"]), params, strict=True)]
        torch._foreach_mul_(updates, [-lr * 0.2 * math.sqrt(max(m.shape)) for m in matrices])
        torch._foreach_mul_(params, 1 - lr * decay)
        torch._foreach_add_(params, updates)

    def _adamw(self, group: dict[str, Any]) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        lr, decay, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        alike: dict[int, list[Tensor]] = defaultdict(list)  # Group parameters sharing the same step count
        for p in params:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["average"] = torch.zeros_like(p)
                state["square"] = torch.zeros_like(p)
            state["step"] += 1
            alike[state["step"]].append(p)
        for t, ps in alike.items():
            grads = [p.grad for p in ps]
            averages = [self.state[p]["average"] for p in ps]
            squares = [self.state[p]["square"] for p in ps]
            torch._foreach_mul_(ps, 1 - lr * decay)
            torch._foreach_lerp_(averages, grads, 1 - beta1)
            torch._foreach_mul_(squares, beta2)
            torch._foreach_addcmul_(squares, grads, grads, value=1 - beta2)
            denominators = torch._foreach_sqrt(squares)
            torch._foreach_div_(denominators, math.sqrt(1 - beta2**t))
            torch._foreach_add_(denominators, eps)
            torch._foreach_addcdiv_(ps, averages, denominators, value=-lr / (1 - beta1**t))


def _orthogonalize(matrices: list[Tensor], steps: int) -> list[Tensor]:
    """Batched bfloat16 quintic Newton-Schulz orthogonalization across heterogeneously shaped 2D matrices.

    Transposes tall matrices to be wide and zero-pads all matrices to the batch max ``(rows, cols)`` so all run in
    one batched ``bmm``/``baddbmm`` loop without altering non-padded entries.
    """
    a, b, c = _NEWTON_SCHULZ
    wide = [m.mT if m.shape[0] > m.shape[1] else m for m in matrices]
    shapes = tuple(tuple(w.shape) for w in wide)
    rows, columns = max(r for r, _ in shapes), max(cols for _, cols in shapes)
    slots = _slots(shapes, rows, columns, wide[0].device)  # Flat scatter/gather indices into padded batch
    flat = torch.cat([w.reshape(-1) for w in wide]).bfloat16()
    x = flat.new_zeros(len(wide) * rows * columns).index_copy_(0, slots, flat).view(len(wide), rows, columns)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)  # Ensure spectral norm <= 1
    for _ in range(steps):  # x <- a*x + (b*G + c*G^2)*x, G = x @ x^T
        gram = x @ x.mT
        x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    pieces = x.view(-1)[slots].split([r * cols for r, cols in shapes])
    return [
        piece.view(w.shape).mT if m.shape[0] > m.shape[1] else piece.view(w.shape)
        for piece, w, m in zip(pieces, wide, matrices, strict=True)
    ]


@functools.lru_cache(maxsize=16)
def _slots(shapes: tuple[tuple[int, int], ...], rows: int, columns: int, device: torch.device) -> Tensor:
    """Cached flat destination indices packing matrices of ``shapes`` into the top-left of ``[N, rows, columns]``."""
    slots = [
        (k * rows * columns + torch.arange(r)[:, None] * columns + torch.arange(c)).flatten()
        for k, (r, c) in enumerate(shapes)
    ]
    return torch.cat(slots).to(device)
