# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compiled, precision-managed single-frame inference wrapper (:class:`NeRDStepper`)."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from ..core.compilation import with_own_code
from ..core.precision import SWITCH_LOCK, Float32Products, Precision
from ..core.signal import Frames
from .nerd import NeRDBase, require_next_state

_AUTOCAST_DTYPES = {Precision.FP16: torch.float16, Precision.BF16: torch.bfloat16}

_TOLERANCES = {Precision.FP32: 1e-4, Precision.TF32: 2e-2, Precision.FP16: 2e-2, Precision.BF16: 1e-1}
"""Max allowed relative error vs. eager IEEE FP32 on the first step (normalized output units, floor 1.0)."""


class NeRDStepper:
    """Runs compiled, no-grad single-frame inference for a fixed ``world_count`` on PyTorch tensors.

    Owns per-world history buffers, applies the configured :class:`Precision` (scoping CUDA's global TF32 flag
    during execution and storing 16-bit linear weights on a snapshot copy for ``fp16``/``bf16``), compiles via
    ``torch.compile(fullgraph=True, dynamic=False)`` on CUDA, and validates the first step against eager IEEE FP32.
    After warm-up, steps perform zero host syncs and are safe to capture inside CUDA graphs.

    Note: Compiled steps specialize to input shapes, strides, and ``inference_mode`` state; warm up with the same
    buffer layouts used inside any CUDA graph. Construct a new stepper if ``nerd`` weights are mutated afterward.

    Args:
        nerd: Model to deploy.
        world_count: Fixed batch/world count.
        precision: Override :class:`Precision` or lowercase string (defaults to ``nerd.precision``).
        compile: Enable ``torch.compile`` (defaults to ``True`` on CUDA, ``False`` on CPU). When using
            ``max_autotune``, call ``torch.set_float32_matmul_precision("high")`` before the first step.

    Attributes:
        nerd: Source model.
        world_count: Number of worlds.
        device: Model device.
        precision: Resolved :class:`Precision` (``"tf32"`` resolves to ``Precision.FP32`` on non-CUDA devices).
        compiled: Whether the step function was compiled with ``torch.compile``.
    """

    def __init__(
        self,
        nerd: NeRDBase,
        world_count: int,
        *,
        precision: Precision | str | None = None,
        compile: bool | None = None,
    ) -> None:
        self.nerd = nerd
        self.world_count = world_count
        self.device = nerd.device
        self.precision = Precision(nerd.precision if precision is None else precision)
        if self.precision is Precision.TF32 and self.device.type != "cuda":
            self.precision = Precision.FP32  # TF32 is CUDA-only
        self.compiled = self.device.type == "cuda" if compile is None else bool(compile)
        # Allocate history outside inference_mode so in-place updates work regardless of caller mode.
        with torch.inference_mode(False):
            self._history = nerd.history(world_count)
        self._checked = False

        dtype = _AUTOCAST_DTYPES.get(self.precision)
        deployed = nerd if dtype is None else _with_linear_dtype(nerd, dtype)
        device_type = self.device.type
        integrates = nerd.prediction.integrates

        def advance(frames: Frames, history: dict[str, Tensor]) -> tuple[Tensor, Frames]:
            with torch.autocast(device_type, dtype=dtype, enabled=dtype is not None):
                prediction = deployed.predict(frames, history)
            return prediction, deployed.system.integrate(frames, prediction) if integrates else {}

        if self.compiled:
            advance = torch.compile(with_own_code(advance), fullgraph=True, dynamic=False)
        self._advance = advance

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({type(self.nerd).__name__}, world_count={self.world_count}, "
            f"device={str(self.device)!r}, precision={self.precision.value!r}, compiled={self.compiled})"
        )

    def predict(self, frames: Frames) -> Tensor:
        """Predict physical-unit outputs ``[world_count, output_count]`` and advance history by one frame."""
        return self._run(frames)[0]

    def step(self, frames: Frames) -> Frames:
        """Advance all worlds by one frame and return next ``STATE`` signals (for integrating predictions)."""
        require_next_state(self.nerd.prediction)
        return self._run(frames)[1]

    @torch.no_grad()
    def _run(self, frames: Frames) -> tuple[Tensor, Frames]:
        """Run one step, verifying against eager IEEE FP32 on the first call."""
        if self._checked:
            return self._deployed(frames)
        before = self.export_history()
        prediction, following = self._deployed(frames)
        self._check(frames, before, prediction, following)
        self._checked = True
        return prediction, following

    def _deployed(self, frames: Frames) -> tuple[Tensor, Frames]:
        """Execute the deployed step with CUDA TF32 configured for :attr:`precision`."""
        if self.device.type != "cuda":
            return self._advance(frames, self._history)
        with SWITCH_LOCK, Float32Products(tensor_float=self.precision is not Precision.FP32):
            return self._advance(frames, self._history)

    def reset(self, world_mask: Tensor | None = None) -> None:
        """Reset history in place for masked worlds ``[world_count]`` bool (all worlds if ``None``)."""
        if world_mask is None:
            world_mask = torch.ones(self.world_count, dtype=torch.bool, device=self.device)
        self.nerd.reset_history(self._history, world_mask)

    def clone(self) -> NeRDStepper:
        """Return a shallow clone sharing weights and compiled step with an independent copy of history."""
        other = copy.copy(self)
        with torch.inference_mode(False):
            other._history = self.export_history()
        return other

    def export_history(self) -> dict[str, Tensor]:
        """Return a cloned snapshot of all per-world history tensors for later :meth:`import_history`."""
        return {name: value.clone() for name, value in self._history.items()}

    def import_history(self, history: dict[str, Tensor]) -> None:
        """Copy a previously exported ``history`` snapshot into this stepper in place."""
        expected = {name: tuple(value.shape) for name, value in self._history.items()}
        given = {name: tuple(value.shape) for name, value in history.items()}
        if given != expected:
            raise ValueError(f"the history of this stepper holds {expected}, got {given}")
        for name, value in self._history.items():
            value.copy_(history[name])

    def _check(self, frames: Frames, history: dict[str, Tensor], prediction: Tensor, following: Frames) -> None:
        """Validate deployed first-step output and integration against eager IEEE FP32."""
        nerd = self.nerd
        std = nerd.output_normalizer.std
        # Compute reference in strict IEEE FP32 regardless of caller autocast or global TF32 flag.
        ieee = Float32Products(tensor_float=False) if self.device.type == "cuda" else nullcontext()
        with SWITCH_LOCK, ieee, torch.autocast(self.device.type, enabled=False):
            expected = nerd.predict(frames, history) / std
        error = ((prediction / std - expected).abs().max() / expected.abs().max().clamp_min(1.0)).item()
        tolerance = _TOLERANCES[self.precision]
        if not error <= tolerance:  # Fails on NaN as well
            training = (
                ", the model is in training mode and holds layers such as dropout (call nerd.eval() first)"
                if nerd.training
                else ""
            )
            raise RuntimeError(
                f"the deployed step of the model differs from its eager step in IEEE float32 by {error:.3g} "
                f"of the normalized output, more than {tolerance:.3g}. The inputs are not finite, the model "
                f"changed after the stepper was created{training}, or the compiler or the "
                f"{self.precision.value} precision is at fault"
            )
        if nerd.prediction.integrates:
            # Both paths integrate the same FP32 prediction, so disagreement indicates a compiler fault.
            for name, value in nerd.system.integrate(frames, prediction).items():
                if not torch.allclose(following[name], value, rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f"the deployed step integrates {name} differently from the eager model")


def _with_linear_dtype(nerd: NeRDBase, dtype: torch.dtype) -> NeRDBase:
    """Deep-copy ``nerd`` with ``nn.Linear`` parameters pre-cast to ``dtype`` (avoids per-step autocast casts)."""
    # Share stateless contact_input to avoid trying to deepcopy cached Warp simulation replicas.
    deployed = copy.deepcopy(nerd, {id(nerd.contact_input): nerd.contact_input})
    for module in deployed.modules():
        if isinstance(module, nn.Linear):
            module.to(dtype)
    return deployed
