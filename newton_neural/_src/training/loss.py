# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Windowed one-step training objectives (:class:`LossBase`, :class:`LossOneStep`) with optional state noise."""

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from ..core.compilation import with_own_code
from ..core.duration import same_duration
from ..core.precision import tensor_float_products
from ..core.signal import Frames
from ..data.windows import TrajectoryBatch
from ..models.nerd import NeRDBase

_VALUE_TOLERANCES = {False: 1e-5, True: 1e-3}
"""Max allowed relative difference between compiled and eager loss on checked calls (floor 1.0), by whether
float32 matrix products run in TF32. TF32 rounds a product to about three digits, and compiled code multiplies in
another order than eager code, by an amount that depends on the GPU."""


class ErrorKind(StrEnum):
    """Per-element error metric applied to normalized predictions."""

    ABSOLUTE = "absolute"
    """L1 absolute error (default; less sensitive to heavy-tailed contact impulses)."""

    SQUARED = "squared"
    """L2 squared error."""


class LossBase(abc.ABC):
    """Stateless callable training objective mapping ``(nerd, batch)`` to a scalar loss.

    ``nerd`` is the model, or its ``torch.nn.parallel.DistributedDataParallel`` wrapper when several processes
    train it. A loss runs the forward pass through the wrapper, which is what averages the gradients across the
    processes, and reads everything else from the wrapped model (``nerd.module``).
    """

    @abc.abstractmethod
    def __call__(self, nerd: NeRDBase | DistributedDataParallel, batch: TrajectoryBatch) -> Tensor:
        """Compute scalar training loss with gradients w.r.t. ``nerd`` parameters."""


def _unwrapped(nerd: NeRDBase | DistributedDataParallel) -> NeRDBase:
    """Return the model itself, whether ``nerd`` is the model or its ``DistributedDataParallel`` wrapper."""
    return nerd.module if isinstance(nerd, DistributedDataParallel) else nerd


class LossOneStep(LossBase):
    """Mean normalized one-step prediction error over valid frames in a trajectory window.

    Scores frames from ``receptive_field - 1`` onward (where full temporal context lies inside the window), plus
    all frames of episode-start windows. For unbounded-history models (``receptive_field is None``), only
    episode-start windows are scored; sample batches with ``windows.loader(batch_size, weights=windows.episode_start)``.

    With ``noise > 0`` (integrating predictions only), perturbs each input state by Gaussian noise scaled by
    ``noise * output_normalizer.std``, recomputes recomputable contact inputs on the perturbed states, and
    supervises against the increment from the perturbed state to the recorded next state.

    To train with several processes, pass the model's ``DistributedDataParallel`` wrapper in every training step
    and the model itself wherever only one process evaluates.

    Args:
        error: :class:`ErrorKind` or string (``"absolute"`` or ``"squared"``).
        noise: Input state-noise scale in units of one-frame standard deviation (must be ``>= 0``).
    """

    def __init__(self, *, error: ErrorKind | str = ErrorKind.ABSOLUTE, noise: float = 0.0) -> None:
        if noise < 0.0:
            raise ValueError(f"noise is a scale and cannot be negative, got {noise}")
        self.error = ErrorKind(error)
        self.noise = noise
        self._compiled: Callable[..., Tensor] | None = None
        # Tracks whether grad-enabled (True) and no-grad (False) compiled paths still await eager verification.
        self._unchecked: set[bool] = set()

    def __call__(self, nerd: NeRDBase | DistributedDataParallel, batch: TrajectoryBatch) -> Tensor:
        """Evaluate mean normalized error over valid frames and output channels."""
        module, nerd = nerd, _unwrapped(nerd)
        if not nerd.fitted:
            raise ValueError(
                "the normalization of the model is not fitted, so features and labels would reach the network "
                "unscaled. Call fit_normalization(windows.loader(batch_size)) on the model once before training"
            )
        if nerd.frame_dt is not None and not same_duration(batch.frame_dt, nerd.frame_dt):
            raise ValueError(
                f"the model steps frames of {nerd.frame_dt} s, and the batch holds frames of {batch.frame_dt} s. "
                "Train a model on data of its own frame duration"
            )
        current, following = nerd.training_frames(batch.frames)
        frame_count = next(iter(current.values())).shape[1]
        field = nerd.receptive_field
        if field is None and not bool(batch.episode_start.any()):
            raise ValueError(
                "the model has unbounded history, so only windows that start an episode train it, and no window "
                "of the batch does. Draw the batches with windows.loader(batch_size, weights=windows.episode_start)"
            )
        # Skip GPU->CPU sync on episode_start when window length already covers the receptive field.
        if field is not None and frame_count < field and not bool(batch.episode_start.all()):
            length = field if following is not None else field - 1
            raise ValueError(
                f"the windows hold {frame_count} frames to predict for and one prediction reads {field} frames, "
                "so only windows that start an episode would train the model. Use "
                f"TrajectoryWindows(dataset, length={length}) or longer"
            )
        if self.noise > 0.0:
            current = self._perturbed(nerd, current)
        arguments = (module, current, following, batch.episode_start)
        if self._compiled is None:
            return self._window_loss(*arguments)
        device_type = batch.episode_start.device.type
        autocast = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None
        loss = self._compiled(self, autocast, *arguments)
        # Verify the first grad and no-grad calls outside autocast (16-bit rounding differs between eager/compiled).
        if loss.requires_grad in self._unchecked and autocast is None:
            self._check(loss, *arguments)
            self._unchecked.remove(loss.requires_grad)
        return loss

    def compile(self, *, check: bool = True, **options: Any) -> None:
        """Compile the core tensor loss with ``torch.compile`` (noise injection and contact FK stay eager).

        When ``check=True``, compares the first grad and first no-grad non-autocast calls against eager evaluation
        and raises ``RuntimeError`` on discrepancy (pass ``check=False`` for stochastic models like dropout). The
        values may differ by ``1e-5`` of the eager loss or of one, whichever is greater, and by ``1e-3`` where
        float32 matrix products run in TF32.

        Args:
            check: Verify first compiled calls against eager loss.
            **options: Keyword arguments forwarded to ``torch.compile``.
        """
        self._compiled = torch.compile(with_own_code(type(self)._window_loss_under), **options)
        self._unchecked = {True, False} if check else set()

    def _perturbed(self, nerd: NeRDBase, current: Frames) -> Frames:
        """Perturb input states with Gaussian noise and re-evaluate recomputable contact inputs."""
        if not nerd.prediction.integrates:
            raise ValueError(
                f"the labels of {type(nerd.prediction).__name__} are recorded for the recorded states and do "
                "not hold for perturbed ones. Train this model with noise=0"
            )
        scale = self.noise * nerd.output_normalizer.std
        lead = next(iter(current.values())).shape[:2]
        noise = scale * torch.randn(*lead, nerd.system.increment_count, device=scale.device)
        current = {**current, **nerd.system.perturb(current, noise)}
        contact_input = nerd.contact_input
        if contact_input is not None and contact_input.recomputable:
            current.update(contact_input.evaluate(nerd.system, current))
        return current

    def _check(self, compiled: Tensor, *arguments: Any) -> None:
        """Raise ``RuntimeError`` if ``compiled`` deviates from eager ``_window_loss`` beyond ``_VALUE_TOLERANCES``."""
        tolerance = _VALUE_TOLERANCES[tensor_float_products(compiled.device)]
        with torch.no_grad():
            # On the model itself, since a forward of the wrapper exchanges buffers with the other processes.
            eager = self._window_loss(_unwrapped(arguments[0]), *arguments[1:])
            error = ((compiled - eager).abs() / eager.abs().clamp_min(1.0)).item()
        if not error <= tolerance:  # Fails on NaN too
            raise RuntimeError(
                f"the compiled loss is {compiled.item():.6g} and the eager loss of the same frames is "
                f"{eager.item():.6g}. They differ by {error:.3g} of the eager loss or of one, whichever is "
                f"greater, more than {tolerance:.3g}. The frames or the weights are not finite, the "
                "compiler is at fault, or the model draws random numbers at every call, as dropout does. "
                "Compile the loss of such a model with compile(check=False)"
            )

    def _window_loss_under(self, autocast: torch.dtype | None, *arguments: Any) -> Tensor:
        """Compiled wrapper receiving explicit ``autocast`` dtype so Dynamo specializes per autocast mode."""
        with torch.autocast(arguments[-1].device.type, dtype=autocast, enabled=autocast is not None):
            return self._window_loss(*arguments)

    def _window_loss(
        self,
        module: NeRDBase | DistributedDataParallel,
        current: Frames,
        following: Frames | None,
        episode_start: Tensor,
    ) -> Tensor:
        """Mean error of the frames that count, with the forward run through ``module``, the model or its wrapper."""
        nerd = _unwrapped(module)
        difference = module(current) - nerd.labels(current, following)
        error = difference.abs() if self.error is ErrorKind.ABSOLUTE else difference.square()
        error = error.mean(dim=-1)
        frame_count = error.shape[1]
        # Unbounded history (receptive_field=None) scores episode-start windows only.
        history_frames = frame_count if nerd.receptive_field is None else nerd.receptive_field - 1
        complete = torch.arange(frame_count, device=error.device) >= history_frames
        counts = (complete | episode_start[:, None]).to(error.dtype)
        # Denominator is guaranteed > 0 by validation in __call__.
        return (error * counts).sum() / counts.sum()
