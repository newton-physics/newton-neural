# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract interfaces for simulated systems (:class:`SystemBase`) and Newton bindings (:class:`BindingBase`).

:class:`SystemBase` encapsulates pure-PyTorch signal definitions, feature transforms, and state integration saved in
checkpoints. :class:`BindingBase` maps those signals onto Newton's Warp state/control arrays at simulation time,
keeping training and offline rollout evaluation independent of a live :class:`newton.Model`.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import newton
from torch import Tensor, nn

from ..core.registry import Part
from ..core.signal import Frames, SignalRole, SignalSpec


class BindingBase(abc.ABC):
    """Bridge mapping a :class:`SystemBase`'s per-world signals onto a :class:`newton.Model`.

    Returned by :meth:`SystemBase.bind`. Simulations, recorders, and live audits interact with Newton models solely
    through these methods. All returned tensors have shape ``[world_count, *shape]`` on the Newton model's device.

    Attributes:
        system: Bound system specification.
        world_count: Number of parallel worlds.
    """

    system: SystemBase
    world_count: int

    @abc.abstractmethod
    def read_state(self, state: newton.State) -> Frames:
        """Read all ``STATE`` signals from ``state``.

        Args:
            state: Source Newton state.

        Returns:
            Dictionary of ``STATE`` tensors ``[world_count, *shape]`` (may be zero-copy views updated in place by
            subsequent solver steps).
        """

    @abc.abstractmethod
    def read_control(self, control: newton.Control) -> Frames:
        """Read all ``CONTROL`` signals from ``control``.

        Args:
            control: Source Newton control.

        Returns:
            Dictionary of ``CONTROL`` tensors ``[world_count, *shape]`` (may be zero-copy views).
        """

    @abc.abstractmethod
    def write_state(self, state: newton.State, frames: Frames, world_mask: Tensor | None = None) -> None:
        """Write ``STATE`` signals into ``state`` without modifying unrelated arrays.

        Args:
            state: Target Newton state.
            frames: Dictionary containing all ``STATE`` signals ``[world_count, *shape]``.
            world_mask: Optional boolean mask ``[world_count]`` selecting worlds to overwrite (all worlds if ``None``).
        """

    @abc.abstractmethod
    def write_control(self, control: newton.Control, frames: Frames) -> None:
        """Write ``CONTROL`` signals into ``control``.

        Args:
            control: Target Newton control.
            frames: Dictionary containing all ``CONTROL`` signals ``[world_count, *shape]``.
        """

    @abc.abstractmethod
    def state_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Return finite ``(low, high)`` sampling bounds ``[*shape]`` for each ``STATE`` signal."""

    @abc.abstractmethod
    def command_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Return ``(low, high)`` sampling bounds ``[*shape]`` for each ``CONTROL`` signal (``+-inf`` if unbounded)."""

    def place(self, frames: Frames, worlds: Tensor) -> Frames:
        """Carry ``STATE`` signals drawn for the first world to the worlds they start.

        A sampler draws the states of one world, within :meth:`state_bounds`. Where the worlds of a model hold the
        system at different places of one scene, a binding overrides this method so that every world starts at
        its own place. The default returns ``frames`` as they are.

        Args:
            frames: ``STATE`` signals ``[len(worlds), *shape]``.
            worlds: Indices ``[len(worlds)]`` of the worlds that the rows of ``frames`` start.
        """
        return frames


class SystemBase(nn.Module, Part, abc.ABC):
    """Pure-PyTorch specification of a simulated system's signals, features, and state integration.

    Holds non-trainable buffers only and reconstructs from :meth:`config`. All frame methods accept either single
    frames ``[world_count, *shape]`` or windows ``[world_count, frame_count, *shape]`` (written ``...`` below) and
    ignore extra signal keys (such as contact signals).

    :meth:`features`, :meth:`increment`, and :meth:`integrate` execute inside compiled steps and losses: avoid
    data-dependent control flow/shapes as well as matmuls or linear layers (so they remain in IEEE FP32 under all
    deployment precisions). Implement :meth:`bind` to run on a :class:`newton.Model`.
    """

    registry: ClassVar[dict[str, type]] = {}

    signals: dict[str, SignalSpec]
    """Ordered ``STATE`` signals followed by ``CONTROL`` signals."""

    feature_count: int
    """Output width of :meth:`features`."""

    increment_count: int
    """Width of :meth:`increment` and :meth:`integrate`."""

    def bind(self, model: newton.Model, articulation: Any = "*") -> BindingBase:
        """Bind this system to a :class:`newton.Model`.

        Args:
            model: Target Newton model.
            articulation: Selector identifying the system in ``model`` (ignored if unused).

        Returns:
            A :class:`BindingBase` reading and writing signals on ``model``.

        Raises:
            NotImplementedError: If this system does not support Newton model binding.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no bind, so it cannot run on a Newton model. Implement bind, which returns "
            "a BindingBase for that Newton model"
        )

    @abc.abstractmethod
    def features(self, frames: Frames) -> Tensor:
        """Convert raw ``STATE`` and ``CONTROL`` signals into flat network features.

        Args:
            frames: Input signal dictionary.

        Returns:
            Feature tensor ``[..., feature_count]`` matching the input floating-point dtype.
        """

    @abc.abstractmethod
    def increment(self, current: Frames, following: Frames) -> Tensor:
        """Compute the state increment carrying ``current`` to ``following``.

        Args:
            current: ``STATE`` signals at time ``t``.
            following: ``STATE`` signals at time ``t + 1``.

        Returns:
            Increment tensor ``[..., increment_count]``.
        """

    @abc.abstractmethod
    def integrate(self, current: Frames, increment: Tensor) -> Frames:
        """Apply ``increment`` to ``current`` ``STATE`` signals (exact inverse of :meth:`increment`).

        Args:
            current: ``STATE`` signals at time ``t``.
            increment: Increment tensor ``[..., increment_count]``.

        Returns:
            ``STATE`` signals at time ``t + 1``.
        """

    def perturb(self, current: Frames, noise: Tensor) -> Frames:
        """Perturb ``current`` ``STATE`` signals by ``noise`` for noisy training (defaults to :meth:`integrate`).

        Args:
            current: ``STATE`` signals to perturb.
            noise: Perturbation vector ``[..., increment_count]``.

        Returns:
            Perturbed ``STATE`` signals.
        """
        return self.integrate(current, noise)

    def error(self, reference: Frames, other: Frames) -> Frames:
        """Compute non-negative per-coordinate errors between two ``STATE`` dictionaries.

        Defaults to ``|other - reference|``. Override for non-Euclidean coordinates such as quaternions.

        Args:
            reference: Reference ``STATE`` signals.
            other: Predicted ``STATE`` signals of matching shapes.

        Returns:
            Per-signal non-negative error tensors ``[..., coordinate_count]`` in each signal's SI unit.
        """
        return {
            name: (other[name] - reference[name]).abs()
            for name, spec in self.signals.items()
            if spec.role is SignalRole.STATE
        }
