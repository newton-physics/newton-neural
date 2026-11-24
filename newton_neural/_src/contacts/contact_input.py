# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract interfaces for contact inputs (:class:`ContactInputBase`) and sensors (:class:`ContactSensorBase`).

A contact input splits contact handling into two components: an on-device sensor reading Newton state/contacts into
reused buffers (CUDA-graph safe), and a PyTorch encoder mapping those signals into a fixed-width contact code. Model
families decide independently where to inject the contact code into their networks.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, ClassVar

import newton
import warp as wp
from torch import nn

from ..core.registry import Part
from ..core.signal import Frames, SignalSpec

if TYPE_CHECKING:
    from ..systems.articulation_binding import BindingArticulation
    from ..systems.system import SystemBase


class ContactSensorBase(abc.ABC):
    """On-device contact sensor bound to a :class:`newton.Model`.

    Preallocates output buffers at construction so :meth:`update` performs zero host syncs and can be captured into
    CUDA graphs.
    """

    overflow: wp.array[wp.int32] | None = None
    """Optional per-world count of dropped contact rows ``(world_count,)`` exposed as ``CONTACT_ROW_OVERFLOW`` by
    :class:`~newton_neural.solvers.SolverNeRD`. ``None`` if unbounded."""

    @abc.abstractmethod
    def update(self, state: newton.State, contacts: newton.Contacts | None) -> Frames:
        """Sense contact signals for ``state`` into internal buffers and return ``[world_count, *shape]`` views.

        Returned views remain valid until the next :meth:`update` call.

        Args:
            state: Current Newton state (body transforms and velocities must be up to date).
            contacts: Detected contacts for ``state`` (required when ``reads_contacts=True``).
        """


class ContactInputBase(Part, abc.ABC):
    """Stateless configuration specifying contact signals, sensor creation, and PyTorch encoder creation.

    All trainable parameters and normalizer buffers live on the module returned by :meth:`encoder` (saved with the
    owning model). Subclasses set :attr:`signals`, :attr:`code_count`, and class flags (:attr:`recomputable`,
    :attr:`reads_contacts`, :attr:`gated`).
    """

    registry: ClassVar[dict[str, type]] = {}

    recomputable: ClassVar[bool] = False
    """``True`` if :meth:`evaluate` can recompute signals purely from joint state (for noisy training and rollouts)."""

    reads_contacts: ClassVar[bool] = False
    """``True`` if :meth:`sensor` requires collision-detected :class:`newton.Contacts` from the caller."""

    gated: ClassVar[bool] = False
    """Hint for model families (honored by :class:`~newton_neural.models.FlashNeRD`) to gate the contact code
    starting near zero when the encoder has randomly initialized trainable weights."""

    signals: dict[str, SignalSpec]
    """``CONTACT`` signals recorded and consumed by this contact input."""

    code_count: int
    """Feature width of the encoded contact vector."""

    def __repr__(self) -> str:
        scalars = (bool, int, float, str)
        entries = [f"{name}={value!r}" for name, value in self.config().items() if isinstance(value, scalars)]
        return f"{type(self).__name__}({', '.join(entries)})"

    @abc.abstractmethod
    def sensor(self, model: newton.Model, binding: BindingArticulation) -> ContactSensorBase:
        """Instantiate an on-device :class:`ContactSensorBase` for the robot bound in ``model``."""

    def evaluate(self, system: SystemBase, frames: Frames) -> Frames:
        """Recompute contact signals purely from ``STATE`` signals in ``frames`` (when ``recomputable=True``).

        Args:
            system: System specification matching ``frames``.
            frames: Signal dictionary containing ``STATE`` tensors with arbitrary leading dimensions.

        Returns:
            Freshly allocated contact signal tensors matching the leading dimensions and device of ``frames``.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot be evaluated from joint state alone")

    @abc.abstractmethod
    def encoder(self) -> nn.Module:
        """Create the PyTorch encoder mapping ``frames`` to contact codes ``[..., code_count]``.

        The returned module must implement ``forward(frames) -> Tensor`` (compile- and graph-safe, ignoring unrelated
        keys in ``frames``) and ``fit_inputs(frames) -> None`` to fit its input normalizer statistics.
        """
