# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""``FlashNeRD``: forward-dynamics model family taking externally sensed contact as input."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from torch import Tensor

from ..backbones.backbone import BackboneBase
from ..backbones.flash import BackboneFlash
from ..contacts.contact_input import ContactInputBase
from ..core.normalizer import FEATURE_LIMIT, Normalizer
from ..core.registry import Part
from ..core.signal import Frames
from ..predictions.prediction import PredictionBase
from ..systems.system import SystemBase
from .flashnerd_network import ContactInjection, FlashNeRDNetwork
from .nerd import NeRDBase, check_config


class FlashNeRD(NeRDBase):
    """Learned forward-dynamics model family with external contact sensing and a streaming temporal backbone.

    Encodes externally sensed contact signals (:class:`~newton_neural.contacts.ContactInputProbes` or
    :class:`~newton_neural.contacts.ContactInputSet`) and injects the contact code into :class:`FlashNeRDNetwork` at
    the sites selected by ``contact_injections``. Predicts either one-frame state increments
    (:class:`~newton_neural.predictions.PredictionStateIncrement` for :class:`~newton_neural.solvers.SolverNeRD`) or
    pre-contact free accelerations (:class:`~newton_neural.predictions.PredictionFreeMotion` for
    :class:`~newton_neural.solvers.SolverNeRDMuJoCo`).

    Under ``torch.autocast``, linear layers compute in reduced precision while feature extraction, normalization,
    attention scores, residual streams, outputs, and integration stay in FP32. See :class:`NeRDBase` for inherited
    attributes.

    Args:
        system: Simulated system specification.
        backbone: Temporal trunk (defaults to :class:`~newton_neural.backbones.BackboneFlash`).
        contact_input: Optional external contact sensing and encoding configuration.
        contact_injections: Network injection points (:class:`ContactInjection` or strings; defaults to all three
            when ``contact_input`` is provided). Must be omitted when ``contact_input is None``.
        prediction: Prediction target (defaults to :class:`~newton_neural.predictions.PredictionStateIncrement`).
        frame_dt: Frame duration [s] (inferred during :meth:`fit_normalization` if omitted).
        readout_hidden_dim: Hidden layer width of the readout MLP.

    Attributes:
        contact_encoder: Contact encoder module created from ``contact_input``, or ``None``.
        network: Underlying :class:`FlashNeRDNetwork`.
        feature_normalizer: Running :class:`Normalizer` for system features.
    """

    part_kinds: ClassVar[dict[str, type[Part]]] = {**NeRDBase.part_kinds, "backbone": BackboneBase}
    """Base part kinds plus ``"backbone"`` for checkpoint serialization.

    :meta hide-value:
    """

    def __init__(
        self,
        system: SystemBase,
        *,
        backbone: BackboneBase | None = None,
        contact_input: ContactInputBase | None = None,
        contact_injections: Sequence[ContactInjection | str] | None = None,
        prediction: PredictionBase | None = None,
        frame_dt: float | None = None,
        readout_hidden_dim: int = 256,
    ) -> None:
        super().__init__(system, contact_input=contact_input, prediction=prediction, frame_dt=frame_dt)
        injections = _injection_points(contact_input, contact_injections)
        self.contact_encoder = None if contact_input is None else contact_input.encoder()
        self.network = FlashNeRDNetwork(
            system.feature_count,
            self.prediction.output_count(system),
            BackboneFlash() if backbone is None else backbone,
            code_count=0 if contact_input is None else contact_input.code_count,
            injections=injections,
            gated=contact_input is not None and contact_input.gated,
            readout_hidden_dim=readout_hidden_dim,
        )
        self.feature_normalizer = Normalizer(system.feature_count)
        check_config(self.backbone)

    @property
    def backbone(self) -> BackboneBase:
        """Temporal backbone module."""
        return self.network.backbone

    @property
    def contact_injections(self) -> tuple[ContactInjection, ...]:
        """Active :class:`ContactInjection` sites in configured order (empty if ``contact_input is None``)."""
        injection = self.network.injection
        return () if injection is None else injection.points

    @property
    def receptive_field(self) -> int | None:
        return self.network.receptive_field

    def fit_inputs(self, frames: Frames) -> None:
        """Update feature and contact-encoder normalizers from ``frames``."""
        self.feature_normalizer.update(self.system.features(frames))
        if self.contact_encoder is not None:
            self.contact_encoder.fit_inputs(frames)

    def inputs(self, frames: Frames) -> tuple[Tensor, Tensor | None]:
        """Return ``(normalized_features, contact_code)`` (features clamped to ``[-10, 10]``; code may be ``None``)."""
        features = self.feature_normalizer.normalize(self.system.features(frames), limit=FEATURE_LIMIT)
        code = None if self.contact_encoder is None else self.contact_encoder(frames)
        return features, code

    def forward(self, frames: Frames) -> Tensor:
        features, code = self.inputs(frames)
        return self.network(features, code)

    def predict(self, frames: Frames, history: dict[str, Tensor]) -> Tensor:
        features, code = self.inputs(frames)
        return self.output_normalizer.denormalize(self.network.step(features, history, code))

    def history(self, world_count: int) -> dict[str, Tensor]:
        return self.network.history(world_count)

    def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
        self.network.reset_history(history, world_mask)

    def settings(self) -> dict[str, Any]:
        """Return ``contact_injections`` and ``readout_hidden_dim`` for checkpoint serialization."""
        return {
            # Omit contact_injections (None) when no contact input is configured.
            "contact_injections": [point.value for point in self.contact_injections] or None,
            "readout_hidden_dim": self.network.readout_hidden.out_features,
        }


def _injection_points(
    contact_input: ContactInputBase | None, contact_injections: Sequence[ContactInjection | str] | None
) -> tuple[ContactInjection, ...]:
    """Validate and normalize ``contact_injections`` against ``contact_input``."""
    places = [point.value for point in ContactInjection]
    if contact_injections is None:
        return () if contact_input is None else tuple(ContactInjection)
    if isinstance(contact_injections, str):
        # Reject bare strings so they are not silently iterated character by character.
        raise ValueError(
            f"contact_injections is the string {str(contact_injections)!r}, and it takes a sequence of places. "
            f"Pass a list or a tuple of one or more of {places}"
        )
    if contact_input is None:
        raise ValueError(
            f"contact_injections is {[str(point) for point in contact_injections]}, the places where the network "
            "adds the code of a contact input, and the model has no contact input. Pass contact_input as well, or "
            "leave contact_injections out"
        )
    if not contact_injections:
        raise ValueError(
            "contact_injections is empty, so the code of the contact input would enter the network nowhere. "
            f"Pass at least one of {places}, or leave it out for all of them"
        )
    unknown = [point for point in contact_injections if point not in places]
    if unknown:
        raise ValueError(
            f"contact_injections names {unknown}, and the network adds the contact code at {places}. Pass a "
            "sequence of ContactInjection values or of their strings"
        )
    points = tuple(ContactInjection(point) for point in contact_injections)
    if len(set(points)) != len(points):
        raise ValueError(
            f"contact_injections names a place twice, {[point.value for point in points]}. Pass each place once"
        )
    return points
