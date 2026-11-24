# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Family-agnostic model base class (:class:`NeRDBase`) used by all recorders, losses, steppers, and solvers."""

from __future__ import annotations

import abc
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

import torch
from torch import Tensor, nn

from ..contacts.contact_input import ContactInputBase
from ..core import checkpoint
from ..core.duration import same_duration
from ..core.normalizer import Normalizer
from ..core.precision import Precision
from ..core.registry import Part, Registered
from ..core.signal import Frames, SignalRole, SignalSpec
from ..data.source import SignalSource
from ..data.windows import TrajectoryBatch
from ..predictions.prediction import PredictionBase, PredictionStateIncrement
from ..systems.system import SystemBase

if TYPE_CHECKING:
    from ..data.simulation import SimulationBase


class NeRDBase(nn.Module, Registered, abc.ABC):
    """Abstract base class for all neural robot dynamics model families.

    Composes a ``system``, a ``prediction``, and an optional ``contact_input`` while centralizing signal assembly,
    label extraction, output normalization, recorder sources, and checkpoint serialization. Subclasses implement
    windowed training (:meth:`forward`), streaming single-frame inference (:meth:`predict`), fixed-shape history
    buffers (:meth:`history`, :meth:`reset_history`), :attr:`receptive_field`, :meth:`fit_inputs`, and
    :meth:`settings`. Verify streaming/window parity with :func:`newton_neural.testing.check_nerd_streaming`.

    Args:
        system: Simulated system specification.
        contact_input: Optional external contact sensing and encoding configuration.
        prediction: Output prediction target (defaults to :class:`PredictionStateIncrement`).
        frame_dt: Frame duration [s] (inferred from data in :meth:`fit_normalization` if omitted).

    Attributes:
        system: Bound :class:`SystemBase`.
        contact_input: Optional :class:`ContactInputBase`.
        prediction: :class:`PredictionBase`.
        output_normalizer: Running :class:`Normalizer` for physical-unit prediction labels.
        frame_dt: Frame duration [s], or ``None`` until fitted.
        precision: Default deployment :class:`Precision` saved in checkpoints (``Precision.TF32`` initially).
        metadata: User-defined JSON-compatible dict persisted in checkpoints.
        signals: Combined dataset signal specs (system, contact input, then prediction labels).
    """

    registry: ClassVar[dict[str, type]] = {}

    part_kinds: ClassVar[dict[str, type[Part]]] = {
        "system": SystemBase,
        "contact_input": ContactInputBase,
        "prediction": PredictionBase,
    }
    """Mapping from part attribute/kwarg name to its registry base class; families extend this for extra parts.

    :meta hide-value:
    """

    def __init__(
        self,
        system: SystemBase,
        *,
        contact_input: ContactInputBase | None = None,
        prediction: PredictionBase | None = None,
        frame_dt: float | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.contact_input = contact_input
        self.prediction = PredictionStateIncrement() if prediction is None else prediction
        self.frame_dt = frame_dt
        self.precision = Precision.TF32
        self.metadata: dict[str, Any] = {}

        contact_signals = {} if contact_input is None else contact_input.signals
        groups = (system.signals, contact_signals, self.prediction.signals(system))
        self.signals: dict[str, SignalSpec] = {name: spec for group in groups for name, spec in group.items()}
        if len(self.signals) != sum(len(group) for group in groups):
            raise ValueError("the system, the contact input and the prediction must name their signals differently")

        self.output_normalizer = Normalizer(self.prediction.output_count(system))
        for part in (system, contact_input, self.prediction):
            if part is not None:
                check_config(part)

    @property
    def fitted(self) -> bool:
        """``True`` if normalizers were fitted via :meth:`fit_normalization` or loaded from a file (no GPU sync)."""
        return self.output_normalizer.fitted

    @property
    @abc.abstractmethod
    def receptive_field(self) -> int | None:
        """Frames (including current) visible to one prediction, or ``None`` for unbounded history."""

    @property
    def device(self) -> torch.device:
        """PyTorch device holding the model buffers."""
        return self.output_normalizer.mean.device

    @torch.no_grad()
    def fit_normalization(self, batches: Iterable[TrajectoryBatch], *, batch_limit: int = 64) -> None:
        """Fit input and output normalizer statistics from up to ``batch_limit`` batches and infer :attr:`frame_dt`.

        Raises:
            ValueError: If batch ``frame_dt`` conflicts with :attr:`frame_dt` or windows are too short.
        """
        for batch in itertools.islice(batches, batch_limit):
            if self.frame_dt is None:
                self.frame_dt = batch.frame_dt
            elif not same_duration(batch.frame_dt, self.frame_dt):
                raise ValueError(f"the model steps frames of {self.frame_dt} s, the data holds {batch.frame_dt} s")
            current, following = self.training_frames(batch.frames)
            self.output_normalizer.update(self.prediction.label(self.system, current, following))
            self.fit_inputs(current)

    @abc.abstractmethod
    def fit_inputs(self, frames: Frames) -> None:
        """Update family-specific input/encoder normalizers from one batch of training frames ``[B, T, *shape]``."""

    @abc.abstractmethod
    def forward(self, frames: Frames) -> Tensor:
        """Predict normalized outputs ``[B, T, output_count]`` over training windows (frame 0 = episode start)."""

    def training_frames(self, frames: Frames) -> tuple[Frames, Frames | None]:
        """Split windows into ``(current, following)`` pairs if ``prediction.integrates``, else ``(frames, None)``."""
        if not self.prediction.integrates:
            return frames, None
        if next(iter(frames.values())).shape[1] < 2:
            raise ValueError(
                f"{type(self.prediction).__name__} is learned from the frame after each frame, and the windows "
                "hold a single frame. Use windows of at least one step"
            )
        current = {name: value[:, :-1] for name, value in frames.items()}
        following = {name: value[:, 1:] for name, value in frames.items()}
        return current, following

    def labels(self, current: Frames, following: Frames | None) -> Tensor:
        """Return normalized supervision targets ``[..., output_count]``."""
        return self.output_normalizer.normalize(self.prediction.label(self.system, current, following))

    @abc.abstractmethod
    def predict(self, frames: Frames, history: dict[str, Tensor]) -> Tensor:
        """Predict one frame ``[B, output_count]`` in physical units and update ``history`` in place (graph-safe)."""

    def step(self, frames: Frames, history: dict[str, Tensor]) -> Frames:
        """Eager, differentiable single-frame step returning next ``STATE`` signals (for integrating predictions).

        ``history`` is updated in place. To differentiate through several steps, pass a clone of it at every step
        (``{name: value.clone() for name, value in history.items()}``), since autograd needs the history that each
        step was computed from.
        """
        require_next_state(self.prediction)
        return self.system.integrate(frames, self.predict(frames, history))

    @abc.abstractmethod
    def history(self, world_count: int) -> dict[str, Tensor]:
        """Allocate zero-initialized per-world FP32 history tensors on :attr:`device` (``{}`` if stateless)."""

    @abc.abstractmethod
    def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
        """Reset history buffers in place for worlds where ``world_mask`` ``[world_count]`` (bool) is ``True``."""

    def sources(self, simulation: SimulationBase) -> list[SignalSource]:
        """Build the :class:`SignalSource` list needed to record a dataset for this model from ``simulation``.

        Captures ``STATE`` and ``CONTACT`` signals pre-step, and ``CONTROL`` and ``LABEL`` signals post-step
        (all cloned so subsequent steps cannot overwrite recorded buffers).
        """
        binding, signals = simulation.binding, self.system.signals
        contact_input = self.contact_input
        if contact_input is not None and contact_input.reads_contacts and simulation.contacts is None:
            raise ValueError(
                f"{type(contact_input).__name__} reads the contacts of the collision detector, and the simulation "
                "detects none. Record from a simulation that runs Newton's collision detection, such as "
                "SimulationNewton(..., collision_pipeline=newton.CollisionPipeline(model))"
            )

        # Read simulation.state dynamically since ping-pong steps swap state objects.
        def read_state() -> Frames:
            return _clones(binding.read_state(simulation.state))

        def read_control() -> Frames:
            return _clones(binding.read_control(simulation.control))

        sources = [SignalSource(_with_role(signals, SignalRole.STATE), read_state)]
        if contact_input is not None:
            sensor = contact_input.sensor(simulation.model, binding)

            def read_contact() -> Frames:
                return _clones(sensor.update(simulation.state, simulation.contacts))

            sources.append(SignalSource(contact_input.signals, read_contact))
        sources.append(SignalSource(_with_role(signals, SignalRole.CONTROL), read_control, after_step=True))
        if self.prediction.signals(self.system):
            sources.append(self.prediction.source(simulation))
        return sources

    @abc.abstractmethod
    def settings(self) -> dict[str, Any]:
        """Return JSON-serializable family-specific constructor kwargs (excluding parts and ``frame_dt``)."""

    def save(self, path: str | Path) -> None:
        """Save weights, normalizers, part configs, settings, ``frame_dt``, ``precision``, and ``metadata``."""
        parts = {kind: getattr(self, kind) for kind in self.part_kinds}
        manifest = {
            "model": type(self).__name__,
            "parts": {
                kind: None if part is None else {"name": type(part).__name__, "config": part.config()}
                for kind, part in parts.items()
            },
            "settings": self.settings(),
            "frame_dt": self.frame_dt,
            "precision": Precision(self.precision).value,
            "signals": [spec.to_dict() for spec in self.signals.values()],
            "metadata": self.metadata,
        }
        checkpoint.write(path, manifest, self.state_dict())

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str | None = None) -> Self:
        """Load a checkpoint saved by :meth:`save` via class registries (``weights_only=True``).

        Calling ``NeRDBase.load`` instantiates whichever registered family wrote the file; calling ``<Family>.load``
        verifies the checkpoint matches ``<Family>``. Validates that reconstructed signal specs match the manifest.
        """
        manifest, state = checkpoint.read(path)
        family = NeRDBase.registered(manifest.get("model"))
        if not issubclass(family, cls):
            raise ValueError(
                f"{path} holds a {family.__name__}, not a {cls.__name__}. Load it with {family.__name__}.load, "
                "or with NeRDBase.load, which returns a model of whichever family wrote the file"
            )
        # Omit None parts so families that do not accept optional parts (e.g. contact_input) construct cleanly.
        parts = {
            kind: family.part_kinds[kind].create(entry["name"], entry["config"])
            for kind, entry in manifest["parts"].items()
            if entry is not None
        }
        nerd = family(parts.pop("system"), **parts, frame_dt=manifest["frame_dt"], **manifest["settings"])
        _check_signals(path, [spec.to_dict() for spec in nerd.signals.values()], manifest["signals"])
        nerd.load_state_dict(state)
        nerd.precision = Precision(manifest["precision"])
        nerd.metadata = manifest["metadata"]
        return nerd.to(device)


def require_next_state(prediction: PredictionBase) -> None:
    """Raise ``RuntimeError`` if ``prediction`` does not integrate into a next state."""
    if not prediction.integrates:
        raise RuntimeError(
            f"{type(prediction).__name__} does not give the next state. Call predict and let a physics solver "
            "integrate its result"
        )


def check_config(part: Part) -> None:
    """Validate eagerly that ``part.config()`` is JSON-serializable."""
    try:
        json.dumps(part.config())
    except TypeError as error:
        name = type(part).__name__
        raise TypeError(f"the config of {name} must be plain data that json can write, but {error}") from error


def _check_signals(path: str | Path, rebuilt: list[dict[str, Any]], saved: list[dict[str, Any]]) -> None:
    """Raise ``ValueError`` if reconstructed model signal specs differ from those saved in ``path``."""
    for index, (now, then) in enumerate(itertools.zip_longest(rebuilt, saved, fillvalue="absent")):
        if now != then:
            raise ValueError(
                f"{path} was written for a model whose signal {index} is {then}, and the parts rebuilt from the "
                f"file give {now}. The class of a part has changed since the file was written. Train and save "
                "the model again, or load the file with the classes that wrote it"
            )


def _with_role(signals: dict[str, SignalSpec], role: SignalRole) -> dict[str, SignalSpec]:
    return {name: spec for name, spec in signals.items() if spec.role is role}


def _clones(frames: Frames) -> Frames:
    return {name: value.clone() for name, value in frames.items()}
