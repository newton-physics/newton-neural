# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal external-style model family, recurrent backbone, and contact input built strictly on public APIs."""

from __future__ import annotations

from typing import Any

import newton
import torch
import warp as wp
from torch import Tensor, nn

from newton_neural import Frames, Normalizer, SignalRole, SignalSpec
from newton_neural.backbones import BackboneBase
from newton_neural.contacts import ContactInputBase, ContactSensorBase
from newton_neural.models import NeRDBase
from newton_neural.predictions import PredictionBase
from newton_neural.systems import BindingArticulation, SystemBase


class MinimalNeRD(NeRDBase):
    """Minimal stateless linear model family (intentionally uses different attribute names than ``FlashNeRD``)."""

    receptive_field = 1

    def __init__(
        self,
        system: SystemBase,
        *,
        contact_input: ContactInputBase | None = None,
        prediction: PredictionBase | None = None,
        frame_dt: float | None = None,
    ) -> None:
        super().__init__(system, contact_input=contact_input, prediction=prediction, frame_dt=frame_dt)
        self.normalizer = Normalizer(system.feature_count)
        self.encoder = None if contact_input is None else contact_input.encoder()
        code_count = 0 if contact_input is None else contact_input.code_count
        self.linear = nn.Linear(system.feature_count + code_count, self.prediction.output_count(system))

    def fit_inputs(self, frames: Frames) -> None:
        self.normalizer.update(self.system.features(frames))
        if self.encoder is not None:
            self.encoder.fit_inputs(frames)

    def forward(self, frames: Frames) -> Tensor:
        inputs = self.normalizer.normalize(self.system.features(frames))
        if self.encoder is not None:
            inputs = torch.cat((inputs, self.encoder(frames)), dim=-1)
        return self.linear(inputs)

    def predict(self, frames: Frames, history: dict[str, Tensor]) -> Tensor:
        return self.output_normalizer.denormalize(self(frames))

    def history(self, world_count: int) -> dict[str, Tensor]:
        return {}

    def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
        pass

    def settings(self) -> dict[str, Any]:
        return {}


class BackboneLeaky(BackboneBase):
    """Minimal recurrent leaky-integrator backbone with unbounded history (``receptive_field = None``)."""

    receptive_field = None

    def __init__(self, *, width: int = 8, leak: float = 0.5) -> None:
        super().__init__()
        self.width = width
        self.leak = leak
        self.inp = nn.Linear(width, width)

    def forward(self, h: Tensor) -> Tensor:
        state = torch.zeros_like(h[:, 0])
        states = []
        for frame in range(h.shape[1]):
            state = self.leak * state + self.inp(h[:, frame])
            states.append(state)
        return torch.stack(states, dim=1)

    def step(self, h: Tensor, history: dict[str, Tensor]) -> Tensor:
        state = self.leak * history["state"] + self.inp(h)
        history["state"].copy_(state)
        return state

    def history(self, world_count: int, *, device: torch.device | str, dtype: torch.dtype) -> dict[str, Tensor]:
        return {"state": torch.zeros(world_count, self.width, device=device, dtype=dtype)}

    def config(self) -> dict[str, Any]:
        return {"width": self.width, "leak": self.leak}


class ContactInputRootHeight(ContactInputBase):
    """Minimal recomputable contact input reading floating-base height ``root_height`` ``[1]`` [m] above ``z = 0``."""

    recomputable = True

    def __init__(self, code_count: int = 4) -> None:
        self.code_count = code_count
        self.signals = {"root_height": SignalSpec("root_height", (1,), SignalRole.CONTACT, unit="m")}

    def sensor(self, model: newton.Model, binding: BindingArticulation) -> ContactSensorBase:
        return _RootHeightSensor(binding)

    def evaluate(self, system: SystemBase, frames: Frames) -> Frames:
        # Floating-base free joint has identity local frames, so joint_q[..., 2:3] is world root height.
        return {"root_height": frames["joint_q"][..., 2:3]}

    def encoder(self) -> nn.Module:
        return _RootHeightEncoder(self.code_count)

    def config(self) -> dict[str, Any]:
        return {"code_count": self.code_count}


class ContactInputRootHeightRecorded(ContactInputRootHeight):
    """Non-recomputable variant of :class:`ContactInputRootHeight`."""

    recomputable = False


class _RootHeightSensor(ContactSensorBase):
    def __init__(self, binding: BindingArticulation) -> None:
        self._root_bodies = wp.to_torch(binding.body_ids)[:, 0].long()  # Base is body 0
        self._height = torch.zeros(binding.world_count, 1, device=self._root_bodies.device)

    def update(self, state: newton.State, contacts: newton.Contacts | None) -> Frames:
        self._height.copy_(wp.to_torch(state.body_q)[self._root_bodies, 2:3])
        return {"root_height": self._height}


class _RootHeightEncoder(nn.Module):
    def __init__(self, code_count: int) -> None:
        super().__init__()
        self.normalizer = Normalizer(1)
        self.project = nn.Linear(1, code_count)

    def forward(self, frames: Frames) -> Tensor:
        return self.project(self.normalizer.normalize(frames["root_height"]))

    def fit_inputs(self, frames: Frames) -> None:
        self.normalizer.update(frames["root_height"])
