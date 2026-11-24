# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Signal schemas and frame dictionaries exchanged between simulations, datasets, models, and solvers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

Frames = dict[str, Tensor]
"""Dictionary mapping signal names to tensors: ``[world_count, *shape]`` for one frame, or
``[world_count, frame_count, *shape]`` for a window of consecutive frames."""

_TORCH_DTYPES = {"float32": torch.float32, "int32": torch.int32, "bool": torch.bool}


class SignalRole(StrEnum):
    """Semantic role of a signal in a model."""

    STATE = "state"
    """Dynamic state variables evolved by the model (e.g., ``joint_q``, ``joint_qd``)."""

    CONTROL = "control"
    """Actuation inputs consumed by the solver (e.g., ``joint_target_q``, ``joint_f``)."""

    CONTACT = "contact"
    """Contact features produced by a contact input sensor."""

    LABEL = "label"
    """Supervision targets recorded from a reference solver (unused at deployment)."""


@dataclass(frozen=True)
class SignalSpec:
    """Schema describing one signal's name, per-world shape, role, dtype, and SI unit."""

    name: str
    """Signal name, a Python identifier (Newton-sourced signals keep Newton's attribute name, e.g., ``joint_q``).

    A dataset stores every signal in a file of this name.
    """

    shape: tuple[int, ...]
    """Shape for one world and one frame."""

    role: SignalRole
    """Role of the signal (:class:`SignalRole`)."""

    dtype: str = "float32"
    """Element dtype string: ``"float32"``, ``"int32"``, or ``"bool"``."""

    unit: str = ""
    """SI unit string (e.g., ``"m or rad"``), or empty if dimensionless."""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(
                f"signal name {self.name!r} is not an identifier. A dataset stores every signal in a file of its "
                "name, so a name holds letters, digits and underscores only"
            )
        if self.dtype not in _TORCH_DTYPES:
            raise ValueError(f"signal {self.name!r} has dtype {self.dtype!r}. Use one of {sorted(_TORCH_DTYPES)}")
        # Coerce JSON-deserialized lists and strings back to tuple[int, ...] and SignalRole.
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "role", SignalRole(self.role))

    @property
    def torch_dtype(self) -> torch.dtype:
        """Corresponding ``torch.dtype`` for :attr:`dtype`."""
        return _TORCH_DTYPES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "role": self.role.value,
            "dtype": self.dtype,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalSpec:
        """Reconstruct a :class:`SignalSpec` from :meth:`to_dict` output."""
        return cls(**data)
