# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model families, the ``NeRDBase`` interface, tensor-level stepping, and inference precision modes.

``FlashNeRD`` is the built-in forward-dynamics family, with ``ContactInjection`` selecting where contact codes enter
its network. ``NeRDBase`` defines the family-agnostic interface for loading checkpoints and implementing custom
families. ``NeRDStepper`` executes compiled single-frame inference directly on tensors, and ``Precision`` enumerates
deployment numeric modes (``fp32``, ``tf32``, ``fp16``, ``bf16``).
"""

from ._src.core.precision import Precision
from ._src.models.flashnerd import FlashNeRD
from ._src.models.flashnerd_network import ContactInjection
from ._src.models.nerd import NeRDBase
from ._src.models.stepper import NeRDStepper

__all__ = ["ContactInjection", "FlashNeRD", "NeRDBase", "NeRDStepper", "Precision"]
