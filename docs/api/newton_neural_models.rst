.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.models
====================

Model families, the ``NeRDBase`` interface, tensor-level stepping, and inference precision modes.

``FlashNeRD`` is the built-in forward-dynamics family, with ``ContactInjection`` selecting where contact codes enter
its network. ``NeRDBase`` defines the family-agnostic interface for loading checkpoints and implementing custom
families. ``NeRDStepper`` executes compiled single-frame inference directly on tensors, and ``Precision`` enumerates
deployment numeric modes (``fp32``, ``tf32``, ``fp16``, ``bf16``).

.. py:module:: newton_neural.models
.. currentmodule:: newton_neural.models

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ContactInjection
   FlashNeRD
   NeRDBase
   NeRDStepper
   Precision
