.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.training
======================

Training objectives and optimizers for PyTorch training loops.

``LossOneStep`` computes normalized one-step prediction error over trajectory windows with optional state-noise
injection and ``torch.compile`` support. ``create_optimizer`` builds either hybrid Muon + AdamW (``OptimizerMuon``)
or plain AdamW. Subclass ``LossBase`` to implement custom training objectives.

.. experimental::

.. py:module:: newton_neural.training
.. currentmodule:: newton_neural.training

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ErrorKind
   LossBase
   LossOneStep
   OptimizerMuon

.. rubric:: Functions

.. autosummary::
   :toctree: _generated

   create_optimizer
