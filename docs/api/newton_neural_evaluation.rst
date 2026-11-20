.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.evaluation
========================

Open-loop rollout metrics and live simulation audits.

``rollout_error`` evaluates multi-step autoregressive drift on recorded trajectories. ``audit_rest`` measures drift
and detrended vibration when controls are held constant, and ``audit_paired`` compares live divergence between a
reference simulation and a neural candidate under shared commands.

.. experimental::

.. py:module:: newton_neural.evaluation
.. currentmodule:: newton_neural.evaluation

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   AuditPaired
   AuditRest
   RolloutStatistics

.. rubric:: Functions

.. autosummary::
   :toctree: _generated

   audit_paired
   audit_rest
   rollout_error
