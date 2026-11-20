.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.predictions
=========================

Prediction targets defining network outputs and supervision labels.

``PredictionStateIncrement`` (default) predicts full one-frame state changes for standalone deployment with
:class:`~newton_neural.solvers.SolverNeRD`. ``PredictionFreeMotion`` predicts unconstrained joint accelerations for
two-way contact coupling inside :class:`~newton_neural.solvers.SolverNeRDMuJoCo`. Subclass ``PredictionBase`` to
define custom prediction targets.

.. py:module:: newton_neural.predictions
.. currentmodule:: newton_neural.predictions

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   PredictionBase
   PredictionFreeMotion
   PredictionStateIncrement
