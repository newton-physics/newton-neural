# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prediction targets defining network outputs and supervision labels.

``PredictionStateIncrement`` (default) predicts full one-frame state changes for standalone deployment with
:class:`~newton_neural.solvers.SolverNeRD`. ``PredictionFreeMotion`` predicts unconstrained joint accelerations for
two-way contact coupling inside :class:`~newton_neural.solvers.SolverNeRDMuJoCo`. Subclass ``PredictionBase`` to
define custom prediction targets.

.. experimental::
"""

from ._src.predictions.prediction import PredictionBase, PredictionFreeMotion, PredictionStateIncrement

__all__ = ["PredictionBase", "PredictionFreeMotion", "PredictionStateIncrement"]
