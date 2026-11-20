# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Neural robot dynamics for the Newton physics engine.

``newton-neural`` trains learned dynamics models from a reference physics solver and runs them as drop-in Newton
solvers across thousands of parallel worlds.

This top-level module exports shared signal types and normalization utilities (:class:`SignalSpec`,
:class:`SignalRole`, :type:`Frames`, :class:`Normalizer`, and :class:`NewtonNeuralWarning`). The main workflow
submodules are :mod:`newton_neural.models`, :mod:`newton_neural.data`, :mod:`newton_neural.training`,
:mod:`newton_neural.evaluation`, and :mod:`newton_neural.solvers`.
"""

from . import (
    backbones,
    contacts,
    data,
    evaluation,
    models,
    predictions,
    solvers,
    systems,
    testing,
    training,
)
from ._src.core.normalizer import Normalizer
from ._src.core.signal import Frames, SignalRole, SignalSpec
from ._src.core.warnings import NewtonNeuralWarning
from ._version import __version__

__all__ = [
    "Frames",
    "NewtonNeuralWarning",
    "Normalizer",
    "SignalRole",
    "SignalSpec",
    "__version__",
    "backbones",
    "contacts",
    "data",
    "evaluation",
    "models",
    "predictions",
    "solvers",
    "systems",
    "testing",
    "training",
]
