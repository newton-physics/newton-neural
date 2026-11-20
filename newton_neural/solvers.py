# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop-in Newton solvers backed by trained neural dynamics models.

``SolverNeRD`` steps standalone state-increment models directly in place of a physics solver. ``SolverNeRDMuJoCo``
runs free-motion models inside MuJoCo Warp for two-way contact coupling with dynamic objects. Both solvers accept
either a model instance or a checkpoint path of any registered family.

``load_nerd`` and ``same_duration`` are helper utilities for custom solvers built around
:class:`~newton_neural.models.NeRDStepper`.

.. experimental::
"""

from ._src.core.duration import same_duration
from ._src.solvers.common import load_nerd
from ._src.solvers.solver_nerd import SolverNeRD
from ._src.solvers.solver_nerd_mujoco import SolverNeRDMuJoCo

__all__ = ["SolverNeRD", "SolverNeRDMuJoCo", "load_nerd", "same_duration"]
