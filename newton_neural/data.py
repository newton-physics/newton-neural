# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trajectory recording, sharded datasets, and GPU-resident training window loaders.

Wrap a Newton model and solver in ``SimulationNewton``, collect episodes using ``TrajectoryRecorder`` with state
samplers (``Sampler*``) and command generators (``Command*``), and load contiguous episode windows with
``TrajectoryWindows`` for training and evaluation. Use ``TrajectoryWriter`` to write datasets from custom simulation
loops.
"""

from ._src.data.commands import (
    CommandBase,
    CommandHeld,
    CommandNearState,
    CommandSmooth,
    CommandSweep,
    CommandSwitch,
)
from ._src.data.dataset import TrajectoryDataset, TrajectoryWriter
from ._src.data.recorder import TrajectoryRecorder
from ._src.data.samplers import (
    SamplerBase,
    SamplerFloatingBase,
    SamplerMixture,
    SamplerPerturb,
    SamplerSobol,
    SamplerUniform,
)
from ._src.data.simulation import SimulationBase
from ._src.data.simulation_newton import SimulationNewton
from ._src.data.source import SignalSource
from ._src.data.windows import TrajectoryBatch, TrajectoryWindows

__all__ = [
    "CommandBase",
    "CommandHeld",
    "CommandNearState",
    "CommandSmooth",
    "CommandSweep",
    "CommandSwitch",
    "SamplerBase",
    "SamplerFloatingBase",
    "SamplerMixture",
    "SamplerPerturb",
    "SamplerSobol",
    "SamplerUniform",
    "SignalSource",
    "SimulationBase",
    "SimulationNewton",
    "TrajectoryBatch",
    "TrajectoryDataset",
    "TrajectoryRecorder",
    "TrajectoryWindows",
    "TrajectoryWriter",
]
