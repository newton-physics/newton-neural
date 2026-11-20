.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.data
==================

Trajectory recording, sharded datasets, and GPU-resident training window loaders.

Wrap a Newton model and solver in ``SimulationNewton``, collect episodes using ``TrajectoryRecorder`` with state
samplers (``Sampler*``) and command generators (``Command*``), and load contiguous episode windows with
``TrajectoryWindows`` for training and evaluation. Use ``TrajectoryWriter`` to write datasets from custom simulation
loops.

.. py:module:: newton_neural.data
.. currentmodule:: newton_neural.data

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   CommandBase
   CommandHeld
   CommandNearState
   CommandSmooth
   CommandSweep
   CommandSwitch
   SamplerBase
   SamplerFloatingBase
   SamplerMixture
   SamplerPerturb
   SamplerSobol
   SamplerUniform
   SignalSource
   SimulationBase
   SimulationNewton
   TrajectoryBatch
   TrajectoryDataset
   TrajectoryRecorder
   TrajectoryWindows
   TrajectoryWriter
