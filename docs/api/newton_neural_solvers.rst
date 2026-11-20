.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.solvers
=====================

Drop-in Newton solvers backed by trained neural dynamics models.

``SolverNeRD`` steps standalone state-increment models directly in place of a physics solver. ``SolverNeRDMuJoCo``
runs free-motion models inside MuJoCo Warp for two-way contact coupling with dynamic objects. Both solvers accept
either a model instance or a checkpoint path of any registered family.

``load_nerd`` and ``same_duration`` are helper utilities for custom solvers built around
:class:`~newton_neural.models.NeRDStepper`.

.. py:module:: newton_neural.solvers
.. currentmodule:: newton_neural.solvers

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   SolverNeRD
   SolverNeRDMuJoCo

.. rubric:: Functions

.. autosummary::
   :toctree: _generated

   load_nerd
   same_duration
