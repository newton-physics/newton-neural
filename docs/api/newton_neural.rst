.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural
=============

Neural robot dynamics for the Newton physics engine.

``newton-neural`` trains learned dynamics models from a reference physics solver and runs them as drop-in Newton
solvers across thousands of parallel worlds.

This top-level module exports shared signal types and normalization utilities (:class:`SignalSpec`,
:class:`SignalRole`, :type:`Frames`, :class:`Normalizer`, and :class:`NewtonNeuralWarning`). The main workflow
submodules are :mod:`newton_neural.models`, :mod:`newton_neural.data`, :mod:`newton_neural.training`,
:mod:`newton_neural.evaluation`, and :mod:`newton_neural.solvers`.

.. experimental::

.. py:module:: newton_neural
.. currentmodule:: newton_neural

.. rubric:: Submodules

- :doc:`newton_neural.backbones <newton_neural_backbones>`
- :doc:`newton_neural.contacts <newton_neural_contacts>`
- :doc:`newton_neural.data <newton_neural_data>`
- :doc:`newton_neural.evaluation <newton_neural_evaluation>`
- :doc:`newton_neural.models <newton_neural_models>`
- :doc:`newton_neural.predictions <newton_neural_predictions>`
- :doc:`newton_neural.solvers <newton_neural_solvers>`
- :doc:`newton_neural.systems <newton_neural_systems>`
- :doc:`newton_neural.testing <newton_neural_testing>`
- :doc:`newton_neural.training <newton_neural_training>`

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NewtonNeuralWarning
   Normalizer
   SignalRole
   SignalSpec

.. rubric:: Type aliases

.. py:type:: Frames
   :canonical: dict[str, torch.Tensor]

   Dictionary mapping signal names to tensors: ``[world_count, *shape]`` for one frame, or
   ``[world_count, frame_count, *shape]`` for a window of consecutive frames.


.. rubric:: Constants

.. list-table::
   :header-rows: 1

   * - Name
     - Value
   * - ``__version__``
     - ``0.1.0.dev0``
