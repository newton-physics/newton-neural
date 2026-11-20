.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.models

Models
======

Every neural dynamics model in ``newton-neural`` is a ``torch.nn.Module`` subclassing :class:`NeRDBase` and composed
of modular, swappable parts. :class:`FlashNeRD` is the first model family included in the library.

The ``NeRDBase`` interface
--------------------------

All library tools (recording, training losses, stepping, rollout evaluation, and Newton solvers) interact with models
strictly through :class:`NeRDBase`. Any custom model family subclassing :class:`NeRDBase` works automatically with
all of them (see :doc:`extending`).

Every model exposes two execution paths for the same underlying function:

- **Windowed training** (:meth:`~NeRDBase.forward`): Evaluates an entire batch of consecutive frame windows in
  parallel and returns normalized predictions for the loss (:doc:`training`).
- **Streaming inference** (:meth:`~NeRDBase.predict`): Evaluates one frame per world at a time via
  :class:`NeRDStepper` or a solver, updating fixed-shape per-world history buffers in place (:doc:`solvers`).

Both paths must produce identical outputs so the deployed model matches the trained one. Use
:func:`~newton_neural.testing.check_nerd_streaming` to verify equivalence for any model family.

Checkpoints and precision
^^^^^^^^^^^^^^^^^^^^^^^^^

Each model stores a default deployment precision in :attr:`NeRDBase.precision` (``"tf32"`` for a new model).
Steppers and solvers inherit this precision unless overridden; see :doc:`performance` for the four precision modes.

Saving a model (:meth:`NeRDBase.save`) writes a single ``.pt`` checkpoint containing weights, normalization
statistics, ``frame_dt``, ``precision``, signal specifications, model class name, part classes and configs, family
settings, and user metadata:

- :meth:`NeRDBase.load` instantiates whichever registered model family wrote the checkpoint (used by solvers).
- Calling ``<Family>.load`` (e.g., ``FlashNeRD.load``) enforces that the checkpoint belongs to that family and raises
  otherwise.

Composable parts
----------------

Every model is built from three core parts. Each part can be swapped independently while keeping the rest of the
pipeline (recording signals, labels, losses, and state integration) unchanged:

.. list-table::
   :widths: 16 32 52
   :header-rows: 1

   * - Part
     - Built-in classes
     - Responsibility
   * - System
     - :class:`~newton_neural.systems.SystemArticulation`
     - Defines state and control signals, maps raw frames to network features, and integrates predicted increments
       back into states.
   * - Prediction
     - :class:`~newton_neural.predictions.PredictionStateIncrement`,
       :class:`~newton_neural.predictions.PredictionFreeMotion`
     - Defines what the network outputs, where supervision labels come from, and which solver consumes the output
       (:doc:`coupling`).
   * - Contact input
     - ``None``, :class:`~newton_neural.contacts.ContactInputProbes`,
       :class:`~newton_neural.contacts.ContactInputSet`
     - Senses contact externally from Newton state or collision buffers and encodes it into a vector for the network
       (:doc:`contact`).

Systems
^^^^^^^

:class:`~newton_neural.systems.SystemArticulation` represents one articulated robot per world with either a fixed or
floating base and revolute, prismatic, and fixed joints. Build it directly from a Newton model via
:meth:`~newton_neural.systems.SystemArticulation.from_model`.

Bodies and actuated joints (revolute and prismatic) are matched by the final segment of their Newton labels, so every
body and every actuated joint within the robot must have a unique label (fixed joints may share labels).

The ``control`` parameter selects which :class:`newton.Control` arrays actuate the robot:

.. list-table::
   :widths: 45 55
   :header-rows: 1

   * - Actuation mode
     - ``control`` argument
   * - Position targets (using Newton's joint drive gains)
     - ``("joint_target_q",)`` *(default)*
   * - Position and velocity targets
     - ``("joint_target_q", "joint_target_qd")``
   * - Direct joint forces / torques
     - ``("joint_f",)``
   * - General actuator inputs
     - ``("joint_act",)``

Everything between the chosen control arrays and the resulting motion, including actuator and PD dynamics simulated
by the reference solver, is learned by the network.

For floating-base robots, :class:`~newton_neural.systems.SystemArticulation` expresses features and increments in the
base's heading frame (origin at the base, yaw aligned with the base about the up axis). This makes predictions
invariant to horizontal translation and heading angle across worlds.

Choosing prediction, contact input, and solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prediction and contact input are chosen when constructing the model; the corresponding solver is selected at
deployment:

.. list-table::
   :widths: 32 22 22 24
   :header-rows: 1

   * - Scenario
     - Prediction
     - Contact input
     - Solver
   * - Robot moves in free space (no contact)
     - ``PredictionStateIncrement``
     - ``None``
     - :class:`~newton_neural.solvers.SolverNeRD`
   * - Contact at known points against flat ground (e.g., feet)
     - ``PredictionStateIncrement``
     - ``ContactInputProbes``
     - :class:`~newton_neural.solvers.SolverNeRD`
   * - Arbitrary contact anywhere on the robot
     - ``PredictionStateIncrement``
     - ``ContactInputSet``
     - :class:`~newton_neural.solvers.SolverNeRD`
   * - Fixed-base robot pushing or manipulating free objects
     - ``PredictionFreeMotion``
     - ``None``
     - :class:`~newton_neural.solvers.SolverNeRDMuJoCo`

With :class:`~newton_neural.predictions.PredictionStateIncrement`, the network predicts the full one-frame state
change (contact effects included) and :class:`~newton_neural.solvers.SolverNeRD` integrates it directly. With
:class:`~newton_neural.predictions.PredictionFreeMotion`, the network predicts unconstrained joint accelerations and
:class:`~newton_neural.solvers.SolverNeRDMuJoCo` hands them to MuJoCo Warp to resolve contacts with objects. See
:doc:`contact` and :doc:`coupling`.

FlashNeRD
---------

:class:`FlashNeRD` is a learned forward-dynamics model family built around **explicit external contact input**:

1. **External contact sensing.** Contact introduces sharp discontinuities that Newton already computes accurately and
   cheaply. Rather than forcing the network to infer contact geometry from joint states, a *contact input* senses
   contact externally via kinematics (:class:`~newton_neural.contacts.ContactInputProbes`) or Newton's collision
   detector (:class:`~newton_neural.contacts.ContactInputSet`), encodes it into a contact code, and injects that code
   into the network at configurable injection points (``contact_injections``; see :ref:`contact-injections`).
2. **Streaming temporal forward dynamics.** Each frame's normalized state/control features are embedded, processed by
   a streaming temporal *backbone* over a short history window, and mapped to a one-frame prediction by a readout
   head. At deployment, only the newest frame is computed while past context lives in fixed-size history buffers.
3. **Standalone or coupled deployment.** Defaults to :class:`~newton_neural.predictions.PredictionStateIncrement`
   for standalone stepping with :class:`~newton_neural.solvers.SolverNeRD`, or swaps to
   :class:`~newton_neural.predictions.PredictionFreeMotion` for two-way coupling via
   :class:`~newton_neural.solvers.SolverNeRDMuJoCo`.

Every argument except ``system`` is optional (defaulting to :class:`~newton_neural.backbones.BackboneFlash`, no
contact input, and :class:`~newton_neural.predictions.PredictionStateIncrement`):

.. code-block:: python

    nerd = FlashNeRD(
        system,
        backbone=None,
        contact_input=None,
        contact_injections=None,
        prediction=None,
    )

Backbones
^^^^^^^^^

The backbone is the temporal trunk mapping embedded frames to hidden states. Backbones are independent of system
features, predictions, and contacts, so custom model families can reuse them too.

.. list-table::
   :widths: 26 22 52
   :header-rows: 1

   * - Backbone
     - Default receptive field
     - Notes
   * - :class:`~newton_neural.backbones.BackboneFlash`
     - 13 frames
     - Default. Runs attention and SwiGLU FFN in parallel from a single packed projection with one shared KV head
       (2 linear layers per block instead of 4).
   * - :class:`~newton_neural.backbones.BackboneTransformer`
     - 29 frames
     - Standard serial multi-head attention + SwiGLU FFN with separate KV per head. Higher step latency than
       ``BackboneFlash``.
   * - :class:`~newton_neural.backbones.BackboneMLP`
     - 4 frames
     - Lowest step cost; concatenates a fixed window of frames. Use ``window=1`` for single-frame predictions such
       as :class:`~newton_neural.predictions.PredictionFreeMotion`.

The model's :attr:`~NeRDBase.receptive_field` is the number of frames (including the current frame) that influence
one prediction. For stacked attention blocks where block :math:`\ell` attends over :math:`S_\ell` frames, spans
compound additively:

.. math::

    R = 1 + \sum_{\ell} (S_\ell - 1)

With ``BackboneFlash``'s default ``spans=(4, 4, 4, 4)``, :math:`R = 1 + 4 \times 3 = 13`. During deployment, each
block maintains a fixed-size per-world ring buffer of past keys and values updated in place, so streaming inference
matches windowed training exactly at constant per-step cost.
