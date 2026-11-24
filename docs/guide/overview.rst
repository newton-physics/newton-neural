.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.models

Overview
========

``newton-neural`` trains neural networks to approximate robot rigid-body dynamics from a reference physics solver
and deploys them directly inside Newton. Newton continues to manage scene construction, kinematics, collision
detection, and sensors; the learned model replaces (or augments) only the dynamics step.

Throughout this documentation, *model* refers to a learned dynamics model (:class:`NeRDBase`), while *Newton model*
refers to :class:`newton.Model`.

Why neural dynamics
-------------------

Reinforcement learning and sampling-based control consume billions of simulation steps. In contact-rich environments,
conventional solvers require small substeps and iterative constraint solves in every parallel world. Learned dynamics
change that cost profile:

- **Fixed-cost stepping.** One forward pass advances all worlds by an entire control frame (replacing 4–8 physics
  substeps). Runtime scales with world count and network size rather than contact count or system stiffness. See
  :doc:`performance`.
- **Differentiability (planned).** End-to-end differentiable steps support gradient-based policy optimization and
  system identification.
- **Real-data fine-tuning (planned).** Models pretrained in simulation can be fine-tuned on real robot recordings to
  reduce the sim-to-real gap.

This preview focuses on **speed**. Tooling for differentiable rollouts and real-world fine-tuning is planned for
future releases.

Integration with Newton
-----------------------

A trained model is a native Newton solver rather than an external wrapper:

- **Standard solver interface.** :class:`~newton_neural.solvers.SolverNeRD` subclasses
  :class:`newton.solvers.SolverBase` and :class:`~newton_neural.solvers.SolverNeRDMuJoCo` subclasses
  :class:`newton.solvers.SolverMuJoCo`. Any simulation loop, viewer, or framework (including
  `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`__) steps them using standard Newton states, controls,
  contacts, and resets. See :doc:`solvers`.
- **Zero-copy device arrays.** PyTorch reads and writes joint state and controls directly in Newton's Warp buffers.
  In replicated scenes, these tensors are zero-copy views with no host round-trips.
- **Warp contact sensing.** Contact inputs run in Warp kernels directly on Newton's body transforms or detected
  contact buffers. See :doc:`contact`.
- **Compiled CUDA-graph steps.** On CUDA devices, ``torch.compile`` compiles the network step, tensor cores
  accelerate matrix multiplications, and the full step (sensing, inference, state write, and forward kinematics)
  replays from a single CUDA graph launch. See :doc:`performance`.

Workflow
--------

The library supports the full pipeline from reference simulation to deployed solver:

.. mermaid::

    flowchart LR
        R["<b>Record</b><br/>SimulationNewton<br/>TrajectoryRecorder"] --> D[("Dataset")]
        D --> T["<b>Train</b><br/>TrajectoryWindows<br/>LossOneStep"]
        T --> E["<b>Evaluate</b><br/>rollout_error<br/>audit_rest, audit_paired"]
        E --> C[("Checkpoint")]
        C --> S["<b>Deploy</b><br/>SolverNeRD<br/>SolverNeRDMuJoCo"]

1. **Record** (:doc:`recording`): Step a reference solver across parallel worlds from sampled states and commands,
   saving all required signals to a sharded dataset.
2. **Train** (:doc:`training`): Fit normalizers and train the model on consecutive frame windows in a plain PyTorch
   loop.
3. **Evaluate** (:doc:`evaluation`): Quantify open-loop rollout drift on held-out worlds and run live rest and paired
   audits against the reference solver.
4. **Deploy** (:doc:`solvers`): Load the ``.pt`` checkpoint into a Newton solver and step it in place of the
   reference solver.

Because datasets capture robot dynamics rather than a specific task, one trained model works for any policy or
planner that uses the same control interface and ``frame_dt``.

Model families
--------------

Every learned dynamics model subclasses :class:`NeRDBase`, which defines the contract used by the recorder, losses,
stepper, evaluation functions, and solvers. New architectures can be added as separate model families without
modifying any surrounding tooling (see :doc:`extending`).

:class:`FlashNeRD` is the first built-in family. It takes externally sensed contact as input and predicts one-frame
state changes or free-space accelerations using a streaming temporal backbone. See :doc:`models`.

Frames and signals
------------------

A **frame** is one model step of duration ``frame_dt`` seconds. Even if the reference solver takes multiple substeps
per frame, the learned model advances one full frame per call.

A **signal** is a named per-world tensor. Signals extracted from Newton retain Newton's names (such as ``joint_q``
and ``joint_target_q``). Every signal has one of four :class:`~newton_neural.SignalRole` values:

.. list-table::
   :widths: 15 50 35
   :header-rows: 1

   * - Role
     - Purpose
     - Examples
   * - ``STATE``
     - Dynamic state variables evolved by the model.
     - ``joint_q``, ``joint_qd``
   * - ``CONTROL``
     - Actuation inputs consumed by the solver.
     - ``joint_target_q``, ``joint_f``
   * - ``CONTACT``
     - Externally sensed contact representations.
     - ``contact_probes``, ``contact_rows``
   * - ``LABEL``
     - Supervision targets recorded from the reference solver (unused at inference).
     - ``joint_qdd_free``

Collections of signals are passed as ``dict[str, torch.Tensor]`` (aliased as :type:`~newton_neural.Frames`). Each
tensor has shape ``[world_count, *shape]`` for a single frame, or ``[world_count, frame_count, *shape]`` for a
temporal window.

Validity and limitations
------------------------

A trained model is valid only for the robot and operating regime captured in its training data:

- **Morphology and gravity.** At construction, solvers verify body labels, joint labels, joint types, kinematic tree
  structure, and gravity against the checkpoint's system specification and raise on mismatch.
- **Fixed frame duration.** Solvers raise if ``step()`` is called with any ``dt`` other than the checkpoint's
  ``frame_dt``.
- **Fixed physical parameters.** Masses, inertias, drive gains, and friction are baked into the learned weights.
  Runtime changes to those Newton model properties do not affect the network;
  :meth:`~newton_neural.solvers.SolverNeRD.notify_model_changed` emits a one-time warning per unsupported change
  flag.
- **Data distribution.** Accuracy degrades outside the states, commands, and contacts represented in the recording.
  Use :doc:`evaluation` to verify accuracy on target trajectories before training policies.
