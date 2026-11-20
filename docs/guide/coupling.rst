.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.solvers

Coupling
========

Standalone :class:`SolverNeRD` evolves only the robot and leaves all other bodies untouched, so it cannot simulate a
robot pushing, grasping, or lifting dynamic objects that push back.

**Two-way coupling** solves this by splitting responsibilities between the neural model and MuJoCo Warp:

- The neural model predicts the robot's **free-space joint acceleration** (``joint_qdd_free``): its acceleration
  from internal dynamics, drives, and gravity *before* contacts and joint limits act.
- MuJoCo Warp computes free accelerations for the rest of the scene, substitutes the model's prediction on the
  robot's DOFs, solves contacts, joint limits, and friction for robot and objects jointly, and integrates the state.

.. mermaid::

    flowchart LR
        A["<b>MuJoCo Warp</b><br/>Free accelerations<br/>for full scene"] --> B["<b>Neural Model</b><br/>Overrides robot<br/>DOF rows"]
        B --> C["<b>MuJoCo Warp</b><br/>Solves contacts, limits,<br/>and friction jointly"]
        C --> D["<b>MuJoCo Warp</b><br/>Integrates state"]

Because the label is captured *before* constraint resolution, you record and train on the robot alone. No scene
objects are needed during training, and the trained model generalizes to arbitrary objects at deployment.

Standalone vs. two-way coupling
-------------------------------

.. list-table::
   :widths: 22 39 39
   :header-rows: 1

   * -
     - Standalone (:class:`SolverNeRD`)
     - Two-way (:class:`SolverNeRDMuJoCo`)
   * - Prediction
     - :class:`~newton_neural.predictions.PredictionStateIncrement`
     - :class:`~newton_neural.predictions.PredictionFreeMotion`
   * - Dynamic objects
     - Not moved by the robot
     - Full two-way interaction (robot pushes objects and feels reaction forces)
   * - Contact resolution
     - Learned by the network via contact input
     - Analytical constraint solve in MuJoCo Warp
   * - Robot base
     - Fixed or floating
     - Fixed only
   * - Step cost
     - One network forward pass *(fast)*
     - Roughly one MuJoCo Warp step

Use **standalone** mode for speed (e.g., locomotion on static terrain). Use **two-way coupling** for manipulation
when you want a learned robot dynamics model (e.g., capturing complex drives or real-world fine-tuning) inside an
analytical contact solver, not for step-time speedup over MuJoCo Warp.

1. Record free accelerations on the robot alone
-----------------------------------------------

The snippet below uses the test fixture arm (fixed base, 6 revolute joints + 2-finger gripper, force-controlled),
then deploys it alongside a free box. For explicit Newton model construction, see ``build_model`` in
:github:`newton_neural/examples/example_coupling.py`.

.. testcode::

    import newton
    import torch

    from newton_neural.backbones import BackboneMLP
    from newton_neural.data import SamplerUniform, SimulationNewton, TrajectoryRecorder, TrajectoryWindows
    from newton_neural.models import FlashNeRD
    from newton_neural.predictions import PredictionFreeMotion
    from newton_neural.solvers import SolverNeRDMuJoCo
    from newton_neural.systems import SystemArticulation
    from newton_neural.tests import fixtures
    from newton_neural.training import LossOneStep, create_optimizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot = fixtures.arm_model(world_count=16, device=device)
    robot.joint_effort_limit.fill_(2.0)  # Bounds sampled joint forces [N or N*m]
    system = SystemArticulation.from_model(robot, "arm", control=("joint_f",))

    backbone = BackboneMLP(window=1)
    nerd = FlashNeRD(system, backbone=backbone, prediction=PredictionFreeMotion()).to(device)
    print(nerd.receptive_field, list(nerd.signals))

.. testoutput::

    1 ['joint_q', 'joint_qd', 'joint_f', 'joint_qdd_free']

Two differences compared to a standalone model:

- **Recorded label signal:** :class:`~newton_neural.predictions.PredictionFreeMotion` records ``joint_qdd_free``
  [rad/s² or m/s²] directly from :class:`newton.solvers.SolverMuJoCo` (requires ``substep_count=1``).
- **Single-frame history:** Free acceleration depends only on the current state and control, so
  ``BackboneMLP(window=1)`` is sufficient and recording can use independent single-step samples
  (``episode_frames=(1, 1)``):

.. testcode::

    mujoco = newton.solvers.SolverMuJoCo(robot)
    simulation = SimulationNewton(robot, mujoco, system, frame_dt=0.005)
    one = torch.ones(system.joint_dof_count, device=device)
    states = SamplerUniform({"joint_q": (-one, one), "joint_qd": (-one, one)})
    recorder = TrajectoryRecorder(
        simulation, nerd, states=states, episode_frames=(1, 1)
    )
    dataset = recorder.record("data/arm", frame_count=64)

2. Train on single frames
-------------------------

Train with ``length=0`` (one frame per window) and ``noise=0`` (default), because recorded ``joint_qdd_free`` labels
correspond strictly to the unperturbed state:

.. testcode::

    windows = TrajectoryWindows(dataset, length=0, device=device)
    nerd.fit_normalization(windows.loader(256))
    loss, optimizer = LossOneStep(), create_optimizer(nerd)
    for batch in windows.loader(256):
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()
    nerd.save("arm.pt")

3. Deploy in a scene with objects
---------------------------------

:class:`SolverNeRDMuJoCo` subclasses :class:`newton.solvers.SolverMuJoCo` and accepts all MuJoCo solver keyword
arguments alongside ``precision``, ``compile``, and ``use_cuda_graph`` (:doc:`performance`):

.. testcode::

    scene = fixtures.arm_with_box_model(world_count=16, device=device)  # Arm + free box
    solver = SolverNeRDMuJoCo(scene, "arm.pt", articulation="arm")

    state_0, state_1, control = scene.state(), scene.state(), scene.control()
    for _ in range(20):
        solver.step(state_0, state_1, control, None, 0.005)
        state_0, state_1 = state_1, state_0

Pass ``articulation="arm"`` so the solver binds to the robot rather than the box articulation.

.. note::

    :meth:`SolverNeRDMuJoCo.reset` restores joint coordinates to Newton model defaults and clears the model's history,
    as :meth:`SolverNeRD.reset` does. Call ``solver.reset(...)`` *before* writing custom joint states, or use
    :meth:`SimulationNewton.restart <newton_neural.data.SimulationNewton.restart>` (see :doc:`solvers`).

Requirements
------------

:class:`SolverNeRDMuJoCo` validates these requirements at construction and raises if any fail:

- Fixed-base articulation with revolute, prismatic, and fixed joints only.
- Model uses :class:`~newton_neural.predictions.PredictionFreeMotion` with ``contact_input=None``.
- MuJoCo Warp backend (``use_mujoco_cpu=False``), integrator in ``{"euler", "implicit", "implicitfast"}``,
  ``enable_sleeping=False``, and ``use_speculative_contacts=False``.
- ``dt == nerd.frame_dt`` on every ``step()`` call.

How injection works inside MuJoCo Warp
--------------------------------------

:class:`SolverNeRDMuJoCo` replaces its instance's MuJoCo Warp step with the equivalent sequence of public MuJoCo Warp
stage calls, inserting a Warp kernel between ``fwd_acceleration`` and ``solve`` that overwrites ``qacc_smooth`` on
the robot's DOFs. At construction, the solver verifies that its staged step reproduces MuJoCo Warp's native step
before enabling acceleration replacement.
