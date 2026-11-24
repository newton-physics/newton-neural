.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton-neural
=============

``newton-neural`` is a learned robot dynamics library for `Newton <https://newton-physics.github.io/newton/>`__.
You record trajectories from a reference physics solver, train a neural network in PyTorch, evaluate its closed-loop
accuracy, and deploy it as a drop-in :class:`newton.solvers.SolverBase` across thousands of parallel worlds.

Models execute entirely on-device, reading Newton's Warp arrays directly, sensing contact in Warp kernels, and
replaying full simulation steps from CUDA graphs. All tools in the library target the common
:class:`~newton_neural.models.NeRDBase` interface, with :class:`~newton_neural.models.FlashNeRD` as the first
built-in model family.

Quickstart
----------

This runnable snippet records, trains, and deploys a :class:`~newton_neural.models.FlashNeRD` model on Newton's
cartpole. Sizes are kept small so it runs in seconds on a CPU; practical models typically record millions of frames
and train for thousands of steps on a GPU.

.. testcode::

    import newton
    import newton.examples

    from newton_neural.data import SimulationNewton, TrajectoryRecorder, TrajectoryWindows
    from newton_neural.models import FlashNeRD
    from newton_neural.solvers import SolverNeRD
    from newton_neural.systems import SystemArticulation
    from newton_neural.training import LossOneStep, create_optimizer

    # 1. Build a replicated Newton scene (64 cartpoles with position-controlled joints).
    robot = newton.ModelBuilder()
    robot.default_joint_cfg.target_ke = 100.0
    robot.default_joint_cfg.target_kd = 10.0
    asset = newton.examples.get_asset("cartpole.urdf")
    robot.add_urdf(asset, floating=False, enable_self_collisions=False)
    scene = newton.ModelBuilder()
    scene.replicate(robot, 64)
    model = scene.finalize()

    # 2. Record trajectories from MuJoCo Warp (4 substeps per 0.02 s frame).
    system = SystemArticulation.from_model(model)
    mujoco = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, mujoco, system, frame_dt=0.02, substep_count=4)
    nerd = FlashNeRD(system).to(simulation.device)  # Untrained model declares which signals to record
    recorder = TrajectoryRecorder(simulation, nerd)
    dataset = recorder.record("data/cartpole", frame_count=64)

    # 3. Train on consecutive frame windows with a standard PyTorch loop.
    windows = TrajectoryWindows(dataset, length=16, device=nerd.device)
    nerd.fit_normalization(windows.loader(256))
    loss, optimizer = LossOneStep(noise=0.1), create_optimizer(nerd)
    for batch in windows.loader(256):
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()
    nerd.save("cartpole.pt")

    # 4. Deploy as a drop-in Newton solver (1 neural step per 0.02 s frame).
    solver = SolverNeRD(model, "cartpole.pt")
    state, control = model.state(), model.control()
    for _ in range(50):
        solver.step(state, state, control, None, 0.02)

Documentation
-------------

.. grid:: 1 2 2 3
    :gutter: 3

    .. grid-item-card:: Overview
        :link: guide/overview
        :link-type: doc

        Architecture, workflow, signal conventions, and model validity bounds.

    .. grid-item-card:: Installation
        :link: guide/installation
        :link-type: doc

        Requirements, uv setup, platform notes, and tested package versions.

    .. grid-item-card:: Examples
        :link: guide/examples
        :link-type: doc

        Four runnable end-to-end examples with side-by-side Newton viewer output.

    .. grid-item-card:: Models
        :link: guide/models
        :link-type: doc

        The ``NeRDBase`` interface, composable parts, and the FlashNeRD family.

    .. grid-item-card:: Recording
        :link: guide/recording
        :link-type: doc

        Sample states and commands to record sharded datasets from a reference solver.

    .. grid-item-card:: Training
        :link: guide/training
        :link-type: doc

        Build trajectory windows, fit normalizers, configure losses, and train in PyTorch.

    .. grid-item-card:: Evaluation
        :link: guide/evaluation
        :link-type: doc

        Measure open-loop rollout error and run live rest and paired audits.

    .. grid-item-card:: Solvers
        :link: guide/solvers
        :link-type: doc

        Deploy models via ``SolverNeRD``, ``SolverNeRDMuJoCo``, and ``NeRDStepper``.

    .. grid-item-card:: Contact Input
        :link: guide/contact
        :link-type: doc

        Choose between fixed ground probes and collision-detected contact sets.

    .. grid-item-card:: Coupling
        :link: guide/coupling
        :link-type: doc

        Couple a learned robot with MuJoCo Warp to push and manipulate free objects.

    .. grid-item-card:: Performance
        :link: guide/performance
        :link-type: doc

        Precision modes (FP32, TF32, FP16, BF16), compilation, CUDA graphs, and timing.

    .. grid-item-card:: API Reference
        :link: api/newton_neural
        :link-type: doc

        Complete reference for all public modules, classes, and functions.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User Guide

    Overview <guide/overview>
    Installation <guide/installation>
    Examples <guide/examples>
    Models <guide/models>
    Recording <guide/recording>
    Training <guide/training>
    Evaluation <guide/evaluation>
    Solvers <guide/solvers>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Concepts

    Contact Input <guide/contact>
    Coupling <guide/coupling>
    Performance <guide/performance>

.. The API Reference toctree is generated by ``docs/generate_api.py`` from the public modules.

.. include:: api/_toctree.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Development

    Extending <guide/extending>
    Development <guide/development>

.. toctree::
    :hidden:
    :caption: Project Links

    GitHub <https://github.com/newton-physics/newton-neural>
