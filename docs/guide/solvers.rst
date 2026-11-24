.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.solvers

Solvers
=======

Trained models deploy directly as Newton solvers. Both solvers accept either an in-memory
:class:`~newton_neural.models.NeRDBase` instance or a checkpoint path, regardless of model family.

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Solver
     - Use when
   * - :class:`SolverNeRD`
     - Standalone neural simulation (:class:`~newton_neural.predictions.PredictionStateIncrement`). Advances the
       robot's joint state directly and updates forward kinematics.
   * - :class:`SolverNeRDMuJoCo`
     - Two-way object manipulation (:class:`~newton_neural.predictions.PredictionFreeMotion`). Predicts the robot's
       free-space joint accelerations while MuJoCo Warp resolves contacts and integrates the full scene
       (:doc:`coupling`).

.. testsetup::

    from guide_setup import cartpole_recording
    from newton_neural.data import TrajectoryWindows

    # Cartpole model from the previous pages, fitted and saved without training.
    model, system, simulation, nerd, reach, dataset = cartpole_recording(frame_count=64)
    nerd.fit_normalization(TrajectoryWindows(dataset, length=16, device=nerd.device).loader(256))
    nerd.save("cartpole.pt")

Stepping a model
----------------

Continuing from :doc:`training` with the cartpole Newton ``model`` and checkpoint ``cartpole.pt``:

.. testcode::

    import torch
    import warp as wp

    from newton_neural.solvers import SolverNeRD

    solver = SolverNeRD(model, "cartpole.pt")
    state, control = model.state(), model.control()
    frame_dt = solver.nerd.frame_dt

    for _ in range(100):
        solver.step(state, state, control, None, frame_dt)

Contract rules for :meth:`SolverNeRD.step`:

- **Exact frame step:** ``dt`` must equal ``solver.nerd.frame_dt`` (no substepping); any other value raises
  ``ValueError``.
- **Matching robot:** The constructor binds to ``articulation`` (default ``"*"``) and verifies that body/joint
  labels, joint types, topology, and gravity match the checkpoint's system.
- **Caller runs collision detection:** ``SolverNeRD`` never invokes a collision pipeline. If the contact input reads
  detected contacts (:class:`~newton_neural.contacts.ContactInputSet`) or you request ``CONTACT_F``, run
  :class:`newton.CollisionPipeline` on ``state_in`` and pass ``contacts`` (:doc:`contact`).
- **Other bodies pass through unchanged:** Non-robot joints keep their ``state_in`` values in ``state_out``.
- **In-place or ping-pong states:** ``state_in`` and ``state_out`` may alias the same :class:`newton.State` or be
  two distinct states swapped each step.

Controls and resets
-------------------

Write controls into :class:`newton.Control` directly or via ``solver.binding.write_control`` using per-world tensors:

.. testcode::

    world_count = solver.binding.world_count
    targets = torch.zeros(world_count, 3, device=solver.nerd.device)
    solver.binding.write_control(control, {"joint_target_q": targets})

When resetting a world, two things must happen: the solver's temporal history for that world must be cleared, and
body transforms must match the new joint coordinates.
:meth:`SimulationNewton.restart <newton_neural.data.SimulationNewton.restart>` handles reset ordering, forward
kinematics, and contact updates automatically for both solvers:

.. testcode::

    from newton_neural.data import SimulationNewton

    simulation = SimulationNewton(model, solver, solver.nerd.system, frame_dt=frame_dt)
    start = {
        "joint_q": torch.zeros(world_count, 3, device=solver.nerd.device),
        "joint_qd": torch.zeros(world_count, 3, device=solver.nerd.device),
    }
    restarting = torch.zeros(world_count, dtype=torch.bool, device=solver.nerd.device)
    restarting[:4] = True

    simulation.restart(start, restarting)
    simulation.step({"joint_target_q": targets})  # Advances simulation.state by one frame

.. note::

    **Calling ``solver.reset()`` directly.** Both solvers follow :meth:`newton.solvers.SolverMuJoCo.reset`. The
    history of the masked worlds is cleared, and ``flags`` selects the joint state that is restored to Newton model
    defaults (all of it by default, none with ``flags=0``). To start from a custom state, call
    ``reset(state, world_mask)`` first and then write ``joint_q`` / ``joint_qd``. :class:`SolverNeRD` reads body
    transforms from its input state, so update them with :func:`newton.eval_fk` after writing.

Runtime options and CUDA graphs
-------------------------------

Both solvers accept ``precision``, ``compile``, and ``use_cuda_graph``. By default on CUDA, the solver uses the
checkpoint's precision (``"tf32"`` for a new model), compiles the network step with ``torch.compile``, and replays
the step from a CUDA graph. See :doc:`performance` for full details.

.. testcode::

    fast = SolverNeRD(model, "cartpole.pt", precision="fp16")
    print(fast.stepper.precision)

.. testoutput::

    fp16

Important rules when CUDA graphs are enabled:

- **Keep buffer objects stable:** Each CUDA graph captures exact device pointers, so the solver caches one graph per
  unique tuple of ``(state_in, state_out, control, contacts, observables)`` object identities (for example, a
  ping-pong loop uses 2 graphs). Creating fresh state/control objects every iteration fills the cache and raises
  ``RuntimeError`` past the limit of 16 graphs. Reuse buffers or set ``use_cuda_graph=False``.
- **Warm up once before external capture:** The first call to ``step()`` compiles the network and validates numerical
  accuracy against eager FP32 (it still advances the simulation normally). If you record your own outer CUDA graph
  with Warp (``THREAD_LOCAL`` or ``RELAXED`` capture mode), run one step *before* entering ``wp.ScopedCapture()``,
  and keep the solver object alive while replaying the graph. Capturing on CPU raises because Warp CPU capture only
  records Warp kernels.

.. code-block:: python

    solver.step(state, state, control, None, frame_dt)  # Warm-up step outside capture

    with wp.ScopedCapture() as capture:
        for _ in range(10):
            solver.step(state, state, control, None, frame_dt)

    wp.capture_launch(capture.graph)  # Replays 10 frames in a single launch

.. _estimated-contact-forces:

Estimated contact forces and overflow
-------------------------------------

Standalone models (:class:`SolverNeRD`) do not compute constraint reaction forces; contact response is learned
implicitly into the predicted motion. To keep Newton contact sensors and RL contact rewards functional,
``SolverNeRD`` supports two observables:

- ``CONTACT_F``: Penalty-based **estimated** normal wrenches computed at ``state_in`` for each rigid contact passed
  in ``contacts``. For penetrating contacts (``gap < 0``), the normal force on body 0 is
  :math:`\mathbf{f} = \frac{1}{2}(k_{e,0} + k_{e,1}) \cdot \text{gap} \cdot \mathbf{n}` [N] (zero friction), paired
  with its torque about body 0's center of mass [N·m] in Newton's world-frame spatial vector layout. Non-penetrating
  and unused slots are zero.
- ``CONTACT_ROW_OVERFLOW``: Per-world count of contact rows dropped when a sensor buffer (e.g.,
  :class:`~newton_neural.contacts.ContactInputSet`) exceeds its ``capacity``.

.. note::

    ``CONTACT_F`` estimates are **not** the actual forces that drove the neural prediction. Their magnitude scales
    with the contact stiffness of the two shapes (``shape_material_ke``), which plays no role in the dynamics the
    model learned, so it is not calibrated against the load. With Newton's default stiffness of ``2.5e3`` N/m, a
    penetration of 1 mm reads as about 2.5 N whatever the robot weighs. The estimate acts along the contact normal
    only. Use it as an indicator of contact and for its direction, and do not use it for force control or direct
    magnitude comparisons against a physics solver. Computing it also requires running collision detection to
    supply ``contacts``, even if your model uses probes or no contact input.

See :ref:`contact-observables` for usage code.

Handling Newton model changes
-----------------------------

Because dynamics parameters (masses, drive gains, friction, gravity, etc.) are baked into trained weights, mutating
them on the Newton model at runtime does not change neural predictions. Calling :meth:`SolverNeRD.notify_model_changed`
emits a one-time :class:`~newton_neural.NewtonNeuralWarning` for each flag listed in
``SolverNeRD.UNSEEN_MODEL_CHANGES``. Joint frame edits do not warn because forward kinematics reads them from the
Newton model every step.

:meth:`SolverNeRDMuJoCo.notify_model_changed` forwards all updates to MuJoCo Warp and warns for properties in
``SolverNeRDMuJoCo.UNSEEN_MODEL_CHANGES`` that affect MuJoCo's constraint solve without affecting the learned free
acceleration.

Tensor-level stepping (``NeRDStepper``)
---------------------------------------

If you want to roll out a model directly on PyTorch tensors without Newton state objects (for example, inside a
sampling-based planner or MPC loop), use :class:`~newton_neural.models.NeRDStepper`:

.. testcode::

    from newton_neural.models import NeRDStepper

    stepper = NeRDStepper(solver.nerd, world_count)
    frames = {**solver.binding.read_state(state), **solver.binding.read_control(control)}
    following = stepper.step(frames)  # Next joint_q and joint_qd
    branch = stepper.clone()  # Shares weights & compiled graph; copies history only

- :meth:`~newton_neural.models.NeRDStepper.step` integrates the prediction and returns the next state signals
  (raises ``RuntimeError`` for non-integrating predictions such as free motion).
- :meth:`~newton_neural.models.NeRDStepper.predict` returns raw predictions in physical units for any model.
- :meth:`~newton_neural.models.NeRDStepper.clone`, :meth:`~newton_neural.models.NeRDStepper.export_history`, and
  :meth:`~newton_neural.models.NeRDStepper.import_history` support branching and restoring rollout histories, while
  :meth:`~newton_neural.models.NeRDStepper.reset` clears history for masked worlds.
