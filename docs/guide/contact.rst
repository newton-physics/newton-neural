.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.contacts

Contact Input
=============

Standalone neural dynamics models need contact information *before* predicting the next frame. A **contact input**
keeps collision geometry analytical: an on-device Warp sensor extracts contact signals from Newton, and a PyTorch
encoder maps them into a fixed-size vector (*contact code*) consumed by the model.

Two built-in contact inputs are provided (omit ``contact_input`` if the robot moves in free space or uses
:doc:`coupling`):

.. list-table::
   :widths: 22 39 39
   :header-rows: 1

   * - Property
     - :class:`ContactInputProbes`
     - :class:`ContactInputSet`
   * - Contact geometry
     - Spheres attached to specific bodies against a single ground plane
     - Any link against any collision shape reported by Newton
   * - Best for
     - Known contact points on flat ground (e.g., quadruped feet)
     - Arbitrary contact locations (e.g., climbing, stairs, hands, platforms)
   * - Collision detection
     - Not required
     - Caller must run :class:`newton.CollisionPipeline` every step
   * - Recorded signals (per world)
     - ``contact_probes`` ``[probe_count, 8]`` (float32)
     - ``contact_rows`` ``[capacity, 14]`` (float32), ``contact_links`` ``[capacity]`` (int32), ``contact_count`` ``[]`` (int32)
   * - Encoded code width
     - ``probe_count * 8``
     - ``slot_count * slot_dim + 1`` (default: ``1025``)
   * - State noise & rollouts
     - Recomputed on perturbed and predicted states via FK
     - Reuses recorded rows (fixed to recorded trajectory)
   * - Runtime overhead
     - One lightweight Warp kernel
     - Collision detection + contact extraction kernels + attention pooling encoder

Prefer :class:`ContactInputProbes` whenever contact happens at known locations against flat ground: it avoids
collision detection overhead and stays exact during noisy training and open-loop rollouts. Switch to
:class:`ContactInputSet` when contacts occur across arbitrary links or obstacle geometry.

The runnable snippets below use the test fixture quadruped (floating base, foot spheres, force-controlled joints).
For complete Newton scene construction, see ``build_model`` in
:github:`newton_neural/examples/example_contact_probes.py` and
:github:`newton_neural/examples/example_contact_set.py`.

.. testcode::

    import newton
    import torch
    import warp as wp

    from newton_neural.contacts import ContactInputProbes, ContactInputSet
    from newton_neural.data import SimulationNewton, TrajectoryRecorder, TrajectoryWindows
    from newton_neural.models import FlashNeRD
    from newton_neural.solvers import SolverNeRD
    from newton_neural.systems import SystemArticulation
    from newton_neural.tests import fixtures

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fixtures.quadruped_model(world_count=16, device=device)
    model.joint_effort_limit.fill_(5.0)  # Bounds sampled joint forces [N*m]
    system = SystemArticulation.from_model(model, "quadruped", control=("joint_f",))

Fixed ground probes (``ContactInputProbes``)
--------------------------------------------

Each probe is a sphere rigidly attached to a robot body. Every frame, one Warp kernel computes 8 float32 features
per probe in the robot frame:

- ``[0]`` **gap** [m]: signed distance from sphere surface to ground plane (negative when penetrating),
- ``[1:4]`` **position** [m]: sphere center in the robot frame,
- ``[4:7]`` **velocity** [m/s]: world velocity of the sphere center expressed in robot-frame axes,
- ``[7]`` **touching**: ``1.0`` if ``gap < touch_distance`` (default ``0.02`` m), else ``0.0``.

:meth:`ContactInputProbes.from_model` places one probe on each body matching an ``fnmatch`` pattern and reads the
ground plane from the Newton model. If a matched body has exactly one collision sphere, its center and radius are
copied automatically; otherwise pass ``radius=...`` to place a probe at the body origin.

.. testcode::

    probes = ContactInputProbes.from_model(model, system, bodies="*_SHANK")
    print(len(probes.probes), probes.signals["contact_probes"].shape)

.. testoutput::

    4 (4, 8)

You can also define probes and the ground plane manually (plane specified in the robot's anchor frame):

.. testcode::

    by_hand = ContactInputProbes(
        [{"body": "LF_SHANK", "point": [0.0, 0.0, -0.25], "radius": 0.03}],
        ground={"normal": [0.0, 0.0, 1.0], "offset": 0.0},
        touch_distance=0.02,
    )

Attach ``probes`` to the model. Recording, noisy training, rollout recomputation, and solver sensing work
automatically:

.. testcode::

    nerd = FlashNeRD(system, contact_input=probes)

.. note::

    Probes sense only a single unbounded plane. Contacts with obstacles, stairs, or other links of the robot are
    invisible to :class:`ContactInputProbes`.

Detected contact sets (``ContactInputSet``)
-------------------------------------------

:class:`ContactInputSet` converts every rigid contact involving a robot link reported by Newton's collision detector
into a row of 14 float32 features in the robot frame (one row per robot link involved, so self-contacts produce two
rows):

- ``[0:3]`` contact point on the link's margin-expanded surface [m],
- ``[3:6]`` unit contact normal pointing from the link into the other body,
- ``[6:9]`` lever arm from the link origin to the contact point [m],
- ``[9]`` surface gap along the normal [m] (negative when penetrating),
- ``[10:13]`` relative surface velocity (link minus other body) [m/s],
- ``[13]`` dynamic flag (``1.0`` if the other body is dynamic, ``0.0`` if static or kinematic).

Each row is paired with its link index (``contact_links``, ``-1`` for padding) and the per-world active row count
(``contact_count``). The learned encoder embeds each row + link ID and uses permutation-invariant softmax attention
pooling across ``slot_count`` slots (default ``16 x 64``), concatenating
``log(1 + count) / log(1 + capacity)`` to form a fixed-width code (all zeros when no contacts are present).

.. testcode::

    contact_set = ContactInputSet.from_system(system, capacity=64)
    nerd = FlashNeRD(system, contact_input=contact_set).to(device)

Set ``capacity`` above the maximum expected contacts per world; excess rows are dropped and counted in
``CONTACT_ROW_OVERFLOW``.

**Recording.** Recording and deployment must use the same collision pipeline. Create a
:class:`newton.CollisionPipeline` (without ``body_pair_reduction``, which alters contact separation semantics and is
rejected by the sensor and :class:`~newton_neural.solvers.SolverNeRD`), pass it to
:class:`~newton_neural.data.SimulationNewton`, and disable MuJoCo's internal collision detector with
``use_mujoco_contacts=False``:

.. testcode::

    pipeline = newton.CollisionPipeline(model)
    mujoco = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=256)
    simulation = SimulationNewton(model, mujoco, system, frame_dt=0.005, collision_pipeline=pipeline)
    recorder = TrajectoryRecorder(simulation, nerd, episode_frames=(32, 64))
    dataset = recorder.record("data/quadruped", frame_count=48)

    windows = TrajectoryWindows(dataset, length=16, device=device)
    nerd.fit_normalization(windows.loader(64))

**Deployment.** Train as in :doc:`training` (skipped below to demonstrate stepping). At runtime, call
``pipeline.collide(state, contacts)`` before each ``solver.step()``:

.. testcode::

    solver = SolverNeRD(model, nerd)
    state, control = model.state(), model.control()
    contacts = pipeline.contacts()
    solver.reset(state)  # Ensures body transforms match initial joint_q

    for _ in range(10):
        pipeline.collide(state, contacts)
        solver.step(state, state, control, contacts, 0.005)

.. _contact-observables:

Contact force and overflow observables
--------------------------------------

Allocate observables once via :meth:`~newton_neural.solvers.SolverNeRD.observables` and pass them into ``step()``:

.. testcode::

    kinds = [solver.ObservableKind.CONTACT_F, solver.ObservableKind.CONTACT_ROW_OVERFLOW]
    observables = solver.observables(kinds, contacts=contacts)

    pipeline.collide(state, contacts)
    solver.step(state, state, control, contacts, 0.005, observables=observables)

    forces = wp.to_torch(observables.contact_f)  # Per contact buffer slot
    overflow = wp.to_torch(observables.contact_row_overflow)  # [world_count + 1], last entry is global world

- ``CONTACT_F``: Penalty-based contact force estimates (see :ref:`estimated-contact-forces`). Available for any
  :class:`~newton_neural.solvers.SolverNeRD` instance whenever ``contacts`` are supplied.
- ``CONTACT_ROW_OVERFLOW``: Rows dropped per world in the current step; increase ``capacity`` if non-zero.

.. _contact-injections:

Contact injection points in ``FlashNeRD``
-----------------------------------------

Contact inputs are decoupled from network architecture; each model family decides where the contact code enters its
network. :class:`~newton_neural.models.FlashNeRD` exposes ``contact_injections`` (accepting
:class:`~newton_neural.models.ContactInjection` enum members or their string values, saved in the checkpoint):

.. testcode::

    nerd = FlashNeRD(system, contact_input=probes, contact_injections=["embedding", "readout"])
    print([place.value for place in nerd.contact_injections])

.. testoutput::

    ['embedding', 'readout']

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Injection point
     - Where the contact code is added
   * - ``"embedding"``
     - Added to the frame embedding before the temporal backbone so contact history propagates across frames.
   * - ``"readout"``
     - Added to the readout hidden layer so current contacts influence the prediction directly.
   * - ``"increment"``
     - Projected directly onto the output increment via a zero-initialized linear map.

By default, all three injection points are enabled (matching the paths taken by raw frame features, so non-gated
probes behave identically to concatenating probe features onto the frame features).

Because :class:`ContactInputSet`'s encoder is trained end to end, its initial output is random. It sets
``gated = True`` so :class:`~newton_neural.models.FlashNeRD` wraps each injection point with a learned gate
initialized near zero, letting contact features enter smoothly as training progresses.
