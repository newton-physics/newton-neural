.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural

Extending
=========

``newton-neural`` is designed to be extended by subclassing. Any custom part or model family automatically works
with recording, training losses, evaluation, checkpointing, and solvers without modifying library code.

.. list-table::
   :widths: 20 32 48
   :header-rows: 1

   * - Extension point
     - Base class
     - Responsibility
   * - Backbone
     - :class:`~newton_neural.backbones.BackboneBase`
     - Temporal trunk mapping embedded frames to hidden states
   * - Prediction
     - :class:`~newton_neural.predictions.PredictionBase`
     - Network output definition, label signals, and supervision target
   * - Contact input
     - :class:`~newton_neural.contacts.ContactInputBase`
     - External contact sensing (Warp/PyTorch) and encoding
   * - System
     - :class:`~newton_neural.systems.SystemBase`
     - Signal layout, feature extraction, state increments/integration, and Newton binding
   * - Model family
     - :class:`~newton_neural.models.NeRDBase`
     - Full network architecture and streaming inference contract

The sections below implement one working example of each extension point, ordered from simplest to most involved.
Every snippet uses only public imports and runs as part of the documentation test suite.

Registration and serialization
------------------------------

Each base class maintains a class-name registry populated automatically via ``__init_subclass__``:

- **Naming:** Registry names are global per base class. Prefix custom classes with their kind (e.g., ``BackbonePair``,
  ``ContactInputHeight``). Re-defining the same class in the same module and qualname (e.g., rerunning a notebook
  cell or reloading a module) updates the entry; registering a colliding name from another location raises
  ``TypeError``.
- **Configs:** Every part implements ``config() -> dict[str, Any]`` returning JSON-serializable plain data (by
  default, constructor keyword arguments) so ``from_config(config)`` and ``Base.create(name, config)`` can rebuild
  it. Model construction validates JSON serializability eagerly so serialization errors surface before training.
- **Loading external classes:** Checkpoints store class names and configs, resolved via registries in
  :meth:`.NeRDBase.load`. Import the module defining your custom class before loading a checkpoint that references
  it. ``load()`` also compares saved signal specs against the rebuilt model and raises ``ValueError`` if a class
  changed its signal schema since saving.

Shared test harness
-------------------

Each part below is exercised through the full lifecycle (recording, one training step, save/load, and single-frame
stepping) using Newton's force-controlled cartpole and the helpers below:

.. testcode::

    import newton
    import newton.examples
    import torch
    from torch import Tensor

    from newton_neural import Frames
    from newton_neural.data import SimulationNewton, TrajectoryDataset, TrajectoryRecorder, TrajectoryWindows
    from newton_neural.models import FlashNeRD, NeRDBase, NeRDStepper
    from newton_neural.systems import SystemArticulation
    from newton_neural.training import LossOneStep, create_optimizer

    WORLD_COUNT = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"


    def replicate(robot: newton.ModelBuilder, *, ground: bool = False) -> newton.Model:
        """Build a replicated Newton model with one ``robot`` per world."""
        scene = newton.ModelBuilder()
        scene.replicate(robot, WORLD_COUNT)
        if ground:
            scene.add_ground_plane()
        model = scene.finalize(device=device)
        model.joint_effort_limit.fill_(5.0)  # Bounds sampled joint forces [N or N*m]
        return model


    def record(nerd: NeRDBase, model: newton.Model, path: str) -> TrajectoryDataset:
        """Record a short dataset of all signals declared by ``nerd`` from MuJoCo Warp."""
        mujoco = newton.solvers.SolverMuJoCo(model, njmax=128)
        simulation = SimulationNewton(model, mujoco, nerd.system, frame_dt=0.02, substep_count=4)
        recorder = TrajectoryRecorder(simulation, nerd, episode_frames=(16, 32))
        return recorder.record(path, frame_count=64)


    def try_out(nerd: FlashNeRD, dataset: TrajectoryDataset, path: str) -> tuple[FlashNeRD, Frames | Tensor]:
        """Run one training step, save and reload the checkpoint, and step one frame."""
        windows = TrajectoryWindows(dataset, length=4, device=device)
        nerd.fit_normalization(windows.loader(32))
        loss, optimizer = LossOneStep(), create_optimizer(nerd)
        batch = next(windows.loader(WORLD_COUNT))
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()

        nerd.save(path)
        loaded = FlashNeRD.load(path, device=device)
        stepper = NeRDStepper(loaded, WORLD_COUNT)
        first_frames = {name: value[:, 0] for name, value in batch.frames.items()}
        if loaded.prediction.integrates:
            return loaded, stepper.step(first_frames)
        return loaded, stepper.predict(first_frames)


    robot = newton.ModelBuilder()
    robot.add_urdf(newton.examples.get_asset("cartpole.urdf"), floating=False, enable_self_collisions=False)
    cartpole = replicate(robot)
    cartpole_system = SystemArticulation.from_model(cartpole, control=("joint_f",))

Custom backbone
---------------

A backbone (:class:`~newton_neural.backbones.BackboneBase`) implements the temporal trunk in two equivalent ways:

- ``forward(h)``: Processes full training windows ``[B, T, width] -> [B, T, width]``, treating frame 0 as an episode
  start.
- ``step(h, history)``: Processes one frame ``[B, width] -> [B, width]`` while updating ``history`` in place, with
  no data-dependent control flow or allocations (required for ``torch.compile`` and CUDA graphs).
- ``history(world_count, *, device, dtype)``: Allocates all-zero initial history tensors (a zeroed row must represent
  a freshly reset world, since resets zero out masked rows).
- ``receptive_field``: Number of frames a hidden state depends on (or ``None`` for unbounded/recurrent backbones;
  train those with ``windows.loader(batch_size, weights=windows.episode_start)``).
- ``config()``: Returns constructor kwargs. Override ``hidden_weights()`` if you want to customize which parameters
  :class:`~newton_neural.training.OptimizerMuon` updates (defaults to all ``nn.Linear`` weights in the backbone).

Below is a minimal two-frame backbone that combines the current embedded frame with the previous one:

.. testcode::

    from typing import Any

    from torch import nn

    from newton_neural.backbones import BackboneBase


    class BackbonePair(BackboneBase):
        """Linear mix of the current embedded frame and the previous frame.

        Args:
            width: Feature width of embedded frames and hidden states.
        """

        receptive_field = 2

        def __init__(self, *, width: int = 64) -> None:
            super().__init__()
            self.width = width
            self.mix = nn.Linear(2 * width, width)

        def forward(self, h: Tensor) -> Tensor:
            # Frame 0 starts an episode and duplicates itself as its own predecessor.
            previous = torch.cat((h[:, :1], h[:, :-1]), dim=1)
            return torch.tanh(self.mix(torch.cat((h, previous), dim=-1)))

        def step(self, h: Tensor, history: dict[str, Tensor]) -> Tensor:
            first = (history["step"] == 0)[:, None]
            previous = torch.where(first, h, history["previous"])
            history["previous"].copy_(h)
            history["step"] += 1
            return torch.tanh(self.mix(torch.cat((h, previous), dim=-1)))

        def history(
            self, world_count: int, *, device: torch.device | str, dtype: torch.dtype
        ) -> dict[str, Tensor]:
            return {
                "previous": torch.zeros(world_count, self.width, device=device, dtype=dtype),
                "step": torch.zeros(world_count, dtype=torch.int64, device=device),
            }

        def config(self) -> dict[str, Any]:
            return {"width": self.width}

.. testcode::
    :hide:

    def define_elsewhere() -> None:
        class BackbonePair(BackboneBase):
            receptive_field = 2

    try:
        define_elsewhere()
    except TypeError:
        pass
    else:
        raise AssertionError("a class named BackbonePair was registered from another place")

Verify streaming/window equivalence, mid-episode resets (even with NaN-poisoned stale history), world isolation,
buffer invariance, and ``torch.autocast`` behavior with :func:`~newton_neural.testing.check_backbone_streaming`:

.. testcode::

    from newton_neural.testing import check_backbone_streaming

    check_backbone_streaming(BackbonePair())
    check_backbone_streaming(BackbonePair(), dtype=torch.float32, autocast=torch.bfloat16)

Plug ``BackbonePair`` into :class:`.FlashNeRD`:

.. testcode::

    nerd = FlashNeRD(cartpole_system, backbone=BackbonePair()).to(device)
    loaded, following = try_out(nerd, record(nerd, cartpole, "data/pair"), "pair.pt")
    print(type(loaded.network.backbone).__name__, loaded.receptive_field, list(following))

.. testoutput::

    BackbonePair 2 ['joint_q', 'joint_qd']

Custom prediction
-----------------

A prediction (:class:`~newton_neural.predictions.PredictionBase`) defines what the network outputs and how its
supervision label is computed from :type:`Frames` (:doc:`overview`):

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Member
     - Contract
   * - ``integrates``
     - Class bool. ``True`` if the model integrates the output itself via ``system.integrate`` (supervision comes
       from ``(current, following)`` pairs and supports state noise). ``False`` if an external solver consumes the
       output (supervision comes from recorded ``LABEL`` signals on each frame; ``following`` is ``None``).
   * - ``signals(system)``
     - ``dict[str, SignalSpec]`` of ``SignalRole.LABEL`` signals to record (empty if ``integrates=True``).
   * - ``output_count(system)``
     - Output vector dimension.
   * - ``label(system, current, following)``
     - Target tensor in physical units.
   * - ``source(simulation)``
     - :class:`~newton_neural.data.SignalSource` capturing the label signals after each simulation step (must clone
       tensors before returning so future steps cannot overwrite them).
   * - ``config()``
     - Constructor kwargs (default ``{}``).

Because :class:`~newton_neural.predictions.PredictionStateIncrement` already covers integrating increments, custom
predictions typically set ``integrates = False`` for consumption by a custom solver. This example predicts one-frame
joint velocity changes ``joint_qd_next - joint_qd``:

.. testcode::

    from newton_neural import SignalRole, SignalSpec
    from newton_neural.data import SignalSource, SimulationBase
    from newton_neural.predictions import PredictionBase
    from newton_neural.systems import SystemBase


    class PredictionVelocityChange(PredictionBase):
        """Predicts one-frame joint velocity change [m/s or rad/s].

        Records ``joint_qd_next`` after each step and supervises against ``joint_qd_next - joint_qd``.
        """

        integrates = False

        def signals(self, system: SystemBase) -> dict[str, SignalSpec]:
            velocity = system.signals["joint_qd"]
            label = SignalSpec("joint_qd_next", velocity.shape, SignalRole.LABEL, unit=velocity.unit)
            return {label.name: label}

        def output_count(self, system: SystemBase) -> int:
            return system.signals["joint_qd"].shape[0]

        def label(self, system: SystemBase, current: Frames, following: Frames | None) -> Tensor:
            return current["joint_qd_next"] - current["joint_qd"]

        def source(self, simulation: SimulationBase) -> SignalSource:
            def read() -> Frames:
                state = simulation.binding.read_state(simulation.state)
                return {"joint_qd_next": state["joint_qd"].clone()}

            return SignalSource(self.signals(simulation.system), read, after_step=True)

.. testcode::

    from newton_neural.backbones import BackboneMLP

    nerd = FlashNeRD(
        cartpole_system,
        backbone=BackboneMLP(width=64, depth=2, window=1),
        prediction=PredictionVelocityChange(),
    ).to(device)
    loaded, change = try_out(nerd, record(nerd, cartpole, "data/velocity"), "velocity.pt")
    print(type(loaded.prediction).__name__, list(loaded.signals), tuple(change.shape))

.. testoutput::

    PredictionVelocityChange ['joint_q', 'joint_qd', 'joint_f', 'joint_qd_next'] (8, 3)

Models with ``integrates = False`` must train with ``noise=0`` (recorded labels do not apply to perturbed states)
and cannot run in :class:`~newton_neural.solvers.SolverNeRD`. Instead, call :meth:`.NeRDStepper.predict` inside a solver that
applies the prediction (analogous to :class:`~newton_neural.solvers.SolverNeRDMuJoCo`).

.. testcode::
    :hide:

    from newton_neural.solvers import SolverNeRD

    try:
        SolverNeRD(cartpole, "velocity.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("SolverNeRD accepted a prediction that does not integrate")

Custom contact input
--------------------

A contact input (:class:`~newton_neural.contacts.ContactInputBase`) is a stateless configuration object that spawns
two runtime components:

- A :class:`~newton_neural.contacts.ContactSensorBase` that reads Newton state/contacts on-device into preallocated
  buffers (no host transfers).
- A PyTorch ``nn.Module`` encoder that normalizes and maps contact signals into a fixed-width contact code (all
  learnable weights and normalizer buffers live on the encoder and are saved in the checkpoint).

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Member
     - Contract
   * - ``signals``
     - ``dict[str, SignalSpec]`` with role ``SignalRole.CONTACT``.
   * - ``code_count``
     - Width of the encoded contact vector.
   * - ``gated``
     - Class bool (default ``False``). Hint telling model families like :class:`.FlashNeRD` to gate the contact code
       starting near zero (recommended when the encoder has randomly initialized trainable weights).
   * - ``reads_contacts``
     - Class bool (default ``False``). Set ``True`` if ``sensor.update()`` requires Newton's ``contacts`` buffer.
   * - ``sensor(model, binding)``
     - Returns a :class:`~newton_neural.contacts.ContactSensorBase` whose ``update(state, contacts)`` writes into
       reused buffers and returns ``[world_count, *shape]`` tensors. If bounded by capacity, set ``sensor.overflow``
       to an int32 Warp array to expose ``CONTACT_ROW_OVERFLOW``.
   * - ``encoder()``
     - Returns an ``nn.Module`` implementing ``forward(frames) -> [..., code_count]`` and ``fit_inputs(frames)``.
   * - ``recomputable``, ``evaluate(system, frames)``
     - Set ``recomputable = True`` and implement ``evaluate`` if the signals depend solely on joint state (enables
       recomputation during noisy training and open-loop rollouts).
   * - ``config()``
     - Constructor kwargs as plain data.

Because floating-base system features omit global base height, this example provides root-body height above ``z = 0``
as a recomputable contact input:

.. testcode::

    import warp as wp

    from newton_neural import Normalizer
    from newton_neural.contacts import ContactInputBase, ContactSensorBase
    from newton_neural.systems import BindingArticulation


    class ContactInputHeight(ContactInputBase):
        """Floating-base root height above ``z = 0`` (signal ``root_height`` ``[1]`` [m]).

        Args:
            code_count: Output width of the contact code.
        """

        recomputable = True

        def __init__(self, code_count: int = 8) -> None:
            self.code_count = code_count
            self.signals = {"root_height": SignalSpec("root_height", (1,), SignalRole.CONTACT, unit="m")}

        def sensor(self, model: newton.Model, binding: BindingArticulation) -> ContactSensorBase:
            return ContactSensorHeight(binding)

        def evaluate(self, system: SystemBase, frames: Frames) -> Frames:
            # Floating-base joint_q starts with root position (x, y, z).
            return {"root_height": frames["joint_q"][..., 2:3]}

        def encoder(self) -> nn.Module:
            return ContactEncoderHeight(self.code_count)

        def config(self) -> dict[str, Any]:
            return {"code_count": self.code_count}


    class ContactSensorHeight(ContactSensorBase):
        """Extracts root body z-coordinate per world from ``state.body_q``."""

        def __init__(self, binding: BindingArticulation) -> None:
            root_bodies = wp.to_torch(binding.body_ids)[:, binding.system.root_body]
            self.root_bodies = root_bodies.long()
            self.height = torch.zeros(binding.world_count, 1, device=root_bodies.device)

        def update(self, state: newton.State, contacts: newton.Contacts | None) -> Frames:
            self.height.copy_(wp.to_torch(state.body_q)[self.root_bodies, 2:3])
            return {"root_height": self.height}


    class ContactEncoderHeight(nn.Module):
        """Normalizes root height and projects it to ``code_count``."""

        def __init__(self, code_count: int) -> None:
            super().__init__()
            self.normalizer = Normalizer(1)
            self.project = nn.Linear(1, code_count)

        def forward(self, frames: Frames) -> Tensor:
            return self.project(self.normalizer.normalize(frames["root_height"]))

        def fit_inputs(self, frames: Frames) -> None:
            self.normalizer.update(frames["root_height"])

For heavier sensors, write a Warp kernel and call :func:`~newton_neural.systems.robot_frame` so sensed quantities
share the same coordinate frame as :class:`~newton_neural.systems.SystemArticulation` features.

For any recomputable contact input, verify that ``evaluate()`` matches ``sensor.update()`` using
:func:`~newton_neural.testing.check_contact_input_recompute`:

.. testcode::

    from newton_neural.testing import check_contact_input_recompute

    robot = newton.ModelBuilder()
    robot.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform((0.0, 0.0, 0.7), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,  # URDF inertias are too small for stable simulation
    )
    quadruped = replicate(robot, ground=True)
    quadruped_system = SystemArticulation.from_model(quadruped, control=("joint_f",))

    check_contact_input_recompute(ContactInputHeight(), quadruped_system, quadruped)

Use ``ContactInputHeight`` in training and :class:`~newton_neural.solvers.SolverNeRD` (where the contact code enters
the network is controlled by the model family; see :ref:`contact-injections` for :class:`.FlashNeRD`):

.. testcode::

    nerd = FlashNeRD(quadruped_system, backbone=BackbonePair(), contact_input=ContactInputHeight())
    nerd.to(device)
    loaded, following = try_out(nerd, record(nerd, quadruped, "data/height"), "height.pt")
    print(type(loaded.contact_input).__name__, list(loaded.signals))

.. testoutput::

    ContactInputHeight ['joint_q', 'joint_qd', 'joint_f', 'root_height']

.. testcode::

    from newton_neural.solvers import SolverNeRD

    solver = SolverNeRD(quadruped, "height.pt")
    state, control = quadruped.state(), quadruped.control()
    solver.reset(state)  # Updates body transforms before first step
    for _ in range(4):
        solver.step(state, state, control, None, 0.02)

Custom system
-------------

A system (:class:`~newton_neural.systems.SystemBase`) defines (1) the pure-PyTorch mapping between raw frames and
network features/increments (buffers only, no trainable weights), and (2) an optional Newton binding via ``bind()``.

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Member
     - Contract
   * - ``signals``
     - Ordered ``dict[str, SignalSpec]`` of ``STATE`` signals followed by ``CONTROL`` signals.
   * - ``feature_count``, ``increment_count``
     - Feature and state-increment widths.
   * - ``features(frames)``
     - Maps frames (single-frame or windowed) to ``[..., feature_count]``.
   * - ``increment(current, following)``
     - Maps consecutive frames to ``[..., increment_count]``.
   * - ``integrate(current, increment)``
     - Exact inverse of ``increment``; returns next ``STATE`` signals.
   * - ``perturb(current, noise)``
     - Applies state noise for noisy training (defaults to ``integrate(current, noise)``).
   * - ``error(reference, other)``
     - Per-coordinate error dictionary used by evaluation (defaults to ``|reference - other|``).
   * - ``bind(model, articulation)``
     - Returns a :class:`~newton_neural.systems.BindingBase` for reading/writing Newton arrays (optional if stepping
       strictly on tensor frames).
   * - ``config()``
     - Constructor kwargs as plain data.

All frame methods must tolerate extra keys in ``frames`` (e.g., contact signals) and support arbitrary leading batch
dimensions. Keep ``features`` and ``integrate`` free of data-dependent control flow and matrix multiplications/linear
layers so they compile cleanly and stay in IEEE FP32 across all deployment precisions.

Subclassing ``SystemArticulation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To customize feature representation for an articulated robot while keeping Newton binding, solvers, and contact
inputs working automatically, subclass :class:`~newton_neural.systems.SystemArticulation`. Below, continuous
revolute angles are encoded as ``(sin(q), cos(q))`` so full rotations produce identical features:

.. testcode::

    class SystemArticulationPeriodic(SystemArticulation):
        """Fixed-base force-controlled articulation with ``(sin, cos)`` features for revolute joints."""

        def __init__(self, spec: SystemArticulation.Spec) -> None:
            super().__init__(spec)
            if spec.floating_base or spec.control != ("joint_f",):
                raise ValueError("SystemArticulationPeriodic supports a fixed base under joint forces only")
            kinds = [joint.type for joint in spec.joints if joint.actuated]
            angles = [index for index, kind in enumerate(kinds) if kind == "revolute"]
            lengths = [index for index, kind in enumerate(kinds) if kind == "prismatic"]
            # Non-persistent buffers are rebuilt from spec on load rather than stored in checkpoints.
            self.register_buffer("angles", torch.tensor(angles, dtype=torch.long), persistent=False)
            self.register_buffer("lengths", torch.tensor(lengths, dtype=torch.long), persistent=False)
            self.feature_count += len(angles)

        def features(self, frames: Frames) -> Tensor:
            joint_q = frames["joint_q"]
            angles, lengths = joint_q[..., self.angles], joint_q[..., self.lengths]
            parts = (angles.sin(), angles.cos(), lengths, frames["joint_qd"], frames["joint_f"])
            return torch.cat(parts, dim=-1)

.. testcode::

    periodic = SystemArticulationPeriodic.from_model(cartpole, control=("joint_f",))
    nerd = FlashNeRD(periodic, backbone=BackbonePair()).to(device)
    loaded, following = try_out(nerd, record(nerd, cartpole, "data/periodic"), "periodic.pt")
    print(type(loaded.system).__name__, loaded.system.feature_count, cartpole_system.feature_count)

.. testoutput::

    SystemArticulationPeriodic 11 9

Because ``bind()`` is inherited, checkpoints deploy straight into :class:`~newton_neural.solvers.SolverNeRD`:

.. testcode::

    solver = SolverNeRD(cartpole, "periodic.pt")
    state, control = cartpole.state(), cartpole.control()
    solver.reset(state)
    solver.step(state, state, control, None, 0.02)

Non-articulated systems (``SystemBase``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non-articulated dynamics (such as a small particle system), subclass :class:`~newton_neural.systems.SystemBase`
directly:

.. testcode::

    class SystemParticles(SystemBase):
        """Small particle system represented as mean-centered positions and velocities.

        Args:
            particle_count: Number of particles per world.
        """

        def __init__(self, particle_count: int) -> None:
            super().__init__()
            self.particle_count = particle_count
            shape = (particle_count, 3)
            self.signals = {
                "particle_q": SignalSpec("particle_q", shape, SignalRole.STATE, unit="m"),
                "particle_qd": SignalSpec("particle_qd", shape, SignalRole.STATE, unit="m/s"),
            }
            self.feature_count = self.increment_count = 6 * particle_count

        def features(self, frames: Frames) -> Tensor:
            position = frames["particle_q"]
            # Center positions so features are translation-invariant.
            centered = position - position.mean(dim=-2, keepdim=True)
            return torch.cat((centered.flatten(-2), frames["particle_qd"].flatten(-2)), dim=-1)

        def increment(self, current: Frames, following: Frames) -> Tensor:
            position_change = following["particle_q"] - current["particle_q"]
            velocity_change = following["particle_qd"] - current["particle_qd"]
            return torch.cat((position_change.flatten(-2), velocity_change.flatten(-2)), dim=-1)

        def integrate(self, current: Frames, increment: Tensor) -> Frames:
            changes = increment.unflatten(-1, (2, self.particle_count, 3))
            position_change, velocity_change = changes.unbind(dim=-3)
            return {
                "particle_q": current["particle_q"] + position_change,
                "particle_qd": current["particle_qd"] + velocity_change,
            }

        def config(self) -> dict[str, Any]:
            return {"particle_count": self.particle_count}

Verify that ``integrate`` inverts ``increment``:

.. testcode::

    particles = SystemParticles(4)
    current, following = (
        {name: torch.randn(2, *spec.shape) for name, spec in particles.signals.items()} for _ in range(2)
    )
    integrated = particles.integrate(current, particles.increment(current, following))
    print(all(torch.allclose(integrated[name], following[name], atol=1e-6) for name in following))

.. testoutput::

    True

That mapping alone is enough to train, save, load, step with :class:`.NeRDStepper`, and run
:func:`~newton_neural.evaluation.rollout_error` using datasets written with
:class:`~newton_neural.data.TrajectoryWriter`:

.. testcode::

    from newton_neural.data import TrajectoryWriter

    FRAME_COUNT = 64

    chain = newton.ModelBuilder()
    for index in range(4):
        chain.add_particle((0.1 * index, 0.0, 1.0), (0.0, 0.0, 0.0), mass=0.0 if index == 0 else 0.1)
    for index in range(3):
        chain.add_spring(index, index + 1, ke=1.0e3, kd=1.0, control=0.0)
    scene = newton.ModelBuilder()
    scene.replicate(chain, WORLD_COUNT)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)

    state, next_state = model.state(), model.state()
    velocity = wp.to_torch(state.particle_qd).view(WORLD_COUNT, 4, 3)
    velocity[:, 1:] = torch.randn(WORLD_COUNT, 3, 3, device=device)
    frames = {"particle_q": [], "particle_qd": []}
    for _ in range(FRAME_COUNT):
        for name, values in frames.items():
            values.append(wp.to_torch(getattr(state, name)).view(WORLD_COUNT, 4, 3).clone())
        state.clear_forces()
        solver.step(state, next_state, None, None, 0.02)
        state, next_state = next_state, state

    episode_start = torch.zeros(WORLD_COUNT, FRAME_COUNT, dtype=torch.bool)
    episode_start[:, 0] = True
    valid = torch.ones(WORLD_COUNT, FRAME_COUNT, dtype=torch.bool)
    stacked = {name: torch.stack(values, dim=1) for name, values in frames.items()}
    with TrajectoryWriter("data/chain", particles.signals, frame_dt=0.02) as writer:
        writer.write(stacked, episode_start, valid)

    nerd = FlashNeRD(particles, backbone=BackbonePair()).to(device)
    loaded, following = try_out(nerd, TrajectoryDataset("data/chain"), "chain.pt")
    print(type(loaded.system).__name__, {name: tuple(value.shape) for name, value in following.items()})

.. testoutput::

    SystemParticles {'particle_q': (8, 4, 3), 'particle_qd': (8, 4, 3)}

.. testcode::
    :hide:

    from newton_neural.evaluation import rollout_error

    errors = rollout_error(loaded, TrajectoryWindows("data/chain", length=4, device=device), horizons=(8,))
    assert errors[8].nrmse > 0.0

*(Flattening all particles into one feature vector suits small particle sets, not high-resolution meshes or cloth.)*

To integrate a non-articulated system into :class:`~newton_neural.data.TrajectoryRecorder`, live audits, and a
custom Newton solver, implement three additional pieces (since :class:`~newton_neural.data.SimulationNewton`,
:class:`~newton_neural.solvers.SolverNeRD`, :class:`~newton_neural.solvers.SolverNeRDMuJoCo`, and built-in contact
inputs target :class:`~newton_neural.systems.BindingArticulation`):

1. A :class:`~newton_neural.systems.BindingBase` subclass (returned by ``system.bind()``) providing
   ``system``, ``world_count``, ``read_state``, ``read_control``, ``write_state``, ``write_control``,
   ``state_bounds``, and ``command_bounds``.
2. A :class:`~newton_neural.data.SimulationBase` subclass wrapping your reference solver with ``step`` and
   ``restart`` (enabling :class:`~newton_neural.data.TrajectoryRecorder`, :meth:`.NeRDBase.sources`, and live
   audits).
3. A Newton :class:`newton.solvers.SolverBase` subclass that reads frames through the binding, steps
   :class:`.NeRDStepper`, and writes predictions back.

.. testcode::
    :hide:

    from newton_neural.evaluation import audit_rest
    from newton_neural.systems import BindingBase


    class BindingParticles(BindingBase):
        """Minimal binding exposing particle arrays as views."""

        def __init__(self, system: SystemParticles, device: torch.device) -> None:
            self.system, self.world_count, self.device = system, WORLD_COUNT, device

        def read_state(self, state: newton.State) -> Frames:
            shape = (WORLD_COUNT, self.system.particle_count, 3)
            return {name: wp.to_torch(getattr(state, name)).view(shape) for name in self.system.signals}

        def read_control(self, control: newton.Control) -> Frames:
            return {}

        def write_state(self, state: newton.State, frames: Frames, world_mask: Tensor | None = None) -> None:
            worlds = slice(None) if world_mask is None else world_mask
            for name, value in self.read_state(state).items():
                value[worlds] = frames[name][worlds]

        def write_control(self, control: newton.Control, frames: Frames) -> None:
            pass

        def state_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
            one = torch.ones(self.system.particle_count, 3, device=self.device)
            return {name: (-one, one) for name in self.system.signals}

        def command_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
            return {}


    class SimulationParticles(SimulationBase):
        """Minimal XPBD particle simulation loop."""

        def __init__(self, model: newton.Model, system: SystemParticles) -> None:
            self.model, self.solver = model, newton.solvers.SolverXPBD(model)
            self.device = torch.device(device)
            self.binding = BindingParticles(system, self.device)
            self.world_count, self.frame_dt, self.substep_count = WORLD_COUNT, 0.02, 1
            self.state, self.next_state = model.state(), model.state()
            self.control, self.contacts = model.control(), None

        def step(self, command: Frames | None = None) -> None:
            self.state.clear_forces()
            self.solver.step(self.state, self.next_state, self.control, None, self.frame_dt)
            self.state, self.next_state = self.next_state, self.state

        def restart(self, frames: Frames, world_mask: Tensor | None = None) -> None:
            self.binding.write_state(self.state, frames, world_mask)


    simulation = SimulationParticles(model, particles)
    simulation.restart(following, torch.arange(WORLD_COUNT, device=device) < 2)
    written = simulation.binding.read_state(simulation.state)["particle_q"]
    assert torch.equal(written[:2], following["particle_q"][:2])
    assert not torch.equal(written[2:], following["particle_q"][2:])
    sources = nerd.sources(simulation)
    assert [list(source.signals) for source in sources] == [["particle_q", "particle_qd"], []]
    assert audit_rest(simulation, frame_count=4).drift_max.keys() == particles.signals.keys()
    recorded = TrajectoryRecorder(simulation, sources, episode_frames=(4, 8)).record("data/chain_recorded", 8)
    assert list(recorded.signals) == ["particle_q", "particle_qd"]

Custom model family
-------------------

To introduce a new network architecture, subclass :class:`~newton_neural.models.NeRDBase`. The base class already
handles signal assembly, label extraction, state integration, recorder sources, output normalization, and
checkpointing. Your subclass provides only the architecture and streaming contract:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Member
     - Contract
   * - ``__init__``
     - Call ``super().__init__(system, contact_input=..., prediction=..., frame_dt=...)`` before building submodules
       (initializes parts, ``self.signals``, and ``self.output_normalizer``).
   * - ``forward(frames)``
     - Windowed training forward pass returning normalized predictions ``[B, T, output_count]`` (frame 0 = episode
       start).
   * - ``predict(frames, history)``
     - Streaming single-frame inference returning predictions ``[B, output_count]`` in physical units while updating
       ``history`` in place (no data-dependent control flow or allocations).
   * - ``history(world_count)``, ``reset_history(history, world_mask)``
     - Allocates fixed-shape per-world history buffers on ``self.device`` and clears masked worlds (return ``{}`` if
       stateless).
   * - ``receptive_field``
     - Number of frames per prediction (including current), or ``None`` if unbounded.
   * - ``fit_inputs(frames)``
     - Updates input/encoder normalizer statistics (called by :meth:`.NeRDBase.fit_normalization`).
   * - ``settings()``
     - Plain-data dict of non-part constructor kwargs stored in checkpoints and passed back on ``load()``.
   * - ``part_kinds``
     - Optional class dict extending ``NeRDBase.part_kinds`` with family-specific part kinds (e.g., ``"backbone"`` in
       :class:`.FlashNeRD`) so checkpoints serialize and rebuild them automatically.

If your family accepts a ``contact_input``, instantiate ``self.contact_encoder = contact_input.encoder()``, call
``self.contact_encoder.fit_inputs(frames)`` inside ``fit_inputs``, and inject ``self.contact_encoder(frames)``
wherever your network needs it.

Below is a minimal stateless family: a single linear layer mapping current frame features to outputs.

.. testcode::

    from newton_neural import Normalizer
    from newton_neural.models import NeRDBase
    from newton_neural.predictions import PredictionBase


    class LinearNeRD(NeRDBase):
        """Stateless linear dynamics model over current-frame features."""

        receptive_field = 1

        def __init__(
            self, system: SystemBase, *, prediction: PredictionBase | None = None, frame_dt: float | None = None
        ) -> None:
            super().__init__(system, prediction=prediction, frame_dt=frame_dt)
            self.feature_normalizer = Normalizer(system.feature_count)
            self.linear = nn.Linear(system.feature_count, self.prediction.output_count(system))

        def fit_inputs(self, frames: Frames) -> None:
            self.feature_normalizer.update(self.system.features(frames))

        def forward(self, frames: Frames) -> Tensor:
            return self.linear(self.feature_normalizer.normalize(self.system.features(frames)))

        def predict(self, frames: Frames, history: dict[str, Tensor]) -> Tensor:
            return self.output_normalizer.denormalize(self(frames))

        def history(self, world_count: int) -> dict[str, Tensor]:
            return {}

        def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
            pass

        def settings(self) -> dict[str, Any]:
            return {}

Record a dataset, fit normalizers, and verify streaming/window parity with
:func:`~newton_neural.testing.check_nerd_streaming` (which checks forward/predict agreement, mid-episode resets even
with NaN history, and history buffer invariance):

.. testcode::

    nerd = LinearNeRD(cartpole_system).to(device)
    dataset = record(nerd, cartpole, "data/linear")
    windows = TrajectoryWindows(dataset, length=4, device=device)
    nerd.fit_normalization(windows.loader(32))
    batch = next(windows.loader(WORLD_COUNT))

.. testcode::

    from newton_neural.testing import check_nerd_streaming

    check_nerd_streaming(nerd, batch.frames)

Train, save, reload through :meth:`.NeRDBase.load`, and deploy via :class:`~newton_neural.solvers.SolverNeRD` (use
``"adamw"`` because ``LinearNeRD`` declares no ``hidden_weights()`` for Muon):

.. testcode::

    loss, optimizer = LossOneStep(), create_optimizer(nerd, "adamw")
    optimizer.zero_grad()
    loss(nerd, batch).backward()
    optimizer.step()
    nerd.save("linear.pt")

    loaded = NeRDBase.load("linear.pt", device=device)
    solver = SolverNeRD(cartpole, "linear.pt")
    state, control = cartpole.state(), cartpole.control()
    solver.reset(state)
    solver.step(state, state, control, None, 0.02)
    print(type(loaded).__name__, type(solver.nerd).__name__, loaded.receptive_field)

.. testoutput::

    LinearNeRD LinearNeRD 1

.. testcode::
    :hide:

    assert type(LinearNeRD.load("linear.pt")) is LinearNeRD
    try:
        FlashNeRD.load("linear.pt")
    except ValueError as error:
        assert "LinearNeRD" in str(error) and "FlashNeRD" in str(error)
    else:
        raise AssertionError("FlashNeRD.load accepted the file of another family")
    try:
        create_optimizer(nerd)
    except ValueError:
        pass
    else:
        raise AssertionError("Muon accepted a model that names no hidden weights")
    errors = rollout_error(loaded, windows, horizons=(2,))
    assert errors[2].nrmse >= 0.0

Recording custom signals
------------------------

:class:`~newton_neural.data.TrajectoryRecorder` accepts either a model (via :meth:`.NeRDBase.sources`) or an
explicit list of :class:`~newton_neural.data.SignalSource` objects. Each source declares its signal specs, a reader
returning freshly cloned ``[world_count, *shape]`` tensors, and whether it runs before or after each simulation step
(``after_step``).

Append custom sources alongside ``nerd.sources(simulation)`` to log extra diagnostics (such as body positions) into
the same dataset; training ignores unused signals automatically:

.. testcode::

    import warp as wp

    from newton_neural import SignalRole, SignalSpec
    from newton_neural.data import SignalSource

    nerd = FlashNeRD(cartpole_system).to(device)
    mujoco = newton.solvers.SolverMuJoCo(cartpole)
    simulation = SimulationNewton(cartpole, mujoco, cartpole_system, frame_dt=0.02, substep_count=4)
    body_count = cartpole.body_count // WORLD_COUNT


    def read_positions() -> Frames:
        body_q = wp.to_torch(simulation.state.body_q).view(WORLD_COUNT, body_count, 7)
        return {"body_position": body_q[..., :3].clone()}


    spec = SignalSpec("body_position", (body_count, 3), SignalRole.LABEL, unit="m")
    positions = SignalSource({spec.name: spec}, read_positions)
    recorder = TrajectoryRecorder(simulation, [*nerd.sources(simulation), positions], episode_frames=(16, 32))
    dataset = recorder.record("data/positions", frame_count=32)
    print(list(dataset.signals))

.. testoutput::

    ['joint_q', 'joint_qd', 'joint_f', 'body_position']

.. testcode::
    :hide:

    windows = TrajectoryWindows(dataset, length=13, device=device)
    nerd.fit_normalization(windows.loader(8))
    assert LossOneStep()(nerd, next(windows.loader(8))).isfinite()

Contributing extensions upstream
--------------------------------

When contributing a class to ``newton-neural``, place implementations under ``newton_neural/_src/`` and re-export
public symbols via ``__all__`` in the top-level module (e.g., :mod:`newton_neural.models`). Check layer boundaries
with ``uv run lint-imports`` (see :doc:`development`).
