.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton_neural.evaluation

Evaluation
==========

A low one-step training loss does not guarantee accurate closed-loop simulation: during deployment, a model feeds
its own predictions back as inputs, allowing small errors to compound. :mod:`newton_neural.evaluation` provides
three complementary checks:

.. list-table::
   :widths: 22 48 30
   :header-rows: 1

   * - Function
     - What it tests
     - Input required
   * - :func:`rollout_error`
     - Open-loop drift over multi-step horizons under recorded controls.
     - :class:`~newton_neural.data.TrajectoryWindows`
   * - :func:`audit_rest`
     - Creep (drift) and jitter (vibration) when controls are held constant.
     - One :class:`~newton_neural.data.SimulationBase`
   * - :func:`audit_paired`
     - Live divergence between candidate and reference solvers from identical states under shared commands.
     - Two :class:`~newton_neural.data.SimulationBase` instances

.. testsetup::

    import itertools

    from guide_setup import cartpole_recording
    from newton_neural.data import TrajectoryWindows
    from newton_neural.training import LossOneStep, create_optimizer

    # Cartpole recording and brief training from the previous pages (fewer worlds and steps).
    model, system, simulation, nerd, reach, dataset = cartpole_recording(frame_count=160)
    train, held_out = TrajectoryWindows(dataset, length=16, device=nerd.device).split(fraction=0.1)
    nerd.fit_normalization(train.loader(256))
    loss, optimizer = LossOneStep(noise=0.1), create_optimizer(nerd)
    for batch in itertools.islice(train.loader(256, epochs=None), 5):
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()

Open-loop rollout error
-----------------------

:func:`rollout_error` rolls a model forward autoregressively using recorded controls and compares the predicted
states against the recording at each requested horizon. Use this metric for checkpoint selection on held-out worlds
and on policy-generated validation recordings.

Continuing from :doc:`training` with the trained model ``nerd`` and ``held_out`` windows:

.. testcode::

    from newton_neural.evaluation import rollout_error

    errors = rollout_error(nerd, held_out, horizons=(1, 16, 64))
    for horizon, statistics in errors.items():
        print(horizon, statistics.p50["joint_q"], statistics.p99["joint_q"], statistics.nrmse)
    after_reset = rollout_error(nerd, held_out, horizons=(1, 16, 64), from_episode_start=True)

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    1 ...
    16 ...
    64 ...

Each horizon maps to a :class:`RolloutStatistics` object:

- ``mean``, ``p50``, ``p90``, ``p99``: Per-signal absolute state errors (via ``system.error``, in each signal's SI
  unit; floating-base orientation is reported as a single rotation angle in radians), aggregated across rollouts and
  coordinates. Because rollout errors are heavy-tailed, inspect ``p50``/``p90``/``p99`` alongside ``mean``.
- ``nrmse``: Prefix-averaged normalized RMSE over frames :math:`1 \dots H`, channels, and rollouts:
  ``system.increment(recorded, predicted) / nerd.output_normalizer.std``. A value of ``1.0`` corresponds to an RMS
  error equal to one typical single-frame state change in the training set.

Key options and caveats:

- **Window length:** ``windows`` can have any length; :func:`rollout_error` re-slices rollouts of length
  ``history + max(horizons)`` from the same worlds.
- **Primed history vs. fresh reset:** By default, rollouts start anywhere in an episode after priming the history
  with ``receptive_field - 1`` recorded frames. Passing ``from_episode_start=True`` evaluates rollouts starting at
  frame 0 with an empty history, matching solver behavior immediately after a reset. RL environments reset frequently,
  so measure both.
- **Subsampling:** ``rollout_count`` defaults to ``4096`` evenly spaced start points (since adjacent-frame starts
  overlap heavily). Pass ``rollout_count=None`` to evaluate every possible rollout.
- **Contact inputs:** Recomputable inputs (:class:`~newton_neural.contacts.ContactInputProbes`) are re-evaluated at
  every predicted state. Non-recomputable inputs (:class:`~newton_neural.contacts.ContactInputSet`) reuse the
  recorded contact rows from the ground-truth trajectory, so long-horizon rollout errors are less representative
  once the predicted trajectory diverges; pair them with :func:`audit_paired`.
- **Deployment parity:** Pass ``precision`` and ``compile`` to match deployment settings (defaults: checkpoint
  precision, compiled on CUDA). Compare against ``precision="fp32"`` before shipping reduced precision
  (:doc:`performance`).

Rest stability audit
--------------------

Models with low rollout error can still slowly creep or oscillate at rest, which RL policies quickly exploit.
:func:`audit_rest` holds the current controls fixed for ``frame_count`` steps and reports two metrics per state
signal:

- **Drift** (``drift_mean``, ``drift_max``): Absolute state difference between frame ``0`` and frame ``frame_count``.
- **Vibration** (``vibration_mean``, ``vibration_max``): RMS residual of each coordinate over time after subtracting
  the least-squares linear trend, separating high-frequency jitter from steady creep. The trend is fitted to the
  system's increments from frame ``0``, so a floating base is detrended as a rotation, as long as it turns less
  than half a turn during the audit.

Bring both the reference and candidate simulations to a steady pose (e.g., holding standing targets) and compare them
side by side. Below, ``model``, ``system``, and ``reach`` come from :doc:`recording`:

.. testcode::

    import newton

    from newton_neural.data import CommandHeld, SimulationNewton
    from newton_neural.evaluation import audit_paired, audit_rest
    from newton_neural.solvers import SolverNeRD

    mujoco = newton.solvers.SolverMuJoCo(model)
    reference = SimulationNewton(model, mujoco, system, frame_dt=0.02, substep_count=4)
    candidate = SimulationNewton(model, SolverNeRD(model, nerd), system, frame_dt=0.02)

    for simulation in (reference, candidate):
        rest = audit_rest(simulation, frame_count=100)
        print(rest.drift_max["joint_q"], rest.vibration_max["joint_q"])

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    ...

Paired live audit
-----------------

:func:`audit_paired` restarts both simulations from the reference's current state, steps both live under the same
command stream for ``frame_count`` frames, and returns per-frame ``mean`` and ``max`` state differences (shape
``[frame_count]`` per signal). The state is that of the robot, so other bodies of a scene, such as a free object,
start where a restart puts them in both simulations:

.. testcode::

    commands = CommandHeld((10, 50), low=-reach, high=reach)
    paired = audit_paired(reference, candidate, commands=commands, frame_count=100)
    print(paired.mean["joint_q"][-1].item(), paired.max["joint_q"][-1].item())

.. testoutput::
    :hide:
    :options: +ELLIPSIS

    ...

Because :func:`audit_paired` runs live simulations (including live collision detection for
:class:`~newton_neural.contacts.ContactInputSet`), it requires no pre-recorded dataset and can probe arbitrary states
reached by stepping the reference under an RL policy.
