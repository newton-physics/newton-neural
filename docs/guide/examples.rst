.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Examples
========

Four self-contained examples ship inside the package. Each script records a dataset from MuJoCo Warp, trains a
:class:`~newton_neural.models.FlashNeRD` model, and runs the trained model side by side against MuJoCo Warp in
Newton's viewer. No external downloads are required, and all hyperparameters live as top-level constants so you can
read or adapt them easily.

.. code-block:: console

    uv run -m newton_neural.examples                                # List available examples
    uv run --extra examples -m newton_neural.examples quickstart    # Run with the OpenGL viewer
    uv run -m newton_neural.examples quickstart --viewer null       # Run headless (no display or extra deps)

Available examples
------------------

.. list-table::
   :widths: 18 28 54
   :header-rows: 1

   * - Example
     - Robot / scene
     - What it demonstrates
   * - :github:`quickstart <newton_neural/examples/example_quickstart.py>`
     - Cartpole with three position-controlled joints
     - Complete Record → Train → Evaluate → Deploy pipeline in one file with an explicit PyTorch training loop.
   * - :github:`contact_probes <newton_neural/examples/example_contact_probes.py>`
     - Quadruped on flat ground
     - Fixed ground-plane contact sensing at known foot locations via
       :class:`~newton_neural.contacts.ContactInputProbes` (no collision detection). See :doc:`contact`.
   * - :github:`contact_set <newton_neural/examples/example_contact_set.py>`
     - Quadruped with randomized box platforms per world
     - General collision-detected contact sensing via
       :class:`~newton_neural.contacts.ContactInputSet`, captured together with collision detection in a single CUDA
       graph.
   * - :github:`coupling <newton_neural/examples/example_coupling.py>`
     - Two-link arm pushing a free box
     - Two-way coupling via :class:`~newton_neural.solvers.SolverNeRDMuJoCo`, where the model predicts free-space
       joint accelerations and MuJoCo Warp resolves arm-box contact. See :doc:`coupling`.

Start with ``quickstart``; the other three examples follow the same structure while varying one component. Each
script's module docstring explains its setup, visual success criteria, and verification command.

On the first run, an example records data and trains before launching the viewer; subsequent runs load the cached
checkpoint immediately. All examples also run on CPU, though recording and training are substantially slower without
a GPU.

For RL policy training examples built on ``newton-neural``, see
`Isaac Lab <https://github.com/isaac-sim/IsaacLab>`__.

CLI options
-----------

Every example accepts Newton's standard example flags plus three library-specific options:

.. list-table::
   :widths: 26 74
   :header-rows: 1

   * - Option
     - Description
   * - ``--world-count``
     - Number of parallel worlds stepped side by side by each solver. Defaults: ``9`` (``quickstart``), ``16``
       (``contact_probes``), ``16`` (``contact_set``), ``4`` (``coupling``). Camera positions are tuned for the
       defaults.
   * - ``--train-steps``
     - Number of optimizer steps. Defaults: ``2000`` (``4000`` for ``coupling``).
   * - ``--cache-dir``
     - Directory for cached datasets and checkpoints.
   * - ``--viewer``
     - Viewer backend: ``gl`` (default), ``null`` (headless), ``usd``, ``rerun``, ``viser``, or ``rtx``.
   * - ``--num-frames``
     - Number of frames to simulate in headless mode (default: ``100``).
   * - ``--test``
     - Evaluate the example's numerical accuracy checks after the final frame and exit non-zero on failure.
   * - ``--device``
     - Target Warp device (e.g., ``cuda:0`` or ``cpu``).

For headless verification (e.g., in CI or on a server without a display), combine ``--viewer null`` and ``--test``:

.. code-block:: console

    uv run -m newton_neural.examples quickstart --viewer null --num-frames 200 --test

Caching behavior
----------------

Datasets and trained checkpoints are cached under ``$XDG_CACHE_HOME/newton-neural/examples`` (or
``~/.cache/newton-neural/examples`` if ``XDG_CACHE_HOME`` is unset) unless overridden with ``--cache-dir``. Each run
prints its active cache path:

.. code-block:: text

    <cache-dir>/<name>/dataset/
    <cache-dir>/<name>/flashnerd_<train-steps>_steps.pt

- If the checkpoint exists, recording and training are skipped.
- Changing ``--train-steps`` reuses the cached dataset and trains a new checkpoint.
- Editing constants in an example file does **not** invalidate the cache; delete ``<cache-dir>/<name>/`` manually to
  re-record and retrain.
- If a recording is interrupted before completion, the partial dataset remains on disk and subsequent runs fail with
  an error pointing to that directory until you delete it.

Viewer notes
------------

The interactive OpenGL viewer requires ``pyglet`` and ``imgui-bundle``, installed via the ``examples`` extra
(``uv run --extra examples``). Running with ``--viewer null`` needs no extra packages and no display.

Each example renders two separate viewer layers (``MuJoCo Warp`` and ``FlashNeRD``) sharing the same Newton model and
offset spatially so you can compare the reference and learned twins side by side. Interactive picking forces in the
viewer are not forwarded to the solvers because learned models do not accept external perturbation forces.
