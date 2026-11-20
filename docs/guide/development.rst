.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Development
===========

``newton-neural`` is part of Newton (a Linux Foundation project). For contribution workflow, CLA requirements, and
PR expectations, see `CONTRIBUTING.md <https://github.com/newton-physics/newton-neural/blob/main/CONTRIBUTING.md>`__.
Code style and API design follow Newton's `coding guidelines
<https://github.com/newton-physics/newton/blob/main/CODING_GUIDELINES.rst>`__.

Environment setup
-----------------

.. code-block:: console

    git clone https://github.com/newton-physics/newton-neural.git
    cd newton-neural
    uv sync

``uv sync`` creates ``.venv`` from ``uv.lock`` with the package installed in editable mode plus the ``dev`` group
(``ruff``, ``import-linter``, ``coverage``).

Running tests
-------------

Unit tests live under ``newton_neural/tests`` and use ``unittest``. Every test runs on CPU and also on CUDA when a
GPU is available:

.. code-block:: console

    uv run -m newton_neural.tests
    uv run -m newton_neural.tests -k test_stepper

Every bug fix or feature should include a unit test that fails without the change. New model families, backbones,
and contact inputs must also pass the conformance checks in :mod:`newton_neural.testing` (:doc:`extending`).

Linting, formatting, and architecture checks
--------------------------------------------

Run all pre-commit hooks (Ruff lint/format, ``uv.lock`` check, ``typos`` spellcheck, Warp array syntax, and import
layering):

.. code-block:: console

    uvx pre-commit run -a

Run the import-layer check standalone:

.. code-block:: console

    uv run lint-imports

Architectural boundaries enforced across ``newton_neural/_src`` (configured in ``pyproject.toml``):

- **Strict layering:** Lower layers must never import higher layers at runtime (``TYPE_CHECKING``-only imports are
  permitted).
- **Explicit public surface:** Only modules directly under ``newton_neural/`` and symbols listed in ``__all__`` are
  public API.
- **Family-agnostic tooling:** Recording, losses, steppers, evaluation, and solvers depend only on
  :class:`~newton_neural.models.NeRDBase` and never import a specific model family.
- **Network-agnostic contact inputs:** Contact inputs sense and encode contact without hardcoding network injection
  points; each model family decides where contact codes enter its network.

License headers
---------------

Every Python file requires an Apache-2.0 SPDX header and every ``.rst`` file requires a CC-BY-4.0 SPDX header
(using the year the file was created):

.. code-block:: python

    # SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
    # SPDX-License-Identifier: Apache-2.0

.. code-block:: rst

    .. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
    .. SPDX-License-Identifier: CC-BY-4.0

Verify headers locally with `Apache SkyWalking Eyes <https://github.com/apache/skywalking-eyes>`__:

.. code-block:: console

    docker run -it --rm -v $(pwd):/github/workspace apache/skywalking-eyes header check
    docker run -it --rm -v $(pwd):/github/workspace apache/skywalking-eyes -c .licenserc-docs.yaml header check

Changelog fragments
-------------------

User-facing pull requests add a fragment file under ``changelog/`` rather than editing ``CHANGELOG.md`` directly
(see :github:`changelog/README.md`). Preview the rendered notes with Towncrier:

.. code-block:: console

    uvx --from towncrier==25.8.0 towncrier build --draft --version X.Y.Z --date YYYY-MM-DD

Documentation
-------------

Sphinx builds treat all warnings as errors (``-W``):

.. code-block:: console

    uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

Run all guide code blocks via Sphinx doctest (executes on GPU if available, otherwise CPU):

.. code-block:: console

    uv run --group docs sphinx-build -W --keep-going -b doctest docs docs/_build/doctest

API reference pages under ``docs/api/`` are generated from each public module's ``__all__`` by
:github:`docs/generate_api.py` (invoked automatically during every Sphinx build) and committed to git. Whenever you
change public exports or top-level module docstrings, regenerate and commit them:

.. code-block:: console

    uv run --group docs python docs/generate_api.py

Docstrings follow Google style with tensor shapes and SI units (types go in annotations, not docstrings).

Benchmarks
----------

Any PR claiming a speedup or modifying the simulation step should report before/after measurements on the same
hardware. Benchmarks live in ``asv/benchmarks`` and run via `asv <https://asv.readthedocs.io/>`__ or directly with
Python:

.. code-block:: console

    uvx --with virtualenv asv machine --yes
    uvx --with virtualenv asv run --launch-method spawn HEAD^!

    # Quick standalone run of a single benchmark file:
    uv run python asv/benchmarks/bench_simulation.py

On machines without a GPU, benchmarks fall back to small CPU sizes to verify correctness. Precision sweeps test
``"fp32"``, ``"tf32"``, and ``"fp16"``.

.. list-table::
   :widths: 46 54
   :header-rows: 1

   * - Benchmark
     - Measures
   * - ``bench_inference.BackboneStep``
     - Single eager FP32 step of each backbone across world counts
   * - ``bench_inference.NeRDStepperStep``
     - Compiled :class:`~newton_neural.models.NeRDStepper` step across world counts and precisions
   * - ``bench_simulation.SolverNeRDStep``
     - Full :class:`~newton_neural.solvers.SolverNeRD` step (with foot probes) across world counts and precisions
   * - ``bench_simulation.ContactInputProbesSensing``
     - Single ``ContactInputProbes`` sensor kernel update across world counts
   * - ``bench_simulation.ContactInputSetSensing``
     - Single ``ContactInputSet`` sensor update (excluding collision detection) across world counts
   * - ``bench_training.TrajectoryWindowsBatch``
     - On-device batch indexing from :class:`~newton_neural.data.TrajectoryWindows` across batch sizes
   * - ``bench_training.LossOneStepGradient``
     - :class:`~newton_neural.training.LossOneStep` forward + backward pass by backbone and batch size (eager and
       compiled on GPU)

Continuous integration
----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Workflow
     - Checks
   * - ``pr.yml``
     - Runs on PRs: ``uv lock --check``, ``ci.yml``, Sphinx HTML build (``-W``), up-to-date ``docs/api/``, and
       Sphinx doctests.
   * - ``ci.yml``
     - Runs on PRs and pushes to ``main``: Ruff lint/format, ``lint-imports``, and CPU unit tests on Python 3.11 and
       3.12.
   * - ``pr_license_check.yml``
     - Verifies Apache-2.0 headers on Python files and CC-BY-4.0 headers on ``.rst`` files.
   * - ``changelog-preview.yml``
     - Validates Towncrier rendering of pending ``changelog/`` fragments.
   * - ``docs-dev.yml``
     - Builds HTML docs on pushes to ``main`` and publishes to GitHub Pages (``/latest/``).
