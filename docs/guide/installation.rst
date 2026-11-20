.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Installation
============

``newton-neural`` is managed with `uv <https://docs.astral.sh/uv/>`__, matching Newton's tooling.

Requirements
------------

All Python dependencies are declared in ``pyproject.toml`` and installed automatically by ``uv``.

.. list-table::
   :widths: 22 28 50
   :header-rows: 1

   * - Requirement
     - Version
     - Notes
   * - Python
     - 3.11+
     - Development is pinned to 3.12 (``.python-version``).
   * - OS
     - Linux (``x86_64`` or ``aarch64``)
     -
   * - GPU
     - NVIDIA GPU supported by Warp and PyTorch
     - Optional. CPU execution is supported for tests, doctests, and small models.
   * - Newton
     - ``>=1.9.0.dev0`` with ``sim`` extra
     - The ``sim`` extra provides MuJoCo and MuJoCo Warp, required for recording from
       :class:`newton.solvers.SolverMuJoCo` and for :class:`~newton_neural.solvers.SolverNeRDMuJoCo`.
       See :doc:`recording` and :doc:`coupling`.
   * - Warp
     - ``>=1.17``
     -
   * - PyTorch
     - ``>=2.11``
     -
   * - NumPy
     - ``>=2``
     -

Install from source
-------------------

Clone the repository and sync the locked environment:

.. code-block:: console

    git clone https://github.com/newton-physics/newton-neural.git
    cd newton-neural
    uv sync

Verify the installation by checking the version and running the unit tests (which run on CPU, and also on GPU if
available):

.. code-block:: console

    uv run python -c "import newton_neural; print(newton_neural.__version__)"
    uv run -m newton_neural.tests

On Linux ``aarch64``, ``pyproject.toml`` resolves PyTorch wheels from the CUDA 13.0 index at
``https://download.pytorch.org/whl/cu130``. On ``x86_64``, PyTorch is installed from PyPI.

Add to an existing uv project
-----------------------------

Install directly from Git:

.. code-block:: console

    uv add "newton-neural @ git+https://github.com/newton-physics/newton-neural.git"

.. note::

    ``uv`` does not propagate ``[tool.uv.sources]`` or custom indexes from dependencies. On ``aarch64``, configure
    the PyTorch CUDA index in your own ``pyproject.toml`` as this repository does.

Optional viewer dependencies
----------------------------

Interactive visualization in the examples requires ``pyglet`` and ``imgui-bundle``, provided by the ``examples``
extra:

.. code-block:: console

    uv run --extra examples -m newton_neural.examples quickstart

Both the viewer packages and MuJoCo Warp are imported lazily when first used, not at ``import newton_neural``.
Running examples headless with ``--viewer null`` requires no extra dependencies. See :doc:`examples`.

Tested versions
---------------

The test suite is verified against the versions locked in ``uv.lock`` on Linux (CPU and NVIDIA GPUs):

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Package
     - Locked version
   * - Python
     - 3.12
   * - Newton
     - 1.9.0.dev0 (pinned git revision in ``uv.lock``)
   * - Warp
     - 1.19.0
   * - MuJoCo / MuJoCo Warp
     - 3.17.0
   * - PyTorch
     - 2.14.1+cu130 (``aarch64``) / 2.15.0 (``x86_64``)
