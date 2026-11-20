# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private implementation modules for ``newton_neural``.

Everything under this subpackage is internal and subject to change without notice; public APIs are re-exported from
the top-level ``newton_neural`` modules.

Subpackages are organized into strict dependency layers enforced by ``lint-imports`` (declared in
``pyproject.toml``), from bottom to top: ``core``, ``systems``, ``backbones``/``data``, ``contacts``/``predictions``,
``models``, ``solvers``/``training``, ``evaluation``, and ``testing``.
"""
