# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit test suite for ``newton_neural``, runnable via ``python -m newton_neural.tests``.

Uses ``unittest`` with per-device registration (CPU plus first visible CUDA GPU, skipping CUDA-only tests when no
GPU is present). Shared support modules: ``unittest_utils`` (device registration and temp dirs), ``fixtures``
(minimal procedural Newton robots), ``extensions`` (public-API custom family/backbone/contact input), and ``helpers``
(small test models, batches, and assertions).
"""
