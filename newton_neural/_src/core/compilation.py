# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Helpers for isolating ``torch.compile`` recompile caches across instances."""

from __future__ import annotations

import types
from collections.abc import Callable


def with_own_code(function: Callable) -> Callable:
    """Return a shallow clone of ``function`` backed by a distinct code object.

    ``torch.compile`` tracks its recompile limit per code object; once exceeded, it falls back to eager execution
    (or raises under ``fullgraph=True``). Giving each stepper or loss its own code object prevents independent
    models, world counts, or batch shapes from exhausting a shared cache. Call once per compiled instance rather
    than per step, since PyTorch retains compiled artifacts for the process lifetime.
    """
    copy = types.FunctionType(
        function.__code__.replace(),
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    return copy
