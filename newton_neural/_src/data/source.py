# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Signal source abstraction decoupling trajectory recording from where signals originate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.signal import Frames, SignalSpec


@dataclass(frozen=True)
class SignalSource:
    """Declares a group of signals and a reader callback invoked once per recorded frame.

    :class:`~newton_neural.data.TrajectoryRecorder` validates each source once at construction by invoking
    :attr:`read` and checking keys, shapes, and dtypes against :attr:`signals`.
    """

    signals: dict[str, SignalSpec]
    """Specs of all signals returned by :attr:`read`, keyed by name."""

    read: Callable[[], Frames]
    """Zero-arg reader returning freshly allocated ``[world_count, *shape]`` tensors matching :attr:`signals`
    (must clone rather than return views into simulation buffers that later steps overwrite)."""

    after_step: bool = False
    """If ``True``, read after ``simulation.step()`` (for consumed controls and post-step labels); if ``False``,
    read before ``simulation.step()`` (for pre-step states and contacts)."""
