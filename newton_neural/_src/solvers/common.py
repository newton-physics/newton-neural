# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared model loading, frame-duration validation, and unseen-model-change warnings for neural solvers."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping

import newton
import torch

from ..core.duration import same_duration
from ..core.warnings import NewtonNeuralWarning
from ..models.nerd import NeRDBase


def load_nerd(nerd: NeRDBase | str | os.PathLike[str], device: torch.device | str) -> NeRDBase:
    """Load or move ``nerd`` onto ``device`` and verify that ``nerd.frame_dt`` is set.

    A model read from a file is put into evaluation mode, since a solver runs inference and nobody else holds
    the model. A model instance keeps the mode its owner set.

    Args:
        nerd: In-memory :class:`NeRDBase` instance or checkpoint file path.
        device: Target PyTorch device.

    Raises:
        ValueError: If ``nerd.frame_dt is None``.
    """
    nerd = nerd.to(device) if isinstance(nerd, NeRDBase) else NeRDBase.load(nerd, device=device).eval()
    if nerd.frame_dt is None:
        raise ValueError(
            "the model has no frame duration, so a solver cannot tell whether it is stepped by the frame the model "
            "was trained on. Fit its normalization, which takes the duration from the data, or build the model "
            "with frame_dt"
        )
    return nerd


def require_frame_step(nerd: NeRDBase, dt: float) -> None:
    """Raise ``ValueError`` if ``dt`` [s] does not match ``nerd.frame_dt`` within relative tolerance."""
    if not same_duration(dt, nerd.frame_dt):
        raise ValueError(
            f"the model was trained on frames of {nerd.frame_dt} s, and the solver is stepped by {dt} s. "
            "Take one solver step per frame of that duration"
        )


def warn_unseen_changes(unseen: Mapping[newton.ModelFlags, str], flags: int, warned: int) -> int:
    """Emit a one-time :class:`NewtonNeuralWarning` for each set flag in ``unseen`` not yet in ``warned``.

    Returns:
        Updated bitmask of warned flags.
    """
    for flag, message in unseen.items():
        if flags & flag and not warned & flag:
            # Point warning stack frame at the caller of solver.notify_model_changed().
            warnings.warn(message, NewtonNeuralWarning, stacklevel=3)
            warned |= flag
    return warned
