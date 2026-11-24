# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Abstract interface for batched Newton simulations driven by recorders and live audits."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import newton
import torch
from torch import Tensor

from ..core.signal import Frames

if TYPE_CHECKING:
    from ..systems.system import BindingBase, SystemBase


class SimulationBase(abc.ABC):
    """Batched Newton simulation of one system per world.

    Every frame advances :attr:`substep_count` solver substeps. Both :meth:`step` and :meth:`restart` maintain two
    invariants: ``state`` has up-to-date body transforms (FK evaluated), and ``contacts`` (when collision detection
    is enabled) reflects ``state``. All tensors reside on :attr:`device` with leading dimension ``world_count``.

    Attributes:
        model: Newton scene model.
        state: Current Newton state (may swap objects on ``step()``; re-read attribute each frame).
        control: Newton control buffer read by the solver.
        contacts: Current detected contacts, or ``None`` if collision detection is disabled/internal.
        solver: Newton solver stepping ``model``.
        binding: :class:`~newton_neural.systems.BindingBase` mapping system signals to ``model``.
        world_count: Number of parallel worlds.
        frame_dt: Frame duration [s].
        substep_count: Number of solver substeps per frame.
        device: PyTorch device of the simulation.
    """

    model: newton.Model
    state: newton.State
    control: newton.Control
    contacts: newton.Contacts | None
    solver: newton.solvers.SolverBase
    binding: BindingBase
    world_count: int
    frame_dt: float
    substep_count: int
    device: torch.device

    @property
    def system(self) -> SystemBase:
        """Simulated system (``binding.system``)."""
        return self.binding.system

    @abc.abstractmethod
    def step(self, command: Frames | None = None) -> None:
        """Advance all worlds by one frame (``frame_dt``).

        Args:
            command: Optional ``CONTROL`` signal tensors ``[world_count, *shape]`` written before stepping; leaves
                existing :attr:`control` unchanged if ``None``.
        """

    @abc.abstractmethod
    def restart(self, frames: Frames, world_mask: Tensor | None = None) -> None:
        """Reset masked worlds to ``frames``, clear solver history, and refresh FK and contacts.

        Args:
            frames: ``STATE`` signals ``[world_count, *shape]`` (unmasked rows ignored).
            world_mask: Optional boolean mask ``[world_count]`` (all worlds if ``None``).
        """

    def state_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Finite ``(low, high)`` bounds per ``STATE`` signal (defaults to :meth:`BindingBase.state_bounds`)."""
        return self.binding.state_bounds()

    def command_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        """``(low, high)`` sampling bounds per ``CONTROL`` signal (defaults to :meth:`BindingBase.command_bounds`)."""
        return self.binding.command_bounds()
