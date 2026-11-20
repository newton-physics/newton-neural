# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standard Newton simulation wrapper (:class:`SimulationNewton`) with CUDA-graphed frame stepping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from ..core.graph import GraphedWork
from ..core.rotation import normalize
from ..core.signal import Frames
from ..systems.articulation_binding import reset_mask_entries
from .simulation import SimulationBase

if TYPE_CHECKING:
    from ..systems.system import SystemBase


class SimulationNewton(SimulationBase):
    """Wraps a :class:`newton.Model` and solver for trajectory recording and live audits.

    Advances ``substep_count`` substeps of ``frame_dt / substep_count`` per frame using ping-pong states, clearing
    external forces before each substep and re-running ``collision_pipeline`` after each substep and reset. On CUDA,
    subsequent frames replay from a CUDA graph via :class:`~newton_neural._src.core.graph.GraphedWork`.

    Requires a generalized-coordinate solver that updates ``state.joint_q`` and ``state.joint_qd`` (such as
    :class:`newton.solvers.SolverMuJoCo`, :class:`newton.solvers.SolverFeatherstone`, or
    :class:`~newton_neural.solvers.SolverNeRD`).

    Args:
        model: Newton scene model.
        solver: Solver instance constructed for ``model``.
        system: System matching one articulation per world in ``model``.
        articulation: Articulation label pattern passed to ``system.bind``.
        frame_dt: Frame duration [s].
        substep_count: Number of solver substeps per frame (default ``1``).
        collision_pipeline: Optional external collision pipeline (``None`` when solver handles contacts internally).
        use_cuda_graph: Replay frames from CUDA graphs on CUDA devices (default ``True``).

    Raises:
        KeyError: If no articulation matches ``articulation``.
        ValueError: If ``solver.model is not model``, ``substep_count < 1``, or ``model`` disagrees with ``system``.
    """

    def __init__(
        self,
        model: newton.Model,
        solver: newton.solvers.SolverBase,
        system: SystemBase,
        *,
        articulation: Any = "*",
        frame_dt: float,
        substep_count: int = 1,
        collision_pipeline: newton.CollisionPipeline | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        if solver.model is not model:
            raise ValueError("the solver steps another model than the one given")
        if substep_count < 1:
            raise ValueError(f"a frame holds at least one solver step, got substep_count {substep_count}")
        self.model, self.solver = model, solver
        self.binding = system.bind(model, articulation)
        self.world_count = self.binding.world_count
        self.frame_dt, self.substep_count = frame_dt, substep_count
        self.device = torch.device(wp.device_to_torch(model.device))
        self.collision_pipeline = collision_pipeline
        self.contacts = None if collision_pipeline is None else collision_pipeline.contacts()
        self.control = model.control()
        self.state = model.state()
        self._next_state = model.state()
        self._frame = GraphedWork(self._advance, model.device, use_cuda_graph=use_cuda_graph)

        # Entry of a restart mask that governs every articulation and every free joint.
        self._articulation_world = torch.as_tensor(
            reset_mask_entries(model, model.articulation_world), device=self.device
        )
        # Index free-joint quaternion slots so sampled raw coordinates are normalized to unit length on restart.
        free = np.flatnonzero(model.joint_type.numpy() == newton.JointType.FREE)
        quaternion_coords = model.joint_q_start.numpy()[free, None] + np.arange(3, 7)
        self._quaternion_coords = torch.as_tensor(quaternion_coords, dtype=torch.long, device=self.device)
        self._quaternion_world = torch.as_tensor(reset_mask_entries(model, model.joint_world)[free], device=self.device)

        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        self._collide(self.state)

    def step(self, command: Frames | None = None) -> None:
        """Write optional ``command`` into :attr:`control` and advance all worlds by one frame."""
        if command is not None:
            self.binding.write_control(self.control, command)
        self._frame.run(self.state, self._next_state)
        if self.substep_count % 2:  # Odd substep count leaves final output in _next_state
            self.state, self._next_state = self._next_state, self.state

    def _advance(self, state: newton.State, next_state: newton.State) -> None:
        """Execute ``substep_count`` ping-pong solver substeps plus collision detection."""
        dt = self.frame_dt / self.substep_count
        for _ in range(self.substep_count):
            state.clear_forces()
            self.solver.step(state, next_state, self.control, self.contacts, dt)
            state, next_state = next_state, state
            self._collide(state)

    def restart(self, frames: Frames, world_mask: Tensor | None = None) -> None:
        """Reset masked worlds to ``frames``, normalize free-joint quaternions, and update FK and contacts.

        Calls ``solver.reset`` first (clearing history and, for solvers that restore joint state, resetting
        non-robot bodies to model defaults), then writes ``frames``, normalizes free-joint quaternions, evaluates
        FK, and runs collision detection.
        """
        # Newton's world_mask has length world_count + 1 (last entry is the global world).
        worlds = torch.zeros(self.world_count + 1, dtype=torch.bool, device=self.device)
        worlds[:-1] = True if world_mask is None else world_mask
        self.solver.reset(self.state, wp.from_torch(worlds, dtype=wp.bool))
        self.binding.write_state(self.state, frames, world_mask)
        joint_q = wp.to_torch(self.state.joint_q)
        quaternions = joint_q[self._quaternion_coords]
        restarted = worlds[self._quaternion_world, None]
        joint_q[self._quaternion_coords] = torch.where(restarted, normalize(quaternions), quaternions)
        articulations = wp.from_torch(worlds[self._articulation_world], dtype=wp.bool)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state, mask=articulations)
        self._collide(self.state)

    def _collide(self, state: newton.State) -> None:
        if self.collision_pipeline is not None:
            self.collision_pipeline.collide(state, self.contacts)
