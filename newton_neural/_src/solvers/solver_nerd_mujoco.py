# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way coupled solver (:class:`SolverNeRDMuJoCo`) injecting neural free accelerations into MuJoCo Warp."""

from __future__ import annotations

import os
from typing import Any, ClassVar

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase, SolverMuJoCo
from torch import Tensor

from ..core.graph import GraphedWork
from ..core.precision import Precision
from ..core.stream import torch_on_warp_stream
from ..models.nerd import NeRDBase
from ..models.stepper import NeRDStepper
from ..predictions.prediction import PredictionFreeMotion
from ..systems.articulation_binding import BindingArticulation, Pattern
from .common import load_nerd, require_frame_step, warn_unseen_changes
from .mujoco_staged_step import replace_mujoco_warp_step

_UNSEEN_CHANGE = (
    "The model predicts the free motion of the robot with the {} it was trained on. A change to those of the "
    "robot reaches only what MuJoCo Warp computes, which is contacts, joint limits and the other bodies."
)
"""Warning template for Newton model mutations reaching MuJoCo Warp but not the neural free-motion weights."""


class SolverNeRDMuJoCo(SolverMuJoCo):
    """MuJoCo Warp solver with learned free-space joint accelerations for one fixed-base robot per world.

    Overrides the robot's pre-constraint accelerations (``qacc_smooth``) inside MuJoCo Warp's staged step while
    MuJoCo Warp solves contacts, joint limits, and friction for robot and dynamic objects jointly. Requires a
    :class:`~newton_neural.predictions.PredictionFreeMotion` model with ``contact_input=None``, MuJoCo Warp GPU
    backend (``use_mujoco_cpu=False``), integrator in ``{"euler", "implicit", "implicitfast"}``,
    ``enable_sleeping=False``, and ``use_speculative_contacts=False``.

    On CUDA, caches CUDA graphs for the neural prediction stage per ``(state_in, control)`` object pair (warm up
    once before any outer ``wp.ScopedCapture``).

    Args:
        model: Newton scene model containing the robot and any dynamic objects.
        nerd: Free-motion :class:`NeRDBase` instance or checkpoint path.
        articulation: Label pattern of the robot's articulation.
        precision: Override :class:`Precision` (defaults to ``nerd.precision``).
        use_cuda_graph: Replay the neural prediction stage from CUDA graphs on CUDA devices (default ``True``).
        compile: Compile the neural step with ``torch.compile`` (defaults to ``True`` on CUDA).
        **mujoco_options: Additional keyword arguments forwarded to :class:`newton.solvers.SolverMuJoCo`.
    """

    UNSEEN_MODEL_CHANGES: ClassVar[dict[newton.ModelFlags, str]] = {
        newton.ModelFlags.BODY_INERTIAL_PROPERTIES: _UNSEEN_CHANGE.format("masses, centres of mass and inertias"),
        newton.ModelFlags.JOINT_DOF_PROPERTIES: _UNSEEN_CHANGE.format("joint gains, friction and armature"),
        newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES: _UNSEEN_CHANGE.format("joint gains, friction and damping"),
        newton.ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES: _UNSEEN_CHANGE.format("joint armature"),
        newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES: _UNSEEN_CHANGE.format("joint rest poses"),
        newton.ModelFlags.ACTUATOR_PROPERTIES: _UNSEEN_CHANGE.format("actuator gains and limits"),
        newton.ModelFlags.MODEL_PROPERTIES: _UNSEEN_CHANGE.format("gravity"),
    }
    """Newton model flags affecting MuJoCo's solve but not the learned free-motion model.

    :meta hide-value:
    """

    def __init__(
        self,
        model: newton.Model,
        nerd: NeRDBase | str | os.PathLike[str],
        *,
        articulation: Pattern = "*",
        precision: Precision | str | None = None,
        use_cuda_graph: bool = True,
        compile: bool | None = None,
        **mujoco_options: Any,
    ) -> None:
        nerd = load_nerd(nerd, wp.device_to_torch(model.device))
        if not isinstance(nerd.prediction, PredictionFreeMotion):
            raise ValueError(
                f"SolverNeRDMuJoCo runs models that predict free motion, and this one uses "
                f"{type(nerd.prediction).__name__}. Step it with SolverNeRD"
            )
        if nerd.contact_input is not None:
            raise ValueError(
                f"MuJoCo Warp resolves every contact of a free motion model, so the model takes no contact input, "
                f"and this one has {type(nerd.contact_input).__name__}"
            )
        # Suppress unseen-change warnings during SolverMuJoCo.__init__'s initial notify_model_changed call.
        self._warned_changes: int | None = None
        super().__init__(model, **mujoco_options)
        self.nerd = nerd
        self.binding = nerd.system.bind(model, articulation)
        self.stepper = NeRDStepper(nerd, self.binding.world_count, precision=precision, compile=compile)
        self._free_acceleration, self._dofs = _free_acceleration_input(self, self.binding)
        # Fallback control handle aliasing model's default control arrays when caller passes control=None.
        self._model_control = model.control(clone_variables=False)
        self.use_cuda_graph = use_cuda_graph and model.device.is_cuda
        self._prediction = GraphedWork(self._predict, model.device, use_cuda_graph=use_cuda_graph)
        self._warned_changes = 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.stepper!r}, use_cuda_graph={self.use_cuda_graph})"

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Predict robot free accelerations and advance the full scene by one frame (``dt == nerd.frame_dt``)."""
        require_frame_step(self.nerd, dt)
        if control is None:
            control = self._model_control
        self._prediction.run(state_in, control)
        super().step(state_in, state_out, control, contacts, dt, observables=observables)

    def reset(
        self,
        state: newton.State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: newton.StateFlags | int | None = None,
    ) -> None:
        """Reset MuJoCo state to Newton model defaults and clear neural streaming history for masked worlds.

        As with :meth:`SolverNeRD.reset`, ``flags`` selects what is restored and ``flags=0`` keeps ``state`` as
        authored. To start from a custom state, call this method first and then write the joint coordinates, or
        use :meth:`SimulationNewton.restart <newton_neural.data.SimulationNewton.restart>`.
        """
        with torch_on_warp_stream(self.stepper.device):
            super().reset(state, world_mask, flags)
            self.stepper.reset(None if world_mask is None else wp.to_torch(world_mask)[: self.binding.world_count])

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        """Forward updates to MuJoCo Warp and warn once per :attr:`UNSEEN_MODEL_CHANGES` flag."""
        super().notify_model_changed(flags)
        if self._warned_changes is not None:
            self._warned_changes = warn_unseen_changes(self.UNSEEN_MODEL_CHANGES, int(flags), self._warned_changes)

    @torch.inference_mode()  # Keep torch.compile specialization consistent across all calls
    def _predict(self, state: newton.State, control: newton.Control) -> None:
        """Write neural free-acceleration predictions into the shared Warp buffer consumed by MuJoCo Warp."""
        prediction = self.stepper.predict(self.binding.read_state(state) | self.binding.read_control(control))
        self._free_acceleration[self._dofs] = prediction


def _free_acceleration_input(solver: SolverMuJoCo, binding: BindingArticulation) -> tuple[Tensor, Tensor]:
    """Attach staged free-acceleration override to ``solver`` and return ``(free_acc_tensor, robot_dof_indices)``."""
    model = solver.model
    dofs = binding.joint_dof_ids
    mask = np.zeros(model.joint_dof_count, dtype=bool)
    mask[dofs] = True
    free_acceleration = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    dof_mask = wp.array(mask, dtype=wp.bool, device=model.device)
    replace_mujoco_warp_step(solver, free_acceleration, dof_mask)
    return wp.to_torch(free_acceleration), torch.as_tensor(dofs, device=wp.device_to_torch(model.device))
