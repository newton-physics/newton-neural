# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prediction targets (:class:`PredictionBase`, :class:`PredictionStateIncrement`, :class:`PredictionFreeMotion`).

Decouples what the network predicts (full one-frame state increments vs. pre-contact free accelerations) from model
families, recorders, and training losses.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from ..core.registry import Part
from ..core.signal import Frames, SignalRole, SignalSpec
from ..data.source import SignalSource
from ..systems.articulation import SystemArticulation
from ..systems.system import SystemBase

if TYPE_CHECKING:
    from ..data.simulation import SimulationBase


class PredictionBase(Part, abc.ABC):
    """Defines a model's output width, extra ``LABEL`` signals to record, and physical-unit supervision target."""

    registry: ClassVar[dict[str, type]] = {}

    integrates: ClassVar[bool]
    """``True`` if the model integrates predictions via ``system.integrate`` (supervision from consecutive frames,
    supports state noise); ``False`` if an external solver consumes predictions (supervision from per-frame recorded
    ``LABEL`` signals, ``following`` is ``None``)."""

    @abc.abstractmethod
    def signals(self, system: SystemBase) -> dict[str, SignalSpec]:
        """Return ``LABEL`` signal specs required during recording (empty if ``integrates=True``)."""

    @abc.abstractmethod
    def output_count(self, system: SystemBase) -> int:
        """Return network output dimension for ``system``."""

    @abc.abstractmethod
    def label(self, system: SystemBase, current: Frames, following: Frames | None) -> Tensor:
        """Compute physical-unit supervision targets ``[..., output_count]``.

        Args:
            system: System specification.
            current: Signals at frame ``t``.
            following: Signals at frame ``t + 1`` (``None`` when ``integrates=False``).
        """

    def source(self, simulation: SimulationBase) -> SignalSource:
        """Create a post-step :class:`SignalSource` capturing this prediction's ``LABEL`` signals."""
        raise NotImplementedError(f"{type(self).__name__} has no label signals to read from a simulation")

    def config(self) -> dict[str, Any]:
        """Return constructor keyword arguments as plain data (empty by default)."""
        return {}


class PredictionStateIncrement(PredictionBase):
    """Full one-frame state increment (contact effects included), integrated directly by the model."""

    integrates: ClassVar[bool] = True

    def signals(self, system: SystemBase) -> dict[str, SignalSpec]:
        """Return ``{}`` (labels are derived from consecutive ``STATE`` frames)."""
        return {}

    def output_count(self, system: SystemBase) -> int:
        """Return ``system.increment_count``."""
        return system.increment_count

    def label(self, system: SystemBase, current: Frames, following: Frames | None) -> Tensor:
        """Return ``system.increment(current, following)`` ``[..., increment_count]``."""
        return system.increment(current, following)


class PredictionFreeMotion(PredictionBase):
    """Pre-contact free joint acceleration ``joint_qdd_free`` [rad/s^2 or m/s^2] for fixed-base articulations.

    Supervises against MuJoCo Warp's ``qacc_smooth`` recorded on each frame; incompatible with state noise because
    recorded labels correspond strictly to unperturbed states.
    """

    integrates: ClassVar[bool] = False

    def signals(self, system: SystemBase) -> dict[str, SignalSpec]:
        """Return ``joint_qdd_free`` ``[joint_dof_count]`` [rad/s^2 or m/s^2]."""
        dof_count = _fixed_base_dof_count(system)
        return {"joint_qdd_free": SignalSpec("joint_qdd_free", (dof_count,), SignalRole.LABEL, unit="rad/s^2 or m/s^2")}

    def output_count(self, system: SystemBase) -> int:
        """Return ``system.joint_dof_count`` (fixed-base articulations only)."""
        return _fixed_base_dof_count(system)

    def label(self, system: SystemBase, current: Frames, following: Frames | None) -> Tensor:
        """Return recorded ``current["joint_qdd_free"]`` ``[..., joint_dof_count]`` [rad/s^2 or m/s^2]."""
        return current["joint_qdd_free"]

    def source(self, simulation: SimulationBase) -> SignalSource:
        """Create a post-step source reading ``qacc_smooth`` from a single-substep MuJoCo Warp simulation.

        Raises:
            ValueError: If ``simulation.system`` is not fixed-base, ``substep_count != 1``, or the solver is not
                MuJoCo Warp (or uses RK4).
        """
        signals = self.signals(simulation.system)
        if simulation.substep_count != 1:
            raise ValueError(
                f"the free acceleration belongs to one solver step, and a frame of the simulation holds "
                f"{simulation.substep_count}. Use one solver step per frame"
            )
        read = _mujoco_free_acceleration(simulation)
        return SignalSource(signals, lambda: {"joint_qdd_free": read()}, after_step=True)


def _fixed_base_dof_count(system: SystemBase) -> int:
    """Validate that ``system`` is a fixed-base :class:`SystemArticulation` and return its DOF count."""
    if not isinstance(system, SystemArticulation) or system.floating_base:
        raise ValueError("PredictionFreeMotion supports fixed base articulations only")
    return system.joint_dof_count


def _mujoco_free_acceleration(simulation: SimulationBase) -> Callable[[], Tensor]:
    """Return a zero-arg callable gathering ``mjw_data.qacc_smooth`` for the bound robot's DOFs."""
    import mujoco

    model, solver = simulation.model, simulation.solver
    if not isinstance(solver, newton.solvers.SolverMuJoCo):
        raise ValueError(
            f"the free acceleration is read from MuJoCo Warp, and the simulation steps with {type(solver).__name__}"
        )
    if solver.use_mujoco_cpu:
        raise ValueError("the free acceleration is read from MuJoCo Warp, and the solver runs MuJoCo on the CPU")
    if solver.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("RK4 evaluates the dynamics several times per step and keeps the last free acceleration only")

    # Map Newton DOF indices to MuJoCo Warp columns (revolute/prismatic accelerations match directly).
    newton_dof = solver.mjc_dof_to_newton_dof.numpy()  # [world_count, mujoco_dof_count], -1 if unmapped
    worlds, columns = np.nonzero(newton_dof >= 0)
    column = np.full(model.joint_dof_count, -1)
    column[newton_dof[worlds, columns]] = columns
    robot_columns = column[simulation.binding.joint_dof_ids]
    if (robot_columns < 0).any():
        raise ValueError("MuJoCo Warp does not simulate every joint of the robot, so it has no free acceleration")

    qacc_smooth = wp.to_torch(solver.mjw_data.qacc_smooth)  # [world_count, mujoco_dof_count]
    index = torch.as_tensor(robot_columns, dtype=torch.long, device=qacc_smooth.device)
    return lambda: qacc_smooth.gather(1, index)
