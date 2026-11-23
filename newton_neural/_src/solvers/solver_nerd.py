# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone Newton solver (:class:`SolverNeRD`) stepping an articulated robot with an integrating neural model."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, cast

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase

from ..contacts.kinematics import contact_gap
from ..core.graph import GraphedWork
from ..core.precision import Precision
from ..models.nerd import NeRDBase
from ..models.stepper import NeRDStepper
from ..systems.articulation_binding import Pattern, reset_mask_entries
from .common import load_nerd, require_frame_step, warn_unseen_changes

_UNSEEN_CHANGE = (
    "The model does not see changes of {} in the Newton model. It moves the robot as in the data it was "
    "trained on, whatever changes afterwards."
)
"""Warning template for Newton model mutations not observed by trained network weights."""


class SolverNeRD(SolverBase):
    """Drop-in :class:`newton.solvers.SolverBase` stepping one robot per world with an integrating neural model.

    Each step reads robot joint state and controls, updates the optional contact sensor, runs the compiled
    :class:`NeRDStepper`, writes the next joint state (preserving non-robot bodies unchanged), and refreshes body
    transforms with ``newton.eval_fk``. Requires ``dt == nerd.frame_dt`` and caller-supplied ``contacts`` whenever
    required by the contact input or ``CONTACT_F``.

    On CUDA, caches up to 16 CUDA graphs keyed by ``(state_in, state_out, control, contacts, observables)`` object
    identities; reuse buffer objects across steps. Supports external ``wp.ScopedCapture`` (``THREAD_LOCAL`` and
    ``RELAXED``) after one warm-up step outside capture.

    Args:
        model: Newton scene model (one matching robot per world).
        nerd: Integrating :class:`NeRDBase` instance or checkpoint path.
        articulation: Articulation label pattern (default ``"*"``).
        precision: Override :class:`Precision` (defaults to ``nerd.precision``).
        use_cuda_graph: Replay steps from CUDA graphs on CUDA devices (default ``True``).
        compile: Compile the model step via ``torch.compile`` (defaults to ``True`` on CUDA).

    Attributes:
        nerd: Loaded model on ``model.device``.
        binding: :class:`~newton_neural.systems.BindingArticulation` locating the robot in ``model``.
        stepper: :class:`NeRDStepper` owning per-world streaming history.
        use_cuda_graph: Active CUDA graph replay flag (``False`` on CPU).
    """

    class ObservableKind(SolverBase.ObservableKind):
        """Standard Newton observable kinds plus ``CONTACT_ROW_OVERFLOW``."""

        CONTACT_ROW_OVERFLOW = "contact_row_overflow"
        """Per-world count of contact rows dropped when exceeding a bounded contact sensor's capacity."""

    @dataclass(eq=False)
    class Observables(SolverBase.Observables):
        """Observable buffers for :class:`SolverNeRD`.

        ``contact_f`` stores penalty-based **estimated** normal wrenches at ``state_in``: for ``gap < 0``,
        ``F = 0.5 * (ke0 + ke1) * gap * normal`` [N] (no friction) plus torque about body 0's CoM [N*m] in world
        frame. The magnitude follows the contact stiffness ``ke`` of the two shapes, which has no part in the
        dynamics the model learned, so it is not calibrated against the load. Read the estimate as an indicator of
        contact and for its direction. It keeps contact sensors and RL contact rewards working and is not meant
        for force control.
        """

        contact_row_overflow: wp.array[wp.int32] | None = SolverBase.Observables.field(  # noqa: RUF009
            dtype=wp.int32, frequency=newton.Model.AttributeFrequency.WORLD
        )
        """Dropped contact rows per world, shape ``(world_count + 1,)`` (last entry is Newton's global world)."""

    SUPPORTED_OBSERVABLES = frozenset({SolverBase.ObservableKind.CONTACT_F, ObservableKind.CONTACT_ROW_OVERFLOW})

    supports_body_pair_reduced_contacts = False
    """Contact rows and estimated contact forces read the gap between the two surfaces. A collision pipeline with
    body pair reduction stores another separation for the contacts it keeps, so a step that reads the contacts
    refuses reduced ones."""

    UNSEEN_MODEL_CHANGES: ClassVar[dict[newton.ModelFlags, str]] = {
        newton.ModelFlags.BODY_INERTIAL_PROPERTIES: _UNSEEN_CHANGE.format("masses, centres of mass and inertias"),
        newton.ModelFlags.JOINT_DOF_PROPERTIES: _UNSEEN_CHANGE.format("joint gains, limits, friction and armature"),
        newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES: _UNSEEN_CHANGE.format(
            "joint gains, limits, friction and damping"
        ),
        newton.ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES: _UNSEEN_CHANGE.format("joint armature"),
        newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES: _UNSEEN_CHANGE.format("joint rest poses"),
        newton.ModelFlags.SHAPE_PROPERTIES: _UNSEEN_CHANGE.format("shape sizes and materials"),
        newton.ModelFlags.MODEL_PROPERTIES: _UNSEEN_CHANGE.format("gravity"),
        newton.ModelFlags.CONSTRAINT_PROPERTIES: _UNSEEN_CHANGE.format("equality constraints and mimic joints"),
        newton.ModelFlags.TENDON_PROPERTIES: _UNSEEN_CHANGE.format("tendons"),
        newton.ModelFlags.ACTUATOR_PROPERTIES: _UNSEEN_CHANGE.format("actuator gains and limits"),
    }
    """Newton model flags ignored by trained weights, mapped to their warning messages.

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
    ) -> None:
        super().__init__(model)
        nerd = load_nerd(nerd, wp.device_to_torch(model.device))
        if not nerd.prediction.integrates:
            raise ValueError(
                f"{type(nerd.prediction).__name__} does not give the next state, which this solver writes. "
                "Step such a model with SolverNeRDMuJoCo"
            )
        self.nerd = nerd
        self.binding = nerd.system.bind(model, articulation)
        self.stepper = NeRDStepper(nerd, self.binding.world_count, precision=precision, compile=compile)
        self._sensor = None if nerd.contact_input is None else nerd.contact_input.sensor(model, self.binding)
        # Fallback control handle aliasing model's default control arrays when caller passes control=None.
        self._model_control = model.control(clone_variables=False)
        self.use_cuda_graph = use_cuda_graph and model.device.is_cuda
        self._work = GraphedWork(self._advance, model.device, use_cuda_graph=use_cuda_graph)
        self._warned_changes = 0
        # Entry of a reset mask that governs every joint coordinate, DOF and articulation.
        device = self.stepper.device
        joint_entry = reset_mask_entries(model, model.joint_world)
        coord_entry = np.repeat(joint_entry, np.diff(model.joint_q_start.numpy()))
        dof_entry = np.repeat(joint_entry, np.diff(model.joint_qd_start.numpy()))
        self._coord_entry = torch.as_tensor(coord_entry, device=device)
        self._dof_entry = torch.as_tensor(dof_entry, device=device)
        self._articulation_entry = torch.as_tensor(reset_mask_entries(model, model.articulation_world), device=device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.stepper!r}, use_cuda_graph={self.use_cuda_graph})"

    @property
    def supported_observables(self) -> frozenset[str]:
        """Supported observable kinds (``CONTACT_F``, plus ``CONTACT_ROW_OVERFLOW`` if sensor tracks overflow)."""
        if self._sensor is not None and self._sensor.overflow is not None:
            return self.SUPPORTED_OBSERVABLES
        return frozenset({self.ObservableKind.CONTACT_F})

    def observables(
        self,
        kinds: Iterable[str],
        *,
        contacts: newton.Contacts | None = None,
        requires_grad: bool | None = None,
    ) -> SolverNeRD.Observables:
        """Allocate requested buffers with a global-world entry in the contact overflow counts."""
        observables = cast(
            SolverNeRD.Observables, super().observables(kinds, contacts=contacts, requires_grad=requires_grad)
        )
        overflow = observables.contact_row_overflow
        # Newton 1.8's WORLD-frequency allocator counts only local worlds.
        if overflow is not None and overflow.shape[0] == self.model.world_count:
            observables.contact_row_overflow = wp.zeros(
                self.model.world_count + 1,
                dtype=overflow.dtype,
                device=overflow.device,
                requires_grad=overflow.requires_grad,
            )
        return observables

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
        """Advance all worlds by one model frame (``dt == nerd.frame_dt``).

        Args:
            state_in: Input state (body transforms must be up to date).
            state_out: Output state (may alias ``state_in``).
            control: Controls (defaults to ``model``'s control arrays if ``None``).
            contacts: Contacts for ``state_in``, required and unreduced if the contact input reads contacts or
                ``CONTACT_F`` is requested. A step that reads no contacts takes any.
            dt: Step duration [s] (must match ``nerd.frame_dt``).
            observables: Optional preallocated :class:`Observables` container.
        """
        require_frame_step(self.nerd, dt)
        contact_input = self.nerd.contact_input
        reads_contacts = contact_input is not None and contact_input.reads_contacts
        if contacts is None and reads_contacts:
            raise ValueError(
                f"{type(contact_input).__name__} reads the contacts of the collision detector, and the step was "
                "given none. Run collision detection on state_in, with newton.CollisionPipeline(model), and pass "
                "its contacts"
            )
        self.validate_observables(observables, contacts)
        # Contact rows and force estimates both read the gap between the surfaces, which reduced contacts lack.
        if reads_contacts or (observables is not None and self.ObservableKind.CONTACT_F in observables.kinds):
            self._require_unreduced_contacts(contacts)
        self._work.run(state_in, state_out, self._model_control if control is None else control, contacts, observables)

    @torch.inference_mode()  # Keep torch.compile specialization consistent across all calls
    def _advance(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts | None,
        observables: SolverNeRD.Observables | None,
    ) -> None:
        """Graph-safe single step (zero host syncs after warm-up)."""
        binding, sensor, model = self.binding, self._sensor, self.model
        requested = frozenset() if observables is None else observables.kinds

        frames = {**binding.read_state(state_in), **binding.read_control(control)}
        if sensor is not None:
            frames.update(sensor.update(state_in, contacts))
        if self.ObservableKind.CONTACT_F in requested:
            # Estimate forces from state_in before in-place FK overwrites body_q.
            self._estimate_contact_forces(state_in, contacts, observables.contact_f)

        following = self.stepper.step(frames)
        if state_out is not state_in:
            # Preserve non-robot joints when stepping between distinct state buffers.
            wp.copy(state_out.joint_q, state_in.joint_q)
            wp.copy(state_out.joint_qd, state_in.joint_qd)
        binding.write_state(state_out, following)
        newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

        if self.ObservableKind.CONTACT_ROW_OVERFLOW in requested:
            wp.copy(observables.contact_row_overflow, sensor.overflow)

    def _estimate_contact_forces(
        self, state: newton.State, contacts: newton.Contacts, contact_f: wp.array[wp.spatial_vector]
    ) -> None:
        """Launch penalty contact-wrench estimation kernel for ``contacts`` at ``state``."""
        model = self.model
        wp.launch(
            _contact_force_estimate_kernel,
            dim=contact_f.shape[0],
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state.body_q,
                model.body_com,
                model.shape_body,
                model.shape_material_ke,
            ],
            outputs=[contact_f],
            device=model.device,
        )

    @torch.no_grad()
    def reset(
        self,
        state: newton.State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: newton.StateFlags | int | None = None,
    ) -> None:
        """Clear the model's history and restore joint state to Newton model defaults for masked worlds.

        Follows :meth:`newton.solvers.SolverMuJoCo.reset`. The history of the masked worlds is always cleared.
        ``flags`` selects what is restored from ``model.joint_q`` and ``model.joint_qd`` (``JOINT_Q``, ``JOINT_QD``,
        or both if ``None``) for every joint of the masked worlds, whose body transforms are then recomputed.
        ``flags=0`` restores nothing and leaves ``state`` as authored.

        To start from a custom state, call this method first, write the joint coordinates, and update body
        transforms with :func:`newton.eval_fk`, or use
        :meth:`SimulationNewton.restart <newton_neural.data.SimulationNewton.restart>`.

        Args:
            state: State to reset (modified in place).
            world_mask: Optional mask ``(world_count + 1,)`` of worlds to reset, whose last entry selects Newton's
                global world (all worlds if ``None``). A model built without worlds is world ``0``.
            flags: Optional :class:`newton.StateFlags` bitmask of the joint state to restore (all if ``None``).
        """
        super().reset(state, world_mask, flags)
        model = self.model
        if world_mask is None:
            worlds = torch.ones(model.world_count + 1, dtype=torch.bool, device=self.stepper.device)
        else:
            worlds = wp.to_torch(world_mask)
        self.stepper.reset(worlds[: self.binding.world_count])

        flags = int(newton.StateFlags.ALL if flags is None else flags)
        restored = False
        for flag, name, entry in (
            (newton.StateFlags.JOINT_Q, "joint_q", self._coord_entry),
            (newton.StateFlags.JOINT_QD, "joint_qd", self._dof_entry),
        ):
            if flags & flag:
                value = wp.to_torch(getattr(state, name), requires_grad=False)
                default = wp.to_torch(getattr(model, name), requires_grad=False)
                value.copy_(torch.where(worlds[entry], default, value))
                restored = True
        if restored:
            articulations = wp.from_torch(worlds[self._articulation_entry], dtype=wp.bool)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state, mask=articulations)

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        """Emit one-time :class:`NewtonNeuralWarning` for any changed :attr:`UNSEEN_MODEL_CHANGES` flags."""
        self._warned_changes = warn_unseen_changes(self.UNSEEN_MODEL_CHANGES, int(flags), self._warned_changes)


@wp.kernel(enable_backward=False)
def _contact_force_estimate_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_material_ke: wp.array[float],
    contact_f: wp.array[wp.spatial_vector],
):
    """Compute penalty normal wrench on body 0 about its CoM for each rigid contact slot."""
    contact = wp.tid()
    wrench = wp.spatial_vector()
    # Guard against detector count exceeding buffer capacity; trailing slots remain zero.
    if contact < wp.min(contact_count[0], contact_shape0.shape[0]):
        shape0 = contact_shape0[contact]
        shape1 = contact_shape1[contact]
        if shape0 >= 0 and shape1 >= 0:
            # Static world shapes (body < 0) use identity transform and world origin for torque.
            X0 = wp.transform_identity()
            X1 = wp.transform_identity()
            com0 = wp.vec3()
            body0 = shape_body[shape0]
            body1 = shape_body[shape1]
            if body0 >= 0:
                X0 = body_q[body0]
                com0 = wp.transform_point(X0, body_com[body0])
            if body1 >= 0:
                X1 = body_q[body1]
            normal = contact_normal[contact]
            support0 = wp.transform_point(X0, contact_point0[contact])
            support1 = wp.transform_point(X1, contact_point1[contact])
            gap = contact_gap(normal, support0, support1, contact_margin0[contact], contact_margin1[contact])
            if gap < 0.0:
                # Normal points from shape 0 into shape 1, so gap < 0 pushes body 0 along -normal.
                stiffness = 0.5 * (shape_material_ke[shape0] + shape_material_ke[shape1])
                force = stiffness * gap * normal
                surface0 = support0 + contact_margin0[contact] * normal
                wrench = wp.spatial_vector(force, wp.cross(surface0 - com0, force))
    contact_f[contact] = wrench
