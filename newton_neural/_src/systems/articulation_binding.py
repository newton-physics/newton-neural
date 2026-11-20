# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton model binding for :class:`SystemArticulation`.

Validates that every world contains the specified articulation, exposes zero-copy strided PyTorch views of Newton's
Warp state/control arrays for replicated scenes (falling back to indexed gather/scatter for non-uniform layouts),
and derives default state/command sampling bounds from Newton joint limits.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from newton.selection import ArticulationView
from torch import Tensor

from ..core.signal import Frames
from .system import BindingBase

if TYPE_CHECKING:
    from .articulation import SystemArticulation

_FRAME_TOLERANCE = 1.0e-4
"""Max tolerance [m] when comparing joint frames and axes against ``SystemArticulation.Spec``."""

_UNLIMITED = 1.0e6
"""Threshold above which Newton joint limits are treated as unbounded (``newton.MAXVAL`` or default ``1e6``)."""

_ANGULAR_REACH = math.pi
"""Default sampling half-range [rad] for unbounded revolute joints."""

_LINEAR_REACH = 1.0
"""Default sampling half-range [m] for unbounded prismatic joints."""

_SPEED_LIMIT = 20.0
"""Cap on default joint velocity sampling bounds [m/s or rad/s]."""

_BASE_REACH = 0.25
"""Default per-axis floating-base position offset around model pose [m]."""

_BASE_LINEAR_SPEED = 1.0
"""Default per-axis floating-base linear velocity bound [m/s]."""

_BASE_ANGULAR_SPEED = 2.0
"""Default per-axis floating-base angular velocity bound [rad/s]."""

Pattern = str | list[str] | re.Pattern[str] | list[int]
"""Articulation label selector accepted by :class:`newton.selection.ArticulationView`."""


def leaf(label: str) -> str:
    """Return the final ``/``-delimited segment of ``label`` (e.g., ``"LF_HIP"`` from ``"/World/Robot/LF_HIP"``)."""
    return label.rsplit("/", maxsplit=1)[-1]


def reset_mask_entries(model: newton.Model, world: wp.array[wp.int32]) -> np.ndarray:
    """Return the entry of a reset mask ``(world_count + 1,)`` that governs each entity with Newton world ``world``.

    The last entry governs Newton's global world (``-1``). A model built without worlds holds every joint in the
    global world and is still one world to a binding and to :class:`newton.solvers.SolverMuJoCo`, so entry ``0``
    governs all of its entities.
    """
    entries = world.numpy().astype(np.int64)
    if model.joint_count and model.joint_world.numpy().max() < 0:
        return np.zeros_like(entries)
    return np.where(entries < 0, model.world_count, entries)


def select_articulation(model: newton.Model, articulation: Pattern) -> ArticulationView:
    """Select exactly one articulation per world matching ``articulation``.

    Args:
        model: Target Newton model.
        articulation: Label pattern matched against full articulation labels.

    Returns:
        Non-homogeneous :class:`ArticulationView` with ``count_per_world == 1`` across all worlds.

    Raises:
        KeyError: If no articulation matches.
        ValueError: If any world has 0 or more than 1 matching articulation.
    """
    # Use homogeneous=False so heterogeneous scenes (e.g., extra per-world obstacles) still match;
    # _locate() validates robot consistency across all worlds.
    try:
        view = ArticulationView(model, articulation, homogeneous=False)
    except KeyError:
        labels = list(dict.fromkeys(model.articulation_label))
        raise KeyError(
            f"pattern {articulation!r} matches no articulation of the Newton model, whose first articulation "
            f"labels are {labels[:4]}"
        ) from None
    if view.count_per_world != 1:
        labels = [model.articulation_label[index] for index in view.articulation_ids.numpy()[0]]
        raise ValueError(
            f"pattern {articulation!r} matches {view.count_per_world} articulations per world "
            f"({', '.join(labels)}). A system holds one articulation per world, so give a pattern that matches "
            "only the robot"
        )
    if view.is_sparse:
        raise ValueError(
            f"pattern {articulation!r} matches an articulation in only {view.world_count} worlds of the Newton model"
        )
    return view


class _Rows:
    """Maps ``[world_count, width]`` signal slices onto a flat Warp array (strided view if uniform, else indexed)."""

    def __init__(self, name: str, rows: np.ndarray, device: torch.device) -> None:
        world_count, width = rows.shape
        self.name = name
        self.shape = (world_count, width)
        self.offset = int(rows[0, 0]) if rows.size else 0
        self.stride = int(rows[1, 0] - rows[0, 0]) if rows.size and world_count > 1 else 0
        regular = self.offset + self.stride * np.arange(world_count)[:, None] + np.arange(width)
        # Uniform per-world stride enables zero-copy as_strided views; irregular layouts use gather/scatter.
        self.index = None if np.array_equal(rows, regular) else torch.as_tensor(rows, dtype=torch.long, device=device)

    def _select(self, flat: Tensor) -> Tensor:
        if self.index is None:
            return flat.as_strided(self.shape, (self.stride, 1), self.offset)
        return flat[self.index]

    def read(self, array: wp.array) -> Tensor:
        return self._select(wp.to_torch(array, requires_grad=False))

    def write(self, array: wp.array, values: Tensor, world_mask: Tensor | None) -> None:
        try:
            values = values.expand(self.shape)  # Allow broadcasting a single row or scalar across worlds
        except RuntimeError:
            raise ValueError(
                f"{self.name} is written as {list(self.shape)}, one row per world, got {list(values.shape)}"
            ) from None
        flat = wp.to_torch(array, requires_grad=False)
        values = values.detach().to(flat.dtype)  # Keep simulator buffers outside PyTorch autograd graphs
        if world_mask is not None:
            values = torch.where(world_mask.unsqueeze(-1), values, self._select(flat))
        if self.index is None:
            self._select(flat).copy_(values)
        else:
            flat[self.index] = values


class BindingArticulation(BindingBase):
    """Binds a :class:`SystemArticulation` to a :class:`newton.Model`.

    Exposes per-world ``[world_count, width]`` tensors on the Newton model's device. For uniformly strided
    (replicated) scenes, :meth:`read_state` and :meth:`read_control` return zero-copy views into Newton's Warp
    buffers; for irregular layouts they return gathered copies and mutations must go through :meth:`write_state` /
    :meth:`write_control`. Control arrays cover actuated (revolute/prismatic) joints only.

    Args:
        model: Target Newton model.
        system: Expected :class:`SystemArticulation`.
        articulation: Label pattern matched against full articulation labels.

    Raises:
        KeyError: If no articulation matches ``articulation``.
        ValueError: If any world's articulation topology, labels, joint types, frames, axes, or gravity differs from
            ``system``.

    Attributes:
        system: Bound :class:`SystemArticulation`.
        world_count: Number of worlds.
        view: Underlying :class:`newton.selection.ArticulationView`.
        body_ids: Warp ``int`` array ``(world_count, body_count)`` of model body indices in ``spec.body_labels`` order.
        root_joint_ids: Warp ``int`` array ``(world_count,)`` of root joint indices.
        joint_coord_ids: NumPy ``int`` array ``[world_count, joint_coord_count]`` indexing ``joint_q``.
        joint_dof_ids: NumPy ``int`` array ``[world_count, joint_dof_count]`` indexing ``joint_qd``.
    """

    def __init__(self, model: newton.Model, system: SystemArticulation, articulation: Pattern = "*") -> None:
        spec = system.spec
        self.system = system
        self.view = select_articulation(model, articulation)
        self.world_count = int(self.view.world_count)
        root_joint_ids, body_ids = _locate(spec, model, self.view)
        self.body_ids = wp.array(body_ids, dtype=int, device=model.device)
        self.root_joint_ids = wp.array(root_joint_ids, dtype=int, device=model.device)
        # Kept as NumPy arrays for host-side index construction.
        self.joint_coord_ids = model.joint_q_start.numpy()[root_joint_ids][:, None] + np.arange(spec.joint_coord_count)
        self.joint_dof_ids = model.joint_qd_start.numpy()[root_joint_ids][:, None] + np.arange(spec.joint_dof_count)

        device = wp.device_to_torch(model.device)
        coords, dofs = self.joint_coord_ids, self.joint_dof_ids
        # Skip floating-base root coordinates (7) and DOFs (6) when indexing actuated controls.
        base_coord_count, self._base_dof_count = (7, 6) if spec.floating_base else (0, 0)
        actuated_coords, actuated_dofs = coords[:, base_coord_count:], dofs[:, self._base_dof_count :]
        self._state = {"joint_q": _Rows("joint_q", coords, device), "joint_qd": _Rows("joint_qd", dofs, device)}
        self._control = {
            name: _Rows(name, actuated_coords if name == "joint_target_q" else actuated_dofs, device)
            for name in spec.control
        }

    def read_state(self, state: newton.State) -> Frames:
        """Read ``joint_q`` ``[world_count, joint_coord_count]`` and ``joint_qd`` ``[world_count, joint_dof_count]``."""
        return {name: rows.read(getattr(state, name)) for name, rows in self._state.items()}

    def read_control(self, control: newton.Control) -> Frames:
        """Read each configured ``CONTROL`` signal ``[world_count, actuated_count]``."""
        return {name: rows.read(getattr(control, name)) for name, rows in self._control.items()}

    def write_state(self, state: newton.State, frames: Frames, world_mask: Tensor | None = None) -> None:
        """Write ``joint_q`` and ``joint_qd`` into ``state`` (call ``newton.eval_fk`` afterward to update body poses).

        Args:
            state: Target Newton state.
            frames: ``joint_q`` and ``joint_qd`` tensors broadcastable to ``[world_count, width]``.
            world_mask: Optional boolean mask ``[world_count]`` selecting worlds to write.

        Raises:
            ValueError: If any tensor cannot broadcast to ``[world_count, width]``.
        """
        for name, rows in self._state.items():
            rows.write(getattr(state, name), frames[name], world_mask)

    def write_control(self, control: newton.Control, frames: Frames) -> None:
        """Write configured ``CONTROL`` signals into ``control``.

        Raises:
            ValueError: If any tensor cannot broadcast to ``[world_count, actuated_count]``.
        """
        for name, rows in self._control.items():
            rows.write(getattr(control, name), frames[name], None)

    def state_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Derive finite ``(low, high)`` sampling bounds for ``joint_q`` and ``joint_qd`` from world 0's limits.

        Unbounded revolute/prismatic limits default to ``+-pi`` rad / ``+-1.0`` m; velocity limits are capped at
        ``20.0`` rad/s or m/s. Floating-base positions span ``+-0.25`` m around world 0's default pose, quaternion
        components span ``[-1, 1]`` (normalized on reset), and base linear/angular velocities span ``+-1.0`` m/s
        and ``+-2.0`` rad/s.
        """
        model = self.view.model
        low, high = self._actuated_reach()
        speed = np.minimum(model.joint_velocity_limit.numpy()[self.joint_dof_ids[0]], _SPEED_LIMIT)
        if self.system.floating_base:
            place = model.joint_q.numpy()[self.joint_coord_ids[0, :3]]
            low = np.concatenate((place - _BASE_REACH, np.full(4, -1.0), low))
            high = np.concatenate((place + _BASE_REACH, np.full(4, 1.0), high))
            speed[:3], speed[3:6] = _BASE_LINEAR_SPEED, _BASE_ANGULAR_SPEED
        return {"joint_q": self._bounds(low, high), "joint_qd": self._bounds(-speed, speed)}

    def command_bounds(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Derive ``(low, high)`` sampling bounds for configured ``CONTROL`` signals from world 0's limits.

        ``joint_target_q`` and ``joint_target_qd`` match :meth:`state_bounds`; ``joint_f`` uses effort limits
        (``+-inf`` if unbounded); ``joint_act`` is unbounded (``+-inf``).
        """
        model = self.view.model
        actuated_dofs = self.joint_dof_ids[0, self._base_dof_count :]
        speed = np.minimum(model.joint_velocity_limit.numpy()[actuated_dofs], _SPEED_LIMIT)
        effort = model.joint_effort_limit.numpy()[actuated_dofs]
        effort = np.where(effort < _UNLIMITED, effort, np.inf)
        unbounded = np.full(len(actuated_dofs), np.inf)
        sides = {
            "joint_target_q": self._actuated_reach(),
            "joint_target_qd": (-speed, speed),
            "joint_f": (-effort, effort),
            "joint_act": (-unbounded, unbounded),
        }
        return {name: self._bounds(*sides[name]) for name in self.system.spec.control}

    def _actuated_reach(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` position limits ``[actuated_count]`` [m or rad] for actuated joints."""
        model, joints = self.view.model, self.system.spec.joints
        # Actuated 1-DOF joints store position limits indexed by DOF id.
        actuated_dofs = self.joint_dof_ids[0, self._base_dof_count :]
        kinds = [joint.type for joint in joints if joint.actuated]
        reach = np.array([_ANGULAR_REACH if kind == "revolute" else _LINEAR_REACH for kind in kinds])
        lower = model.joint_limit_lower.numpy()[actuated_dofs]
        upper = model.joint_limit_upper.numpy()[actuated_dofs]
        return np.where(lower > -_UNLIMITED, lower, -reach), np.where(upper < _UNLIMITED, upper, reach)

    def _bounds(self, low: np.ndarray, high: np.ndarray) -> tuple[Tensor, Tensor]:
        device = wp.device_to_torch(self.view.model.device)
        return tuple(torch.as_tensor(side, dtype=torch.float32, device=device) for side in (low, high))


def _locate(
    spec: SystemArticulation.Spec, model: newton.Model, view: ArticulationView
) -> tuple[np.ndarray, np.ndarray]:
    """Validate every world's articulation against ``spec`` and return ``(root_joint_ids, body_ids)``."""
    joint_count = len(spec.joints)
    articulations = view.articulation_ids.numpy()[:, 0]
    root_joint_ids = model.articulation_start.numpy()[articulations]
    joint_counts = model.articulation_end.numpy()[articulations] - root_joint_ids
    if (joint_counts != joint_count).any():
        world = int(np.argmax(joint_counts != joint_count))
        raise ValueError(
            f"the articulation in world {world} has {joint_counts[world]} joints, the system has {joint_count}"
        )
    joints = root_joint_ids[:, None] + np.arange(joint_count)  # [world_count, joint_count]

    types = model.joint_type.numpy()[joints]
    expected_types = np.array([newton.JointType[joint.type.upper()] for joint in spec.joints])
    if (types != expected_types).any():
        world, joint = (int(index) for index in np.argwhere(types != expected_types)[0])
        raise ValueError(
            f"joint {model.joint_label[joints[world, joint]]!r} in world {world} is "
            f"{newton.JointType(types[world, joint]).name.lower()}, the system has "
            f"{spec.joints[joint].type} joint {spec.joint_labels[joint]!r} there"
        )

    # Order bodies by spec.body_labels column index.
    body_ids = np.empty_like(joints)
    body_ids[:, [joint.child for joint in spec.joints]] = model.joint_child.numpy()[joints]
    parents = np.array([joint.parent for joint in spec.joints])
    wrong = model.joint_parent.numpy()[joints] != np.where(parents >= 0, body_ids[:, parents], -1)
    if wrong.any():
        world = int(np.argwhere(wrong)[0, 0])
        raise ValueError(f"the joints in world {world} connect other bodies than the system's joint tree")

    for world, (world_joints, world_bodies) in enumerate(zip(joints.tolist(), body_ids.tolist(), strict=True)):
        for kind, found, expected in (
            ("joint", [leaf(model.joint_label[joint]) for joint in world_joints], spec.joint_labels),
            ("body", [leaf(model.body_label[body]) for body in world_bodies], spec.body_labels),
        ):
            if tuple(found) != expected:
                raise ValueError(
                    f"the {kind} labels in world {world} differ from the system's. "
                    f"The Newton model has {found}, the system has {list(expected)}"
                )

    # Verify joint frames and axes (excluding root parent_xform, which places each world's robot).
    child_frames = np.array([joint.child_xform for joint in spec.joints])
    parent_frames = np.array([joint.parent_xform for joint in spec.joints])
    same = _same_frames(model.joint_X_c.numpy()[joints], child_frames)
    same[:, 1:] &= _same_frames(model.joint_X_p.numpy()[joints[:, 1:]], parent_frames[1:])
    axial = np.flatnonzero([joint.actuated for joint in spec.joints])
    axes = np.array([spec.joints[joint].axis for joint in axial]).reshape(-1, 3)  # Safe when axial is empty
    found_axes = model.joint_axis.numpy()[model.joint_qd_start.numpy()[joints[:, axial]]]
    same[:, axial] &= np.abs(found_axes - axes).max(-1) < _FRAME_TOLERANCE
    if not same.all():
        world, joint = (int(index) for index in np.argwhere(~same)[0])
        raise ValueError(
            f"joint {model.joint_label[joints[world, joint]]!r} in world {world} has another frame or axis than "
            f"joint {spec.joint_labels[joint]!r} of the system, so the Newton model holds another robot than the "
            "system"
        )

    gravity = model.gravity.numpy()[: view.world_count]
    differs = ~np.isclose(gravity, spec.gravity, atol=1e-4).all(axis=-1)
    if differs.any():
        world = int(np.argmax(differs))
        raise ValueError(
            f"the gravity {gravity[world].tolist()} of world {world} differs from {list(spec.gravity)}, which the "
            "system holds"
        )
    return root_joint_ids, body_ids


def _same_frames(found: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Compare transforms ``[..., 7]`` within :data:`_FRAME_TOLERANCE`, invariant to quaternion sign."""
    position = np.abs(found[..., :3] - expected[..., :3]).max(-1)
    sign = np.sign((found[..., 3:] * expected[..., 3:]).sum(-1, keepdims=True))
    rotation = np.abs(found[..., 3:] - sign * expected[..., 3:]).max(-1)
    return np.maximum(position, rotation) < _FRAME_TOLERANCE
