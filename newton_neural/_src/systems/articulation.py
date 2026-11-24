# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulated rigid-body robot system (fixed or floating base) and Warp robot-frame helpers.

Serializes the robot's kinematic tree as plain JSON-compatible data (:class:`SystemArticulation.Spec`) stored in
checkpoints, maps generalized coordinates into heading-invariant features for floating bases, and exposes matching
Warp ``@wp.func`` helpers (:func:`robot_frame`, :func:`robot_frame_floating`) for on-device contact sensors.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Self

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from ..core import rotation as rot
from ..core.signal import Frames, SignalRole, SignalSpec
from .articulation_binding import BindingArticulation, Pattern, leaf, select_articulation
from .system import SystemBase

_JOINT_SIZES = {"free": (7, 6), "revolute": (1, 1), "prismatic": (1, 1), "fixed": (0, 0)}
"""``(coord_count, dof_count)`` per supported joint type."""

_ACTUATED_TYPES = ("revolute", "prismatic")
"""Joint types driven by control arrays."""

_CONTROL_UNITS = {
    "joint_target_q": "m or rad",
    "joint_target_qd": "m/s or rad/s",
    "joint_f": "N or N*m",
    "joint_act": "N or N*m",
}
"""Supported :class:`newton.Control` arrays and their SI units."""

_IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


@wp.func
def robot_frame_floating(root_body_q: wp.transform, up: wp.vec3) -> wp.transform:
    """Compute the heading frame of a floating base from its world transform.

    Origin is at ``root_body_q`` translation; orientation is the twist component of ``root_body_q`` about ``up``.
    Warp counterpart to the heading frame computed by :class:`SystemArticulation`.

    Args:
        root_body_q: World transform of the root body.
        up: Unit vector opposite gravity (``system.up``).

    Returns:
        Transform from robot frame to world frame.
    """
    q = wp.transform_get_rotation(root_body_q)
    along = q[0] * up[0] + q[1] * up[1] + q[2] * up[2]
    heading = wp.quat(along * up[0], along * up[1], along * up[2], q[3])
    # A 180-degree rotation about a horizontal axis has undefined yaw; default to identity.
    heading = wp.where(wp.length(heading) > 1.0e-8, wp.normalize(heading), wp.quat_identity())
    return wp.transform(wp.transform_get_translation(root_body_q), heading)


@wp.func
def robot_frame(
    world: int,
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    joint_child: wp.array[int],
    up: wp.vec3,
    root_joint_ids: wp.array[int],
    floating_base: wp.bool,
) -> wp.transform:
    """Return the robot frame for ``world`` (heading frame if floating base, root anchor frame if fixed base).

    Matches the coordinate frame used by :class:`SystemArticulation` features so Warp contact kernels express
    positions and velocities in the same basis.

    Args:
        world: World index.
        body_q: World transforms of all bodies (``state.body_q``).
        joint_X_p: Parent transforms of all joints (``model.joint_X_p``).
        joint_child: Child body index per joint (``model.joint_child``).
        up: Unit vector opposite gravity (``system.up``).
        root_joint_ids: Root joint index per world (``binding.root_joint_ids``).
        floating_base: Whether the root joint is a free joint.

    Returns:
        Transform from robot frame to world frame.
    """
    root_joint = root_joint_ids[world]
    if floating_base:
        return robot_frame_floating(body_q[joint_child[root_joint]], up)
    return joint_X_p[root_joint]


class SystemArticulation(SystemBase):
    """Generalized-coordinate articulated robot (one articulation per world, fixed or floating base).

    Supports ``free`` (root only), ``revolute``, ``prismatic``, and ``fixed`` joints.

    **Signals:** ``joint_q`` ``[joint_coord_count]`` [m or rad, ``xyzw`` unit quaternion for floating base] and
    ``joint_qd`` ``[joint_dof_count]`` [m/s or rad/s] in Newton's layout, followed by one ``[actuated_count]``
    control tensor per name in ``spec.control`` (revolute and prismatic joints only).

    **Robot frame:** For a floating base with ``up = -gravity / |gravity|``, origin is ``joint_q[0:3]`` and rotation
    is heading ``R_h = twist(joint_q[3:7], up)``. For a fixed base, features use raw joint coordinates directly while
    Warp sensors (:func:`robot_frame`) use the world's root anchor transform ``model.joint_X_p[root_joint]``.

    **Features:** Fixed base: ``[joint_q, joint_qd, controls]``. Floating base (with base rotation ``R_b``):
    ``[joint_q[7:], R_h^T v_base (3), R_h^T w_base (3), joint_qd[6:], R_b^T (gravity / |gravity|) (3), controls]``,
    omitting global base position and heading for horizontal translation/yaw invariance.

    **Increments:** Fixed base: ``[dq, dqd]``. Floating base: ``[R_h^T dp_base (3), R_h^T r (3), dq[7:],
    R_h^T dv_base (3), R_h^T dw_base (3), dqd[6:]]``, where ``r`` is the rotation vector of
    ``q_following * q_current^-1`` [rad] and ``R_h`` is the heading at ``current``. :meth:`integrate` returns a
    normalized canonical quaternion (``w >= 0``).

    Args:
        spec: Kinematic tree and control specification (:class:`SystemArticulation.Spec`).

    Raises:
        ValueError: If ``spec`` has unsupported joint/control types, duplicate body or actuated-joint labels, or
            inconsistent topology.

    Attributes:
        spec: Underlying :class:`SystemArticulation.Spec`.
        floating_base: ``True`` if the root joint is a free joint.
        joint_coord_count: Width of ``joint_q``.
        joint_dof_count: Width of ``joint_qd``.
        actuated_count: Number of actuated (revolute + prismatic) joints.
        includes_actuators: ``True`` when driven by targets rather than direct forces (``joint_f``, ``joint_act``),
            meaning actuator dynamics are learned inside the model.
        root_body: Index of the root body in ``spec.body_labels``.
        anchor: Root joint parent transform ``(x, y, z, qx, qy, qz, qw)`` [m].
        up: Unit vector opposite gravity ``[3]`` (zeros if gravity is zero).
        down: Unit vector along gravity ``[3]`` (zeros if gravity is zero).
    """

    up: Tensor
    down: Tensor

    @dataclass(frozen=True)
    class Joint:
        """Kinematic definition of a single joint."""

        type: str
        """Lowercase :class:`newton.JointType` name: ``"free"``, ``"revolute"``, ``"prismatic"``, or ``"fixed"``."""

        parent: int
        """Parent body index in ``Spec.body_labels``, or ``-1`` for the world."""

        child: int
        """Child body index in ``Spec.body_labels``."""

        parent_xform: tuple[float, ...] = _IDENTITY
        """Joint transform in parent body frame ``(x, y, z, qx, qy, qz, qw)`` [m]."""

        child_xform: tuple[float, ...] = _IDENTITY
        """Joint transform in child body frame ``(x, y, z, qx, qy, qz, qw)`` [m]."""

        axis: tuple[float, ...] = (0.0, 0.0, 0.0)
        """Unit motion axis in joint frame for revolute/prismatic joints (zeros otherwise)."""

        def __post_init__(self) -> None:
            # Coerce JSON-deserialized lists back to float tuples.
            for name in ("parent_xform", "child_xform", "axis"):
                object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))

        @property
        def actuated(self) -> bool:
            """``True`` for revolute and prismatic joints."""
            return self.type in _ACTUATED_TYPES

    @dataclass(frozen=True)
    class Spec:
        """Plain-data kinematic specification of an articulated robot.

        Body labels and actuated-joint labels must each be unique within the articulation (fixed joints may share
        labels).
        """

        control: tuple[str, ...]
        """Names of :class:`newton.Control` arrays consumed (subset of ``joint_target_q``, ``joint_target_qd``,
        ``joint_f``, ``joint_act``)."""

        gravity: tuple[float, ...]
        """World-frame gravity vector [m/s^2]."""

        body_labels: tuple[str, ...]
        """Leaf label of each body in model order."""

        joint_labels: tuple[str, ...]
        """Leaf label of each joint in model order."""

        joints: tuple[SystemArticulation.Joint, ...]
        """Joint tree in topological order (parents before children, root joint first)."""

        body_com: tuple[tuple[float, ...], ...]
        """Center of mass of each body in its body frame [m]."""

        def __post_init__(self) -> None:
            # Coerce JSON-deserialized lists and dicts into immutable tuples and Joint instances.
            joint_type = SystemArticulation.Joint
            coerced = {
                "control": tuple(self.control),
                "gravity": tuple(float(x) for x in self.gravity),
                "body_labels": tuple(self.body_labels),
                "joint_labels": tuple(self.joint_labels),
                "joints": tuple(j if isinstance(j, joint_type) else joint_type(**j) for j in self.joints),
                "body_com": tuple(tuple(float(x) for x in com) for com in self.body_com),
            }
            for name, value in coerced.items():
                object.__setattr__(self, name, value)

        @property
        def joint_coord_count(self) -> int:
            """Total coordinate count (width of ``joint_q``)."""
            return sum(_JOINT_SIZES[joint.type][0] for joint in self.joints)

        @property
        def joint_dof_count(self) -> int:
            """Total degree-of-freedom count (width of ``joint_qd``)."""
            return sum(_JOINT_SIZES[joint.type][1] for joint in self.joints)

        @property
        def floating_base(self) -> bool:
            """``True`` if the root joint is a free joint."""
            return self.joints[0].type == "free"

        @property
        def actuated_joint_labels(self) -> tuple[str, ...]:
            """Labels of actuated joints in control-column order."""
            return tuple(label for label, joint in zip(self.joint_labels, self.joints, strict=True) if joint.actuated)

        def to_dict(self) -> dict[str, Any]:
            """Serialize to JSON-compatible plain data."""
            return _plain(dataclasses.asdict(self))

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> Self:
            """Reconstruct a :class:`Spec` from :meth:`to_dict` output."""
            return cls(**data)

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        _validate(spec)
        self.spec = spec
        self.floating_base = spec.floating_base
        self.joint_coord_count = spec.joint_coord_count
        self.joint_dof_count = spec.joint_dof_count
        self.actuated_count = spec.joint_dof_count - (6 if spec.floating_base else 0)
        self.includes_actuators = not {"joint_f", "joint_act"} & set(spec.control)
        self.root_body = spec.joints[0].child
        self.anchor = spec.joints[0].parent_xform

        gravity = torch.tensor(spec.gravity, dtype=torch.float32)
        down = gravity / gravity.norm().clamp_min(1e-12)
        # Derived from spec; non-persistent avoids duplicating them in state_dict.
        self.register_buffer("down", down, persistent=False)
        self.register_buffer("up", -down, persistent=False)

        self.signals = {
            "joint_q": SignalSpec("joint_q", (self.joint_coord_count,), SignalRole.STATE, unit="m or rad"),
            "joint_qd": SignalSpec("joint_qd", (self.joint_dof_count,), SignalRole.STATE, unit="m/s or rad/s"),
        }
        for name in spec.control:
            self.signals[name] = SignalSpec(name, (self.actuated_count,), SignalRole.CONTROL, unit=_CONTROL_UNITS[name])
        state_count = self.joint_coord_count + self.joint_dof_count
        # Floating base replaces position (3) + quat (4) with projected gravity (3) in features (-4),
        # and replaces quat (4) with a 3D rotation vector in increments (-1).
        self.feature_count = state_count + self.actuated_count * len(spec.control) - (4 if self.floating_base else 0)
        self.increment_count = state_count - (1 if self.floating_base else 0)

    @classmethod
    def from_model(
        cls, model: newton.Model, articulation: Pattern = "*", *, control: str | tuple[str, ...] = ("joint_target_q",)
    ) -> Self:
        """Build a :class:`SystemArticulation` from ``model`` using world 0 as the template.

        Args:
            model: Source Newton model.
            articulation: Label pattern matched against full articulation labels (must select one articulation per
                world).
            control: Name or tuple of :class:`newton.Control` arrays driving the robot.

        Raises:
            KeyError: If no articulation matches ``articulation``.
            ValueError: If any world has != 1 match, gravity varies across worlds, or a joint type is unsupported.
        """
        view = select_articulation(model, articulation)
        articulation_id = int(view.articulation_ids.numpy()[0, 0])
        joint_ids = range(
            int(model.articulation_start.numpy()[articulation_id]), int(model.articulation_end.numpy()[articulation_id])
        )
        joint_type, joint_qd_start = model.joint_type.numpy(), model.joint_qd_start.numpy()
        joint_parent, joint_child = model.joint_parent.numpy(), model.joint_child.numpy()
        joint_X_p, joint_X_c, joint_axis = model.joint_X_p.numpy(), model.joint_X_c.numpy(), model.joint_axis.numpy()
        body_ids = sorted(int(joint_child[j]) for j in joint_ids)
        local = {body: index for index, body in enumerate(body_ids)}
        joints = []
        for j in joint_ids:
            kind = newton.JointType(int(joint_type[j])).name.lower()
            parent = int(joint_parent[j])
            if parent >= 0 and parent not in local:
                raise ValueError(
                    f"joint {model.joint_label[j]!r} attaches the articulation to body {model.body_label[parent]!r} "
                    "outside it. A system's root joint attaches to the world"
                )
            joints.append(
                cls.Joint(
                    type=kind,
                    parent=local.get(parent, -1),
                    child=local[int(joint_child[j])],
                    parent_xform=joint_X_p[j].tolist(),
                    child_xform=joint_X_c[j].tolist(),
                    axis=joint_axis[joint_qd_start[j]].tolist() if kind in _ACTUATED_TYPES else (0.0, 0.0, 0.0),
                )
            )
        gravity = model.gravity.numpy()[: view.world_count]
        if not np.allclose(gravity, gravity[0]):
            raise ValueError("gravity differs between the worlds of the Newton model. A system holds one gravity")
        return cls(
            cls.Spec(
                control=(control,) if isinstance(control, str) else tuple(control),
                gravity=gravity[0].tolist(),
                body_labels=tuple(leaf(model.body_label[body]) for body in body_ids),
                joint_labels=tuple(leaf(model.joint_label[j]) for j in joint_ids),
                joints=tuple(joints),
                body_com=model.body_com.numpy()[body_ids].tolist(),
            )
        )

    def config(self) -> dict[str, Any]:
        """Return ``{"spec": spec.to_dict()}`` for checkpoint serialization."""
        return {"spec": self.spec.to_dict()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Reconstruct from :meth:`config`."""
        return cls(cls.Spec.from_dict(config["spec"]))

    def extra_repr(self) -> str:
        """Summary string showing base type, coordinate/DOF counts, and control signals."""
        return (
            f"floating_base={self.floating_base}, joint_coord_count={self.joint_coord_count}, "
            f"joint_dof_count={self.joint_dof_count}, control={self.spec.control}"
        )

    def bind(self, model: newton.Model, articulation: Pattern = "*") -> BindingArticulation:
        """Bind this system to ``model`` and validate every world against :attr:`spec`.

        Args:
            model: Target Newton model.
            articulation: Label pattern matched against full articulation labels.

        Returns:
            :class:`BindingArticulation` exposing per-world state/control tensors and bounds.

        Raises:
            KeyError: If no articulation matches ``articulation``.
            ValueError: If any world's articulation or gravity disagrees with :attr:`spec`.
        """
        return BindingArticulation(model, self, articulation)

    def kinematic_model(self, world_count: int, *, device: wp.DeviceLike | torch.device) -> newton.Model:
        """Build a minimal shape-free :class:`newton.Model` of this articulation for standalone forward kinematics.

        Running ``newton.eval_fk`` reproduces the original body poses and CoM-referenced velocities without needing
        the full scene. Bodies have dummy unit mass/inertia and original ``body_com``; fixed bases sit at
        :attr:`anchor`. Compatible with :meth:`bind`.

        Args:
            world_count: Number of replicated worlds.
            device: Target Warp or PyTorch device.

        Returns:
            Finalized Newton model with one articulation labeled ``"robot"`` per world.
        """
        spec = self.spec
        robot = newton.ModelBuilder(gravity=spec.gravity)
        for label, com in zip(spec.body_labels, spec.body_com, strict=True):
            robot.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=com, label=label)
        joint_ids = []
        for label, joint in zip(spec.joint_labels, spec.joints, strict=True):
            frames = {
                "parent_xform": wp.transform(*joint.parent_xform),
                "child_xform": wp.transform(*joint.child_xform),
                "label": label,
            }
            if joint.type == "free":
                joint_ids.append(robot.add_joint_free(joint.child, **frames))
            elif joint.type == "fixed":
                joint_ids.append(robot.add_joint_fixed(joint.parent, joint.child, **frames))
            elif joint.type == "revolute":
                joint_ids.append(robot.add_joint_revolute(joint.parent, joint.child, axis=joint.axis, **frames))
            else:
                joint_ids.append(robot.add_joint_prismatic(joint.parent, joint.child, axis=joint.axis, **frames))
        robot.add_articulation(joint_ids, label="robot")
        scene = newton.ModelBuilder(gravity=spec.gravity)
        scene.replicate(robot, world_count)
        return scene.finalize(device=str(device))

    def features(self, frames: Frames) -> Tensor:
        """Extract network features ``[..., feature_count]`` from ``joint_q``, ``joint_qd``, and controls."""
        joint_q, joint_qd = frames["joint_q"], frames["joint_qd"]
        controls = [frames[name] for name in self.spec.control]
        if not self.floating_base:
            return torch.cat((joint_q, joint_qd, *controls), dim=-1)
        base = rot.normalize(joint_q[..., 3:7])
        heading = rot.twist(base, self.up.to(joint_q))
        # Rotate linear and angular base velocities [..., 2, 3] into the heading frame together.
        velocity = rot.rotate_inverse(heading.unsqueeze(-2), joint_qd[..., :6].unflatten(-1, (2, 3)))
        gravity = rot.rotate_inverse(base, self.down.to(joint_q))
        return torch.cat((joint_q[..., 7:], velocity.flatten(-2), joint_qd[..., 6:], gravity, *controls), dim=-1)

    def increment(self, current: Frames, following: Frames) -> Tensor:
        """Compute heading-frame state increments ``[..., increment_count]`` from ``current`` to ``following``."""
        q0, q1 = current["joint_q"], following["joint_q"]
        qd0, qd1 = current["joint_qd"], following["joint_qd"]
        if not self.floating_base:
            return torch.cat((q1 - q0, qd1 - qd0), dim=-1)
        base0, base1 = rot.normalize(q0[..., 3:7]), rot.normalize(q1[..., 3:7])
        heading = rot.twist(base0, self.up.to(q0))
        turn = rot.to_rotation_vector(rot.multiply(base1, rot.conjugate(base0)))
        delta_qd = qd1 - qd0
        # Rotate position delta, rotation delta, and base linear/angular velocity deltas into heading frame.
        world = torch.stack((q1[..., :3] - q0[..., :3], turn, delta_qd[..., :3], delta_qd[..., 3:6]), dim=-2)
        local = rot.rotate_inverse(heading.unsqueeze(-2), world)
        return torch.cat(
            (
                local[..., :2, :].flatten(-2),
                q1[..., 7:] - q0[..., 7:],
                local[..., 2:, :].flatten(-2),
                delta_qd[..., 6:],
            ),
            dim=-1,
        )

    def integrate(self, current: Frames, increment: Tensor) -> Frames:
        """Apply ``increment`` to ``current`` ``joint_q`` and ``joint_qd`` (inverts :meth:`increment`).

        For floating bases, returns a normalized canonical quaternion (``w >= 0``).
        """
        joint_q, joint_qd = current["joint_q"], current["joint_qd"]
        if not self.floating_base:
            count = self.joint_coord_count
            return {"joint_q": joint_q + increment[..., :count], "joint_qd": joint_qd + increment[..., count:]}
        count = self.joint_coord_count - 7  # Actuated joint coordinates after the 7D floating base
        base = rot.normalize(joint_q[..., 3:7])
        heading = rot.twist(base, self.up.to(joint_q))
        local = torch.cat((increment[..., :6], increment[..., 6 + count : 12 + count]), dim=-1).unflatten(-1, (4, 3))
        move, turn, linear, angular = rot.rotate(heading.unsqueeze(-2), local).unbind(dim=-2)
        turned = rot.canonicalize(rot.normalize(rot.multiply(rot.from_rotation_vector(turn), base)))
        return {
            "joint_q": torch.cat(
                (joint_q[..., :3] + move, turned, joint_q[..., 7:] + increment[..., 6 : 6 + count]), dim=-1
            ),
            "joint_qd": torch.cat(
                (
                    joint_qd[..., :3] + linear,
                    joint_qd[..., 3:6] + angular,
                    joint_qd[..., 6:] + increment[..., 12 + count :],
                ),
                dim=-1,
            ),
        }

    def error(self, reference: Frames, other: Frames) -> Frames:
        """Compute per-coordinate absolute state errors.

        For floating bases, collapses the 4D base quaternion difference into a single geodesic rotation angle in
        ``[0, pi]`` rad, yielding ``joint_q`` shape ``[..., joint_coord_count - 3]``: base position (3) [m],
        orientation angle (1) [rad], and actuated joint coordinates [m or rad].
        """
        errors = super().error(reference, other)
        if self.floating_base:
            turn = rot.multiply(other["joint_q"][..., 3:7], rot.conjugate(reference["joint_q"][..., 3:7]))
            angle = torch.linalg.vector_norm(rot.to_rotation_vector(turn), dim=-1, keepdim=True)
            error = errors["joint_q"]
            errors["joint_q"] = torch.cat((error[..., :3], angle, error[..., 7:]), dim=-1)
        return errors


def _plain(value: Any) -> Any:
    """Recursively convert tuples to lists for JSON serialization."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _validate(spec: SystemArticulation.Spec) -> None:
    """Validate ``spec`` topology, labels, joint types, controls, and gravity."""
    joint_count = len(spec.joints)
    if joint_count == 0:
        raise ValueError("the articulation has no joints")
    if not (len(spec.joint_labels) == len(spec.body_labels) == len(spec.body_com) == joint_count):
        raise ValueError(
            f"the spec holds {joint_count} joints, {len(spec.joint_labels)} joint labels, {len(spec.body_labels)} "
            f"body labels and {len(spec.body_com)} centres of mass. A joint tree has one body per joint"
        )
    # Require unique leaf labels for bodies and actuated joints; fixed joints may share names in USD assets.
    for kind, labels in (("body", spec.body_labels), ("actuated joint", spec.actuated_joint_labels)):
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(
                f"the {kind} labels {repeated} repeat among {list(labels)}. Every {kind} of the robot is found by "
                "the final component of its label, so give each a label of its own"
            )
    for index, (label, joint) in enumerate(zip(spec.joint_labels, spec.joints, strict=True)):
        if joint.type not in _JOINT_SIZES:
            raise ValueError(f"joint {label!r} is a {joint.type} joint. Supported types are {', '.join(_JOINT_SIZES)}")
        if (joint.parent < 0) != (index == 0):
            raise ValueError(
                f"joint {label!r} has parent {joint.parent}. Only the root joint, which comes first, "
                "attaches to the world"
            )
        if joint.type == "free" and index > 0:
            raise ValueError(f"joint {label!r} is a free joint inside the tree. Only the root joint may be free")
    root = spec.joints[0]
    if root.type == "free" and not np.allclose((*root.parent_xform, *root.child_xform), (*_IDENTITY, *_IDENTITY)):
        raise ValueError(
            f"the free joint {spec.joint_labels[0]!r} has a parent or child transform. A floating base needs "
            "both to be the identity, so that its joint coordinates are the pose of the base body"
        )
    unknown = [name for name in spec.control if name not in _CONTROL_UNITS]
    if unknown or len(set(spec.control)) != len(spec.control):
        raise ValueError(f"control must hold distinct names among {', '.join(_CONTROL_UNITS)}, got {spec.control}")
    if spec.floating_base and not any(spec.gravity):
        raise ValueError("a floating base needs gravity, which defines the up direction of its heading frame")
