# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Procedural Newton test fixtures (quadruped, fixed-base arm, box scenes) and random frame generators.

All ``*_model`` builders return a finalized :class:`newton.Model` with ``world_count`` worlds, gravity
``(0, 0, -9.81)`` m/s^2, a ground plane at ``z = 0``, and one ``"quadruped"`` or ``"arm"`` articulation per world.
"""

from __future__ import annotations

import newton
import torch
import warp as wp

from newton_neural import Frames
from newton_neural._src.systems.articulation import SystemArticulation

WORLD_SPACING = (2.0, 0.0, 0.0)
"""Spacing between consecutive worlds [m]."""

QUADRUPED_LEGS = ("LF", "RF", "LH", "RH")
"""Quadruped leg prefixes in joint order."""

ARM_ANCHOR = (0.1, -0.2, 0.3)
"""Arm root joint offset in each world [m]."""

_INERTIA = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
"""Default diagonal inertia [kg*m^2] for shape-less links."""


def add_quadruped(builder: newton.ModelBuilder) -> None:
    """Add a 13-body floating-base quadruped (``"quadruped"``, 19 coords / 18 DOFs) with 4 foot spheres."""
    base = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.55), wp.quat_identity()), mass=10.0, com=(0.02, -0.01, 0.03), label="base"
    )
    builder.add_shape_box(base, hx=0.3, hy=0.15, hz=0.08, label="trunk")
    joints = [builder.add_joint_free(base, label="floating_base")]
    for leg, (x, y) in zip(QUADRUPED_LEGS, ((0.25, 0.15), (0.25, -0.15), (-0.25, 0.15), (-0.25, -0.15)), strict=True):
        hip = builder.add_link(mass=1.0, inertia=_INERTIA, com=(0.0, 0.01, 0.0), label=f"{leg}_HIP")
        thigh = builder.add_link(mass=1.0, com=(0.0, 0.0, -0.12), label=f"{leg}_THIGH")
        shank = builder.add_link(mass=0.5, com=(0.0, 0.0, -0.1), label=f"{leg}_SHANK")
        builder.add_shape_capsule(
            thigh, xform=wp.transform((0.0, 0.0, -0.125), wp.quat_identity()), radius=0.03, half_height=0.095
        )
        builder.add_shape_sphere(
            shank, xform=wp.transform((0.0, 0.0, -0.25), wp.quat_identity()), radius=0.03, label=f"{leg}_FOOT"
        )
        joints += [
            builder.add_joint_revolute(
                base,
                hip,
                axis=(1.0, 0.0, 0.0),
                parent_xform=wp.transform((x, y, 0.0), wp.quat_rpy(0.0, 0.0, 0.1)),
                label=f"{leg}_HAA",
            ),
            builder.add_joint_revolute(
                hip,
                thigh,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform((0.0, 0.05 if y > 0 else -0.05, 0.0), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, 0.01), wp.quat_identity()),
                label=f"{leg}_HFE",
            ),
            builder.add_joint_revolute(
                thigh,
                shank,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform((0.0, 0.0, -0.25), wp.quat_identity()),
                label=f"{leg}_KFE",
            ),
        ]
    builder.add_articulation(joints, label="quadruped")


def add_arm(builder: newton.ModelBuilder) -> None:
    """Add a 9-body fixed-base 6-DOF arm + 2-finger gripper (``"arm"``, 8 coords / 8 DOFs) at :data:`ARM_ANCHOR`."""
    parent = -1
    joints = []
    axes = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    for i, axis in enumerate(axes):
        link = builder.add_link(mass=1.0, com=(0.0, 0.01 * i, 0.05), label=f"link{i + 1}")
        builder.add_shape_capsule(link, radius=0.03, half_height=0.05)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=axis,
                parent_xform=wp.transform(ARM_ANCHOR if i == 0 else (0.0, 0.0, 0.15), wp.quat_rpy(0.0, 0.0, 0.2 * i)),
                child_xform=wp.transform((0.0, 0.0, -0.02), wp.quat_identity()),
                label=f"joint{i + 1}",
            )
        )
        parent = link
    hand = builder.add_link(mass=0.5, label="hand")
    builder.add_shape_box(hand, hx=0.03, hy=0.05, hz=0.02)
    joints.append(
        builder.add_joint_fixed(
            parent, hand, parent_xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), label="hand_joint"
        )
    )
    for side, y in (("left", 0.03), ("right", -0.03)):
        finger = builder.add_link(mass=0.1, com=(0.0, 0.0, 0.02), label=f"{side}_finger")
        builder.add_shape_box(finger, hx=0.01, hy=0.005, hz=0.03)
        joints.append(
            builder.add_joint_prismatic(
                hand,
                finger,
                axis=(0.0, 1.0 if y > 0 else -1.0, 0.0),
                parent_xform=wp.transform((0.0, y, 0.05), wp.quat_identity()),
                label=f"{side}_finger_joint",
            )
        )
    builder.add_articulation(joints, label="arm")


def _add_box(builder: newton.ModelBuilder, index: int) -> None:
    """Add a 0.06 m free cube ``box<index>`` resting on the ground."""
    box = builder.add_body(
        xform=wp.transform((0.4, 0.1 * index, 0.03), wp.quat_identity()), mass=0.2, label=f"box{index}"
    )
    builder.add_shape_box(box, hx=0.03, hy=0.03, hz=0.03)


def _replicated(robot: newton.ModelBuilder, world_count: int, device: torch.device | str) -> newton.Model:
    scene = newton.ModelBuilder()
    scene.replicate(robot, world_count, spacing=WORLD_SPACING)
    scene.add_ground_plane()
    return scene.finalize(device=str(device))


def quadruped_model(world_count: int, device: torch.device | str) -> newton.Model:
    """Return ``world_count`` replicated quadrupeds above a flat ground plane."""
    robot = newton.ModelBuilder()
    add_quadruped(robot)
    return _replicated(robot, world_count, device)


def arm_model(world_count: int, device: torch.device | str) -> newton.Model:
    """Return ``world_count`` replicated fixed-base arms above a ground plane."""
    robot = newton.ModelBuilder()
    add_arm(robot)
    return _replicated(robot, world_count, device)


def arm_with_box_model(world_count: int, device: torch.device | str) -> newton.Model:
    """Return ``world_count`` replicated worlds each with a free ``box0`` followed by ``arm`` (non-zero row offset)."""
    robot = newton.ModelBuilder()
    _add_box(robot, 0)
    add_arm(robot)
    return _replicated(robot, world_count, device)


def arm_with_uneven_boxes_model(world_count: int, device: torch.device | str) -> newton.Model:
    """Return worlds with 1 box (even worlds) or 2 boxes (odd worlds) before ``arm`` (tests non-strided indexing)."""
    scene = newton.ModelBuilder()
    for world in range(world_count):
        robot = newton.ModelBuilder()
        for index in range(1 + world % 2):
            _add_box(robot, index)
        add_arm(robot)
        scene.add_world(robot, xform=wp.transform((WORLD_SPACING[0] * world, 0.0, 0.0), wp.quat_identity()))
    scene.add_ground_plane()
    return scene.finalize(device=str(device))


def random_frames(
    system: SystemArticulation,
    world_count: int,
    frame_count: int | None = None,
    *,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> Frames:
    """Generate valid random ``STATE`` and ``CONTROL`` tensors (unit quaternions for floating bases)."""
    generator = torch.Generator().manual_seed(seed)
    lead = (world_count,) if frame_count is None else (world_count, frame_count)
    frames = {name: torch.randn(*lead, *spec.shape, generator=generator) for name, spec in system.signals.items()}
    joint_q = frames["joint_q"]
    joint_q *= 0.3
    if system.floating_base:
        joint_q[..., :2] *= 10.0
        joint_q[..., 2] += 0.6
        joint_q[..., 3:7] = torch.nn.functional.normalize(torch.randn(*lead, 4, generator=generator), dim=-1)
    return {name: value.to(device) for name, value in frames.items()}
