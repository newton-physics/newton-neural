# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Differentiable, branch-free quaternion and rotation-vector operations on PyTorch tensors.

Quaternions use Newton's ``(x, y, z, w)`` layout with shape ``[..., 4]``; 3D vectors and rotation vectors have shape
``[..., 3]``. All functions broadcast across leading dimensions and are safe for ``torch.compile`` and CUDA graphs.
"""

from __future__ import annotations

import torch
from torch import Tensor


def normalize(q: Tensor) -> Tensor:
    """Normalize quaternions ``[..., 4]`` to unit length."""
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def canonicalize(q: Tensor) -> Tensor:
    """Flip quaternion signs where ``w < 0`` so each rotation has a unique representative ``[..., 4]``."""
    return torch.where(q[..., 3:4] < 0, -q, q)


def conjugate(q: Tensor) -> Tensor:
    """Return the conjugate (inverse rotation) of unit quaternions ``[..., 4]``."""
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    """Compose unit quaternions ``a * b`` ``[..., 4]`` (applies ``b`` first, then ``a``)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3D vectors ``v`` ``[..., 3]`` by unit quaternions ``q`` ``[..., 4]``."""
    xyz, v = torch.broadcast_tensors(q[..., :3], v)  # torch.linalg.cross does not broadcast across differing ranks
    t = 2.0 * torch.linalg.cross(xyz, v)
    return v + q[..., 3:4] * t + torch.linalg.cross(xyz, t)


def rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate 3D vectors ``v`` ``[..., 3]`` by the inverse of unit quaternions ``q`` ``[..., 4]``."""
    return rotate(conjugate(q), v)


def from_rotation_vector(r: Tensor) -> Tensor:
    """Convert rotation vectors ``[..., 3]`` (axis * angle [rad]) to unit quaternions ``[..., 4]``."""
    angle = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
    half = 0.5 * angle
    # Small-angle Taylor expansion of sin(theta / 2) / theta avoids 0 / 0 loss of precision.
    scale = torch.where(angle < 1e-6, 0.5 - angle.square() / 48.0, torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat((r * scale, torch.cos(half)), dim=-1)


def to_rotation_vector(q: Tensor) -> Tensor:
    """Convert unit quaternions ``[..., 4]`` to rotation vectors ``[..., 3]`` with angle in ``[0, pi]`` rad."""
    q = canonicalize(q)
    xyz, w = q[..., :3], q[..., 3:4]
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    # Small-angle limit of theta / sin(theta / 2) -> 2 / w avoids division by ~0.
    scale = torch.where(sin_half < 1e-6, 2.0 / w.clamp_min(1e-12), angle / sin_half.clamp_min(1e-12))
    return xyz * scale


def twist(q: Tensor, axis: Tensor) -> Tensor:
    """Extract the twist component of ``q = swing * twist`` about ``axis`` (with ``w >= 0``).

    For ``axis = up``, this yields the heading (yaw) quaternion of ``q``. Returns the identity quaternion for a
    180-degree rotation about an axis perpendicular to ``axis``.

    Args:
        q: Unit quaternions ``[..., 4]``.
        axis: Unit axis vectors ``[..., 3]`` or shared ``[3]``.

    Returns:
        Canonical unit twist quaternions ``[..., 4]``.
    """
    along = (q[..., :3] * axis).sum(dim=-1, keepdim=True)
    projected = torch.cat((along * axis, q[..., 3:4]), dim=-1)
    norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    identity = torch.zeros_like(projected)
    identity[..., 3] = 1.0
    return canonicalize(torch.where(norm > 1e-8, projected / norm.clamp_min(1e-12), identity))
