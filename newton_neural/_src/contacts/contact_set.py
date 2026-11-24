# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collision-detected contact set (:class:`ContactInputSet`) with permutation-invariant slot pooling.

Extracts all rigid contacts involving robot links from Newton's contact buffers in two CUDA-graph-safe Warp kernels
and pools them into a fixed-size learned contact code independent of contact count and order.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Self

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor, nn

from ..core.normalizer import FEATURE_LIMIT, Normalizer
from ..core.signal import Frames, SignalRole, SignalSpec
from ..systems.articulation import SystemArticulation, robot_frame
from ..systems.articulation_binding import BindingArticulation
from .contact_input import ContactInputBase, ContactSensorBase
from .kinematics import contact_gap, point_velocity, require_unreduced

ROW_WIDTH = 14
"""Per-row features: ``[position (3), normal (3), lever_arm (3), gap (1), rel_velocity (3), other_dynamic (1)]``."""

_DYNAMIC = int(newton.BodyFlags.DYNAMIC)


@wp.func
def _lookup(table: wp.array[wp.int32], body: int) -> int:
    """Return ``table[body]``, or ``-1`` for static world geometry (``body < 0``)."""
    if body < 0:
        return -1
    return table[body]


@wp.func
def _body_transform(body: int, body_q: wp.array[wp.transform]) -> wp.transform:
    """World transform of ``body``, or identity for static world geometry (``body < 0``)."""
    if body < 0:
        return wp.transform_identity()
    return body_q[body]


@wp.func
def _is_dynamic(body: int, body_flags: wp.array[wp.int32]) -> float:
    """Return ``1.0`` if ``body`` is dynamic, ``0.0`` if static or kinematic."""
    if body < 0:
        return 0.0
    return wp.where((body_flags[body] & _DYNAMIC) != 0, 1.0, 0.0)


@wp.func
def _add_row(
    world: int,
    link: int,
    to_frame: wp.transform,
    point: wp.vec3,
    normal: wp.vec3,
    origin: wp.vec3,
    gap: float,
    velocity: wp.vec3,
    other_moves: float,
    rows: wp.array3d[float],
    links: wp.array2d[wp.int32],
    count: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    """Atomically claim a slot in ``world`` and write one robot-link contact row in the robot frame."""
    row = wp.atomic_add(count, world, 1)
    if row >= links.shape[1]:
        # Roll back count so it stays clamped to capacity and record overflow separately.
        wp.atomic_sub(count, world, 1)
        wp.atomic_add(overflow, world, 1)
        return
    position = wp.transform_point(to_frame, point)
    direction = wp.transform_vector(to_frame, normal)
    lever = wp.transform_vector(to_frame, point - origin)
    relative = wp.transform_vector(to_frame, velocity)
    for axis in range(3):
        rows[world, row, axis] = position[axis]
        rows[world, row, 3 + axis] = direction[axis]
        rows[world, row, 6 + axis] = lever[axis]
        rows[world, row, 10 + axis] = relative[axis]
    rows[world, row, 9] = gap
    rows[world, row, 13] = other_moves
    links[world, row] = link


@wp.kernel(enable_backward=False)
def _clear_rows_kernel(
    rows: wp.array3d[float],
    links: wp.array2d[wp.int32],
    count: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    world, row = wp.tid()
    # One thread per slot clears only previously populated rows; per-world loops bottleneck sensor latency.
    if links[world, row] >= 0:
        for channel in range(ROW_WIDTH):
            rows[world, row, channel] = 0.0
        links[world, row] = -1
    if row == 0:
        count[world] = 0
        overflow[world] = 0


@wp.kernel(enable_backward=False)
def _contact_rows_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_offset0: wp.array[wp.vec3],
    contact_offset1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[float],
    contact_margin1: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_flags: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_child: wp.array[wp.int32],
    up: wp.vec3,
    root_joint_ids: wp.array[wp.int32],
    floating_base: wp.bool,
    body_world: wp.array[wp.int32],
    body_link: wp.array[wp.int32],
    rows: wp.array3d[float],
    links: wp.array2d[wp.int32],
    count: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
):
    contact = wp.tid()
    # Guard against detector count exceeding the allocated rigid_contact buffer size.
    if contact >= wp.min(contact_count[0], contact_shape0.shape[0]):
        return
    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    if shape0 < 0 or shape1 < 0:  # Unused/invalid contact slot
        return
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    link0 = _lookup(body_link, body0)
    link1 = _lookup(body_link, body1)
    world = wp.max(_lookup(body_world, body0), _lookup(body_world, body1))
    if world < 0:
        return

    # Transform body-local support points and offsets to world coordinates.
    X0 = _body_transform(body0, body_q)
    X1 = _body_transform(body1, body_q)
    normal = contact_normal[contact]
    support0 = wp.transform_point(X0, contact_point0[contact])
    support1 = wp.transform_point(X1, contact_point1[contact])
    gap = contact_gap(normal, support0, support1, contact_margin0[contact], contact_margin1[contact])
    point0 = support0 + wp.transform_vector(X0, contact_offset0[contact])
    point1 = support1 + wp.transform_vector(X1, contact_offset1[contact])
    velocity0 = point_velocity(body0, point0, body_q, body_qd, body_com)
    velocity1 = point_velocity(body1, point1, body_q, body_qd, body_com)

    frame = robot_frame(world, body_q, joint_X_p, joint_child, up, root_joint_ids, floating_base)
    to_frame = wp.transform_inverse(frame)

    if link0 >= 0:
        _add_row(
            world,
            link0,
            to_frame,
            point0,
            normal,
            wp.transform_get_translation(X0),
            gap,
            velocity0 - velocity1,
            _is_dynamic(body1, body_flags),
            rows,
            links,
            count,
            overflow,
        )
    if link1 >= 0:
        _add_row(
            world,
            link1,
            to_frame,
            point1,
            -normal,
            wp.transform_get_translation(X1),
            gap,
            velocity1 - velocity0,
            _is_dynamic(body0, body_flags),
            rows,
            links,
            count,
            overflow,
        )


class ContactInputSet(ContactInputBase):
    """Contact input built from arbitrary rigid contacts reported by Newton's collision detector.

    Emits one row per robot link involved in each detected contact (self-contacts emit two rows) in the robot frame
    (heading frame for floating bases, anchor frame for fixed bases), then pools all rows per world into a
    permutation-invariant code of size ``slot_count * slot_dim + 1``.

    **Signals:**

    - ``contact_rows`` ``[capacity, 14]`` float32 (zero-padded):

      - ``[0:3]`` contact point on the margin-expanded link surface [m],
      - ``[3:6]`` unit normal pointing from link into the other body,
      - ``[6:9]`` lever arm from link origin to contact point [m],
      - ``[9]`` signed surface gap along normal [m] (negative = penetration),
      - ``[10:13]`` relative surface velocity (link minus other body) [m/s],
      - ``[13]`` ``1.0`` if other body is dynamic, ``0.0`` if static or kinematic.

    - ``contact_links`` ``[capacity]`` int32: link index in ``system.spec.body_labels``, or ``-1`` for padding.
    - ``contact_count`` ``[]`` int32: number of populated rows (``0 .. capacity``).

    **Code:** Flattened attention-pooled slots ``[slot_count * slot_dim]`` concatenated with
    ``log(1 + count) / log(1 + capacity)``. All zeros when no contacts exist. Marks ``gated = True`` because the
    encoder is trained end to end.

    Args:
        link_count: Number of bodies in the robot.
        capacity: Maximum contact rows retained per world (excess rows increment ``overflow``).
        slot_count: Number of attention pooling slots.
        slot_dim: Width of each pooling slot.
        link_embedding_dim: Width of the learned per-link embedding.
    """

    recomputable: ClassVar[bool] = False
    reads_contacts: ClassVar[bool] = True
    gated: ClassVar[bool] = True

    def __init__(
        self,
        link_count: int,
        *,
        capacity: int = 64,
        slot_count: int = 16,
        slot_dim: int = 64,
        link_embedding_dim: int = 16,
    ) -> None:
        sizes = {
            "link_count": link_count,
            "capacity": capacity,
            "slot_count": slot_count,
            "slot_dim": slot_dim,
            "link_embedding_dim": link_embedding_dim,
        }
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f"{name} must be at least 1, got {size}")
        self.link_count = link_count
        self.capacity = capacity
        self.slot_count = slot_count
        self.slot_dim = slot_dim
        self.link_embedding_dim = link_embedding_dim
        self.code_count = slot_count * slot_dim + 1
        self.signals = {
            "contact_rows": SignalSpec("contact_rows", (capacity, ROW_WIDTH), SignalRole.CONTACT),
            "contact_links": SignalSpec("contact_links", (capacity,), SignalRole.CONTACT, dtype="int32"),
            "contact_count": SignalSpec("contact_count", (), SignalRole.CONTACT, dtype="int32"),
        }

    @classmethod
    def from_system(cls, system: SystemArticulation, **sizes: int) -> Self:
        """Create a :class:`ContactInputSet` sized to ``len(system.spec.body_labels)``."""
        return cls(len(system.spec.body_labels), **sizes)

    def config(self) -> dict[str, Any]:
        """Return constructor arguments as JSON-serializable plain data."""
        return {
            "link_count": self.link_count,
            "capacity": self.capacity,
            "slot_count": self.slot_count,
            "slot_dim": self.slot_dim,
            "link_embedding_dim": self.link_embedding_dim,
        }

    def sensor(self, model: newton.Model, binding: BindingArticulation) -> ContactSensorSet:
        """Create a :class:`ContactSensorSet` for ``binding`` in ``model``."""
        return ContactSensorSet(self, model, binding)

    def encoder(self) -> ContactEncoderSet:
        """Create a :class:`ContactEncoderSet` for this configuration."""
        return ContactEncoderSet(self)


class ContactSensorSet(ContactSensorBase):
    """On-device Warp sensor extracting :class:`ContactInputSet` rows from :class:`newton.Contacts`.

    Does not invoke collision detection; the caller runs :class:`newton.CollisionPipeline` and passes ``contacts``
    to :meth:`update`. On GPU, atomic slot allocation means row order (and which rows overflow when exceeding
    ``capacity``) is non-deterministic across runs.

    Attributes:
        overflow: Per-world count of rows dropped past ``capacity`` ``[world_count]`` int32 (excludes contacts
            dropped by Newton's own collision buffer).
    """

    def __init__(self, contact_input: ContactInputSet, model: newton.Model, binding: BindingArticulation) -> None:
        body_ids = binding.body_ids.numpy()
        world_count, link_count = body_ids.shape
        if link_count != contact_input.link_count:
            raise ValueError(
                f"the robot has {link_count} bodies, the contact set was built for {contact_input.link_count}"
            )
        device = model.device
        shape = (world_count, contact_input.capacity)
        self._rows = wp.zeros((*shape, ROW_WIDTH), dtype=float, device=device)
        self._links = wp.full(shape, -1, dtype=wp.int32, device=device)
        self._count = wp.zeros(world_count, dtype=wp.int32, device=device)
        self.overflow = wp.zeros(world_count, dtype=wp.int32, device=device)
        self._outputs = [self._rows, self._links, self._count, self.overflow]
        self._frames = {
            "contact_rows": wp.to_torch(self._rows, requires_grad=False),
            "contact_links": wp.to_torch(self._links, requires_grad=False),
            "contact_count": wp.to_torch(self._count, requires_grad=False),
        }

        # Lookup tables mapping model body index -> (world, link), or -1 for non-robot bodies.
        body_world = np.full(model.body_count, -1, dtype=np.int32)
        body_link = np.full(model.body_count, -1, dtype=np.int32)
        body_world[body_ids] = np.arange(world_count)[:, None]
        body_link[body_ids] = np.arange(link_count)
        # Constant kernel inputs (everything except per-step state and contacts).
        self._inputs = [
            model.body_com,
            model.body_flags,
            model.shape_body,
            model.joint_X_p,
            model.joint_child,
            wp.vec3(*binding.system.up.tolist()),
            binding.root_joint_ids,
            binding.system.floating_base,
            wp.array(body_world, dtype=wp.int32, device=device),
            wp.array(body_link, dtype=wp.int32, device=device),
        ]

    def update(self, state: newton.State, contacts: newton.Contacts | None) -> Frames:
        """Populate per-world contact rows from ``state`` and unreduced ``contacts`` (returns buffer views)."""
        if contacts is None:
            raise ValueError(
                "ContactInputSet reads the contacts of the collision detector and was given none. Run collision "
                "detection on the state, with newton.CollisionPipeline(model), and pass its contacts"
            )
        require_unreduced(contacts)
        device = self._rows.device
        wp.launch(_clear_rows_kernel, dim=self._links.shape, outputs=self._outputs, device=device)
        wp.launch(
            _contact_rows_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state.body_q,
                state.body_qd,
                *self._inputs,
            ],
            outputs=self._outputs,
            device=device,
        )
        return dict(self._frames)


class ContactEncoderSet(nn.Module):
    """Permutation-invariant attention-pooling encoder for :class:`ContactInputSet`.

    Normalizes valid rows, concatenates learned link embeddings, projects through ``Linear -> RMSNorm -> SiLU``,
    and pools into ``slot_count`` slots via learned softmax attention weights (in FP32 even under ``autocast``),
    appending ``log(1 + count) / log(1 + capacity)``. Padding rows are masked out and empty sets produce zeros.
    """

    def __init__(self, contact_input: ContactInputSet) -> None:
        super().__init__()
        self.capacity = contact_input.capacity
        self.normalizer = Normalizer(ROW_WIDTH)
        self.link_embedding = nn.Embedding(contact_input.link_count, contact_input.link_embedding_dim)
        # Bias omitted: link_embedding absorbs per-link offset and bias-free matmul is ~2x faster here.
        self.row = nn.Linear(ROW_WIDTH + contact_input.link_embedding_dim, contact_input.slot_dim, bias=False)
        self.norm = nn.RMSNorm(contact_input.slot_dim, eps=1e-6)
        self.score = nn.Linear(contact_input.slot_dim, contact_input.slot_count, bias=False)
        # Orthogonal init diversifies initial slot attention weights across rows.
        nn.init.orthogonal_(self.score.weight)

    def forward(self, frames: Frames) -> Tensor:
        """Encode ``contact_rows``, ``contact_links``, and ``contact_count`` to ``[..., slot_count * slot_dim + 1]``."""
        links = frames["contact_links"]
        valid = (links >= 0).unsqueeze(-1)
        features = self.normalizer.normalize(frames["contact_rows"], limit=FEATURE_LIMIT)
        # Zero out padding before projection so uninitialized or non-finite padding cannot corrupt outputs.
        features = torch.where(valid, features, 0.0)
        encoded = self.row(torch.cat((features, self.link_embedding(links.clamp_min(0))), dim=-1))
        encoded = nn.functional.silu(self.norm(encoded.to(features.dtype)))

        scores = self.score(encoded).to(features.dtype)  # [..., capacity, slot_count]
        # Finite floor prevents NaN softmax/gradients when all rows are masked; zeroed out via torch.where below.
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.where(valid, torch.softmax(scores, dim=-2), 0.0)
        # Keep pooling reduction in full precision under autocast.
        with torch.autocast(encoded.device.type, enabled=False):
            slots = weights.transpose(-1, -2) @ encoded  # [..., slot_count, slot_dim]

        count = torch.log1p(frames["contact_count"].to(features.dtype)) / math.log1p(self.capacity)
        return torch.cat((slots.flatten(-2), count.unsqueeze(-1)), dim=-1)

    def fit_inputs(self, frames: Frames) -> None:
        """Update row normalizer statistics over valid (``contact_links >= 0``) rows."""
        self.normalizer.update(frames["contact_rows"], frames["contact_links"] >= 0)
