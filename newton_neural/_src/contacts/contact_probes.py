# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphere contact probes (:class:`ContactInputProbes`) sensed analytically against a single ground plane.

Requires no collision detection and is fully recomputable via forward kinematics during noisy training and open-loop
rollouts. Live sensing runs in a single CUDA-graph-safe Warp kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fnmatch import fnmatchcase
from typing import Any, ClassVar, Self

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor, nn

from ..core import rotation as rot
from ..core.normalizer import FEATURE_LIMIT, Normalizer
from ..core.signal import Frames, SignalRole, SignalSpec
from ..core.stream import torch_on_warp_stream
from ..systems.articulation import SystemArticulation, robot_frame
from ..systems.articulation_binding import BindingArticulation, Pattern
from .contact_input import ContactInputBase, ContactSensorBase
from .kinematics import point_velocity

SIGNAL = "contact_probes"
"""Signal name storing probe features."""

FEATURE_COUNT = 8
"""Features per probe: ``[gap (1), position (3), velocity (3), touching (1)]``."""


@wp.kernel(enable_backward=False)
def _probe_features_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_X_p: wp.array[wp.transform],
    joint_child: wp.array[int],
    up: wp.vec3,
    body_ids: wp.array2d[int],
    root_joint_ids: wp.array[int],
    floating_base: wp.bool,
    probe_column: wp.array[int],
    probe_point: wp.array[wp.vec3],
    probe_radius: wp.array[float],
    ground_normal: wp.vec3,
    ground_offset: float,
    touch_distance: float,
    features: wp.array3d[float],
):
    world, probe = wp.tid()
    # Anchor transform (robot placement; identity for floating bases).
    anchor = joint_X_p[root_joint_ids[world]]
    frame = robot_frame(world, body_q, joint_X_p, joint_child, up, root_joint_ids, floating_base)

    body = body_ids[world, probe_column[probe]]
    point = wp.transform_point(body_q[body], probe_point[probe])
    velocity = point_velocity(body, point, body_q, body_qd, body_com)

    height = wp.dot(ground_normal, wp.transform_point(wp.transform_inverse(anchor), point)) - ground_offset
    gap = height - probe_radius[probe]
    to_frame = wp.transform_inverse(frame)
    position = wp.transform_point(to_frame, point)
    velocity = wp.transform_vector(to_frame, velocity)

    features[world, probe, 0] = gap
    for axis in range(3):
        features[world, probe, 1 + axis] = position[axis]
        features[world, probe, 4 + axis] = velocity[axis]
    features[world, probe, 7] = wp.where(gap < touch_distance, 1.0, 0.0)


def _require_keys(what: str, entry: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Validate that ``entry`` contains all required ``keys``."""
    missing = [key for key in keys if key not in entry]
    if missing:
        raise ValueError(f"{what} needs the keys {list(keys)} and lacks {missing}. It holds {list(entry)}")


class ContactInputProbes(ContactInputBase):
    """Fixed sphere probes on robot bodies sensed analytically against a single ground plane.

    Suited to known contact locations on flat ground (e.g., quadruped feet); ignores all other geometry.

    **Frames:** ``ground`` is expressed in the robot's *anchor frame* (root joint parent transform, identity for a
    floating base). Probe positions and velocities are expressed in the *robot frame* (heading frame for a floating
    base, anchor frame for a fixed base), keeping features invariant to horizontal translation and yaw on flat
    ground.

    **Signal:** ``contact_probes`` ``[probe_count, 8]`` float32 per world:

    - ``[0]`` ``gap`` [m]: signed distance from sphere surface to ground plane (negative = penetration),
    - ``[1:4]`` sphere center position [m] in the robot frame,
    - ``[4:7]`` world velocity of the sphere center [m/s] expressed in robot-frame axes,
    - ``[7]`` ``touching``: ``1.0`` if ``gap < touch_distance``, else ``0.0``.

    **Code:** Per-channel normalized features clamped to ``[-10, 10]`` and flattened to ``[probe_count * 8]``.
    Has no trainable weights, so ``gated = False``.

    Args:
        probes: Sequence of ``{"body": label, "point": [x, y, z], "radius": r}`` dicts (``body`` is the leaf label;
            ``point`` [m] is in body frame; ``radius`` [m]).
        ground: Ground plane ``{"normal": [x, y, z], "offset": d}`` in the anchor frame (unit ``normal`` pointing
            away from ground; plane satisfies ``normal . p = offset`` [m]).
        touch_distance: Gap threshold [m] below which ``touching == 1.0``.

    Raises:
        ValueError: If ``probes`` is empty, keys are missing, or ``ground["normal"]`` is not unit length.
    """

    recomputable: ClassVar[bool] = True
    reads_contacts: ClassVar[bool] = False
    gated: ClassVar[bool] = False

    def __init__(
        self,
        probes: Sequence[dict[str, Any]],
        *,
        ground: dict[str, Any],
        touch_distance: float = 0.02,
    ) -> None:
        for index, probe in enumerate(probes):
            _require_keys(f"probe {index}", probe, ("body", "point", "radius"))
        _require_keys("ground", ground, ("normal", "offset"))
        self.probes = [
            {"body": str(probe["body"]), "point": [float(x) for x in probe["point"]], "radius": float(probe["radius"])}
            for probe in probes
        ]
        self.ground = {"normal": [float(x) for x in ground["normal"]], "offset": float(ground["offset"])}
        self.touch_distance = float(touch_distance)
        if not self.probes:
            raise ValueError("a probe contact input needs at least one probe")
        if not math.isclose(math.hypot(*self.ground["normal"]), 1.0, abs_tol=1e-4):
            raise ValueError(f"the ground normal {self.ground['normal']} is not a unit vector")

        probe_count = len(self.probes)
        self.signals = {
            SIGNAL: SignalSpec(SIGNAL, (probe_count, FEATURE_COUNT), SignalRole.CONTACT, unit="m and m/s"),
        }
        self.code_count = probe_count * FEATURE_COUNT
        # Cached kinematic replicas keyed by (world_count, device) for evaluate().
        self._replicas: dict[
            tuple[int, torch.device], tuple[newton.Model, newton.State, BindingArticulation, ContactSensorProbes]
        ] = {}

    @classmethod
    def from_model(
        cls,
        model: newton.Model,
        system: SystemArticulation,
        *,
        bodies: str | Sequence[str],
        radius: float | None = None,
        touch_distance: float = 0.02,
        articulation: Pattern = "*",
    ) -> Self:
        """Create probes on bodies matching ``bodies`` and extract the static ground plane from ``model``.

        If a matched body has exactly one collision sphere, its center and radius are copied; otherwise a probe of
        size ``radius`` is placed at the body origin.

        Args:
            model: Newton model containing the robot and exactly one static plane shape.
            system: Robot system specification.
            bodies: One or more :mod:`fnmatch` patterns matched against ``system.spec.body_labels``.
            radius: Fallback probe radius [m] for matched bodies without a single collision sphere.
            touch_distance: Gap threshold [m] for ``touching``.
            articulation: Label pattern of the robot's articulation in ``model``.

        Raises:
            ValueError: If any pattern matches no body, ``radius`` is missing when needed, ``model`` does not
                have exactly one static plane shape, or the plane lies elsewhere for the robot of one world than
                for that of another.
        """
        binding = system.bind(model, articulation)
        labels = system.spec.body_labels
        patterns = [bodies] if isinstance(bodies, str) else list(bodies)
        unmatched = [pattern for pattern in patterns if not any(fnmatchcase(label, pattern) for label in labels)]
        if unmatched:
            raise ValueError(f"no body matches {unmatched}. The system's bodies are {list(labels)}")

        shape_body, shape_type = model.shape_body.numpy(), model.shape_type.numpy()
        shape_transform, shape_scale = model.shape_transform.numpy(), model.shape_scale.numpy()
        collides = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        spheres = collides & (shape_type == int(newton.GeoType.SPHERE))
        probes = []
        # Inspect world 0 (binding already validated that all worlds share the same robot).
        for label, body in zip(labels, binding.body_ids.numpy()[0], strict=True):
            if not any(fnmatchcase(label, pattern) for pattern in patterns):
                continue
            shapes = np.flatnonzero(spheres & (shape_body == body))
            if len(shapes) == 1:
                point, probe_radius = shape_transform[shapes[0], :3], shape_scale[shapes[0], 0]
            elif radius is not None:
                point, probe_radius = (0.0, 0.0, 0.0), radius
            else:
                raise ValueError(
                    f"body {label!r} has {len(shapes)} collision spheres. A probe on a body without a single "
                    "collision sphere sits at the body origin and needs a radius"
                )
            probes.append({"body": label, "point": point, "radius": probe_radius})

        planes = np.flatnonzero((shape_body == -1) & (shape_type == int(newton.GeoType.PLANE)))
        if len(planes) != 1:
            raise ValueError(
                f"the Newton model holds {len(planes)} static plane shapes. Probes sense exactly one ground plane. "
                "For any other scene, give the plane to the constructor"
            )
        # The plane in the frame every world's robot is mounted in, which is where the probes sense it.
        mounts = torch.as_tensor(model.joint_X_p.numpy()[binding.root_joint_ids.numpy()])
        plane = torch.as_tensor(shape_transform[planes[0]])
        up = rot.rotate(plane[3:], torch.tensor([0.0, 0.0, 1.0]))  # Newton planes face +Z locally
        normals = rot.rotate_inverse(mounts[:, 3:], up)
        offsets = (normals * rot.rotate_inverse(mounts[:, 3:], plane[:3] - mounts[:, :3])).sum(dim=-1)
        elsewhere = ((normals - normals[0]).abs().amax(dim=-1) > 1e-4) | ((offsets - offsets[0]).abs() > 1e-4)
        if elsewhere.any():
            world = int(elsewhere.nonzero()[0])
            raise ValueError(
                f"the ground plane lies {offsets[world].item():.4g} m along {normals[world].tolist()} from the mount "
                f"of the robot in world {world}, and {offsets[0].item():.4g} m along {normals[0].tolist()} from that "
                "in world 0. Probes sense one plane, fixed in the frame the robot is mounted in, so mount the robot "
                "of every world at one height and tilt over the plane"
            )
        ground = {"normal": normals[0].tolist(), "offset": offsets[0].item()}
        return cls(probes, ground=ground, touch_distance=touch_distance)

    def config(self) -> dict[str, Any]:
        """Return ``probes``, ``ground``, and ``touch_distance`` as JSON-serializable plain data."""
        return {
            "probes": [{**probe, "point": list(probe["point"])} for probe in self.probes],
            "ground": {**self.ground, "normal": list(self.ground["normal"])},
            "touch_distance": self.touch_distance,
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # Exclude cached Warp kinematic replicas (not deepcopyable); rebuilt lazily on demand.
        return self.from_config(self.config())

    def sensor(self, model: newton.Model, binding: BindingArticulation) -> ContactSensorProbes:
        """Create a :class:`ContactSensorProbes` for ``binding`` in ``model``."""
        return ContactSensorProbes(self, model, binding)

    @torch.no_grad()
    def evaluate(self, system: SystemArticulation, frames: Frames) -> Frames:
        """Recompute ``contact_probes`` ``[..., probe_count, 8]`` from ``joint_q`` and ``joint_qd`` via FK.

        Caches a minimal kinematic Newton replica per ``(world_count, device)``. Non-differentiable.
        """
        joint_q = frames["joint_q"]
        lead = joint_q.shape[:-1]
        world_count, device = math.prod(lead), joint_q.device
        if (world_count, device) not in self._replicas:
            model = system.kinematic_model(world_count, device=device)
            binding = system.bind(model)
            self._replicas[world_count, device] = (model, model.state(), binding, self.sensor(model, binding))
        model, state, binding, sensor = self._replicas[world_count, device]
        # A copy, because sensor.features is overwritten by subsequent calls.
        features = joint_q.new_empty(*lead, len(self.probes), FEATURE_COUNT)
        # The state copy, the kinematics and the result copy run on one CUDA stream, in this order.
        with torch_on_warp_stream(device):
            binding.write_state(state, {name: frames[name].flatten(0, -2) for name in ("joint_q", "joint_qd")})
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            features.copy_(sensor.update(state, None)[SIGNAL].unflatten(0, lead))
        return {SIGNAL: features}

    def encoder(self) -> ContactEncoderProbes:
        """Create a :class:`ContactEncoderProbes` for this configuration."""
        return ContactEncoderProbes(self)


class ContactSensorProbes(ContactSensorBase):
    """On-device Warp sensor evaluating :class:`ContactInputProbes` in one kernel launch (CUDA-graph safe).

    Args:
        contact_input: Probe and ground-plane configuration.
        model: Target Newton model.
        binding: Robot articulation binding in ``model``.

    Attributes:
        features: Preallocated output buffer ``[world_count, probe_count, 8]``.
    """

    def __init__(self, contact_input: ContactInputProbes, model: newton.Model, binding: BindingArticulation) -> None:
        probes = contact_input.probes
        labels = binding.system.spec.body_labels  # Matches binding.body_ids column order
        unknown = [probe["body"] for probe in probes if probe["body"] not in labels]
        if unknown:
            raise ValueError(
                f"the probes name the bodies {unknown}, which the robot does not have. It has {list(labels)}"
            )
        device = model.device

        self.features = wp.zeros((binding.world_count, len(probes), FEATURE_COUNT), dtype=float, device=device)
        self._view = wp.to_torch(self.features, requires_grad=False)
        # Constant kernel inputs (everything except per-step state.body_q and state.body_qd).
        self._inputs = [
            model.body_com,
            model.joint_X_p,
            model.joint_child,
            wp.vec3(*binding.system.up.tolist()),
            binding.body_ids,
            binding.root_joint_ids,
            binding.system.floating_base,
            wp.array([labels.index(probe["body"]) for probe in probes], dtype=int, device=device),
            wp.array([probe["point"] for probe in probes], dtype=wp.vec3, device=device),
            wp.array([probe["radius"] for probe in probes], dtype=float, device=device),
            wp.vec3(*contact_input.ground["normal"]),
            contact_input.ground["offset"],
            contact_input.touch_distance,
        ]

    def update(self, state: newton.State, contacts: newton.Contacts | None) -> Frames:
        """Launch the probe kernel on ``state`` and return ``{"contact_probes": view}`` (ignores ``contacts``)."""
        wp.launch(
            _probe_features_kernel,
            dim=self.features.shape[:2],
            inputs=[state.body_q, state.body_qd, *self._inputs],
            outputs=[self.features],
            device=self.features.device,
        )
        return {SIGNAL: self._view}


class ContactEncoderProbes(nn.Module):
    """Normalizes probe features per channel, clamps to ``[-10, 10]``, and flattens to ``[probe_count * 8]``."""

    def __init__(self, contact_input: ContactInputProbes) -> None:
        super().__init__()
        self.normalizer = Normalizer(contact_input.code_count)

    def forward(self, frames: Frames) -> Tensor:
        """Encode ``frames["contact_probes"]`` ``[..., probe_count, 8]`` into ``[..., probe_count * 8]``."""
        return self.normalizer.normalize(frames[SIGNAL].flatten(-2), limit=FEATURE_LIMIT)

    def fit_inputs(self, frames: Frames) -> None:
        """Update running normalization statistics from ``frames["contact_probes"]``."""
        self.normalizer.update(frames[SIGNAL].flatten(-2))
