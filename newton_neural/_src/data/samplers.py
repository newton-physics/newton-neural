# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Initial-state samplers for task-agnostic episode resets during trajectory recording."""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence

import torch
from torch import Tensor

from ..core import rotation
from ..core.signal import Frames

Bounds = dict[str, tuple[Tensor, Tensor]]
"""Per-signal ``(low, high)`` bounds, each shaped ``[*shape]`` for one world."""


class SamplerBase(abc.ABC):
    """Abstract sampler generating initial episode states."""

    def start(self, bounds: Bounds) -> None:  # noqa: B027
        """Receive default state bounds from the simulation before sampling begins."""

    @abc.abstractmethod
    def sample(self, count: int) -> Frames:
        """Draw ``count`` initial states as ``[count, *shape]`` tensors per ``STATE`` signal."""


class _SamplerBox(SamplerBase):
    """Base sampler over per-signal box bounds (falls back to simulation bounds if omitted)."""

    def __init__(self, bounds: Bounds | None = None) -> None:
        self.bounds = bounds

    def start(self, bounds: Bounds) -> None:
        if self.bounds is None:
            self.bounds = bounds

    def _box(self) -> Bounds:
        if self.bounds is None:
            raise RuntimeError(f"{type(self).__name__} has no bounds. Pass bounds or start it with a simulation's")
        return self.bounds


class SamplerUniform(_SamplerBox):
    """Independent uniform sampling per channel within box bounds.

    Args:
        bounds: Optional per-signal ``(low, high)`` bounds (defaults to simulation state bounds).
        generator: Optional PyTorch random generator on the bounds device.
    """

    def __init__(self, bounds: Bounds | None = None, *, generator: torch.Generator | None = None) -> None:
        super().__init__(bounds)
        self.generator = generator

    def sample(self, count: int) -> Frames:
        out = {}
        for name, (low, high) in self._box().items():
            unit = torch.rand(count, *low.shape, device=low.device, generator=self.generator)
            out[name] = low + (high - low) * unit
        return out


class SamplerSobol(_SamplerBox):
    """Low-discrepancy scrambled Sobol sequence over box bounds (drawn in host blocks and moved to device).

    Args:
        bounds: Optional per-signal ``(low, high)`` bounds (defaults to simulation state bounds).
        seed: Scrambling seed.
        block_size: Number of points generated per CPU batch.
    """

    def __init__(self, bounds: Bounds | None = None, *, seed: int = 0, block_size: int = 1 << 16) -> None:
        super().__init__(bounds)
        self.seed, self.block_size = seed, block_size
        self._engine: torch.quasirandom.SobolEngine | None = None

    def sample(self, count: int) -> Frames:
        bounds = self._box()
        sizes = [low.numel() for low, _ in bounds.values()]
        if self._engine is None:
            self._engine = torch.quasirandom.SobolEngine(sum(sizes), scramble=True, seed=self.seed)
            self._points = torch.empty(0, sum(sizes), device=next(iter(bounds.values()))[0].device)
        if len(self._points) < count:
            fresh = self._engine.draw(max(self.block_size, count - len(self._points)))
            self._points = torch.cat((self._points, fresh.to(self._points.device)))
        points, self._points = self._points[:count], self._points[count:]
        out = {}
        for (name, (low, high)), unit in zip(bounds.items(), points.split(sizes, dim=1), strict=True):
            out[name] = low + (high - low) * unit.view(count, *low.shape)
        return out


class SamplerFloatingBase(SamplerBase):
    """Samples floating-base robot states around a nominal ``pose`` using intuitive physical perturbations.

    Draws base position uniformly inside ``position``, base tilt uniformly over a horizontal disk of radius
    ``tilt`` (world Z up, preserving heading), joint coordinates uniformly within ``+-joint_reach``, and base/joint
    velocities uniformly within their speed bounds.

    Args:
        pose: Nominal single-world ``joint_q`` ``[joint_coord_count]`` (``[pos(3), quat_xyzw(4), joints...]``).
        position: ``(low, high)`` base translation offsets from ``pose`` [m].
        tilt: Max tilt angle [rad] in ``[0, pi]`` about a horizontal world axis.
        linear_speed: Per-axis base linear velocity bound [m/s].
        angular_speed: Per-axis base angular velocity bound [rad/s].
        joint_reach: Actuated joint offset bound [m or rad], scalar or ``[joint_count]``.
        joint_speed: Actuated joint velocity bound [m/s or rad/s], scalar or ``[joint_count]``.
        generator: Optional random generator on ``pose.device``.
    """

    def __init__(
        self,
        pose: Tensor,
        *,
        position: tuple[Sequence[float], Sequence[float]] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        tilt: float = 0.0,
        linear_speed: float = 0.0,
        angular_speed: float = 0.0,
        joint_reach: Tensor | Sequence[float] | float = 0.0,
        joint_speed: Tensor | Sequence[float] | float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        if pose.dim() != 1 or len(pose) < 7:
            raise ValueError(
                f"pose must be joint_q of one world, the seven coordinates of the base and the joints, got "
                f"{list(pose.shape)}. Pass one row, such as joint_q[0]"
            )
        self.pose = pose
        self.position = torch.as_tensor(position, dtype=pose.dtype, device=pose.device)
        if self.position.shape != (2, 3):
            raise ValueError(
                f"position must be the lowest and the highest offset along x, y and z, got "
                f"{list(self.position.shape)}. Pass two triples, such as ((0, 0, 0), (0, 0, 0.1))"
            )
        self.tilt = tilt
        base_speed = torch.tensor([linear_speed] * 3 + [angular_speed] * 3, dtype=pose.dtype, device=pose.device)
        self.joint_reach = self._per_joint("joint_reach", joint_reach)
        self.speed = torch.cat((base_speed, self._per_joint("joint_speed", joint_speed)))
        self.generator = generator

    def _per_joint(self, name: str, bound: Tensor | Sequence[float] | float) -> Tensor:
        """Broadcast a scalar or per-joint bound to ``[joint_count]`` on ``pose.device``."""
        joint_count = len(self.pose) - 7
        bound = torch.as_tensor(bound, dtype=self.pose.dtype, device=self.pose.device)
        if bound.shape not in ((), (joint_count,)):
            raise ValueError(
                f"{name} must be one number or one value for each of the {joint_count} joints of the pose, got "
                f"{list(bound.shape)}"
            )
        return bound.expand(joint_count)

    def sample(self, count: int) -> Frames:
        pose = self.pose
        low, high = self.position
        position = pose[:3] + low + (high - low) * self._rand(count, 3)
        # sqrt(u) yields uniform area density over the disk of radius tilt.
        angle = self.tilt * self._rand(count, 1).sqrt()
        direction = 2.0 * math.pi * self._rand(count, 1)
        turn = angle * torch.cat((direction.cos(), direction.sin(), torch.zeros_like(direction)), dim=1)
        # Apply horizontal tilt in world frame on top of nominal pose orientation.
        orientation = rotation.normalize(rotation.multiply(rotation.from_rotation_vector(turn), pose[3:7]))
        joints = pose[7:] + self.joint_reach * self._centered(count, len(pose) - 7)
        return {
            "joint_q": torch.cat((position, orientation, joints), dim=1),
            "joint_qd": self.speed * self._centered(count, len(self.speed)),
        }

    def _rand(self, *shape: int) -> Tensor:
        return torch.rand(*shape, dtype=self.pose.dtype, device=self.pose.device, generator=self.generator)

    def _centered(self, *shape: int) -> Tensor:
        """Uniform samples in ``[-1, 1]``."""
        return 2.0 * self._rand(*shape) - 1.0


class SamplerPerturb(SamplerBase):
    """Wraps another sampler and adds per-signal zero-mean Gaussian noise.

    Args:
        sampler: Base state sampler.
        scale: Noise standard deviation per signal name in signal units (omitted signals are unchanged).
        generator: Optional random generator on the sampled tensor device.
    """

    def __init__(
        self, sampler: SamplerBase, scale: dict[str, float], *, generator: torch.Generator | None = None
    ) -> None:
        self.sampler, self.scale, self.generator = sampler, scale, generator

    def start(self, bounds: Bounds) -> None:
        self.sampler.start(bounds)

    def sample(self, count: int) -> Frames:
        out = self.sampler.sample(count)
        for name, scale in self.scale.items():
            value = out[name]
            out[name] = value + scale * torch.randn(value.shape, device=value.device, generator=self.generator)
        return out


class SamplerMixture(SamplerBase):
    """Draws each state from one of several samplers chosen according to ``weights``.

    Component selection runs on CPU to avoid device-to-host syncs when slicing sub-batches.

    Args:
        samplers: Mapping from sampler instance to positive relative weight (all samplers must emit identical keys).
        generator: Optional CPU random generator for component selection.
    """

    def __init__(self, samplers: dict[SamplerBase, float], *, generator: torch.Generator | None = None) -> None:
        self.samplers = list(samplers)
        self.weights = torch.tensor(list(samplers.values()), dtype=torch.float64)
        self.generator = generator

    def start(self, bounds: Bounds) -> None:
        for sampler in self.samplers:
            sampler.start(bounds)

    def sample(self, count: int) -> Frames:
        choice = torch.multinomial(self.weights, count, replacement=True, generator=self.generator)
        out: Frames = {}
        for i, sampler in enumerate(self.samplers):
            rows = torch.nonzero(choice == i).flatten()
            if len(rows) == 0:
                continue
            part = sampler.sample(len(rows))
            if out and part.keys() != out.keys():
                raise ValueError(
                    f"the samplers of a mixture must return the same signals, got {sorted(part)} and {sorted(out)}"
                )
            for name, value in part.items():
                if name not in out:
                    out[name] = value.new_empty(count, *value.shape[1:])
                out[name][rows.to(value.device)] = value
        return out
