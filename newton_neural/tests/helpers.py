# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared factories for small test models, synthetic frames/batches, simulations, and assertions."""

from __future__ import annotations

import contextlib
import functools
import unittest
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import newton
import torch
from torch import Tensor

from newton_neural import Frames
from newton_neural.backbones import BackboneBase, BackboneFlash, BackboneMLP, BackboneTransformer
from newton_neural.contacts import ContactInputBase, ContactInputProbes, ContactInputSet
from newton_neural.data import SamplerUniform, SimulationNewton, TrajectoryBatch
from newton_neural.models import FlashNeRD, NeRDBase
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.tests.extensions import MinimalNeRD
from newton_neural.training import LossOneStep, create_optimizer

WORLD_COUNT = 4
FRAME_DT = 0.02
SUBSTEP_COUNT = 4
"""Solver substeps per frame for :func:`make_arm_simulation`."""


@functools.cache
def _spec(robot: str) -> SystemArticulation.Spec:
    model = getattr(fixtures, f"{robot}_model")(1, "cpu")
    return SystemArticulation.from_model(model, robot).spec


def make_system(robot: str) -> SystemArticulation:
    """Build a fresh :class:`SystemArticulation` for fixture ``"quadruped"`` or ``"arm"``."""
    return SystemArticulation(_spec(robot))


def make_backbone() -> BackboneFlash:
    """Build a tiny ``BackboneFlash`` with receptive field 3."""
    return BackboneFlash(head_count=2, head_dim=8, ffn_dim=16, spans=(2, 2))


def make_frames(
    nerd: NeRDBase, world_count: int, frame_count: int | None = None, *, device: Any = "cpu", seed: int = 0
) -> Frames:
    """Generate valid random frames covering all signals declared by ``nerd``."""
    frames = fixtures.random_frames(nerd.system, world_count, frame_count, device=device, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    lead = frames["joint_q"].shape[:-1]
    contact_input = nerd.contact_input
    if isinstance(contact_input, ContactInputSet):
        capacity = contact_input.capacity
        count = torch.randint(capacity + 1, lead, generator=generator, dtype=torch.int32)
        padding = torch.arange(capacity) >= count[..., None]
        links = torch.randint(contact_input.link_count, (*lead, capacity), generator=generator, dtype=torch.int32)
        rows = torch.randn(*lead, *contact_input.signals["contact_rows"].shape, generator=generator)
        frames["contact_rows"] = rows.masked_fill(padding[..., None], 0.0).to(device)
        frames["contact_links"] = links.masked_fill(padding, -1).to(device)
        frames["contact_count"] = count.to(device)
    elif contact_input is not None:
        frames.update(contact_input.evaluate(nerd.system, frames))
    for name, spec in nerd.prediction.signals(nerd.system).items():
        frames[name] = torch.randn(*lead, *spec.shape, generator=generator).to(device)
    return frames


def make_batch(nerd: NeRDBase, world_count: int, frame_count: int, *, device: Any = "cpu", seed: int = 0):
    """Build a synthetic :class:`TrajectoryBatch` with ``episode_start=False``."""
    frames = make_frames(nerd, world_count, frame_count, device=device, seed=seed)
    return TrajectoryBatch(frames, torch.zeros(world_count, dtype=torch.bool, device=device), FRAME_DT)


def make_nerd(
    robot: str | SystemArticulation,
    device: Any,
    *,
    family: type[NeRDBase] = FlashNeRD,
    backbone: Callable[[], BackboneBase] = make_backbone,
    fitted_on: tuple[int, int] = (16, 6),
    **parts: Any,
) -> NeRDBase:
    """Build a deterministic, fitted small model with perturbed weights (activating zero-init skip paths)."""
    torch.manual_seed(0)
    system = make_system(robot) if isinstance(robot, str) else robot
    if issubclass(family, FlashNeRD):
        parts = {"backbone": backbone(), "readout_hidden_dim": 16, **parts}
    nerd = family(system, **parts)
    with torch.no_grad():
        for parameter in nerd.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    nerd.fit_normalization([make_batch(nerd, *fitted_on)])
    return nerd.to(device)


CONTACT_INPUTS: dict[str, Callable[[], ContactInputBase | None]] = {
    "no_contact_input": lambda: None,
    "probes": lambda: ContactInputProbes.from_model(
        fixtures.quadruped_model(1, "cpu"), make_system("quadruped"), bodies="*_SHANK"
    ),
    "contact_set": lambda: ContactInputSet.from_system(make_system("quadruped"), capacity=6, slot_count=2, slot_dim=8),
}
"""Quadruped contact-input configurations tested across tools."""

NERDS: dict[str, dict[str, Any]] = {
    "flash": {"backbone": make_backbone},
    "flash_span_one": {"backbone": lambda: BackboneFlash(head_count=2, head_dim=8, ffn_dim=16, spans=(1, 1))},
    "transformer": {"backbone": lambda: BackboneTransformer(width=16, head_count=2, depth=2, span=2, ffn_dim=32)},
    "mlp": {"backbone": lambda: BackboneMLP(width=16, depth=2, window=2)},
    "minimal": {"family": MinimalNeRD},
}
"""Model configurations (kwargs for :func:`make_nerd`) covering all built-in backbones plus ``MinimalNeRD``."""

BACKBONES: dict[str, Callable[[], BackboneBase]] = {
    "flash": lambda: BackboneFlash(head_count=2, head_dim=8, ffn_dim=16, spans=(3, 1, 4)),
    "transformer": lambda: BackboneTransformer(width=16, head_count=4, depth=2, span=3, ffn_dim=32),
    "mlp": lambda: BackboneMLP(width=16, depth=2, window=3),
}
"""Small instances of all three built-in backbones with mixed spans/window lengths."""


def make_arm_simulation(
    device: Any, world_count: int, *, driven: bool = True, **solver_options: Any
) -> SimulationNewton:
    """Create a contact-free Featherstone arm simulation (``FRAME_DT``, ``SUBSTEP_COUNT`` substeps)."""
    model = fixtures.arm_model(world_count, device)
    if driven:
        model.joint_target_ke.fill_(40.0)
        model.joint_target_kd.fill_(4.0)
        model.joint_armature.fill_(0.1)
    system = SystemArticulation.from_model(model, "arm")
    solver = newton.solvers.SolverFeatherstone(model, **solver_options)
    return SimulationNewton(model, solver, system, frame_dt=FRAME_DT, substep_count=SUBSTEP_COUNT)


def near_rest(simulation: SimulationNewton) -> SamplerUniform:
    """Uniform arm state sampler within ``+-0.5`` [m/rad and m/s or rad/s] of zero."""
    half = torch.full((simulation.system.joint_dof_count,), 0.5, device=simulation.device)
    generator = torch.Generator(device=simulation.device).manual_seed(0)
    return SamplerUniform({"joint_q": (-half, half), "joint_qd": (-half, half)}, generator=generator)


def frame_of(frames: Frames, index: int) -> Frames:
    """Slice frame ``index`` from windowed frames ``[world_count, frame_count, ...]``."""
    return {name: value[:, index] for name, value in frames.items()}


def process_settings() -> tuple[str | None, str, str, str]:
    """Snapshot PyTorch's global FP32/TF32 matmul settings (legacy enum, global, CUDA, and MKLDNN flags)."""
    backends = torch.backends
    try:
        name = torch.get_float32_matmul_precision()
    except RuntimeError:
        name = None
    return name, backends.fp32_precision, backends.cuda.matmul.fp32_precision, backends.mkldnn.matmul.fp32_precision


@contextlib.contextmanager
def process_precision(setting: str) -> Iterator[None]:
    """Temporarily apply one of PyTorch's FP32/TF32 matmul configuration modes and restore on exit."""
    backends = torch.backends
    name, *switches = process_settings()
    if setting == "allow_tf32":
        backends.cuda.matmul.allow_tf32 = True
    elif setting == "cuda_tf32":
        backends.cuda.matmul.fp32_precision = "tf32"
    elif setting == "global_tf32":
        backends.fp32_precision = "tf32"
    else:
        torch.set_float32_matmul_precision(setting)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(name)
        backends.fp32_precision, backends.cuda.matmul.fp32_precision, backends.mkldnn.matmul.fp32_precision = switches


@contextlib.contextmanager
def recorded_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Capture all warnings emitted within the block (including repeated warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def assert_frames_close(test: unittest.TestCase, actual: Frames, expected: Frames, tolerance: float = 1e-5) -> None:
    """Assert two frame dicts share identical keys and agree within ``tolerance``."""
    test.assertEqual(actual.keys(), expected.keys())
    for name, value in expected.items():
        error = (actual[name] - value).abs().max().item()
        test.assertLessEqual(error, tolerance, f"{name} differs by {error:.3g}")


def assert_close(actual: Tensor, expected: Tensor, tolerance: float = 1e-9) -> None:
    """Assert ``actual`` matches ``expected`` within ``tolerance * max(1, max(|expected|))``."""
    scale = max(1.0, expected.abs().max().item())
    torch.testing.assert_close(actual, expected, atol=tolerance * scale, rtol=0.0)


def pendulum_windows(system: SystemArticulation, world_count: int, frame_count: int, *, device: Any, seed: int = 0):
    """Synthesize independent damped driven-pendulum trajectories (``qdd = 40(target - q) - 4qd - 5sin(q)``)."""
    generator = torch.Generator().manual_seed(seed)
    count = system.joint_dof_count
    joint_q = 2.0 * torch.rand(world_count, count, generator=generator) - 1.0
    joint_qd = torch.randn(world_count, count, generator=generator)
    target = 2.0 * torch.rand(world_count, count, generator=generator) - 1.0
    frames: dict[str, list[Tensor]] = {"joint_q": [], "joint_qd": [], "joint_target_q": []}
    for _ in range(frame_count):
        frames["joint_q"].append(joint_q)
        frames["joint_qd"].append(joint_qd)
        frames["joint_target_q"].append(target)
        joint_qd = joint_qd + FRAME_DT * (40.0 * (target - joint_q) - 4.0 * joint_qd - 5.0 * torch.sin(joint_q))
        joint_q = joint_q + FRAME_DT * joint_qd
    stacked = {name: torch.stack(values, dim=1).to(device) for name, values in frames.items()}
    return TrajectoryBatch(stacked, torch.ones(world_count, dtype=torch.bool, device=device), FRAME_DT)


def train_on_pendulums(device: torch.device, optimizer_name: str = "muon") -> tuple[float, float]:
    """Train a small FlashNeRD on synthetic pendulum windows for 200 steps and return ``(loss_before, loss_after)``."""
    torch.manual_seed(0)
    system = make_system("arm")
    nerd = FlashNeRD(system, backbone=make_backbone(), readout_hidden_dim=32)
    train = pendulum_windows(system, 128, 8, device="cpu", seed=0)
    nerd.fit_normalization([train])
    nerd.to(device)
    train, held_out = train.to(device), pendulum_windows(system, 256, 8, device=device, seed=1)
    loss, optimizer = LossOneStep(), create_optimizer(nerd, optimizer_name, lr=3e-3)
    with torch.no_grad():
        before = loss(nerd, held_out).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss(nerd, train).backward()
        optimizer.step()
    with torch.no_grad():
        return before, loss(nerd, held_out).item()
