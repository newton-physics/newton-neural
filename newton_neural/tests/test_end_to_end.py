# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Record -> Train -> Save/Load -> Deploy tests for probes, contact sets, and free-motion coupling."""

from __future__ import annotations

import itertools
import unittest

import newton
import numpy as np
import torch

from newton_neural.backbones import BackboneFlash, BackboneMLP
from newton_neural.contacts import ContactInputBase, ContactInputProbes, ContactInputSet
from newton_neural.data import (
    SamplerFloatingBase,
    SamplerUniform,
    SimulationNewton,
    TrajectoryDataset,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.models import FlashNeRD, NeRDBase, NeRDStepper
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.tests.helpers import assert_frames_close, frame_of
from newton_neural.tests.unittest_utils import FolderTestCase, add_function_test
from newton_neural.training import LossBase, LossOneStep, create_optimizer

WORLD_COUNT = 64
FRAME_DT = 0.005
BATCH_SIZE = 256
STEP_COUNT = 300
"""Training steps of every run."""


class TestEndToEnd(FolderTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)


def _quadruped_simulation(device: torch.device, *, newton_contacts: bool) -> SimulationNewton:
    """The quadruped on the ground, driven by joint forces and stepped by MuJoCo.

    Args:
        device: Device of the simulation.
        newton_contacts: Whether MuJoCo solves the contacts a Newton collision pipeline detects, which the
            simulation then holds, instead of detecting contacts itself.
    """
    model = fixtures.quadruped_model(WORLD_COUNT, device)
    model.joint_effort_limit.fill_(5.0)  # bounds the sampled joint forces [N*m]
    system = SystemArticulation.from_model(model, "quadruped", control=("joint_f",))
    solver = newton.solvers.SolverMuJoCo(model, njmax=256, use_mujoco_contacts=not newton_contacts)
    pipeline = newton.CollisionPipeline(model) if newton_contacts else None
    return SimulationNewton(
        model, solver, system, articulation="quadruped", frame_dt=FRAME_DT, collision_pipeline=pipeline
    )


def _crouched(simulation: SimulationNewton) -> SamplerFloatingBase:
    """Starting states of the quadruped crouched on its feet, close to upright and slow.

    The legs are folded, front feet forward and hind feet backward with the shanks near vertical.
    """
    device = simulation.device
    low, high = simulation.state_bounds()["joint_q"]
    place = (low[:2] + high[:2]) / 2  # where the Newton model places the base of the first world
    upright = torch.tensor([0.43, 0.0, 0.0, 0.0, 1.0], device=device)  # height [m] and a quaternion (x, y, z, w)
    front = torch.tensor([0.0, -1.05, 1.05], device=device)  # hip abduction, hip flexion, knee [rad]
    reach = torch.tensor([0.1, 0.15, 0.25], device=device)
    return SamplerFloatingBase(
        torch.cat((place, upright, front, front, -front, -front)),
        position=((-0.25, -0.25, -0.03), (0.25, 0.25, 0.03)),
        tilt=0.1,
        linear_speed=0.5,
        angular_speed=0.5,
        joint_reach=reach.repeat(4),
        joint_speed=0.5,
        generator=torch.Generator(device=device).manual_seed(0),
    )


def _saved_beside(test: TestEndToEnd, nerd: FlashNeRD, dataset: TrajectoryDataset) -> FlashNeRD:
    """Save the untrained model a dataset was recorded for into the dataset folder, as a recording script does.

    Returns:
        The model a training script starts from, loaded from that file.
    """
    nerd.save(dataset.path / "flashnerd.pt")
    loaded = FlashNeRD.load(dataset.path / "flashnerd.pt", device=nerd.device)
    test.assertFalse(loaded.fitted)
    test.assertIsNone(loaded.frame_dt)
    test.assertEqual(loaded.signals, TrajectoryDataset(dataset.path).signals)
    return loaded


def _recorded(
    test: TestEndToEnd, simulation: SimulationNewton, contact_input: ContactInputBase
) -> tuple[FlashNeRD, TrajectoryDataset]:
    """A small model of the quadruped with ``contact_input``, and 96 frames per world recorded for it.

    The model comes back as a training script gets it, untrained from its file in the dataset folder.
    """
    backbone = BackboneFlash(head_count=2, head_dim=16, ffn_dim=64, spans=(2, 2))
    nerd = FlashNeRD(simulation.binding.system, backbone=backbone, contact_input=contact_input, readout_hidden_dim=64)
    nerd.to(simulation.device)
    recorder = TrajectoryRecorder(simulation, nerd, states=_crouched(simulation), episode_frames=(32, 64))
    dataset = recorder.record(test.folder / "data", 96)
    test.assertEqual(dataset.signals, nerd.signals)
    test.assertTrue(dataset.arrays(0)["valid"].all())
    return _saved_beside(test, nerd, dataset), dataset


def _train(nerd: NeRDBase, windows: TrajectoryWindows, loss: LossBase) -> tuple[float, float]:
    """Fit the normalization of ``nerd`` and train it for :data:`STEP_COUNT` steps on four fifths of the worlds.

    Returns:
        The one step error without noise on the windows of the other worlds, before and after training.
    """
    train, held_out = windows.split(0.2)
    nerd.fit_normalization(train.loader(BATCH_SIZE))
    optimizer = create_optimizer(nerd, lr=3e-3)
    unseen = held_out.batch(torch.arange(len(held_out), device=windows.device))
    error = LossOneStep()
    with torch.no_grad():
        before = error(nerd, unseen).item()
    for batch in itertools.islice(train.loader(BATCH_SIZE, epochs=None), STEP_COUNT):
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()
    with torch.no_grad():
        return before, error(nerd, unseen).item()


def _assert_deploys(test: TestEndToEnd, nerd: NeRDBase, windows: TrajectoryWindows) -> None:
    """Save and load ``nerd`` and expect a stepper of the loaded model to step along recorded windows as it does."""
    nerd.save(test.folder / "model.pt")
    loaded = NeRDBase.load(test.folder / "model.pt", device=nerd.device)
    stepper = NeRDStepper(loaded, WORLD_COUNT, precision="fp32")  # the arithmetic of the eager model
    history = nerd.history(WORLD_COUNT)
    frames = windows.batch(torch.arange(WORLD_COUNT, device=windows.device)).frames
    with torch.no_grad():
        for index in range(windows.length + 1):
            frame = frame_of(frames, index)
            assert_frames_close(test, stepper.step(frame), nerd.step(frame, history), tolerance=1e-4)


def test_fixed_contact(test: TestEndToEnd, device: torch.device) -> None:
    """Record the quadruped with probes on its feet, train with input noise, and deploy the saved model."""
    simulation = _quadruped_simulation(device, newton_contacts=False)
    contact_input = ContactInputProbes.from_model(simulation.model, simulation.binding.system, bodies="*_SHANK")
    nerd, dataset = _recorded(test, simulation, contact_input)
    # The feet stand on the ground in a good share of the frames and are in the air in the others.
    touching = dataset.arrays(0)["contact_probes"][..., 7].mean()
    test.assertTrue(0.2 < touching < 0.8, f"the probes touch in {touching:.0%} of the frames")

    windows = TrajectoryWindows(dataset, 4, device=device)
    # With noise the probes are evaluated again on every perturbed state.
    before, after = _train(nerd, windows, LossOneStep(noise=0.05))
    test.assertLess(after, 0.6 * before)
    _assert_deploys(test, nerd, windows)


def test_variable_contact(test: TestEndToEnd, device: torch.device) -> None:
    """Record the quadruped with the contacts Newton detects as a set, train, and deploy the saved model."""
    simulation = _quadruped_simulation(device, newton_contacts=True)
    contact_input = ContactInputSet.from_system(simulation.system, slot_count=4, slot_dim=16)
    nerd, dataset = _recorded(test, simulation, contact_input)
    arrays = dataset.arrays(0)
    links, count = arrays["contact_links"], arrays["contact_count"]
    test.assertEqual((links.dtype, count.dtype), (np.int32, np.int32))
    # Every frame holds as many rows as it counts, each on a link of the robot, and the count changes as the
    # robot moves on the ground.
    np.testing.assert_array_equal((links >= 0).sum(axis=-1), count)
    test.assertLess(links.max(), contact_input.link_count)
    test.assertGreater(count.min(), 0)
    test.assertLess(count.max(), contact_input.capacity)
    test.assertGreater(count.max(), count.min())

    windows = TrajectoryWindows(dataset, 4, device=device)
    test.assertEqual(windows.batch(torch.arange(2, device=device)).frames["contact_links"].dtype, torch.int32)
    before, after = _train(nerd, windows, LossOneStep())
    test.assertLess(after, 0.6 * before)
    _assert_deploys(test, nerd, windows)


def test_free_motion(test: TestEndToEnd, device: torch.device) -> None:
    """Record independent samples of the arm's free acceleration and train a single frame model on them."""
    model = fixtures.arm_model(WORLD_COUNT, device)
    model.joint_effort_limit.fill_(2.0)  # bounds the sampled joint forces [N or N*m]
    system = SystemArticulation.from_model(model, "arm", control=("joint_f",))
    solver = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, solver, system, articulation="arm", frame_dt=FRAME_DT)
    backbone = BackboneMLP(width=64, depth=2, window=1)
    nerd = FlashNeRD(system, backbone=backbone, prediction=PredictionFreeMotion(), readout_hidden_dim=64).to(device)
    test.assertEqual(nerd.receptive_field, 1)

    # Every frame starts from a newly sampled state within a radian (or meter) and a radian per second of rest.
    one = torch.ones(system.joint_dof_count, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    states = SamplerUniform({"joint_q": (-one, one), "joint_qd": (-one, one)}, generator=generator)
    recorder = TrajectoryRecorder(simulation, nerd, states=states, episode_frames=(1, 1))
    dataset = recorder.record(test.folder / "data", 128)
    test.assertEqual(dataset.signals, nerd.signals)
    test.assertTrue(dataset.arrays(0)["episode_start"].all())

    windows = TrajectoryWindows(dataset, 0, device=device)
    test.assertEqual(len(windows), WORLD_COUNT * 128)
    before, after = _train(_saved_beside(test, nerd, dataset), windows, LossOneStep())
    test.assertLess(after, 0.5 * before)


for _function in (test_fixed_contact, test_variable_contact, test_free_motion):
    add_function_test(TestEndToEnd, _function.__name__, _function)


if __name__ == "__main__":
    unittest.main()
