# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared cartpole recording fixture used in ``.. testsetup::`` blocks across Sphinx guide pages."""

import newton
import newton.examples
import torch
from torch import Tensor

from newton_neural.data import CommandHeld, SamplerUniform, SimulationNewton, TrajectoryDataset, TrajectoryRecorder
from newton_neural.models import FlashNeRD
from newton_neural.systems import SystemArticulation


def cartpole_recording(
    frame_count: int,
) -> tuple[newton.Model, SystemArticulation, SimulationNewton, FlashNeRD, Tensor, TrajectoryDataset]:
    """Record a 32-world cartpole dataset into ``data/cartpole`` and save an untrained ``flashnerd.pt`` inside it."""
    robot = newton.ModelBuilder()
    robot.default_joint_cfg.target_ke = 100.0
    robot.default_joint_cfg.target_kd = 10.0
    robot.add_urdf(newton.examples.get_asset("cartpole.urdf"), floating=False, enable_self_collisions=False)
    scene = newton.ModelBuilder()
    scene.replicate(robot, 32)
    model = scene.finalize()
    system = SystemArticulation.from_model(model)
    simulation = SimulationNewton(model, newton.solvers.SolverMuJoCo(model), system, frame_dt=0.02, substep_count=4)
    nerd = FlashNeRD(system).to(simulation.device)
    reach = torch.tensor([1.0, 1.5, 1.5], device=simulation.device)
    recorder = TrajectoryRecorder(
        simulation,
        nerd,
        states=SamplerUniform({"joint_q": (-reach, reach), "joint_qd": (-reach, reach)}),
        commands=CommandHeld((10, 50), low=-reach, high=reach),
        episode_frames=(100, 200),
    )
    dataset = recorder.record("data/cartpole", frame_count=frame_count)
    nerd.save(dataset.path / "flashnerd.pt")
    return model, system, simulation, nerd, reach, dataset
