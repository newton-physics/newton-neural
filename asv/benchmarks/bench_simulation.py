# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for full ``SolverNeRD`` steps and standalone contact-sensor kernels.

Uses small CPU world counts when no GPU is available.
"""

import newton
import torch
import warp as wp

wp.config.log_level = wp.LOG_WARNING

from newton_neural.contacts import ContactInputProbes, ContactInputSet
from newton_neural.models import FlashNeRD
from newton_neural.solvers import SolverNeRD
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORLD_COUNTS = (1, 1024, 16384) if DEVICE.type == "cuda" else (1, 16)
PRECISIONS = ("fp32", "tf32", "fp16")
FRAME_DT = 0.02  # Frame duration [s]
ROBOT = "quadruped"  # Articulation label in fixture
FEET = "*_SHANK"  # Bodies carrying foot probe spheres


class SolverNeRDStep:
    """Full ``SolverNeRD`` step (probe sensing, inference, state write, FK; CUDA-graphed on GPU) for a quadruped."""

    params = (WORLD_COUNTS, PRECISIONS)
    param_names = ("world_count", "precision")
    number = 20
    repeat = 5

    def setup(self, world_count, precision):
        model = fixtures.quadruped_model(world_count, DEVICE)
        system = SystemArticulation.from_model(model, ROBOT)
        probes = ContactInputProbes.from_model(model, system, bodies=FEET)
        nerd = FlashNeRD(system, contact_input=probes, frame_dt=FRAME_DT)
        # Zero weights keep the robot stationary across repeats while preserving exact compute cost.
        for parameter in nerd.parameters():
            torch.nn.init.zeros_(parameter)
        self.solver = SolverNeRD(model, nerd, precision=precision)
        self.state, self.control = model.state(), model.control()
        self.solver.reset(self.state)
        for _ in range(3):  # Warm up: compile, validate, and capture CUDA graph
            self.solver.step(self.state, self.state, self.control, None, FRAME_DT)

    def time_step(self, world_count, precision):
        self.solver.step(self.state, self.state, self.control, None, FRAME_DT)
        wp.synchronize_device(self.solver.model.device)


class ContactInputProbesSensing:
    """Single ``ContactInputProbes`` kernel update across world counts."""

    params = (WORLD_COUNTS,)
    param_names = ("world_count",)
    number = 20
    repeat = 5

    def setup(self, world_count):
        self.model = fixtures.quadruped_model(world_count, DEVICE)
        system = SystemArticulation.from_model(self.model, ROBOT)
        probes = ContactInputProbes.from_model(self.model, system, bodies=FEET)
        self.sensor = probes.sensor(self.model, system.bind(self.model, ROBOT))
        self.state = self.model.state()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.sensor.update(self.state, None)  # Preload kernel

    def time_update(self, world_count):
        self.sensor.update(self.state, None)
        wp.synchronize_device(self.model.device)


class ContactInputSetSensing:
    """Single ``ContactInputSet`` kernel update (excluding collision detection) across world counts."""

    params = (WORLD_COUNTS,)
    param_names = ("world_count",)
    number = 20
    repeat = 5

    def setup(self, world_count):
        self.model = fixtures.quadruped_model(world_count, DEVICE)
        system = SystemArticulation.from_model(self.model, ROBOT)
        contact_set = ContactInputSet.from_system(system)
        self.sensor = contact_set.sensor(self.model, system.bind(self.model, ROBOT))
        self.state = self.model.state()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        pipeline = newton.CollisionPipeline(self.model)
        self.contacts = pipeline.contacts()
        pipeline.collide(self.state, self.contacts)
        self.sensor.update(self.state, self.contacts)  # Preload kernels

    def time_update(self, world_count):
        self.sensor.update(self.state, self.contacts)
        wp.synchronize_device(self.model.device)


if __name__ == "__main__":
    from newton.utils import run_benchmark

    for benchmark in (SolverNeRDStep, ContactInputProbesSensing, ContactInputSetSensing):
        run_benchmark(benchmark)
