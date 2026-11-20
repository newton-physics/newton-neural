# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for ``TrajectoryWindows`` batch indexing and ``LossOneStep`` forward/backward passes.

Uses small CPU sizes when no GPU is available.
"""

import tempfile

import torch
import warp as wp

wp.config.log_level = wp.LOG_WARNING

from newton_neural.backbones import BackboneBase
from newton_neural.data import TrajectoryBatch, TrajectoryWindows, TrajectoryWriter
from newton_neural.models import FlashNeRD
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.training import LossOneStep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORLD_COUNT, FRAME_COUNT = (2048, 256) if DEVICE.type == "cuda" else (16, 64)
BATCH_SIZES = (256, 4096) if DEVICE.type == "cuda" else (64,)
WINDOW_LENGTH = 16  # Steps per window
FRAME_DT = 0.02  # Frame duration [s]
COMPILED = (False, True) if DEVICE.type == "cuda" else (False,)


class TrajectoryWindowsBatch:
    """On-device gathering of 17-frame quadruped windows across batch sizes."""

    params = (BATCH_SIZES,)
    param_names = ("batch_size",)
    number = 20
    repeat = 5

    def setup(self, batch_size):
        # Synthetic random dataset (one episode per world).
        system = SystemArticulation.from_model(fixtures.quadruped_model(1, "cpu"), "quadruped")
        frames = fixtures.random_frames(system, WORLD_COUNT, FRAME_COUNT)
        valid = torch.ones(WORLD_COUNT, FRAME_COUNT, dtype=torch.bool)
        episode_start = torch.zeros_like(valid)
        episode_start[:, 0] = True
        self.folder = tempfile.TemporaryDirectory()
        writer = TrajectoryWriter(self.folder.name, system.signals, FRAME_DT)
        writer.write(frames, episode_start, valid)

        self.windows = TrajectoryWindows(writer.close(), WINDOW_LENGTH, device=DEVICE)
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        self.indices = torch.randint(len(self.windows), (batch_size,), device=DEVICE, generator=generator)

    def time_batch(self, batch_size):
        self.windows.batch(self.indices)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)

    def teardown(self, batch_size):
        self.folder.cleanup()


class LossOneStepGradient:
    """``LossOneStep`` forward + backward pass (TF32 enabled) across backbones, batch sizes, and compile modes."""

    params = (("BackboneFlash", "BackboneTransformer", "BackboneMLP"), BATCH_SIZES, COMPILED)
    param_names = ("backbone", "batch_size", "compiled")
    number = 20
    repeat = 5

    def setup(self, backbone, batch_size, compiled):
        # ASV runs each parameter combo in its own process.
        torch.set_float32_matmul_precision("high")
        system = SystemArticulation.from_model(fixtures.quadruped_model(1, "cpu"), "quadruped")
        self.nerd = FlashNeRD(system, backbone=BackboneBase.create(backbone, {})).to(DEVICE)
        frames = fixtures.random_frames(system, batch_size, WINDOW_LENGTH + 1, device=DEVICE)
        # Mark windows as episode starts so all frames contribute regardless of backbone receptive field.
        self.batch = TrajectoryBatch(frames, torch.ones(batch_size, dtype=torch.bool, device=DEVICE), FRAME_DT)
        self.nerd.fit_normalization([self.batch])
        self.loss = LossOneStep(noise=0.1)
        if compiled:
            self.loss.compile()
        self.time_gradient(backbone, batch_size, compiled)  # Warm up: compile and verify first call

    def time_gradient(self, backbone, batch_size, compiled):
        self.nerd.zero_grad()
        self.loss(self.nerd, self.batch).backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)


if __name__ == "__main__":
    from newton.utils import run_benchmark

    for benchmark in (TrajectoryWindowsBatch, LossOneStepGradient):
        run_benchmark(benchmark)
