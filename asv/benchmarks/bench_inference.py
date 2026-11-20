# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for standalone backbone and ``NeRDStepper`` single-frame inference latency.

Isolates network cost from contact sensing and Newton overhead. Uses small CPU world counts when no GPU is present.
"""

import torch
import warp as wp

wp.config.log_level = wp.LOG_WARNING

from newton_neural.backbones import BackboneBase
from newton_neural.models import FlashNeRD, NeRDStepper
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORLD_COUNTS = (1, 1024, 16384) if DEVICE.type == "cuda" else (1, 16)
PRECISIONS = ("fp32", "tf32", "fp16")
FRAME_DT = 0.02  # Frame duration [s]


def synchronize() -> None:
    """Block until queued CUDA kernels finish."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


class BackboneStep:
    """Eager FP32 single-frame latency for each built-in backbone across world counts."""

    params = (("BackboneFlash", "BackboneTransformer", "BackboneMLP"), WORLD_COUNTS)
    param_names = ("backbone", "world_count")
    number = 20
    repeat = 5

    def setup(self, backbone, world_count):
        self.backbone = BackboneBase.create(backbone, {}).to(DEVICE)
        self.frame = torch.randn(world_count, self.backbone.width, device=DEVICE)
        self.history = self.backbone.history(world_count, device=DEVICE, dtype=torch.float32)

    def time_step(self, backbone, world_count):
        with torch.no_grad():
            self.backbone.step(self.frame, self.history)
        synchronize()


class NeRDStepperStep:
    """Compiled ``NeRDStepper`` latency for a floating-base quadruped FlashNeRD across world counts and precisions."""

    params = (WORLD_COUNTS, PRECISIONS)
    param_names = ("world_count", "precision")
    number = 20
    repeat = 5

    def setup(self, world_count, precision):
        system = SystemArticulation.from_model(fixtures.quadruped_model(1, DEVICE), "quadruped")
        nerd = FlashNeRD(system, frame_dt=FRAME_DT).to(DEVICE)
        self.stepper = NeRDStepper(nerd, world_count, precision=precision)
        self.frames = fixtures.random_frames(system, world_count, device=DEVICE)
        self.stepper.step(self.frames)  # Warm up: compile and validate first step

    def time_step(self, world_count, precision):
        self.stepper.step(self.frames)
        synchronize()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    for benchmark in (BackboneStep, NeRDStepperStep):
        run_benchmark(benchmark)
