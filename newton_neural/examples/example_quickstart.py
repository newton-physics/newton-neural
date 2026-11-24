# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Quickstart: end-to-end Record -> Train -> Evaluate -> Deploy on a position-controlled cartpole.

Records trajectories from MuJoCo Warp under random piecewise-constant joint targets, trains a FlashNeRD model in a
plain PyTorch loop, evaluates open-loop rollout error on held-out worlds, saves a checkpoint, and runs SolverNeRD
side by side with MuJoCo Warp. Reuses cached datasets and checkpoints on subsequent runs.

Success criteria:
- Paired cartpoles (MuJoCo Warp vs. FlashNeRD) track random targets changing every second with matching motion.
- After 200 frames (4 s), max joint-coordinate discrepancy across all worlds is < 0.1 m or rad.

Runs:
- `python -m newton_neural.examples quickstart`
- `python -m newton_neural.examples quickstart --viewer null --num-frames 200 --test`
"""

import argparse
import itertools
import time
from pathlib import Path

import newton
import newton.examples
import newton.viewer
import torch
import warp as wp

import newton_neural.examples
from newton_neural.data import (
    CommandHeld,
    SamplerUniform,
    SimulationNewton,
    TrajectoryDataset,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.evaluation import rollout_error
from newton_neural.models import FlashNeRD
from newton_neural.solvers import SolverNeRD
from newton_neural.systems import SystemArticulation
from newton_neural.training import LossOneStep, create_optimizer

FRAME_DT = 0.02  # Frame duration [s] (1 neural step per frame)
SUBSTEPS = 4  # MuJoCo Warp substeps per frame
REACH = (1.0, 1.5, 1.5)  # Max joint target range: cart [m], two poles [rad]
HOLD_FRAMES = (10, 50)  # Random hold duration per target [frames]
VIEW_HOLD_FRAMES = 50  # Target refresh interval during playback [frames] (1 s)

RECORD_WORLD_COUNT = 2048  # Parallel worlds during recording
RECORD_FRAME_COUNT = 256  # Frames per world (~524k total frames)
EPISODE_FRAMES = (100, 200)  # Episode length range before reset [frames]

WINDOW = 16  # Steps per training window (>= 13 receptive field of BackboneFlash)
BATCH_SIZE = 1024  # Windows per optimizer step
LEARNING_RATE = 2e-3  # Peak LR (cosine-annealed to 0)
NOISE = 0.1  # Input state noise [std devs of one-frame change]
HORIZON = 100  # Open-loop rollout horizon for evaluation [frames]

TOLERANCE = 0.1  # Max allowed joint-coordinate discrepancy vs. MuJoCo Warp [m or rad]


def build_model(world_count: int) -> newton.Model:
    """Build ``world_count`` replicated cartpoles with position-controlled joints."""
    cartpole = newton.ModelBuilder()
    cartpole.default_joint_cfg.target_ke = 100.0
    cartpole.default_joint_cfg.target_kd = 10.0
    cartpole.add_urdf(newton.examples.get_asset("cartpole.urdf"), floating=False, enable_self_collisions=False)
    scene = newton.ModelBuilder()
    scene.replicate(cartpole, world_count)
    return scene.finalize()


def record(nerd: FlashNeRD, path: Path) -> TrajectoryDataset:
    """Record cartpole trajectories from MuJoCo Warp under random piecewise-constant joint targets."""
    model = build_model(RECORD_WORLD_COUNT)
    solver = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, solver, nerd.system, frame_dt=FRAME_DT, substep_count=SUBSTEPS)
    reach = torch.tensor(REACH, device=simulation.device)
    recorder = TrajectoryRecorder(
        simulation,
        nerd,
        states=SamplerUniform({"joint_q": (-reach, reach), "joint_qd": (-reach, reach)}),
        commands=CommandHeld(HOLD_FRAMES, low=-reach, high=reach),
        episode_frames=EPISODE_FRAMES,
    )
    start = time.perf_counter()
    dataset = recorder.record(path, RECORD_FRAME_COUNT)
    elapsed = time.perf_counter() - start
    print(f"Recorded {dataset.frame_count} frames for each of {dataset.world_count} worlds in {elapsed:.1f} s.")
    return dataset


def train(nerd: FlashNeRD, windows: TrajectoryWindows, step_count: int) -> None:
    """Train ``nerd`` on trajectory windows with a cosine-annealed PyTorch loop."""
    # Enable TF32 matmuls on tensor-core GPUs for faster training.
    torch.set_float32_matmul_precision("high")
    nerd.fit_normalization(windows.loader(BATCH_SIZE))
    # Keep loss eager here: compile startup overhead outweighs gains for short runs.
    loss = LossOneStep(noise=NOISE)
    optimizer = create_optimizer(nerd, lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_count)
    start = time.perf_counter()
    for step, batch in enumerate(itertools.islice(windows.loader(BATCH_SIZE, epochs=None), step_count), start=1):
        optimizer.zero_grad()
        error = loss(nerd, batch)
        error.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == step_count:
            print(f"Step {step} of {step_count}, loss {error.item():.4f}, {time.perf_counter() - start:.0f} s.")


def evaluate(nerd: FlashNeRD, held_out: TrajectoryWindows) -> None:
    """Print open-loop rollout error on held-out worlds."""
    error = rollout_error(nerd, held_out, horizons=(HORIZON,))[HORIZON]
    print(
        f"Rolled out alone for {HORIZON} frames on held out worlds, the joint coordinates are off by "
        f"{error.p50['joint_q']:.4f} m or rad at the median and by {error.p99['joint_q']:.4f} at the 99th percentile."
    )


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame = 0

        self.model = build_model(args.world_count)
        device = wp.device_to_torch(self.model.device)

        # Record, train, and evaluate unless a cached checkpoint already exists.
        folder = newton_neural.examples.cache_folder(args, "quickstart")
        checkpoint = folder / f"flashnerd_{args.train_steps}_steps.pt"
        if not checkpoint.exists():
            torch.manual_seed(0)
            nerd = FlashNeRD(SystemArticulation.from_model(self.model)).to(device)
            dataset = newton_neural.examples.cached_dataset(folder / "dataset")
            if dataset is None:
                dataset = record(nerd, folder / "dataset")
            windows, held_out = TrajectoryWindows(dataset, WINDOW, device=device).split()
            train(nerd, windows, args.train_steps)
            evaluate(nerd, held_out)
            nerd.save(checkpoint)

        # Step MuJoCo Warp and FlashNeRD side by side on separate states under shared controls.
        self.solver_mujoco = newton.solvers.SolverMuJoCo(self.model)
        self.solver_nerd = SolverNeRD(self.model, checkpoint)
        self.state_mujoco = self.model.state()
        self.state_nerd = self.model.state()
        self.control = self.model.control()
        # Dedicated RNG keeps playback targets identical across cached and freshly trained runs.
        self.reach = torch.tensor(REACH, device=device).expand(args.world_count, -1)
        self.generator = torch.Generator(device).manual_seed(0)

        # Render MuJoCo Warp and FlashNeRD in separate layers offset 1 m apart.
        for layer, shift in (("MuJoCo Warp", 0.0), ("FlashNeRD", 1.0)):
            self.viewer.activate(layer)
            self.viewer.set_model(self.model)
            self.viewer.set_world_offsets((3.0, 6.0, 0.0))
            self.viewer.set_layer_transform(layer, (shift, 0.0, 0.0))
        self.viewer.set_camera(pos=wp.vec3(14.0, 0.0, 5.5), pitch=-18.0, yaw=180.0)

    def step(self) -> None:
        if self.frame % VIEW_HOLD_FRAMES == 0:  # Sample fresh joint targets
            unit = torch.rand(self.reach.shape, device=self.reach.device, generator=self.generator)
            targets = {"joint_target_q": self.reach * (2.0 * unit - 1.0)}
            self.solver_nerd.binding.write_control(self.control, targets)
        # Advance MuJoCo Warp via substeps and FlashNeRD in one frame step.
        for _ in range(SUBSTEPS):
            self.solver_mujoco.step(self.state_mujoco, self.state_mujoco, self.control, None, FRAME_DT / SUBSTEPS)
        self.solver_nerd.step(self.state_nerd, self.state_nerd, self.control, None, FRAME_DT)
        self.sim_time += FRAME_DT
        self.frame += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        for layer, state in (("MuJoCo Warp", self.state_mujoco), ("FlashNeRD", self.state_nerd)):
            self.viewer.activate(layer)
            self.viewer.log_state(state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        binding = self.solver_nerd.binding
        error = binding.system.error(binding.read_state(self.state_mujoco), binding.read_state(self.state_nerd))
        worst = error["joint_q"].max().item()
        print(f"After {self.frame} frames the joint coordinates differ by {worst:.4f} m or rad at most.")
        if not worst < TOLERANCE:
            raise ValueError(f"the model is {worst:.3f} m or rad from MuJoCo Warp, more than {TOLERANCE}")


if __name__ == "__main__":
    parser = newton_neural.examples.create_parser(world_count=9, train_steps=2000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
