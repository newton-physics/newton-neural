# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Coupling: two-link arm pushing a free box via two-way coupling (``SolverNeRDMuJoCo``).

Trains a single-frame MLP free-motion model on independent samples of the arm alone (``PredictionFreeMotion``), then
deploys it inside MuJoCo Warp to push a heavy free box unseen during training. A heavy box keeps pushing trajectories
well-conditioned since light boxes amplify sub-milliradian joint differences into centimeter displacements.

Success criteria:
- Paired arms sweep into their heavy boxes, push them a few centimeters, and stall short of their joint targets.
- After 100 rendered frames (2 s), median discrepancy is < 0.03 rad (arm) and < 0.03 m (box), while the MuJoCo
  Warp arm is held > 0.3 rad short of its joint targets.

Runs:
- `python -m newton_neural.examples coupling`
- `python -m newton_neural.examples coupling --viewer null --num-frames 100 --test`
"""

import argparse
import math
import time
from pathlib import Path

import newton
import newton.examples
import newton.viewer
import numpy as np
import torch
import warp as wp

import newton_neural.examples
from newton_neural.backbones import BackboneMLP
from newton_neural.data import (
    CommandNearState,
    SamplerUniform,
    SimulationNewton,
    TrajectoryDataset,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.models import FlashNeRD
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.solvers import SolverNeRDMuJoCo
from newton_neural.systems import SystemArticulation
from newton_neural.training import LossOneStep

SIM_DT = 0.005  # Single solver step [s] (matches model frame_dt for free-motion coupling)
SUBSTEPS = 4  # Solver steps per rendered frame
START = (-0.9, 0.0)  # Initial arm joint angles [rad]
TARGET = (0.6, 0.3)  # Sweep endpoint joint targets past the box [rad]
SWEEP_TIME = 2.0  # Half-period of cosine sweep [s]

REACH = 1.5  # Sampled joint angle half-range [rad]
SPEED = 3.0  # Sampled joint velocity half-range [rad/s]
TARGET_REACH = 1.0  # Max sampled target offset from current joint angle [rad]
RECORD_WORLD_COUNT = 4096  # Parallel worlds during recording
RECORD_FRAME_COUNT = 64  # Independent samples per world (~262k total)

BATCH_SIZE = 4096  # Samples per optimizer step
LEARNING_RATE = 2e-3  # Peak LR

TOLERANCE = 0.03  # Max median discrepancy vs. MuJoCo Warp: arm [rad] and box [m]
HELD_BACK = 0.3  # Minimum target lag proving box resistance on MuJoCo Warp's arm [rad]


def build_model(world_count: int, *, seed: int) -> newton.Model:
    """Build ``world_count`` scenes pairing a two-link arm with a heavy free box in its sweep path."""
    along_x = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
    arm = newton.ModelBuilder()
    parent, joints = -1, []
    for link, joint, angle in zip(("upper_arm", "forearm"), ("shoulder", "elbow"), START, strict=True):
        body = arm.add_link(label=link)
        arm.add_shape_capsule(body, xform=wp.transform((0.2, 0.0, 0.0), along_x), radius=0.03, half_height=0.2)
        joints.append(
            arm.add_joint_revolute(
                parent,
                body,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform((0.0, 0.0, 0.05) if parent < 0 else (0.4, 0.0, 0.0), wp.quat_identity()),
                target_ke=100.0,
                target_kd=10.0,
                armature=0.1,  # Rotor inertia keeping forearm accelerations well-scaled
                label=joint,
            )
        )
        arm.joint_q[-1] = arm.joint_target_q[-1] = angle
        parent = body
    arm.add_articulation(joints, label="arm")

    heavy = newton.ModelBuilder.ShapeConfig(density=6000.0)  # ~15 kg box
    random = np.random.default_rng(seed)
    scene = newton.ModelBuilder()
    for _ in range(world_count):
        distance = random.uniform(0.4, 0.65)  # Box placement along arm reach [m]
        scene.begin_world()
        box = scene.add_body(xform=wp.transform((distance, 0.0, 0.05), wp.quat_identity()), label="box")
        scene.add_shape_box(box, hx=0.08, hy=0.08, hz=0.05, cfg=heavy)
        scene.add_builder(arm)
        scene.end_world()
    scene.add_ground_plane()
    return scene.finalize()


def record(nerd: FlashNeRD, path: Path) -> TrajectoryDataset:
    """Record independent single-step free-acceleration samples of the arm."""
    model = build_model(RECORD_WORLD_COUNT, seed=0)
    solver = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, solver, nerd.system, articulation="arm", frame_dt=SIM_DT)
    reach = torch.full((2,), REACH, device=simulation.device)
    speed = torch.full((2,), SPEED, device=simulation.device)
    recorder = TrajectoryRecorder(
        simulation,
        nerd,
        states=SamplerUniform({"joint_q": (-reach, reach), "joint_qd": (-speed, speed)}),
        commands=CommandNearState(TARGET_REACH),
        episode_frames=(1, 1),
    )
    start = time.perf_counter()
    dataset = recorder.record(path, RECORD_FRAME_COUNT)
    elapsed = time.perf_counter() - start
    print(f"Recorded {dataset.frame_count} samples for each of {dataset.world_count} worlds in {elapsed:.1f} s.")
    return dataset


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.sim_time = 0.0

        self.model = build_model(args.world_count, seed=1)
        device = wp.device_to_torch(self.model.device)

        # Record and train a single-frame free-motion MLP unless cached.
        folder = newton_neural.examples.cache_folder(args, "coupling")
        checkpoint = folder / f"flashnerd_{args.train_steps}_steps.pt"
        if not checkpoint.exists():
            torch.manual_seed(0)
            system = SystemArticulation.from_model(self.model, "arm")
            nerd = FlashNeRD(system, backbone=BackboneMLP(window=1), prediction=PredictionFreeMotion()).to(device)
            dataset = newton_neural.examples.cached_dataset(folder / "dataset")
            if dataset is None:
                dataset = record(nerd, folder / "dataset")
            windows = TrajectoryWindows(dataset, 0, device=device)
            newton_neural.examples.train(
                nerd, windows, LossOneStep(), args.train_steps, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
            )
            nerd.save(checkpoint)

        # Step MuJoCo Warp and SolverNeRDMuJoCo side by side on separate states.
        self.solvers = {
            "MuJoCo Warp": newton.solvers.SolverMuJoCo(self.model),
            "FlashNeRD": SolverNeRDMuJoCo(self.model, checkpoint, articulation="arm"),
        }
        self.states = {name: self.model.state() for name in self.solvers}
        self.control = self.model.control()
        self.binding = self.solvers["FlashNeRD"].binding
        self.start = torch.tensor(START, device=device).expand(args.world_count, -1)
        self.target = torch.tensor(TARGET, device=device).expand(args.world_count, -1)
        # Warm up once outside capture (compiles & validates), then capture all substeps into one graph.
        self.simulate()
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        # Render MuJoCo Warp and FlashNeRD in separate layers offset 1.7 m along Y.
        for index, layer in enumerate(self.solvers):
            self.viewer.activate(layer)
            self.viewer.set_model(self.model)
            self.viewer.set_world_offsets((2.0, 6.0, 0.0))
            self.viewer.set_layer_transform(layer, (0.0, 1.7 * index, 0.0))
        self.viewer.set_camera(pos=wp.vec3(6.5, 0.85, 5.5), pitch=-42.0, yaw=180.0)

    def simulate(self) -> None:
        for _ in range(SUBSTEPS):
            for name, solver in self.solvers.items():
                solver.step(self.states[name], self.states[name], self.control, None, SIM_DT)

    def step(self) -> None:
        # Cosine-interpolate joint targets smoothly between START and TARGET.
        blend = 0.5 - 0.5 * math.cos(math.pi * self.sim_time / SWEEP_TIME)
        targets = {"joint_target_q": self.start + blend * (self.target - self.start)}
        self.binding.write_control(self.control, targets)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += SUBSTEPS * SIM_DT

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        for layer, state in self.states.items():
            self.viewer.activate(layer)
            self.viewer.log_state(state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        mujoco, twin = self.states.values()
        targets = self.binding.read_control(self.control)["joint_target_q"]

        def arm(state: newton.State) -> torch.Tensor:  # Per-world arm joint angles [rad]
            return self.binding.read_state(state)["joint_q"]

        def box(state: newton.State) -> torch.Tensor:  # Per-world box position [m] (body 0 of each world)
            return wp.to_torch(state.body_q).view(len(targets), -1, 7)[:, 0, :3]

        # Median across worlds of max arm joint discrepancy [rad] and box Euclidean displacement [m].
        arm_apart = (arm(twin) - arm(mujoco)).abs().amax(dim=1).median().item()
        box_apart = (box(twin) - box(mujoco)).norm(dim=1).median().item()
        mujoco_short = (arm(mujoco) - targets).abs().amax(dim=1).median().item()
        print(
            f"After {round(self.sim_time / (SUBSTEPS * SIM_DT))} rendered frames, in the median world, the model is "
            f"{arm_apart:.4f} rad and {box_apart:.4f} m from MuJoCo Warp, whose arm stops {mujoco_short:.3f} rad "
            f"short of its joint targets."
        )
        if not max(arm_apart, box_apart) < TOLERANCE:
            raise ValueError(f"the model is more than {TOLERANCE} rad or m from MuJoCo Warp")
        if not mujoco_short > HELD_BACK:
            raise ValueError(f"the box holds the arm of MuJoCo Warp less than {HELD_BACK} rad short of its targets")


if __name__ == "__main__":
    parser = newton_neural.examples.create_parser(world_count=4, train_steps=4000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
