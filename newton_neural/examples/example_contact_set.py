# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Contact Set: quadruped on randomized box platforms using collision-detected contacts (``ContactInputSet``).

Feeds rigid contacts from ``newton.CollisionPipeline`` into a permutation-invariant contact set encoder, then
captures collision detection and the neural step together in one CUDA graph on unseen platform geometries. Trains
briefly for a few-second comparison (not long-horizon stability or fall recovery).

Success criteria:
- Paired quadrupeds shift and sway on flat ground and randomized box platforms with matching motion.
- After 100 frames (2 s), median error is < 0.05 m base position, < 0.1 rad base orientation, < 0.05 rad joints,
  and zero contact-row overflows occur.

Runs:
- `python -m newton_neural.examples contact_set`
- `python -m newton_neural.examples contact_set --viewer null --num-frames 100 --test`
"""

import argparse
import time
from pathlib import Path

import newton
import newton.examples
import newton.viewer
import numpy as np
import torch
import warp as wp

import newton_neural.examples
from newton_neural.contacts import ContactInputSet
from newton_neural.data import (
    CommandHeld,
    SamplerFloatingBase,
    SimulationNewton,
    TrajectoryDataset,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.models import FlashNeRD
from newton_neural.solvers import SolverNeRD
from newton_neural.systems import SystemArticulation
from newton_neural.training import LossOneStep

FRAME_DT = 0.02  # Frame duration [s]
SUBSTEPS = 4  # MuJoCo Warp substeps per frame (each with collision detection)
STANCE = (0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6)  # Nominal standing joint angles [rad]
SWING = 0.15  # Target offset around STANCE [rad]
HOLD_FRAMES = (10, 50)  # Random target hold duration [frames]
VIEW_HOLD_FRAMES = 50  # Target refresh interval during playback [frames] (1 s)
PLATFORM_HEIGHT = (0.02, 0.06)  # Randomized platform height range [m]
PLATFORM_DISTANCE = (0.2, 0.7)  # Randomized platform distance in front of robot [m]

RECORD_WORLD_COUNT = 1024  # Parallel worlds during recording (each with a randomized platform)
RECORD_FRAME_COUNT = 256  # Frames per world (~262k total frames, ~1 GB of contact rows)
EPISODE_FRAMES = (100, 200)  # Valid episode length range [frames]
SETTLE_FRAME_COUNT = 25  # Leading settle frames marked invalid after reset
START_POSITION = ((-0.1, -0.05, -0.02), (0.1, 0.05, 0.02))  # Initial base offset bounds [m]
START_TILT = 0.06  # Max initial base tilt [rad]
START_SPEED = (0.2, 0.3, 1.0)  # Initial speed bounds: base linear [m/s], base angular [rad/s], joints [rad/s]

WINDOW = 16  # Steps per training window (>= 13 receptive field of BackboneFlash)
BATCH_SIZE = 1024  # Windows per optimizer step
LEARNING_RATE = 2e-3  # Peak LR
NOISE = 0.1  # Input state noise [std devs of one-frame change]

TOLERANCE = (0.05, 0.1, 0.05)  # Median error limits: base pos [m], base rot [rad], worst joint [rad]


def build_model(world_count: int, *, seed: int) -> newton.Model:
    """Build ``world_count`` quadrupeds on flat ground, each facing a randomly placed static box platform."""
    quadruped = newton.ModelBuilder()
    quadruped.default_joint_cfg.armature = 0.06
    quadruped.default_joint_cfg.target_ke = 500.0
    quadruped.default_joint_cfg.target_kd = 20.0
    quadruped.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform((0.0, 0.0, 0.52), wp.quat_identity()),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,  # URDF inertias are too small for stable simulation
    )
    quadruped.joint_q[-12:] = STANCE
    quadruped.joint_target_q[-12:] = STANCE

    random = np.random.default_rng(seed)
    scene = newton.ModelBuilder()
    for _ in range(world_count):
        height, distance = random.uniform(*PLATFORM_HEIGHT), random.uniform(*PLATFORM_DISTANCE)
        place = wp.transform((distance, 0.0, 0.5 * height), wp.quat_identity())
        scene.begin_world()
        scene.add_builder(quadruped)
        scene.add_shape_box(body=-1, xform=place, hx=0.15, hy=0.35, hz=0.5 * height)
        scene.end_world()
    scene.add_ground_plane()
    model = scene.finalize()
    # Evaluate FK so body transforms reflect STANCE before initial collision detection.
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    return model


def record(nerd: FlashNeRD, path: Path) -> TrajectoryDataset:
    """Record quadruped trajectories and detected contact rows using Newton's CollisionPipeline."""
    model = build_model(RECORD_WORLD_COUNT, seed=0)
    simulation = SimulationNewton(
        model,
        newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=128),
        nerd.system,
        frame_dt=FRAME_DT,
        substep_count=SUBSTEPS,
        collision_pipeline=newton.CollisionPipeline(model),
    )
    states = SamplerFloatingBase(
        simulation.binding.read_state(model.state())["joint_q"][0],
        position=START_POSITION,
        tilt=START_TILT,
        linear_speed=START_SPEED[0],
        angular_speed=START_SPEED[1],
        joint_reach=SWING,
        joint_speed=START_SPEED[2],
    )
    stance = torch.tensor(STANCE, device=simulation.device)
    recorder = TrajectoryRecorder(
        simulation,
        nerd,
        states=states,
        commands=CommandHeld(HOLD_FRAMES, low=stance - SWING, high=stance + SWING),
        episode_frames=EPISODE_FRAMES,
        settle_frame_count=SETTLE_FRAME_COUNT,
    )
    start = time.perf_counter()
    dataset = recorder.record(path, RECORD_FRAME_COUNT)
    elapsed = time.perf_counter() - start
    rows = dataset.arrays(0)["contact_count"]
    print(
        f"Recorded {dataset.frame_count} frames for each of {dataset.world_count} worlds in {elapsed:.1f} s. A frame "
        f"holds {rows.mean():.1f} contact rows on average and {rows.max()} at most, and the contact set has room "
        f"for {nerd.contact_input.capacity}."
    )
    return dataset


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame = 0

        # Use a different seed than training so playback tests unseen platform geometries.
        self.model = build_model(args.world_count, seed=1)
        device = wp.device_to_torch(self.model.device)

        folder = newton_neural.examples.cache_folder(args, "contact_set")
        checkpoint = folder / f"flashnerd_{args.train_steps}_steps.pt"
        if not checkpoint.exists():
            torch.manual_seed(0)
            system = SystemArticulation.from_model(self.model)
            nerd = FlashNeRD(system, contact_input=ContactInputSet.from_system(system)).to(device)
            dataset = newton_neural.examples.cached_dataset(folder / "dataset")
            if dataset is None:
                dataset = record(nerd, folder / "dataset")
            windows = TrajectoryWindows(dataset, WINDOW, device=device)
            loss = LossOneStep(noise=NOISE)
            newton_neural.examples.train(
                nerd, windows, loss, args.train_steps, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
            )
            nerd.save(checkpoint)

        # Both solvers use separate states and Newton-detected contact buffers.
        self.solver_mujoco = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=128)
        self.solver_nerd = SolverNeRD(self.model, checkpoint)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.state_mujoco = self.model.state()
        self.state_nerd = self.model.state()
        self.contacts_mujoco = self.collision_pipeline.contacts()
        self.contacts_nerd = self.collision_pipeline.contacts()
        self.control = self.model.control()
        overflow = SolverNeRD.ObservableKind.CONTACT_ROW_OVERFLOW
        self.observables = self.solver_nerd.observables({overflow}, contacts=self.contacts_nerd)
        # Dedicated RNG keeps playback targets identical across cached and freshly trained runs.
        self.stance = torch.tensor(STANCE, device=device).expand(args.world_count, -1)
        self.generator = torch.Generator(device).manual_seed(0)

        # Let quadrupeds settle in MuJoCo Warp, then sync FlashNeRD to that pose.
        for _ in range(SETTLE_FRAME_COUNT):
            self.simulate_mujoco()
        self.state_nerd.assign(self.state_mujoco)
        # Warm up once outside capture (compiles & validates), then capture collision + both solvers into one graph.
        self.simulate()
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        # Render MuJoCo Warp and FlashNeRD in separate layers offset 1.1 m along Y.
        for layer, shift in (("MuJoCo Warp", 0.0), ("FlashNeRD", 1.1)):
            self.viewer.activate(layer)
            self.viewer.set_model(self.model)
            self.viewer.set_world_offsets((2.5, 2.8, 0.0))
            self.viewer.set_layer_transform(layer, (0.0, shift, 0.0))
        self.viewer.set_camera(pos=wp.vec3(10.0, -7.0, 5.0), pitch=-25.0, yaw=150.0)

    def simulate_mujoco(self) -> None:
        for _ in range(SUBSTEPS):
            self.collision_pipeline.collide(self.state_mujoco, self.contacts_mujoco)
            self.solver_mujoco.step(
                self.state_mujoco, self.state_mujoco, self.control, self.contacts_mujoco, FRAME_DT / SUBSTEPS
            )

    def simulate(self) -> None:
        self.simulate_mujoco()
        self.collision_pipeline.collide(self.state_nerd, self.contacts_nerd)
        self.solver_nerd.step(
            self.state_nerd, self.state_nerd, self.control, self.contacts_nerd, FRAME_DT, observables=self.observables
        )

    def step(self) -> None:
        if self.frame % VIEW_HOLD_FRAMES == 0:  # Sample fresh joint targets
            unit = torch.rand(self.stance.shape, device=self.stance.device, generator=self.generator)
            targets = {"joint_target_q": self.stance + SWING * (2.0 * unit - 1.0)}
            self.solver_nerd.binding.write_control(self.control, targets)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
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
        # Floating-base joint_q error layout: [base_pos (3), base_angle (1), joints (12)].
        position, angle, joints = error["joint_q"].split((3, 1, 12), dim=1)
        found = (position.norm(dim=1).median().item(), angle.median().item(), joints.amax(dim=1).median().item())
        print(
            f"After {self.frame} frames the median twin is off by {found[0]:.3f} m and "
            f"{found[1]:.3f} rad at its base and by {found[2]:.3f} rad at its worst joint."
        )
        if not all(value < tolerance for value, tolerance in zip(found, TOLERANCE, strict=True)):
            raise ValueError(f"the median twin is off by more than {TOLERANCE} at its base [m, rad] or joints [rad]")
        dropped = int(wp.to_torch(self.observables.contact_row_overflow).sum())
        if dropped:
            raise ValueError(f"{dropped} contact rows found no room in the contact set. Raise its capacity")


if __name__ == "__main__":
    parser = newton_neural.examples.create_parser(world_count=16, train_steps=2000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
