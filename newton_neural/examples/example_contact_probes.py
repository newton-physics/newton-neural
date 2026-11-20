# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Contact Probes: quadruped on flat ground using foot probes (``ContactInputProbes``).

Senses ground contact analytically at four foot spheres without collision detection. Probes are recomputed via FK
on perturbed training states and open-loop rollout states. Trains briefly for a few-second visual comparison (not
for long-horizon stability or recovery from falls).

Success criteria:
- Paired quadrupeds (MuJoCo Warp vs. FlashNeRD) shift and sway on flat ground with matching motion.
- After 100 frames (2 s), median error is < 0.05 m base position, < 0.1 rad base orientation, and < 0.05 rad joints.

Runs:
- `python -m newton_neural.examples contact_probes`
- `python -m newton_neural.examples contact_probes --viewer null --num-frames 100 --test`
"""

import argparse
import time
from pathlib import Path

import newton
import newton.examples
import newton.viewer
import torch
import warp as wp

import newton_neural.examples
from newton_neural.contacts import ContactInputProbes
from newton_neural.data import (
    CommandHeld,
    SamplerFloatingBase,
    SimulationNewton,
    TrajectoryDataset,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.evaluation import rollout_error
from newton_neural.models import FlashNeRD
from newton_neural.solvers import SolverNeRD
from newton_neural.systems import SystemArticulation
from newton_neural.training import LossOneStep

FRAME_DT = 0.02  # Frame duration [s]
SUBSTEPS = 4  # MuJoCo Warp substeps per frame
STANCE = (0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6)  # Nominal standing joint angles [rad]
SWING = 0.15  # Target offset around STANCE [rad]
HOLD_FRAMES = (10, 50)  # Random target hold duration [frames]
VIEW_HOLD_FRAMES = 50  # Target refresh interval during playback [frames] (1 s)
FEET = ("LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK")  # Foot-bearing shank bodies
FOOT = {"point": [0.0, 0.0, -0.25], "radius": 0.02}  # Foot sphere center in shank frame and radius [m]

RECORD_WORLD_COUNT = 2048  # Parallel worlds during recording
RECORD_FRAME_COUNT = 256  # Frames per world (~524k total frames)
EPISODE_FRAMES = (100, 200)  # Valid episode length range [frames]
SETTLE_FRAME_COUNT = 25  # Leading settle frames marked invalid after reset
START_POSITION = ((-0.1, -0.05, -0.02), (0.1, 0.05, 0.02))  # Initial base offset bounds [m]
START_TILT = 0.06  # Max initial base tilt [rad]
START_SPEED = (0.2, 0.3, 1.0)  # Initial speed bounds: base linear [m/s], base angular [rad/s], joints [rad/s]

WINDOW = 16  # Steps per training window (>= 13 receptive field of BackboneFlash)
BATCH_SIZE = 1024  # Windows per optimizer step
LEARNING_RATE = 2e-3  # Peak LR
NOISE = 0.1  # Input state noise [std devs of one-frame change]
HORIZON = 100  # Open-loop rollout horizon [frames]

TOLERANCE = (0.05, 0.1, 0.05)  # Median error limits: base pos [m], base rot [rad], worst joint [rad]


def build_model(world_count: int) -> newton.Model:
    """Build ``world_count`` replicated quadrupeds on a flat ground plane."""
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
    scene = newton.ModelBuilder()
    scene.replicate(quadruped, world_count)
    scene.add_ground_plane()
    return scene.finalize()


def record(nerd: FlashNeRD, path: Path) -> TrajectoryDataset:
    """Record quadruped trajectories and foot probe signals from MuJoCo Warp."""
    model = build_model(RECORD_WORLD_COUNT)
    solver = newton.solvers.SolverMuJoCo(model, njmax=128)
    simulation = SimulationNewton(model, solver, nerd.system, frame_dt=FRAME_DT, substep_count=SUBSTEPS)
    # Sample starting poses near standing height with small tilt and velocities.
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
    print(f"Recorded {dataset.frame_count} frames for each of {dataset.world_count} worlds in {elapsed:.1f} s.")
    return dataset


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame = 0

        self.model = build_model(args.world_count)
        device = wp.device_to_torch(self.model.device)

        # Record, train, and evaluate with foot probes unless a cached checkpoint exists.
        folder = newton_neural.examples.cache_folder(args, "contact_probes")
        checkpoint = folder / f"flashnerd_{args.train_steps}_steps.pt"
        if not checkpoint.exists():
            torch.manual_seed(0)
            probes = [{"body": body, **FOOT} for body in FEET]
            contact_input = ContactInputProbes(probes, ground={"normal": [0.0, 0.0, 1.0], "offset": 0.0})
            nerd = FlashNeRD(SystemArticulation.from_model(self.model), contact_input=contact_input).to(device)
            dataset = newton_neural.examples.cached_dataset(folder / "dataset")
            if dataset is None:
                dataset = record(nerd, folder / "dataset")
            windows, held_out = TrajectoryWindows(dataset, WINDOW, device=device).split()
            loss = LossOneStep(noise=NOISE)
            newton_neural.examples.train(
                nerd, windows, loss, args.train_steps, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE
            )
            error = rollout_error(nerd, held_out, horizons=(HORIZON,))[HORIZON]
            print(
                f"Rolled out alone for {HORIZON} frames on held out worlds, the joint coordinates are off by "
                f"{error.p50['joint_q']:.4f} m or rad at the median and by {error.p90['joint_q']:.4f} at the "
                "90th percentile."
            )
            nerd.save(checkpoint)

        # Step MuJoCo Warp and FlashNeRD side by side (no external collision pipeline needed for probes).
        self.solver_mujoco = newton.solvers.SolverMuJoCo(self.model, njmax=128)
        self.solver_nerd = SolverNeRD(self.model, checkpoint)
        self.state_mujoco = self.model.state()
        self.state_nerd = self.model.state()
        self.control = self.model.control()
        # Dedicated RNG keeps playback targets identical across cached and freshly trained runs.
        self.stance = torch.tensor(STANCE, device=device).expand(args.world_count, -1)
        self.generator = torch.Generator(device).manual_seed(0)

        # Let quadrupeds settle onto the ground in MuJoCo Warp, then sync FlashNeRD to that pose.
        for _ in range(SETTLE_FRAME_COUNT):
            self.simulate_mujoco()
        self.state_nerd.assign(self.state_mujoco)

        # Render MuJoCo Warp and FlashNeRD in separate layers offset 1 m along Y.
        for layer, shift in (("MuJoCo Warp", 0.0), ("FlashNeRD", 1.0)):
            self.viewer.activate(layer)
            self.viewer.set_model(self.model)
            self.viewer.set_world_offsets((2.5, 2.5, 0.0))
            self.viewer.set_layer_transform(layer, (0.0, shift, 0.0))
        self.viewer.set_camera(pos=wp.vec3(10.0, -7.0, 5.0), pitch=-25.0, yaw=150.0)

    def simulate_mujoco(self) -> None:
        for _ in range(SUBSTEPS):
            self.solver_mujoco.step(self.state_mujoco, self.state_mujoco, self.control, None, FRAME_DT / SUBSTEPS)

    def step(self) -> None:
        if self.frame % VIEW_HOLD_FRAMES == 0:  # Sample fresh joint targets
            unit = torch.rand(self.stance.shape, device=self.stance.device, generator=self.generator)
            targets = {"joint_target_q": self.stance + SWING * (2.0 * unit - 1.0)}
            self.solver_nerd.binding.write_control(self.control, targets)
        self.simulate_mujoco()
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
        # Floating-base joint_q error layout: [base_pos (3), base_angle (1), joints (12)].
        position, angle, joints = error["joint_q"].split((3, 1, 12), dim=1)
        found = (position.norm(dim=1).median().item(), angle.median().item(), joints.amax(dim=1).median().item())
        print(
            f"After {self.frame} frames the median twin is off by {found[0]:.3f} m and "
            f"{found[1]:.3f} rad at its base and by {found[2]:.3f} rad at its worst joint."
        )
        if not all(value < tolerance for value, tolerance in zip(found, TOLERANCE, strict=True)):
            raise ValueError(f"the median twin is off by more than {TOLERANCE} at its base [m, rad] or joints [rad]")


if __name__ == "__main__":
    parser = newton_neural.examples.create_parser(world_count=16, train_steps=2000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
