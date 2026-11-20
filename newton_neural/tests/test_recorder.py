# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``TrajectoryRecorder``: episodes, divergence resets, partial recovery, contacts, and labels."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import newton
import numpy as np
import torch
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.core.signal import SignalRole, SignalSpec
from newton_neural.contacts import ContactInputSet
from newton_neural.data import (
    CommandHeld,
    SamplerBase,
    SamplerFloatingBase,
    SamplerSobol,
    SamplerUniform,
    SignalSource,
    SimulationNewton,
    TrajectoryDataset,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.models import FlashNeRD
from newton_neural.predictions import PredictionFreeMotion, PredictionStateIncrement
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.tests.extensions import ContactInputRootHeight, ContactInputRootHeightRecorded
from newton_neural.tests.helpers import make_backbone
from newton_neural.tests.unittest_utils import FolderTestCase, add_function_test

FRAME_DT = 0.005
WORLD_COUNT = 4
EFFORT_LIMIT = 2.0
"""Effort limit given to every joint of the arm [N or N*m], which bounds the sampled joint forces."""


def _arm_simulation(
    device: torch.device, solver: str = "featherstone", *, control: tuple[str, ...] = ("joint_f",), box: bool = False
) -> SimulationNewton:
    """The arm in the air, alone or behind a free box, stepped once per frame without contacts."""
    model = (fixtures.arm_with_box_model if box else fixtures.arm_model)(WORLD_COUNT, device)
    model.joint_effort_limit.fill_(EFFORT_LIMIT)
    system = SystemArticulation.from_model(model, "arm", control=control)
    if solver == "mujoco":
        # Explicit Euler without contacts, so that a step is the free acceleration times the step.
        stepper = newton.solvers.SolverMuJoCo(model, integrator="euler", disable_contacts=True)
    else:
        stepper = newton.solvers.SolverFeatherstone(model)
    return SimulationNewton(model, stepper, system, articulation="arm", frame_dt=FRAME_DT)


def _sources(simulation: SimulationNewton) -> list[SignalSource]:
    """The state before each step and the controls the solver consumed in it, as new tensors."""

    def with_role(role: SignalRole) -> dict[str, SignalSpec]:
        return {name: spec for name, spec in simulation.binding.system.signals.items() if spec.role == role}

    def clones(frames: Frames) -> Frames:
        return {name: value.clone() for name, value in frames.items()}

    binding = simulation.binding
    return [
        SignalSource(with_role(SignalRole.STATE), lambda: clones(binding.read_state(simulation.state))),
        SignalSource(
            with_role(SignalRole.CONTROL), lambda: clones(binding.read_control(simulation.control)), after_step=True
        ),
    ]


def _near_rest(simulation: SimulationNewton, seed: int = 0) -> SamplerUniform:
    """Starting states within a radian (or meter) and a radian (or meter) per second of rest."""
    one = torch.ones(simulation.binding.system.joint_dof_count, device=simulation.device)
    generator = torch.Generator(device=simulation.device).manual_seed(seed)
    return SamplerUniform({"joint_q": (-one, one), "joint_qd": (-one, one)}, generator=generator)


class _CommandLogged(CommandHeld):
    """Held random commands that keep what they issued and replace the command of chosen worlds at chosen frames.

    Args:
        replaced: The value that takes the place of the whole command, by ``(frame, world)``.
    """

    def __init__(self, replaced: dict[tuple[int, int], float] | None = None) -> None:
        super().__init__()
        self.replaced = replaced or {}
        self.issued: list[Tensor] = []

    def next(self, frames: Frames | None = None) -> Tensor:
        command = super().next(frames).clone()
        for (frame, world), value in self.replaced.items():
            if frame == len(self.issued):
                command[world] = value
        self.issued.append(command)
        return command


class TestTrajectoryRecorder(FolderTestCase):
    pass


def test_dataset_layout(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Write every signal of the sources with its dtype and shape, the requested frames, and the metadata."""
    simulation = _arm_simulation(device, control=("joint_target_q", "joint_f"))
    low = torch.full((16,), -0.5, device=device)
    recorder = TrajectoryRecorder(
        simulation,
        _sources(simulation),
        states=_near_rest(simulation),
        commands=CommandHeld(low=low, high=-low),
        episode_frames=(5, 9),
        seed=3,
    )
    test.assertEqual(recorder.signals, simulation.binding.system.signals)
    dataset = recorder.record(test.folder / "data", 21, shard_frame_count=8)

    test.assertEqual(dataset.signals, simulation.binding.system.signals)
    test.assertEqual(dataset.frame_dt, FRAME_DT)
    test.assertEqual(dataset.world_count, WORLD_COUNT)
    test.assertEqual(dataset.frame_count, 21)
    test.assertIn(f"{WORLD_COUNT * 21} frames of {WORLD_COUNT} worlds", repr(dataset))
    test.assertEqual([shard["frame_count"] for shard in dataset.shards], [8, 8, 5])
    test.assertEqual([shard["continues"] for shard in dataset.shards], [False, True, True])
    test.assertEqual(
        dataset.generation,
        {
            "states": "SamplerUniform",
            "commands": "CommandHeld",
            "episode_frames": [5, 9],
            "settle_frame_count": 0,
            "seed": 3,
            "world_count": WORLD_COUNT,
            "complete": True,
        },
    )
    for shard, frame_count in enumerate((8, 8, 5)):
        arrays = dataset.arrays(shard)
        test.assertEqual(set(arrays), {"joint_q", "joint_qd", "joint_target_q", "joint_f", "episode_start", "valid"})
        for name, spec in dataset.signals.items():
            test.assertEqual(arrays[name].shape, (WORLD_COUNT, frame_count, *spec.shape))
            test.assertEqual(arrays[name].dtype, np.float32)
            test.assertTrue(np.isfinite(arrays[name]).all())
        test.assertTrue(arrays["valid"].all())
    test.assertTrue(dataset.arrays(0)["episode_start"][:, 0].all())
    # Both control signals were split from one command vector, each between its bounds.
    test.assertLessEqual(np.abs(dataset.arrays(0)["joint_target_q"]).max(), 0.5)
    test.assertLessEqual(np.abs(dataset.arrays(0)["joint_f"]).max(), 0.5)
    test.assertGreater(np.abs(dataset.arrays(0)["joint_f"] - dataset.arrays(0)["joint_target_q"]).max(), 0.0)


def test_frames_consecutive(test: TestTrajectoryRecorder, device: torch.device, solver: str) -> None:
    """Record states that follow from one another under the recorded controls, and flag where worlds restarted."""
    simulation = _arm_simulation(device, solver)
    commands = _CommandLogged()
    recorder = TrajectoryRecorder(
        simulation, _sources(simulation), states=_near_rest(simulation), commands=commands, episode_frames=(4, 7)
    )
    frame_count = 20
    arrays = recorder.record(test.folder / "data", frame_count, shard_frame_count=frame_count).arrays(0)
    frames = {name: torch.as_tensor(np.array(value), device=device) for name, value in arrays.items()}
    episode_start = frames["episode_start"]

    # The recorded controls are the commands the solver was given, between the bounds from the effort limits.
    test.assertTrue(torch.equal(frames["joint_f"], torch.stack(commands.issued, dim=1)))
    test.assertLessEqual(float(frames["joint_f"].abs().max()), EFFORT_LIMIT)
    test.assertGreater(float(frames["joint_f"].abs().max()), 0.5 * EFFORT_LIMIT)
    test.assertTrue(frames["valid"].all())

    # Every world starts with an episode, and its episodes last between 4 and 7 frames.
    test.assertTrue(episode_start[:, 0].all())
    for world in range(WORLD_COUNT):
        starts = torch.nonzero(episode_start[world]).flatten()
        lengths = starts.diff()
        test.assertTrue(((lengths >= 4) & (lengths <= 7)).all())
        test.assertLessEqual(frame_count - int(starts[-1]), 7)

    # A second simulation reaches frame t + 1 from frame t, except where a new episode begins.
    replay = _arm_simulation(device, solver)
    restarts = 0
    for t in range(frame_count - 1):
        replay.restart({name: frames[name][:, t] for name in ("joint_q", "joint_qd")})
        replay.step({"joint_f": frames["joint_f"][:, t]})
        reached = replay.binding.read_state(replay.state)
        for name in ("joint_q", "joint_qd"):
            follows = torch.isclose(reached[name], frames[name][:, t + 1], rtol=1e-5, atol=1e-5).all(dim=1)
            test.assertTrue(torch.equal(follows, ~episode_start[:, t + 1]), f"{name} at frame {t + 1}")
        restarts += int(episode_start[:, t + 1].sum())
    test.assertGreaterEqual(restarts, 2 * WORLD_COUNT)


def test_diverged_worlds_restart(test: TestTrajectoryRecorder, device: torch.device, solver: str) -> None:
    """Restart a world whose state stops being finite or grows beyond bounds, and mark its frame invalid."""
    simulation = _arm_simulation(device, solver)
    # World 2 is driven with forces that are not numbers in frame 5, world 0 with enormous forces in frame 8.
    commands = _CommandLogged({(5, 2): float("nan"), (8, 0): 1.0e9})
    recorder = TrajectoryRecorder(
        simulation, _sources(simulation), states=_near_rest(simulation), commands=commands, episode_frames=(50, 50)
    )
    arrays = recorder.record(test.folder / "data", 12).arrays(0)

    valid = np.ones((WORLD_COUNT, 12), dtype=bool)
    valid[2, 5] = valid[0, 8] = False
    np.testing.assert_array_equal(arrays["valid"], valid)
    episode_start = np.zeros((WORLD_COUNT, 12), dtype=bool)
    episode_start[:, 0] = episode_start[2, 6] = episode_start[0, 9] = True
    np.testing.assert_array_equal(arrays["episode_start"], episode_start)
    # Every recorded state is sound, since a frame holds the state its step starts from.
    for name in ("joint_q", "joint_qd"):
        test.assertTrue(np.isfinite(arrays[name]).all())
        test.assertLess(np.abs(arrays[name]).max(), 100.0)
    test.assertTrue(np.isnan(arrays["joint_f"][2, 5]).all())
    test.assertTrue(np.isfinite(arrays["joint_f"][valid]).all())


def test_independent_samples(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Start every frame of every world from a newly sampled state when episodes last one frame."""
    simulation = _arm_simulation(device)
    recorder = TrajectoryRecorder(simulation, _sources(simulation), episode_frames=(1, 1), seed=5)
    arrays = recorder.record(test.folder / "data", 6).arrays(0)
    test.assertTrue(arrays["episode_start"].all())
    test.assertTrue(arrays["valid"].all())

    # The default sampler is a Sobol sequence over the state bounds, drawn for all worlds at every frame.
    sampler = SamplerSobol(simulation.state_bounds(), seed=5)
    for frame in range(6):
        expected = sampler.sample(WORLD_COUNT)
        for name in ("joint_q", "joint_qd"):
            np.testing.assert_allclose(arrays[name][:, frame], expected[name].cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_settle_frame_count(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Mark the frames of settling invalid and begin the episode, of its full length, after them."""
    simulation = _arm_simulation(device)
    recorder = TrajectoryRecorder(
        simulation, _sources(simulation), states=_near_rest(simulation), episode_frames=(4, 4), settle_frame_count=3
    )
    arrays = recorder.record(test.folder / "data", 21).arrays(0)
    frame = np.arange(21) % 7  # three frames of settling, then four of the episode
    np.testing.assert_array_equal(arrays["valid"], np.broadcast_to(frame >= 3, (WORLD_COUNT, 21)))
    np.testing.assert_array_equal(arrays["episode_start"], np.broadcast_to(frame == 3, (WORLD_COUNT, 21)))


def test_broken_world_settles_again(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Restart a world that breaks in the middle of an episode with the frames of settling and a full episode."""
    simulation = _arm_simulation(device)
    commands = _CommandLogged({(4, 1): float("nan"), (15, 2): float("nan")})  # in an episode, and while settling
    recorder = TrajectoryRecorder(
        simulation,
        _sources(simulation),
        states=_near_rest(simulation),
        commands=commands,
        episode_frames=(4, 4),
        settle_frame_count=3,
    )
    dataset = recorder.record(test.folder / "data", 21, shard_frame_count=8)
    flags = {
        name: np.concatenate([dataset.arrays(shard)[name] for shard in range(3)], axis=1)
        for name in ("valid", "episode_start")
    }
    frame = np.arange(21)
    # Worlds that never break settle for three frames and run for four, over and over.
    for world in (0, 3):
        np.testing.assert_array_equal(flags["valid"][world], frame % 7 >= 3)
        np.testing.assert_array_equal(flags["episode_start"][world], frame % 7 == 3)
    # World 1 breaks at the second frame of its first episode, frame 4, and starts over at frame 5.
    np.testing.assert_array_equal(flags["valid"][1], (frame == 3) | ((frame >= 5) & ((frame - 5) % 7 >= 3)))
    np.testing.assert_array_equal(flags["episode_start"][1], (frame == 3) | ((frame >= 5) & ((frame - 5) % 7 == 3)))
    # World 2 breaks at frame 15, the second frame of its third settling, and settles anew from frame 16.
    np.testing.assert_array_equal(flags["valid"][2], ((frame < 14) & (frame % 7 >= 3)) | (frame >= 19))
    np.testing.assert_array_equal(flags["episode_start"][2], ((frame < 14) & (frame % 7 == 3)) | (frame == 19))


def test_records_for_a_model(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Take the model to record for, ask it for its sources, and record what a list of the same sources records."""
    datasets = []
    for name in ("model", "sources"):
        simulation = _arm_simulation(device, "mujoco")
        nerd = FlashNeRD(simulation.system, backbone=make_backbone(), prediction=PredictionFreeMotion()).to(device)
        recorder = TrajectoryRecorder(
            simulation,
            nerd if name == "model" else nerd.sources(simulation),
            states=_near_rest(simulation),
            episode_frames=(1, 1),
        )
        test.assertEqual(recorder.signals, nerd.signals)
        datasets.append(recorder.record(test.folder / name, 6))
    test.assertEqual(list(datasets[0].signals), ["joint_q", "joint_qd", "joint_f", "joint_qdd_free"])
    for name, values in datasets[0].arrays(0).items():
        np.testing.assert_array_equal(values, datasets[1].arrays(0)[name])


def test_failed_recording_leaves_a_dataset(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Break a recording in its third shard, see the error, and open the two shards that were complete."""

    class CommandFailing(CommandHeld):
        """Held commands that raise at their twentieth frame."""

        frame = 0

        def next(self, frames: Frames | None = None) -> Tensor:
            self.frame += 1
            if self.frame == 20:
                raise RuntimeError("the simulation broke")
            return super().next(frames)

    def recorder(commands: CommandHeld) -> TrajectoryRecorder:
        simulation = _arm_simulation(device)
        return TrajectoryRecorder(
            simulation, _sources(simulation), states=_near_rest(simulation), commands=commands, episode_frames=(4, 7)
        )

    with test.assertRaisesRegex(RuntimeError, "the simulation broke"):
        recorder(CommandFailing()).record(test.folder / "failed", 40, shard_frame_count=8)
    failed = TrajectoryDataset(test.folder / "failed")
    test.assertFalse(failed.generation["complete"])
    test.assertEqual((failed.world_count, failed.frame_count, len(failed.shards)), (WORLD_COUNT, 16, 2))
    whole = recorder(CommandHeld()).record(test.folder / "whole", 40, shard_frame_count=8)
    test.assertTrue(whole.generation["complete"])
    # The frames before the failure are those of a recording that runs to its end.
    for shard in range(2):
        for name, values in failed.arrays(shard).items():
            np.testing.assert_array_equal(values, whole.arrays(shard)[name])
    test.assertGreater(len(TrajectoryWindows(failed, 3, device=device)), 0)


def test_model_beside_the_dataset(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Save the model a recording was made for into the dataset folder, and read both back undisturbed."""
    simulation = _arm_simulation(device)
    nerd = FlashNeRD(simulation.system, backbone=make_backbone()).to(device)
    path = test.folder / "data"
    # Before the recording or after it, the file of the model is not in the way of the dataset.
    nerd.save(path / "flashnerd.pt")
    recorder = TrajectoryRecorder(simulation, nerd, states=_near_rest(simulation), episode_frames=(6, 9))
    dataset = recorder.record(path, 12)
    nerd.save(dataset.path / "flashnerd.pt")
    test.assertEqual(sorted(entry.name for entry in path.iterdir()), ["flashnerd.pt", "meta.json", "shard-00000"])

    reopened = TrajectoryDataset(path)
    test.assertEqual((reopened.signals, reopened.shards), (dataset.signals, dataset.shards))
    test.assertEqual(reopened.signals, nerd.signals)
    loaded = FlashNeRD.load(path / "flashnerd.pt", device=device)
    test.assertEqual(loaded.system.config(), nerd.system.config())
    test.assertFalse(loaded.fitted)
    # A second recording into the folder is refused for the dataset that is there, not for the model.
    with test.assertRaisesRegex(FileExistsError, "holds a dataset"):
        recorder.record(path, 12)
    test.assertEqual(len(TrajectoryWindows(path, 3, device=device)), len(TrajectoryWindows(dataset, 3, device=device)))


def test_sampler_takes_the_count(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Ask a sampler for a number of states and nothing else, at the first start and at every restart."""

    class SamplerRest(SamplerBase):
        """Every state at rest in the zero pose. It keeps the counts it was asked for."""

        def __init__(self) -> None:
            self.counts: list[int] = []

        def sample(self, count: int) -> Frames:
            self.counts.append(count)
            zeros = torch.zeros(count, 8, device=device)
            return {"joint_q": zeros, "joint_qd": zeros.clone()}

    simulation = _arm_simulation(device)
    states = SamplerRest()
    recorder = TrajectoryRecorder(simulation, _sources(simulation), states=states, episode_frames=(3, 3))
    arrays = recorder.record(test.folder / "data", 7).arrays(0)
    test.assertEqual(states.counts, [WORLD_COUNT] * 3)  # at the start and after the frames 2 and 5
    test.assertTrue((arrays["joint_q"][:, [0, 3, 6]] == 0.0).all())


def test_floating_bases_start_in_their_own_worlds(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Record worlds that are spaced apart and expect every floating base to start around its own world's place,
    with the default sampler and with one that draws around the pose of the first world."""
    model = fixtures.quadruped_model(3, device)
    system = SystemArticulation.from_model(model, "quadruped")
    for sampled in (False, True):
        simulation = SimulationNewton(model, newton.solvers.SolverMuJoCo(model, njmax=128), system, frame_dt=FRAME_DT)
        rest = simulation.binding.read_state(model.state())["joint_q"].clone()
        test.assertGreater((rest[1:, 0] - rest[0, 0]).min().item(), 1.0)  # the worlds of the fixture are spaced
        states = SamplerFloatingBase(rest[0], position=((-0.1, -0.1, 0.0), (0.1, 0.1, 0.0))) if sampled else None
        TrajectoryRecorder(simulation, FlashNeRD(system).to(device), states=states)
        start = simulation.binding.read_state(simulation.state)["joint_q"]
        test.assertLessEqual((start[:, :2] - rest[:, :2]).abs().max().item(), 0.25 + 1e-6)
        # A state handed to the simulation itself is written where it says.
        simulation.restart({"joint_q": rest[:1].expand(3, -1), "joint_qd": torch.zeros(3, 18, device=device)})
        written = simulation.binding.read_state(simulation.state)["joint_q"][:, :3]
        test.assertTrue(torch.equal(written, rest[:1, :3].expand(3, -1)))


def test_windows(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Load a recorded dataset as training windows that stay inside episodes."""
    simulation = _arm_simulation(device)
    recorder = TrajectoryRecorder(
        simulation, _sources(simulation), states=_near_rest(simulation), episode_frames=(6, 6)
    )
    dataset = recorder.record(test.folder / "data", 18, shard_frame_count=7)
    windows = TrajectoryWindows(dataset, 3, device=device)
    # Three episodes of six frames per world, each with three windows of four frames.
    test.assertEqual(len(windows), WORLD_COUNT * 3 * 3)
    batch = windows.batch(torch.arange(len(windows), device=device))
    test.assertEqual(batch.frame_dt, FRAME_DT)
    test.assertEqual(batch.frames.keys(), dataset.signals.keys())
    for name, spec in dataset.signals.items():
        test.assertEqual(batch.frames[name].shape, (len(windows), 4, *spec.shape))
        test.assertEqual(batch.frames[name].dtype, torch.float32)
    test.assertEqual(int(batch.episode_start.sum()), WORLD_COUNT * 3)
    # Independent samples are single frame windows. None holds a step, since the frame after it starts anew.
    samples = TrajectoryRecorder(simulation, _sources(simulation), episode_frames=(1, 1))
    single = TrajectoryWindows(samples.record(test.folder / "samples", 6), 0)
    test.assertEqual(len(single), WORLD_COUNT * 6)
    test.assertEqual(len(single.with_length(1)), 0)


def test_free_motion_label(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Record the free acceleration, which without contacts and limits is the velocity change over the step."""
    simulation = _arm_simulation(device, "mujoco", box=True)
    prediction = PredictionFreeMotion()
    source = prediction.source(simulation)
    test.assertTrue(source.after_step)
    test.assertEqual(source.signals, prediction.signals(simulation.binding.system))
    recorder = TrajectoryRecorder(
        simulation, [*_sources(simulation), source], states=_near_rest(simulation), episode_frames=(5, 8)
    )
    dataset = recorder.record(test.folder / "data", 24)
    test.assertEqual(dataset.signals["joint_qdd_free"].role, SignalRole.LABEL)
    arrays = dataset.arrays(0)
    test.assertEqual(arrays["joint_qdd_free"].shape, (WORLD_COUNT, 24, 8))
    test.assertEqual(arrays["joint_qdd_free"].dtype, np.float32)

    continues = ~arrays["episode_start"][:, 1:]
    difference = (arrays["joint_qd"][:, 1:] - arrays["joint_qd"][:, :-1])[continues] / FRAME_DT
    label = arrays["joint_qdd_free"][:, :-1][continues]
    test.assertGreater(np.abs(label).max(), 10.0)
    np.testing.assert_allclose(label, difference, rtol=1e-4, atol=1e-3)


def test_free_motion_requirements(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Refuse to read free accelerations from a simulation that cannot provide them."""
    with test.assertRaisesRegex(NotImplementedError, "no label signals"):
        PredictionStateIncrement().source(_arm_simulation(device))
    prediction = PredictionFreeMotion()
    with test.assertRaisesRegex(ValueError, "SolverFeatherstone"):
        prediction.source(_arm_simulation(device, "featherstone"))
    if device.type == "cpu":
        model = fixtures.arm_model(1, device)  # the CPU version of MuJoCo steps one world
        system = SystemArticulation.from_model(model, control=("joint_f",))
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
        with test.assertRaisesRegex(ValueError, "on the CPU"):
            prediction.source(SimulationNewton(model, solver, system, frame_dt=FRAME_DT))

    model = fixtures.arm_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, control=("joint_f",))
    solver = newton.solvers.SolverMuJoCo(model, integrator="rk4")
    with test.assertRaisesRegex(ValueError, "RK4"):
        prediction.source(SimulationNewton(model, solver, system, frame_dt=FRAME_DT))
    solver = newton.solvers.SolverMuJoCo(model)
    with test.assertRaisesRegex(ValueError, "one solver step per frame"):
        prediction.source(SimulationNewton(model, solver, system, frame_dt=FRAME_DT, substep_count=2))

    model = fixtures.quadruped_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model)
    solver = newton.solvers.SolverMuJoCo(model, njmax=128)
    with test.assertRaisesRegex(ValueError, "fixed base"):
        prediction.source(SimulationNewton(model, solver, system, frame_dt=FRAME_DT))


def test_refuses_faulty_sources(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Refuse a source that returns another dtype, shape or set of signals than it declares, when built."""
    simulation = _arm_simulation(device)
    state, control = _sources(simulation)

    def faulty(change: Callable[[Frames], Frames]) -> list[SignalSource]:
        return [SignalSource(state.signals, lambda: change(state.read())), control]

    with test.assertRaisesRegex(ValueError, r"'joint_qd' returns torch.float64 \[4, 8\].*torch.float32 \[4, 8\]"):
        TrajectoryRecorder(simulation, faulty(lambda frames: {**frames, "joint_qd": frames["joint_qd"].double()}))
    with test.assertRaisesRegex(ValueError, r"'joint_q' returns torch.float32 \[8\].*torch.float32 \[4, 8\]"):
        TrajectoryRecorder(simulation, faulty(lambda frames: {**frames, "joint_q": frames["joint_q"][0]}))
    with test.assertRaisesRegex(
        ValueError, r"declares the signals \['joint_q', 'joint_qd'\] and returns \['joint_q'\]"
    ):
        TrajectoryRecorder(simulation, faulty(lambda frames: {"joint_q": frames["joint_q"]}))


def test_contact_set_needs_collision_pipeline(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Record a model with a contact set from a simulation that detects no contacts and expect the remedy named."""
    model = fixtures.quadruped_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model)
    nerd = FlashNeRD(system, backbone=make_backbone(), contact_input=ContactInputSet.from_system(system))
    nerd.to(device)

    def simulation(pipeline: newton.CollisionPipeline | None) -> SimulationNewton:
        solver = newton.solvers.SolverFeatherstone(model)
        return SimulationNewton(model, solver, system, frame_dt=FRAME_DT, collision_pipeline=pipeline)

    blind = simulation(None)
    remedy = r"the simulation detects none.*collision_pipeline=newton\.CollisionPipeline\(model\)"
    with test.assertRaisesRegex(ValueError, "ContactInputSet reads the contacts.*" + remedy):
        TrajectoryRecorder(blind, nerd)
    sighted = simulation(newton.CollisionPipeline(model))
    test.assertIn("contact_rows", TrajectoryRecorder(sighted, nerd).signals)

    # What needs the pipeline is an input that reads detected contacts. One that senses the state alone, be it
    # recomputable or not, records from a simulation without one.
    for contact_input in (ContactInputRootHeight(), ContactInputRootHeightRecorded()):
        test.assertFalse(contact_input.reads_contacts)
        sensing = FlashNeRD(system, backbone=make_backbone(), contact_input=contact_input).to(device)
        test.assertIn("root_height", TrajectoryRecorder(blind, sensing).signals)


def test_refuses_states_beyond_limit(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Refuse starting states that count as diverged, which would leave every recorded frame invalid."""
    simulation = _arm_simulation(device)
    one = torch.ones(8, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    far = SamplerUniform({"joint_q": (-one, one), "joint_qd": (2.0e3 * one, 3.0e3 * one)}, generator=generator)
    with test.assertRaisesRegex(ValueError, "4 of 4 worlds start from a state"):
        TrajectoryRecorder(simulation, _sources(simulation), states=far)


def test_refuses_bad_arguments(test: TestTrajectoryRecorder, device: torch.device) -> None:
    """Refuse a signal that two sources provide, episodes without frames and commands without finite bounds."""
    simulation = _arm_simulation(device)
    sources = _sources(simulation)
    with test.assertRaisesRegex(ValueError, "more than one source"):
        TrajectoryRecorder(simulation, [*sources, sources[0]])
    for options in ({"episode_frames": (0, 4)}, {"episode_frames": (5, 4)}, {"settle_frame_count": -1}):
        with test.assertRaisesRegex(ValueError, "episode_frames is|settle_frame_count cannot"):
            TrajectoryRecorder(simulation, sources, **options)

    # A model without effort limits leaves the joint forces unbounded, until the command brings its bounds.
    model = fixtures.arm_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, control=("joint_f",))
    unbounded = SimulationNewton(model, newton.solvers.SolverFeatherstone(model), system, frame_dt=FRAME_DT)
    with test.assertRaisesRegex(ValueError, "finite bounds"):
        TrajectoryRecorder(unbounded, _sources(unbounded))
    recorder = TrajectoryRecorder(unbounded, _sources(unbounded), commands=CommandHeld(low=-1.0, high=1.0))
    arrays = recorder.record(test.folder / "data", 4).arrays(0)
    test.assertLessEqual(np.abs(arrays["joint_f"]).max(), 1.0)


for _solver in ("mujoco", "featherstone"):
    for _function in (test_frames_consecutive, test_diverged_worlds_restart):
        add_function_test(TestTrajectoryRecorder, f"{_function.__name__}_{_solver}", _function, solver=_solver)
for _function in (
    test_dataset_layout,
    test_independent_samples,
    test_settle_frame_count,
    test_broken_world_settles_again,
    test_records_for_a_model,
    test_failed_recording_leaves_a_dataset,
    test_model_beside_the_dataset,
    test_sampler_takes_the_count,
    test_floating_bases_start_in_their_own_worlds,
    test_windows,
    test_free_motion_label,
    test_free_motion_requirements,
    test_refuses_faulty_sources,
    test_contact_set_needs_collision_pipeline,
    test_refuses_states_beyond_limit,
    test_refuses_bad_arguments,
):
    add_function_test(TestTrajectoryRecorder, _function.__name__, _function)


if __name__ == "__main__":
    unittest.main()
