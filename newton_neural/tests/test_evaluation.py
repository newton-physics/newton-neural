# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``rollout_error``, ``audit_rest``, and ``audit_paired`` against analytical references."""

from __future__ import annotations

import itertools
import math
import unittest
from unittest import mock

import newton
import numpy as np
import torch

from newton_neural import Frames
from newton_neural._src.contacts.contact_input import ContactInputBase
from newton_neural._src.core import rotation
from newton_neural._src.evaluation import rollout
from newton_neural.data import (
    CommandHeld,
    SamplerBase,
    SimulationNewton,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.evaluation import RolloutStatistics, audit_paired, audit_rest, rollout_error
from newton_neural.models import FlashNeRD, NeRDBase, NeRDStepper, Precision
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.tests.extensions import BackboneLeaky, ContactInputRootHeight, ContactInputRootHeightRecorded
from newton_neural.tests.helpers import (
    FRAME_DT,
    SUBSTEP_COUNT,
    frame_of,
    make_arm_simulation,
    make_backbone,
    make_nerd,
    make_system,
    near_rest,
)
from newton_neural.tests.unittest_utils import FolderTestCase, add_function_test
from newton_neural.training import LossOneStep, create_optimizer

PRIME_COUNT = 2
"""Frames that fill the history of a model with the backbone of ``make_backbone``, its receptive field less one."""


def _quadruped_simulation(device: torch.device, world_count: int) -> SimulationNewton:
    """The quadruped falling freely, stepped by Featherstone's algorithm without contacts."""
    model = fixtures.quadruped_model(world_count, device)
    system = SystemArticulation.from_model(model, "quadruped")
    solver = newton.solvers.SolverFeatherstone(model)
    return SimulationNewton(model, solver, system, frame_dt=FRAME_DT, substep_count=SUBSTEP_COUNT)


def _nerd(simulation: SimulationNewton, contact_input: ContactInputBase | None = None) -> FlashNeRD:
    """A small untrained model of the simulation's system, on its device.

    It is deployed in IEEE float32 unless a rollout is given another precision, so that the errors of its
    rollouts are those of the eager model stepped by hand.
    """
    torch.manual_seed(0)
    nerd = FlashNeRD(simulation.system, backbone=make_backbone(), contact_input=contact_input, readout_hidden_dim=32)
    nerd.precision = Precision.FP32
    return nerd.to(simulation.device)


def _held_commands() -> CommandHeld:
    """Position targets within half a radian (or meter) of zero, each held for 0.4 to 1.2 s."""
    return CommandHeld((20, 60), low=-0.5, high=0.5)


class TestEvaluation(FolderTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(torch.compiler.reset)

    def record(
        self,
        simulation: SimulationNewton,
        nerd: NeRDBase,
        frame_count: int,
        states: SamplerBase | None = None,
    ) -> TrajectoryWindows:
        """Record episodes of 60 to 100 frames under held targets and fit the model's normalization to them.

        Args:
            simulation: The simulation to record.
            nerd: The model whose signals are recorded.
            frame_count: Frames to record in every world.
            states: Where episodes start. Anywhere within the simulation's state bounds by default.

        Returns:
            Windows of 4 steps over the recording, on the simulation's device.
        """
        recorder = TrajectoryRecorder(
            simulation, nerd, states=states, commands=_held_commands(), episode_frames=(60, 100)
        )
        dataset = recorder.record(self.folder / "data", frame_count)
        windows = TrajectoryWindows(dataset, 4, device=simulation.device)
        nerd.fit_normalization(windows.loader(min(256, len(windows))))
        return windows


def test_error_floating_base(test: TestEvaluation, device: torch.device) -> None:
    """Compare base orientations as one rotation angle whatever the sign of the quaternions."""
    system = make_system("quadruped").to(device)
    reference = fixtures.random_frames(system, 5, 3, device=device)
    turn = torch.tensor([0.0, 0.0, math.sin(0.35), math.cos(0.35)], device=device)  # 0.7 rad about z
    other = {name: value.clone() for name, value in reference.items()}
    other["joint_q"][..., 3:7] = -rotation.multiply(turn, reference["joint_q"][..., 3:7])
    other["joint_q"][..., 0] += 0.25
    other["joint_q"][..., 9] -= 0.5
    other["joint_qd"][..., 4] += 2.0

    error = system.error(reference, other)
    test.assertEqual(list(error), ["joint_q", "joint_qd"])
    test.assertEqual(error["joint_q"].shape, (5, 3, system.joint_coord_count - 3))
    test.assertEqual(error["joint_qd"].shape, (5, 3, system.joint_dof_count))
    expected = torch.zeros(system.joint_coord_count - 3, device=device)
    expected[0], expected[3], expected[6] = 0.25, 0.7, 0.5  # position x, the angle, joint coordinate 9
    test.assertTrue(torch.allclose(error["joint_q"], expected.expand(5, 3, -1), atol=1e-4))
    test.assertTrue(torch.allclose(error["joint_qd"].sum(dim=-1), torch.full((5, 3), 2.0, device=device), atol=1e-5))


def _assert_rollouts_by_hand(
    test: TestEvaluation,
    nerd: NeRDBase,
    frames: Frames,
    prime_count: int,
    statistics: dict[int, RolloutStatistics],
) -> None:
    """Step the model by hand through windows and expect the statistics that ``rollout_error`` reported.

    Args:
        test: The test case that asserts.
        nerd: The model that was rolled out.
        frames: The windows that were measured, ``[rollout_count, prime_count + max(horizons) + 1, *shape]``.
        prime_count: Recorded frames at the start of each window that fill the history. None for a rollout
            from the start of an episode.
        statistics: What ``rollout_error`` returned for the horizons 1 to the last, at least.
    """
    count = len(next(iter(frames.values())))
    system, scale = nerd.system, nerd.output_normalizer.std
    with torch.no_grad():
        history = nerd.history(count)
        for frame in range(prime_count):
            nerd.predict(frame_of(frames, frame), history)
        state, squares = frame_of(frames, prime_count), torch.zeros(count, device=nerd.device)
        for horizon in range(1, max(statistics) + 1):
            current = frame_of(frames, prime_count + horizon - 1) | state
            state = nerd.step(current, history)
            reference = frame_of(frames, prime_count + horizon)
            squares += (system.increment(reference, state) / scale).square().mean(dim=-1)
            if horizon not in statistics:
                continue
            result = statistics[horizon]
            test.assertAlmostEqual(result.nrmse, (squares / horizon).mean().sqrt().item(), delta=1e-3 * result.nrmse)
            for name in ("joint_q", "joint_qd"):
                error = (state[name] - reference[name]).abs().flatten().sort().values
                test.assertAlmostEqual(result.mean[name], error.mean().item(), delta=1e-3 * result.mean[name])
                for quantile, value in ((0.5, result.p50), (0.9, result.p90), (0.99, result.p99)):
                    # The value of the nearest rank, which for a rank half way between two is either of them.
                    rank = quantile * (len(error) - 1)
                    below, above = error[math.floor(rank)].item(), error[math.ceil(rank)].item()
                    test.assertGreaterEqual(value[name], below * (1 - 1e-3))
                    test.assertLessEqual(value[name], above * (1 + 1e-3))
                test.assertLessEqual(result.p50[name], result.p90[name])
                test.assertLessEqual(result.p90[name], result.p99[name])
                test.assertLessEqual(result.p99[name], error.max().item() * (1 + 1e-3))


def test_rollout_matches_single_steps(test: TestEvaluation, device: torch.device) -> None:
    """Reproduce the statistics at horizons 0, 1 and 2 by stepping the model by hand from a primed history."""
    simulation = make_arm_simulation(device, 8)
    nerd = _nerd(simulation)
    windows = test.record(simulation, nerd, 100, near_rest(simulation)).with_length(PRIME_COUNT + 2)
    count = len(windows)
    test.assertGreater(count, 2)
    # Two batches, the second with one window and the rest repeated, to check that repeats are not counted.
    statistics = rollout_error(nerd, windows, horizons=(0, 1, 2), rollout_count=None, batch_size=count - 1)
    frames = windows.batch(torch.arange(count, device=device)).frames
    _assert_rollouts_by_hand(test, nerd, frames, PRIME_COUNT, statistics)

    # Linear layers in reduced precision change the errors, and only a little.
    reduced = rollout_error(nerd=nerd, windows=windows, horizons=(2,), precision=Precision.BF16)[2]
    test.assertNotEqual(reduced.nrmse, statistics[2].nrmse)
    test.assertAlmostEqual(reduced.nrmse, statistics[2].nrmse, delta=0.05 * statistics[2].nrmse)
    # The stepper of the rollouts takes the precision and the choice of compilation that the call names.
    with mock.patch.object(rollout, "NeRDStepper", wraps=NeRDStepper) as stepper:
        eager = rollout_error(nerd, windows, horizons=(2,), precision="tf32", compile=False)[2]
    test.assertEqual(stepper.call_args.kwargs, {"precision": "tf32", "compile": False})
    test.assertAlmostEqual(eager.nrmse, statistics[2].nrmse, delta=0.05 * statistics[2].nrmse)

    start = statistics[0]
    for values in (start.mean, start.p50, start.p90, start.p99):
        test.assertEqual(values, {"joint_q": 0.0, "joint_qd": 0.0})
    test.assertEqual(start.nrmse, 0.0)


def test_rollout_count(test: TestEvaluation, device: torch.device) -> None:
    """Measure the asked number of rollouts, spread evenly over the windows, and all of them when there are fewer."""
    simulation = make_arm_simulation(device, 8)
    nerd = _nerd(simulation)
    windows = test.record(simulation, nerd, 100, near_rest(simulation)).with_length(PRIME_COUNT + 4)
    count = len(windows)
    test.assertGreater(count, 400)
    every = rollout_error(nerd, windows, horizons=(4,), rollout_count=None)
    for rollout_count in (1, 7, 100):
        some = rollout_error(nerd, windows, horizons=(4,), rollout_count=rollout_count, batch_size=64)
        chosen = torch.arange(rollout_count, device=device) * count // rollout_count
        _assert_rollouts_by_hand(test, nerd, windows.batch(chosen).frames, PRIME_COUNT, some)
        test.assertNotEqual(some[4].nrmse, every[4].nrmse)
    # As many rollouts as there are windows, or more, are all of them, which is also what the default gives here.
    for rollout_count in (count, count + 1, 10 * count):
        test.assertEqual(rollout_error(nerd, windows, horizons=(4,), rollout_count=rollout_count), every)
    test.assertEqual(rollout_error(nerd, windows, horizons=(4,)), every)
    with test.assertRaisesRegex(ValueError, "rollout_count must be 1 or more, or None for every rollout, got 0"):
        rollout_error(nerd, windows, horizons=(4,), rollout_count=0)


def test_rollout_from_episode_start(test: TestEvaluation, device: torch.device) -> None:
    """Start every rollout at the first frame of an episode with an empty history, where no frame primes it."""
    simulation = make_arm_simulation(device, 8)
    nerd = _nerd(simulation)
    windows = test.record(simulation, nerd, 200, near_rest(simulation))
    horizons = (1, 6)
    from_start = rollout_error(nerd, windows, horizons=horizons, from_episode_start=True, rollout_count=None)

    # By hand, from the windows of six steps that begin an episode, without a frame before the first step.
    cut = windows.with_length(6)
    starts = torch.nonzero(cut.episode_start).flatten()
    test.assertGreater(len(starts), 8)  # more than the first episode of every world
    test.assertLess(len(starts), len(cut))
    frames = cut.batch(starts).frames
    _assert_rollouts_by_hand(test, nerd, frames, 0, from_start)
    # A full history gives other predictions for other frames.
    anywhere = rollout_error(nerd, windows, horizons=horizons, rollout_count=None)
    test.assertNotEqual(from_start[1].nrmse, anywhere[1].nrmse)

    # A sample of the starts, spread evenly over them.
    some = rollout_error(nerd, windows, horizons=horizons, from_episode_start=True, rollout_count=5)
    chosen = starts[torch.arange(5, device=device) * len(starts) // 5]
    _assert_rollouts_by_hand(test, nerd, cut.batch(chosen).frames, 0, some)

    # Nothing has to prime a model with unbounded history at the start of an episode.
    unbounded = FlashNeRD(simulation.system, backbone=BackboneLeaky()).to(device)
    unbounded.fit_normalization(windows.loader(256))
    leaky = rollout_error(unbounded, windows, horizons=horizons, from_episode_start=True, rollout_count=None)
    _assert_rollouts_by_hand(test, unbounded, frames, 0, leaky)
    with test.assertRaisesRegex(
        ValueError, r"no episode of 151 valid frames.*150 frames after the start of an episode"
    ):
        rollout_error(nerd, windows, horizons=(150,), from_episode_start=True)


def test_rollout_error_falls_with_training(test: TestEvaluation, device: torch.device) -> None:
    """Train a small model on the driven arm and expect errors that fall with training and grow with the horizon."""
    simulation = make_arm_simulation(device, 48)
    nerd = _nerd(simulation)
    train, held_out = test.record(simulation, nerd, 200, near_rest(simulation)).split(0.25)
    horizons = (1, 8, 32)
    before = rollout_error(nerd, held_out, horizons=horizons, batch_size=512)

    loss, optimizer = LossOneStep(), create_optimizer(nerd, lr=3e-3)
    batches = train.loader(128, epochs=None, generator=torch.Generator(device=device).manual_seed(0))
    for batch in itertools.islice(batches, 200):
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()
    after = rollout_error(nerd, held_out, horizons=horizons, batch_size=512)

    for horizon in horizons:
        test.assertLess(after[horizon].nrmse, before[horizon].nrmse)
        for name in ("joint_q", "joint_qd"):
            test.assertLess(after[horizon].p50[name], 0.6 * before[horizon].p50[name])
            test.assertLess(after[horizon].p50[name], after[horizon].p90[name])
            test.assertLess(after[horizon].p90[name], after[horizon].p99[name])
    for short, long in ((1, 8), (8, 32)):
        test.assertLess(after[short].nrmse, after[long].nrmse)
        for name in ("joint_q", "joint_qd"):
            test.assertLess(after[short].p50[name], after[long].p50[name])
            test.assertLess(after[short].mean[name], after[long].mean[name])


def test_rollout_recomputes_contact(test: TestEvaluation, device: torch.device) -> None:
    """Evaluate a recomputable contact input at the predicted states and keep the recording of any other."""
    simulation = _quadruped_simulation(device, 8)
    recomputing = _nerd(simulation, ContactInputRootHeight())
    with torch.no_grad():  # let the height weigh on the predictions of the untrained model
        recomputing.contact_encoder.project.weight.mul_(8.0)
    windows = test.record(simulation, recomputing, 100)
    recorded = _nerd(simulation, ContactInputRootHeightRecorded())
    recorded.load_state_dict(recomputing.state_dict())
    recorded.frame_dt = recomputing.frame_dt

    first = rollout_error(recomputing, windows, horizons=(1, 16))
    second = rollout_error(recorded, windows, horizons=(1, 16))
    # The first step starts from a recorded state, whose contact signals both models agree on.
    test.assertAlmostEqual(first[1].nrmse, second[1].nrmse, delta=1e-4 * first[1].nrmse)
    test.assertGreater(abs(first[16].nrmse - second[16].nrmse), 1e-2 * first[16].nrmse)
    test.assertEqual(list(first[16].p50), ["joint_q", "joint_qd"])


def test_rollout_refuses_bad_arguments(test: TestEvaluation, device: torch.device) -> None:
    """Refuse models that cannot roll out alone or were never fitted, bad horizons, other frame durations and
    short episodes."""
    simulation = make_arm_simulation(device, 4)
    nerd = _nerd(simulation)
    windows = test.record(simulation, nerd, 100, near_rest(simulation))
    with test.assertRaisesRegex(ValueError, "horizons"):
        rollout_error(nerd, windows, horizons=())
    with test.assertRaisesRegex(ValueError, "horizons"):
        rollout_error(nerd, windows, horizons=(4, -1))
    with test.assertRaisesRegex(
        ValueError, r"no episode of 103 valid frames.*100 frames after 2 frames of history.*shorter.*episode_frames"
    ):
        rollout_error(nerd, windows, horizons=(100,))
    with test.assertRaisesRegex(ValueError, "next state"):
        rollout_error(make_nerd("arm", device, prediction=PredictionFreeMotion()), windows)
    # A model that was never fitted is no trained model, with or without a frame duration.
    unbounded = FlashNeRD(simulation.system, backbone=BackboneLeaky()).to(device)
    timed = FlashNeRD(simulation.system, backbone=make_backbone(), frame_dt=FRAME_DT).to(device)
    for unfitted in (unbounded, timed):
        with test.assertRaisesRegex(ValueError, "never fitted.*Pass the trained model"):
            rollout_error(unfitted, windows)
    unbounded.fit_normalization(windows.loader(256))
    with test.assertRaisesRegex(ValueError, "unbounded history.*from_episode_start=True"):
        rollout_error(unbounded, windows)
    nerd.frame_dt = 2 * FRAME_DT
    with test.assertRaisesRegex(ValueError, "frames of"):
        rollout_error(nerd, windows, horizons=(1,))


def test_audit_rest(test: TestEvaluation, device: torch.device) -> None:
    """Report almost no motion for an arm that holds its targets and a clear drift for one that falls."""
    holding = make_arm_simulation(device, 4)
    for _ in range(600):  # let the arm sag into the pose its springs hold against gravity
        holding.step()
    rest = audit_rest(holding, frame_count=50)
    falling = audit_rest(make_arm_simulation(device, 4, driven=False), frame_count=50)

    for name in ("joint_q", "joint_qd"):
        test.assertLess(rest.drift_max[name], 1e-3)
        test.assertLess(rest.vibration_max[name], 1e-3)
        test.assertGreater(falling.drift_mean[name], 0.1)
        test.assertGreater(falling.vibration_mean[name], 100 * rest.vibration_max[name])
        for audit in (rest, falling):
            test.assertLessEqual(audit.drift_mean[name], audit.drift_max[name])
            test.assertLessEqual(audit.vibration_mean[name], audit.vibration_max[name])
    with test.assertRaisesRegex(ValueError, "frame_count"):
        audit_rest(holding, frame_count=0)


def test_audit_rest_matches_line_fit(test: TestEvaluation, device: torch.device) -> None:
    """Reproduce drift and vibration of a falling arm from its frames with a least squares line."""
    frame_count = 20
    audit = audit_rest(make_arm_simulation(device, 3, driven=False), frame_count=frame_count)

    simulation = make_arm_simulation(device, 3, driven=False)
    time = np.arange(frame_count + 1)
    frames: dict[str, list[np.ndarray]] = {"joint_q": [], "joint_qd": []}
    for _ in time:
        for name, value in simulation.binding.read_state(simulation.state).items():
            # A copy, since the simulation goes on writing the arrays it hands out. [worlds * coordinates]
            frames[name].append(value.clone().flatten().cpu().numpy())
        simulation.step()
    for name, values in frames.items():
        series = np.stack(values).astype(np.float64)
        line = np.polynomial.polynomial.polyval(time, np.polynomial.polynomial.polyfit(time, series, 1)).T
        vibration = np.sqrt(np.mean((series - line) ** 2, axis=0))
        drift = np.abs(series[-1] - series[0])
        test.assertAlmostEqual(audit.drift_max[name], drift.max(), delta=1e-4 * drift.max())
        test.assertAlmostEqual(audit.drift_mean[name], drift.mean(), delta=1e-4 * drift.mean())
        test.assertAlmostEqual(audit.vibration_max[name], vibration.max(), delta=1e-3 * vibration.max())
        test.assertAlmostEqual(audit.vibration_mean[name], vibration.mean(), delta=1e-3 * vibration.mean())


def test_audit_rest_free_fall(test: TestEvaluation, device: torch.device) -> None:
    """Report what a quadruped gains in free fall as drift, and its steady gain of speed as no vibration."""
    frame_count = 20
    audit = audit_rest(_quadruped_simulation(device, 3), frame_count=frame_count)
    duration, step = frame_count * FRAME_DT, FRAME_DT / SUBSTEP_COUNT
    speed = 9.81 * duration
    # The solver adds the change of the velocity before it moves the base, which falls one step further for it.
    distance = 0.5 * speed * (duration + step)
    test.assertAlmostEqual(audit.drift_max["joint_qd"], speed, delta=1e-3 * speed)
    test.assertAlmostEqual(audit.drift_max["joint_q"], distance, delta=1e-3 * distance)
    test.assertLess(audit.vibration_max["joint_qd"], 1e-4)
    test.assertGreater(audit.vibration_max["joint_q"], 0.01)


class _Replay:
    """States written by hand, handed out one per step as a simulation would."""

    def __init__(self, system: SystemArticulation, states: list[Frames]) -> None:
        self.system, self.device = system, states[0]["joint_q"].device
        self.binding, self.state = self, None
        self._states, self._frame = states, 0

    def read_state(self, state: None) -> Frames:
        return self._states[self._frame]

    def step(self, command: Frames | None = None) -> None:
        self._frame += 1


def test_audit_rest_quaternion_sign(test: TestEvaluation, device: torch.device) -> None:
    """Report the heading jitter of a floating base, whichever sign its quaternion takes from frame to frame."""
    system = make_system("quadruped")
    frame_count, jitter = 50, 1.0e-4
    start = fixtures.random_frames(system, 2, device=device, seed=0)
    for heading in (0.0, math.pi):  # at half a turn w crosses zero, and the quaternion with w >= 0 changes sign
        states = []
        for frame in range(frame_count + 1):
            half = 0.5 * (heading + jitter * (1.0 if frame % 2 else -1.0))
            joint_q = start["joint_q"].clone()
            joint_q[:, 3:7] = rotation.canonicalize(torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)]))
            states.append({"joint_q": joint_q, "joint_qd": start["joint_qd"]})
        audit = audit_rest(_Replay(system, states), frame_count=frame_count)
        test.assertAlmostEqual(audit.vibration_max["joint_q"], jitter, delta=0.05 * jitter)
        test.assertLess(audit.vibration_max["joint_qd"], 1e-6)


def test_audit_paired(test: TestEvaluation, device: torch.device) -> None:
    """Report no difference between identical simulations and a growing one for another solver setting."""
    frame_count = 40
    torch.manual_seed(0)
    reference = make_arm_simulation(device, 4)
    for _ in range(10):  # the candidate must take over a state that is not the one both were built with
        reference.step()
    same = audit_paired(reference, make_arm_simulation(device, 4), commands=_held_commands(), frame_count=frame_count)
    # A solver that renews its mass matrix at every eighth step only moves the arm a little differently.
    coarse = make_arm_simulation(device, 4, update_mass_matrix_interval=8)
    other = audit_paired(reference, coarse, commands=_held_commands(), frame_count=frame_count)

    for name in ("joint_q", "joint_qd"):
        for audit in (same, other):
            test.assertEqual(audit.mean[name].shape, (frame_count,))
            test.assertEqual(audit.max[name].shape, (frame_count,))
            test.assertEqual(audit.mean[name].device.type, device.type)
            test.assertTrue(bool((audit.mean[name] <= audit.max[name]).all()))
        test.assertLessEqual(same.max[name].max().item(), 1e-6)
        test.assertGreater(other.mean[name][0].item(), 0.0)
    test.assertGreater(other.mean["joint_q"][-1].item(), 10 * other.mean["joint_q"][0].item())
    # Without a command the worlds are driven as a recorder drives them by default.
    default = audit_paired(reference, make_arm_simulation(device, 4), frame_count=4)
    test.assertLessEqual(default.max["joint_q"].max().item(), 1e-6)


def test_audit_paired_refuses_bad_arguments(test: TestEvaluation, device: torch.device) -> None:
    """Refuse simulations of other systems or world counts, no frames, and commands without finite bounds."""
    reference = make_arm_simulation(device, 4)
    with test.assertRaisesRegex(ValueError, "worlds"):
        audit_paired(reference, make_arm_simulation(device, 2), commands=_held_commands(), frame_count=4)
    with test.assertRaisesRegex(ValueError, "signals"):
        audit_paired(reference, _quadruped_simulation(device, 4), commands=_held_commands(), frame_count=4)
    with test.assertRaisesRegex(ValueError, "frame_count"):
        audit_paired(reference, make_arm_simulation(device, 4), commands=_held_commands(), frame_count=0)

    model = fixtures.arm_model(4, device)
    system = SystemArticulation.from_model(model, "arm", control=("joint_f",))
    unbounded = SimulationNewton(model, newton.solvers.SolverFeatherstone(model), system, frame_dt=FRAME_DT)
    with test.assertRaisesRegex(ValueError, "finite bounds"):
        audit_paired(unbounded, unbounded, commands=CommandHeld(), frame_count=4)


for _function in (
    test_error_floating_base,
    test_rollout_matches_single_steps,
    test_rollout_count,
    test_rollout_from_episode_start,
    test_rollout_error_falls_with_training,
    test_rollout_recomputes_contact,
    test_rollout_refuses_bad_arguments,
    test_audit_rest,
    test_audit_rest_matches_line_fit,
    test_audit_rest_free_fall,
    test_audit_rest_quaternion_sign,
    test_audit_paired,
    test_audit_paired_refuses_bad_arguments,
):
    add_function_test(TestEvaluation, _function.__name__, _function)


if __name__ == "__main__":
    unittest.main()
