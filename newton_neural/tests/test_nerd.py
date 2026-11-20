# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``NeRDBase`` family-agnostic contract across ``FlashNeRD`` and external ``MinimalNeRD``."""

from __future__ import annotations

import unittest

import newton
import torch

from newton_neural import SignalRole, SignalSpec
from newton_neural.contacts import ContactInputProbes
from newton_neural.data import (
    CommandHeld,
    SimulationNewton,
    TrajectoryBatch,
    TrajectoryRecorder,
    TrajectoryWindows,
)
from newton_neural.evaluation import rollout_error
from newton_neural.models import FlashNeRD, NeRDBase, NeRDStepper
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.solvers import SolverNeRD
from newton_neural.testing import check_nerd_streaming
from newton_neural.tests import fixtures
from newton_neural.tests.extensions import BackboneLeaky, ContactInputRootHeight, MinimalNeRD
from newton_neural.tests.helpers import (
    FRAME_DT,
    NERDS,
    WORLD_COUNT,
    assert_frames_close,
    frame_of,
    make_arm_simulation,
    make_backbone,
    make_frames,
    make_nerd,
    make_system,
    near_rest,
)
from newton_neural.tests.unittest_utils import FolderTestCase, add_function_test
from newton_neural.training import LossOneStep, create_optimizer


def _recorded_windows(test: TestNeRD, simulation: SimulationNewton, nerd: NeRDBase) -> TrajectoryWindows:
    """Windows of a small dataset recorded for ``nerd`` from the arm, from which the model takes its statistics."""
    commands = CommandHeld((20, 60), low=-0.5, high=0.5)
    states = near_rest(simulation)
    recorder = TrajectoryRecorder(simulation, nerd, states=states, commands=commands, episode_frames=(40, 64))
    dataset = recorder.record(test.folder / "data", 64)
    test.assertEqual(dataset.signals, nerd.signals)
    windows = TrajectoryWindows(dataset, 8, device=simulation.device)
    nerd.fit_normalization(windows.loader(256))
    test.assertTrue(nerd.fitted)
    test.assertEqual(nerd.frame_dt, FRAME_DT)
    return windows


def _lower_the_loss(test: TestNeRD, nerd: NeRDBase, batch: TrajectoryBatch) -> None:
    """Take a few steps of a plain training loop and expect a lower loss."""
    loss, optimizer = LossOneStep(), create_optimizer(nerd, "adamw", lr=1e-2)
    with torch.no_grad():
        before = loss(nerd, batch).item()
    for _ in range(20):
        optimizer.zero_grad()
        loss(nerd, batch).backward()
        optimizer.step()
    with torch.no_grad():
        test.assertLess(loss(nerd, batch).item(), before)


def test_minimal_family_through_every_tool(test: TestNeRD, device: torch.device) -> None:
    """Record for a model of the minimal family, train it, step it, roll it out, solve with it and save it."""
    simulation = make_arm_simulation(device, 16)
    torch.manual_seed(0)
    nerd = MinimalNeRD(simulation.system).to(device)
    test.assertFalse(nerd.fitted)
    test.assertIsNone(nerd.contact_input)

    # The recorder reads the sources of the model, and the model takes its statistics from the recording.
    windows = _recorded_windows(test, simulation, nerd)
    batch = windows.batch(torch.arange(min(256, len(windows)), device=device))
    _lower_the_loss(test, nerd, batch)
    check_nerd_streaming(nerd, batch.frames)

    # The stepper deploys the model, and the rollout error rolls it out against the recording.
    world_count = simulation.world_count
    frames = frame_of(windows.batch(torch.arange(world_count, device=device)).frames, 0)
    stepper = NeRDStepper(nerd, world_count, precision="fp32")  # the arithmetic of the eager model
    with torch.no_grad():
        expected = nerd.step(frames, nerd.history(world_count))
    assert_frames_close(test, stepper.step(frames), expected, tolerance=1e-4)
    statistics = rollout_error(nerd, windows, horizons=(1, 4), rollout_count=64)
    test.assertEqual(list(statistics), [1, 4])
    test.assertTrue(all(0.0 <= value.nrmse < float("inf") for value in statistics.values()))

    # The saved model comes back as its own family and steps the robot as a Newton solver.
    path = test.folder / "model.pt"
    nerd.save(path)
    loaded = NeRDBase.load(path, device=device)
    test.assertIs(type(loaded), MinimalNeRD)
    test.assertEqual((loaded.signals, loaded.frame_dt, loaded.fitted), (nerd.signals, FRAME_DT, True))
    solver = SolverNeRD(simulation.model, path, precision="fp32")  # the arithmetic of the eager model
    test.assertIs(type(solver.nerd), MinimalNeRD)
    state, control = simulation.model.state(), simulation.model.control()
    solver.reset(state)
    solver.binding.write_state(state, frames)
    solver.binding.write_control(control, frames)
    newton.eval_fk(simulation.model, state.joint_q, state.joint_qd, state)
    solver.step(state, state, control, None, FRAME_DT)
    assert_frames_close(test, solver.binding.read_state(state), expected, tolerance=1e-4)


def test_minimal_family_with_contact_input(test: TestNeRD, device: torch.device) -> None:
    """Give a model of the minimal family probes and record for it, train it, step it and solve with it."""
    simulation = make_arm_simulation(device, 16)
    system = simulation.system
    probes = ContactInputProbes.from_model(simulation.model, system, bodies="*_finger", radius=0.01)
    torch.manual_seed(0)
    nerd = MinimalNeRD(system, contact_input=probes).to(device)
    test.assertEqual(list(nerd.signals), [*system.signals, "contact_probes"])
    test.assertEqual(nerd.linear.in_features, system.feature_count + probes.code_count)

    # The recorder senses the probes with every state, and the encoder takes its statistics from the recording.
    windows = _recorded_windows(test, simulation, nerd)
    batch = windows.batch(torch.arange(min(256, len(windows)), device=device))
    _lower_the_loss(test, nerd, batch)
    check_nerd_streaming(nerd, batch.frames)
    # A loss with noise evaluates the probes again on the perturbed states, and the model reads what they give.
    with torch.no_grad():
        test.assertTrue(torch.isfinite(LossOneStep(noise=0.1)(nerd, batch)))
        lifted = {**batch.frames, "contact_probes": batch.frames["contact_probes"] + 1.0}
        test.assertFalse(torch.equal(nerd(lifted), nerd(batch.frames)))

    # The stepper deploys the model on recorded frames, which hold the probes of their states. The rollout error
    # evaluates the probes again at every state the model predicts.
    world_count = simulation.world_count
    frames = frame_of(windows.batch(torch.arange(world_count, device=device)).frames, 0)
    with torch.no_grad():
        expected = nerd.step(frames, nerd.history(world_count))
    stepper = NeRDStepper(nerd, world_count, precision="fp32")  # the arithmetic of the eager model
    assert_frames_close(test, stepper.step(frames), expected, tolerance=1e-4)
    statistics = rollout_error(nerd, windows, horizons=(4,), rollout_count=64)
    test.assertTrue(0.0 <= statistics[4].nrmse < float("inf"))

    # The solver rebuilds the family with its contact input from the file and senses the probes itself.
    path = test.folder / "model.pt"
    nerd.save(path)
    solver = SolverNeRD(simulation.model, path, precision="fp32")  # the arithmetic of the eager model
    test.assertIs(type(solver.nerd), MinimalNeRD)
    test.assertEqual(solver.nerd.contact_input.config(), probes.config())
    state, control = simulation.model.state(), simulation.model.control()
    solver.reset(state)
    solver.binding.write_state(state, frames)
    solver.binding.write_control(control, frames)
    newton.eval_fk(simulation.model, state.joint_q, state.joint_qd, state)
    solver.step(state, state, control, None, FRAME_DT)
    assert_frames_close(test, solver.binding.read_state(state), expected, tolerance=1e-4)


def test_streaming_of_both_families(test: TestNeRD, device: torch.device) -> None:
    """Check that models of both families predict frame by frame what their forward gives over windows."""
    models = (
        make_nerd("arm", device, family=MinimalNeRD),
        make_nerd("arm", device),
        make_nerd("quadruped", device, contact_input=ContactInputRootHeight()),
        make_nerd("quadruped", device, contact_input=ContactInputRootHeight(), backbone=BackboneLeaky),
    )
    for nerd in models:
        check_nerd_streaming(nerd, make_frames(nerd, WORLD_COUNT, 9, device=device))


def test_differentiable_rollout(test: TestNeRD, device: torch.device) -> None:
    """Differentiate a rollout of three steps, which takes a clone of the history at every step."""
    for name, options in NERDS.items():
        nerd = make_nerd("arm", device, **options)
        start = make_frames(nerd, WORLD_COUNT, device=device)
        start["joint_q"].requires_grad_()
        frames, history = dict(start), nerd.history(WORLD_COUNT)
        for _ in range(3):
            history = {key: value.clone() for key, value in history.items()}
            frames |= nerd.step(frames, history)
        (frames["joint_q"].sum() + frames["joint_qd"].sum()).backward()
        gradient = start["joint_q"].grad
        test.assertTrue(torch.isfinite(gradient).all(), name)
        test.assertGreater(gradient.abs().sum().item(), 0.0, name)
        weights = [parameter.grad for parameter in nerd.parameters() if parameter.grad is not None]
        test.assertTrue(any(weight.abs().sum().item() > 0.0 for weight in weights), name)


def test_load_returns_the_family_of_the_file(test: TestNeRD, device: torch.device) -> None:
    """Load files of two families through the base class, and expect each family to refuse the other's file."""
    models = {family: make_nerd("arm", device, family=family) for family in (MinimalNeRD, FlashNeRD)}
    for family, nerd in models.items():
        path = test.folder / f"{family.__name__}.pt"
        nerd.save(path)
        frames = make_frames(nerd, WORLD_COUNT, device=device)
        for loaded in (NeRDBase.load(path, device=device), family.load(path, device=device)):
            test.assertIs(type(loaded), family)
            with torch.no_grad():
                expected = nerd.predict(frames, nerd.history(WORLD_COUNT))
                test.assertTrue(torch.equal(loaded.predict(frames, loaded.history(WORLD_COUNT)), expected))
    with test.assertRaisesRegex(ValueError, "holds a MinimalNeRD, not a FlashNeRD.*NeRDBase.load"):
        FlashNeRD.load(test.folder / "MinimalNeRD.pt")
    with test.assertRaisesRegex(ValueError, "holds a FlashNeRD, not a MinimalNeRD"):
        MinimalNeRD.load(test.folder / "FlashNeRD.pt")


def test_training_frames(test: unittest.TestCase, device: torch.device) -> None:
    """Split windows into frame pairs for a state increment model and keep every frame for a free motion model."""
    nerd = make_nerd("arm", device)
    window = make_frames(nerd, WORLD_COUNT, 5, device=device)
    current, following = nerd.training_frames(window)
    for name, value in window.items():
        test.assertTrue(torch.equal(current[name], value[:, :-1]))
        test.assertTrue(torch.equal(following[name], value[:, 1:]))
    with test.assertRaisesRegex(ValueError, "at least one step"):
        nerd.training_frames(frame_of(window, slice(0, 1)))

    nerd = FlashNeRD(make_system("arm"), backbone=make_backbone(), prediction=PredictionFreeMotion()).to(device)
    window = make_frames(nerd, WORLD_COUNT, 5, device=device)
    current, following = nerd.training_frames(window)
    test.assertIs(current, window)
    test.assertIsNone(following)
    single = frame_of(window, slice(0, 1))
    test.assertIs(nerd.training_frames(single)[0], single)

    # The statistics cover every frame, and windows of single frames are enough to fit them.
    starts = torch.ones(WORLD_COUNT, dtype=torch.bool, device=device)
    nerd.fit_normalization([TrajectoryBatch(window, starts, FRAME_DT), TrajectoryBatch(single, starts, FRAME_DT)])
    test.assertEqual(nerd.feature_normalizer.count.item(), WORLD_COUNT * 6)
    test.assertEqual(nerd.output_normalizer.count.item(), WORLD_COUNT * 6)
    labels = torch.cat((window["joint_qdd_free"].flatten(0, 1), single["joint_qdd_free"].flatten(0, 1)))
    torch.testing.assert_close(nerd.output_normalizer.mean, labels.mean(dim=0), atol=1e-5, rtol=0)


def test_free_motion_model(test: unittest.TestCase, device: torch.device) -> None:
    """Build a free motion model and expect its label signal, raw predictions and a refusal to step."""
    nerd = make_nerd("arm", device, prediction=PredictionFreeMotion())
    test.assertEqual(list(nerd.signals), ["joint_q", "joint_qd", "joint_target_q", "joint_qdd_free"])
    window = make_frames(nerd, WORLD_COUNT, 3, device=device)
    current, following = frame_of(window, 0), frame_of(window, 1)
    expected = nerd.output_normalizer.normalize(current["joint_qdd_free"])
    test.assertTrue(torch.equal(nerd.labels(current, following), expected))
    history = nerd.history(WORLD_COUNT)
    test.assertEqual(nerd.predict(current, history).shape, (WORLD_COUNT, nerd.system.joint_dof_count))
    with test.assertRaisesRegex(RuntimeError, "physics solver"):
        nerd.step(current, history)


def test_sources(test: unittest.TestCase, device: torch.device) -> None:
    """Read a model's sources around a simulation step and expect state and contact before it, control after it."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    model = fixtures.quadruped_model(WORLD_COUNT, device)
    solver = newton.solvers.SolverFeatherstone(model)
    simulation = SimulationNewton(model, solver, nerd.system, articulation="quadruped", frame_dt=1e-3)
    sources = nerd.sources(simulation)
    state_source, contact_source, control_source = sources
    signals = [list(source.signals) for source in sources]
    test.assertEqual(signals, [["joint_q", "joint_qd"], ["root_height"], ["joint_target_q"]])
    test.assertEqual([source.after_step for source in sources], [False, False, True])

    before = {**state_source.read(), **contact_source.read()}
    test.assertTrue(torch.equal(before["root_height"], before["joint_q"][:, 2:3]))
    command = {"joint_target_q": torch.full((WORLD_COUNT, 12), 0.25, device=device)}
    simulation.step(command)
    control = control_source.read()
    test.assertTrue(torch.equal(control["joint_target_q"], command["joint_target_q"]))
    simulation.step({"joint_target_q": torch.zeros(WORLD_COUNT, 12, device=device)})
    after = {**state_source.read(), **contact_source.read()}
    test.assertFalse(torch.equal(after["joint_q"], before["joint_q"]))
    test.assertTrue(torch.equal(after["root_height"], after["joint_q"][:, 2:3]))
    # What was read stays as it was while the simulation moves on, so the reads are copies.
    test.assertTrue(torch.equal(before["root_height"], before["joint_q"][:, 2:3]))
    test.assertTrue(torch.equal(control["joint_target_q"], command["joint_target_q"]))

    for source in sources:
        for name, value in source.read().items():
            test.assertEqual(value.shape, (WORLD_COUNT, *nerd.signals[name].shape))
            test.assertEqual(value.dtype, nerd.signals[name].torch_dtype)


def test_sources_with_labels(test: unittest.TestCase, device: torch.device) -> None:
    """Read the sources of a free motion model on a MuJoCo simulation and expect the label source last."""
    nerd = make_nerd("arm", device, prediction=PredictionFreeMotion())
    model = fixtures.arm_model(WORLD_COUNT, device)
    solver = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, solver, nerd.system, articulation="arm", frame_dt=2e-3)
    sources = nerd.sources(simulation)
    signals = [list(source.signals) for source in sources]
    test.assertEqual(signals, [["joint_q", "joint_qd"], ["joint_target_q"], ["joint_qdd_free"]])
    test.assertEqual([source.after_step for source in sources], [False, True, True])
    simulation.step()
    label = sources[-1].read()["joint_qdd_free"]
    test.assertEqual(label.shape, (WORLD_COUNT, nerd.system.joint_dof_count))
    test.assertTrue(torch.isfinite(label).all())


class TestNeRD(FolderTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(torch.compiler.reset)

    def test_families_register_by_class_name(self) -> None:
        """Expect every family under its class name in the registry of the base class, and not the base class."""
        self.assertIs(NeRDBase.registry["FlashNeRD"], FlashNeRD)
        self.assertIs(NeRDBase.registry["MinimalNeRD"], MinimalNeRD)
        self.assertNotIn("NeRDBase", NeRDBase.registry)
        self.assertIs(FlashNeRD.registry, NeRDBase.registry)

    def test_signal_names_must_differ(self) -> None:
        """Give a contact input a signal named like a state signal and expect a ValueError."""
        contact_input = ContactInputRootHeight()
        contact_input.signals = {"joint_q": SignalSpec("joint_q", (1,), SignalRole.CONTACT)}
        with self.assertRaisesRegex(ValueError, "differently"):
            FlashNeRD(make_system("quadruped"), backbone=make_backbone(), contact_input=contact_input)

    def test_minimal_family_checkpoint(self) -> None:
        """Save a model of the minimal family and expect its class, its two parts and no settings in the file."""
        path = self.folder / "model.pt"
        make_nerd("arm", "cpu", family=MinimalNeRD).save(path)
        manifest = torch.load(path, weights_only=True)["manifest"]
        self.assertEqual(manifest["model"], "MinimalNeRD")
        names = {kind: part and part["name"] for kind, part in manifest["parts"].items()}
        expected = {"system": "SystemArticulation", "contact_input": None, "prediction": "PredictionStateIncrement"}
        self.assertEqual(names, expected)
        self.assertEqual(manifest["settings"], {})

    def test_family_must_implement_the_interface(self) -> None:
        """Leave a member out of a family and expect a TypeError that names it when a model is built."""
        with self.assertRaisesRegex(TypeError, "abstract"):
            NeRDBase(make_system("arm"))

        class NeRDWithoutPredict(MinimalNeRD):
            predict = NeRDBase.predict

        with self.assertRaisesRegex(TypeError, "abstract.*NeRDWithoutPredict.*predict"):
            NeRDWithoutPredict(make_system("arm"))


for _function in (
    test_minimal_family_through_every_tool,
    test_minimal_family_with_contact_input,
    test_streaming_of_both_families,
    test_differentiable_rollout,
    test_load_returns_the_family_of_the_file,
    test_training_frames,
    test_free_motion_model,
    test_sources,
    test_sources_with_labels,
):
    add_function_test(TestNeRD, _function.__name__, _function)


if __name__ == "__main__":
    unittest.main()
