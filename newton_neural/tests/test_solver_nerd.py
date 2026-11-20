# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SolverNeRD``: eager parity, contacts, resets, CUDA graphs, observables, and warnings."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from newton_neural import Frames, NewtonNeuralWarning
from newton_neural._src.core import rotation as rot
from newton_neural._src.systems.articulation_binding import BindingArticulation
from newton_neural.contacts import ContactInputBase, ContactInputProbes, ContactInputSet, ContactSensorBase
from newton_neural.data import SimulationNewton
from newton_neural.models import FlashNeRD, Precision
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.solvers import SolverNeRD
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.tests.extensions import ContactInputRootHeight
from newton_neural.tests.helpers import (
    FRAME_DT,
    WORLD_COUNT,
    assert_frames_close,
    make_backbone,
    make_nerd,
    recorded_warnings,
)
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

STAND_HEIGHT = 0.5
"""Base height [m] at which the quadruped's feet press into the ground and its links touch each other."""


def _clones(frames: Frames) -> Frames:
    return {name: value.clone() for name, value in frames.items()}


def _nerd(
    model: newton.Model,
    articulation: str,
    contact_input: Callable[[newton.Model, SystemArticulation], ContactInputBase] | None = None,
) -> FlashNeRD:
    """A small fitted model of the robot in ``model``, with the system read from the Newton model a solver steps.

    It is deployed in IEEE float32 unless a solver is given another precision, so that a step of a solver is
    the step of the eager model to rounding.
    """
    system = SystemArticulation.from_model(model, articulation)
    contact_input = None if contact_input is None else contact_input(model, system)
    nerd = make_nerd(system, wp.device_to_torch(model.device), contact_input=contact_input)
    nerd.precision = Precision.FP32
    return nerd


def _probes(model: newton.Model, system: SystemArticulation) -> ContactInputProbes:
    return ContactInputProbes.from_model(model, system, bodies="*_SHANK")


def _contact_set(model: newton.Model, system: SystemArticulation) -> ContactInputSet:
    return ContactInputSet.from_system(system, slot_count=4, slot_dim=8, link_embedding_dim=4)


def _small_contact_set(model: newton.Model, system: SystemArticulation) -> ContactInputSet:
    return ContactInputSet.from_system(system, capacity=5, slot_count=4, slot_dim=8, link_embedding_dim=4)


DROPPED_ROWS = 3
"""Rows per world that the sensor of :class:`ContactInputDropping` reports as dropped."""


class ContactInputDropping(ContactInputRootHeight):
    """A contact input written outside the library whose sensor reports dropped rows, as one with a capacity does."""

    def sensor(self, model: newton.Model, binding: BindingArticulation) -> ContactSensorBase:
        sensor = super().sensor(model, binding)
        sensor.overflow = wp.full(binding.world_count, DROPPED_ROWS, dtype=wp.int32, device=model.device)
        return sensor


def _pose(solver: SolverNeRD, state: newton.State, control: newton.Control, seed: int) -> None:
    """Restart the solver, then write random joint state and controls of the robot and update its bodies."""
    frames = fixtures.random_frames(solver.nerd.system, WORLD_COUNT, device=solver.nerd.device, seed=seed)
    solver.reset(state)
    solver.binding.write_state(state, frames)
    solver.binding.write_control(control, frames)
    newton.eval_fk(solver.model, state.joint_q, state.joint_qd, state)


def _stand(solver: SolverNeRD, state: newton.State, height: float | Tensor, seed: int = 0) -> None:
    """Put the quadruped upright with bent legs, its base ``height`` [m] above the ground. The history is kept."""
    frames = fixtures.random_frames(solver.nerd.system, WORLD_COUNT, device=solver.nerd.device, seed=seed)
    frames["joint_q"][:, 2] = height
    frames["joint_q"][:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=solver.nerd.device)
    solver.binding.write_state(state, frames)
    newton.eval_fk(solver.model, state.joint_q, state.joint_qd, state)


class _Reference:
    """The eager model stepped beside a solver, on copies of the frames the solver reads."""

    def __init__(self, solver: SolverNeRD) -> None:
        self.nerd, self.binding = solver.nerd, solver.binding
        self.history = self.nerd.history(WORLD_COUNT)

    def step(self, state: newton.State, control: newton.Control, contact: Frames | None = None) -> Frames:
        """The next STATE signals from ``state``, with the contact signals ``contact`` when the model has any."""
        frames = _clones({**self.binding.read_state(state), **self.binding.read_control(control)})
        if contact is not None:
            frames.update(_clones(contact))
        with torch.no_grad():
            return self.nerd.step(frames, self.history)

    def reset(self, world_mask: Tensor | None = None) -> None:
        if world_mask is None:
            world_mask = torch.ones(WORLD_COUNT, dtype=torch.bool, device=self.nerd.device)
        self.nerd.reset_history(self.history, world_mask)


def _assert_state(
    test: unittest.TestCase, solver: SolverNeRD, state: newton.State, expected: Frames, tolerance: float = 1e-4
) -> None:
    """Require ``state`` to hold the robot's ``expected`` joint state and the body transforms that belong to it."""
    assert_frames_close(test, solver.binding.read_state(state), expected, tolerance=tolerance)
    current = solver.model.state()
    current.joint_q.assign(state.joint_q)
    current.joint_qd.assign(state.joint_qd)
    newton.eval_fk(solver.model, current.joint_q, current.joint_qd, current)
    np.testing.assert_allclose(state.body_q.numpy(), current.body_q.numpy(), atol=1e-5)
    np.testing.assert_allclose(state.body_qd.numpy(), current.body_qd.numpy(), atol=1e-4)


def _host_contact_wrenches(model: newton.Model, contacts: newton.Contacts, state: newton.State) -> np.ndarray:
    """The contact force estimates of every detected rigid contact, computed on the host, ``[contact_count, 6]``.

    The force on body 0 is the mean contact stiffness of the two shapes times the penetration, against the
    normal [N]. The torque is taken about the centre of mass of body 0, or the world origin for a static shape [N*m].
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes = [contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]]
    points = [contacts.rigid_contact_point0.numpy()[:count], contacts.rigid_contact_point1.numpy()[:count]]
    margins = [contacts.rigid_contact_margin0.numpy()[:count], contacts.rigid_contact_margin1.numpy()[:count]]
    normal = torch.as_tensor(contacts.rigid_contact_normal.numpy()[:count])
    body_q, body_com = state.body_q.numpy(), model.body_com.numpy()
    identity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def to_world(side: int, local: np.ndarray) -> Tensor:
        bodies = model.shape_body.numpy()[shapes[side]]
        pose = torch.as_tensor(np.where(bodies[:, None] >= 0, body_q[bodies], identity))
        return pose[:, :3] + rot.rotate(pose[:, 3:], torch.as_tensor(local))

    supports = [to_world(side, points[side]) for side in (0, 1)]
    gap = ((supports[1] - supports[0]) * normal).sum(-1) - torch.as_tensor(margins[0] + margins[1])
    stiffness = model.shape_material_ke.numpy()
    push = torch.as_tensor(0.5 * (stiffness[shapes[0]] + stiffness[shapes[1]])) * gap.clamp(max=0.0)
    force = push[:, None] * normal
    body0 = model.shape_body.numpy()[shapes[0]]
    com = to_world(0, np.where(body0[:, None] >= 0, body_com[body0], 0.0).astype(np.float32))
    surface = supports[0] + torch.as_tensor(margins[0])[:, None] * normal
    return torch.cat([force, torch.linalg.cross(surface - com, force)], dim=-1).numpy()


def _robot_model(scene: str, device: torch.device) -> newton.Model:
    """The fixture model ``scene``, such as ``"quadruped"`` or ``"arm_with_box"``."""
    return getattr(fixtures, f"{scene}_model")(WORLD_COUNT, device)


def test_matches_model(
    test: unittest.TestCase,
    device: torch.device,
    scene: str,
    robot: str,
    in_place: bool,
    graphed: bool,
    precision: Precision,
) -> None:
    """Step a solver over several frames and expect the eager model's step from the same frames each time."""
    model = _robot_model(scene, device)
    solver = SolverNeRD(
        model,
        _nerd(model, robot),
        articulation=robot,
        precision=precision,
        use_cuda_graph=graphed,
        compile=graphed,
    )
    test.assertIs(solver.stepper.precision, precision)
    reference = _Reference(solver)
    state_in, control = model.state(), model.control()
    state_out = state_in if in_place else model.state()
    _pose(solver, state_in, control, seed=1)
    tolerance = 1e-4 if precision is Precision.FP32 else 1e-1
    for _ in range(6):
        expected = reference.step(state_in, control)
        solver.step(state_in, state_out, control, None, FRAME_DT)
        _assert_state(test, solver, state_out, expected, tolerance)
        state_in, state_out = state_out, state_in


def test_model_control(test: unittest.TestCase, device: torch.device) -> None:
    """Step without a control object and expect the controls stored in the Newton model to be used."""
    model = _robot_model("arm", device)
    solver = SolverNeRD(model, _nerd(model, "arm"), use_cuda_graph=False, compile=False)
    state, model_control = model.state(), model.control(clone_variables=False)
    _pose(solver, state, model_control, seed=2)
    test.assertGreater(wp.to_torch(model.joint_target_q).abs().max().item(), 0.0)
    expected = _Reference(solver).step(state, model_control)
    solver.step(state, state, None, None, FRAME_DT)
    _assert_state(test, solver, state, expected)


def test_repr(test: unittest.TestCase, device: torch.device) -> None:
    """Print a solver and expect its stepper and whether it replays from CUDA graphs, which none does on a
    device without CUDA."""
    model = _robot_model("arm", device)
    for asked in (True, False):
        solver = SolverNeRD(model, _nerd(model, "arm"), use_cuda_graph=asked, compile=False)
        replays = asked and device.type == "cuda"
        test.assertIs(solver.use_cuda_graph, replays)
        test.assertEqual(repr(solver), f"SolverNeRD({solver.stepper!r}, use_cuda_graph={replays})")


def test_other_articulations_carry_over(test: unittest.TestCase, device: torch.device, in_place: bool) -> None:
    """Step an arm beside a free box and expect the box to keep its joint state and body transform."""
    model = _robot_model("arm_with_box", device)
    solver = SolverNeRD(model, _nerd(model, "arm"), articulation="arm", use_cuda_graph=False, compile=False)
    state_in, control = model.state(), model.control()
    state_out = state_in if in_place else model.state()
    # The box is the first articulation of every world. Move it and give it a velocity.
    box_q = wp.to_torch(state_in.joint_q).view(WORLD_COUNT, -1)[:, :7]
    box_qd = wp.to_torch(state_in.joint_qd).view(WORLD_COUNT, -1)[:, :6]
    box_q[:, :3] += torch.tensor([0.1, -0.2, 0.3], device=device)
    box_qd[:] = torch.tensor([0.5, 0.0, -0.5, 0.0, 1.0, 0.0], device=device)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    _pose(solver, state_in, control, seed=4)
    box_body = [index for index, label in enumerate(model.body_label) if label.endswith("box0")]
    test.assertEqual(len(box_body), WORLD_COUNT)
    before = box_q.clone(), box_qd.clone(), state_in.body_q.numpy()[box_body], state_in.body_qd.numpy()[box_body]

    expected = _Reference(solver).step(state_in, control)
    solver.step(state_in, state_out, control, None, FRAME_DT)
    _assert_state(test, solver, state_out, expected)
    torch.testing.assert_close(wp.to_torch(state_out.joint_q).view(WORLD_COUNT, -1)[:, :7], before[0])
    torch.testing.assert_close(wp.to_torch(state_out.joint_qd).view(WORLD_COUNT, -1)[:, :6], before[1])
    np.testing.assert_allclose(state_out.body_q.numpy()[box_body], before[2], atol=1e-6)
    np.testing.assert_allclose(state_out.body_qd.numpy()[box_body], before[3], atol=1e-6)


def test_dt_mismatch(test: unittest.TestCase, device: torch.device) -> None:
    """Step with another time step than the model's frame and expect an error that names both."""
    model = _robot_model("arm", device)
    solver = SolverNeRD(model, _nerd(model, "arm"), use_cuda_graph=False, compile=False)
    state = model.state()
    with test.assertRaisesRegex(ValueError, rf"{FRAME_DT} s.*0\.005 s"):
        solver.step(state, state, None, None, 0.005)
    solver.step(state, state, None, None, FRAME_DT * (1.0 + 1e-7))  # within rounding of the frame


def test_checkpoint_path(test: unittest.TestCase, device: torch.device) -> None:
    """Build one solver from a saved model's path and one from the model object and expect the same step."""
    model = _robot_model("quadruped", device)
    nerd = _nerd(model, "quadruped", _probes)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.pt"
        nerd.save(path)
        solvers = [SolverNeRD(model, nerd=given, use_cuda_graph=False, compile=False) for given in (nerd, str(path))]
    test.assertIs(solvers[0].nerd, nerd)
    test.assertEqual(solvers[1].nerd.device, device)
    states = []
    for solver in solvers:
        state, control = model.state(), model.control()
        _pose(solver, state, control, seed=5)
        solver.step(state, state, control, None, FRAME_DT)
        states.append(_clones(solver.binding.read_state(state)))
    assert_frames_close(test, states[1], states[0], tolerance=1e-6)


def test_refuses_mismatch(test: unittest.TestCase, device: torch.device) -> None:
    """Give a solver a model of another robot, or one that does not predict the next state, and expect an error."""
    arm = _robot_model("arm", device)
    quadruped = _robot_model("quadruped", device)
    with test.assertRaisesRegex(ValueError, "joints"):
        SolverNeRD(quadruped, _nerd(arm, "arm"), compile=False)
    with test.assertRaisesRegex(ValueError, "SolverNeRDMuJoCo"):
        SolverNeRD(arm, make_nerd("arm", device, prediction=PredictionFreeMotion()), compile=False)
    without_frame = FlashNeRD(SystemArticulation.from_model(arm, "arm"), backbone=make_backbone())
    with test.assertRaisesRegex(ValueError, "no frame duration.*build the model with frame_dt"):
        SolverNeRD(arm, without_frame, compile=False)


def test_reset(test: unittest.TestCase, device: torch.device) -> None:
    """Reset two of four worlds and expect their history cleared and the joint state that ``flags`` names restored."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped"), use_cuda_graph=False, compile=False)
    reference = _Reference(solver)
    state, control = model.state(), model.control()
    _pose(solver, state, control, seed=6)
    for _ in range(3):
        reference.step(state, control)
        solver.step(state, state, control, None, FRAME_DT)
    history = solver.stepper.export_history()
    default = _clones(solver.binding.read_state(model.state()))
    before = _clones(solver.binding.read_state(state))

    mask = torch.tensor([True, False, False, True], device=device)
    wp.to_torch(state.body_q).add_(1.0)  # stale in every world
    stale = state.body_q.numpy().copy()
    newton_mask = wp.array([True, False, False, True, False], dtype=wp.bool, device=model.device)

    # A reset that restores nothing clears the history of its worlds and keeps the state as it was written.
    solver.reset(state, newton_mask, flags=0)
    for name, value in solver.stepper.export_history().items():
        test.assertFalse(value[mask].any(), name)
        torch.testing.assert_close(value[~mask], history[name][~mask])
    for name, value in solver.binding.read_state(state).items():
        test.assertTrue(torch.equal(value, before[name]), name)
    np.testing.assert_array_equal(state.body_q.numpy(), stale)

    # A reset of the velocities alone leaves the positions.
    solver.reset(state, newton_mask, flags=newton.StateFlags.JOINT_QD)
    found = solver.binding.read_state(state)
    test.assertTrue(torch.equal(found["joint_q"], before["joint_q"]))
    test.assertTrue(torch.equal(found["joint_qd"][mask], default["joint_qd"][mask]))
    test.assertTrue(torch.equal(found["joint_qd"][~mask], before["joint_qd"][~mask]))

    # A reset without flags restores the model's defaults, and the bodies of its worlds follow.
    solver.reset(state, newton_mask)
    reference.reset(mask)
    for name, value in solver.binding.read_state(state).items():
        test.assertTrue(torch.equal(value[mask], default[name][mask]), name)
        test.assertTrue(torch.equal(value[~mask], before[name][~mask]), name)
    body_q = state.body_q.numpy().reshape(WORLD_COUNT, -1, 7)
    restarted = mask.cpu().numpy()
    np.testing.assert_array_equal(body_q[~restarted], stale.reshape(WORLD_COUNT, -1, 7)[~restarted])
    np.testing.assert_allclose(body_q[restarted, 0], default["joint_q"][mask, :7].cpu().numpy(), atol=1e-6)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)  # the worlds left alone still hold stale bodies
    expected = reference.step(state, control)
    solver.step(state, state, control, None, FRAME_DT)
    _assert_state(test, solver, state, expected)
    with test.assertRaises(ValueError):  # Newton's mask has one more entry than there are worlds
        solver.reset(state, wp.zeros(WORLD_COUNT, dtype=wp.bool, device=model.device))


def test_reset_restores_whole_worlds(test: unittest.TestCase, device: torch.device) -> None:
    """Restore every joint of the masked worlds, the boxes beside the arm among them, in worlds of unequal size."""
    model = _robot_model("arm_with_uneven_boxes", device)
    solver = SolverNeRD(model, _nerd(model, "arm"), articulation="arm", use_cuda_graph=False, compile=False)
    state = model.state()
    default_q, default_qd = model.joint_q.numpy().copy(), model.joint_qd.numpy().copy()
    wp.to_torch(state.joint_q).add_(0.1)
    wp.to_torch(state.joint_qd).add_(1.0)
    moved_q, moved_qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()

    mask = np.array([False, True, True, False, False])
    solver.reset(state, wp.array(mask, dtype=wp.bool, device=model.device))
    joint_world = model.joint_world.numpy()
    coords = np.repeat(mask[joint_world], np.diff(model.joint_q_start.numpy()))
    dofs = np.repeat(mask[joint_world], np.diff(model.joint_qd_start.numpy()))
    test.assertTrue(coords.any() and not coords.all())
    np.testing.assert_array_equal(state.joint_q.numpy(), np.where(coords, default_q, moved_q))
    np.testing.assert_array_equal(state.joint_qd.numpy(), np.where(dofs, default_qd, moved_qd))

    expected = model.state()
    expected.joint_q.assign(state.joint_q)
    expected.joint_qd.assign(state.joint_qd)
    newton.eval_fk(model, expected.joint_q, expected.joint_qd, expected)
    bodies = mask[model.body_world.numpy()]
    np.testing.assert_allclose(state.body_q.numpy()[bodies], expected.body_q.numpy()[bodies], atol=1e-6)


def test_reset_without_worlds(test: unittest.TestCase, device: torch.device) -> None:
    """Reset a robot built without worlds, which lies in Newton's global world and is world 0 of a mask."""
    builder = newton.ModelBuilder()
    fixtures.add_arm(builder)
    builder.add_ground_plane()
    model = builder.finalize(device=str(device))
    test.assertLess(model.joint_world.numpy().max(), 0)
    solver = SolverNeRD(model, _nerd(model, "arm"), use_cuda_graph=False, compile=False)
    default = model.joint_q.numpy().copy()

    def moved() -> newton.State:
        state = model.state()
        wp.to_torch(state.joint_q).add_(0.1)
        wp.to_torch(state.joint_qd).add_(1.0)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state

    state = moved()
    solver.step(state, state, model.control(), None, FRAME_DT)
    history = solver.stepper.export_history()
    test.assertTrue(any(value.any() for value in history.values()))

    # The entry of world 0 governs the robot, as it does in SolverMuJoCo, and the entry of the global world nothing.
    state = moved()
    solver.reset(state, wp.array([False, True], dtype=wp.bool, device=model.device))
    test.assertFalse(np.isclose(state.joint_q.numpy(), default).any())
    for name, value in solver.stepper.export_history().items():
        test.assertTrue(torch.equal(value, history[name]), name)
    solver.reset(state, wp.array([True, False], dtype=wp.bool, device=model.device))
    np.testing.assert_array_equal(state.joint_q.numpy(), default)
    test.assertFalse(any(value.any() for value in solver.stepper.export_history().values()))
    expected = model.state()
    newton.eval_fk(model, expected.joint_q, expected.joint_qd, expected)
    np.testing.assert_allclose(state.body_q.numpy(), expected.body_q.numpy(), atol=1e-6)


def test_restart_through_simulation(test: unittest.TestCase, device: torch.device) -> None:
    """Restart worlds through a simulation and expect the written state, bodies that follow it and no history."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped"))
    simulation = SimulationNewton(model, solver, solver.nerd.system, frame_dt=FRAME_DT)
    binding = simulation.binding
    for _ in range(3):
        simulation.step()
    before = _clones(binding.read_state(simulation.state))
    history = solver.stepper.export_history()
    start = fixtures.random_frames(solver.nerd.system, WORLD_COUNT, device=device, seed=7)
    mask = torch.tensor([True, False, False, True], device=device)

    simulation.restart(start, mask)
    for name, value in binding.read_state(simulation.state).items():
        torch.testing.assert_close(value[mask], start[name][mask])
        test.assertTrue(torch.equal(value[~mask], before[name][~mask]), name)
    for name, value in solver.stepper.export_history().items():
        test.assertFalse(value[mask].any(), name)
        torch.testing.assert_close(value[~mask], history[name][~mask])
    bodies = model.state()
    newton.eval_fk(model, simulation.state.joint_q, simulation.state.joint_qd, bodies)
    np.testing.assert_allclose(simulation.state.body_q.numpy(), bodies.body_q.numpy(), atol=1e-5)


def test_caller_capture(test: unittest.TestCase, device: torch.device, relaxed: bool, graphed: bool) -> None:
    """Record a step into a graph the caller captures, replay it and expect it to follow the state between replays."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped", _probes), use_cuda_graph=graphed)
    contact_input, system = solver.nerd.contact_input, solver.nerd.system
    reference = _Reference(solver)
    state, control = model.state(), model.control()
    _pose(solver, state, control, seed=8)

    def expected_step() -> Frames:
        return reference.step(state, control, contact_input.evaluate(system, solver.binding.read_state(state)))

    stream = wp.get_stream(model.device)
    mode = wp.CaptureMode.THREAD_LOCAL
    if relaxed:  # as a caller captures on a stream of its own
        stream = wp.stream_from_torch(torch.cuda.Stream(device=device))
        mode = wp.CaptureMode.RELAXED
    with test.assertRaisesRegex(RuntimeError, "one step before"):
        with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream, capture_mode=mode):
            solver.step(state, state, control, None, FRAME_DT)
    with torch.inference_mode():  # an environment may step in inference mode and capture outside it
        for _ in range(2):
            expected = expected_step()
            solver.step(state, state, control, None, FRAME_DT)
            _assert_state(test, solver, state, expected)
    with wp.ScopedStream(stream), wp.ScopedCapture(stream=stream, capture_mode=mode) as capture:
        solver.step(state, state, control, None, FRAME_DT)

    # Memory that torch hands out on the stream of the capture must not be memory the graph writes.
    with torch.cuda.stream(torch.cuda.ExternalStream(stream.cuda_stream, device=device)):
        bystanders = [torch.full((WORLD_COUNT, width), 7.0, device=device) for width in (8, 16, 19, 37, 64, 256)]
    for replay in range(5):
        if replay == 2:
            _pose(solver, state, control, seed=9)
            reference.reset()
        expected = expected_step()
        wp.capture_launch(capture.graph)
        _assert_state(test, solver, state, expected)
    for bystander in bystanders:
        test.assertTrue((bystander == 7.0).all().item())


def test_capture_on_cpu(test: unittest.TestCase, device: torch.device) -> None:
    """Step inside a Warp capture on the CPU, which cannot record torch work, and expect an error."""
    model = _robot_model("arm", device)
    solver = SolverNeRD(model, _nerd(model, "arm"))
    state = model.state()
    solver.step(state, state, None, None, FRAME_DT)
    with test.assertRaisesRegex(RuntimeError, "CPU"), wp.ScopedCapture(device=model.device):
        solver.step(state, state, None, None, FRAME_DT)


def test_probes(test: unittest.TestCase, device: torch.device, graphed: bool) -> None:
    """Step a model with contact probes and expect the eager step with probes evaluated from the same state."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped", _probes), use_cuda_graph=graphed, compile=graphed)
    contact_input, system = solver.nerd.contact_input, solver.nerd.system
    test.assertEqual(solver.supported_observables, {solver.ObservableKind.CONTACT_F})
    with test.assertRaises(ValueError):
        solver.observables({solver.ObservableKind.CONTACT_ROW_OVERFLOW})
    reference = _Reference(solver)
    state_in, state_out, control = model.state(), model.state(), model.control()
    _pose(solver, state_in, control, seed=10)
    for step in range(4):
        _stand(solver, state_in, STAND_HEIGHT, seed=step)
        probes = contact_input.evaluate(system, solver.binding.read_state(state_in))
        test.assertTrue(probes["contact_probes"][..., 7].any().item())  # some probes touch the ground
        expected = reference.step(state_in, control, probes)
        solver.step(state_in, state_out, control, None, FRAME_DT)
        _assert_state(test, solver, state_out, expected)
        state_in, state_out = state_out, state_in


def test_contact_set(test: unittest.TestCase, device: torch.device, graphed: bool) -> None:
    """Step a model with a contact set on detected contacts and expect the eager step on the same contact rows."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped", _contact_set), use_cuda_graph=graphed, compile=graphed)
    sensor = solver.nerd.contact_input.sensor(model, solver.binding)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    reference = _Reference(solver)
    state, control = model.state(), model.control()
    _pose(solver, state, control, seed=11)
    # The step is refused before any of it runs, also before the first one, which a graph capture cannot hold.
    with test.assertRaisesRegex(ValueError, "ContactInputSet reads the contacts.*the step was given none"):
        solver.step(state, state, control, None, FRAME_DT)
    test.assertFalse(solver.stepper.export_history()["step"].any())
    for step in range(4):
        _stand(solver, state, STAND_HEIGHT, seed=step)
        pipeline.collide(state, contacts)
        rows = sensor.update(state, contacts)
        test.assertGreater(rows["contact_count"].min().item(), 0)
        expected = reference.step(state, control, rows)
        solver.step(state, state, control, contacts, FRAME_DT)
        _assert_state(test, solver, state, expected)


def test_contact_row_overflow(test: unittest.TestCase, device: torch.device) -> None:
    """Give the contact set fewer rows than a world needs and expect the dropped rows in the solver's observable."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped", _small_contact_set), use_cuda_graph=False)
    kinds = solver.ObservableKind
    test.assertEqual(solver.supported_observables, {kinds.CONTACT_F, kinds.CONTACT_ROW_OVERFLOW})
    observables = solver.observables({kinds.CONTACT_ROW_OVERFLOW})
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state = model.state()
    _stand(solver, state, STAND_HEIGHT)
    pipeline.collide(state, contacts)
    sensor = solver.nerd.contact_input.sensor(model, solver.binding)
    sensor.update(state, contacts)
    expected = sensor.overflow.numpy().copy()
    test.assertTrue((expected > 0).all())
    solver.step(state, state, None, contacts, FRAME_DT, observables=observables)
    # The final entry of a per world observable is Newton's global world.
    np.testing.assert_array_equal(observables.contact_row_overflow.numpy(), [*expected, 0])


def test_overflow_of_any_sensor(test: unittest.TestCase, device: torch.device) -> None:
    """Offer the overflow observable for any contact sensor that reports dropped rows, and for no other."""
    model = _robot_model("quadruped", device)
    test.assertIsNone(ContactSensorBase.overflow)
    silent = SolverNeRD(model, _nerd(model, "quadruped", lambda *_: ContactInputRootHeight()))
    test.assertEqual(silent.supported_observables, {silent.ObservableKind.CONTACT_F})

    nerd = _nerd(model, "quadruped", lambda *_: ContactInputDropping())
    solver = SolverNeRD(model, nerd, use_cuda_graph=False)
    kinds = solver.ObservableKind
    test.assertEqual(solver.supported_observables, {kinds.CONTACT_F, kinds.CONTACT_ROW_OVERFLOW})
    observables = solver.observables({kinds.CONTACT_ROW_OVERFLOW})
    state = model.state()
    _stand(solver, state, STAND_HEIGHT)
    solver.step(state, state, None, None, FRAME_DT, observables=observables)
    np.testing.assert_array_equal(observables.contact_row_overflow.numpy(), [DROPPED_ROWS] * WORLD_COUNT + [0])


def test_contact_force_estimates(test: unittest.TestCase, device: torch.device, graphed: bool) -> None:
    """Request contact forces and expect Newton's layout, zeros without penetration and a push out of the ground."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped"), use_cuda_graph=graphed, compile=graphed)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observables = solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)
    test.assertEqual(observables.contact_f.dtype, wp.spatial_vector)
    test.assertEqual(observables.contact_f.shape, (contacts.rigid_contact_max + contacts.soft_contact_max,))
    state = model.state()
    with test.assertRaisesRegex(ValueError, "Contacts"):
        solver.step(state, state, None, None, FRAME_DT, observables=observables)

    ground = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))[0])
    shape_world = model.shape_world.numpy()
    even = torch.tensor([True, False, True, False], device=device)
    # Three steps. The first one is taken as it is, later ones replay a graph when the solver uses graphs.
    for sunk in (even, torch.zeros_like(even), ~even):
        _stand(solver, state, torch.where(sunk, 0.27, 2.0))
        pipeline.collide(state, contacts)
        expected = _host_contact_wrenches(model, contacts, state)
        solver.step(state, state, None, contacts, FRAME_DT, observables=observables)
        test.assertIs(observables.contacts, contacts)
        wrenches = observables.contact_f.numpy()
        count = len(expected)
        np.testing.assert_allclose(wrenches[:count], expected, rtol=1e-4, atol=1e-2)
        test.assertFalse(wrenches[count:].any())

        # A foot in the ground is pushed up, and a robot that stands clear of the ground gets no force from it.
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        on_ground = (shape0 == ground) | (shape1 == ground)
        world = shape_world[np.where(shape0 == ground, shape1, shape0)[on_ground]]
        up = np.where(shape0 == ground, -1.0, 1.0)[on_ground] * wrenches[:count, 2][on_ground]
        for index, in_ground in enumerate(sunk.tolist()):
            pushed = up[world == index]
            if in_ground:
                test.assertGreater(pushed.max(), 0.0)
                test.assertGreaterEqual(pushed.min(), 0.0)
            else:
                test.assertFalse(pushed.any())


def test_refuses_reduced_contacts(test: unittest.TestCase, device: torch.device) -> None:
    """Step with contacts of a pipeline that reduces them per body pair and expect the step to be refused."""
    model = _robot_model("quadruped", device)
    solver = SolverNeRD(model, _nerd(model, "quadruped"), use_cuda_graph=False, compile=False)
    reduction = newton.CollisionPipeline.BodyPairReductionConfig()
    reduced = newton.CollisionPipeline(model, body_pair_reduction=reduction).contacts()
    test.assertTrue(reduced.rigid_contact_body_pair_reduced)
    state = model.state()
    _stand(solver, state, STAND_HEIGHT)
    with test.assertRaisesRegex(ValueError, "body pair reduction.*without body_pair_reduction"):
        solver.step(state, state, None, reduced, FRAME_DT)
    test.assertFalse(solver.stepper.export_history()["step"].any())
    solver.step(state, state, None, newton.CollisionPipeline(model).contacts(), FRAME_DT)


def test_new_objects_every_step(test: unittest.TestCase, device: torch.device) -> None:
    """Step with new contacts at every step and expect an error once more graphs would be kept than the limit."""
    model = _robot_model("arm", device)
    solver = SolverNeRD(model, _nerd(model, "arm"), compile=False)
    pipeline = newton.CollisionPipeline(model)
    state, control = model.state(), model.control()
    kept = [pipeline.contacts() for _ in range(16)]
    solver.step(state, state, control, kept[0], FRAME_DT)  # the first step runs as it is and records nothing
    for contacts in kept:
        solver.step(state, state, control, contacts, FRAME_DT)
    with test.assertRaisesRegex(RuntimeError, "16 combinations.*same few objects.*use_cuda_graph=False"):
        solver.step(state, state, control, pipeline.contacts(), FRAME_DT)
    # A combination that was recorded still replays, and a solver without graphs takes any number of objects.
    before = _clones(solver.binding.read_state(state))
    solver.step(state, state, control, kept[3], FRAME_DT)
    test.assertFalse(torch.equal(solver.binding.read_state(state)["joint_q"], before["joint_q"]))
    eager = SolverNeRD(model, _nerd(model, "arm"), use_cuda_graph=False, compile=False)
    for _ in range(18):
        eager.step(state, state, control, pipeline.contacts(), FRAME_DT)


def test_notify_model_changed(test: unittest.TestCase, device: torch.device) -> None:
    """Notify the solver of model changes and expect one warning per kind of change the network does not see."""
    model = _robot_model("arm", device)
    solver = SolverNeRD(model, _nerd(model, "arm"), compile=False)
    flags, unseen = newton.ModelFlags, SolverNeRD.UNSEEN_MODEL_CHANGES
    shown = (flags.BODY_INERTIAL_PROPERTIES, flags.SHAPE_PROPERTIES)
    for notified, expected in (
        (flags.JOINT_PROPERTIES, []),  # the forward kinematics of every step applies changed joint frames
        (flags.BODY_INERTIAL_PROPERTIES, [unseen[flags.BODY_INERTIAL_PROPERTIES]]),
        (flags.BODY_INERTIAL_PROPERTIES, []),
        (shown[0] | shown[1] | flags.JOINT_PROPERTIES, [unseen[flags.SHAPE_PROPERTIES]]),
        (flags.ALL, [text for flag, text in unseen.items() if flag not in shown]),
        (flags.ALL, []),
    ):
        with recorded_warnings() as caught:
            solver.notify_model_changed(notified)
        test.assertTrue(all(issubclass(warning.category, NewtonNeuralWarning) for warning in caught))
        test.assertEqual([str(warning.message) for warning in caught], expected)
        # A warning names the line that notified the solver.
        test.assertTrue(all(warning.filename == __file__ for warning in caught))
    test.assertIn("masses", unseen[flags.BODY_INERTIAL_PROPERTIES])
    test.assertEqual(len(unseen), 10)


class TestSolverNeRD(unittest.TestCase):
    """The solver against the eager model, on every device."""


cuda_devices = get_cuda_test_devices()
for mode_name, use_graph in (("eager", False), ("graphed", True)):
    mode_devices = cuda_devices if use_graph else None
    # In the last scene the arm sits at uneven rows, so its state is gathered and scattered, not viewed.
    for scene_name, robot_name in (("quadruped", "quadruped"), ("arm", "arm"), ("arm_with_uneven_boxes", "arm")):
        for layout, same_state in (("two_states", False), ("in_place", True)):
            add_function_test(
                TestSolverNeRD,
                f"test_matches_model_{scene_name}_{layout}_{mode_name}",
                test_matches_model,
                devices=mode_devices,
                scene=scene_name,
                robot=robot_name,
                in_place=same_state,
                graphed=use_graph,
                precision=Precision.FP32,
            )
    for function in (test_probes, test_contact_set, test_contact_force_estimates):
        add_function_test(
            TestSolverNeRD, f"{function.__name__}_{mode_name}", function, devices=mode_devices, graphed=use_graph
        )
for precision_kind in (Precision.TF32, Precision.FP16):
    add_function_test(
        TestSolverNeRD,
        f"test_matches_model_quadruped_{precision_kind.value}_graphed",
        test_matches_model,
        devices=cuda_devices,
        scene="quadruped",
        robot="quadruped",
        in_place=False,
        graphed=True,
        precision=precision_kind,
    )
for layout, same_state in (("two_states", False), ("in_place", True)):
    add_function_test(
        TestSolverNeRD,
        f"test_other_articulations_carry_over_{layout}",
        test_other_articulations_carry_over,
        in_place=same_state,
    )
for function in (
    test_model_control,
    test_repr,
    test_dt_mismatch,
    test_checkpoint_path,
    test_refuses_mismatch,
    test_reset,
    test_reset_restores_whole_worlds,
    test_reset_without_worlds,
    test_restart_through_simulation,
    test_contact_row_overflow,
    test_overflow_of_any_sensor,
    test_refuses_reduced_contacts,
    test_notify_model_changed,
):
    add_function_test(TestSolverNeRD, function.__name__, function)
add_function_test(TestSolverNeRD, "test_new_objects_every_step", test_new_objects_every_step, devices=cuda_devices)
for capture_name, relaxed_capture in (("thread_local", False), ("relaxed", True)):
    for mode_name, use_graph in (("eager", False), ("graphed", True)):
        add_function_test(
            TestSolverNeRD,
            f"test_caller_capture_{capture_name}_{mode_name}",
            test_caller_capture,
            devices=cuda_devices,
            relaxed=relaxed_capture,
            graphed=use_graph,
        )
add_function_test(TestSolverNeRD, "test_capture_on_cpu", test_capture_on_cpu, devices=[torch.device("cpu")])


if __name__ == "__main__":
    unittest.main()
