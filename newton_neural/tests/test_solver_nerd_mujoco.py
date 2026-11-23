# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SolverNeRDMuJoCo``: staged-step parity, free-acc injection, box pushing, resets, and graphs."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_neural import NewtonNeuralWarning
from newton_neural._src.solvers import mujoco_staged_step as staged_module
from newton_neural._src.solvers import solver_nerd_mujoco as solver_module
from newton_neural._src.solvers.solver_nerd_mujoco import SolverNeRDMuJoCo
from newton_neural.data import SimulationNewton
from newton_neural.models import FlashNeRD, Precision
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures, helpers
from newton_neural.tests.extensions import ContactInputRootHeight
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

FRAME_DT = 0.005
WORLD_COUNT = 2

SWEEP_POSE = (0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""Joint coordinates [rad, m] that stretch the arm out sideways, so its first joint sweeps the hand around."""

SWEEP_SPEED = 3.0
"""Velocity of the first joint during a sweep [rad/s]."""


def _nerd(device: torch.device, *, coasting: bool = False) -> FlashNeRD:
    """A small free motion model of the arm for frames of :data:`FRAME_DT`.

    Its weights are random. A coasting model predicts no free acceleration, so the robot keeps its joint
    velocities until a constraint acts on it. The model is deployed in IEEE float32 unless a solver is given
    another precision, so that what a solver hands MuJoCo is the prediction of the eager model to rounding.
    """
    nerd = helpers.make_nerd("arm", device, prediction=PredictionFreeMotion())
    nerd.frame_dt = FRAME_DT
    nerd.precision = Precision.FP32
    if coasting:
        with torch.no_grad():
            nerd.output_normalizer.mean.zero_()
            nerd.output_normalizer.std.fill_(1e-6)
    return nerd


def _random_state(model: newton.Model, system: SystemArticulation) -> newton.State:
    """The model's initial state with the arm at a random pose and velocity."""
    state = model.state()
    frames = fixtures.random_frames(system, WORLD_COUNT, device=wp.device_to_torch(model.device))
    system.bind(model, "arm").write_state(state, frames)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _sweep_state(model: newton.Model, system: SystemArticulation) -> newton.State:
    """The arm stretched out and sweeping, with the box of the model, if any, in the air just ahead of the hand."""
    binding = system.bind(model, "arm")
    state = model.state()
    frames = binding.read_state(state)
    frames["joint_q"][:] = torch.tensor(SWEEP_POSE, device=frames["joint_q"].device)
    frames["joint_qd"][:, 0] = SWEEP_SPEED
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    if model.joint_coord_count > WORLD_COUNT * system.joint_coord_count:
        hand = wp.to_torch(state.body_q)[wp.to_torch(binding.body_ids)[:, 6].long(), :3]
        anchor = wp.to_torch(model.joint_X_p)[wp.to_torch(binding.root_joint_ids).long(), :3]
        # The hand moves along the horizontal tangent of its circle around the first joint's vertical axis.
        radius = hand - anchor
        tangent = torch.nn.functional.normalize(torch.stack([-radius[:, 1], radius[:, 0]], dim=-1), dim=-1)
        box = wp.to_torch(state.joint_q).view(WORLD_COUNT, -1)[:, :3]  # the box is first in every world
        box.copy_(hand)
        box[:, :2] += 0.075 * tangent
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run(
    solver: newton.solvers.SolverBase,
    state: newton.State,
    step_count: int,
    control: newton.Control | None = None,
    *,
    collide: bool = False,
) -> newton.State:
    """The state after ``step_count`` steps of :data:`FRAME_DT` from ``state``, which is left as it was.

    With ``collide`` the solver is given the contacts Newton detects before every step.
    """
    model = solver.model
    state_in, state_out = model.state(), model.state()
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        getattr(state_in, name).assign(getattr(state, name))
    pipeline = newton.CollisionPipeline(model) if collide else None
    contacts = pipeline.contacts() if collide else None
    for _ in range(step_count):
        if collide:
            pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, FRAME_DT)
        state_in, state_out = state_out, state_in
    return state_in


def _newton_order(solver: SolverMuJoCo, values: wp.array) -> torch.Tensor:
    """Per degree of freedom values of MuJoCo Warp ``[world_count, nv]`` in Newton's order, ``[world_count, -1]``."""
    newton_dof = wp.to_torch(solver.mjc_dof_to_newton_dof).long()
    ordered = torch.zeros(solver.model.joint_dof_count, device=newton_dof.device)
    ordered[newton_dof.flatten()] = wp.to_torch(values).flatten()
    return ordered.view(newton_dof.shape[0], -1)


def _joints(state: newton.State, name: str = "joint_q") -> torch.Tensor:
    """A joint array of ``state`` with one row per world, ``[WORLD_COUNT, -1]``."""
    return wp.to_torch(getattr(state, name)).view(WORLD_COUNT, -1)


class TestSolverNeRDMuJoCo(unittest.TestCase):
    pass


def test_replaced_acceleration_reproduces_mujoco(test: TestSolverNeRDMuJoCo, device: torch.device, box: bool) -> None:
    """Reproduce MuJoCo's own step when the free acceleration it computed is fed back as the replacement."""
    model = (fixtures.arm_with_box_model if box else fixtures.arm_model)(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, "arm", control=("joint_f",))
    binding = system.bind(model, "arm")
    for integrator in ("euler", "implicit", "implicitfast"):
        reference = SimulationNewton(
            model, SolverMuJoCo(model, integrator=integrator), system, articulation="arm", frame_dt=FRAME_DT
        )
        label = PredictionFreeMotion().source(reference)
        solver = SolverMuJoCo(model, integrator=integrator)
        free_acceleration, dofs = solver_module._free_acceleration_input(solver, binding)
        test.assertIn("_mujoco_warp_step", vars(solver))

        reference.restart(fixtures.random_frames(system, WORLD_COUNT, device=device))
        state_in, state_out = model.state(), model.state()
        generator = torch.Generator().manual_seed(0)
        for _ in range(4):
            state_in.joint_q.assign(reference.state.joint_q)
            state_in.joint_qd.assign(reference.state.joint_qd)
            reference.step({"joint_f": torch.randn(WORLD_COUNT, system.actuated_count, generator=generator).to(device)})
            free_acceleration[dofs] = label.read()["joint_qdd_free"]
            solver.step(state_in, state_out, reference.control, None, FRAME_DT)
            for name in ("joint_q", "joint_qd"):
                expected, found = getattr(reference.state, name).numpy(), getattr(state_out, name).numpy()
                np.testing.assert_allclose(found, expected, rtol=1e-5, atol=3e-5, err_msg=f"{integrator} {name}")
        test.assertGreater(free_acceleration[dofs].abs().max().item(), 10.0)


def test_prediction_replaces_robot_rows(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Give MuJoCo the model's prediction as the robot's free acceleration and leave the box its own."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    solver = SolverNeRDMuJoCo(model, nerd, articulation="arm")
    test.assertIn("_mujoco_warp_step", vars(solver))  # the solver's own stages of MuJoCo Warp's step
    plain = SolverMuJoCo(model)
    state, control = _random_state(model, nerd.system), model.control()
    wp.to_torch(control.joint_target_q).normal_()

    frames = solver.binding.read_state(state) | solver.binding.read_control(control)
    with torch.no_grad():
        prediction = nerd.predict(frames, nerd.history(WORLD_COUNT))
    learned = _run(solver, state, 1, control)
    own = _run(plain, state, 1, control)

    found, mujoco = _newton_order(solver, solver.mjw_data.qacc_smooth), _newton_order(plain, plain.mjw_data.qacc_smooth)
    # The box is first in every world and falls freely. The arm's eight degrees of freedom follow.
    torch.testing.assert_close(found[:, 6:], prediction, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(found[:, :6], mujoco[:, :6], rtol=1e-5, atol=1e-5)
    test.assertGreater((found[:, 6:] - mujoco[:, 6:]).abs().max().item(), 1.0)
    velocity, own_velocity = _joints(learned, "joint_qd"), _joints(own, "joint_qd")
    test.assertGreater((velocity[:, 6:] - own_velocity[:, 6:]).abs().max().item(), 1e-3)
    torch.testing.assert_close(velocity[:, :6], own_velocity[:, :6], rtol=1e-5, atol=1e-5)


def test_staged_step_reproduces_mujoco(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Reproduce MuJoCo Warp's own step with its stages while no free acceleration is replaced."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    state, control = _random_state(model, SystemArticulation.from_model(model, "arm")), model.control()
    wp.to_torch(control.joint_target_q).normal_()
    replacement = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
    mask = wp.zeros(model.joint_dof_count, dtype=wp.bool, device=model.device)
    for options in ({}, {"use_mujoco_contacts": False}, {"integrator": "implicit"}):
        staged, plain = SolverMuJoCo(model, **options), SolverMuJoCo(model, **options)
        staged_module.replace_mujoco_warp_step(staged, replacement, mask)
        collide = not options.get("use_mujoco_contacts", True)
        found = _run(staged, state, 4, control, collide=collide)
        expected = _run(plain, state, 4, control, collide=collide)
        for name in ("joint_q", "joint_qd"):
            np.testing.assert_allclose(
                getattr(found, name).numpy(),
                getattr(expected, name).numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"{options} {name}",
            )
        moved = found.joint_q.numpy() - state.joint_q.numpy()
        test.assertGreater(np.abs(moved).max(), 1e-3)


def test_construction_takes_no_step(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Leave MuJoCo Warp as it is at construction, so that the first step starts from the state it is given."""
    model = fixtures.arm_model(WORLD_COUNT, device)
    with mock.patch.object(SolverMuJoCo, "step") as step:
        solver = SolverNeRDMuJoCo(model, _nerd(device, coasting=True), articulation="arm")
    step.assert_not_called()
    state = model.state()
    _joints(state)[:, 1] = 0.5
    test.assertAlmostEqual(_joints(_run(solver, state, 1))[0, 1].item(), 0.5, delta=1e-2)


def test_staged_step_guard(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Install MuJoCo Warp's stages under the options they can run and refuse the others."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)

    def replace(model: newton.Model = model, **options) -> SolverMuJoCo:
        solver = SolverMuJoCo(model, **options)
        replacement = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=model.device)
        mask = wp.zeros(model.joint_dof_count, dtype=wp.bool, device=model.device)
        staged_module.replace_mujoco_warp_step(solver, replacement, mask)
        return solver

    for options in ({}, {"use_mujoco_contacts": False}, {"integrator": "implicit"}):
        test.assertIn("_mujoco_warp_step", vars(replace(**options)))
    with (
        mock.patch.object(staged_module, "STAGES", (*staged_module.STAGES, "no_such_stage")),
        test.assertRaisesRegex(RuntimeError, "no function no_such_stage"),
    ):
        replace()
    with test.assertRaisesRegex(ValueError, "integration"):
        replace(integrator="rk4")
    with test.assertRaisesRegex(ValueError, "speculative"):
        replace(use_mujoco_contacts=False, use_speculative_contacts=True)
    with test.assertRaisesRegex(ValueError, "sleep"):
        replace(enable_sleeping=True)
    if device.type == "cpu":
        with test.assertRaisesRegex(ValueError, "use_mujoco_cpu"):
            replace(fixtures.arm_model(1, device), use_mujoco_cpu=True)  # this backend steps one world


def test_refuses_what_it_cannot_run(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Refuse models, robots, MuJoCo settings and step durations the solver cannot honour."""
    model = fixtures.arm_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    with test.assertRaisesRegex(ValueError, "PredictionStateIncrement"):
        SolverNeRDMuJoCo(model, helpers.make_nerd("arm", device))
    with test.assertRaisesRegex(ValueError, "contact input"):
        parts = {"contact_input": ContactInputRootHeight(), "prediction": PredictionFreeMotion()}
        SolverNeRDMuJoCo(model, helpers.make_nerd("arm", device, **parts))
    unfitted = _nerd(device)
    unfitted.frame_dt = None
    with test.assertRaisesRegex(ValueError, "no frame duration.*build the model with frame_dt"):
        SolverNeRDMuJoCo(model, unfitted)
    # A model of free motion cannot be made for a floating base, and the arm's model fits no other robot.
    with test.assertRaisesRegex(ValueError, "fixed base"):
        helpers.make_nerd("quadruped", device, prediction=PredictionFreeMotion())
    with test.assertRaisesRegex(ValueError, "joints"):
        SolverNeRDMuJoCo(fixtures.quadruped_model(WORLD_COUNT, device), nerd, njmax=128)

    with test.assertRaisesRegex(ValueError, "integration"):
        SolverNeRDMuJoCo(model, nerd, integrator="rk4")
    with test.assertRaisesRegex(ValueError, "sleep"):
        SolverNeRDMuJoCo(model, nerd, enable_sleeping=True)
    if device.type == "cpu":
        with test.assertRaisesRegex(ValueError, "use_mujoco_cpu"):
            SolverNeRDMuJoCo(fixtures.arm_model(1, device), nerd, use_mujoco_cpu=True)
    with test.assertRaisesRegex(ValueError, "speculative"):
        SolverNeRDMuJoCo(model, nerd, use_mujoco_contacts=False, use_speculative_contacts=True)

    solver = SolverNeRDMuJoCo(model, nerd)
    state = model.state()
    with test.assertRaisesRegex(ValueError, "0.005 s.*0.01 s"):
        solver.step(state, state, None, None, 2.0 * FRAME_DT)


def test_loads_checkpoint(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Step a saved model in half precision close to the model in memory in full precision."""
    model = fixtures.arm_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    state = _random_state(model, nerd.system)
    expected = _run(SolverNeRDMuJoCo(model, nerd), state, 2).joint_qd.numpy()
    with tempfile.TemporaryDirectory() as folder:
        nerd.save(Path(folder) / "arm.pt")
        solver = SolverNeRDMuJoCo(model, nerd=Path(folder) / "arm.pt", precision="fp16")
    test.assertIsNot(solver.nerd, nerd)
    test.assertEqual(solver.nerd.device.type, device.type)
    test.assertEqual(solver.stepper.precision, Precision.FP16)
    np.testing.assert_allclose(_run(solver, state, 2).joint_qd.numpy(), expected, atol=1e-3)


def test_reset_forgets_masked_worlds(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Clear the history and restore the joint state of the worlds a reset selects, and of no other."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    solver = SolverNeRDMuJoCo(model, nerd, articulation="arm")
    state = _run(solver, _random_state(model, nerd.system), 3)
    before = solver.stepper.export_history()
    joint_q, default = _joints(state).clone(), _joints(model)

    solver.reset(state, wp.array([True, False, False], dtype=wp.bool, device=model.device))
    after = solver.stepper.export_history()
    test.assertEqual(after["step"].tolist(), [0, 3])
    for name, value in after.items():
        test.assertEqual(value[0].abs().max().item(), 0.0, name)
        test.assertTrue(torch.equal(value[1], before[name][1]), name)
    test.assertTrue(torch.equal(_joints(state)[0], default[0]))
    test.assertTrue(torch.equal(_joints(state)[1], joint_q[1]))

    solver.reset(state)
    test.assertEqual(solver.stepper.export_history()["step"].tolist(), [0, 0])
    test.assertTrue(torch.equal(_joints(state), default))
    # A reset solver starts over, so it repeats what a new solver does from the same state.
    again = _run(solver, state, 2)
    fresh = _run(SolverNeRDMuJoCo(model, nerd, articulation="arm"), state, 2)
    np.testing.assert_allclose(again.joint_qd.numpy(), fresh.joint_qd.numpy(), rtol=1e-4, atol=1e-5)


def test_reset_on_caller_stream(test: TestSolverNeRDMuJoCo, device: torch.device, torch_caller: bool) -> None:
    """MuJoCo and neural history must consume the same mask when either library runs on a non-default stream."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    solver = SolverNeRDMuJoCo(model, nerd, articulation="arm", compile=False, use_cuda_graph=False)
    state = _run(solver, _random_state(model, nerd.system), 3)
    before = _joints(state).clone()
    test.assertFalse(torch.equal(before[0], _joints(model)[0]))
    mask = torch.zeros(WORLD_COUNT + 1, dtype=torch.bool, device=device)
    warp_mask = wp.from_torch(mask, dtype=wp.bool)
    stream = torch.cuda.Stream(device)
    torch.cuda.synchronize(device)
    caller_stream = torch.cuda.stream(stream) if torch_caller else wp.ScopedStream(wp.stream_from_torch(stream))
    with caller_stream:
        # Warm up both the mask producer and the reset before checking their ordering.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1)
            mask[:1].fill_(False)
        solver.reset(state, warp_mask)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)  # leave the mask update queued while the other stream can run
            mask[:1].fill_(True)
        solver.reset(state, warp_mask)
    torch.cuda.synchronize(device)
    test.assertEqual(solver.stepper.export_history()["step"].tolist(), [0, 3])
    test.assertTrue(torch.equal(_joints(state)[0], _joints(model)[0]))
    test.assertTrue(torch.equal(_joints(state)[1], before[1]))


def test_restart_order(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Keep a state written after a reset and lose one written before, unless the reset restores nothing."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    solver = SolverNeRDMuJoCo(model, nerd, articulation="arm")
    simulation = SimulationNewton(model, solver, nerd.system, articulation="arm", frame_dt=FRAME_DT)
    binding = simulation.binding
    for _ in range(3):
        simulation.step()
    before = {name: value.clone() for name, value in binding.read_state(simulation.state).items()}
    start = fixtures.random_frames(nerd.system, WORLD_COUNT, device=device, seed=3)

    simulation.restart(start, torch.tensor([True, False], device=device))
    for name, value in binding.read_state(simulation.state).items():
        torch.testing.assert_close(value[0], start[name][0])
        test.assertTrue(torch.equal(value[1], before[name][1]), name)
    test.assertEqual(solver.stepper.export_history()["step"].tolist(), [0, 3])

    # A host that resets the solver itself writes afterwards, because the reset restores the model's joint state.
    state = simulation.state
    solver.reset(state)
    binding.write_state(state, start)
    for name, value in binding.read_state(state).items():
        torch.testing.assert_close(value, start[name])
    solver.reset(state)
    test.assertTrue(torch.equal(_joints(state), _joints(model)))
    # A reset that restores nothing keeps the state written before it.
    binding.write_state(state, start)
    solver.reset(state, flags=0)
    for name, value in binding.read_state(state).items():
        torch.testing.assert_close(value, start[name])


def test_replays_from_graphs(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Step the same from the solver's own CUDA graph, from a caller's graph and without any graph."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    start = _random_state(model, nerd.system)
    eager = SolverNeRDMuJoCo(model, nerd, articulation="arm", use_cuda_graph=False)
    expected = _run(eager, start, 6).joint_qd.numpy()
    test.assertTrue(np.isfinite(expected).all())

    # Six steps alternate between two pairs of states, so each pair's graph is recorded once and replayed twice.
    graphed = SolverNeRDMuJoCo(model, nerd, articulation="arm")
    np.testing.assert_allclose(_run(graphed, start, 6).joint_qd.numpy(), expected, rtol=1e-3, atol=1e-4)
    # A solver says whether it replays from graphs of its own, which none does on a device without CUDA.
    replays = device.type == "cuda"
    test.assertEqual((eager.use_cuda_graph, graphed.use_cuda_graph), (False, replays))
    test.assertEqual(repr(graphed), f"SolverNeRDMuJoCo({graphed.stepper!r}, use_cuda_graph={replays})")

    if device.type == "cuda":
        solver = SolverNeRDMuJoCo(model, nerd, articulation="arm")
        state_0, state_1 = _run(solver, start, 0), model.state()
        solver.step(state_0, state_1, None, None, FRAME_DT)
        solver.step(state_1, state_0, None, None, FRAME_DT)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_0, state_1, None, None, FRAME_DT)
            solver.step(state_1, state_0, None, None, FRAME_DT)
        # A capture records the two steps without taking them.
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(state_0.joint_qd.numpy(), expected, rtol=1e-3, atol=1e-4)


def test_inference_mode_change(test: TestSolverNeRDMuJoCo, device: torch.device, warm_up_inside: bool) -> None:
    """Take the first step inside or outside inference mode and later steps in the other mode, and compile once."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    nerd = _nerd(device)
    start = _random_state(model, nerd.system)
    eager = SolverNeRDMuJoCo(model, nerd, articulation="arm", use_cuda_graph=False, compile=False)
    expected = _run(eager, start, 4).joint_qd.numpy()

    solver = SolverNeRDMuJoCo(model, nerd, articulation="arm", compile=True)
    state_0, state_1 = _run(solver, start, 0), model.state()
    with torch.inference_mode(warm_up_inside):
        solver.step(state_0, state_1, None, None, FRAME_DT)
    # A reinforcement learning loop warms up in one mode and records its steps into a CUDA graph in the other.
    # A graph capture cannot compile, so the later steps must find the step compiled. On CUDA they are recorded.
    with torch.inference_mode(not warm_up_inside), torch.compiler.set_stance("fail_on_recompile"):
        for _ in range(3):
            state_0, state_1 = state_1, state_0
            solver.step(state_0, state_1, None, None, FRAME_DT)
    np.testing.assert_allclose(state_1.joint_qd.numpy(), expected, rtol=1e-3, atol=1e-4)


def test_pushes_a_box(test: TestSolverNeRDMuJoCo, device: torch.device, collide: bool) -> None:
    """Push a free box with the hand, and let the box slow the robot."""
    nerd = _nerd(device, coasting=True)
    alone = fixtures.arm_model(WORLD_COUNT, device)
    scene = fixtures.arm_with_box_model(WORLD_COUNT, device)
    options = {"use_mujoco_contacts": not collide}
    step_count = 40

    unhindered = _run(
        SolverNeRDMuJoCo(alone, nerd, **options), _sweep_state(alone, nerd.system), step_count, collide=collide
    )
    # The model predicts no free acceleration, so an arm that meets nothing keeps sweeping at its starting speed.
    test.assertAlmostEqual(_joints(unhindered)[0, 0].item(), SWEEP_SPEED * step_count * FRAME_DT, delta=1e-3)

    start = _sweep_state(scene, nerd.system)
    # The arm by its label and by its articulation indices in the scene, where a box comes before every arm.
    arms = [index for index, label in enumerate(scene.articulation_label) if label == "arm"]
    test.assertEqual(arms, [1, 3])
    for articulation in ("arm", arms):
        solver = SolverNeRDMuJoCo(scene, nerd, articulation=articulation, **options)
        end = _run(solver, start, step_count, collide=collide)
        # The box is the first seven joint coordinates of a world and the arm the last eight.
        travel = (_joints(end)[:, :2] - _joints(start)[:, :2]).norm(dim=-1)
        test.assertGreater(travel.min().item(), 0.1)
        slowed = (_joints(end)[:, 7:] - _joints(unhindered)).abs().max().item()
        test.assertGreater(slowed, 0.05)
        # The bodies the solver wrote are those that forward kinematics gives for its joint state.
        bodies = scene.state()
        newton.eval_fk(scene, end.joint_q, end.joint_qd, bodies)
        np.testing.assert_allclose(end.body_q.numpy(), bodies.body_q.numpy(), atol=1e-4)


def test_keeps_robot_constraints(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Stop the robot at its joint limit and at the ground, and follow a changed limit."""
    nerd = _nerd(device, coasting=True)
    model = fixtures.arm_model(WORLD_COUNT, device)
    binding = nerd.system.bind(model, "arm")
    first_dofs = torch.as_tensor(binding.joint_dof_ids[:, 0], device=device)
    wp.to_torch(model.joint_limit_lower)[first_dofs] = -0.2
    wp.to_torch(model.joint_limit_upper)[first_dofs] = 0.2
    solver = SolverNeRDMuJoCo(model, nerd)
    start = _sweep_state(model, nerd.system)

    # Unhindered, the first joint of the arm would reach 0.3 rad in these steps.
    test.assertLess(_joints(_run(solver, start, 20))[:, 0].max().item(), 0.22)

    wp.to_torch(model.joint_limit_upper)[first_dofs] = 0.4
    # MuJoCo Warp takes the limit over. The network, which does not see it, is why the solver warns.
    with test.assertWarns(NewtonNeuralWarning):
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    solver.reset(model.state())
    # Now it passes the old limit and turns back at the new one, short of the 0.45 rad it would reach.
    angle = _joints(_run(solver, start, 30))[:, 0]
    test.assertGreater(angle.min().item(), 0.3)
    test.assertLess(angle.max().item(), 0.42)

    def hand_height(second_joint: float, speed: float, step_count: int) -> torch.Tensor:
        """Height of the hand [m] after steps that start with the second joint at an angle [rad] and a speed [rad/s]."""
        solver.reset(model.state())
        state = model.state()
        frames = binding.read_state(state)
        frames["joint_q"][:, 1] = second_joint
        frames["joint_qd"][:, 1] = speed
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        end = _run(solver, state, step_count)
        return wp.to_torch(end.body_q)[wp.to_torch(binding.body_ids)[:, 6].long(), 2]

    # Swinging down at 3 rad/s for 0.2 s, the arm would reach the pose that puts the hand under the ground.
    test.assertLess(hand_height(1.9 + 3.0 * 40 * FRAME_DT, 0.0, 0).max().item(), -0.1)
    test.assertGreater(hand_height(1.9, 3.0, 40).min().item(), 0.0)


def test_notify_model_changed(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Build the solver without a warning, then expect one warning per kind of change the network does not see."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    flags, unseen = newton.ModelFlags, SolverNeRDMuJoCo.UNSEEN_MODEL_CHANGES
    with warnings.catch_warnings():
        # The constructor of SolverMuJoCo notifies itself of the whole model, which is no change.
        warnings.simplefilter("error", NewtonNeuralWarning)
        solver = SolverNeRDMuJoCo(model, _nerd(device), articulation="arm")
    for notified, expected in (
        (
            flags.SHAPE_PROPERTIES | flags.JOINT_PROPERTIES,
            [],
        ),  # MuJoCo Warp applies them, and the network needs neither
        (flags.BODY_INERTIAL_PROPERTIES, [unseen[flags.BODY_INERTIAL_PROPERTIES]]),
        (flags.BODY_INERTIAL_PROPERTIES | flags.MODEL_PROPERTIES, [unseen[flags.MODEL_PROPERTIES]]),
        (
            flags.ALL,
            [
                text
                for flag, text in unseen.items()
                if flag not in (flags.BODY_INERTIAL_PROPERTIES, flags.MODEL_PROPERTIES)
            ],
        ),
        (flags.ALL, []),
    ):
        with helpers.recorded_warnings() as caught:
            solver.notify_model_changed(notified)
        caught = [warning for warning in caught if issubclass(warning.category, NewtonNeuralWarning)]
        test.assertEqual([str(warning.message) for warning in caught], expected)
        # A warning names the line that notified the solver.
        test.assertTrue(all(warning.filename == __file__ for warning in caught))
    test.assertIn("masses", unseen[flags.BODY_INERTIAL_PROPERTIES])


def test_labels_are_free_accelerations(test: TestSolverNeRDMuJoCo, device: torch.device) -> None:
    """Read as labels the free accelerations MuJoCo Warp's data holds for the arm, in Newton's order."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, "arm", control=("joint_f",))
    simulation = SimulationNewton(model, SolverMuJoCo(model), system, articulation="arm", frame_dt=FRAME_DT)
    solver = simulation.solver
    source = PredictionFreeMotion().source(simulation)

    simulation.restart(fixtures.random_frames(system, WORLD_COUNT, device=device))
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        simulation.step({"joint_f": torch.randn(WORLD_COUNT, system.actuated_count, generator=generator).to(device)})
        label = source.read()["joint_qdd_free"]
        test.assertEqual(label.shape, (WORLD_COUNT, system.joint_dof_count))
        test.assertGreater(label.abs().max().item(), 10.0)
        # The arm's degrees of freedom follow the six of the box in every world.
        from_data = _newton_order(solver, solver.mjw_data.qacc_smooth)[:, 6:]
        torch.testing.assert_close(label, from_data, rtol=0.0, atol=1e-6)


for _box in (False, True):
    add_function_test(
        TestSolverNeRDMuJoCo,
        f"test_replaced_acceleration_reproduces_mujoco_{'box' if _box else 'arm'}",
        test_replaced_acceleration_reproduces_mujoco,
        box=_box,
    )
for _warm_up_inside in (False, True):
    add_function_test(
        TestSolverNeRDMuJoCo,
        f"test_inference_mode_change_warm_up_{'inside' if _warm_up_inside else 'outside'}",
        test_inference_mode_change,
        warm_up_inside=_warm_up_inside,
    )
for _collide in (False, True):
    add_function_test(
        TestSolverNeRDMuJoCo,
        f"test_pushes_a_box_{'newton' if _collide else 'mujoco'}_contacts",
        test_pushes_a_box,
        collide=_collide,
    )
for _function in (
    test_prediction_replaces_robot_rows,
    test_staged_step_reproduces_mujoco,
    test_construction_takes_no_step,
    test_staged_step_guard,
    test_refuses_what_it_cannot_run,
    test_loads_checkpoint,
    test_keeps_robot_constraints,
    test_reset_forgets_masked_worlds,
    test_restart_order,
    test_replays_from_graphs,
    test_notify_model_changed,
    test_labels_are_free_accelerations,
):
    add_function_test(TestSolverNeRDMuJoCo, _function.__name__, _function)

for _caller in ("torch", "warp"):
    add_function_test(
        TestSolverNeRDMuJoCo,
        f"test_reset_on_{_caller}_stream",
        test_reset_on_caller_stream,
        devices=get_cuda_test_devices(),
        torch_caller=_caller == "torch",
    )


if __name__ == "__main__":
    unittest.main()
