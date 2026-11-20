# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SimulationNewton``: substep parity, FK/contact freshness, masked restarts, and CUDA graphs."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.core import rotation as rot
from newton_neural.data import SimulationBase, SimulationNewton
from newton_neural.systems import SystemArticulation
from newton_neural.tests import fixtures
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

FRAME_DT = 0.01
WORLD_COUNT = 4

ROBOTS: dict[str, Callable[..., newton.Model]] = {"arm": fixtures.arm_model, "quadruped": fixtures.quadruped_model}

SOLVERS: dict[str, Callable[[newton.Model], newton.solvers.SolverBase]] = {
    # The constraint rows of four feet on the ground exceed MuJoCo Warp's default capacity.
    "mujoco": lambda model: newton.solvers.SolverMuJoCo(model, njmax=128),
    "featherstone": newton.solvers.SolverFeatherstone,
    "xpbd": newton.solvers.SolverXPBD,
}


def _commands(system: SystemArticulation, frame_count: int, device: torch.device) -> list[Frames]:
    """Random joint forces [N or N*m] for ``frame_count`` frames, each ``[WORLD_COUNT, actuated_count]``."""
    generator = torch.Generator().manual_seed(0)
    forces = torch.randn(frame_count, WORLD_COUNT, system.actuated_count, generator=generator).to(device)
    return [{"joint_f": force} for force in forces]


def _solver_alone(
    model: newton.Model,
    solver: newton.solvers.SolverBase,
    system: SystemArticulation,
    commands: list[Frames],
    substep_count: int,
    *,
    in_place: bool,
    collide: bool,
) -> newton.State:
    """The state a solver reaches from the model's initial state in a loop written as Newton's examples do."""
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model) if collide else None
    contacts = pipeline.contacts() if collide else None
    binding = system.bind(model)
    for command in commands:
        binding.write_control(control, command)
        for _ in range(substep_count):
            state.clear_forces()
            if collide:
                pipeline.collide(state, contacts)
            if in_place:
                solver.step(state, state, control, contacts, FRAME_DT / substep_count)
            else:
                solver.step(state, next_state, control, contacts, FRAME_DT / substep_count)
                state, next_state = next_state, state
    return state


def _forward_kinematics(model: newton.Model, state: newton.State) -> np.ndarray:
    """Body transforms ``[body_count, 7]`` that follow from the joint coordinates of ``state``."""
    scratch = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, scratch)
    return scratch.body_q.numpy()


def _contact_rows(contacts: newton.Contacts) -> np.ndarray:
    """The detected rigid contacts ``[contact_count, 11]`` (shapes, points, normal) in a fixed order."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    rows = np.concatenate(
        [
            contacts.rigid_contact_shape0.numpy()[:count, None],
            contacts.rigid_contact_shape1.numpy()[:count, None],
            contacts.rigid_contact_point0.numpy()[:count],
            contacts.rigid_contact_point1.numpy()[:count],
            contacts.rigid_contact_normal.numpy()[:count],
        ],
        axis=1,
    )
    return rows[np.lexsort(rows.T[::-1])]


def _detected(model: newton.Model, state: newton.State) -> np.ndarray:
    """The contact rows a new collision pipeline detects for ``state``."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return _contact_rows(contacts)


class TestSimulationNewton(unittest.TestCase):
    pass


def test_step_matches_solver(test: TestSimulationNewton, device: torch.device, robot: str, solver: str) -> None:
    """Advance the state exactly as the solver alone does, with one and with several solver steps per frame."""
    in_place = solver == "mujoco"  # the other solvers are stepped with two states that swap
    collide = solver != "mujoco"  # MuJoCo Warp detects contacts itself
    # On CUDA, contacts are found and their forces summed in an order that changes from run to run.
    tolerance = 1e-6 if device.type == "cpu" else 1e-3
    for substep_count in (1, 3):
        model = ROBOTS[robot](WORLD_COUNT, device)
        system = SystemArticulation.from_model(model, control=("joint_f",))
        commands = _commands(system, 5, device)
        simulation = SimulationNewton(
            model,
            SOLVERS[solver](model),
            system,
            frame_dt=FRAME_DT,
            substep_count=substep_count,
            collision_pipeline=newton.CollisionPipeline(model) if collide else None,
        )
        test.assertIsInstance(simulation, SimulationBase)
        test.assertEqual(simulation.world_count, WORLD_COUNT)
        test.assertEqual(simulation.device, device)
        test.assertEqual((simulation.frame_dt, simulation.substep_count), (FRAME_DT, substep_count))
        initial = simulation.state.body_q.numpy().copy()
        for command in commands:
            simulation.step(command)

        expected = _solver_alone(
            model, SOLVERS[solver](model), system, commands, substep_count, in_place=in_place, collide=collide
        )
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            reached, reference = getattr(simulation.state, name).numpy(), getattr(expected, name).numpy()
            np.testing.assert_allclose(reached, reference, rtol=tolerance, atol=tolerance, err_msg=name)
        test.assertGreater(np.abs(simulation.state.body_q.numpy() - initial).max(), 1e-3)
        consumed = simulation.binding.read_control(simulation.control)["joint_f"]
        test.assertTrue(torch.equal(consumed, commands[-1]["joint_f"]))


def test_body_transforms_current(test: TestSimulationNewton, device: torch.device, robot: str, solver: str) -> None:
    """Keep body transforms consistent with the joint coordinates at construction and after every step."""
    model = ROBOTS[robot](WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, control=("joint_f",))
    simulation = SimulationNewton(model, SOLVERS[solver](model), system, frame_dt=FRAME_DT, substep_count=2)
    np.testing.assert_allclose(simulation.state.body_q.numpy(), _forward_kinematics(model, simulation.state), atol=1e-5)
    for command in _commands(system, 3, device):
        simulation.step(command)
        np.testing.assert_allclose(
            simulation.state.body_q.numpy(), _forward_kinematics(model, simulation.state), atol=1e-5
        )


def test_restart_masked(test: TestSimulationNewton, device: torch.device, robot: str, solver: str) -> None:
    """Write the state of the masked worlds only, with unit quaternions and current body transforms."""
    model = ROBOTS[robot](WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, control=("joint_f",))
    simulation = SimulationNewton(model, SOLVERS[solver](model), system, frame_dt=FRAME_DT)
    for command in _commands(system, 3, device):
        simulation.step(command)
    binding = simulation.binding
    before = {name: value.clone() for name, value in binding.read_state(simulation.state).items()}
    bodies_before = simulation.state.body_q.numpy().copy()

    frames = fixtures.random_frames(system, WORLD_COUNT, device=device, seed=1)
    written = {name: value.clone() for name, value in frames.items()}
    if system.floating_base:
        frames["joint_q"][:, 3:7] *= 3.0  # a sampled quaternion has no unit length
    mask = torch.tensor([True, False, True, False], device=device)
    simulation.restart(frames, mask)

    state = binding.read_state(simulation.state)
    for name in ("joint_q", "joint_qd"):
        test.assertTrue(torch.equal(state[name][~mask], before[name][~mask]), name)
        torch.testing.assert_close(state[name][mask], written[name][mask], atol=1e-6, rtol=0.0, msg=name)
    bodies = simulation.state.body_q.numpy()
    kept = binding.body_ids.numpy()[~mask.cpu().numpy()]
    np.testing.assert_array_equal(bodies[kept], bodies_before[kept])
    np.testing.assert_allclose(bodies, _forward_kinematics(model, simulation.state), atol=1e-5)

    simulation.restart(frames)
    state = binding.read_state(simulation.state)
    torch.testing.assert_close(state["joint_q"], written["joint_q"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(simulation.state.body_q.numpy(), _forward_kinematics(model, simulation.state), atol=1e-5)


def test_restart_resets_solver(test: TestSimulationNewton, device: torch.device) -> None:
    """Reset the solver for the masked worlds only, which with MuJoCo puts a moved box back in its place."""
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, "arm", control=("joint_f",))
    solver = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, solver, system, articulation="arm", frame_dt=FRAME_DT)
    # The box holds the first seven joint coordinates of each world. Lift it and let it fall for a while.
    place = model.joint_q.numpy().reshape(WORLD_COUNT, -1)[:, :7]
    wp.to_torch(simulation.state.joint_q).view(WORLD_COUNT, -1)[:, 2] += 0.5
    for _ in range(3):
        simulation.step()
    frames = fixtures.random_frames(system, WORLD_COUNT, device=device, seed=2)
    mask = torch.tensor([False, True, True, False], device=device)
    simulation.restart(frames, mask)

    box = simulation.state.joint_q.numpy().reshape(WORLD_COUNT, -1)[:, :7]
    restarted = mask.cpu().numpy()
    np.testing.assert_allclose(box[restarted], place[restarted], atol=1e-6)
    test.assertTrue((box[~restarted, 2] - place[~restarted, 2] > 0.4).all())
    state = simulation.binding.read_state(simulation.state)
    torch.testing.assert_close(state["joint_q"][mask], frames["joint_q"][mask])
    # MuJoCo forgot the forces it applied to the restarted worlds, and kept those of the others.
    simulation.step({"joint_f": torch.ones(WORLD_COUNT, system.actuated_count, device=device)})
    simulation.restart(frames, mask)
    applied = solver.mjw_data.qfrc_applied.numpy()
    test.assertTrue((applied[restarted] == 0.0).all())
    test.assertTrue((applied[~restarted, 6:] == 1.0).all())


def test_restart_without_worlds(test: TestSimulationNewton, device: torch.device) -> None:
    """Restart a robot built without worlds, which lies in Newton's global world, with current body transforms."""
    builder = newton.ModelBuilder()
    fixtures.add_quadruped(builder)
    builder.add_ground_plane()
    model = builder.finalize(device=str(device))
    test.assertLess(model.joint_world.numpy().max(), 0)
    system = SystemArticulation.from_model(model, control=("joint_f",))
    simulation = SimulationNewton(model, newton.solvers.SolverFeatherstone(model), system, frame_dt=FRAME_DT)
    frames = fixtures.random_frames(system, 1, device=device, seed=3)
    written = {name: value.clone() for name, value in frames.items()}
    frames["joint_q"][:, 3:7] *= 3.0  # a sampled quaternion has no unit length
    simulation.restart(frames)

    state = simulation.binding.read_state(simulation.state)
    for name in ("joint_q", "joint_qd"):
        torch.testing.assert_close(state[name], written[name], atol=1e-6, rtol=0.0, msg=name)
    expected = model.state()
    expected.joint_q.assign(simulation.state.joint_q)
    expected.joint_qd.assign(simulation.state.joint_qd)
    newton.eval_fk(model, expected.joint_q, expected.joint_qd, expected)
    np.testing.assert_allclose(simulation.state.body_q.numpy(), expected.body_q.numpy(), atol=1e-6)


def test_contacts_current(test: TestSimulationNewton, device: torch.device) -> None:
    """Hold the contacts of the current state after every step and after a restart."""
    model = fixtures.quadruped_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model, control=("joint_f",))
    simulation = SimulationNewton(
        model,
        newton.solvers.SolverFeatherstone(model),
        system,
        frame_dt=FRAME_DT,
        substep_count=2,
        collision_pipeline=newton.CollisionPipeline(model),
    )
    np.testing.assert_array_equal(_contact_rows(simulation.contacts), _detected(model, simulation.state))
    for command in _commands(system, 4, device):
        previous = _contact_rows(simulation.contacts)
        simulation.step(command)
        current = _contact_rows(simulation.contacts)
        np.testing.assert_array_equal(current, _detected(model, simulation.state))
        test.assertFalse(current.shape == previous.shape and np.array_equal(current, previous))

    # Lift two worlds off the ground. Their feet leave it, and the other worlds keep their contacts.
    frames = {name: value.clone() for name, value in simulation.binding.read_state(simulation.state).items()}
    frames["joint_q"][:, 2] += 5.0
    mask = torch.tensor([True, False, True, False], device=device)
    simulation.restart(frames, mask)
    current = _contact_rows(simulation.contacts)
    np.testing.assert_array_equal(current, _detected(model, simulation.state))
    shape_world = model.shape_world.numpy()
    pairs = shape_world[current[:, :2].astype(int)]  # world of both shapes of every contact, -1 for the ground
    on_ground = pairs[(pairs == -1).any(axis=1)].max(axis=1)
    test.assertEqual(set(on_ground.tolist()), {1, 3})


def test_no_collision_pipeline(test: TestSimulationNewton, device: torch.device) -> None:
    """Hold no contacts without a collision pipeline."""
    model = fixtures.arm_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model)
    simulation = SimulationNewton(model, newton.solvers.SolverFeatherstone(model), system, frame_dt=FRAME_DT)
    simulation.step()
    test.assertIsNone(simulation.contacts)
    test.assertIsNone(simulation.collision_pipeline)


def test_bounds_are_those_of_the_binding(test: TestSimulationNewton, device: torch.device) -> None:
    """Hand out the bounds of the binding, and turn a bound that is written as a state into a valid one."""
    model = fixtures.quadruped_model(WORLD_COUNT, device)
    system = SystemArticulation.from_model(model)
    simulation = SimulationNewton(model, newton.solvers.SolverFeatherstone(model), system, frame_dt=FRAME_DT)
    bounds = simulation.state_bounds()
    for found, expected in (
        (bounds, simulation.binding.state_bounds()),
        (simulation.command_bounds(), simulation.binding.command_bounds()),
    ):
        test.assertEqual(list(found), list(expected))
        for name, (low, high) in expected.items():
            test.assertTrue(torch.equal(found[name][0], low) and torch.equal(found[name][1], high), name)

    # The lower bound is no valid state, since its quaternion has no unit length. Writing it makes it one.
    simulation.restart({name: bound.expand(WORLD_COUNT, -1) for name, (bound, _) in bounds.items()})
    quaternion = simulation.binding.read_state(simulation.state)["joint_q"][:, 3:7]
    torch.testing.assert_close(quaternion, rot.normalize(-torch.ones(WORLD_COUNT, 4, device=device)))


def test_step_on_a_stream_of_the_caller(test: TestSimulationNewton, device: torch.device) -> None:
    """Restart and step from a busy PyTorch stream of the caller and expect the frame of the default stream, the
    one that starts from the state given and takes the command given."""
    model = fixtures.arm_model(2, device)
    system = SystemArticulation.from_model(model, "arm", control=("joint_f",))
    solver = newton.solvers.SolverMuJoCo(model)
    simulation = SimulationNewton(model, solver, system, articulation="arm", frame_dt=FRAME_DT)
    start = fixtures.random_frames(system, 2, device=device, seed=3)
    state = {name: start[name] for name in ("joint_q", "joint_qd")}
    command = {"joint_f": torch.ones(2, system.actuated_count, device=device)}

    def frame(busy: bool, force: float) -> Tensor:
        simulation.restart({name: torch.zeros_like(value) for name, value in state.items()})
        simulation.step({"joint_f": torch.zeros_like(command["joint_f"])})
        torch.cuda.synchronize(device)
        if not busy:
            simulation.restart(state)
            simulation.step({"joint_f": force * command["joint_f"]})
            return simulation.binding.read_state(simulation.state)["joint_qd"].clone()
        with torch.cuda.stream(torch.cuda.Stream(device)):
            work = torch.randn(4096, 4096, device=device)
            for _ in range(24):
                work = torch.nn.functional.normalize(work @ work)
            simulation.restart({name: value.clone() for name, value in state.items()})
            simulation.step({"joint_f": force * command["joint_f"].clone()})
            found = simulation.binding.read_state(simulation.state)["joint_qd"].clone()
        torch.cuda.synchronize(device)
        return found

    expected = frame(busy=False, force=1.0)
    test.assertGreater((expected - frame(busy=False, force=0.0)).abs().max().item(), 1e-3)  # the command counts
    for _ in range(3):
        torch.testing.assert_close(frame(busy=True, force=1.0), expected, rtol=1e-5, atol=1e-6)


def test_refuses_bad_arguments(test: TestSimulationNewton, device: torch.device) -> None:
    """Refuse a solver of another model, a frame without solver steps and a model with external actuators."""
    model = fixtures.arm_model(2, device)
    system = SystemArticulation.from_model(model)
    other = newton.solvers.SolverFeatherstone(fixtures.arm_model(2, device))
    with test.assertRaisesRegex(ValueError, "another model"):
        SimulationNewton(model, other, system, frame_dt=FRAME_DT)
    with test.assertRaisesRegex(ValueError, "substep_count"):
        SimulationNewton(model, newton.solvers.SolverFeatherstone(model), system, frame_dt=FRAME_DT, substep_count=0)
    # Newton leaves external actuators to the caller of the solver, and the simulation would step without them.
    robot = newton.ModelBuilder()
    fixtures.add_arm(robot)
    robot.add_actuator(newton.actuators.DrivePD, index=0, kp=100.0, kd=10.0)
    actuated = robot.finalize(device=str(device))
    with test.assertRaisesRegex(ValueError, "external actuators"):
        SimulationNewton(actuated, newton.solvers.SolverFeatherstone(actuated), system, frame_dt=FRAME_DT)


for _robot, _solver in (
    ("arm", "mujoco"),
    ("arm", "featherstone"),
    ("quadruped", "mujoco"),
    ("quadruped", "featherstone"),
    ("quadruped", "xpbd"),
):
    add_function_test(
        TestSimulationNewton,
        f"test_step_matches_solver_{_robot}_{_solver}",
        test_step_matches_solver,
        robot=_robot,
        solver=_solver,
    )
for _robot, _solver in (("arm", "mujoco"), ("quadruped", "featherstone")):
    for _function in (test_body_transforms_current, test_restart_masked):
        add_function_test(
            TestSimulationNewton, f"{_function.__name__}_{_robot}_{_solver}", _function, robot=_robot, solver=_solver
        )
for _function in (
    test_restart_resets_solver,
    test_restart_without_worlds,
    test_contacts_current,
    test_no_collision_pipeline,
    test_bounds_are_those_of_the_binding,
    test_refuses_bad_arguments,
):
    add_function_test(TestSimulationNewton, _function.__name__, _function)
add_function_test(
    TestSimulationNewton,
    test_step_on_a_stream_of_the_caller.__name__,
    test_step_on_a_stream_of_the_caller,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
