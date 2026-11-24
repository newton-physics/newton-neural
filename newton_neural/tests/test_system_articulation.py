# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SystemArticulation`` and ``BindingArticulation``: invariance, zero-copy views, and bounds."""

from __future__ import annotations

import dataclasses
import json
import math
import unittest
from collections.abc import Callable
from typing import Any

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.core import rotation as rot
from newton_neural._src.core.signal import SignalRole
from newton_neural._src.systems.articulation import robot_frame_floating
from newton_neural._src.systems.articulation_binding import BindingArticulation, leaf
from newton_neural.systems import BindingBase, SystemArticulation, SystemBase, robot_frame
from newton_neural.tests import fixtures
from newton_neural.tests.unittest_utils import add_function_test

CONTROL = ("joint_target_q", "joint_f")

FIXTURES: dict[str, tuple[Callable[..., newton.Model], str]] = {
    "quadruped": (fixtures.quadruped_model, "quadruped"),
    "arm": (fixtures.arm_model, "arm"),
    "arm_with_box": (fixtures.arm_with_box_model, "arm"),
    "arm_with_uneven_boxes": (fixtures.arm_with_uneven_boxes_model, "arm"),
}
"""Model function and articulation pattern of every fixture."""


@wp.kernel
def _floating_frames_kernel(
    up: wp.vec3,
    root_body: int,
    body_ids: wp.array2d[int],
    body_q: wp.array[wp.transform],
    frames: wp.array[wp.transform],
):
    world = wp.tid()
    frames[world] = robot_frame_floating(body_q[body_ids[world, root_body]], up)


@wp.kernel
def _robot_frames_kernel(
    body_q: wp.array[wp.transform],
    joint_X_p: wp.array[wp.transform],
    joint_child: wp.array[int],
    up: wp.vec3,
    root_joint_ids: wp.array[int],
    floating_base: wp.bool,
    frames: wp.array[wp.transform],
):
    world = wp.tid()
    frames[world] = robot_frame(world, body_q, joint_X_p, joint_child, up, root_joint_ids, floating_base)


def _system(
    fixture: str, world_count: int, device: torch.device
) -> tuple[newton.Model, SystemArticulation, BindingArticulation]:
    """A fixture's model with the system that describes its robot and the system's binding."""
    model_function, articulation = FIXTURES[fixture]
    model = model_function(world_count, device)
    system = SystemArticulation.from_model(model, articulation, control=CONTROL).to(device)
    return model, system, system.bind(model, articulation)


def _floating_frames(system: SystemArticulation, binding: BindingArticulation, state: newton.State) -> Tensor:
    """The robot frame of a floating base in every world ``[world_count, 7]``, as a Warp kernel computes it."""
    frames = wp.zeros(binding.world_count, dtype=wp.transform, device=state.body_q.device)
    wp.launch(
        _floating_frames_kernel,
        dim=binding.world_count,
        inputs=[wp.vec3(*system.up.tolist()), system.root_body, binding.body_ids, state.body_q],
        outputs=[frames],
        device=state.body_q.device,
    )
    return wp.to_torch(frames).clone()


def _replica_frames(system: SystemArticulation, binding: BindingArticulation, model: newton.Model) -> Tensor:
    """The transform ``[world_count, 7]`` that carries a fixed base from the system's anchor, where the
    kinematic replica holds it, to the root joint of each world."""
    root = wp.to_torch(model.joint_X_p)[wp.to_torch(binding.root_joint_ids).long()]
    anchor = root.new_tensor(system.anchor)
    turn = rot.multiply(root[:, 3:], rot.conjugate(anchor[3:]).expand_as(root[:, 3:]))
    return torch.cat((root[:, :3] - rot.rotate(turn, anchor[:3]), turn), dim=-1)


def _posed(model: newton.Model, binding: BindingArticulation, frames: Frames) -> tuple[newton.State, Tensor, Tensor]:
    """A state holding ``frames``, with the poses ``[world_count, body_count, 7]`` and the velocities
    ``[world_count, body_count, 6]`` of the articulation's bodies from forward kinematics."""
    state = model.state()
    binding.write_state(state, frames)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_ids = wp.to_torch(binding.body_ids).long()
    return state, wp.to_torch(state.body_q)[body_ids], wp.to_torch(state.body_qd)[body_ids]


def _model_rows(model: newton.Model, articulation: str, labels: tuple[str, ...], starts: np.ndarray) -> np.ndarray:
    """Rows ``[world_count, len(labels)]`` where the joints with ``labels`` start in a joint array, found by
    walking the labels of the Newton model, independently of the binding."""
    articulation_world = model.articulation_world.numpy()
    joint_articulation = model.joint_articulation.numpy()
    rows = np.zeros((model.world_count, len(labels)), dtype=np.int64)
    for joint, label in enumerate(model.joint_label):
        index = joint_articulation[joint]
        if leaf(model.articulation_label[index]) == articulation and leaf(label) in labels:
            rows[articulation_world[index], labels.index(leaf(label))] = starts[joint]
    return rows


def _windows(system: SystemArticulation, device: torch.device) -> tuple[Frames, Frames]:
    """Float64 windows of consecutive random frames, frames 0 to 4 and frames 1 to 5 of 4 worlds."""
    frames = {k: v.double() for k, v in fixtures.random_frames(system, 4, 6, device=device).items()}
    return {k: v[:, :-1] for k, v in frames.items()}, {k: v[:, 1:] for k, v in frames.items()}


def test_from_model(test: unittest.TestCase, device: torch.device, fixture: str, world_count: int) -> None:
    """Describe a fixture's robot and check its sizes, labels and signals."""
    model_function, articulation = FIXTURES[fixture]
    system = SystemArticulation.from_model(model_function(world_count, device), articulation, control=CONTROL)
    spec = system.spec
    if fixture == "quadruped":
        test.assertTrue(system.floating_base)
        test.assertEqual((spec.joint_coord_count, spec.joint_dof_count, system.actuated_count), (19, 18, 12))
        test.assertEqual(spec.body_labels[:3], ("base", "LF_HIP", "LF_THIGH"))
        test.assertEqual(spec.joint_labels[:2], ("floating_base", "LF_HAA"))
        test.assertEqual([joint.type for joint in spec.joints], ["free"] + ["revolute"] * 12)
        test.assertEqual(spec.actuated_joint_labels[:2], ("LF_HAA", "LF_HFE"))
        # joint_q[7:] (12), base velocities (6), joint_qd[6:] (12), gravity direction (3), two controls (24)
        test.assertEqual((system.feature_count, system.increment_count), (57, 36))
    else:
        test.assertFalse(system.floating_base)
        test.assertEqual((spec.joint_coord_count, spec.joint_dof_count, system.actuated_count), (8, 8, 8))
        test.assertEqual(spec.body_labels[5:], ("link6", "hand", "left_finger", "right_finger"))
        test.assertEqual([joint.type for joint in spec.joints], ["revolute"] * 6 + ["fixed"] + ["prismatic"] * 2)
        test.assertEqual(spec.actuated_joint_labels[5:], ("joint6", "left_finger_joint", "right_finger_joint"))
        test.assertEqual((system.feature_count, system.increment_count), (32, 16))
        test.assertEqual(spec.joints[7].axis, (0.0, 1.0, 0.0))
        np.testing.assert_allclose(np.array(system.anchor[1:3]), fixtures.ARM_ANCHOR[1:], rtol=1e-6)
    test.assertEqual(spec.joints[0].parent, -1)
    test.assertEqual(system.root_body, 0)
    np.testing.assert_allclose(spec.gravity, (0.0, 0.0, -9.81), rtol=1e-6)
    torch.testing.assert_close(system.up, torch.tensor([0.0, 0.0, 1.0]))
    test.assertEqual(list(system.signals), ["joint_q", "joint_qd", *CONTROL])
    test.assertEqual(system.signals["joint_q"].shape, (spec.joint_coord_count,))
    test.assertEqual(system.signals["joint_qd"].role, SignalRole.STATE)
    test.assertEqual(system.signals["joint_f"].shape, (system.actuated_count,))
    test.assertEqual(system.signals["joint_f"].role, SignalRole.CONTROL)
    test.assertEqual(len(list(system.parameters())), 0)
    for frame_count in (None, 3):
        frames = fixtures.random_frames(system, 2, frame_count, device=device)
        features = system.to(device).features(frames)
        test.assertEqual(features.shape, (*frames["joint_q"].shape[:-1], system.feature_count))
        test.assertEqual(features.dtype, torch.float32)


def test_config_round_trip(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Rebuild a system from its config after a pass through JSON text, directly and through the registry."""
    model_function, articulation = FIXTURES[fixture]
    system = SystemArticulation.from_model(model_function(2, device), articulation, control=CONTROL)
    config = json.loads(json.dumps(system.config()))
    test.assertEqual(SystemArticulation.Spec.from_dict(config["spec"]), system.spec)
    for rebuilt in (SystemArticulation.from_config(config), SystemBase.create("SystemArticulation", config)):
        test.assertIsInstance(rebuilt, SystemArticulation)
        test.assertEqual(rebuilt.spec, system.spec)
        test.assertEqual(rebuilt.config(), system.config())
        test.assertEqual(list(rebuilt.signals.values()), list(system.signals.values()))
        frames = fixtures.random_frames(system, 2)
        torch.testing.assert_close(rebuilt.features(frames), system.features(frames))


def test_integrate_inverts_increment(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Integrate the increment between random frames, with any base rotation and quaternion sign, and reach them."""
    model_function, articulation = FIXTURES[fixture]
    system = SystemArticulation.from_model(model_function(1, device), articulation).to(device)
    current, following = _windows(system, device)
    if system.floating_base:
        # a base that turns by almost half a turn in one frame, and one whose quaternion flips sign without turning
        turn = rot.from_rotation_vector(torch.tensor([1.86, 0.0, 2.48], dtype=torch.float64, device=device))
        following["joint_q"][0, 0, 3:7] = rot.multiply(turn, current["joint_q"][0, 0, 3:7])
        following["joint_q"][1, 0, 3:7] = -current["joint_q"][1, 0, 3:7]
    increment = system.increment(current, following)
    test.assertEqual(increment.shape, (4, 5, system.increment_count))
    reached = system.integrate(current, increment)
    expected = following["joint_q"].clone()
    if system.floating_base:
        test.assertGreater(float(increment[0, 0, 3:6].norm()), 3.0)
        test.assertLess(float(increment[1, 0, 3:6].norm()), 1e-9)
        test.assertGreaterEqual(float(reached["joint_q"][..., 6].min()), 0.0)
        expected[..., 3:7] = rot.canonicalize(rot.normalize(expected[..., 3:7]))
    torch.testing.assert_close(reached["joint_q"], expected, atol=1e-9, rtol=0)
    torch.testing.assert_close(reached["joint_qd"], following["joint_qd"], atol=1e-9, rtol=0)
    # float32 frames of one instant, as a solver steps them
    single = {k: v[:, 0].float() for k, v in current.items()}
    after = {k: v[:, 0].float() for k, v in following.items()}
    step = system.increment(single, after)
    reached = system.integrate(single, step)
    torch.testing.assert_close(reached["joint_q"], expected[:, 0].float(), atol=1e-4, rtol=0)
    torch.testing.assert_close(reached["joint_qd"], after["joint_qd"], atol=1e-4, rtol=0)
    # the default perturbation applies the noise as an increment
    torch.testing.assert_close(system.perturb(single, step)["joint_q"], reached["joint_q"], atol=0, rtol=0)


def test_features_ignore_place_and_heading(test: unittest.TestCase, device: torch.device) -> None:
    """Move the whole robot and turn it about the vertical, and get the same features and increments."""
    system = SystemArticulation.from_model(fixtures.quadruped_model(1, device), control=CONTROL).to(device)
    current, following = _windows(system, device)
    turn = rot.from_rotation_vector(torch.tensor([0.0, 0.0, 2.3], dtype=torch.float64, device=device))
    shift = torch.tensor([20.0, -7.0, 3.0], dtype=torch.float64, device=device)

    def moved(frames: Frames) -> Frames:
        joint_q, joint_qd = frames["joint_q"].clone(), frames["joint_qd"].clone()
        joint_q[..., :3] = rot.rotate(turn, joint_q[..., :3]) + shift
        joint_q[..., 3:7] = rot.multiply(turn, joint_q[..., 3:7])
        joint_qd[..., :3] = rot.rotate(turn, joint_qd[..., :3])
        joint_qd[..., 3:6] = rot.rotate(turn, joint_qd[..., 3:6])
        return {**frames, "joint_q": joint_q, "joint_qd": joint_qd}

    torch.testing.assert_close(system.features(moved(current)), system.features(current), atol=1e-9, rtol=0)
    torch.testing.assert_close(
        system.increment(moved(current), moved(following)), system.increment(current, following), atol=1e-9, rtol=0
    )
    # the sign of the base quaternion carries no information either
    flipped = {**current, "joint_q": current["joint_q"].clone()}
    flipped["joint_q"][..., 3:7] *= -1.0
    torch.testing.assert_close(system.features(flipped), system.features(current), atol=1e-12, rtol=0)
    # the tilt is seen, and the gravity direction of an upright base points down its own z axis
    upright = {k: v[:, 0].clone() for k, v in current.items()}
    upright["joint_q"][:, 3:7] = turn
    gravity = system.features(upright)[:, 30:33]
    torch.testing.assert_close(gravity, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64, device=device).expand(4, 3))


def test_binding_round_trip(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Write frames into a Newton state, read them back, and write only the worlds a mask selects."""
    model_function, articulation = FIXTURES[fixture]
    model = model_function(3, device)
    system = SystemArticulation.from_model(model, articulation=articulation, control=CONTROL)
    binding = system.bind(model, articulation=articulation)
    test.assertEqual(binding.world_count, 3)
    test.assertEqual(binding.view.count_per_world, 1)
    # The binding built by hand is the one the system hands out, with the Newton model first.
    by_hand = BindingArticulation(model, system, articulation=articulation)
    test.assertIs(by_hand.system, system)
    np.testing.assert_array_equal(by_hand.body_ids.numpy(), binding.body_ids.numpy())
    state = model.state()
    untouched = state.joint_q.numpy().copy()
    frames = fixtures.random_frames(system, 3, device=device)
    binding.write_state(state, frames)
    read = binding.read_state(state)
    test.assertEqual(list(read), ["joint_q", "joint_qd"])
    torch.testing.assert_close(read["joint_q"], frames["joint_q"])
    torch.testing.assert_close(read["joint_qd"], frames["joint_qd"])
    # everything outside the articulation, such as the boxes next to an arm, keeps its value
    rows = _model_rows(model, articulation, system.spec.joint_labels[:1], model.joint_q_start.numpy())
    inside = (rows + np.arange(system.joint_coord_count)).ravel()
    outside = np.setdiff1d(np.arange(model.joint_coord_count), inside)
    np.testing.assert_array_equal(state.joint_q.numpy()[outside], untouched[outside])
    np.testing.assert_array_equal(state.joint_q.numpy()[inside], frames["joint_q"].cpu().numpy().ravel())

    other = fixtures.random_frames(system, 3, device=device, seed=1)
    mask = torch.tensor([True, False, True], device=device)
    binding.write_state(state, other, mask)
    read = binding.read_state(state)
    for key in ("joint_q", "joint_qd"):
        torch.testing.assert_close(read[key][mask], other[key][mask])
        torch.testing.assert_close(read[key][~mask], frames[key][~mask])
    # writing what was read changes nothing
    before = state.joint_q.numpy().copy()
    binding.write_state(state, binding.read_state(state))
    np.testing.assert_array_equal(state.joint_q.numpy(), before)
    # a single row goes to every world that the mask selects, as torch broadcasts it
    binding.write_state(state, {key: value[0] for key, value in other.items()}, mask)
    read = binding.read_state(state)
    for key in ("joint_q", "joint_qd"):
        torch.testing.assert_close(read[key][mask], other[key][:1].expand(2, -1))
        torch.testing.assert_close(read[key][~mask], frames[key][~mask])
    # a signal whose shape does not broadcast is refused by name, with both shapes
    width = system.joint_dof_count
    with test.assertRaisesRegex(ValueError, rf"joint_qd is written as \[3, {width}\].*got \[2, {width}\]"):
        binding.write_state(state, {**frames, "joint_qd": frames["joint_qd"][:2]})


def test_binding_is_zero_copy(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Check that tensors read from evenly spaced rows share memory with the Warp arrays, in both directions."""
    model_function, articulation = FIXTURES[fixture]
    model = model_function(3, device)
    system = SystemArticulation.from_model(model, articulation, control=CONTROL)
    binding = system.bind(model, articulation)
    state, control = model.state(), model.control()
    first_row = int(_model_rows(model, articulation, system.spec.joint_labels[:1], model.joint_q_start.numpy())[0, 0])
    joint_q = binding.read_state(state)["joint_q"]
    test.assertEqual(joint_q.data_ptr(), state.joint_q.ptr + 4 * first_row)
    joint_q[2, 1] = 123.0
    test.assertEqual(int((state.joint_q.numpy() == 123.0).sum()), 1)
    state.joint_qd.fill_(0.5)
    expected = torch.full((3, system.joint_dof_count), 0.5, device=device)
    torch.testing.assert_close(binding.read_state(state)["joint_qd"], expected)
    joint_f = binding.read_control(control)["joint_f"]
    joint_f.fill_(-2.0)
    test.assertEqual(int((control.joint_f.numpy() == -2.0).sum()), 3 * system.actuated_count)


def test_binding_gathers_uneven_rows(test: unittest.TestCase, device: torch.device) -> None:
    """Read copies where worlds place the robot at uneven rows, and reach the state through the write methods."""
    model = fixtures.arm_with_uneven_boxes_model(3, device)
    system = SystemArticulation.from_model(model, "arm", control=CONTROL)
    binding = system.bind(model, "arm")
    state = model.state()
    read = binding.read_state(state)["joint_q"]
    read.fill_(9.0)  # a copy, which the state does not see
    test.assertEqual(int((state.joint_q.numpy() == 9.0).sum()), 0)
    binding.write_state(state, {"joint_q": read, "joint_qd": binding.read_state(state)["joint_qd"]})
    test.assertEqual(int((state.joint_q.numpy() == 9.0).sum()), 3 * system.joint_coord_count)
    torch.testing.assert_close(binding.read_state(state)["joint_q"], read)


def test_control_rows(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Read and write each control array at the rows of the actuated joints, found by label in every world."""
    model_function, articulation = FIXTURES[fixture]
    model = model_function(3, device)
    names = ("joint_target_q", "joint_target_qd", "joint_f", "joint_act")
    system = SystemArticulation.from_model(model, articulation, control=names)
    # One control is also named by its string alone.
    test.assertEqual(SystemArticulation.from_model(model, articulation, control="joint_f").spec.control, ("joint_f",))
    binding = system.bind(model, articulation)
    spec = system.spec
    actuated = tuple(
        label for label, joint in zip(spec.joint_labels, spec.joints, strict=True) if joint.type != "fixed"
    )[1 if system.floating_base else 0 :]
    test.assertEqual(len(actuated), system.actuated_count)
    control = model.control()
    for array_name in names:
        starts = model.joint_q_start.numpy() if array_name == "joint_target_q" else model.joint_qd_start.numpy()
        rows = _model_rows(model, articulation, actuated, starts)
        array = getattr(control, array_name)
        values = np.arange(array.shape[0], dtype=np.float32)
        array.assign(values)
        read = binding.read_control(control)[array_name]
        test.assertEqual(read.shape, (3, system.actuated_count))
        np.testing.assert_array_equal(read.cpu().numpy(), values[rows])

        written = torch.full((3, system.actuated_count), -1.0, device=device) - torch.arange(3, device=device)[:, None]
        binding.write_control(control, {**binding.read_control(control), array_name: written})
        result = getattr(control, array_name).numpy()
        np.testing.assert_array_equal(result[rows], written.cpu().numpy())
        rest = np.setdiff1d(np.arange(array.shape[0]), rows.ravel())
        np.testing.assert_array_equal(result[rest], values[rest])
        with test.assertRaisesRegex(ValueError, rf"{array_name} is written as \[3, \d+\].*got \[2, \d+\]"):
            binding.write_control(control, {**binding.read_control(control), array_name: written[:2]})


def test_kinematic_model(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Pose the original model and its joints-only replica alike and compare every body's pose and velocity."""
    model_function, articulation = FIXTURES[fixture]
    model = model_function(3, device)
    system = SystemArticulation.from_model(model, articulation)
    binding = system.bind(model, articulation)
    frames = fixtures.random_frames(system, 3, device=device)
    _, body_q, body_qd = _posed(model, binding, frames)

    replica = system.kinematic_model(3, device=device)
    test.assertEqual((replica.world_count, replica.shape_count), (3, 0))
    test.assertEqual(replica.body_count, 3 * len(system.spec.body_labels))
    replica_binding = system.bind(replica)
    _, replica_q, replica_qd = _posed(replica, replica_binding, frames)

    if not system.floating_base:
        # the replica's worlds coincide at the anchor, from where each world's robot is one transform away
        frame = _replica_frames(system, binding, model)[:, None]
        replica_q = torch.cat(
            (
                frame[..., :3] + rot.rotate(frame[..., 3:], replica_q[..., :3]),
                rot.multiply(frame[..., 3:].expand_as(replica_q[..., 3:]), replica_q[..., 3:]),
            ),
            dim=-1,
        )
        torch.testing.assert_close(frame[0, 0], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device))
        test.assertAlmostEqual(float(frame[2, 0, 0] - frame[1, 0, 0]), fixtures.WORLD_SPACING[0], places=5)
    torch.testing.assert_close(body_q[..., :3], replica_q[..., :3], atol=2e-5, rtol=0)
    alignment = (body_q[..., 3:] * replica_q[..., 3:]).sum(-1).abs()
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1e-5, rtol=0)
    torch.testing.assert_close(body_qd, replica_qd, atol=1e-4, rtol=0)


def test_robot_frame_floating(test: unittest.TestCase, device: torch.device) -> None:
    """Compare the Warp robot frame of a floating base with the base position and the twist of its rotation."""
    model = fixtures.quadruped_model(4, device)
    system = SystemArticulation.from_model(model).to(device)
    binding = system.bind(model)
    frames = fixtures.random_frames(system, 4, device=device)
    state, _, _ = _posed(model, binding, frames)
    frame = _floating_frames(system, binding, state)
    torch.testing.assert_close(frame[:, :3], frames["joint_q"][:, :3])
    heading = rot.twist(rot.normalize(frames["joint_q"][:, 3:7]), system.up)
    torch.testing.assert_close(rot.canonicalize(frame[:, 3:]), heading, atol=1e-6, rtol=0)


def test_robot_frame(test: unittest.TestCase, device: torch.device) -> None:
    """Expect the Warp robot frame of a world to be the heading frame of a floating base and the anchor frame
    of a fixed base, where the root joint of that world attaches it."""
    for fixture in ("quadruped", "arm_with_uneven_boxes"):
        model, system, binding = _system(fixture, 3, device)
        state, _, _ = _posed(model, binding, fixtures.random_frames(system, 3, device=device))
        frames = wp.zeros(3, dtype=wp.transform, device=model.device)
        wp.launch(
            _robot_frames_kernel,
            dim=3,
            inputs=[
                state.body_q,
                model.joint_X_p,
                model.joint_child,
                wp.vec3(*system.up.tolist()),
                binding.root_joint_ids,
                system.floating_base,
            ],
            outputs=[frames],
            device=model.device,
        )
        if system.floating_base:
            expected = _floating_frames(system, binding, state)
        else:
            expected = wp.to_torch(model.joint_X_p)[wp.to_torch(binding.root_joint_ids).long()]
            # The worlds stand apart, so each has an anchor frame of its own.
            test.assertFalse(torch.equal(expected[0], expected[1]))
        test.assertTrue(torch.equal(wp.to_torch(frames), expected), fixture)


def test_mismatch_is_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Bind systems to models with other joints, labels, types or gravity and get a ValueError each time."""
    arm_model = fixtures.arm_model(2, device)
    arm = SystemArticulation.from_model(arm_model)
    quadruped = SystemArticulation.from_model(fixtures.quadruped_model(2, device))
    with test.assertRaisesRegex(ValueError, "9 joints, the system has 13"):
        quadruped.bind(arm_model)

    spec = arm.spec
    labels = ("joint1", "elbow", *spec.joint_labels[2:])
    with test.assertRaisesRegex(ValueError, "joint labels in world 0 differ"):
        SystemArticulation(dataclasses.replace(spec, joint_labels=labels)).bind(arm_model)
    bodies = (*spec.body_labels[:-1], "thumb")
    with test.assertRaisesRegex(ValueError, "body labels in world 0 differ"):
        SystemArticulation(dataclasses.replace(spec, body_labels=bodies)).bind(arm_model)
    joints = (*spec.joints[:-1], dataclasses.replace(spec.joints[-1], type="revolute"))
    with test.assertRaisesRegex(ValueError, "'right_finger_joint' in world 0 is prismatic"):
        SystemArticulation(dataclasses.replace(spec, joints=joints)).bind(arm_model)
    joints = (*spec.joints[:-1], dataclasses.replace(spec.joints[-1], parent=5))
    with test.assertRaisesRegex(ValueError, "connect other bodies"):
        SystemArticulation(dataclasses.replace(spec, joints=joints)).bind(arm_model)
    with test.assertRaisesRegex(ValueError, "gravity"):
        SystemArticulation(dataclasses.replace(spec, gravity=(0.0, 0.0, -1.62))).bind(arm_model)
    # the message names the first world whose gravity differs, with that gravity
    uneven = fixtures.arm_model(3, device)
    gravity = uneven.gravity.numpy()
    gravity[2] = (0.0, 0.0, -1.62)
    uneven.gravity.assign(gravity)
    with test.assertRaisesRegex(ValueError, r"gravity \[.*-1\.62\d*\] of world 2 differs from \[.*-9\.81\d*\]"):
        arm.bind(uneven)
    # a robot that is moved and turned about gravity meets the gravity of the system, a tilted one does not
    robot = newton.ModelBuilder()
    fixtures.add_arm(robot)

    def mounted(*rotations: wp.quat) -> newton.Model:
        scene = newton.ModelBuilder()
        for index, rotation in enumerate(rotations):
            scene.add_world(robot, xform=wp.transform((2.0 * index, 1.0, 0.5), rotation))
        return scene.finalize(device=str(device))

    turned = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.0)
    upside_down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    arm.bind(mounted(wp.quat_identity(), turned))
    with test.assertRaisesRegex(ValueError, r"gravity \[.*9\.81\d*\] of world 1 differs from \[.*-9\.81\d*\].*tilted"):
        arm.bind(mounted(wp.quat_identity(), upside_down))
    # a system read from the tilted robot holds its mount
    SystemArticulation.from_model(mounted(upside_down)).bind(mounted(upside_down, upside_down))
    # a robot with the same labels and tree, and one link a centimetre longer or one axis turned
    frame = spec.joints[2].parent_xform
    longer = dataclasses.replace(spec.joints[2], parent_xform=(frame[0], frame[1], frame[2] + 0.01, *frame[3:]))
    turned = dataclasses.replace(spec.joints[2], axis=(1.0, 0.0, 0.0))
    for joint in (longer, turned):
        joints = (*spec.joints[:2], joint, *spec.joints[3:])
        with test.assertRaisesRegex(ValueError, "another frame or axis"):
            SystemArticulation(dataclasses.replace(spec, joints=joints)).bind(arm_model)
    # a quaternion and its negative are one frame
    flipped = dataclasses.replace(spec.joints[2], parent_xform=(*frame[:3], *(-x for x in frame[3:])))
    SystemArticulation(dataclasses.replace(spec, joints=(*spec.joints[:2], flipped, *spec.joints[3:]))).bind(arm_model)

    # a world whose robot differs from the first world's
    scene = newton.ModelBuilder()
    for label in ("joint6", "wrist"):
        robot = newton.ModelBuilder()
        fixtures.add_arm(robot)
        robot.joint_label[5] = label
        scene.add_world(robot)
    with test.assertRaisesRegex(ValueError, "joint labels in world 1 differ"):
        arm.bind(scene.finalize(device=str(device)))


def test_selection_is_one_articulation_per_world(test: unittest.TestCase, device: torch.device) -> None:
    """Refuse patterns that match several articulations per world or none, and accept a model without worlds."""
    model = fixtures.arm_with_box_model(2, device)
    with test.assertRaisesRegex(ValueError, "matches 2 articulations per world"):
        SystemArticulation.from_model(model)
    arm = SystemArticulation.from_model(model, "arm")
    with test.assertRaisesRegex(ValueError, "matches 2 articulations per world"):
        arm.bind(model)
    with test.assertRaisesRegex(KeyError, r"'humanoid' matches no articulation.*\['box0_articulation', 'arm'\]"):
        SystemArticulation.from_model(model, "humanoid")
    scene = newton.ModelBuilder()
    for add_robot in (fixtures.add_arm, fixtures.add_quadruped):
        robot = newton.ModelBuilder()
        add_robot(robot)
        scene.add_world(robot)
    with test.assertRaisesRegex(ValueError, "in only 1 worlds"):
        SystemArticulation.from_model(scene.finalize(device=str(device)), "arm")

    builder = newton.ModelBuilder()
    fixtures.add_arm(builder)
    single = builder.finalize(device=str(device))
    system = SystemArticulation.from_model(single)
    binding = system.bind(single)
    test.assertEqual(binding.world_count, 1)
    test.assertEqual(binding.read_state(single.state())["joint_q"].shape, (1, 8))


def test_unsupported_articulations_are_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Refuse ball joints, unknown controls, a floating base without gravity, an offset free joint and labels
    of bodies or actuated joints that repeat."""
    builder = newton.ModelBuilder()
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="base")
    arm = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="arm")
    builder.add_articulation([builder.add_joint_free(base), builder.add_joint_ball(base, arm, label="shoulder")])
    with test.assertRaisesRegex(ValueError, "'shoulder' is a ball joint"):
        SystemArticulation.from_model(builder.finalize(device=str(device)))

    model = fixtures.quadruped_model(1, device)
    with test.assertRaisesRegex(ValueError, "control must hold distinct names"):
        SystemArticulation.from_model(model, control=("joint_torque",))
    spec = SystemArticulation.from_model(model).spec
    with test.assertRaisesRegex(ValueError, "needs gravity"):
        SystemArticulation(dataclasses.replace(spec, gravity=(0.0, 0.0, 0.0)))
    offset = dataclasses.replace(spec.joints[0], parent_xform=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    with test.assertRaisesRegex(ValueError, "free joint 'floating_base' has a parent or child transform"):
        SystemArticulation(dataclasses.replace(spec, joints=(offset, *spec.joints[1:])))

    # Two bodies or two actuated joints that share the final component of their label, as left and right
    # copies of one part do, cannot be told apart by it. Fixed joints may share one, as those of a USD asset do.
    robot = newton.ModelBuilder()
    inertia = wp.mat33(np.eye(3))
    base = robot.add_link(mass=1.0, inertia=inertia, label="robot/base")
    joints = [robot.add_joint_fixed(-1, base, label="robot/base/PhysicsFixedJoint")]
    for side in ("left", "right"):
        tip = robot.add_link(mass=1.0, inertia=inertia, label=f"robot/{side}_tip")
        joints.append(robot.add_joint_revolute(base, tip, axis=(0.0, 0.0, 1.0), label=f"robot/{side}_hinge"))
    flange = robot.add_link(mass=1.0, inertia=inertia, label="robot/flange")
    joints.append(robot.add_joint_fixed(tip, flange, label="robot/flange/PhysicsFixedJoint"))
    robot.add_articulation(joints, label="robot")
    model = robot.finalize(device=str(device))
    system = SystemArticulation.from_model(model)
    spec = system.spec
    test.assertEqual(spec.joint_labels, ("PhysicsFixedJoint", "left_hinge", "right_hinge", "PhysicsFixedJoint"))
    test.assertEqual(system.bind(model).world_count, 1)
    bodies = ("base", "tip", "tip", "flange")
    with test.assertRaisesRegex(ValueError, r"body labels \['tip'\] repeat among \['base', 'tip', 'tip', 'flange'\]"):
        SystemArticulation(dataclasses.replace(spec, body_labels=bodies))
    hinges = ("PhysicsFixedJoint", "hinge", "hinge", "PhysicsFixedJoint")
    with test.assertRaisesRegex(ValueError, r"actuated joint labels \['hinge'\] repeat.*a label of its own"):
        SystemArticulation(dataclasses.replace(spec, joint_labels=hinges))


def test_spec_derives_its_counts(test: unittest.TestCase, device: torch.device) -> None:
    """Expect the widths of the joint state and the kind of base to follow from the joints of a spec alone."""
    expected = {"quadruped": (19, 18, True), "arm": (8, 8, False)}
    for fixture, (coord_count, dof_count, floating_base) in expected.items():
        spec = _system(fixture, 1, device)[1].spec
        test.assertEqual((spec.joint_coord_count, spec.joint_dof_count, spec.floating_base), expected[fixture])
        # A spec stores none of the three, so it cannot contradict its joints.
        stored = {field.name for field in dataclasses.fields(spec)}
        test.assertFalse(stored & {"joint_coord_count", "joint_dof_count", "floating_base"})
        test.assertEqual(set(spec.to_dict()), stored)
        with test.assertRaises(TypeError):
            dataclasses.replace(spec, joint_coord_count=coord_count - 1)
        with test.assertRaises(AttributeError):
            spec.floating_base = not floating_base
        # Without its last joint, which has one coordinate and one degree of freedom, a spec counts one less.
        shorter = dataclasses.replace(
            spec,
            joints=spec.joints[:-1],
            joint_labels=spec.joint_labels[:-1],
            body_labels=spec.body_labels[:-1],
            body_com=spec.body_com[:-1],
        )
        test.assertEqual((shorter.joint_coord_count, shorter.joint_dof_count), (coord_count - 1, dof_count - 1))
        system = SystemArticulation(shorter)
        test.assertEqual((system.joint_coord_count, system.joint_dof_count), (coord_count - 1, dof_count - 1))


def test_printed_form(test: unittest.TestCase, device: torch.device) -> None:
    """Print a system and expect the kind of base, the joint counts and the controls."""
    system = SystemArticulation.from_model(fixtures.quadruped_model(1, device))
    test.assertEqual(
        repr(system),
        "SystemArticulation(floating_base=True, joint_coord_count=19, joint_dof_count=18, control=('joint_target_q',))",
    )
    arm = SystemArticulation.from_model(fixtures.arm_model(1, device), control=("joint_target_q", "joint_f"))
    test.assertIn("floating_base=False, joint_coord_count=8, joint_dof_count=8", repr(arm))
    test.assertIn("control=('joint_target_q', 'joint_f')", repr(arm))


def test_includes_actuators(test: unittest.TestCase, device: torch.device) -> None:
    """Expect a system to stand for the robot with its actuators exactly when it consumes no joint forces."""
    model = fixtures.arm_model(1, device)
    for control, expected in (
        (("joint_target_q",), True),
        (("joint_target_q", "joint_target_qd"), True),
        (("joint_f",), False),
        (("joint_target_q", "joint_act"), False),
        ((), True),
    ):
        test.assertIs(SystemArticulation.from_model(model, control=control).includes_actuators, expected, control)


def test_binding_interface(test: unittest.TestCase, device: torch.device) -> None:
    """Expect the binding of an articulation to be a binding, a system without a place to say so, and a system
    that lacks a part of the mapping to be refused when it is built."""
    model = fixtures.arm_model(2, device)
    system = SystemArticulation.from_model(model)
    binding = system.bind(model)
    test.assertIsInstance(binding, BindingBase)
    test.assertIs(binding.system, system)
    members = ("read_state", "read_control", "write_state", "write_control", "state_bounds", "command_bounds")
    test.assertEqual(BindingBase.__abstractmethods__, frozenset(members))
    with test.assertRaisesRegex(TypeError, "abstract"):
        BindingBase()

    class SystemWithoutIntegrate(SystemBase):
        def features(self, frames: Frames) -> Tensor:
            return frames["joint_q"]

        def increment(self, current: Frames, following: Frames) -> Tensor:
            return following["joint_q"] - current["joint_q"]

        def config(self) -> dict[str, Any]:
            return {}

    with test.assertRaisesRegex(TypeError, "abstract.*SystemWithoutIntegrate.*integrate"):
        SystemWithoutIntegrate()

    class SystemWithoutPlace(SystemWithoutIntegrate):
        def integrate(self, current: Frames, increment: Tensor) -> Frames:
            return {"joint_q": current["joint_q"] + increment}

    with test.assertRaisesRegex(NotImplementedError, "SystemWithoutPlace has no bind.*BindingBase"):
        SystemWithoutPlace().bind(model)


def test_joint_rows(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Expect the binding to name the rows of the joint arrays that hold the robot in every world."""
    model, system, binding = _system(fixture, 3, device)
    coords, dofs = binding.joint_coord_ids, binding.joint_dof_ids
    for rows, count in ((coords, system.joint_coord_count), (dofs, system.joint_dof_count)):
        test.assertIsInstance(rows, np.ndarray)
        test.assertTrue(np.issubdtype(rows.dtype, np.integer))
        test.assertEqual(rows.shape, (3, count))
    state = model.state()
    wp.to_torch(state.joint_q).copy_(torch.arange(model.joint_coord_count, device=device))
    wp.to_torch(state.joint_qd).copy_(torch.arange(model.joint_dof_count, device=device))
    frames = binding.read_state(state)
    test.assertTrue(torch.equal(frames["joint_q"].cpu(), torch.as_tensor(coords, dtype=torch.float32)))
    test.assertTrue(torch.equal(frames["joint_qd"].cpu(), torch.as_tensor(dofs, dtype=torch.float32)))
    # The rows start where the root joint of the robot starts, world by world.
    root_joints = binding.root_joint_ids.numpy()
    np.testing.assert_array_equal(coords[:, 0], model.joint_q_start.numpy()[root_joints])
    np.testing.assert_array_equal(dofs[:, 0], model.joint_qd_start.numpy()[root_joints])


def test_bounds(test: unittest.TestCase, device: torch.device) -> None:
    """Read the bounds of the first world's robot from the Newton model, with the documented ones where it has none."""
    model = fixtures.arm_with_uneven_boxes_model(3, device)
    control = ("joint_target_q", "joint_target_qd", "joint_f", "joint_act")
    system = SystemArticulation.from_model(model, "arm", control=control)
    binding = system.bind(model, "arm")
    # Limits of the eight degrees of freedom of the arm in the first world, where one box precedes it. The
    # second world, where two boxes do, gets other limits, which the bounds must not read.
    first, second = binding.joint_dof_ids[0], binding.joint_dof_ids[1]
    lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    velocity, effort = model.joint_velocity_limit.numpy(), model.joint_effort_limit.numpy()
    lower[first], upper[first] = -newton.MAXVAL, newton.MAXVAL
    lower[first[1]], upper[first[1]] = -0.5, 0.7
    upper[first[3]] = 2.0
    lower[first[4]], upper[first[5]] = 5.0, -4.0  # one limit each, farther from zero than the default reach
    lower[first[6:]], upper[first[6:]] = 0.0, 0.04
    velocity[first], effort[first] = 1.0e6, 5.0
    velocity[first[0]], velocity[first[7]], effort[first[2]] = 3.0, 50.0, 1.0e6
    lower[second], upper[second], velocity[second], effort[second] = -0.1, 0.1, 0.5, 0.25
    for array, values in (
        (model.joint_limit_lower, lower),
        (model.joint_limit_upper, upper),
        (model.joint_velocity_limit, velocity),
        (model.joint_effort_limit, effort),
    ):
        array.assign(values)

    pi = math.pi
    q_low = torch.tensor([-pi, -0.5, -pi, -pi, 5.0, -4.0 - pi, 0.0, 0.0], device=device)
    q_high = torch.tensor([pi, 0.7, pi, 2.0, 5.0 + pi, -4.0, 0.04, 0.04], device=device)
    speed = torch.tensor([3.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], device=device)
    force = torch.tensor([5.0, 5.0, math.inf, 5.0, 5.0, 5.0, 5.0, 5.0], device=device)
    unbounded = torch.full((8,), math.inf, device=device)
    expected = {
        "joint_q": (q_low, q_high),
        "joint_qd": (-speed, speed),
        "joint_target_q": (q_low, q_high),
        "joint_target_qd": (-speed, speed),
        "joint_f": (-force, force),
        "joint_act": (-unbounded, unbounded),
    }
    state, command = binding.state_bounds(), binding.command_bounds()
    test.assertEqual(list(state), ["joint_q", "joint_qd"])
    test.assertEqual(list(command), list(control))
    for name, (low, high) in (state | command).items():
        test.assertTrue(torch.equal(low, expected[name][0]), name)
        test.assertTrue(torch.equal(high, expected[name][1]), name)
        test.assertEqual((low.dtype, low.device.type), (torch.float32, device.type), name)

    # Joints without limits reach pi rad, or 1 m for a prismatic joint.
    model = fixtures.arm_model(2, device)
    low, high = SystemArticulation.from_model(model).bind(model).state_bounds()["joint_q"]
    reach = torch.tensor([pi] * 6 + [1.0] * 2, device=device)  # six angles, then two lengths
    test.assertTrue(torch.equal(low, -reach) and torch.equal(high, reach))

    # A floating base starts near its place in the first world, with any orientation and bounded velocities.
    model = fixtures.quadruped_model(2, device)
    binding = SystemArticulation.from_model(model).bind(model)
    place = torch.as_tensor(model.joint_q.numpy()[:3], device=device)
    low, high = binding.state_bounds()["joint_q"]
    test.assertTrue(torch.equal(low[:7], torch.cat((place - 0.25, -torch.ones(4, device=device)))))
    test.assertTrue(torch.equal(high[:7], torch.cat((place + 0.25, torch.ones(4, device=device)))))
    test.assertTrue(torch.equal(high[7:], torch.full((12,), pi, device=device)))
    speed = torch.tensor([1.0] * 3 + [2.0] * 3 + [20.0] * 12, device=device)
    slowest, fastest = binding.state_bounds()["joint_qd"]
    test.assertTrue(torch.equal(slowest, -speed) and torch.equal(fastest, speed))
    target_low, target_high = binding.command_bounds()["joint_target_q"]
    test.assertTrue(torch.equal(target_high, high[7:]))
    test.assertTrue(torch.equal(target_low, low[7:]))


class TestSystemArticulation(unittest.TestCase):
    pass


for _fixture in FIXTURES:
    for _world_count in (1, 3):
        add_function_test(
            TestSystemArticulation,
            f"test_from_model_{_fixture}_{_world_count}",
            test_from_model,
            fixture=_fixture,
            world_count=_world_count,
        )
    for _function in (test_binding_round_trip, test_control_rows, test_kinematic_model, test_joint_rows):
        add_function_test(TestSystemArticulation, f"{_function.__name__}_{_fixture}", _function, fixture=_fixture)
for _fixture in ("quadruped", "arm", "arm_with_box"):
    add_function_test(
        TestSystemArticulation, f"test_binding_is_zero_copy_{_fixture}", test_binding_is_zero_copy, fixture=_fixture
    )
for _fixture in ("quadruped", "arm"):
    for _function in (test_config_round_trip, test_integrate_inverts_increment):
        add_function_test(TestSystemArticulation, f"{_function.__name__}_{_fixture}", _function, fixture=_fixture)
for _function in (
    test_features_ignore_place_and_heading,
    test_binding_gathers_uneven_rows,
    test_robot_frame_floating,
    test_robot_frame,
    test_mismatch_is_refused,
    test_selection_is_one_articulation_per_world,
    test_unsupported_articulations_are_refused,
    test_spec_derives_its_counts,
    test_printed_form,
    test_includes_actuators,
    test_binding_interface,
    test_bounds,
):
    add_function_test(TestSystemArticulation, _function.__name__, _function)


if __name__ == "__main__":
    unittest.main()
