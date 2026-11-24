# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ContactInputSet``: contact features, permutation/padding invariance, overflow, and CUDA graphs."""

from __future__ import annotations

import json
import math
import unittest

import newton
import numpy as np
import torch
import warp as wp
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.contacts.contact_input import ContactInputBase
from newton_neural._src.contacts.contact_set import ROW_WIDTH, ContactEncoderSet, ContactInputSet
from newton_neural._src.core import rotation as rot
from newton_neural._src.core.signal import SignalRole
from newton_neural._src.systems.articulation import SystemArticulation
from newton_neural._src.systems.articulation_binding import BindingArticulation
from newton_neural.tests.fixtures import add_arm, quadruped_model, random_frames
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

WORLD_COUNT = 3

SPHERE_RADIUS = 0.05
SPHERE_MARGIN = 0.002
PLANE_MARGIN = 0.001
BALL_RADIUS = 0.04
FOOT_DROP = 0.2
"""How far the foot link of the hopper hangs below its base [m]."""
SPHERE_SHIFT = 0.1
"""How far the hopper's sphere sits in front of the origin of its foot link [m]."""


class Scene:
    """A model with its robot, a state, a collision pipeline and the contact set's sensor."""

    def __init__(self, model: newton.Model, articulation: str, *, capacity: int = 64) -> None:
        self.model = model
        self.system = SystemArticulation.from_model(model, articulation)
        self.binding: BindingArticulation = self.system.bind(model, articulation)
        self.state = model.state()
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        self.contact_input = ContactInputSet.from_system(self.system, capacity=capacity)
        self.sensor = self.contact_input.sensor(model, self.binding)

    def collide(self, frames: Frames) -> None:
        """Write the robot's joint state, update the body poses and detect the contacts."""
        self.binding.write_state(self.state, frames)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.pipeline.collide(self.state, self.contacts)

    def sense(self) -> Frames:
        """The sensor's signals for the detected contacts."""
        return self.sensor.update(self.state, self.contacts)


def _quadruped_frames(scene: Scene, *, height: float | Tensor = 0.5, seed: int = 0) -> Frames:
    """A bent quadruped per world with its own heading, tilt and velocity, its base ``height`` [m] above the ground."""
    world_count = scene.binding.world_count
    device = wp.device_to_torch(scene.model.device)
    frames = random_frames(scene.system, world_count, device=device, seed=seed)
    turn = torch.tensor([0.1, -0.05, 0.0], device=device) + torch.arange(world_count, device=device)[:, None] * (
        torch.tensor([0.02, 0.03, 0.9], device=device)
    )
    frames["joint_q"][:, 2] = height
    frames["joint_q"][:, 3:7] = rot.from_rotation_vector(turn)
    return frames


def _arm_touching_box_model(world_count: int, device: torch.device) -> newton.Model:
    """The arm of the fixtures with a free box pressed against its links, each world mounted with its own turn."""
    scene = newton.ModelBuilder()
    for world in range(world_count):
        robot = newton.ModelBuilder()
        add_arm(robot)
        # At zero joint angles the arm is a column above its anchor, which this box overlaps by 2 cm.
        box = robot.add_body(xform=wp.transform((0.16, -0.2, 1.0), wp.quat_identity()), mass=0.2, label="box")
        robot.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
        mount = wp.transform((2.0 * world, 0.5, 0.0), wp.quat_rpy(0.0, 0.0, 0.6 * world + 0.3))
        scene.add_world(robot, xform=mount)
    scene.add_ground_plane()
    return scene.finalize(device=str(device))


def _hopper_model(world_count: int, device: torch.device) -> newton.Model:
    """A floating base with a foot link that carries one sphere, and a free ball, above the ground plane.

    The foot link hangs :data:`FOOT_DROP` below the base and its sphere sits :data:`SPHERE_SHIFT` in front of
    the link's origin. The ball's free joint follows the robot's joints in every world.
    """
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    robot = newton.ModelBuilder()
    base = robot.add_link(mass=1.0, inertia=inertia, label="base")
    foot = robot.add_link(mass=0.5, inertia=inertia, label="foot")
    robot.add_shape_sphere(
        foot,
        xform=wp.transform((SPHERE_SHIFT, 0.0, 0.0), wp.quat_identity()),
        radius=SPHERE_RADIUS,
        cfg=newton.ModelBuilder.ShapeConfig(margin=SPHERE_MARGIN),
    )
    joints = [
        robot.add_joint_free(base, label="floating_base"),
        robot.add_joint_fixed(
            base, foot, parent_xform=wp.transform((0.0, 0.0, -FOOT_DROP), wp.quat_identity()), label="ankle"
        ),
    ]
    robot.add_articulation(joints, label="hopper")
    ball = robot.add_body(mass=0.1, label="ball")
    robot.add_shape_sphere(ball, radius=BALL_RADIUS, cfg=newton.ModelBuilder.ShapeConfig(margin=0.0))
    scene = newton.ModelBuilder()
    scene.replicate(robot, world_count, spacing=(3.0, 0.0, 0.0))
    scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(margin=PLANE_MARGIN))
    return scene.finalize(device=str(device))


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the vector ``v`` ``[3]`` by the quaternion ``q`` ``[4]`` (xyzw)."""
    return v + 2.0 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def _rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the vector ``v`` ``[3]`` by the inverse of the quaternion ``q`` ``[4]`` (xyzw)."""
    return _rotate(q * np.array([-1.0, -1.0, -1.0, 1.0]), v)


def _host_rows(scene: Scene) -> list[np.ndarray]:
    """The rows of every world, computed on the host from the contact buffer in the buffer's order.

    Returns:
        Per world an array ``[row_count, 1 + 14]`` that holds the link and then the row.
    """
    model, system, contacts = scene.model, scene.system, scene.contacts
    body_q = scene.state.body_q.numpy().astype(np.float64)
    body_qd = scene.state.body_qd.numpy().astype(np.float64)
    body_com = model.body_com.numpy().astype(np.float64)
    body_flags = model.body_flags.numpy()
    shape_body = model.shape_body.numpy()
    body_ids = scene.binding.body_ids.numpy()
    place = {int(body): (world, link) for world, bodies in enumerate(body_ids) for link, body in enumerate(bodies)}

    frames = []  # origin [3] and quaternion [4] of the robot frame of every world
    for world, bodies in enumerate(body_ids):
        if system.floating_base:
            base = body_q[bodies[system.root_body]]
            heading = rot.twist(torch.from_numpy(base[3:]), system.up.double().cpu())
            frames.append((base[:3], heading.numpy()))
        else:
            anchor = model.joint_X_p.numpy()[scene.binding.root_joint_ids.numpy()[world]].astype(np.float64)
            frames.append((anchor[:3], anchor[3:]))

    count = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    shapes = np.stack((contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()), axis=1)
    points = np.stack((contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy()), axis=1)
    offsets = np.stack((contacts.rigid_contact_offset0.numpy(), contacts.rigid_contact_offset1.numpy()), axis=1)
    margins = np.stack((contacts.rigid_contact_margin0.numpy(), contacts.rigid_contact_margin1.numpy()), axis=1)
    normals = contacts.rigid_contact_normal.numpy().astype(np.float64)

    rows: list[list[np.ndarray]] = [[] for _ in body_ids]
    for contact in range(count):
        bodies = shape_body[shapes[contact]]
        support, surface, velocity, origin, dynamic = [], [], [], [], []
        for side, body in enumerate(bodies):
            point = points[contact, side].astype(np.float64)
            offset = offsets[contact, side].astype(np.float64)
            if body < 0:
                support.append(point)
                surface.append(point + offset)
                velocity.append(np.zeros(3))
                origin.append(np.zeros(3))
                dynamic.append(0.0)
                continue
            position, quaternion = body_q[body, :3], body_q[body, 3:]
            support.append(position + _rotate(quaternion, point))
            surface.append(support[-1] + _rotate(quaternion, offset))
            com = position + _rotate(quaternion, body_com[body])
            velocity.append(body_qd[body, :3] + np.cross(body_qd[body, 3:], surface[-1] - com))
            origin.append(position)
            dynamic.append(float(body_flags[body] & int(newton.BodyFlags.DYNAMIC) != 0))
        gap = normals[contact] @ (support[1] - support[0]) - margins[contact].sum()
        for side, body in enumerate(bodies):
            if int(body) not in place:
                continue
            world, link = place[int(body)]
            frame_origin, frame_quaternion = frames[world]
            sign = 1.0 if side == 0 else -1.0
            row = np.concatenate(
                (
                    [link],
                    _rotate_inverse(frame_quaternion, surface[side] - frame_origin),
                    _rotate_inverse(frame_quaternion, sign * normals[contact]),
                    _rotate_inverse(frame_quaternion, surface[side] - origin[side]),
                    [gap],
                    _rotate_inverse(frame_quaternion, velocity[side] - velocity[1 - side]),
                    [dynamic[1 - side]],
                )
            )
            rows[world].append(row)
    return [np.array(world_rows).reshape(-1, 1 + ROW_WIDTH) for world_rows in rows]


def _sensed_rows(test: unittest.TestCase, frames: Frames) -> list[np.ndarray]:
    """The valid rows of every world as ``[row_count, 1 + 14]`` with the link first, after checking the padding."""
    rows = frames["contact_rows"].cpu().numpy()
    links = frames["contact_links"].cpu().numpy()
    counts = frames["contact_count"].cpu().numpy()
    sensed = []
    for world, count in enumerate(counts):
        test.assertTrue((links[world, :count] >= 0).all())
        test.assertTrue((links[world, count:] == -1).all())
        test.assertTrue((rows[world, count:] == 0.0).all())
        sensed.append(np.concatenate((links[world, :count, None], rows[world, :count]), axis=1))
    return sensed


def _assert_subset(test: unittest.TestCase, found: np.ndarray, expected: np.ndarray, tolerance: float = 1e-4) -> None:
    """Check that every row of ``found`` equals a row of its own in ``expected``, whatever the order."""
    test.assertLessEqual(len(found), len(expected))
    remaining = list(range(len(expected)))
    for row in found:
        errors = np.abs(expected[remaining] - row).max(axis=1)
        best = int(np.argmin(errors))
        test.assertLess(errors[best], tolerance, f"no expected row matches {row}")
        remaining.pop(best)


def _assert_rows_match_host(test: unittest.TestCase, scene: Scene, frames: Frames) -> list[np.ndarray]:
    """Check that the sensor's rows of every world equal, as a set, the rows computed on the host. Returns them."""
    expected = _host_rows(scene)
    for found, wanted in zip(_sensed_rows(test, frames), expected, strict=True):
        test.assertEqual(len(found), len(wanted))
        _assert_subset(test, found, wanted)
    test.assertTrue((scene.sensor.overflow.numpy() == 0).all())
    return expected


def test_rows_match_host_quadruped(test: unittest.TestCase, device: torch.device) -> None:
    """Compare the rows of quadrupeds standing on the ground with rows computed in NumPy from the same contacts."""
    scene = Scene(quadruped_model(WORLD_COUNT, device), "quadruped")
    scene.collide(_quadruped_frames(scene))
    rows = np.concatenate(_assert_rows_match_host(test, scene, scene.sense()))
    test.assertTrue((rows[:, 1 + 13] == 0.0).any(), "no contact with the ground")
    test.assertTrue((rows[:, 1 + 13] == 1.0).any(), "no contact between two links")
    test.assertGreater(len(np.unique(rows[:, 0])), 4)


def test_rows_match_host_arm_and_box(test: unittest.TestCase, device: torch.device) -> None:
    """Compare the rows of fixed base arms that touch a moving box with rows computed in NumPy."""
    scene = Scene(_arm_touching_box_model(WORLD_COUNT, device), "arm")
    frames = random_frames(scene.system, WORLD_COUNT, device=device, seed=1)
    frames["joint_q"] *= 0.2
    # The box's free joint follows the arm's 8 degrees of freedom in every world.
    joint_qd = scene.state.joint_qd.numpy().reshape(WORLD_COUNT, -1)
    joint_qd[:, 8:] = np.linspace(-1.0, 1.0, WORLD_COUNT * 6).reshape(WORLD_COUNT, 6)
    scene.state.joint_qd.assign(joint_qd.reshape(-1))
    scene.collide(frames)
    _assert_rows_match_host(test, scene, scene.sense())
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    shapes = np.stack((scene.contacts.rigid_contact_shape0.numpy(), scene.contacts.rigid_contact_shape1.numpy()))
    touched = scene.model.shape_body.numpy()[shapes[:, :count]]
    boxes = [body for body, label in enumerate(scene.model.body_label) if label.endswith("box")]
    test.assertEqual(len(boxes), WORLD_COUNT)
    for box in boxes:
        test.assertIn(box, touched, "an arm does not touch its box")


def test_sphere_on_plane(test: unittest.TestCase, device: torch.device) -> None:
    """Check gap, normal, position, lever arm and velocity of a sphere over a plane and under a ball by hand."""
    scene = Scene(_hopper_model(2, device), "hopper")
    yaw = torch.tensor([0.0, 2.0], device=device)
    foot_height = torch.tensor([0.08, 0.04], device=device)  # the first sphere hovers, the second one penetrates
    velocity = torch.tensor([[0.3, -0.2, 0.0], [-0.1, 0.4, 0.0]], device=device)
    spin = torch.tensor([0.5, -1.5], device=device)
    frames = {
        "joint_q": torch.zeros(2, 7, device=device),
        "joint_qd": torch.zeros(2, 6, device=device),
    }
    world_x = scene.binding.read_state(scene.state)["joint_q"][:, 0].clone()
    frames["joint_q"][:, 0] = world_x
    frames["joint_q"][:, 2] = foot_height + FOOT_DROP
    frames["joint_q"][:, 3:7] = rot.from_rotation_vector(yaw[:, None] * torch.tensor([0.0, 0.0, 1.0], device=device))
    frames["joint_qd"][:, :3] = velocity
    frames["joint_qd"][:, 5] = spin
    # The ball rests 1 cm above the top of the sphere of the first world and far away in the second.
    ball_gap = 0.01
    forward = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), dim=-1)
    ball = frames["joint_q"][:, :3] + SPHERE_SHIFT * forward
    ball[:, 2] = foot_height + torch.tensor([SPHERE_RADIUS + BALL_RADIUS + ball_gap, 5.0], device=device)
    joint_q = scene.state.joint_q.numpy().reshape(2, -1)
    joint_q[:, 7:10] = ball.cpu().numpy()
    scene.state.joint_q.assign(joint_q.reshape(-1))
    scene.collide(frames)
    sensed = _sensed_rows(test, scene.sense())

    # The base has its centre of mass at its origin, so the contact point moves with the base's velocity and
    # with its spin about the vertical through the base. All of it is expressed in the heading frame.
    heading_velocity = torch.stack(
        (
            yaw.cos() * velocity[:, 0] + yaw.sin() * velocity[:, 1],
            -yaw.sin() * velocity[:, 0] + yaw.cos() * velocity[:, 1] + spin * SPHERE_SHIFT,
            torch.zeros_like(yaw),
        ),
        dim=-1,
    ).cpu()
    reach = SPHERE_RADIUS + SPHERE_MARGIN
    for world in range(2):
        ground = [
            1.0,  # the foot link
            *(SPHERE_SHIFT, 0.0, -FOOT_DROP - reach),
            *(0.0, 0.0, -1.0),
            *(SPHERE_SHIFT, 0.0, -reach),
            float(foot_height[world]) - SPHERE_RADIUS - SPHERE_MARGIN - PLANE_MARGIN,
            *heading_velocity[world].tolist(),
            0.0,
        ]
        expected = [ground]
        if world == 0:
            under_ball = [
                1.0,
                *(SPHERE_SHIFT, 0.0, -FOOT_DROP + reach),
                *(0.0, 0.0, 1.0),
                *(SPHERE_SHIFT, 0.0, reach),
                ball_gap - SPHERE_MARGIN,
                *heading_velocity[world].tolist(),
                1.0,
            ]
            expected.append(under_ball)
        test.assertEqual(len(sensed[world]), len(expected))
        _assert_subset(test, sensed[world], np.array(expected), tolerance=1e-5)
    # The robot's sphere is the second shape of the contact with the plane and the first of the one with the ball,
    # so both sides of a contact were turned into rows.
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    first_body = scene.model.shape_body.numpy()[scene.contacts.rigid_contact_shape0.numpy()[:count]]
    robot_first = np.isin(first_body, scene.binding.body_ids.numpy())
    test.assertTrue(robot_first.any() and not robot_first.all())


def test_worlds_do_not_mix(test: unittest.TestCase, device: torch.device) -> None:
    """Lift every second quadruped off the ground and expect ground contact in the other worlds only."""
    scene = Scene(quadruped_model(4, device), "quadruped")
    lifted = torch.arange(4, device=device) % 2 == 1
    scene.collide(_quadruped_frames(scene, height=torch.where(lifted, 5.0, 0.5)))
    rows = _assert_rows_match_host(test, scene, scene.sense())
    for world, world_rows in enumerate(rows):
        touches_ground = bool((world_rows[:, 1 + 13] == 0.0).any())
        test.assertEqual(touches_ground, not bool(lifted[world]))

    # Rows a world no longer has must turn back into padding.
    scene.collide(_quadruped_frames(scene, height=torch.where(lifted, 0.5, 5.0)))
    _assert_rows_match_host(test, scene, scene.sense())


def test_overflow(test: unittest.TestCase, device: torch.device) -> None:
    """Give the sensor fewer rows than a world needs and expect the surplus to be dropped and counted."""
    capacity = 5
    scene = Scene(quadruped_model(WORLD_COUNT, device), "quadruped", capacity=capacity)
    scene.collide(_quadruped_frames(scene))
    sensed = _sensed_rows(test, scene.sense())
    overflow = scene.sensor.overflow.numpy()
    for world, expected in enumerate(_host_rows(scene)):
        test.assertGreater(len(expected), capacity)
        test.assertEqual(len(sensed[world]), capacity)
        test.assertEqual(overflow[world], len(expected) - capacity)
        _assert_subset(test, sensed[world], expected)
        if device.type == "cpu":  # the kernel reads the buffer in order there
            np.testing.assert_allclose(sensed[world], expected[:capacity], atol=1e-4)


def test_entries_without_shape_give_no_row(test: unittest.TestCase, device: torch.device) -> None:
    """Take the first shape out of every detected contact, as a cleared buffer holds it, and expect no row."""
    scene = Scene(quadruped_model(WORLD_COUNT, device), "quadruped")
    scene.collide(_quadruped_frames(scene))
    test.assertGreater(scene.sense()["contact_count"].sum().item(), 0)
    scene.contacts.rigid_contact_shape0.fill_(-1)
    frames = scene.sense()
    test.assertEqual(frames["contact_count"].sum().item(), 0)
    test.assertTrue((frames["contact_links"] == -1).all().item())


def test_sensor_keeps_the_up_of_the_system(test: unittest.TestCase, device: torch.device) -> None:
    """Turn the gravity of the Newton model under a sensor and expect the rows to stay in the system's frame."""
    scene = Scene(quadruped_model(WORLD_COUNT, device), "quadruped")
    scene.collide(_quadruped_frames(scene))
    expected = _assert_rows_match_host(test, scene, scene.sense())
    gravity = scene.model.gravity.numpy()
    gravity[:] = (3.0, 0.0, -9.3)
    scene.model.gravity.assign(gravity)
    for found, wanted in zip(_sensed_rows(test, scene.sense()), expected, strict=True):
        test.assertEqual(len(found), len(wanted))
        _assert_subset(test, found, wanted)


def test_sensor_replays_from_graph(test: unittest.TestCase, device: torch.device) -> None:
    """Record the sensor into a CUDA graph and replay it on the contacts of new states."""
    scene = Scene(quadruped_model(WORLD_COUNT, device), "quadruped")
    scene.collide(_quadruped_frames(scene))
    scene.sense()  # kernels load outside the capture
    with wp.ScopedCapture(device=scene.model.device) as capture:
        frames = scene.sense()
    for seed, height in ((1, 0.45), (2, 5.0), (3, 0.55)):
        scene.collide(_quadruped_frames(scene, height=height, seed=seed))
        wp.capture_launch(capture.graph)
        _assert_rows_match_host(test, scene, frames)


class TestContactSetSensor(unittest.TestCase):
    """The sensor of the contact set, which builds rows on the device from the contacts a collision pipeline reports."""

    def test_missing_contacts(self) -> None:
        """Expect an error that names the remedy when the sensor is given no contacts."""
        scene = Scene(quadruped_model(1, "cpu"), "quadruped")
        with self.assertRaisesRegex(ValueError, r"Run collision detection.*newton\.CollisionPipeline\(model\)"):
            scene.sensor.update(scene.state, None)

    def test_reduced_contacts(self) -> None:
        """Expect an error for the contacts of a pipeline that reduces them per body pair."""
        scene = Scene(quadruped_model(1, "cpu"), "quadruped")
        reduction = newton.CollisionPipeline.BodyPairReductionConfig()
        pipeline = newton.CollisionPipeline(scene.model, body_pair_reduction=reduction)
        reduced = pipeline.contacts()
        self.assertTrue(reduced.rigid_contact_body_pair_reduced)
        scene.collide(_quadruped_frames(scene))
        pipeline.collide(scene.state, reduced)
        with self.assertRaisesRegex(ValueError, "body pair reduction.*without body_pair_reduction"):
            scene.sensor.update(scene.state, reduced)
        # The contacts of a pipeline that keeps every contact give rows.
        self.assertGreater(scene.sense()["contact_count"].item(), 0)

    def test_link_count_mismatch(self) -> None:
        """Expect an error when the contact set was built for a robot with another number of bodies."""
        scene = Scene(quadruped_model(1, "cpu"), "quadruped")
        with self.assertRaisesRegex(ValueError, "13 bodies"):
            ContactInputSet(4).sensor(scene.model, scene.binding)


for sensor_test in (
    test_rows_match_host_quadruped,
    test_rows_match_host_arm_and_box,
    test_sphere_on_plane,
    test_worlds_do_not_mix,
    test_overflow,
    test_entries_without_shape_give_no_row,
    test_sensor_keeps_the_up_of_the_system,
):
    add_function_test(TestContactSetSensor, sensor_test.__name__, sensor_test)
add_function_test(
    TestContactSetSensor,
    "test_sensor_replays_from_graph",
    test_sensor_replays_from_graph,
    devices=get_cuda_test_devices(),
)


CONTACT_INPUT = ContactInputSet(5, capacity=12, slot_count=4, slot_dim=16, link_embedding_dim=8)


def _encoder(device: torch.device) -> ContactEncoderSet:
    """An encoder of :data:`CONTACT_INPUT` with fitted statistics."""
    torch.manual_seed(0)
    encoder = CONTACT_INPUT.encoder().to(device)
    encoder.fit_inputs(_contact_frames((64,), device, seed=7))
    return encoder


def _contact_frames(lead: tuple[int, ...], device: torch.device, *, seed: int = 0) -> Frames:
    """Random contact sets of :data:`CONTACT_INPUT` with zero padding. The first is empty and the second full."""
    generator = torch.Generator().manual_seed(seed)
    capacity = CONTACT_INPUT.capacity
    count = torch.randint(0, capacity + 1, lead, generator=generator, dtype=torch.int32)
    count.view(-1)[0] = 0
    count.view(-1)[1] = capacity
    valid = torch.arange(capacity) < count.unsqueeze(-1)
    links = torch.randint(0, CONTACT_INPUT.link_count, (*lead, capacity), generator=generator, dtype=torch.int32)
    rows = 0.5 + 2.0 * torch.randn(*lead, capacity, ROW_WIDTH, generator=generator)
    frames = {
        "contact_rows": torch.where(valid.unsqueeze(-1), rows, 0.0),
        "contact_links": torch.where(valid, links, -1),
        "contact_count": count,
    }
    return {name: value.to(device) for name, value in frames.items()}


def _with_garbage(frames: Frames) -> Frames:
    """A copy of ``frames`` whose padding rows hold large and non-finite values."""
    rows = frames["contact_rows"].clone()
    garbage = torch.tensor([float("nan"), float("inf"), -1e30, 7.0], device=rows.device)
    rows[frames["contact_links"] < 0] = garbage.repeat(ROW_WIDTH)[:ROW_WIDTH]
    return {**frames, "contact_rows": rows}


def _pooled_by_hand(encoder: ContactEncoderSet, frames: Frames) -> Tensor:
    """The code of each contact set ``[B, code_count]``, pooled one set and one slot at a time from its valid rows."""
    codes = []
    for rows, links, count in zip(*(frames[name] for name in CONTACT_INPUT.signals), strict=True):
        rows, links = rows[: int(count)].double(), links[: int(count)].long()
        features = ((rows - encoder.normalizer.mean) / encoder.normalizer.std).clamp(-10.0, 10.0)
        encoded = encoder.row(torch.cat((features, encoder.link_embedding(links)), dim=-1))
        encoded = torch.nn.functional.silu(encoder.norm(encoded))
        slots = []
        for scoring in encoder.score.weight:
            weights = torch.softmax(encoded @ scoring, dim=0)
            slots.append((weights.unsqueeze(-1) * encoded).sum(dim=0))
        fill = math.log(1 + int(count)) / math.log(1 + CONTACT_INPUT.capacity)
        codes.append(torch.cat((*slots, torch.tensor([fill], dtype=torch.float64, device=rows.device))))
    return torch.stack(codes)


def test_code_matches_pooling_by_hand(test: unittest.TestCase, device: torch.device) -> None:
    """Compare the code with slots pooled one at a time from the valid rows alone."""
    encoder = _encoder(device).double()
    frames = _contact_frames((6,), device)
    frames["contact_rows"] = frames["contact_rows"].double()
    with torch.no_grad():
        code = encoder(frames)
        expected = _pooled_by_hand(encoder, frames)
    test.assertEqual(code.shape, (6, CONTACT_INPUT.code_count))
    test.assertLess((code - expected).abs().max().item(), 1e-12)


def test_code_ignores_row_order(test: unittest.TestCase, device: torch.device) -> None:
    """Permute the rows of every set, padding included, and expect the same code."""
    encoder = _encoder(device)
    frames = _contact_frames((8,), device)
    order = torch.randperm(CONTACT_INPUT.capacity, generator=torch.Generator().manual_seed(1)).to(device)
    permuted = {
        "contact_rows": frames["contact_rows"][:, order],
        "contact_links": frames["contact_links"][:, order],
        "contact_count": frames["contact_count"],
    }
    with torch.no_grad():
        code = encoder(frames)
        test.assertGreater(code.abs().max().item(), 0.1)
        test.assertLess((encoder(permuted) - code).abs().max().item(), 1e-5)


def test_code_ignores_padding(test: unittest.TestCase, device: torch.device) -> None:
    """Fill the padding rows with large and non-finite values and expect the code and its gradients unchanged."""
    encoder = _encoder(device)
    frames = _contact_frames((8,), device)
    gradients = []
    for variant in (frames, _with_garbage(frames)):
        encoder.zero_grad()
        code = encoder(variant)
        code.square().sum().backward()
        gradients.append([parameter.grad.clone() for parameter in encoder.parameters()])
        test.assertTrue(torch.equal(code, encoder(frames)))
    for clean, garbage in zip(*gradients, strict=True):
        test.assertTrue(torch.isfinite(clean).all())
        test.assertTrue(torch.equal(clean, garbage))


def test_empty_set_gives_zero_code(test: unittest.TestCase, device: torch.device) -> None:
    """Encode sets without a valid row and expect a code of exact zeros, with finite gradients."""
    encoder = _encoder(device)
    frames = _contact_frames((4,), device)
    frames["contact_links"].fill_(-1)
    frames["contact_count"].zero_()
    code = encoder(_with_garbage(frames))
    test.assertTrue((code == 0.0).all())
    code.sum().backward()
    for parameter in encoder.parameters():
        test.assertTrue(torch.isfinite(parameter.grad).all())


def test_window_matches_single_frames(test: unittest.TestCase, device: torch.device) -> None:
    """Encode a window of frames at once and expect the codes of its frames encoded one by one."""
    encoder = _encoder(device)
    frames = _contact_frames((3, 5), device)
    with torch.no_grad():
        code = encoder(frames)
        test.assertEqual(code.shape, (3, 5, CONTACT_INPUT.code_count))
        for frame in range(5):
            single = encoder({name: value[:, frame] for name, value in frames.items()})
            test.assertLess((code[:, frame] - single).abs().max().item(), 1e-6)


def test_autocast_keeps_full_precision(test: unittest.TestCase, device: torch.device) -> None:
    """Encode under autocast and expect a float32 code close to the full precision one."""
    encoder = _encoder(device)
    frames = _with_garbage(_contact_frames((8,), device))
    with torch.no_grad():
        expected = encoder(frames)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            code = encoder(frames)
    test.assertEqual(code.dtype, torch.float32)
    test.assertLess((code - expected).abs().max().item(), 3e-2)
    # The count is not rounded to the autocast type.
    test.assertTrue(torch.equal(code[:, -1], expected[:, -1]))
    # Nor are the pooled sums. They hold values that the autocast type cannot.
    slots = code[:, :-1]
    test.assertFalse(torch.equal(slots, slots.to(torch.bfloat16).float()))


def test_compiled_encoder_matches_eager(test: unittest.TestCase, device: torch.device) -> None:
    """Compile the encoder as one static graph and compare it with the eager encoder."""
    encoder = _encoder(device)
    frames = _with_garbage(_contact_frames((8,), device))
    test.addCleanup(torch.compiler.reset)
    compiled = torch.compile(encoder, fullgraph=True, dynamic=False)
    with torch.no_grad():
        test.assertLess((compiled(frames) - encoder(frames)).abs().max().item(), 1e-5)


def test_normalization_ignores_padding(test: unittest.TestCase, device: torch.device) -> None:
    """Fit the statistics on sets with garbage in the padding and expect the statistics of the valid rows."""
    encoder = CONTACT_INPUT.encoder().to(device)
    frames = _contact_frames((16, 3), device)
    encoder.fit_inputs(_with_garbage(frames))
    valid = frames["contact_rows"][frames["contact_links"] >= 0].double()
    test.assertEqual(encoder.normalizer.count.item(), len(valid))
    test.assertLess((encoder.normalizer.mean - valid.mean(dim=0)).abs().max().item(), 1e-5)
    test.assertLess((encoder.normalizer.std - valid.std(dim=0, correction=0)).abs().max().item(), 1e-5)


class TestContactSetEncoder(unittest.TestCase):
    """The encoder of the contact set, which pools valid rows softly into slots."""


for encoder_test in (
    test_code_matches_pooling_by_hand,
    test_code_ignores_row_order,
    test_code_ignores_padding,
    test_empty_set_gives_zero_code,
    test_window_matches_single_frames,
    test_autocast_keeps_full_precision,
    test_normalization_ignores_padding,
):
    add_function_test(TestContactSetEncoder, encoder_test.__name__, encoder_test)
add_function_test(
    TestContactSetEncoder,
    "test_compiled_encoder_matches_eager",
    test_compiled_encoder_matches_eager,
    devices=get_cuda_test_devices(),
)


class TestContactInputSet(unittest.TestCase):
    """The contact set as a part, with its signals, its sizes and rebuilding it from a config."""

    def test_signals_and_sizes(self) -> None:
        """Check the signals, the width of the code and the kind of input."""
        contact_input = ContactInputSet(9)
        self.assertEqual(
            {name: (spec.shape, spec.dtype, spec.role) for name, spec in contact_input.signals.items()},
            {
                "contact_rows": ((64, 14), "float32", SignalRole.CONTACT),
                "contact_links": ((64,), "int32", SignalRole.CONTACT),
                "contact_count": ((), "int32", SignalRole.CONTACT),
            },
        )
        self.assertEqual(contact_input.code_count, 16 * 64 + 1)
        # What kind of input it is stands on the class, where a reader of the class and of an instance find it.
        self.assertTrue(ContactInputSet.gated)
        self.assertTrue(ContactInputSet.reads_contacts)
        self.assertFalse(ContactInputSet.recomputable)
        self.assertNotIn("gated", vars(contact_input))
        with self.assertRaises(NotImplementedError):
            contact_input.evaluate(None, {})
        self.assertEqual(
            repr(contact_input),
            "ContactInputSet(link_count=9, capacity=64, slot_count=16, slot_dim=64, link_embedding_dim=16)",
        )

    def test_from_system_takes_one_link_per_body(self) -> None:
        """Build the contact set from a system and expect one link per body of the system."""
        system = SystemArticulation.from_model(quadruped_model(1, "cpu"), "quadruped")
        contact_input = ContactInputSet.from_system(system, capacity=32)
        self.assertEqual(contact_input.link_count, 13)
        self.assertEqual(contact_input.capacity, 32)

    def test_from_system_sizes(self) -> None:
        """Expect the sizes of ``from_system`` to be those of the constructor, with the same defaults."""
        system = SystemArticulation.from_model(quadruped_model(1, "cpu"), "quadruped")
        self.assertEqual(ContactInputSet.from_system(system).config(), ContactInputSet(13).config())
        given = {"capacity": 8, "slot_count": 2, "slot_dim": 4, "link_embedding_dim": 2}
        self.assertEqual(ContactInputSet.from_system(system, **given).config(), {"link_count": 13, **given})
        # A size is given by its name, and a name the constructor does not have is refused.
        with self.assertRaises(TypeError):
            ContactInputSet.from_system(system, 8)
        with self.assertRaises(TypeError):
            ContactInputSet.from_system(system, slot_width=8)

    def test_config_round_trip(self) -> None:
        """Rebuild the contact set by name from its config after a trip through JSON."""
        contact_input = ContactInputSet(5, capacity=8, slot_count=2, slot_dim=4, link_embedding_dim=2)
        config = json.loads(json.dumps(contact_input.config()))
        rebuilt = ContactInputBase.create("ContactInputSet", config)
        self.assertIsInstance(rebuilt, ContactInputSet)
        self.assertEqual(rebuilt.config(), contact_input.config())
        self.assertEqual(rebuilt.signals, contact_input.signals)
        self.assertEqual(rebuilt.code_count, 2 * 4 + 1)
        state = contact_input.encoder().state_dict()
        self.assertEqual(
            {key: value.shape for key, value in rebuilt.encoder().state_dict().items()},
            {key: value.shape for key, value in state.items()},
        )

    def test_invalid_sizes(self) -> None:
        """Expect an error for a size below one."""
        with self.assertRaises(ValueError):
            ContactInputSet(0)
        with self.assertRaises(ValueError):
            ContactInputSet(4, capacity=0)


if __name__ == "__main__":
    unittest.main()
