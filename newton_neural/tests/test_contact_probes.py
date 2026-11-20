# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ContactInputProbes``: geometry, finite-diff velocity, FK parity, invariance, and CUDA graphs."""

from __future__ import annotations

import copy
import json
import unittest
from collections.abc import Callable
from typing import Any

import newton
import torch
import warp as wp
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.contacts.contact_input import ContactInputBase
from newton_neural._src.contacts.contact_probes import (
    FEATURE_COUNT,
    SIGNAL,
    ContactEncoderProbes,
    ContactInputProbes,
)
from newton_neural._src.core import rotation as rot
from newton_neural._src.core.signal import SignalRole
from newton_neural._src.systems.articulation_binding import BindingArticulation
from newton_neural.systems import SystemArticulation
from newton_neural.testing import check_contact_input_recompute
from newton_neural.tests import fixtures
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

FIXTURES: dict[str, tuple[Callable[..., newton.Model], str, dict[str, Any]]] = {
    "quadruped": (fixtures.quadruped_model, "quadruped", {"bodies": "*_SHANK"}),
    "arm": (fixtures.arm_model, "arm", {"bodies": ["*_finger", "link3"], "radius": 0.01}),
    "arm_with_uneven_boxes": (fixtures.arm_with_uneven_boxes_model, "arm", {"bodies": ["*_finger"], "radius": 0.01}),
}
"""Model function, articulation pattern and probe selection of every fixture."""

FOOT_RADIUS = 0.03
"""Radius of the quadruped's foot spheres [m]."""

STANDING_GAP = 0.55 - 0.01 - 0.25 - 0.25 - FOOT_RADIUS
"""Gap of the quadruped's feet in its default pose [m], from base height, hip offset, thigh, shank and foot radius."""


def _scene(
    fixture: str, world_count: int, device: torch.device
) -> tuple[newton.Model, SystemArticulation, ContactInputProbes]:
    """A fixture's model, its system on ``device`` and the probes :meth:`ContactInputProbes.from_model` finds."""
    build, articulation, selection = FIXTURES[fixture]
    model = build(world_count, device)
    system = SystemArticulation.from_model(model, articulation).to(device)
    return model, system, ContactInputProbes.from_model(model, system, articulation=articulation, **selection)


def _pose(
    fixture: str, model: newton.Model, system: SystemArticulation, frames: Frames
) -> tuple[BindingArticulation, newton.State]:
    """The binding of a fixture's robot in ``model`` and a state of ``model`` posed at ``frames``."""
    binding = system.bind(model, FIXTURES[fixture][1])
    state = model.state()
    binding.write_state(state, frames)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return binding, state


def _sense(
    fixture: str, model: newton.Model, system: SystemArticulation, contact_input: ContactInputProbes, frames: Frames
) -> Tensor:
    """The features a new live sensor gives on ``model`` posed at ``frames``, ``[world_count, probe_count, 8]``."""
    binding, state = _pose(fixture, model, system, frames)
    return contact_input.sensor(model, binding).update(state, None)[SIGNAL].clone()


def _assert_close(test: unittest.TestCase, actual: Tensor, expected: Tensor, tolerance: float) -> None:
    test.assertEqual(actual.shape, expected.shape)
    error = (actual.double() - expected.double()).abs().max().item()
    test.assertLessEqual(error, tolerance)


def test_gap_is_analytic(test: unittest.TestCase, device: torch.device) -> None:
    """Check that the gap of a standing and of a lifted robot is the height of its feet above the plane."""
    model, system, contact_input = _scene("quadruped", 2, device)
    frames = {name: value.clone() for name, value in system.bind(model).read_state(model.state()).items()}
    lift = 0.5
    frames["joint_q"][1, 2] += lift
    features = _sense("quadruped", model, system, contact_input, frames)
    expected = torch.tensor([[STANDING_GAP] * 4, [STANDING_GAP + lift] * 4], device=device)
    _assert_close(test, features[..., 0], expected, 1e-6)
    # only the standing world is within the touch distance of 0.02 m
    test.assertEqual(features[..., 7].tolist(), [[1.0] * 4, [0.0] * 4])

    # A tilted and shifted plane, against the probe centres the features themselves report. The default pose
    # has no heading, so a centre is the base position plus the position feature.
    normal, offset = torch.tensor([0.0, 0.6, 0.8], device=device), 0.1
    tilted = ContactInputProbes(contact_input.probes, ground={"normal": normal.tolist(), "offset": offset})
    centre = frames["joint_q"][:, None, :3] + features[..., 1:4]
    _assert_close(test, tilted.evaluate(system, frames)[SIGNAL][..., 0], centre @ normal - offset - FOOT_RADIUS, 1e-5)


def test_features_follow_body_poses(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Check the gap and the position of every probe against the pose of its body, at random joint states."""
    model, system, contact_input = _scene(fixture, 3, device)
    frames = fixtures.random_frames(system, 3, device=device, seed=7)
    binding, state = _pose(fixture, model, system, frames)
    features = contact_input.sensor(model, binding).update(state, None)[SIGNAL]

    columns = [system.spec.body_labels.index(probe["body"]) for probe in contact_input.probes]
    body_q = wp.to_torch(state.body_q)[wp.to_torch(binding.body_ids).long()[:, columns]]  # [3, probe_count, 7]
    points = torch.tensor([probe["point"] for probe in contact_input.probes], device=device)
    radius = torch.tensor([probe["radius"] for probe in contact_input.probes], device=device)
    centre = body_q[..., :3] + rot.rotate(body_q[..., 3:], points)
    # the ground of the fixtures is the plane z = 0
    _assert_close(test, features[..., 0], centre[..., 2] - radius, 1e-5)
    if system.floating_base:
        # the heading frame keeps the up axis, so the third position is the height above the base
        _assert_close(test, features[..., 3], centre[..., 2] - frames["joint_q"][:, None, 2], 1e-5)
    else:
        # the anchors of the fixture are not rotated, so positions are offsets from the anchor of each world
        anchor = wp.to_torch(model.joint_X_p)[wp.to_torch(binding.root_joint_ids).long()]
        _assert_close(test, features[..., 1:4], centre - anchor[:, None, :3], 1e-5)


def test_velocity_matches_finite_difference(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Check that the probe velocity is the rate at which the probe position changes as the joints move."""
    _, system, contact_input = _scene(fixture, 1, device)
    frames = fixtures.random_frames(system, 4, device=device, seed=1)
    joint_q, joint_qd = frames["joint_q"], frames["joint_qd"]
    if system.floating_base:
        joint_qd[:, 3:6] = 0.0  # a base that does not turn keeps its heading, so the robot frame only translates

    def position(step: float) -> Tensor:
        """Probe positions ``[4, probe_count, 3]`` after moving along the velocities for ``step`` seconds."""
        moved = joint_q.clone()
        if system.floating_base:
            moved[:, :3] += step * joint_qd[:, :3]
            moved[:, 7:] += step * joint_qd[:, 6:]
        else:
            moved += step * joint_qd
        return contact_input.evaluate(system, {"joint_q": moved, "joint_qd": joint_qd})[SIGNAL][..., 1:4]

    step = 1e-2
    expected = (position(step) - position(-step)) / (2.0 * step)
    if system.floating_base:
        # positions are relative to the base, velocities are relative to the world
        heading = rot.twist(joint_q[:, 3:7], system.up)
        expected += rot.rotate_inverse(heading, joint_qd[:, :3]).unsqueeze(1)
    _assert_close(test, contact_input.evaluate(system, frames)[SIGNAL][..., 4:7], expected, 2e-3)


def test_sensor_matches_evaluate(test: unittest.TestCase, device: torch.device, fixture: str) -> None:
    """Check that the live sensor and the evaluation from joint state give the same features."""
    model, system, contact_input = _scene(fixture, 3, device)
    frames = fixtures.random_frames(system, 3, device=device, seed=2)
    sensed = _sense(fixture, model, system, contact_input, frames)
    evaluated = contact_input.evaluate(system, frames)[SIGNAL]
    test.assertEqual(sensed.shape, (3, len(contact_input.probes), FEATURE_COUNT))
    test.assertEqual((evaluated.dtype, evaluated.device), (torch.float32, frames["joint_q"].device))
    _assert_close(test, evaluated, sensed, 1e-4)
    check_contact_input_recompute(contact_input, system, model, articulation=FIXTURES[fixture][1])


def test_sensor_keeps_the_up_of_the_system(test: unittest.TestCase, device: torch.device) -> None:
    """Turn the gravity of the Newton model under a sensor and expect the features to stay in the system's frame."""
    model, system, contact_input = _scene("quadruped", 2, device)
    frames = fixtures.random_frames(system, 2, device=device, seed=8)
    binding, state = _pose("quadruped", model, system, frames)
    sensor = contact_input.sensor(model, binding)
    sensed = sensor.update(state, None)[SIGNAL].clone()
    gravity = model.gravity.numpy()
    gravity[:] = (3.0, 0.0, -9.3)
    model.gravity.assign(gravity)
    # The system keeps the up direction it was described with, and so do its features and the probes.
    test.assertTrue(torch.equal(sensor.update(state, None)[SIGNAL], sensed))
    _assert_close(test, contact_input.evaluate(system, frames)[SIGNAL], sensed, 1e-4)


def test_features_ignore_place_and_heading(test: unittest.TestCase, device: torch.device) -> None:
    """Check that the features of a floating robot do not change when it moves along the ground or turns about up."""
    _, system, contact_input = _scene("quadruped", 1, device)
    frames = fixtures.random_frames(system, 4, device=device, seed=3)
    joint_q, joint_qd = frames["joint_q"], frames["joint_qd"]
    generator = torch.Generator().manual_seed(0)
    shift = 5.0 * torch.randn(4, 3, generator=generator).to(device)
    shift[:, 2] = 0.0
    turn = rot.from_rotation_vector(3.0 * torch.randn(4, 1, generator=generator).to(device) * system.up)
    moved = {
        "joint_q": torch.cat(
            (rot.rotate(turn, joint_q[:, :3]) + shift, rot.multiply(turn, joint_q[:, 3:7]), joint_q[:, 7:]), dim=-1
        ),
        "joint_qd": torch.cat(
            (rot.rotate(turn, joint_qd[:, :3]), rot.rotate(turn, joint_qd[:, 3:6]), joint_qd[:, 6:]), dim=-1
        ),
    }
    features = contact_input.evaluate(system, frames)[SIGNAL]
    _assert_close(test, contact_input.evaluate(system, moved)[SIGNAL], features, 1e-4)


def test_spaced_worlds_agree(test: unittest.TestCase, device: torch.device) -> None:
    """Check that worlds placed apart give the same features for the same joint state."""
    fixture = "arm_with_uneven_boxes"
    model, system, contact_input = _scene(fixture, 4, device)
    frames = {name: value.expand(4, -1) for name, value in fixtures.random_frames(system, 1, device=device).items()}
    features = _sense(fixture, model, system, contact_input, frames)
    _assert_close(test, features, features[:1].expand(4, -1, -1), 1e-5)
    _assert_close(test, features, contact_input.evaluate(system, frames)[SIGNAL], 1e-5)


def test_sensor_updates_its_views(test: unittest.TestCase, device: torch.device) -> None:
    """Check that the sensor fills the same memory on every update, so earlier views follow the state."""
    model, system, contact_input = _scene("quadruped", 2, device)
    binding = system.bind(model)
    sensor = contact_input.sensor(model, binding)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    view = sensor.update(state, None)[SIGNAL]
    test.assertEqual(wp.to_torch(sensor.features).data_ptr(), view.data_ptr())
    _assert_close(test, view[..., 0], torch.full((2, 4), STANDING_GAP, device=device), 1e-6)

    frames = fixtures.random_frames(system, 2, device=device, seed=4)
    binding.write_state(state, frames)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    test.assertEqual(sensor.update(state, None)[SIGNAL].data_ptr(), view.data_ptr())
    _assert_close(test, view, contact_input.evaluate(system, frames)[SIGNAL], 1e-4)


def test_sensor_replays_from_graph(test: unittest.TestCase, device: torch.device) -> None:
    """Check that an update recorded into a CUDA graph senses the current state on every replay."""
    model, system, contact_input = _scene("quadruped", 2, device)
    binding = system.bind(model)
    sensor = contact_input.sensor(model, binding)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    with wp.ScopedCapture(device=model.device) as capture:
        view = sensor.update(state, None)[SIGNAL]
    for seed in (5, 6):
        frames = fixtures.random_frames(system, 2, device=device, seed=seed)
        binding.write_state(state, frames)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        wp.capture_launch(capture.graph)
        _assert_close(test, view, contact_input.evaluate(system, frames)[SIGNAL], 1e-4)


def test_evaluate_windows(test: unittest.TestCase, device: torch.device) -> None:
    """Check that a window evaluates like its frames, as a copy without gradient in the type of the frames."""
    _, system, contact_input = _scene("quadruped", 1, device)
    frames = {name: value.double() for name, value in fixtures.random_frames(system, 2, 3, device=device).items()}
    frames["joint_q"].requires_grad_()
    window = contact_input.evaluate(system, frames)[SIGNAL]
    test.assertEqual(window.shape, (2, 3, 4, FEATURE_COUNT))
    test.assertEqual((window.dtype, window.device), (torch.float64, frames["joint_q"].device))
    test.assertFalse(window.requires_grad)
    kept = window.clone()
    for frame in range(3):
        single = contact_input.evaluate(system, {name: value[:, frame] for name, value in frames.items()})[SIGNAL]
        test.assertEqual(single.shape, (2, 4, FEATURE_COUNT))
        _assert_close(test, single, window[:, frame], 1e-6)
    # evaluating again poses the same replica, which must not reach results handed out before
    contact_input.evaluate(system, {name: value.flip(1) for name, value in frames.items()})
    test.assertTrue(torch.equal(window, kept))


def test_from_model(test: unittest.TestCase, device: torch.device) -> None:
    """Check that probes are found on sphere shapes and at body origins, and the ground in the anchor frame."""
    _, _, contact_input = _scene("quadruped", 2, device)
    for probe, leg in zip(contact_input.probes, fixtures.QUADRUPED_LEGS, strict=True):
        test.assertEqual(probe["body"], f"{leg}_SHANK")
        test.assertEqual(probe["point"], [0.0, 0.0, -0.25])
        test.assertAlmostEqual(probe["radius"], FOOT_RADIUS, places=6)
    test.assertEqual(contact_input.ground, {"normal": [0.0, 0.0, 1.0], "offset": 0.0})
    test.assertEqual(contact_input.touch_distance, 0.02)

    # the arm has no spheres, and its ground lies below the anchor of its root joint
    _, _, contact_input = _scene("arm", 3, device)
    test.assertEqual(
        contact_input.probes,
        [{"body": body, "point": [0.0, 0.0, 0.0], "radius": 0.01} for body in ("link3", "left_finger", "right_finger")],
    )
    test.assertEqual(contact_input.ground["normal"], [0.0, 0.0, 1.0])
    test.assertAlmostEqual(contact_input.ground["offset"], -fixtures.ARM_ANCHOR[2], places=6)

    # The touch distance of the constructor can be given.
    model, system, _ = _scene("quadruped", 1, device)
    chosen = ContactInputProbes.from_model(model, system, bodies="*_SHANK", touch_distance=0.05)
    test.assertEqual(chosen.touch_distance, 0.05)


def test_from_model_refusals(test: unittest.TestCase, device: torch.device) -> None:
    """Check that a model without a ground plane, an unmatched pattern and a missing radius are refused."""
    model, system, _ = _scene("quadruped", 1, device)
    with test.assertRaisesRegex(ValueError, r"no body matches \['\*_FOOT'\]"):
        ContactInputProbes.from_model(model, system, bodies=["*_SHANK", "*_FOOT"])
    with test.assertRaisesRegex(ValueError, "body 'base' has 0 collision spheres"):
        ContactInputProbes.from_model(model, system, bodies="base")

    robot = newton.ModelBuilder()
    fixtures.add_quadruped(robot)
    scene = newton.ModelBuilder()
    scene.replicate(robot, 1)
    with test.assertRaisesRegex(ValueError, "holds 0 static plane shapes"):
        ContactInputProbes.from_model(scene.finalize(device=str(device)), system, bodies="*_SHANK")


def test_config_round_trip(test: unittest.TestCase, device: torch.device) -> None:
    """Check that the registry rebuilds the contact input from its config, and that a copy senses alike."""
    _, system, found = _scene("quadruped", 1, device)
    contact_input = ContactInputProbes(found.probes, ground=found.ground, touch_distance=0.05)
    config = json.loads(json.dumps(contact_input.config()))
    test.assertEqual(set(config), {"probes", "ground", "touch_distance"})
    rebuilt = ContactInputBase.create("ContactInputProbes", config)
    test.assertIsInstance(rebuilt, ContactInputProbes)
    test.assertEqual(rebuilt.config(), contact_input.config())
    test.assertEqual(rebuilt.signals, contact_input.signals)

    frames = fixtures.random_frames(system, 2, device=device)
    features = contact_input.evaluate(system, frames)[SIGNAL]
    # the copy is taken after an evaluation, when the original holds a posed replica
    for other in (rebuilt, copy.deepcopy(contact_input)):
        test.assertTrue(torch.equal(other.evaluate(system, frames)[SIGNAL], features))


def test_contract(test: unittest.TestCase, device: torch.device) -> None:
    """Check the signal, the code width and the kind of input, and that invalid arguments are refused."""
    model, system, contact_input = _scene("quadruped", 1, device)
    spec = contact_input.signals[SIGNAL]
    test.assertEqual((spec.name, spec.shape, spec.role, spec.dtype), (SIGNAL, (4, 8), SignalRole.CONTACT, "float32"))
    test.assertEqual(contact_input.code_count, 32)
    # What kind of input it is stands on the class, where a reader of the class and of an instance find it.
    test.assertTrue(ContactInputProbes.recomputable)
    test.assertFalse(ContactInputProbes.reads_contacts)
    test.assertFalse(ContactInputProbes.gated)
    test.assertEqual(repr(contact_input), "ContactInputProbes(touch_distance=0.02)")

    ground = contact_input.ground
    with test.assertRaisesRegex(ValueError, "at least one probe"):
        ContactInputProbes([], ground=ground)
    with test.assertRaisesRegex(ValueError, "not a unit vector"):
        ContactInputProbes(contact_input.probes, ground={"normal": [0.0, 0.0, 2.0], "offset": 0.0})
    misnamed = {"body": "base", "position": [0.0, 0.0, 0.0], "radius": 0.01}
    with test.assertRaisesRegex(ValueError, r"probe 1 needs the keys \['body', 'point', 'radius'\].*lacks \['point'\]"):
        ContactInputProbes([contact_input.probes[0], misnamed], ground=ground)
    with test.assertRaisesRegex(ValueError, r"ground needs the keys \['normal', 'offset'\] and lacks \['offset'\]"):
        ContactInputProbes(contact_input.probes, ground={"normal": [0.0, 0.0, 1.0]})
    stray = ContactInputProbes([{"body": "tail", "point": [0.0, 0.0, 0.0], "radius": 0.01}], ground=ground)
    with test.assertRaisesRegex(ValueError, r"name the bodies \['tail'\]"):
        stray.sensor(model, system.bind(model))


def test_encoder(test: unittest.TestCase, device: torch.device) -> None:
    """Check that the encoder normalizes after fitting, clamps to ten, flattens windows and saves its statistics."""
    _, system, contact_input = _scene("quadruped", 1, device)
    frames = contact_input.evaluate(system, fixtures.random_frames(system, 64, 2, device=device))
    encoder = contact_input.encoder().to(device)
    raw = encoder(frames)
    test.assertEqual(raw.shape, (64, 2, contact_input.code_count))
    _assert_close(test, raw, frames[SIGNAL].flatten(-2).clamp(-10.0, 10.0), 0.0)

    encoder.fit_inputs(frames)
    code = encoder(frames).flatten(0, 1)
    varies = frames[SIGNAL].flatten(-2).flatten(0, 1).std(dim=0) > 1e-6
    test.assertGreater(int(varies.sum()), 24)
    _assert_close(test, code.mean(dim=0), torch.zeros_like(code[0]), 1e-4)
    _assert_close(test, code.std(dim=0, correction=0), varies.float(), 1e-4)

    far = encoder({SIGNAL: 1000.0 * frames[SIGNAL]})
    test.assertEqual((far.min().item(), far.max().item()), (-10.0, 10.0))

    restored = ContactEncoderProbes(contact_input).to(device)
    restored.load_state_dict(encoder.state_dict())
    test.assertTrue(torch.equal(restored(frames), encoder(frames)))


class TestContactProbes(unittest.TestCase):
    pass


for _fixture in ("quadruped", "arm"):
    for _function in (
        test_features_follow_body_poses,
        test_velocity_matches_finite_difference,
        test_sensor_matches_evaluate,
    ):
        add_function_test(TestContactProbes, f"{_function.__name__}_{_fixture}", _function, fixture=_fixture)
add_function_test(
    TestContactProbes,
    "test_sensor_matches_evaluate_arm_with_uneven_boxes",
    test_sensor_matches_evaluate,
    fixture="arm_with_uneven_boxes",
)
for _function in (
    test_gap_is_analytic,
    test_sensor_keeps_the_up_of_the_system,
    test_features_ignore_place_and_heading,
    test_spaced_worlds_agree,
    test_sensor_updates_its_views,
    test_evaluate_windows,
    test_from_model,
    test_from_model_refusals,
    test_config_round_trip,
    test_contract,
    test_encoder,
):
    add_function_test(TestContactProbes, _function.__name__, _function)
add_function_test(
    TestContactProbes,
    test_sensor_replays_from_graph.__name__,
    test_sensor_replays_from_graph,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
