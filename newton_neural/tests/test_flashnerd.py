# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``FlashNeRD``: contact injection, streaming parity, normalization, and autocast precision."""

from __future__ import annotations

import unittest
from typing import Any

import torch

from newton_neural.backbones import BackboneFlash
from newton_neural.contacts import ContactInputBase, ContactInputSet
from newton_neural.data import TrajectoryBatch
from newton_neural.models import ContactInjection, FlashNeRD
from newton_neural.testing import check_nerd_streaming
from newton_neural.tests.extensions import BackboneLeaky, ContactInputRootHeight
from newton_neural.tests.helpers import (
    FRAME_DT,
    WORLD_COUNT,
    assert_frames_close,
    frame_of,
    make_backbone,
    make_batch,
    make_frames,
    make_nerd,
    make_system,
    train_on_pendulums,
)
from newton_neural.tests.unittest_utils import add_function_test

ROBOTS = ("quadruped", "arm")


def test_signals_and_shapes(test: unittest.TestCase, device: torch.device, robot: str, contact: bool) -> None:
    """Build a model and expect the documented signals and shapes of inputs, labels, predictions and steps."""
    nerd = make_nerd(robot, device, contact_input=ContactInputRootHeight() if contact else None)
    system = nerd.system
    expected = list(system.signals) + (["root_height"] if contact else [])
    test.assertEqual(list(nerd.signals), expected)
    test.assertEqual(nerd.receptive_field, 3)
    test.assertEqual(nerd.device.type, device.type)

    window = make_frames(nerd, WORLD_COUNT, 5, device=device)
    current, following = frame_of(window, 0), frame_of(window, 1)
    for frames, lead in ((current, (WORLD_COUNT,)), (window, (WORLD_COUNT, 5))):
        features, code = nerd.inputs(frames)
        test.assertEqual(features.shape, (*lead, system.feature_count))
        test.assertEqual(features.dtype, torch.float32)
        if contact:
            test.assertEqual(code.shape, (*lead, nerd.contact_input.code_count))
        else:
            test.assertIsNone(code)
    test.assertEqual(nerd.labels(current, following).shape, (WORLD_COUNT, system.increment_count))
    test.assertEqual(nerd.labels(window, window).shape, (WORLD_COUNT, 5, system.increment_count))

    history = nerd.history(WORLD_COUNT)
    test.assertTrue(all(value.shape[0] == WORLD_COUNT and value.device == nerd.device for value in history.values()))
    test.assertEqual(nerd.predict(current, history).shape, (WORLD_COUNT, system.increment_count))
    following = nerd.step(current, history)
    test.assertEqual(list(following), ["joint_q", "joint_qd"])
    for name, value in following.items():
        test.assertEqual(value.shape, (WORLD_COUNT, *system.signals[name].shape))
    test.assertEqual(history["step"].tolist(), [2] * WORLD_COUNT)


def test_features_are_clamped(test: unittest.TestCase, device: torch.device) -> None:
    """Feed a state far outside the fitted data and expect normalized features within plus or minus 10."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    frames["joint_qd"] = frames["joint_qd"] * 1e6
    features, _ = nerd.inputs(frames)
    test.assertEqual(features.abs().max().item(), 10.0)


def test_steps_match_window_forward(test: unittest.TestCase, device: torch.device, robot: str, contact: bool) -> None:
    """Step a model frame by frame, across resets, and expect the predictions of the windowed forward the loss
    uses, to the rounding of float32."""
    nerd = make_nerd(robot, device, contact_input=ContactInputRootHeight() if contact else None)
    window = make_frames(nerd, WORLD_COUNT, 2 * nerd.receptive_field + 1, device=device)
    check_nerd_streaming(nerd, window, tolerance=1e-6)


def test_reset_history_forgets(test: unittest.TestCase, device: torch.device) -> None:
    """Reset some worlds and expect them to predict as from a fresh history while the others carry on."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    window = make_frames(nerd, WORLD_COUNT, 4, device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    with torch.no_grad():
        history, fresh = nerd.history(WORLD_COUNT), nerd.history(WORLD_COUNT)
        for index in range(3):
            nerd.predict(frame_of(window, index), history)
        carried = {name: value.clone() for name, value in history.items()}
        nerd.reset_history(history, world_mask=mask)
        reset = nerd.predict(frame_of(window, 3), history)
        test.assertTrue(torch.equal(reset[mask], nerd.predict(frame_of(window, 3), fresh)[mask]))
        test.assertTrue(torch.equal(reset[~mask], nerd.predict(frame_of(window, 3), carried)[~mask]))


def test_fit_normalization(test: unittest.TestCase, device: torch.device) -> None:
    """Fit a model on batches and expect unit statistics, the frame duration of the data and the batch limit kept."""
    nerd = FlashNeRD(make_system("quadruped"), backbone=make_backbone(), contact_input=ContactInputRootHeight())
    nerd.to(device)
    test.assertIsNone(nerd.frame_dt)
    batches = [make_batch(nerd, 64, 5, device=device, seed=seed) for seed in range(3)]
    nerd.fit_normalization([])  # no batch, so nothing is fitted
    test.assertFalse(nerd.fitted)
    nerd.fit_normalization(batches, batch_limit=2)
    test.assertTrue(nerd.fitted)
    test.assertEqual(nerd.frame_dt, FRAME_DT)
    test.assertEqual(nerd.feature_normalizer.count.item(), 2 * 64 * 4)
    test.assertEqual(nerd.contact_encoder.normalizer.count.item(), 2 * 64 * 4)

    frames = {name: torch.cat([batch.frames[name] for batch in batches[:2]]) for name in batches[0].frames}
    current, following = frame_of(frames, slice(0, -1)), frame_of(frames, slice(1, None))
    for normalized in (nerd.inputs(current)[0], nerd.labels(current, following)):
        flat = normalized.flatten(0, 1)
        test.assertLess(flat.mean(dim=0).abs().max().item(), 1e-3)
        test.assertLess((flat.std(dim=0) - 1.0).abs().max().item(), 1e-2)

    slower = TrajectoryBatch(batches[0].frames, batches[0].episode_start, 2 * FRAME_DT)
    with test.assertRaisesRegex(ValueError, "frames of"):
        nerd.fit_normalization([slower])


def test_training_lowers_error(test: unittest.TestCase, device: torch.device) -> None:
    """Train a small model on pendulum dynamics and expect a tenth of the one step error on unseen windows."""
    before, after = train_on_pendulums(device)
    test.assertLess(after, 0.1 * before)


def test_autocast_keeps_state_math(test: unittest.TestCase, device: torch.device, robot: str) -> None:
    """Run a model under autocast and expect float32 features, labels and states, with exact features and labels."""
    nerd = make_nerd(robot, device, contact_input=ContactInputRootHeight() if robot == "quadruped" else None)
    window = make_frames(nerd, WORLD_COUNT, 2, device=device)
    current, following = frame_of(window, 0), frame_of(window, 1)
    with torch.no_grad():
        features, labels = nerd.inputs(current)[0], nerd.labels(current, following)
        stepped = nerd.step(current, nerd.history(WORLD_COUNT))
        with torch.autocast(device.type, dtype=torch.bfloat16):
            test.assertTrue(torch.equal(nerd.inputs(current)[0], features))
            test.assertTrue(torch.equal(nerd.labels(current, following), labels))
            reduced = nerd.step(current, nerd.history(WORLD_COUNT))
    test.assertTrue(all(value.dtype == torch.float32 for value in reduced.values()))
    assert_frames_close(test, reduced, stepped, tolerance=0.1)


class TestFlashNeRD(unittest.TestCase):
    def test_default_parts(self) -> None:
        """Build a model with defaults and expect the default backbone, a state increment and TensorFloat-32."""
        nerd = FlashNeRD(make_system("arm"))
        self.assertEqual(nerd.network.backbone.config(), BackboneFlash().config())
        self.assertTrue(nerd.prediction.integrates)
        self.assertIsNone(nerd.contact_input)
        self.assertIsNone(nerd.contact_encoder)
        self.assertEqual(nerd.contact_injections, ())
        self.assertEqual(nerd.precision.value, "tf32")
        self.assertEqual(nerd.metadata, {})
        self.assertEqual(nerd.network.readout_hidden.out_features, 256)

    def test_contact_injections_default(self) -> None:
        """Build a model with a contact input and expect all three places, in a member that cannot be set."""
        nerd = FlashNeRD(make_system("quadruped"), backbone=make_backbone(), contact_input=ContactInputRootHeight())
        self.assertEqual(nerd.contact_injections, tuple(ContactInjection))
        self.assertEqual(nerd.contact_injections, ("embedding", "readout", "increment"))
        with self.assertRaises(AttributeError):
            nerd.contact_injections = ("readout",)

    def test_contact_injections_subset(self) -> None:
        """Choose places and expect a map for each of them alone, and later frames reached from the embedding only.

        The code of a frame acts on the prediction of that frame at every place. Only through the embedding does
        it enter the history that the predictions of the frames after it read.
        """
        cases = ((["embedding"], True), (["readout"], False), (["increment"], False), (["increment", "readout"], False))
        for places, reaches_later in cases:
            with self.subTest(places=places):
                nerd = make_nerd("quadruped", "cpu", contact_input=ContactInputRootHeight(), contact_injections=places)
                self.assertEqual(nerd.contact_injections, tuple(ContactInjection(place) for place in places))
                widths = {"embedding": nerd.backbone.width, "readout": 16, "increment": nerd.system.increment_count}
                injection = nerd.network.injection
                self.assertEqual(injection.widths, tuple(widths[place] for place in places))
                self.assertEqual(injection.project.weight.shape, (sum(injection.widths), 4))

                window = make_frames(nerd, WORLD_COUNT, 3)
                lifted = {**window, "root_height": window["root_height"].clone()}
                lifted["root_height"][:, 0] += 1.0
                with torch.no_grad():
                    change = (nerd(lifted) - nerd(window)).abs().amax(dim=(0, 2))
                self.assertGreater(change[0].item(), 0.0)
                self.assertEqual(bool(change[1:].any()), reaches_later)

    def test_gate_follows_the_contact_input(self) -> None:
        """Expect a gate at each chosen place for a contact set, which asks for one, and none for an input that
        does not ask."""
        system, places = make_system("quadruped"), ["readout", "embedding"]

        def gate(contact_input: ContactInputBase) -> torch.nn.Parameter | None:
            nerd = FlashNeRD(system, backbone=make_backbone(), contact_input=contact_input, contact_injections=places)
            return nerd.network.injection.gate

        contact_set = ContactInputSet(len(system.spec.body_labels), capacity=8, slot_count=2, slot_dim=4)
        self.assertEqual(gate(contact_set).shape, (2,))
        self.assertIsNone(gate(ContactInputRootHeight()))

    def test_contact_injections_refusals(self) -> None:
        """Expect places without a contact input, one string, no place, an unknown place and a repeated place to be
        refused."""
        system, backbone, contact_input = make_system("quadruped"), make_backbone(), ContactInputRootHeight()
        with self.assertRaisesRegex(ValueError, r"\['readout'\].*no contact input\. Pass contact_input as well"):
            FlashNeRD(system, backbone=backbone, contact_injections=[ContactInjection.READOUT])
        refusals = {
            "readout": r"the string 'readout'.*a sequence of places\. Pass a list or a tuple",
            (): "is empty.*Pass at least one of",
            ("readout", "everywhere"): r"names \['everywhere'\].*at \['embedding', 'readout', 'increment'\]\. Pass a",
            ("readout", ContactInjection.READOUT): r"a place twice, \['readout', 'readout'\]\. Pass each place once",
        }
        for places, message in refusals.items():
            with self.subTest(places=places), self.assertRaisesRegex(ValueError, message):
                FlashNeRD(system, backbone=backbone, contact_input=contact_input, contact_injections=places)

    def test_part_configs_are_checked_at_composition(self) -> None:
        """Compose a model from a part that a checkpoint could not store and expect a TypeError right away."""

        class BackboneOfTensors(BackboneLeaky):
            def config(self) -> dict[str, Any]:
                return {"width": torch.tensor(self.width), "leak": self.leak}

        class ContactInputOfTensors(ContactInputRootHeight):
            def config(self) -> dict[str, Any]:
                return {"code_count": torch.tensor(self.code_count)}

        with self.assertRaisesRegex(TypeError, "config of BackboneOfTensors must be plain data.*Tensor"):
            FlashNeRD(make_system("arm"), backbone=BackboneOfTensors())
        with self.assertRaisesRegex(TypeError, "config of ContactInputOfTensors must be plain data.*Tensor"):
            FlashNeRD(make_system("quadruped"), backbone=make_backbone(), contact_input=ContactInputOfTensors())


for robot_name in ROBOTS:
    for with_contact in (False, True) if robot_name == "quadruped" else (False,):
        suffix = f"{robot_name}{'_contact' if with_contact else ''}"
        options = {"robot": robot_name, "contact": with_contact}
        add_function_test(TestFlashNeRD, f"test_signals_and_shapes_{suffix}", test_signals_and_shapes, **options)
        add_function_test(
            TestFlashNeRD, f"test_steps_match_window_forward_{suffix}", test_steps_match_window_forward, **options
        )
    add_function_test(
        TestFlashNeRD, f"test_autocast_keeps_state_math_{robot_name}", test_autocast_keeps_state_math, robot=robot_name
    )
for function in (
    test_features_are_clamped,
    test_reset_history_forgets,
    test_fit_normalization,
    test_training_lowers_error,
):
    add_function_test(TestFlashNeRD, function.__name__, function)


if __name__ == "__main__":
    unittest.main()
