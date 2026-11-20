# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``PredictionBase``, ``PredictionStateIncrement``, and ``PredictionFreeMotion``."""

from __future__ import annotations

import json
import unittest

import torch

from newton_neural._src.core.signal import SignalRole, SignalSpec
from newton_neural.predictions import PredictionBase, PredictionFreeMotion, PredictionStateIncrement
from newton_neural.systems import SystemArticulation, SystemBase
from newton_neural.tests import fixtures
from newton_neural.tests.unittest_utils import add_function_test


def test_state_increment(test: unittest.TestCase, device: torch.device) -> None:
    """Check that the state increment needs no recorded signal and labels frames with the system's increment."""
    prediction = PredictionStateIncrement()
    test.assertTrue(prediction.integrates)
    for model in (fixtures.quadruped_model(1, device), fixtures.arm_model(1, device)):
        system = SystemArticulation.from_model(model).to(device)
        test.assertEqual(prediction.signals(system), {})
        test.assertEqual(prediction.output_count(system), system.increment_count)
        frames = fixtures.random_frames(system, 3, 5, device=device)
        current = {name: value[:, :-1] for name, value in frames.items()}
        following = {name: value[:, 1:] for name, value in frames.items()}
        label = prediction.label(system, current, following)
        test.assertEqual(label.shape, (3, 4, system.increment_count))
        torch.testing.assert_close(label, system.increment(current, following))


def test_free_motion(test: unittest.TestCase, device: torch.device) -> None:
    """Check that free motion records the free joint acceleration and labels each frame with it."""
    prediction = PredictionFreeMotion()
    test.assertFalse(prediction.integrates)
    system = SystemArticulation.from_model(fixtures.arm_model(1, device)).to(device)
    signals = prediction.signals(system)
    test.assertEqual(list(signals), ["joint_qdd_free"])
    signal = signals["joint_qdd_free"]
    test.assertEqual((signal.name, signal.shape, signal.role), ("joint_qdd_free", (8,), SignalRole.LABEL))
    test.assertEqual((signal.dtype, signal.unit), ("float32", "rad/s^2 or m/s^2"))
    test.assertEqual(prediction.output_count(system), system.joint_dof_count)
    frames = fixtures.random_frames(system, 3, 5, device=device)
    frames["joint_qdd_free"] = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(1)).to(device)
    current = {name: value[:, :-1] for name, value in frames.items()}
    following = {name: value[:, 1:] for name, value in frames.items()}
    torch.testing.assert_close(prediction.label(system, current, following), frames["joint_qdd_free"][:, :-1])


def test_free_motion_refuses_a_floating_base(test: unittest.TestCase, device: torch.device) -> None:
    """Ask free motion for the signals and the output width of a floating base and get a ValueError."""
    system = SystemArticulation.from_model(fixtures.quadruped_model(1, device))
    prediction = PredictionFreeMotion()
    with test.assertRaisesRegex(ValueError, "fixed base articulations only"):
        prediction.signals(system)
    with test.assertRaisesRegex(ValueError, "fixed base articulations only"):
        prediction.output_count(system)


class TestPredictions(unittest.TestCase):
    def test_registry_rebuilds_predictions(self) -> None:
        """Rebuild each prediction from its class name and its config after a pass through JSON text."""
        for prediction in (PredictionStateIncrement(), PredictionFreeMotion()):
            config = json.loads(json.dumps(prediction.config()))
            rebuilt = PredictionBase.create(type(prediction).__name__, config)
            self.assertIs(type(rebuilt), type(prediction))
        unknown = "unknown PredictionBase 'PredictionContactForce'. The registered ones are .*PredictionFreeMotion"
        with self.assertRaisesRegex(ValueError, unknown):
            PredictionBase.create("PredictionContactForce", {})

    def test_incomplete_prediction_is_refused(self) -> None:
        """Build a prediction that lacks a method of the interface and expect an error that names the method."""

        class PredictionWithoutLabel(PredictionBase):
            integrates = False

            def signals(self, system: SystemBase) -> dict[str, SignalSpec]:
                return {}

            def output_count(self, system: SystemBase) -> int:
                return 1

        with self.assertRaisesRegex(TypeError, "abstract.*PredictionWithoutLabel.*label"):
            PredictionWithoutLabel()


add_function_test(TestPredictions, "test_state_increment", test_state_increment)
add_function_test(TestPredictions, "test_free_motion", test_free_motion)
add_function_test(TestPredictions, "test_free_motion_refuses_a_floating_base", test_free_motion_refuses_a_floating_base)


if __name__ == "__main__":
    unittest.main()
