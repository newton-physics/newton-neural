# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying ``newton_neural.testing`` conformance checks reject faulty extensions and avoid side effects."""

from __future__ import annotations

import copy
import subprocess
import sys
import unittest

import torch
from torch import Tensor

from newton_neural import Frames
from newton_neural.models import FlashNeRD
from newton_neural.testing import check_backbone_streaming, check_nerd_streaming
from newton_neural.tests.extensions import BackboneLeaky, MinimalNeRD
from newton_neural.tests.helpers import BACKBONES, WORLD_COUNT, make_frames, make_system


class BackboneLeakyStale(BackboneLeaky):
    """A faulty backbone that keeps its history through a reset."""

    def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
        pass


class TestConformance(unittest.TestCase):
    """The checks an extension runs in its own tests, held to catch a fault and to change nothing."""

    def test_conformance_check_rejects_a_faulty_backbone(self) -> None:
        """Expect the conformance check to name the reset when a backbone keeps its history through one."""
        with self.assertRaisesRegex(AssertionError, "after its reset"):
            check_backbone_streaming(BackboneLeakyStale())

    def test_streaming_check_rejects_faulty_models(self) -> None:
        """Expect the streaming check to name a prediction that differs from forward and a reset that forgets
        nothing."""

        class NeRDShifted(MinimalNeRD):
            def predict(self, frames: Frames, history: dict[str, Tensor]) -> Tensor:
                return super().predict(frames, history) + self.output_normalizer.std

        class NeRDStale(FlashNeRD):
            def reset_history(self, history: dict[str, Tensor], world_mask: Tensor) -> None:
                pass

        system = make_system("arm")
        shifted = NeRDShifted(system)
        with self.assertRaisesRegex(AssertionError, "an uninterrupted sequence differs from forward"):
            check_nerd_streaming(shifted, make_frames(shifted, WORLD_COUNT, 9))
        stale = NeRDStale(system, backbone=BackboneLeaky())
        with self.assertRaisesRegex(AssertionError, "the worlds reset at frame"):
            check_nerd_streaming(stale, make_frames(stale, WORLD_COUNT, 9))

    def test_conformance_checks_are_part_of_the_library(self) -> None:
        """Import the package in a new interpreter and expect the checks without the package's own tests."""
        script = "import sys, newton_neural.testing; print('newton_neural.tests' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.split(), ["False"])

    def test_conformance_check_leaves_the_backbone_unchanged(self) -> None:
        """Run the conformance check and expect the parameters of the checked backbone to be untouched."""
        backbone = BACKBONES["flash"]()
        before = copy.deepcopy(backbone.state_dict())
        check_backbone_streaming(backbone)
        for key, value in backbone.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
