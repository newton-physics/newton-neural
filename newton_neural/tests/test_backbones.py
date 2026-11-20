# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for temporal backbones: streaming parity, span-1 fast path, config round-trip, and validation."""

from __future__ import annotations

import json
import unittest

import torch
from torch import Tensor, nn

from newton_neural.backbones import BackboneBase, BackboneFlash, BackboneMLP, BackboneTransformer
from newton_neural.testing import check_backbone_streaming
from newton_neural.tests.extensions import BackboneLeaky
from newton_neural.tests.helpers import BACKBONES, WORLD_COUNT, assert_close, process_precision
from newton_neural.tests.unittest_utils import add_function_test

ATTENTION_BACKBONES = {
    "flash": lambda span: BackboneFlash(head_count=4, head_dim=8, ffn_dim=16, spans=(span, span)),
    "transformer": lambda span: BackboneTransformer(width=16, head_count=4, depth=2, span=span, ffn_dim=32),
}
"""The two backbones that attend, with every block attending over ``span`` frames."""


def test_conformance(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Run the public conformance check on the backbone in float64 and in float32."""
    check_backbone_streaming(BACKBONES[backbone](), device=device)
    check_backbone_streaming(BACKBONES[backbone](), device=device, dtype=torch.float32)


def test_span_of_one_matches_general_attention(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Give a backbone of span 1 the weights of one of span 2 and expect, bit for bit, what the general attention
    of the wider one computes for a frame that has no past, over a window and frame by frame."""
    torch.manual_seed(0)
    narrow, wide = (ATTENTION_BACKBONES[backbone](span).to(device).eval() for span in (1, 2))
    for parameter in wide.parameters():
        nn.init.normal_(parameter, std=0.3)
    # Every weight but the lag bias, which has one column per frame of the span and no effect on a single frame.
    shared = {key: value for key, value in wide.state_dict().items() if not key.endswith("lag_bias")}
    test.assertEqual(narrow.load_state_dict(shared, strict=False).unexpected_keys, [])
    h = torch.randn(WORLD_COUNT, 5, narrow.width, device=device)
    histories = [stack.history(WORLD_COUNT, device=device, dtype=torch.float32) for stack in (narrow, wide)]
    everywhere = torch.ones(WORLD_COUNT, dtype=torch.bool, device=device)
    # In IEEE arithmetic, since a product in TensorFloat-32 would round the values the general attention sums.
    with torch.no_grad(), process_precision("highest"):
        for window in h.split(1, dim=1):
            # A window of one frame has no past, and neither has a world that was just reset.
            test.assertTrue(torch.equal(narrow(window), wide(window)))
            wide.reset_history(histories[1], everywhere)
            test.assertTrue(torch.equal(narrow.step(window[:, 0], histories[0]), wide.step(window[:, 0], histories[1])))
            for index in range(len(narrow.blocks)):  # the ring is written as the general step writes it
                rings = [history[f"ring{index}"][:, 0] for history in histories]
                test.assertTrue(torch.equal(*rings))
                test.assertTrue(rings[0].any())
        # The frames of a window do not see each other, and a step is the forward of its frame.
        assert_close(narrow(h), torch.cat([narrow(window) for window in h.split(1, dim=1)], dim=1), 1e-6)
    test.assertEqual(histories[0]["step"].tolist(), [5] * WORLD_COUNT)
    check_backbone_streaming(ATTENTION_BACKBONES[backbone](1), device=device)
    check_backbone_streaming(ATTENTION_BACKBONES[backbone](1), device=device, dtype=torch.float32)


class TestBackbones(unittest.TestCase):
    """The streaming contract of every backbone, on every device, and the backbones as parts that are rebuilt
    from a config."""

    def test_config_round_trip(self) -> None:
        """Rebuild every backbone through the registry from its config after a trip through JSON."""
        backbones = [*(make() for make in BACKBONES.values()), BackboneFlash(), BackboneTransformer(), BackboneMLP()]
        for backbone in [*backbones, BackboneLeaky(width=4, leak=0.25)]:
            with self.subTest(backbone=type(backbone).__name__):
                config = json.loads(json.dumps(backbone.config()))
                rebuilt = BackboneBase.create(type(backbone).__name__, config)
                self.assertIs(type(rebuilt), type(backbone))
                self.assertEqual(rebuilt.config(), backbone.config())
                self.assertEqual((rebuilt.width, rebuilt.receptive_field), (backbone.width, backbone.receptive_field))
                rebuilt.load_state_dict(backbone.state_dict())
        default = {"head_count": 8, "head_dim": 64, "ffn_dim": 1024, "spans": [4, 4, 4, 4]}
        self.assertEqual(BackboneFlash().config(), default)
        with self.assertRaises(ValueError):
            BackboneBase.create("BackboneMissing", {})

    def test_receptive_field_and_width(self) -> None:
        """Expect the frames that the blocks of each backbone reach back over, and the default width."""
        self.assertEqual(BackboneFlash().receptive_field, 13)
        self.assertEqual(BackboneFlash(spans=(3, 1, 4)).receptive_field, 6)
        self.assertEqual(BackboneTransformer().receptive_field, 29)
        self.assertEqual(BackboneTransformer(depth=3, span=5).receptive_field, 13)
        self.assertEqual(BackboneMLP().receptive_field, 4)
        self.assertEqual((BackboneFlash().width, BackboneTransformer().width, BackboneMLP().width), (512, 512, 512))

    def test_invalid_sizes(self) -> None:
        """Expect an error for head counts and sizes that do not fit together."""
        with self.assertRaises(ValueError):
            BackboneTransformer(width=16, head_count=3)
        with self.assertRaises(ValueError):
            BackboneFlash(spans=(4, 0))
        with self.assertRaises(ValueError):
            BackboneMLP(window=0)

    def test_hyperparameters_are_keyword_only(self) -> None:
        """Give each backbone a positional hyperparameter and expect a TypeError."""
        for backbone in (BackboneFlash, BackboneTransformer, BackboneMLP):
            with self.assertRaises(TypeError):
                backbone(16)

    def test_incomplete_backbone_is_refused(self) -> None:
        """Build a backbone that lacks a method of the interface and expect an error that names the method."""

        class BackboneWithoutStep(BackboneBase):
            receptive_field = 1

            def forward(self, h: Tensor) -> Tensor:
                return h

            def history(self, world_count: int, *, device: torch.device | str, dtype: torch.dtype) -> dict[str, Tensor]:
                return {}

        with self.assertRaisesRegex(TypeError, "abstract.*BackboneWithoutStep.*step"):
            BackboneWithoutStep()


for backbone_name in BACKBONES:
    add_function_test(TestBackbones, f"test_conformance_{backbone_name}", test_conformance, backbone=backbone_name)
for backbone_name in ATTENTION_BACKBONES:
    add_function_test(
        TestBackbones,
        f"test_span_of_one_matches_general_attention_{backbone_name}",
        test_span_of_one_matches_general_attention,
        backbone=backbone_name,
    )


if __name__ == "__main__":
    unittest.main()
