# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``Normalizer``: batch accumulation, masking, constant channels, clamping, and ``fitted`` state."""

from __future__ import annotations

import unittest

import torch

from newton_neural._src.core.normalizer import FEATURE_LIMIT, Normalizer
from newton_neural.tests.unittest_utils import add_function_test


def _samples(device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    scale = torch.tensor([1.0, 100.0, 0.01, 5.0])
    offset = torch.tensor([0.0, -50.0, 3.0, 1000.0])
    return (torch.randn(6, 50, 4, generator=generator) * scale + offset).to(device)


def test_batches_accumulate(test: unittest.TestCase, device: torch.device) -> None:
    """Update over several batches and compare with the statistics of all samples at once."""
    x = _samples(device)
    normalizer = Normalizer(4).to(device)
    for batch in x:
        normalizer.update(batch)
    flat = x.reshape(-1, 4).double()
    test.assertEqual(float(normalizer.count), 300.0)
    torch.testing.assert_close(normalizer.mean, flat.mean(0).float())
    torch.testing.assert_close(normalizer.std, flat.std(0, correction=0).float())


def test_mask_selects_samples(test: unittest.TestCase, device: torch.device) -> None:
    """Update with a validity mask and compare with the statistics of the valid samples alone."""
    x = _samples(device)
    mask = torch.rand(6, 50, generator=torch.Generator().manual_seed(1)).to(device) < 0.3
    x[~mask] = 1e6  # padding must not leak into any statistic
    normalizer = Normalizer(4).to(device)
    normalizer.update(x, mask)
    normalizer.update(x, torch.zeros_like(mask))  # a batch without valid samples changes nothing
    valid = x[mask].double()
    test.assertEqual(float(normalizer.count), float(mask.sum()))
    torch.testing.assert_close(normalizer.mean, valid.mean(0).float())
    torch.testing.assert_close(normalizer.std, valid.std(0, correction=0).float())


def test_constant_channel_keeps_unit_scale(test: unittest.TestCase, device: torch.device) -> None:
    """Fit a channel that never varies and check that normalizing it only removes its mean."""
    x = _samples(device)
    x[..., 2] = 7.0
    normalizer = Normalizer(4).to(device)
    normalizer.update(x)
    test.assertEqual(float(normalizer.std[2]), 1.0)
    torch.testing.assert_close(normalizer.normalize(x)[..., 2], torch.zeros(6, 50, device=device))


def test_normalize_round_trip(test: unittest.TestCase, device: torch.device) -> None:
    """Normalize to zero mean and unit deviation, denormalize back, and restore the statistics from a state dict."""
    x = _samples(device)
    normalizer = Normalizer(4).to(device)
    normalizer.update(x)
    y = normalizer.normalize(x).reshape(-1, 4)
    torch.testing.assert_close(y.mean(0), torch.zeros(4, device=device), atol=1e-4, rtol=0)
    torch.testing.assert_close(y.std(0, correction=0), torch.ones(4, device=device), atol=1e-4, rtol=0)
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(x)), x, rtol=1e-5, atol=1e-4)
    restored = Normalizer(4).to(device)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored.normalize(x), normalizer.normalize(x))
    # A state dict holds the statistics that normalizing reads and the sums they are updated from.
    test.assertEqual(set(normalizer.state_dict()), {"mean", "std", "count", "_mean64", "_squares64"})


def test_limit_clamps_the_normalized(test: unittest.TestCase, device: torch.device) -> None:
    """Normalize samples far from the fitted ones and expect the limit to bound them, and nothing without one."""
    x = _samples(device)
    normalizer = Normalizer(4).to(device)
    normalizer.update(x)
    far = 50.0 * x + 7.0
    unlimited = normalizer.normalize(far)
    test.assertGreater(unlimited.abs().max().item(), 100.0)
    test.assertEqual(FEATURE_LIMIT, 10.0)
    limited = normalizer.normalize(far, limit=FEATURE_LIMIT)
    test.assertTrue(torch.equal(limited, unlimited.clamp(-10.0, 10.0)))
    test.assertEqual(limited.abs().max().item(), 10.0)
    # Samples inside the limit pass unchanged.
    test.assertTrue(torch.equal(normalizer.normalize(x, limit=1.0e3), normalizer.normalize(x)))


def test_fitted_follows_the_samples(test: unittest.TestCase, device: torch.device) -> None:
    """Expect ``fitted`` once the statistics hold a sample, from an update or from a loaded state dict."""
    x = _samples(device)
    normalizer = Normalizer(4).to(device)
    test.assertFalse(normalizer.fitted)
    normalizer.update(x, torch.zeros(6, 50, dtype=torch.bool, device=device))  # a batch without valid samples
    test.assertFalse(normalizer.fitted)
    normalizer.update(x)
    test.assertTrue(normalizer.fitted)
    restored = Normalizer(4).to(device)
    restored.load_state_dict(normalizer.state_dict())
    test.assertTrue(restored.fitted)
    restored.load_state_dict(Normalizer(4).state_dict())
    test.assertFalse(restored.fitted)


class TestNormalizer(unittest.TestCase):
    pass


add_function_test(TestNormalizer, "test_batches_accumulate", test_batches_accumulate)
add_function_test(TestNormalizer, "test_mask_selects_samples", test_mask_selects_samples)
add_function_test(TestNormalizer, "test_constant_channel_keeps_unit_scale", test_constant_channel_keeps_unit_scale)
add_function_test(TestNormalizer, "test_normalize_round_trip", test_normalize_round_trip)
add_function_test(TestNormalizer, "test_limit_clamps_the_normalized", test_limit_clamps_the_normalized)
add_function_test(TestNormalizer, "test_fitted_follows_the_samples", test_fitted_follows_the_samples)


if __name__ == "__main__":
    unittest.main()
