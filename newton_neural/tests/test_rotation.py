# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quaternion and rotation-vector utilities (conventions, composition, small-angle, 180-deg)."""

from __future__ import annotations

import math
import unittest

import torch

from newton_neural._src.core import rotation as rot


def _random_quaternions(count: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return rot.normalize(torch.randn(count, 4, dtype=torch.float64, generator=generator))


def _random_vectors(count: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, dtype=torch.float64, generator=generator)


class TestRotation(unittest.TestCase):
    def test_rotate_is_right_handed(self) -> None:
        """Check that a quarter turn about z takes x to y."""
        q = rot.from_rotation_vector(torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64))
        rotated = rot.rotate(q, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(rotated, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=0)

    def test_multiply_composes_and_inverse_undoes(self) -> None:
        """Check that a product rotates like its factors in turn, and that the inverse rotation undoes a rotation."""
        a, b, v = _random_quaternions(50, 0), _random_quaternions(50, 1), _random_vectors(50)
        torch.testing.assert_close(rot.rotate(rot.multiply(a, b), v), rot.rotate(a, rot.rotate(b, v)))
        torch.testing.assert_close(rot.rotate_inverse(a, rot.rotate(a, v)), v)
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(50, 4)
        torch.testing.assert_close(rot.multiply(a, rot.conjugate(a)), identity, atol=1e-12, rtol=0)

    def test_rotation_vector_round_trip(self) -> None:
        """Convert quaternions to rotation vectors and back, for large, tiny and zero rotations."""
        q = rot.canonicalize(_random_quaternions(100))
        torch.testing.assert_close(rot.from_rotation_vector(rot.to_rotation_vector(q)), q, atol=1e-10, rtol=0)
        self.assertLessEqual(float(rot.to_rotation_vector(q).norm(dim=-1).max()), math.pi + 1e-9)
        small = _random_vectors(10) * 1e-8
        small[0] = 0.0
        torch.testing.assert_close(rot.to_rotation_vector(rot.from_rotation_vector(small)), small, atol=1e-14, rtol=0)

    def test_rotation_vector_ignores_quaternion_sign(self) -> None:
        """Check that a quaternion and its negative give the same rotation vector."""
        q = _random_quaternions(50)
        torch.testing.assert_close(rot.to_rotation_vector(-q), rot.to_rotation_vector(q))

    def test_rotation_vector_gradient_is_finite_at_zero(self) -> None:
        """Differentiate the rotation vector conversions at the identity rotation."""
        vector = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        rot.to_rotation_vector(rot.from_rotation_vector(vector)).sum().backward()
        torch.testing.assert_close(vector.grad, torch.ones(3, dtype=torch.float64))

    def test_twist_leaves_a_swing_about_a_perpendicular_axis(self) -> None:
        """Split rotations into swing and twist about a tilted axis and check each part's axis."""
        q = _random_quaternions(100)
        axis = torch.nn.functional.normalize(torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64), dim=0)
        twist = rot.twist(q, axis)
        # the twist turns about the axis
        self.assertLess(float(torch.linalg.cross(twist[:, :3], axis.expand(100, 3)).abs().max()), 1e-12)
        self.assertGreaterEqual(float(twist[:, 3].min()), 0.0)
        torch.testing.assert_close(twist.norm(dim=-1), torch.ones(100, dtype=torch.float64))
        # what remains turns about an axis perpendicular to it
        swing = rot.multiply(q, rot.conjugate(twist))
        self.assertLess(float((swing[:, :3] @ axis).abs().max()), 1e-12)

    def test_twist_follows_a_turn_about_the_axis(self) -> None:
        """Turn rotations about the axis and check that their twist turns with them."""
        q = _random_quaternions(100)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        turn = rot.from_rotation_vector(axis * torch.linspace(-3.0, 3.0, 100, dtype=torch.float64).unsqueeze(-1))
        expected = rot.canonicalize(rot.multiply(turn, rot.twist(q, axis)))
        torch.testing.assert_close(rot.twist(rot.multiply(turn, q), axis), expected, atol=1e-10, rtol=0)

    def test_twist_of_a_perpendicular_half_turn_is_the_identity(self) -> None:
        """Check that a half turn about a horizontal axis has no heading."""
        half_turn = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]])
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, 4)
        torch.testing.assert_close(rot.twist(half_turn, torch.tensor([0.0, 0.0, 1.0])), identity)


if __name__ == "__main__":
    unittest.main()
