# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for state samplers: bounds coverage, mixtures, perturbations, and floating-base tilt invariance."""

from __future__ import annotations

import inspect
import math
import unittest

import torch
from torch import Tensor

from newton_neural import Frames
from newton_neural._src.core import rotation
from newton_neural._src.data.samplers import (
    SamplerBase,
    SamplerFloatingBase,
    SamplerMixture,
    SamplerPerturb,
    SamplerSobol,
    SamplerUniform,
)
from newton_neural.tests.unittest_utils import add_function_test


class TestSamplers(unittest.TestCase):
    pass


class _SamplerCounting(SamplerBase):
    """A sampler written as the interface asks, which takes the count alone. It numbers the states it draws."""

    def __init__(self, device: torch.device) -> None:
        self.device, self.drawn_count = device, 0

    def sample(self, count: int) -> Frames:
        numbers = torch.arange(self.drawn_count, self.drawn_count + count, device=self.device)
        self.drawn_count += count
        return {"joint_q": numbers[:, None].expand(count, 3).float()}


def _bounds(device: torch.device) -> dict[str, tuple[Tensor, Tensor]]:
    """Bounds of a coordinate signal ``[3]`` and a velocity signal ``[2, 2]`` on the device."""
    return {
        "joint_q": (torch.tensor([-1.0, 0.0, 2.0], device=device), torch.tensor([1.0, 0.5, 6.0], device=device)),
        "joint_qd": (torch.full((2, 2), -3.0, device=device), torch.full((2, 2), 5.0, device=device)),
    }


def _constant(value: float, device: torch.device) -> SamplerUniform:
    """A sampler whose every state equals ``value``."""
    return SamplerUniform({name: (torch.full_like(low, value),) * 2 for name, (low, _) in _bounds(device).items()})


def _assert_within(test: TestSamplers, states: Frames, bounds: dict[str, tuple[Tensor, Tensor]], count: int) -> None:
    test.assertEqual(states.keys(), bounds.keys())
    for name, (low, high) in bounds.items():
        test.assertEqual(states[name].shape, (count, *low.shape))
        test.assertEqual(states[name].dtype, torch.float32)
        test.assertEqual(states[name].device, low.device)
        test.assertTrue(((states[name] >= low) & (states[name] <= high)).all())


def test_sampler_uniform(test: TestSamplers, device: torch.device) -> None:
    """Draw states between the bounds that spread over the whole range."""
    bounds = _bounds(device)
    sampler = SamplerUniform(bounds, generator=torch.Generator(device).manual_seed(0))
    states = sampler.sample(4096)
    _assert_within(test, states, bounds, 4096)
    for name, (low, high) in bounds.items():
        test.assertTrue(((states[name].mean(dim=0) - (low + high) / 2).abs() < 0.03 * (high - low)).all())
        test.assertTrue((states[name].amax(dim=0) - states[name].amin(dim=0) > 0.98 * (high - low)).all())
    test.assertFalse(torch.equal(sampler.sample(16)["joint_q"], sampler.sample(16)["joint_q"]))


def test_sampler_sobol(test: TestSamplers, device: torch.device) -> None:
    """Cover every channel of the range evenly, also when the points come from several blocks and calls."""
    bounds = _bounds(device)
    sampler = SamplerSobol(bounds, block_size=256)
    parts = [sampler.sample(count) for count in (100, 156, 256, 500, 12)]  # 1024 points over four blocks
    for part, count in zip(parts, (100, 156, 256, 500, 12), strict=True):
        _assert_within(test, part, bounds, count)
    bin_count = 16
    for name, (low, high) in bounds.items():
        unit = (torch.cat([part[name] for part in parts]) - low) / (high - low)
        bins = (unit * bin_count).long().clamp_(max=bin_count - 1).flatten(1)
        for channel in bins.unbind(1):
            counts = torch.bincount(channel, minlength=bin_count)
            # A random sample of 1024 points scatters by about 8 around 64 a bin. The sequence is exact up to rounding.
            test.assertTrue(((counts - 1024 // bin_count).abs() <= 1).all())
    other = SamplerSobol(bounds, seed=1).sample(8)
    test.assertFalse(torch.equal(other["joint_q"], SamplerSobol(bounds).sample(8)["joint_q"]))


def test_sampler_perturb(test: TestSamplers, device: torch.device) -> None:
    """Add noise of the given standard deviation to the listed signals and leave the others as sampled."""
    sampler = SamplerPerturb(_constant(2.0, device), {"joint_q": 0.1}, generator=torch.Generator(device).manual_seed(0))
    states = sampler.sample(4096)
    noise = states["joint_q"] - 2.0
    test.assertEqual(noise.shape, (4096, 3))
    test.assertAlmostEqual(noise.std().item(), 0.1, delta=0.005)
    test.assertAlmostEqual(noise.mean().item(), 0.0, delta=0.005)
    test.assertTrue((states["joint_qd"] == 2.0).all())


def test_sampler_mixture(test: TestSamplers, device: torch.device) -> None:
    """Take each state from one sampler chosen by weight, and ask each sampler for as many states as it gives."""
    count = 4096
    mixture = SamplerMixture(
        {_constant(0.0, device): 1.0, _constant(1.0, device): 3.0}, generator=torch.Generator().manual_seed(0)
    )
    states = mixture.sample(count)
    unit = {name: (torch.zeros_like(low), torch.ones_like(low)) for name, (low, _) in _bounds(device).items()}
    _assert_within(test, states, unit, count)
    from_second = states["joint_q"][:, 0] == 1.0
    test.assertAlmostEqual(from_second.float().mean().item(), 0.75, delta=0.04)
    # A state comes from one sampler as a whole.
    test.assertTrue((states["joint_q"] == from_second[:, None]).all())
    test.assertTrue((states["joint_qd"] == from_second[:, None, None]).all())

    # Every sampler is asked for the number of states it contributes, with the count as its one argument.
    counting = _SamplerCounting(device), _SamplerCounting(device)
    numbered = SamplerMixture({counting[0]: 1.0, counting[1]: 1.0}, generator=torch.Generator().manual_seed(0))
    numbers = numbered.sample(count)["joint_q"][:, 0].sort().values
    drawn = counting[0].drawn_count, counting[1].drawn_count
    test.assertEqual(sum(drawn), count)
    expected = torch.cat([torch.arange(n, device=device) for n in drawn]).sort().values
    test.assertTrue(torch.equal(numbers, expected.float()))
    # A mixture is set by its samplers with their weights and by a generator, and every sampler draws from a
    # count alone.
    test.assertEqual(list(inspect.signature(SamplerMixture).parameters), ["samplers", "generator"])
    for sampler in (SamplerBase, SamplerUniform, SamplerSobol, SamplerFloatingBase, SamplerPerturb, SamplerMixture):
        test.assertEqual(list(inspect.signature(sampler.sample).parameters), ["self", "count"], sampler.__name__)
    with test.assertRaisesRegex(ValueError, "same signals"):
        SamplerMixture({_constant(0.0, device): 1.0, counting[0]: 1.0}).sample(64)
    # A sampler without a sample is refused where it is built.
    with test.assertRaisesRegex(TypeError, "abstract class SamplerBase .*sample"):
        SamplerBase()


def test_sampler_start(test: TestSamplers, device: torch.device) -> None:
    """Use the bounds a sampler is started with unless it has its own, and pass them on to the samplers inside."""
    bounds = _bounds(device)
    for sampler in (SamplerUniform(), SamplerSobol()):
        with test.assertRaisesRegex(RuntimeError, "no bounds"):
            sampler.sample(4)
        sampler.start(bounds)
        _assert_within(test, sampler.sample(64), bounds, 64)
    own = {"joint_q": (torch.full((3,), 7.0, device=device), torch.full((3,), 8.0, device=device))}
    for sampler in (SamplerUniform(own), SamplerSobol(own)):
        sampler.start(bounds)
        _assert_within(test, sampler.sample(64), own, 64)
    nested = SamplerMixture({SamplerPerturb(SamplerUniform(), {}): 1.0, SamplerSobol(): 1.0})
    nested.start(bounds)
    _assert_within(test, nested.sample(64), bounds, 64)


def test_sampler_floating_base(test: TestSamplers, device: torch.device) -> None:
    """Draw states of a floating base within every bound, turned by at most the tilt, with the heading kept."""
    count, tilt = 8192, 0.4
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    heading = torch.tensor([0.0, 0.0, math.sin(0.6), math.cos(0.6)], device=device)  # 1.2 rad about the vertical
    joints = torch.tensor([0.3, -0.2, 0.9], device=device)
    # A pose whose quaternion is not of unit length, as a hand written one may be.
    pose = torch.cat((torch.tensor([1.0, -2.0, 0.5], device=device), 2.0 * heading, joints))
    reach = torch.tensor([0.1, 0.0, 0.5], device=device)
    sampler = SamplerFloatingBase(
        pose,
        position=((-0.1, 0.0, 0.0), (0.3, 0.0, 0.2)),
        tilt=tilt,
        linear_speed=1.5,
        angular_speed=2.5,
        joint_reach=reach,
        joint_speed=3.0,
        generator=torch.Generator(device).manual_seed(0),
    )
    states = sampler.sample(count)
    test.assertEqual(list(states), ["joint_q", "joint_qd"])
    test.assertEqual((states["joint_q"].shape, states["joint_qd"].shape), ((count, 10), (count, 9)))
    test.assertEqual((states["joint_q"].device, states["joint_q"].dtype), (device, torch.float32))
    position, orientation, drawn_joints = states["joint_q"].split((3, 4, 3), dim=1)

    def assert_fills(values: Tensor, low: Tensor, high: Tensor) -> None:
        """Expect values ``[count, width]`` within the bounds ``[width]`` that reach both ends of each range."""
        test.assertTrue(((values >= low - 1e-6) & (values <= high + 1e-6)).all())
        span = high - low
        test.assertTrue((values.amin(dim=0) <= low + 0.01 * span + 1e-6).all())
        test.assertTrue((values.amax(dim=0) >= high - 0.01 * span - 1e-6).all())

    offset_low, offset_high = torch.tensor([[-0.1, 0.0, 0.0], [0.3, 0.0, 0.2]], device=device)
    assert_fills(position, pose[:3] + offset_low, pose[:3] + offset_high)
    assert_fills(drawn_joints, joints - reach, joints + reach)
    speed = torch.tensor([1.5] * 3 + [2.5] * 3 + [3.0] * 3, device=device)
    assert_fills(states["joint_qd"], -speed, speed)

    # Unit quaternions, turned from the pose about a horizontal axis by an angle up to the tilt.
    test.assertTrue(torch.allclose(orientation.norm(dim=1), torch.ones(count, device=device), atol=1e-6))
    turn = rotation.to_rotation_vector(rotation.multiply(orientation, rotation.conjugate(heading)))
    angle = turn.norm(dim=1)
    test.assertLessEqual(angle.max().item(), tilt + 1e-5)
    test.assertGreater(angle.max().item(), 0.99 * tilt)
    test.assertLess(turn[:, 2].abs().max().item(), 1e-5)
    # The turns are spread evenly over the disc. Half of them lie within the tilt divided by the root of two,
    # and their axes point in every direction.
    test.assertAlmostEqual((angle < tilt / math.sqrt(2.0)).float().mean().item(), 0.5, delta=0.03)
    test.assertLess(turn[:, :2].mean(dim=0).abs().max().item(), 0.03 * tilt)
    quadrant = (turn[:, 0] > 0).long() + 2 * (turn[:, 1] > 0).long()
    test.assertTrue((torch.bincount(quadrant, minlength=4) > 0.22 * count).all())
    # The heading of the library, the twist about the vertical, is that of the pose in every state.
    test.assertTrue(torch.allclose(rotation.twist(orientation, up), heading.expand(count, 4), atol=1e-5))

    # Without bounds every state is the pose at rest, with its quaternion of unit length.
    still = SamplerFloatingBase(pose).sample(4)
    expected = torch.cat((pose[:3], heading, joints))
    test.assertTrue(torch.allclose(still["joint_q"], expected.expand(4, 10), atol=1e-6))
    test.assertTrue((still["joint_qd"] == 0.0).all())
    # A base without joints, and bounds per joint given as a list.
    test.assertEqual(SamplerFloatingBase(pose[:7], tilt=0.1).sample(3)["joint_qd"].shape, (3, 6))
    listed = SamplerFloatingBase(pose, joint_speed=[1.0, 0.0, 2.0]).sample(64)["joint_qd"]
    test.assertTrue((listed[:, 7] == 0.0).all() and listed[:, 8].abs().max().item() > 1.0)

    with test.assertRaisesRegex(ValueError, r"pose must be joint_q of one world.*got \[2, 10\]"):
        SamplerFloatingBase(pose.expand(2, 10))
    with test.assertRaisesRegex(ValueError, r"pose must be joint_q of one world.*got \[3\]"):
        SamplerFloatingBase(pose[:3])
    with test.assertRaisesRegex(ValueError, r"joint_reach must be one number or one value for each of the 3 joints"):
        SamplerFloatingBase(pose, joint_reach=[0.1, 0.2])
    with test.assertRaisesRegex(ValueError, r"position must be the lowest and the highest offset.*got \[3\]"):
        SamplerFloatingBase(pose, position=(0.1, 0.1, 0.1))


add_function_test(TestSamplers, "test_sampler_uniform", test_sampler_uniform)
add_function_test(TestSamplers, "test_sampler_sobol", test_sampler_sobol)
add_function_test(TestSamplers, "test_sampler_perturb", test_sampler_perturb)
add_function_test(TestSamplers, "test_sampler_mixture", test_sampler_mixture)
add_function_test(TestSamplers, "test_sampler_floating_base", test_sampler_floating_base)
add_function_test(TestSamplers, "test_sampler_start", test_sampler_start)


if __name__ == "__main__":
    unittest.main()
