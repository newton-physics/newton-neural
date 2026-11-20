# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``FlashNeRDNetwork``: streaming parity, contact injections, zero-init skips, compile, and graphs."""

from __future__ import annotations

import unittest
from typing import Any

import torch
from torch import Tensor, nn

from newton_neural._src.models.flashnerd_network import ContactInjection, FlashNeRDNetwork
from newton_neural.testing import check_backbone_streaming
from newton_neural.tests.extensions import BackboneLeaky
from newton_neural.tests.helpers import BACKBONES, WORLD_COUNT, assert_close
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

FEATURE_COUNT = 7
OUTPUT_COUNT = 5
CODE_COUNT = 6
READOUT_HIDDEN_DIM = 16

CODE = {"code_count": CODE_COUNT, "injections": tuple(ContactInjection)}
"""Network options for a contact code that enters at every injection point."""

HISTORY_SHAPES = {
    "flash": {"ring0": (3, 16), "ring1": (1, 16), "ring2": (4, 16), "step": ()},
    "transformer": {"ring0": (3, 32), "ring1": (3, 32), "step": ()},
    "mlp": {"frames": (3, 16), "step": ()},
}
"""Shape of every history tensor of the backbones of ``BACKBONES``, for one world."""


def _network(
    name: str, device: torch.device, *, dtype: torch.dtype = torch.float64, **options: Any
) -> FlashNeRDNetwork:
    """A small network around the backbone ``name`` with every parameter random, so that no term vanishes."""
    torch.manual_seed(0)
    network = FlashNeRDNetwork(
        FEATURE_COUNT, OUTPUT_COUNT, BACKBONES[name](), readout_hidden_dim=READOUT_HIDDEN_DIM, **options
    )
    for parameter in network.parameters():
        nn.init.normal_(parameter, std=0.3)
    return network.to(device=device, dtype=dtype).eval()


def _frames(network: FlashNeRDNetwork, *, frame_count: int | None = None, width: int = FEATURE_COUNT) -> Tensor:
    """Random frames ``[WORLD_COUNT, frame_count, width]``, by default longer than twice the receptive field."""
    weight = network.norm.weight
    frame_count = frame_count or 2 * network.receptive_field + 3
    return torch.randn(WORLD_COUNT, frame_count, width, dtype=torch.float64).to(weight)


def _stream(network: FlashNeRDNetwork, x: Tensor, history: dict[str, Tensor], code: Tensor | None = None) -> Tensor:
    """Step through the frames ``x`` ``[B, T, feature_count]`` and stack the predictions ``[B, T, output_count]``."""
    codes = [None] * x.shape[1] if code is None else code.unbind(1)
    return torch.stack([network.step(frame, history, c) for frame, c in zip(x.unbind(1), codes, strict=True)], dim=1)


def test_step_matches_forward(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Step through a sequence longer than the receptive field and compare with the whole window forward."""
    network = _network(backbone, device)
    x = _frames(network)
    with torch.no_grad():
        assert_close(_stream(network, x, network.history(WORLD_COUNT)), network(x))


def test_staggered_resets(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Reset every world at a different frame and compare each with a forward that starts at its reset."""
    network = _network(backbone, device)
    x = _frames(network)
    starts = torch.arange(WORLD_COUNT, device=device) * x.shape[1] // WORLD_COUNT
    history = network.history(WORLD_COUNT)
    with torch.no_grad():
        outputs = []
        for frame in range(x.shape[1]):
            network.reset_history(history, starts == frame)
            outputs.append(network.step(x[:, frame], history))
        stepped = torch.stack(outputs, dim=1)
        for world, start in enumerate(starts.tolist()):
            assert_close(stepped[world : world + 1, start:], network(x[world : world + 1, start:]))


def test_reset_forgets_nan_history(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Fill the history of one world with NaN, reset it and expect the finite output of a fresh world."""
    network = _network(backbone, device)
    x = _frames(network)
    history = network.history(WORLD_COUNT)
    mask = torch.arange(WORLD_COUNT, device=device) == 1
    with torch.no_grad():
        _stream(network, x[:, :4], history)
        for value in history.values():
            if value.is_floating_point():
                value[mask] = torch.nan
        network.reset_history(history, mask)
        y = network.step(x[:, 4], history)
        test.assertTrue(torch.isfinite(y).all())
        assert_close(y[mask], network(x[mask, 4:5])[:, 0])
        assert_close(y[~mask], network(x[~mask, :5])[:, -1])


def test_stale_history_cannot_leak(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Mark a world as fresh while its stored values are NaN and check that they never reach its output."""
    network = _network(backbone, device)
    x = _frames(network)
    history = network.history(WORLD_COUNT)
    for key, value in history.items():
        if key != "step":
            value[1] = torch.nan
    with torch.no_grad():
        assert_close(_stream(network, x, history), network(x))


def test_worlds_are_independent(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Feed one world NaN and check that the outputs of the other worlds do not change."""
    network = _network(backbone, device)
    x = _frames(network)
    poisoned = x.clone()
    poisoned[-1] = torch.nan
    with torch.no_grad():
        stepped = _stream(network, poisoned, network.history(WORLD_COUNT))
        test.assertTrue(stepped[-1].isnan().all())
        assert_close(stepped[:-1], network(x)[:-1])


def test_history_layout(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Check that the history has fixed shapes and the requested dtype, starts at zero and is updated in place."""
    network = _network(backbone, device, dtype=torch.float32)
    x = _frames(network)
    with torch.no_grad():
        reference = _stream(network, x, network.history(WORLD_COUNT))
        for dtype in (torch.float32, torch.float16):
            history = network.history(WORLD_COUNT, dtype=dtype)
            created = dict(history)
            shapes = {key: (WORLD_COUNT, *shape) for key, shape in HISTORY_SHAPES[backbone].items()}
            test.assertEqual({key: tuple(value.shape) for key, value in history.items()}, shapes)
            for key, value in history.items():
                test.assertEqual(value.dtype, torch.int64 if key == "step" else dtype)
                test.assertEqual(value.device, device)
                test.assertFalse(value.any())
            stepped = _stream(network, x, history)
            for key, value in created.items():
                test.assertIs(history[key], value)
                test.assertEqual(tuple(value.shape), shapes[key])
                test.assertEqual(value.dtype, torch.int64 if key == "step" else dtype)
            test.assertTrue((history["step"] == x.shape[1]).all())
            # A float16 history rounds what it stores, so its predictions are close but not equal.
            assert_close(stepped, reference, 1e-6 if dtype == torch.float32 else 2e-2)
    for value in _network(backbone, device).history(WORLD_COUNT).values():  # the defaults are the network's own
        test.assertEqual((value.device, value.dtype in (torch.float64, torch.int64)), (device, True))


def test_autocast(test: unittest.TestCase, device: torch.device, backbone: str, autocast: torch.dtype) -> None:
    """Run forward and steps under autocast and expect agreement, with float32 norms, history and output."""
    network = _network(backbone, device, dtype=torch.float32)
    x = _frames(network)
    norm_dtypes = []
    for module in network.modules():
        if isinstance(module, nn.RMSNorm):
            module.register_forward_pre_hook(lambda _, inputs: norm_dtypes.append(inputs[0].dtype))
    history = network.history(WORLD_COUNT)
    with torch.no_grad():
        exact = network(x)
        with torch.autocast(device.type, dtype=autocast):
            whole = network(x)
            stepped = _stream(network, x, history)
    test.assertEqual({whole.dtype, stepped.dtype, *norm_dtypes}, {torch.float32})
    test.assertTrue(all(value.dtype in (torch.float32, torch.int64) for value in history.values()))
    assert_close(stepped, whole, 5e-2)
    assert_close(whole, exact, 1e-1)
    check_backbone_streaming(BACKBONES[backbone](), device=device, dtype=torch.float32, autocast=autocast)


def test_compiled_step_matches_eager(
    test: unittest.TestCase, device: torch.device, backbone: str, autocast: torch.dtype | None = None
) -> None:
    """Compile the step as one static graph and compare it with the eager step, across a reset."""
    network = _network(backbone, device, dtype=torch.float32, **CODE, gated=True)
    x = _frames(network, frame_count=8)
    code = _frames(network, frame_count=8, width=CODE_COUNT)
    mask = torch.arange(WORLD_COUNT, device=device) % 2 == 0
    tolerance = 1e-5 if autocast is None else 2e-2

    def step(x: Tensor, history: dict[str, Tensor], code: Tensor) -> Tensor:
        with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
            return network.step(x, history, code)

    test.addCleanup(torch.compiler.reset)
    compiled = torch.compile(step, fullgraph=True, dynamic=False)
    histories = network.history(WORLD_COUNT), network.history(WORLD_COUNT)
    with torch.no_grad():
        for frame in range(x.shape[1]):
            if frame == 5:
                for history in histories:
                    network.reset_history(history, mask)
            expected = step(x[:, frame], histories[0], code[:, frame])
            assert_close(compiled(x[:, frame], histories[1], code[:, frame]), expected, tolerance)
    for key, value in histories[0].items():
        assert_close(histories[1][key].double(), value.double(), tolerance)


def test_cuda_graph_replay(test: unittest.TestCase, device: torch.device, backbone: str) -> None:
    """Capture the step in a CUDA graph, replay it frame by frame and compare with the eager step, across a reset."""
    network = _network(backbone, device, dtype=torch.float32)
    x = _frames(network, frame_count=8)
    mask = torch.arange(WORLD_COUNT, device=device) % 2 == 0
    frame_input = torch.zeros_like(x[:, 0])
    histories = network.history(WORLD_COUNT), network.history(WORLD_COUNT)
    with torch.no_grad():
        network.step(frame_input, network.history(WORLD_COUNT))  # libraries initialize outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = network.step(frame_input, histories[1])
        for frame in range(x.shape[1]):
            if frame == 5:
                for history in histories:
                    network.reset_history(history, mask)
            frame_input.copy_(x[:, frame])
            graph.replay()
            assert_close(output, network.step(x[:, frame], histories[0]), 1e-5)


class TestFlashNeRDNetwork(unittest.TestCase):
    """The network of the FlashNeRD family, with its streaming around every backbone on every device, its
    contact injection, its readout and its hidden weights."""

    def test_zero_code_equals_network_without_code(self) -> None:
        """Share the weights with a network built without a contact code and expect its output for a zero code."""
        for gated in (False, True):
            with self.subTest(gated=gated):
                coded = _network("flash", torch.device("cpu"), **CODE, gated=gated)
                plain = _network("flash", torch.device("cpu"))
                self.assertIsNone(plain.injection)
                missing, unexpected = plain.load_state_dict(coded.state_dict(), strict=False)
                self.assertEqual(missing, [])
                self.assertTrue(unexpected and all(key.startswith("injection.") for key in unexpected))
                x = _frames(coded)
                code = torch.zeros(*x.shape[:2], CODE_COUNT, dtype=x.dtype)
                with torch.no_grad():
                    assert_close(coded(x, code), plain(x), 1e-12)
                    stepped = _stream(coded, x, coded.history(WORLD_COUNT), code)
                    assert_close(stepped, _stream(plain, x, plain.history(WORLD_COUNT)), 1e-12)
                    self.assertGreater((coded(x, torch.ones_like(code)) - plain(x)).abs().max().item(), 1e-3)

    def test_code_is_required_exactly_when_built_with_one(self) -> None:
        """Expect an error for a missing code, an unexpected code and a code without injection points."""
        coded = _network("mlp", torch.device("cpu"), code_count=CODE_COUNT, injections=("embedding",))
        plain = _network("mlp", torch.device("cpu"))
        x = _frames(plain)
        code = torch.zeros(*x.shape[:2], CODE_COUNT, dtype=x.dtype)
        with self.assertRaises(ValueError):
            coded(x)
        with self.assertRaises(ValueError):
            coded.step(x[:, 0], coded.history(WORLD_COUNT))
        with self.assertRaises(ValueError):
            plain(x, code)
        with self.assertRaises(ValueError):
            FlashNeRDNetwork(FEATURE_COUNT, OUTPUT_COUNT, BACKBONES["mlp"](), code_count=CODE_COUNT)
        with self.assertRaises(ValueError):
            FlashNeRDNetwork(FEATURE_COUNT, OUTPUT_COUNT, BACKBONES["mlp"](), injections=("readout",))
        with self.assertRaises(ValueError):
            FlashNeRDNetwork(
                FEATURE_COUNT, OUTPUT_COUNT, BACKBONES["mlp"](), code_count=CODE_COUNT, injections=("input",)
            )

    def test_direct_paths_start_at_zero(self) -> None:
        """Build a fresh network and expect the direct path and the increment injection to add nothing yet."""
        torch.manual_seed(0)
        network = FlashNeRDNetwork(FEATURE_COUNT, OUTPUT_COUNT, BACKBONES["flash"](), **CODE)
        self.assertFalse(network.direct.weight.any() or network.direct.bias.any())
        x = torch.randn(WORLD_COUNT, 3, FEATURE_COUNT)
        code = torch.randn(WORLD_COUNT, 3, CODE_COUNT)
        with torch.no_grad():
            terms = network.injection(code)
            self.assertFalse(terms[ContactInjection.INCREMENT].any())
            self.assertTrue(terms[ContactInjection.EMBEDDING].any() and terms[ContactInjection.READOUT].any())
            h = network.backbone(network.embed(x) + terms[ContactInjection.EMBEDDING])
            p = network.readout_hidden(torch.cat((network.norm(h), x), dim=-1)) + terms[ContactInjection.READOUT]
            assert_close(network(x, code), network.readout_out(torch.relu(p)), 1e-6)

    def test_injection_math(self) -> None:
        """Compute gated and ungated injection by hand from the weights, for all points and for each alone."""
        for gated in (False, True):
            for points in (tuple(ContactInjection), *((point,) for point in ContactInjection)):
                with self.subTest(gated=gated, points=points):
                    options = {"code_count": CODE_COUNT, "injections": points, "gated": gated}
                    network = _network("transformer", torch.device("cpu"), **options)
                    injection = network.injection
                    widths = {"embedding": 16, "readout": READOUT_HIDDEN_DIM, "increment": OUTPUT_COUNT}
                    self.assertEqual(injection.project.weight.shape, (sum(widths[p] for p in points), CODE_COUNT))
                    self.assertIsNone(injection.project.bias)
                    x = _frames(network)
                    code = _frames(network, width=CODE_COUNT)
                    maps = dict(zip(points, injection.project.weight.split([widths[p] for p in points]), strict=True))
                    gates = torch.sigmoid(injection.gate) if gated else torch.ones(len(points), dtype=x.dtype)
                    term = dict.fromkeys(ContactInjection, 0.0)
                    for gate, (point, weight) in zip(gates, maps.items(), strict=True):
                        term[point] = gate * (code @ weight.T)
                    with torch.no_grad():
                        h = network.backbone(network.embed(x) + term[ContactInjection.EMBEDDING])
                        p = network.readout_hidden(torch.cat((network.norm(h), x), dim=-1))
                        p = p + term[ContactInjection.READOUT]
                        y = network.readout_out(torch.relu(p)) + network.direct(x) + term[ContactInjection.INCREMENT]
                        assert_close(network(x, code), y, 1e-12)
                        assert_close(_stream(network, x, network.history(WORLD_COUNT), code), y)
        fresh = FlashNeRDNetwork(FEATURE_COUNT, OUTPUT_COUNT, BACKBONES["mlp"](), **CODE, gated=True)
        assert_close(torch.sigmoid(fresh.injection.gate), torch.full((3,), 0.1), 1e-6)

    def test_forward_trains_every_parameter(self) -> None:
        """Differentiate a loss on the whole window forward and expect a gradient for every parameter."""
        for name in BACKBONES:
            with self.subTest(backbone=name):
                network = _network(name, torch.device("cpu"), **CODE, gated=True)
                x = _frames(network)
                network(x, _frames(network, width=CODE_COUNT)).square().sum().backward()
                # A block that attends over one frame gives all its weight to it, whatever its query and bias are.
                idle = {"backbone.blocks.1.attention.lag_bias"} if name == "flash" else set()
                silent = {
                    key
                    for key, parameter in network.named_parameters()
                    if parameter.grad is None or not parameter.grad.any()
                }
                self.assertEqual(silent, idle)

    def test_hidden_weights(self) -> None:
        """Expect the weights of the hidden linear layers only, without embedding, output layers and injection."""
        counts = {"flash": 3 * 2 + 1, "transformer": 2 * 4 + 1, "mlp": 2 + 1}
        for name, count in counts.items():
            with self.subTest(backbone=name):
                network = _network(name, torch.device("cpu"), **CODE)
                hidden = network.hidden_weights()
                excluded = (network.embed, network.readout_out, network.direct, network.injection.project)
                linear = [module for module in network.modules() if isinstance(module, nn.Linear)]
                expected = {id(module.weight) for module in linear if module not in excluded}
                self.assertEqual(len(hidden), count)
                self.assertEqual({id(weight) for weight in hidden}, expected)
                self.assertTrue(all(weight.dim() == 2 for weight in hidden))
                self.assertEqual(len(network.backbone.hidden_weights()), count - 1)

    def test_receptive_field(self) -> None:
        """Expect the receptive field of the backbone, and exactly the frames inside it to change a prediction."""
        for name in BACKBONES:
            with self.subTest(backbone=name):
                network = _network(name, torch.device("cpu"))
                reach = network.receptive_field
                self.assertEqual(reach, network.backbone.receptive_field)
                x = _frames(network)
                outside, inside = x.clone(), x.clone()
                outside[:, :-reach] += 1.0
                inside[:, -reach] += 1.0
                with torch.no_grad():
                    last = network(x)[:, -1]
                    assert_close(network(outside)[:, -1], last, 1e-12)
                    self.assertGreater((network(inside)[:, -1] - last).abs().max().item(), 1e-6)

    def test_reset_history_by_world_mask(self) -> None:
        """Reset one world of a history through a backbone and through its network, naming the argument
        ``world_mask``, and expect zeros in that world alone."""
        for make in BACKBONES.values():
            backbone = make()
            for owner in (backbone, FlashNeRDNetwork(FEATURE_COUNT, OUTPUT_COUNT, backbone)):
                history = backbone.history(2, device="cpu", dtype=torch.float32)
                for value in history.values():
                    value.fill_(1)
                owner.reset_history(history, world_mask=torch.tensor([True, False]))
                for value in history.values():
                    self.assertFalse(value[0].any())
                    self.assertTrue(value[1].all())

    def test_extension_backbone(self) -> None:
        """Run a backbone defined outside the package through the shell and the conformance check."""
        check_backbone_streaming(BackboneLeaky())
        network = FlashNeRDNetwork(FEATURE_COUNT, OUTPUT_COUNT, BackboneLeaky()).double()
        self.assertIsNone(network.receptive_field)
        x = torch.randn(WORLD_COUNT, 9, FEATURE_COUNT, dtype=torch.float64)
        with torch.no_grad():
            assert_close(_stream(network, x, network.history(WORLD_COUNT)), network(x))
        self.assertEqual(len(network.hidden_weights()), 2)


cuda_devices = get_cuda_test_devices()
for backbone_name in BACKBONES:
    for function in (
        test_step_matches_forward,
        test_staggered_resets,
        test_reset_forgets_nan_history,
        test_stale_history_cannot_leak,
        test_worlds_are_independent,
        test_history_layout,
    ):
        add_function_test(
            TestFlashNeRDNetwork, f"{function.__name__}_{backbone_name}", function, backbone=backbone_name
        )
    for autocast_dtype in (torch.float16, torch.bfloat16):
        add_function_test(
            TestFlashNeRDNetwork,
            f"test_autocast_{str(autocast_dtype).removeprefix('torch.')}_{backbone_name}",
            test_autocast,
            devices=cuda_devices if autocast_dtype is torch.float16 else None,  # the CPU casts to bfloat16 alone
            backbone=backbone_name,
            autocast=autocast_dtype,
        )
    for function in (test_compiled_step_matches_eager, test_cuda_graph_replay):
        add_function_test(
            TestFlashNeRDNetwork,
            f"{function.__name__}_{backbone_name}",
            function,
            devices=cuda_devices,
            backbone=backbone_name,
        )
add_function_test(
    TestFlashNeRDNetwork,
    "test_compiled_step_matches_eager_float16_flash",
    test_compiled_step_matches_eager,
    devices=cuda_devices,
    backbone="flash",
    autocast=torch.float16,
)


if __name__ == "__main__":
    unittest.main()
