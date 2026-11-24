# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``OptimizerMuon`` and ``create_optimizer``: PyTorch parity, param splitting, and state dicts."""

from __future__ import annotations

import copy
import io
import math
import unittest

import torch
from torch import Tensor, nn

from newton_neural.backbones import BackboneFlash, BackboneTransformer
from newton_neural.models import FlashNeRD, NeRDBase
from newton_neural.tests.extensions import ContactInputRootHeight
from newton_neural.tests.helpers import NERDS, make_backbone, make_batch, make_nerd, train_on_pendulums
from newton_neural.tests.unittest_utils import add_function_test
from newton_neural.training import LossOneStep, OptimizerMuon, create_optimizer


def _nerd(backbone: str) -> FlashNeRD:
    return make_nerd("quadruped", "cpu", contact_input=ContactInputRootHeight(), **NERDS[backbone])


def _groups(optimizer: torch.optim.Optimizer, nerd: NeRDBase) -> tuple[set[str], set[str]]:
    """The names of the parameters Muon and AdamW update."""
    names = {id(p): name for name, p in nerd.named_parameters()}
    muon = {names[id(p)] for g in optimizer.param_groups if g["muon"] for p in g["params"]}
    adamw = {names[id(p)] for g in optimizer.param_groups if not g["muon"] for p in g["params"]}
    return muon, adamw


def _step(optimizer: torch.optim.Optimizer, params: list[Tensor], grads: list[Tensor]) -> None:
    for p, g in zip(params, grads, strict=True):
        p.grad = g.clone()
    optimizer.step()


def test_adamw_trains(test: unittest.TestCase, device: torch.device) -> None:
    """Train a small model with AdamW by name and expect a tenth of the one step error on unseen windows."""
    before, after = train_on_pendulums(device, "adamw")
    test.assertLess(after, 0.1 * before)


def test_parameter_without_gradient(test: unittest.TestCase, device: torch.device, optimizer_name: str) -> None:
    """Take a training step of models whose blocks attend over one frame and expect the lag biases, which get no
    gradient there, to stay and every other parameter to move."""
    backbones = (
        lambda: BackboneFlash(head_count=2, head_dim=8, ffn_dim=16, spans=(1, 1)),
        lambda: BackboneTransformer(width=16, head_count=2, depth=2, span=1, ffn_dim=32),
    )
    for backbone in backbones:
        nerd = make_nerd("arm", device, backbone=backbone)
        optimizer = create_optimizer(nerd, optimizer_name, lr=1e-2)
        before = {key: parameter.detach().clone() for key, parameter in nerd.named_parameters()}
        LossOneStep()(nerd, make_batch(nerd, 8, 4, device=device)).backward()
        optimizer.step()
        idle = {key for key, parameter in nerd.named_parameters() if parameter.grad is None}
        test.assertEqual(idle, {f"network.backbone.blocks.{index}.attention.lag_bias" for index in range(2)})
        moved = {key for key, parameter in nerd.named_parameters() if not torch.equal(parameter, before[key])}
        test.assertEqual(moved, set(before) - idle)


class TestOptim(unittest.TestCase):
    def test_muon_takes_the_hidden_layers(self) -> None:
        """Create the default optimizer for each backbone and expect Muon on hidden weights, AdamW on the rest."""
        for backbone in ("flash", "transformer", "mlp"):
            with self.subTest(backbone=backbone):
                nerd = _nerd(backbone)
                muon, adamw = _groups(create_optimizer(nerd), nerd)
                params = dict(nerd.named_parameters())
                self.assertEqual(muon | adamw, set(params))
                self.assertFalse(muon & adamw)
                hidden = nerd.network.hidden_weights()
                self.assertEqual(muon, {name for name, p in params.items() if any(p is w for w in hidden)})
                self.assertTrue(muon)
                self.assertTrue(all(params[name].ndim >= 2 for name in muon))
                self.assertTrue(all(name in adamw for name, p in params.items() if p.ndim < 2))
                # The layers that read features and write predictions, and everything about contact.
                reading_and_writing = {
                    "network.embed.weight",
                    "network.readout_out.weight",
                    "network.direct.weight",
                    "network.injection.project.weight",
                    "contact_encoder.project.weight",
                }
                self.assertLessEqual(reading_and_writing, adamw)
                self.assertTrue(all("lag_bias" not in name for name in muon))

    def test_muon_takes_named_hidden_weights(self) -> None:
        """Ask for Muon on a bare backbone and get its hidden weights, and on a module that names none an error."""
        backbone = make_backbone()
        optimizer = create_optimizer(module=backbone)
        muon = {id(p) for group in optimizer.param_groups if group["muon"] for p in group["params"]}
        self.assertEqual(muon, {id(weight) for weight in backbone.hidden_weights()})
        self.assertTrue(muon)
        with self.assertRaisesRegex(ValueError, r"names no hidden weights.*hidden_weights\(\).*name='adamw'"):
            create_optimizer(nn.Linear(4, 4))
        self.assertIs(type(create_optimizer(nn.Linear(4, 4), "adamw")), torch.optim.AdamW)

    def test_adamw_by_name(self) -> None:
        """Ask for AdamW by name and expect torch's AdamW with the given settings, and an error for other names."""
        nerd = _nerd("mlp")
        optimizer = create_optimizer(nerd, "adamw", lr=2e-3, weight_decay=1e-5)
        self.assertIs(type(optimizer), torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 2e-3)
        self.assertEqual(optimizer.defaults["weight_decay"], 1e-5)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), len(list(nerd.parameters())))
        with self.assertRaisesRegex(ValueError, "unknown optimizer"):
            create_optimizer(nerd, "sgd")

    def test_adamw_part_steps_as_torch_adamw(self) -> None:
        """Step vectors with OptimizerMuon and with torch's AdamW and expect the same parameters."""
        torch.manual_seed(0)
        params = [torch.randn(5, requires_grad=True), torch.randn(3, requires_grad=True)]
        twins = [p.detach().clone().requires_grad_() for p in params]
        ours = OptimizerMuon(params, lr=1e-2, weight_decay=0.1, betas=(0.8, 0.9), eps=1e-6)
        theirs = torch.optim.AdamW(twins, lr=1e-2, weight_decay=0.1, betas=(0.8, 0.9), eps=1e-6)
        for _ in range(5):
            grads = [torch.randn_like(p) for p in params]
            _step(ours, params, grads)
            _step(theirs, twins, grads)
        for p, twin in zip(params, twins, strict=True):
            self.assertTrue(torch.allclose(p, twin, atol=1e-7))

    def test_muon_part_steps_as_torch_muon(self) -> None:
        """Step wide and tall matrices in one batch and expect the steps of torch's Muon matched to AdamW."""
        torch.manual_seed(0)
        weights = [torch.randn(*shape, requires_grad=True) for shape in ((16, 24), (40, 12), (8, 8), (24, 16))]
        twins = [w.detach().clone().requires_grad_() for w in weights]
        ours = OptimizerMuon(weights, lr=1e-2, weight_decay=0.1)
        theirs = torch.optim.Muon(twins, lr=1e-2, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
        starts = [w.detach().clone() for w in weights]
        for _ in range(3):
            grads = [torch.randn_like(w) for w in weights]
            _step(ours, weights, grads)
            _step(theirs, twins, grads)
        for w, twin, start in zip(weights, twins, starts, strict=True):
            a, b = (w - start).detach().flatten(), (twin - start).detach().flatten()
            self.assertGreater(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 0.999)
            self.assertLess(abs(a.norm().item() / b.norm().item() - 1), 0.01)

    def test_convolution_kernels_step_as_matrices(self) -> None:
        """Step a convolution kernel from zero and expect a nearly orthogonal step of the size of an AdamW step."""
        torch.manual_seed(0)
        kernel = torch.zeros(32, 16, 3, 3, requires_grad=True)
        optimizer = OptimizerMuon([kernel], lr=1e-3, weight_decay=0.0)
        _step(optimizer, [kernel], [torch.randn(32, 16, 3, 3)])
        step = -kernel.detach().reshape(32, -1) / (1e-3 * 0.2 * math.sqrt(16 * 9))
        singular = torch.linalg.svdvals(step)
        self.assertGreater(singular.min().item(), 0.6)
        self.assertLess(singular.max().item(), 1.25)
        rms = kernel.detach().square().mean().sqrt().item()
        self.assertTrue(0.1e-3 < rms < 0.25e-3)  # about 0.2 times the learning rate, as an AdamW step

    def test_groups_by_shape_unless_told(self) -> None:
        """Pass a matrix and a vector with and without a group mark and expect the split by shape or by mark."""
        weight, bias = torch.randn(4, 3, requires_grad=True), torch.randn(4, requires_grad=True)
        optimizer = OptimizerMuon([weight, bias])
        self.assertEqual([(g["muon"], len(g["params"])) for g in optimizer.param_groups], [(True, 1), (False, 1)])
        optimizer = OptimizerMuon([{"params": [weight, bias], "muon": False}])
        self.assertEqual([g["muon"] for g in optimizer.param_groups], [False])
        with self.assertRaisesRegex(ValueError, "matrices"):
            OptimizerMuon([{"params": [bias], "muon": True}])

    def test_saved_optimizer_resumes(self) -> None:
        """Save an optimizer under a schedule, load it into a twin and expect identical next steps."""
        torch.manual_seed(0)
        nerd = _nerd("flash")
        optimizer = create_optimizer(nerd, lr=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-5)
        params = list(nerd.parameters())
        for _ in range(2):
            _step(optimizer, params, [torch.randn_like(p) for p in params])
            schedule.step()
        self.assertTrue(all(g["lr"] < 1e-3 for g in optimizer.param_groups))
        twin = copy.deepcopy(nerd)
        resumed = create_optimizer(twin, lr=1e-3)
        saved = io.BytesIO()
        torch.save(optimizer.state_dict(), saved)
        saved.seek(0)
        resumed.load_state_dict(torch.load(saved))
        grads = [torch.randn_like(p) for p in params]
        _step(optimizer, params, grads)
        _step(resumed, list(twin.parameters()), grads)
        for p, q in zip(params, twin.parameters(), strict=True):
            self.assertTrue(torch.equal(p, q))


add_function_test(TestOptim, "test_adamw_trains", test_adamw_trains)
for _optimizer_name in ("muon", "adamw"):
    add_function_test(
        TestOptim,
        f"test_parameter_without_gradient_{_optimizer_name}",
        test_parameter_without_gradient,
        optimizer_name=_optimizer_name,
    )


if __name__ == "__main__":
    unittest.main()
