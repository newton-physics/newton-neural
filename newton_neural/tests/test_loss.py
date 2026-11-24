# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``LossOneStep``: L1/L2 accuracy, history masking, noise, contact recomputation, compile, and
training in several processes."""

from __future__ import annotations

import copy
import multiprocessing
import queue
import socket
import traceback
import unittest
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import torch
import torch.distributed
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from newton_neural.backbones import BackboneBase, BackboneMLP
from newton_neural.data import TrajectoryBatch
from newton_neural.models import FlashNeRD, NeRDBase
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.tests.extensions import BackboneLeaky, ContactInputRootHeight, ContactInputRootHeightRecorded
from newton_neural.tests.helpers import (
    CONTACT_INPUTS,
    NERDS,
    WORLD_COUNT,
    frame_of,
    make_backbone,
    make_batch,
    make_nerd,
    make_system,
    pendulum_windows,
    process_precision,
)
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices
from newton_neural.training import ErrorKind, LossBase, LossOneStep, create_optimizer

FRAME_COUNT = 6
"""Frames per window, which gives 5 predictions."""

VALUE_TOLERANCE = 1e-5
"""Largest difference from the eager loss that a checked call of a compiled loss accepts in IEEE float32, relative
to the eager loss or to one, whichever is greater."""

TENSOR_FLOAT_VALUE_TOLERANCE = 1e-3
"""The same in TensorFloat-32, which rounds a product to about three digits. Compiled code multiplies in another
order than eager code, and how far that moves the loss depends on the GPU."""

COMPILED_GRADIENT_TOLERANCE = 1e-4
"""Largest difference from the gradient of the eager loss that a compiled loss may show in IEEE float32, relative
to its length."""

TENSOR_FLOAT_GRADIENT_TOLERANCE = 1e-1
"""The same in TensorFloat-32 for the small models of the tests. Rounding moves the error of a few outputs across
zero, which flips their gradient under the absolute error, and the architecture of the GPU decides how many
products round."""

TENSOR_CORE_WINDOW_COUNT = 64
"""Windows with which TensorFloat-32 changes the losses of the small models of the tests. cuBLAS computes the
products of a few rows in IEEE arithmetic whatever is allowed."""

TRAIN_STEP_COUNT = 40
"""Optimizer steps of the training that compares the compiled with the eager loss."""

RANK_COUNT = 2
"""Processes that train one model together."""

DISTRIBUTED_TIMEOUT = 600.0
"""Longest wait [s] for the report of a process that trains with others, which compiles its loss on a GPU."""

COMPILED_CASES = (
    ("flash", "no_contact_input", "absolute", 0.0),
    ("flash", "probes", "squared", 0.3),
    ("flash", "contact_set", "absolute", 0.3),
    ("flash_span_one", "no_contact_input", "squared", 0.0),
    ("transformer", "no_contact_input", "squared", 0.3),
    ("transformer", "probes", "absolute", 0.0),
    ("transformer", "contact_set", "squared", 0.0),
    ("mlp", "no_contact_input", "absolute", 0.3),
    ("mlp", "probes", "squared", 0.3),
    ("mlp", "contact_set", "absolute", 0.0),
    ("minimal", "probes", "absolute", 0.3),
)
"""The models whose compiled loss is compared with the eager one, as a name in ``NERDS``, the contact input, the
error and the noise. Every backbone and every contact input meets both errors
and trains with and without noise, and one model attends over single frames, which leaves it parameters without
a gradient. They are more than the variants PyTorch's compiler keeps of one function."""


def _nerd_and_its_windows(
    device: torch.device, nerd_name: str, contact: str = "no_contact_input"
) -> tuple[NeRDBase, TrajectoryBatch]:
    """A small model of the quadruped, a name in ``NERDS``, and the windows it took its statistics from, enough
    of them for TensorFloat-32 to change its loss."""
    size = (TENSOR_CORE_WINDOW_COUNT, FRAME_COUNT)
    nerd = make_nerd("quadruped", device, contact_input=CONTACT_INPUTS[contact](), fitted_on=size, **NERDS[nerd_name])
    return nerd, make_batch(nerd, *size, device=device)


def _starting(batch: TrajectoryBatch, starts: list[bool]) -> TrajectoryBatch:
    """The batch with its windows marked as starting an episode or not."""
    episode_start = torch.tensor(starts, device=batch.episode_start.device)
    return TrajectoryBatch(batch.frames, episode_start, batch.frame_dt)


def _frame_errors(nerd: NeRDBase, batch: TrajectoryBatch) -> Tensor:
    """Mean absolute normalized error of every prediction of every window, ``[batch_size, frame_count - 1]``."""
    current, following = frame_of(batch.frames, slice(0, -1)), frame_of(batch.frames, slice(1, None))
    return (nerd(current) - nerd.labels(current, following)).abs().mean(dim=-1)


def test_error_kinds(test: unittest.TestCase, device: torch.device) -> None:
    """Compare both error kinds with the errors of predictions stepped frame by frame from an episode start."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    batch = _starting(make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device), [True] * WORLD_COUNT)
    history = nerd.history(WORLD_COUNT)
    differences = []
    with torch.no_grad():
        for index in range(FRAME_COUNT - 1):
            current, following = frame_of(batch.frames, index), frame_of(batch.frames, index + 1)
            predicted = nerd.output_normalizer.normalize(nerd.predict(current, history))
            differences.append(predicted - nerd.labels(current, following))
        difference = torch.stack(differences)
        test.assertTrue(torch.allclose(LossOneStep()(nerd, batch), difference.abs().mean(), rtol=1e-4))
        squared = LossOneStep(error=ErrorKind.SQUARED)(nerd, batch)
        test.assertTrue(torch.allclose(squared, difference.square().mean(), rtol=1e-4))
        test.assertTrue(torch.equal(LossOneStep(error="squared")(nerd, batch), squared))
    loss = LossOneStep()(nerd=nerd, batch=batch)
    loss.backward()
    test.assertTrue(all(parameter.grad is not None for parameter in nerd.parameters()))


def test_frames_that_count(test: unittest.TestCase, device: torch.device) -> None:
    """Expect frames without full history to count only in windows that start an episode."""
    nerd = make_nerd("quadruped", device)  # a receptive field of 3, so two frames lack history
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    loss = LossOneStep()
    with torch.no_grad():
        errors = _frame_errors(nerd, batch)
        test.assertTrue(torch.allclose(loss(nerd, batch), errors[:, 2:].mean(), rtol=1e-5))
        started = _starting(batch, [True] * WORLD_COUNT)
        test.assertTrue(torch.allclose(loss(nerd, started), errors.mean(), rtol=1e-5))
        mixed = _starting(batch, [True, False, False, True])
        expected = (errors[0].sum() + errors[1, 2:].sum() + errors[2, 2:].sum() + errors[3].sum()) / 16
        test.assertTrue(torch.allclose(loss(nerd, mixed), expected, rtol=1e-5))


def test_frames_that_count_by_history(
    test: unittest.TestCase, device: torch.device, backbone: Callable[[], BackboneBase]
) -> None:
    """Expect every frame to count with a one frame history, and only episode starts with an unbounded one,
    which refuses a batch without one."""
    nerd = make_nerd("arm", device, backbone=backbone)
    batch = _starting(make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device), [False, True, False, False])
    with torch.no_grad():
        errors = _frame_errors(nerd, batch)
        expected = errors.mean() if nerd.receptive_field == 1 else errors[1].mean()
        test.assertTrue(torch.allclose(LossOneStep()(nerd, batch), expected, rtol=1e-5))
        if nerd.receptive_field is None:
            nowhere = _starting(batch, [False] * WORLD_COUNT)
            with test.assertRaisesRegex(ValueError, r"unbounded history.*weights=windows\.episode_start"):
                LossOneStep()(nerd, nowhere)


def test_noise(test: unittest.TestCase, device: torch.device) -> None:
    """Expect noise to change what the model reads, at random and the same under a seed, and to spare the batch."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    recorded = {name: value.clone() for name, value in batch.frames.items()}
    noisy = LossOneStep(noise=0.5)
    with torch.no_grad():
        clean = LossOneStep()(nerd, batch).item()
        torch.manual_seed(0)
        first = noisy(nerd, batch).item()
        torch.manual_seed(0)
        test.assertEqual(noisy(nerd, batch).item(), first)
        test.assertNotEqual(noisy(nerd, batch).item(), first)
        test.assertNotEqual(first, clean)
    for name, value in recorded.items():
        test.assertTrue(torch.equal(batch.frames[name], value))
    noisy(nerd, batch).backward()
    test.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in nerd.parameters()))


def test_noise_recomputes_contact(test: unittest.TestCase, device: torch.device) -> None:
    """Record contact signals that contradict the states and expect noise to replace them if recomputable."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    wrong_frames = {**batch.frames, "root_height": batch.frames["root_height"] + 5.0}
    wrong = TrajectoryBatch(wrong_frames, batch.episode_start, batch.frame_dt)
    recorded_only = FlashNeRD(
        make_system("quadruped"),
        backbone=make_backbone(),
        contact_input=ContactInputRootHeightRecorded(),
        readout_hidden_dim=16,
    )
    recorded_only.load_state_dict(nerd.state_dict())
    recorded_only.to(device)
    clean, slight = LossOneStep(), LossOneStep(noise=1e-9)  # too little noise to change a state
    with torch.no_grad():
        right = clean(nerd, batch).item()
        contradicted = clean(nerd, wrong).item()
        test.assertGreater(abs(contradicted - right), 1e-3)
        test.assertAlmostEqual(slight(nerd, wrong).item(), right, delta=1e-5)
        test.assertAlmostEqual(slight(recorded_only, wrong).item(), contradicted, delta=1e-5)


def test_noise_is_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Train a free motion model without noise, ask for noise and expect a ValueError."""
    nerd = make_nerd("arm", device, prediction=PredictionFreeMotion())
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    loss = LossOneStep()(nerd, batch)
    loss.backward()
    test.assertTrue(torch.isfinite(loss))
    with test.assertRaisesRegex(ValueError, "noise=0"):
        LossOneStep(noise=0.1)(nerd, batch)


def test_labels_of_the_frame(test: unittest.TestCase, device: torch.device) -> None:
    """Expect a free motion model to learn from every frame of a window, also from windows of single frames."""
    nerd = make_nerd(
        "arm", device, prediction=PredictionFreeMotion(), backbone=lambda: BackboneMLP(width=16, depth=1, window=1)
    )
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    with torch.no_grad():
        errors = (nerd(batch.frames) - nerd.labels(batch.frames, None)).abs().mean(dim=-1)
        test.assertEqual(errors.shape, (WORLD_COUNT, FRAME_COUNT))
        test.assertTrue(torch.allclose(LossOneStep()(nerd, batch), errors.mean(), rtol=1e-5))
        # Each frame on its own is a window. With a one frame history the predictions stay the same.
        frames = {name: value.flatten(0, 1).unsqueeze(1) for name, value in batch.frames.items()}
        starts = torch.zeros(WORLD_COUNT * FRAME_COUNT, dtype=torch.bool, device=device)
        single = TrajectoryBatch(frames, starts, batch.frame_dt)
        test.assertTrue(torch.allclose(LossOneStep()(nerd, single), errors.mean(), rtol=1e-5))

        # A longer history is complete in a single frame only where that frame starts an episode, so single
        # frames train such a model when every one of them does.
        longer = make_nerd("arm", device, prediction=PredictionFreeMotion())
        for start_count in (0, 3):
            starts[:start_count] = True
            with test.assertRaisesRegex(ValueError, "length=2"):
                LossOneStep()(longer, single)
        starts[:] = True
        errors = (longer(frames) - longer.labels(frames, None)).abs().mean(dim=-1)
        test.assertTrue(torch.allclose(LossOneStep()(longer, single), errors.mean(), rtol=1e-5))
    loss = LossOneStep()(nerd, single)
    loss.backward()
    test.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in nerd.parameters()))


def test_single_frames_are_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Give a model that predicts state increments windows of single frames and expect a ValueError."""
    nerd = make_nerd("arm", device)
    batch = make_batch(nerd, WORLD_COUNT, 1, device=device)
    with test.assertRaisesRegex(ValueError, "at least one step"):
        LossOneStep()(nerd, batch)


def test_short_windows_are_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Give windows with fewer frames to predict for than the receptive field and expect them refused."""
    nerd = make_nerd("arm", device)  # a receptive field of 3
    short = make_batch(nerd, WORLD_COUNT, 3, device=device)  # two frames to predict for
    with torch.no_grad():
        for starts in ([False] * WORLD_COUNT, [True, False, False, True]):
            with test.assertRaisesRegex(ValueError, "2 frames to predict for.*reads 3 frames.*length=3"):
                LossOneStep()(nerd, _starting(short, starts))
        # Every frame of a window that starts an episode has its whole past, so such windows may be short.
        started = _starting(short, [True] * WORLD_COUNT)
        test.assertTrue(torch.allclose(LossOneStep()(nerd, started), _frame_errors(nerd, short).mean(), rtol=1e-5))
        # Windows of the suggested length hold one frame with its full history each.
        enough = make_batch(nerd, WORLD_COUNT, 4, device=device)
        expected = _frame_errors(nerd, enough)[:, 2].mean()
        test.assertTrue(torch.allclose(LossOneStep()(nerd, enough), expected, rtol=1e-5))


def test_other_frame_duration_is_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Train a fitted model on a batch of another frame duration and expect an error that names both."""
    nerd = make_nerd("arm", device)
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    LossOneStep()(nerd, batch)
    slower = TrajectoryBatch(batch.frames, batch.episode_start, 10.0 * batch.frame_dt)
    with test.assertRaisesRegex(ValueError, rf"{batch.frame_dt} s.*{slower.frame_dt} s"):
        LossOneStep()(nerd, slower)


def test_unfitted_model_is_refused(test: unittest.TestCase, device: torch.device) -> None:
    """Train a model whose normalization was never fitted and expect an error that names the missing call."""
    fitted = make_nerd("arm", device)
    batch = make_batch(fitted, WORLD_COUNT, FRAME_COUNT, device=device)
    nerd = FlashNeRD(make_system("arm"), backbone=make_backbone(), readout_hidden_dim=16, frame_dt=batch.frame_dt)
    nerd.to(device)
    with test.assertRaisesRegex(ValueError, r"not fitted.*fit_normalization\(windows.loader\(batch_size\)\)"):
        LossOneStep()(nerd, batch)
    # The statistics of a fitted model count as fitted wherever they are loaded, also through a module that
    # holds the model. Those of a model that never was fitted do not.
    nerd.load_state_dict(fitted.state_dict())
    with torch.no_grad():
        test.assertTrue(torch.equal(LossOneStep()(nerd, batch), LossOneStep()(fitted, batch)))
    unfitted = FlashNeRD(make_system("arm"), backbone=make_backbone(), readout_hidden_dim=16)
    nerd.load_state_dict(unfitted.state_dict())
    test.assertFalse(nerd.fitted)
    holder, source = torch.nn.ModuleDict({"nerd": nerd}), torch.nn.ModuleDict({"nerd": fitted})
    holder.load_state_dict(source.state_dict())
    test.assertTrue(nerd.fitted)
    # A model is fitted once the statistics of its outputs hold samples, however they got there.
    unfitted.to(device)
    current, following = unfitted.training_frames(batch.frames)
    test.assertFalse(unfitted.fitted)
    unfitted.output_normalizer.update(unfitted.prediction.label(unfitted.system, current, following))
    test.assertTrue(unfitted.fitted)


def _gradient(nerd: NeRDBase) -> Tensor:
    """The gradient that ``backward`` left on the parameters of a model, as one vector."""
    gradients = [torch.zeros_like(p) if p.grad is None else p.grad for p in nerd.parameters()]
    return torch.cat([gradient.flatten() for gradient in gradients])


def _assert_compiled_matches_eager(
    test: unittest.TestCase, nerd: NeRDBase, batch: TrajectoryBatch, **settings: Any
) -> LossOneStep:
    """Compile a loss and require the value and the gradient of the eager loss under the same seed.

    First while float32 matrix products run in TensorFloat-32, as a training script sets it, where the first call
    checks itself. Then in IEEE float32, where compiled and eager code agree to rounding on every GPU.

    Returns:
        The compiled loss.
    """
    eager, compiled = LossOneStep(**settings), LossOneStep(**settings)
    compiled.compile(fullgraph=True)  # so that the compiler raises where it would give up and run the loss eagerly
    for setting, value_tolerance, gradient_tolerance in (
        ("high", TENSOR_FLOAT_VALUE_TOLERANCE, TENSOR_FLOAT_GRADIENT_TOLERANCE),
        ("highest", VALUE_TOLERANCE, COMPILED_GRADIENT_TOLERANCE),
    ):
        with process_precision(setting):
            for seed in range(2):
                found = []
                for loss in (eager, compiled):
                    torch.manual_seed(seed)
                    nerd.zero_grad()
                    value = loss(nerd, batch)
                    value.backward()
                    found.append((value.detach(), _gradient(nerd)))
                (expected, expected_gradient), (value, gradient) = found
                error = ((value - expected).abs() / expected.abs().clamp_min(1.0)).item()
                test.assertLessEqual(error, value_tolerance, f"{setting}, the loss differs by {error:.3g}")
                error = ((gradient - expected_gradient).norm() / expected_gradient.norm()).item()
                test.assertLessEqual(error, gradient_tolerance, f"{setting}, the gradient differs by {error:.3g}")
    return compiled


def test_compiled_loss_matches_eager(
    test: unittest.TestCase, device: torch.device, nerd_name: str, contact: str, error: str, noise: float
) -> None:
    """Compile the loss of a model that learns the change to the next frame and expect the eager loss and
    gradient."""
    nerd, batch = _nerd_and_its_windows(device, nerd_name, contact)
    _assert_compiled_matches_eager(test, nerd, batch, error=error, noise=noise)


def test_compiled_loss_of_recorded_labels(test: unittest.TestCase, device: torch.device) -> None:
    """Compile the loss of a free motion model, whose labels are recorded with the frame, and expect the eager
    loss and gradient, also for a call that tracks no gradient."""
    nerd = make_nerd("arm", device, prediction=PredictionFreeMotion())
    batch = make_batch(nerd, TENSOR_CORE_WINDOW_COUNT, FRAME_COUNT, device=device)
    compiled = _assert_compiled_matches_eager(test, nerd, batch)
    with torch.no_grad():
        expected = LossOneStep()(nerd, batch)
        for _ in range(2):
            test.assertTrue(torch.allclose(compiled(nerd, batch), expected, rtol=VALUE_TOLERANCE))


class LossScaled(LossOneStep):
    """The one step loss times ``scale``, which takes the loss of a model far below or far above one."""

    scale = 1.0

    def _window_loss(self, *arguments: Any) -> Tensor:
        return self.scale * super()._window_loss(*arguments)


def _compiler_off_by(offset: float) -> Any:
    """Replace ``torch.compile`` by a stand-in whose function returns ``offset`` more than the one it is given."""
    return mock.patch.object(torch, "compile", lambda function, **_: lambda *arguments: function(*arguments) + offset)


def test_compiled_loss_checks_itself(test: unittest.TestCase, device: torch.device) -> None:
    """Give the loss compilers that are off by half and by twice what its check accepts and expect the first
    call of each kind to raise for twice and to pass for half, after which the eager loss is evaluated no more.
    Expect no check under autocast and none when it is switched off, and a tolerance that grows with a loss
    far above one and stays for a loss far below it."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    eager = LossOneStep()(nerd, batch)
    eager.backward()
    expected, expected_gradient = eager.detach(), _gradient(nerd)

    def compiled_loss(offset: float, scale: float = 1.0, **options: Any) -> LossOneStep:
        loss = LossScaled()
        loss.scale = scale
        with _compiler_off_by(offset):
            loss.compile(**options)
        return loss

    tolerance = VALUE_TOLERANCE * max(1.0, expected.item())
    # A value that is off raises at every call until the loss is compiled anew, with gradients and without.
    loss = compiled_loss(2.0 * tolerance)
    for _ in range(2):
        with test.assertRaisesRegex(RuntimeError, r"compiled loss is.*eager loss.*compile\(check=False\)"):
            loss(nerd, batch)
        with test.assertRaisesRegex(RuntimeError, "compiled loss is"), torch.no_grad():
            loss(nerd, batch)

    # Compiled and eager code round at other places under autocast, so such a call is left for the next one.
    loss = compiled_loss(2.0 * tolerance)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        loss(nerd, batch)
    with test.assertRaisesRegex(RuntimeError, "compiled loss is"):
        loss(nerd, batch)
    # A loss compiled without the check returns what its compiler made of it.
    loss = compiled_loss(2.0 * tolerance, check=False)
    test.assertTrue(torch.equal(loss(nerd, batch), expected + 2.0 * tolerance))

    # Within the tolerance the loss returned is the compiled one, which the caller differentiates.
    offset = 0.5 * tolerance
    loss = compiled_loss(offset)
    window_loss, tracked = loss._window_loss, []

    def eager_loss(*arguments: Any) -> Tensor:
        tracked.append(torch.is_grad_enabled())
        return window_loss(*arguments)

    loss._window_loss = eager_loss
    for call in range(2):
        tracked.clear()
        nerd.zero_grad()
        value = loss(nerd, batch)
        value.backward()
        test.assertTrue(torch.equal(value, expected + offset))
        test.assertTrue(torch.allclose(_gradient(nerd), expected_gradient, rtol=1e-4, atol=1e-7))
        with torch.no_grad():
            test.assertTrue(torch.equal(loss(nerd, batch), expected + offset))
        # The stand-in evaluates the loss once a call. The first call of each kind evaluates it once more for
        # the check, without gradients, and no later call does.
        test.assertEqual(tracked, [True, False, False, False] if call == 0 else [True, False])
    # Compiling again brings the check back.
    with _compiler_off_by(2.0 * tolerance):
        loss.compile()
    with test.assertRaisesRegex(RuntimeError, "compiled loss is"):
        loss(nerd, batch)

    # In TensorFloat-32 the check accepts what the rounding of the products moves, and no more.
    if device.type == "cuda":
        with process_precision("high"):
            loose = TENSOR_FLOAT_VALUE_TOLERANCE * max(1.0, expected.item())
            compiled_loss(0.5 * loose)(nerd, batch)
            with test.assertRaisesRegex(RuntimeError, "compiled loss is"):
                compiled_loss(2.0 * loose)(nerd, batch)

    # A copy of a compiled loss computes with its own settings.
    squared = copy.deepcopy(compiled_loss(0.0))
    squared.error = ErrorKind.SQUARED
    test.assertTrue(torch.equal(squared(nerd, batch), LossOneStep(error="squared")(nerd, batch)))

    # The difference counts relative to the eager loss or to one, whichever is greater.
    for scale in (1e-3, 1e3):
        tolerance = VALUE_TOLERANCE * max(1.0, scale * expected.item())
        with torch.no_grad():
            compiled_loss(0.5 * tolerance, scale)(nerd, batch)
            with test.assertRaisesRegex(RuntimeError, "compiled loss is"):
                compiled_loss(2.0 * tolerance, scale)(nerd, batch)

    # Frames that are not finite fail the check as well, whatever the compiler did.
    frames = {**batch.frames, "joint_qd": torch.full_like(batch.frames["joint_qd"], torch.nan)}
    with test.assertRaisesRegex(RuntimeError, "not finite"):
        compiled_loss(0.0)(nerd, TrajectoryBatch(frames, batch.episode_start, batch.frame_dt))


def test_compiled_loss_trains(test: unittest.TestCase, device: torch.device) -> None:
    """Train a small model for a few steps with the eager loss and again with the compiled loss from the same
    seed, and expect the loss to fall by the same steps."""

    def train(compile: bool) -> Tensor:
        torch.manual_seed(0)
        system = make_system("arm")
        nerd = FlashNeRD(system, backbone=make_backbone(), readout_hidden_dim=32)
        windows = pendulum_windows(system, 128, 8, device="cpu")
        nerd.fit_normalization([windows])
        nerd.to(device)
        windows = windows.to(device)
        loss, optimizer = LossOneStep(noise=0.1), create_optimizer(nerd, lr=3e-3)
        if compile:
            loss.compile()
        losses = []
        for _ in range(TRAIN_STEP_COUNT):
            optimizer.zero_grad()
            error = loss(nerd, windows)
            error.backward()
            optimizer.step()
            losses.append(error.detach())
        return torch.stack(losses)

    eager, compiled = train(False), train(True)
    test.assertLess(compiled[-1], 0.5 * compiled[0])
    test.assertTrue(torch.allclose(compiled, eager, rtol=2e-3), f"{compiled.tolist()} and {eager.tolist()}")


def test_compiled_loss_under_autocast(test: unittest.TestCase, device: torch.device) -> None:
    """Call a compiled loss under autocast first, where it is not checked, and expect a float32 loss near the
    eager one with the same noise under a seed, then outside of it, where it is."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    batch = make_batch(nerd, WORLD_COUNT, FRAME_COUNT, device=device)
    eager, compiled = LossOneStep(noise=0.3), LossOneStep(noise=0.3)
    compiled.compile()
    torch.manual_seed(0)
    expected = eager(nerd, batch)
    torch.manual_seed(0)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        reduced = compiled(nerd, batch)
    test.assertEqual(reduced.dtype, torch.float32)
    test.assertTrue(torch.allclose(reduced, expected, rtol=5e-2))
    torch.manual_seed(0)
    test.assertTrue(torch.allclose(compiled(nerd, batch), expected, rtol=1e-4))


def test_compiled_loss_under_either_autocast_type(test: unittest.TestCase, device: torch.device) -> None:
    """Call one compiled loss under bfloat16 and then under float16 autocast, for a model whose blocks attend
    over one frame, and expect each call nearer to the eager loss of its own type than to that of the other,
    so that neither call computed in the type of the other."""
    nerd, batch = _nerd_and_its_windows(device, "flash_span_one")
    eager, compiled = LossOneStep(), LossOneStep()
    compiled.compile()
    expected, found = [], []
    with torch.no_grad():
        for dtype in (torch.bfloat16, torch.float16):
            with torch.autocast(device.type, dtype=dtype):
                expected.append(eager(nerd, batch))
                found.append(compiled(nerd, batch))
    test.assertNotEqual(*expected)
    for value, own, other in zip(found, expected, reversed(expected), strict=True):
        test.assertLess((value - own).abs(), (value - other).abs())


def _distributed_worker(rank: int, port: int, device_type: str, compiled: bool, reports: Any) -> None:
    """Train as process ``rank`` of :data:`RANK_COUNT` for three steps and report the gradient of the first."""
    try:
        torch.distributed.init_process_group(
            "gloo" if device_type == "cpu" else "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=RANK_COUNT,
        )
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
        nerd = make_nerd("arm", device)  # the same weights in every process
        # Other statistics in every process. The wrapper hands those of the first process to all of them.
        nerd.fit_normalization([make_batch(nerd, 16, FRAME_COUNT, device=device, seed=50 + rank)])
        wrapper = DistributedDataParallel(nerd)
        optimizer = create_optimizer(nerd, lr=1e-3)
        losses = (LossOneStep(), LossOneStep(), LossOneStep(noise=0.1))
        report = {}
        for step, loss in enumerate(losses):
            if compiled:
                loss.compile()
            batch = make_batch(nerd, 8, FRAME_COUNT, device=device, seed=100 + 10 * step + rank)  # its own data
            optimizer.zero_grad()
            loss(wrapper, batch).backward()
            if step == 0:
                report["gradient"] = _gradient(nerd).cpu().numpy()
                report["mean"] = nerd.output_normalizer.mean.cpu().numpy()
            optimizer.step()
        report["weights"] = torch.cat([p.detach().flatten() for p in nerd.parameters()]).cpu().numpy()
        reports.put((rank, report))
    except Exception:
        reports.put((rank, traceback.format_exc()))
    finally:
        reports.close()
        reports.join_thread()


def _train_distributed(test: unittest.TestCase, device: torch.device, *, compiled: bool) -> list[dict[str, Any]]:
    """Run :func:`_distributed_worker` in :data:`RANK_COUNT` processes and return their reports by rank."""
    context = multiprocessing.get_context("spawn")
    reports = context.Queue()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    processes = [
        context.Process(target=_distributed_worker, args=(rank, port, device.type, compiled, reports))
        for rank in range(RANK_COUNT)
    ]
    for process in processes:
        process.start()
    found: dict[int, Any] = {}
    try:
        # Stop at the first failure. The other processes then wait for one that is gone and are killed below.
        while len(found) < RANK_COUNT and not any(isinstance(report, str) for report in found.values()):
            rank, report = reports.get(timeout=DISTRIBUTED_TIMEOUT)
            found[rank] = report
    except queue.Empty:
        pass
    finally:
        failed = any(isinstance(report, str) for report in found.values()) or len(found) < RANK_COUNT
        for process in processes:
            process.join(timeout=0.0 if failed else 30.0)
            if process.is_alive():
                process.kill()
    for rank, report in found.items():
        if isinstance(report, str):
            test.fail(f"process {rank} failed\n{report}")
    if len(found) < RANK_COUNT:
        test.fail(f"only the processes {sorted(found)} of {RANK_COUNT} reported within {DISTRIBUTED_TIMEOUT:g} s")
    return [found[rank] for rank in range(RANK_COUNT)]


def _assert_one_gradient(
    test: unittest.TestCase, device: torch.device, reports: list[dict[str, Any]], tolerance: float
) -> None:
    """Require every process to hold the mean of the gradients that the processes would compute alone."""
    alone = []
    for rank in range(RANK_COUNT):
        nerd = make_nerd("arm", device)
        nerd.fit_normalization([make_batch(nerd, 16, FRAME_COUNT, device=device, seed=50)])  # of the first process
        LossOneStep()(nerd, make_batch(nerd, 8, FRAME_COUNT, device=device, seed=100 + rank)).backward()
        alone.append(_gradient(nerd))
        np.testing.assert_allclose(reports[rank]["mean"], nerd.output_normalizer.mean.cpu().numpy(), rtol=1e-6)
    expected = torch.stack(alone).mean(dim=0).cpu().numpy()
    test.assertGreater(np.linalg.norm(alone[0].cpu().numpy() - expected), 0.1 * np.linalg.norm(expected))
    for report in reports:
        error = np.linalg.norm(report["gradient"] - expected) / np.linalg.norm(expected)
        test.assertLess(error, tolerance)
        np.testing.assert_allclose(report["gradient"], reports[0]["gradient"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(report["weights"], reports[0]["weights"], rtol=1e-5, atol=1e-7)


def test_distributed_training(test: unittest.TestCase, device: torch.device) -> None:
    """Train in two processes through ``DistributedDataParallel`` and expect one gradient, the mean of both."""
    _assert_one_gradient(test, device, _train_distributed(test, device, compiled=False), tolerance=1e-5)


def test_distributed_training_compiled(test: unittest.TestCase, device: torch.device) -> None:
    """Train on two GPUs through ``DistributedDataParallel`` with a compiled loss and expect one gradient."""
    if torch.cuda.device_count() < RANK_COUNT:
        test.skipTest(f"the test trains on {RANK_COUNT} GPUs")
    reports = _train_distributed(test, device, compiled=True)
    _assert_one_gradient(test, device, reports, tolerance=COMPILED_GRADIENT_TOLERANCE)


class TestLoss(unittest.TestCase):
    def test_settings(self) -> None:
        """Create losses with valid and invalid settings and expect the invalid ones refused."""
        loss = LossOneStep()
        self.assertIsInstance(loss, LossBase)
        self.assertIs(loss.error, ErrorKind.ABSOLUTE)
        self.assertEqual(loss.noise, 0.0)
        with self.assertRaises(ValueError):
            LossOneStep(error="cubed")
        with self.assertRaisesRegex(ValueError, "negative"):
            LossOneStep(noise=-0.1)
        with self.assertRaises(TypeError):
            LossOneStep("squared")  # the settings are keyword only
        with self.assertRaises(TypeError):
            LossBase()


for function in (
    test_error_kinds,
    test_frames_that_count,
    test_noise,
    test_noise_recomputes_contact,
    test_noise_is_refused,
    test_labels_of_the_frame,
    test_single_frames_are_refused,
    test_short_windows_are_refused,
    test_other_frame_duration_is_refused,
    test_unfitted_model_is_refused,
):
    add_function_test(TestLoss, function.__name__, function)
add_function_test(
    TestLoss,
    "test_frames_that_count_by_history_one_frame",
    test_frames_that_count_by_history,
    backbone=lambda: BackboneMLP(width=16, depth=1, window=1),
)
add_function_test(
    TestLoss,
    "test_frames_that_count_by_history_unbounded",
    test_frames_that_count_by_history,
    backbone=BackboneLeaky,
)
add_function_test(TestLoss, "test_compiled_loss_checks_itself", test_compiled_loss_checks_itself)
add_function_test(TestLoss, "test_distributed_training", test_distributed_training, devices=[torch.device("cpu")])
add_function_test(
    TestLoss, "test_distributed_training_compiled", test_distributed_training_compiled, devices=get_cuda_test_devices()
)
for nerd_name, contact_name, error_kind, noise_scale in COMPILED_CASES:
    add_function_test(
        TestLoss,
        f"test_compiled_loss_matches_eager_{nerd_name}_{contact_name}",
        test_compiled_loss_matches_eager,
        devices=get_cuda_test_devices(),
        nerd_name=nerd_name,
        contact=contact_name,
        error=error_kind,
        noise=noise_scale,
    )
for function in (
    test_compiled_loss_of_recorded_labels,
    test_compiled_loss_trains,
    test_compiled_loss_under_autocast,
    test_compiled_loss_under_either_autocast_type,
):
    add_function_test(TestLoss, function.__name__, function, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
