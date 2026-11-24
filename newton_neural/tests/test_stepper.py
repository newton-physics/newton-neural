# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``NeRDStepper``: precisions, first-step verification, history clone/export/reset, and TF32."""

from __future__ import annotations

import itertools
import threading
import unittest
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

from newton_neural import Frames
from newton_neural.backbones import BackboneMLP
from newton_neural.models import NeRDBase, NeRDStepper, Precision
from newton_neural.predictions import PredictionFreeMotion
from newton_neural.tests import fixtures
from newton_neural.tests.extensions import ContactInputRootHeight, MinimalNeRD
from newton_neural.tests.helpers import (
    CONTACT_INPUTS,
    NERDS,
    WORLD_COUNT,
    frame_of,
    make_frames,
    make_nerd,
    make_system,
    process_precision,
    process_settings,
)
from newton_neural.tests.unittest_utils import add_function_test, get_cuda_test_devices

TOLERANCES = {Precision.FP32: 1e-4, Precision.TF32: 2e-2, Precision.FP16: 2e-2, Precision.BF16: 1e-1}
"""Largest accepted difference from the eager model per precision, relative to the largest normalized output."""

TENSOR_CORE_WORLD_COUNT = 64
"""Worlds with which TensorFloat-32 changes the predictions of the small models of the tests. cuBLAS computes
the products of a few rows in IEEE arithmetic whatever is allowed."""


def _assert_step_close(
    test: unittest.TestCase, nerd: NeRDBase, frames: Frames, actual: Frames, expected: Frames
) -> None:
    """Require a step to match the eager model within the tolerance of the model's precision.

    The two next states are compared as the normalized network outputs they came from.
    """
    reference = nerd.labels(frames, expected)
    scale = max(1.0, reference.abs().max().item())
    error = (nerd.labels(frames, actual) - reference).abs().max().item() / scale
    test.assertLessEqual(error, TOLERANCES[nerd.precision], f"the step differs from the eager model by {error:.3g}")


def _reference_step(nerd: NeRDBase, frames: Frames, history: dict[str, Tensor]) -> Frames:
    with torch.no_grad():
        return nerd.step(frames, history)


def _eager_prediction(nerd: NeRDBase, frames: Frames) -> Tensor:
    """The normalized prediction of the eager model for the first frame of an episode."""
    with torch.no_grad():
        return nerd.predict(frames, nerd.history(len(frames["joint_q"]))) / nerd.output_normalizer.std


def _first_prediction(nerd: NeRDBase, frames: Frames, precision: Precision, compile: bool) -> Tensor:
    """The normalized prediction of a new stepper for the first frame of an episode."""
    stepper = NeRDStepper(nerd, len(frames["joint_q"]), precision=precision, compile=compile)
    return stepper.predict(frames) / nerd.output_normalizer.std


def _ieee_and_tensor_float(test: unittest.TestCase, nerd: NeRDBase, frames: Frames) -> tuple[Tensor, Tensor]:
    """The eager predictions in IEEE arithmetic and with TensorFloat-32, for a test that tells the two apart.

    Skips the test on a device where the two do not differ, as where cuBLAS computes products of this size in
    IEEE arithmetic whatever is allowed.
    """
    with process_precision("highest"):
        ieee = _eager_prediction(nerd, frames)
    with process_precision("high"):
        tensor_float = _eager_prediction(nerd, frames)
    if (tensor_float - ieee).abs().max() < 1e-5:
        test.skipTest("TensorFloat-32 does not change the matrix products of this model on this device")
    return ieee, tensor_float


def _watch_steps(stepper: NeRDStepper, read: Callable[[], object]) -> list:
    """Make every later step of ``stepper`` call ``read`` inside the step, and return the list of what it read."""
    deployed, seen = stepper._advance, []

    def advance(frames: Frames, history: dict[str, Tensor]) -> tuple[Tensor, Frames]:
        seen.append(read())
        return deployed(frames, history)

    stepper._advance = advance
    return seen


def test_matches_model(
    test: unittest.TestCase, device: torch.device, robot: str, contact: bool, precision: Precision, compile: bool
) -> None:
    """Step a stepper and the eager model through frames with masked and full resets and expect the same states."""
    nerd = make_nerd(robot, device, contact_input=ContactInputRootHeight() if contact else None)
    nerd.precision = precision
    stepper = NeRDStepper(nerd, WORLD_COUNT, compile=compile)
    window = make_frames(nerd, WORLD_COUNT, 9, device=device)
    history = nerd.history(WORLD_COUNT)
    mask = torch.tensor([True, False, True, False], device=device)
    everywhere = torch.ones(WORLD_COUNT, dtype=torch.bool, device=device)
    for index in range(9):
        if index == 3:
            stepper.reset(world_mask=mask)
            nerd.reset_history(history, mask)
        if index == 6:
            stepper.reset()
            nerd.reset_history(history, everywhere)
        frames = frame_of(window, index)
        actual = stepper.step(frames)
        _assert_step_close(test, nerd, frames, actual, _reference_step(nerd, frames, history))
    test.assertEqual(list(actual), ["joint_q", "joint_qd"])
    test.assertTrue(all(value.dtype == torch.float32 for value in actual.values()))
    test.assertEqual(nerd.network.embed.weight.dtype, torch.float32)  # a reduced precision works on a copy


def test_predict(test: unittest.TestCase, device: torch.device, free_motion: bool) -> None:
    """Predict frame by frame, for a model that integrates and for one that does not, and expect the eager
    model's prediction, one row per world, with the history advanced by every call."""
    nerd = make_nerd("arm", device, prediction=PredictionFreeMotion() if free_motion else None)
    stepper = NeRDStepper(nerd, WORLD_COUNT)
    window = make_frames(nerd, WORLD_COUNT, 4, device=device)
    history = nerd.history(WORLD_COUNT)
    scale = nerd.output_normalizer.std
    for index in range(4):
        frames = frame_of(window, index)
        actual = stepper.predict(frames)
        with torch.no_grad():
            expected = nerd.predict(frames, history)
        test.assertIsInstance(actual, Tensor)
        test.assertEqual(actual.shape, (WORLD_COUNT, nerd.prediction.output_count(nerd.system)))
        error = ((actual - expected) / scale).abs().max().item() / max(1.0, (expected / scale).abs().max().item())
        test.assertLessEqual(error, TOLERANCES[nerd.precision], f"frame {index} differs by {error:.3g}")
    test.assertEqual(stepper.export_history()["step"].tolist(), [4] * WORLD_COUNT)


def test_step_needs_integration(test: unittest.TestCase, device: torch.device) -> None:
    """Step a free motion model and expect the error of the eager model's step, with the history untouched."""
    nerd = make_nerd("arm", device, prediction=PredictionFreeMotion())
    stepper = NeRDStepper(nerd, WORLD_COUNT, compile=False)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    with test.assertRaises(RuntimeError) as eager:
        nerd.step(frames, nerd.history(WORLD_COUNT))
    with test.assertRaises(RuntimeError) as deployed:
        stepper.step(frames)
    test.assertEqual(str(deployed.exception), str(eager.exception))
    test.assertIn("PredictionFreeMotion does not give the next state. Call predict", str(deployed.exception))
    test.assertEqual(stepper.export_history()["step"].tolist(), [0] * WORLD_COUNT)


def test_steps_newton_state(test: unittest.TestCase, device: torch.device) -> None:
    """Step from the strided views a binding reads from a Newton state, write the result back and expect parity."""
    nerd = make_nerd("arm", device)
    model = fixtures.arm_with_box_model(WORLD_COUNT, device)
    binding = nerd.system.bind(model, "arm")
    state, control = model.state(), model.control()
    start = make_frames(nerd, WORLD_COUNT, device=device)
    binding.write_state(state, start)
    binding.write_control(control, start)
    stepper = NeRDStepper(nerd, WORLD_COUNT)
    history = nerd.history(WORLD_COUNT)
    for _ in range(3):
        frames = {**binding.read_state(state), **binding.read_control(control)}
        test.assertFalse(frames["joint_q"].is_contiguous())
        expected = _reference_step(nerd, frames, history)
        following = stepper.step(frames)
        _assert_step_close(test, nerd, frames, following, expected)
        binding.write_state(state, following)
        test.assertTrue(torch.equal(binding.read_state(state)["joint_q"], following["joint_q"]))


def test_clone(test: unittest.TestCase, device: torch.device) -> None:
    """Clone a stepper mid sequence, branch the two and expect each to follow its own frames."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    stepper = NeRDStepper(nerd, WORLD_COUNT)
    window = make_frames(nerd, WORLD_COUNT, 6, device=device)
    branch = make_frames(nerd, WORLD_COUNT, 6, device=device, seed=1)
    histories = nerd.history(WORLD_COUNT), nerd.history(WORLD_COUNT)
    for index in range(3):
        stepper.step(frame_of(window, index))
        for history in histories:
            _reference_step(nerd, frame_of(window, index), history)
    clone = stepper.clone()
    test.assertIs(clone.nerd, nerd)
    for index in range(3, 6):
        for frames, runner, history in zip((window, branch), (stepper, clone), histories, strict=True):
            frame = frame_of(frames, index)
            _assert_step_close(test, nerd, frame, runner.step(frame), _reference_step(nerd, frame, history))


def test_history_export_and_import(test: unittest.TestCase, device: torch.device) -> None:
    """Export the history, step on, import it back and expect the same steps again from the same frames."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    stepper = NeRDStepper(nerd, WORLD_COUNT)
    window = make_frames(nerd, WORLD_COUNT, 6, device=device)
    for index in range(3):
        stepper.step(frame_of(window, index))
    saved = stepper.export_history()
    first = [stepper.step(frame_of(window, index)) for index in range(3, 6)]
    test.assertEqual(saved["step"].tolist(), [3] * WORLD_COUNT)  # the export is a copy, later steps leave it
    stepper.import_history(saved)
    for index, expected in zip(range(3, 6), first, strict=True):
        actual = stepper.step(frame_of(window, index))
        for name, value in expected.items():
            test.assertTrue(torch.equal(actual[name], value))
    test.assertEqual(saved["step"].tolist(), [3] * WORLD_COUNT)  # the import copied, the stepper owns its history

    other = NeRDStepper(nerd, WORLD_COUNT + 1, compile=False)
    with test.assertRaisesRegex(ValueError, "history of this stepper"):
        other.import_history(saved)
    with test.assertRaisesRegex(ValueError, "history of this stepper"):
        stepper.import_history({"step": saved["step"]})


def test_inference_mode(test: unittest.TestCase, device: torch.device) -> None:
    """Create and step a stepper inside inference mode, then use it outside, and expect the eager model's steps."""
    nerd = make_nerd("arm", device)
    nerd.precision = Precision.BF16  # the copy of the weights is then made inside inference mode too
    window = make_frames(nerd, WORLD_COUNT, 4, device=device)
    history = nerd.history(WORLD_COUNT)
    with torch.inference_mode():
        stepper = NeRDStepper(nerd, WORLD_COUNT)
        inside = [stepper.step(frame_of(window, index)) for index in range(2)]
    outside = [stepper.step(frame_of(window, index)) for index in range(2, 4)]
    test.assertFalse(any(value.is_inference() or value.requires_grad for value in outside[-1].values()))
    for index, actual in enumerate(inside + outside):
        frames = frame_of(window, index)
        _assert_step_close(test, nerd, frames, actual, _reference_step(nerd, frames, history))
    saved = stepper.export_history()
    stepper.reset(torch.tensor([True, False, False, False], device=device))
    test.assertEqual(stepper.export_history()["step"].tolist(), [0, 4, 4, 4])
    stepper.import_history(saved)
    test.assertEqual(stepper.export_history()["step"].tolist(), [4] * WORLD_COUNT)


def test_device_and_precision(test: unittest.TestCase, device: torch.device) -> None:
    """Create a stepper for a model and expect steps on the device of the model, in the precision asked for."""
    nerd = make_nerd("arm", device)
    stepper = NeRDStepper(nerd=nerd, world_count=WORLD_COUNT, precision="bf16")
    test.assertIs(stepper.nerd, nerd)
    test.assertEqual(stepper.device, nerd.device)
    test.assertIs(stepper.precision, Precision.BF16)
    following = stepper.step(make_frames(nerd, WORLD_COUNT, device=device))
    test.assertTrue(all(value.device.type == device.type for value in following.values()))
    with test.assertRaises(ValueError):
        NeRDStepper(nerd, WORLD_COUNT, precision="fp8")
    nerd.precision = "fp16"  # the precision of the model may be set as a string too
    test.assertIs(NeRDStepper(nerd, WORLD_COUNT).precision, Precision.FP16)


def test_first_step_check(test: unittest.TestCase, device: torch.device) -> None:
    """Make the deployed step disagree with the model and expect the first step to raise a RuntimeError."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)

    stepper = NeRDStepper(nerd, WORLD_COUNT, precision=Precision.FP32)
    deployed = stepper._advance

    def wrong_prediction(frames: Frames, history: dict[str, Tensor]) -> tuple[Tensor, Frames]:
        prediction, following = deployed(frames, history)
        return prediction + 0.01 * nerd.output_normalizer.std, following

    stepper._advance = wrong_prediction
    with test.assertRaisesRegex(RuntimeError, "differs from its eager step"):
        stepper.step(frames)

    def wrong_integration(frames: Frames, history: dict[str, Tensor]) -> tuple[Tensor, Frames]:
        prediction, following = deployed(frames, history)
        return prediction, {**following, "joint_qd": following["joint_qd"] + 0.01}

    stepper._advance = wrong_integration
    with test.assertRaisesRegex(RuntimeError, "integrates joint_qd differently"):
        stepper.step(frames)

    stepper._advance = wrong_prediction
    with test.assertRaisesRegex(RuntimeError, "differs from its eager step"):
        stepper.predict(frames)  # a prediction is checked like a step

    stepper._advance = deployed
    stepper.reset()
    stepper.predict(frames)  # the untouched step passes its check, and later calls are not checked again
    stepper._advance = wrong_prediction
    stepper.step(frames)
    stepper.predict(frames)


def test_first_step_check_of_inputs_and_weights(test: unittest.TestCase, device: torch.device) -> None:
    """Step from non-finite frames, and after changing the model under a reduced precision, and expect errors."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    broken = {**frames, "joint_q": torch.full_like(frames["joint_q"], torch.nan)}
    with test.assertRaisesRegex(RuntimeError, "not finite"):
        NeRDStepper(nerd, WORLD_COUNT, compile=False).step(broken)

    stepper = NeRDStepper(nerd, WORLD_COUNT, precision=Precision.BF16, compile=False)
    with torch.no_grad():
        nerd.network.readout_out.bias.add_(1.0)
    with test.assertRaisesRegex(RuntimeError, "model changed"):
        stepper.step(frames)


def test_first_step_inside_autocast(test: unittest.TestCase, device: torch.device) -> None:
    """Take the first step of ``fp32`` inside an autocast region of the caller and expect the eager model's
    float32 prediction, which the check computes with autocast off."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    expected = _eager_prediction(nerd, frames)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        actual = _first_prediction(nerd, frames, Precision.FP32, compile=False)
    test.assertTrue(torch.equal(actual, expected))


def test_first_step_check_by_precision(test: unittest.TestCase, device: torch.device, precision: Precision) -> None:
    """Shift a deployed prediction by half and by twice the tolerance of the precision its stepper names and
    expect the first step to pass and to raise, whatever the process allows."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    with process_precision("highest"):  # the eager model in IEEE float32, so the shift is all that differs
        ieee = _eager_prediction(nerd, frames)
    scale = ieee.abs().max().clamp_min(1.0)

    def shifted_by(tolerances: float) -> NeRDStepper:
        stepper = NeRDStepper(nerd, WORLD_COUNT, precision=precision, compile=False)
        shift = tolerances * TOLERANCES[stepper.precision] * scale
        prediction = (ieee + shift) * nerd.output_normalizer.std
        stepper._advance = lambda frames, history: (prediction, nerd.system.integrate(frames, prediction))
        return stepper

    for setting in ("highest", "high"):
        with process_precision(setting):
            shifted_by(0.5).step(frames)
            with test.assertRaisesRegex(RuntimeError, "differs from its eager step in IEEE float32"):
                shifted_by(2.0).step(frames)


def test_default_precision(test: unittest.TestCase, device: torch.device) -> None:
    """Compose models of two families and expect TensorFloat-32, which a stepper takes over compiled on CUDA
    and names IEEE float32 on a device without it."""
    on_cuda = device.type == "cuda"
    test.assertEqual([precision.value for precision in Precision], ["fp32", "tf32", "fp16", "bf16"])
    for nerd in (make_nerd("arm", device), MinimalNeRD(make_system("arm")).to(device)):
        test.assertIs(nerd.precision, Precision.TF32)
        stepper = NeRDStepper(nerd, WORLD_COUNT)
        test.assertIs(stepper.precision, Precision.TF32 if on_cuda else Precision.FP32)
        test.assertIs(stepper.compiled, on_cuda)


def test_repr(test: unittest.TestCase, device: torch.device) -> None:
    """Expect the repr of a stepper to name the family, the worlds, the device, the precision and compilation."""
    stepper = NeRDStepper(make_nerd("arm", device), WORLD_COUNT, precision="fp16", compile=False)
    expected = f"NeRDStepper(FlashNeRD, world_count=4, device='{device}', precision='fp16', compiled=False)"
    test.assertEqual(repr(stepper), expected)
    stepper = NeRDStepper(MinimalNeRD(make_system("arm")).to(device), 2, compile=True)
    precision = "tf32" if device.type == "cuda" else "fp32"  # the default of a model, as the device runs it
    expected = f"NeRDStepper(MinimalNeRD, world_count=2, device='{device}', precision='{precision}', compiled=True)"
    test.assertEqual(repr(stepper), expected)


def test_fp32_is_strict(test: unittest.TestCase, device: torch.device, compile: bool) -> None:
    """Step in a process that allows TensorFloat-32 and expect IEEE products in single precision alone."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, TENSOR_CORE_WORLD_COUNT, device=device)
    ieee, tensor_float = _ieee_and_tensor_float(test, nerd, frames)
    with process_precision("high"):
        strict = _first_prediction(nerd, frames, Precision.FP32, compile)
        default = _first_prediction(nerd, frames, Precision.TF32, compile)
    rounding = (tensor_float - ieee).abs().max().item()
    if compile:
        # A compiled step orders its sums in another way than the eager model, which is far less than the rounding.
        test.assertLess((strict - ieee).abs().max().item(), 0.02 * rounding)
        test.assertGreater((default - ieee).abs().max().item(), 0.1 * rounding)
    else:
        test.assertTrue(torch.equal(strict, ieee))
        test.assertTrue(torch.equal(default, tensor_float))


def test_process_precision(test: unittest.TestCase, device: torch.device, process: str) -> None:
    """Step in a process whose float32 matrix products were set in one of PyTorch's ways and expect
    TensorFloat-32 inside every step but those of ``fp32``, and the process as it was after every step, also
    after one that raises."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    broken = {**frames, "joint_q": torch.full_like(frames["joint_q"], torch.nan)}
    switch = torch.backends.cuda.matmul
    generator = torch.Generator(device=device).manual_seed(0)
    a, b = (torch.randn(512, 512, device=device, generator=generator) for _ in range(2))

    def failing(frames: Frames, history: dict[str, Tensor]) -> tuple[Tensor, Frames]:
        raise RuntimeError("the step fails")

    with process_precision(process):
        found, product = process_settings(), a @ b
        for precision in Precision:
            stepper = NeRDStepper(nerd, WORLD_COUNT, precision=precision)
            tensor_float = _watch_steps(stepper, lambda: switch.fp32_precision == "tf32")
            for _ in range(2):  # the first step with its check, and a later one
                stepper.step(frames)
                test.assertEqual(process_settings(), found)
            test.assertEqual(tensor_float, [precision is not Precision.FP32] * 2)
            with test.assertRaisesRegex(RuntimeError, "not finite"):  # raised by the check of the first step
                NeRDStepper(nerd, WORLD_COUNT, precision=precision, compile=False).step(broken)
            test.assertEqual(process_settings(), found)
            stepper._advance = failing
            with test.assertRaisesRegex(RuntimeError, "the step fails"):  # raised inside the deployed step
                stepper.step(frames)
            test.assertEqual(process_settings(), found)
        # The products of the caller are computed as they were before.
        test.assertTrue(torch.equal(a @ b, product))
        # A switch that inherited its value inherits again, and one with a value of its own keeps it.
        torch.backends.fp32_precision = "ieee"
        test.assertEqual(switch.fp32_precision == "ieee", process in ("highest", "global_tf32"))


def test_steps_take_turns(test: unittest.TestCase, device: torch.device) -> None:
    """Hold a step open and expect a step of the other arithmetic in a second thread to wait until it has ended,
    so that neither computes in the arithmetic of the other nor puts back what the other set. A step of
    ``fp32`` and one of ``tf32`` are held in turn, since one of the two writes the switch in any process."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    found = process_settings()
    for names in (("fp32", "tf32"), ("tf32", "fp32")):
        steppers = [NeRDStepper(nerd, WORLD_COUNT, precision=name, compile=False) for name in names]
        inside, leave = threading.Event(), threading.Event()

        def held(inside: threading.Event = inside, leave: threading.Event = leave) -> bool:
            inside.set()
            leave.wait()
            return torch.backends.cuda.matmul.fp32_precision == "tf32"

        tensor_float = _watch_steps(steppers[0], held)
        first, second = (threading.Thread(target=stepper.predict, args=(frames,)) for stepper in steppers)
        first.start()
        test.assertTrue(inside.wait(timeout=60))
        second.start()
        second.join(timeout=0.5)
        waits = second.is_alive()
        leave.set()
        first.join()
        second.join()
        test.assertTrue(waits)
        test.assertEqual(tensor_float, [names[0] == "tf32"])
        test.assertEqual(process_settings(), found)


def test_cpu_sets_nothing(test: unittest.TestCase, device: torch.device) -> None:
    """Step on the CPU in every precision and expect the settings of the process untouched inside every step."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, WORLD_COUNT, device=device)
    found = process_settings()
    for precision in Precision:
        stepper = NeRDStepper(nerd, WORLD_COUNT, precision=precision, compile=False)
        inside = _watch_steps(stepper, process_settings)
        stepper.step(frames)
        test.assertEqual(inside, [found])


def test_graph_keeps_its_precision(test: unittest.TestCase, device: torch.device, precision: Precision) -> None:
    """Record a step into a CUDA graph, replay it under either precision of the process and expect the arithmetic
    of the stepper's precision both times."""
    nerd = make_nerd("arm", device)
    frames = make_frames(nerd, TENSOR_CORE_WORLD_COUNT, device=device)
    ieee, tensor_float = _ieee_and_tensor_float(test, nerd, frames)
    rounding = (tensor_float - ieee).abs().max().item()
    stepper = NeRDStepper(nerd, TENSOR_CORE_WORLD_COUNT, precision=precision)
    # Recorded while the process allows what the precision does not, and the other way round.
    with process_precision("high" if precision is Precision.FP32 else "highest"):
        stepper.predict(frames)  # the first step compiles and checks itself, which a graph cannot record
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prediction = stepper.predict(frames)
    replays = []
    for setting in ("highest", "high"):
        stepper.reset()
        with process_precision(setting):
            graph.replay()
        replays.append(prediction / nerd.output_normalizer.std)
    test.assertTrue(torch.equal(*replays))
    error = (replays[0] - ieee).abs().max().item()
    if precision is Precision.FP32:
        test.assertLess(error, 0.02 * rounding)
    else:
        test.assertGreater(error, 0.1 * rounding)


REDUCED_OPERATIONS = {
    "linear",
    *("silu", "relu", "mul", "add"),  # what the networks compute from the results of their linear layers
    *("cat", "split", "chunk", "view", "reshape", "expand", "__getitem__"),  # which only rearrange values
    *("float", "to", "__get__"),  # the conversions, and reads of a shape or a type
}
"""The operations of a deployed step that may read or return a 16 bit tensor. No sum, matrix product, softmax
or norm is among them, so attention scores, pooling and normalization are float32."""


class _Operations(TorchFunctionMode):
    """Collects the names of the torch operations that read or return a tensor of ``dtype`` while it is entered.

    An operation that autocast lowers is given float32 tensors, so only what it returns shows that it computed
    in ``dtype``.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype, self.names = dtype, set()

    def __torch_function__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        values = itertools.chain.from_iterable(
            value if isinstance(value, (tuple, list)) else (value,)
            for value in (*args, *(kwargs or {}).values(), result)
        )
        if any(isinstance(value, Tensor) and value.dtype == self.dtype for value in values):
            self.names.add(function.__name__)
        return result


def test_what_is_16_bit(test: unittest.TestCase, device: torch.device, precision: Precision) -> None:
    """Deploy models of both families, with every backbone and contact input, in a 16 bit precision and expect
    16 bit weights in the linear layers, no operation but those of :data:`REDUCED_OPERATIONS` to read or return
    a 16 bit tensor, and a float32 contact code, prediction, state and history."""
    reduced, single = torch.float16 if precision is Precision.FP16 else torch.bfloat16, {torch.float32}
    for name, contact in itertools.product(NERDS, CONTACT_INPUTS):
        with test.subTest(nerd=name, contact=contact):
            nerd = make_nerd("quadruped", device, contact_input=CONTACT_INPUTS[contact](), **NERDS[name])
            frames = make_frames(nerd, WORLD_COUNT, device=device)
            given = {name: value.clone() for name, value in frames.items()}
            encoder = () if nerd.contact_input is None else type(nerd.contact_input.encoder())
            seen: dict[str, set[torch.dtype]] = {"weights": set(), "code": set()}

            def watch(module: nn.Module, inputs: tuple, output: Tensor, encoder=encoder, seen=seen) -> None:
                if isinstance(module, nn.Linear):
                    seen["weights"].update(parameter.dtype for parameter in module.parameters())
                elif isinstance(module, encoder):
                    seen["code"].add(output.dtype)

            stepper = NeRDStepper(nerd, WORLD_COUNT, precision=precision, compile=False)
            stepper.predict(frames)  # the first step also runs the eager model in float32, to check itself
            with nn.modules.module.register_module_forward_hook(watch), _Operations(reduced) as operations:
                prediction = stepper.predict(frames)
                following = stepper.step(frames)

            test.assertEqual(seen["weights"], {reduced})
            test.assertLessEqual(operations.names, REDUCED_OPERATIONS)
            test.assertLessEqual(seen["code"], single)
            test.assertEqual({prediction.dtype, *(value.dtype for value in following.values())}, single)
            test.assertLessEqual({value.dtype for value in stepper.export_history().values()}, single | {torch.int64})
            for name, value in given.items():
                test.assertEqual(frames[name].dtype, value.dtype)
                test.assertTrue(torch.equal(frames[name], value))
            # The model keeps its weights. The stepper deploys a copy.
            test.assertEqual({parameter.dtype for parameter in nerd.parameters()}, single)


def test_many_steppers(test: unittest.TestCase, device: torch.device) -> None:
    """Compile steppers for nine world counts in one process and expect every one to step as the eager model."""
    nerd = make_nerd("arm", device, backbone=lambda: BackboneMLP(width=16, depth=1, window=2))
    for world_count in range(1, 10):
        stepper = NeRDStepper(nerd, world_count, compile=True)
        frames = make_frames(nerd, world_count, device=device)
        expected = _reference_step(nerd, frames, nerd.history(world_count))
        _assert_step_close(test, nerd, frames, stepper.step(frames), expected)


def test_cuda_graph(test: unittest.TestCase, device: torch.device, precision: Precision) -> None:
    """Record a compiled step in a CUDA graph, replay it frame by frame across a reset and expect the eager steps."""
    nerd = make_nerd("quadruped", device, contact_input=ContactInputRootHeight())
    nerd.precision = precision
    stepper = NeRDStepper(nerd, WORLD_COUNT)
    window = make_frames(nerd, WORLD_COUNT, 7, device=device)
    history = nerd.history(WORLD_COUNT)
    mask = torch.tensor([False, True, True, False], device=device)
    frames = {name: value.clone() for name, value in frame_of(window, 0).items()}
    stepper.step(frames)  # the first step compiles and checks itself, which a graph cannot record
    _reference_step(nerd, frames, history)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        following = stepper.step(frames)
    for index in range(1, 7):
        if index == 4:
            stepper.reset(mask)
            nerd.reset_history(history, mask)
        for name, value in frames.items():
            value.copy_(window[name][:, index])
        graph.replay()
        _assert_step_close(test, nerd, frames, following, _reference_step(nerd, frames, history))


class TestStepper(unittest.TestCase):
    """The stepper against the eager model, on every device."""


cuda_devices = get_cuda_test_devices()
for compiled in (False, True):
    for robot_name, with_contact in (("quadruped", True), ("arm", False)):
        for precision_kind in Precision:
            add_function_test(
                TestStepper,
                f"test_matches_model_{robot_name}_{precision_kind.value}_{'compiled' if compiled else 'eager'}",
                test_matches_model,
                devices=cuda_devices if compiled else None,
                robot=robot_name,
                contact=with_contact,
                precision=precision_kind,
                compile=compiled,
            )
for function in (
    test_step_needs_integration,
    test_steps_newton_state,
    test_clone,
    test_history_export_and_import,
    test_inference_mode,
    test_device_and_precision,
    test_first_step_check,
    test_first_step_check_of_inputs_and_weights,
    test_first_step_inside_autocast,
    test_default_precision,
    test_repr,
):
    add_function_test(TestStepper, function.__name__, function)
for precision_kind in Precision:
    add_function_test(
        TestStepper,
        f"test_first_step_check_by_precision_{precision_kind.value}",
        test_first_step_check_by_precision,
        precision=precision_kind,
    )
for precision_kind in (Precision.FP16, Precision.BF16):
    add_function_test(
        TestStepper, f"test_what_is_16_bit_{precision_kind.value}", test_what_is_16_bit, precision=precision_kind
    )
add_function_test(TestStepper, "test_cpu_sets_nothing", test_cpu_sets_nothing, devices=[torch.device("cpu")])
for compiled in (False, True):
    add_function_test(
        TestStepper,
        f"test_fp32_is_strict_{'compiled' if compiled else 'eager'}",
        test_fp32_is_strict,
        devices=cuda_devices,
        compile=compiled,
    )
for process_kind in ("highest", "high", "allow_tf32", "cuda_tf32", "global_tf32"):
    add_function_test(
        TestStepper,
        f"test_process_precision_{process_kind}",
        test_process_precision,
        devices=cuda_devices,
        process=process_kind,
    )
add_function_test(TestStepper, "test_steps_take_turns", test_steps_take_turns, devices=cuda_devices)
for precision_kind in (Precision.FP32, Precision.TF32):
    add_function_test(
        TestStepper,
        f"test_graph_keeps_its_precision_{precision_kind.value}",
        test_graph_keeps_its_precision,
        devices=cuda_devices,
        precision=precision_kind,
    )
for kind, without_integration in (("state_increment", False), ("free_motion", True)):
    add_function_test(TestStepper, f"test_predict_{kind}", test_predict, free_motion=without_integration)
add_function_test(TestStepper, "test_many_steppers", test_many_steppers, devices=cuda_devices)
for precision_kind in (Precision.FP32, Precision.TF32, Precision.FP16):
    add_function_test(
        TestStepper,
        f"test_cuda_graph_{precision_kind.value}",
        test_cuda_graph,
        devices=cuda_devices,
        precision=precision_kind,
    )


if __name__ == "__main__":
    unittest.main()
