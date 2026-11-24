# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conformance checks verifying streaming parity and contact recomputation contracts for extensions."""

from __future__ import annotations

import copy
from collections.abc import Callable

import newton
import torch
from torch import Tensor

from ..backbones.backbone import BackboneBase
from ..contacts.contact_input import ContactInputBase
from ..core.signal import Frames
from ..models.nerd import NeRDBase
from ..systems.articulation import SystemArticulation
from ..systems.articulation_binding import Pattern


@torch.no_grad()
def check_backbone_streaming(
    backbone: BackboneBase,
    *,
    world_count: int = 4,
    frame_count: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    autocast: torch.dtype | None = None,
    tolerance: float | None = None,
) -> None:
    """Verify that ``backbone.step`` matches ``backbone.forward`` across uninterrupted runs, resets, and NaN isolation.

    Perturbs a copy of ``backbone`` so zero-initialized parameters participate, then checks: (1) uninterrupted
    streaming matches ``forward``, (2) mid-episode resets (even from NaN-poisoned history) match fresh ``forward``
    slices, (3) a world fed NaNs does not affect other worlds, and (4) history buffers are mutated in place with
    stable shapes and dtypes.

    Args:
        backbone: Backbone instance to validate (not modified).
        world_count: Parallel worlds (``>= 2``).
        frame_count: Sequence length (defaults to ``2 * receptive_field + 3`` or ``16`` if unbounded).
        device: Target device.
        dtype: Parameter/input/history dtype.
        autocast: Optional ``torch.autocast`` dtype.
        tolerance: Relative tolerance (floor 1.0); defaults to ``1e-9`` (FP64), ``1e-4`` (FP32), or ``5e-2``
            (autocast).
    """
    device = torch.device(device)
    if tolerance is None:
        tolerance = 5e-2 if autocast is not None else 1e-9 if dtype == torch.float64 else 1e-4
    if frame_count is None:
        frame_count = 16 if backbone.receptive_field is None else 2 * backbone.receptive_field + 3
    if world_count < 2 or frame_count < world_count:
        raise ValueError(f"need at least 2 worlds and a frame per world, got {world_count=} and {frame_count=}")

    generator = torch.Generator().manual_seed(0)
    backbone = copy.deepcopy(backbone).to(device=device, dtype=dtype).eval()
    for parameter in backbone.parameters():
        noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
        parameter.add_(0.1 * noise.to(parameter))
    x = torch.randn(world_count, frame_count, backbone.width, generator=generator, dtype=torch.float64)
    x = x.to(device=device, dtype=dtype)
    # Stagger episode resets across worlds (world 0 starts at frame 0 and runs uninterrupted).
    starts = torch.arange(world_count, device=device) * frame_count // world_count

    with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
        whole = backbone(x)
        history = backbone.history(world_count, device=device, dtype=dtype)
        layout = {name: (value, value.shape, value.dtype) for name, value in history.items()}
        for name, value in history.items():
            assert value.shape[0] == world_count, f"history {name!r} is not [world_count, ...]"
            assert value.dtype == dtype or not value.is_floating_point(), f"history {name!r} is not {dtype}"
            assert not value.any(), f"history {name!r} does not start at zero"

        def step(frame: int) -> Tensor:
            return backbone.step(x[:, frame], history)

        stepped = _stream(backbone, step, frame_count, history)
        _assert_close(stepped, whole, tolerance, "an uninterrupted sequence")

        stepped = _stream(backbone, step, frame_count, history, starts=starts)
        for world, start in enumerate(starts.tolist()):
            fresh = backbone(x[world : world + 1, start:])
            what = f"world {world} after its reset at frame {start}"
            _assert_close(stepped[world : world + 1, start:], fresh, tolerance, what)

        x[-1] = torch.nan
        backbone.reset_history(history, torch.ones(world_count, dtype=torch.bool, device=device))
        stepped = _stream(backbone, step, frame_count, history)
        _assert_close(stepped[:-1], whole[:-1], tolerance, "the worlds beside one fed NaN")

    for name, (value, shape, value_dtype) in layout.items():
        assert history[name] is value, f"history {name!r} was replaced instead of updated in place"
        assert value.shape == shape and value.dtype == value_dtype, f"history {name!r} changed shape or dtype"


@torch.no_grad()
def check_nerd_streaming(nerd: NeRDBase, frames: Frames, *, tolerance: float = 1e-4) -> None:
    """Verify that ``nerd.predict`` streaming matches ``nerd.forward`` over uninterrupted runs and mid-episode resets.

    Checks normalized streaming outputs against ``forward(frames)``, staggered resets with NaN-poisoned stale
    history, and in-place history buffer stability.

    Args:
        nerd: Model to validate (not modified).
        frames: Windowed signals ``[world_count, frame_count, *shape]`` on ``nerd.device``.
        tolerance: Relative tolerance on normalized outputs (floor 1.0).
    """
    world_count, frame_count = next(iter(frames.values())).shape[:2]
    whole = nerd(frames)
    history = nerd.history(world_count)
    layout = {name: (value, value.shape, value.dtype) for name, value in history.items()}

    def step(frame: int) -> Tensor:
        current = {name: value[:, frame] for name, value in frames.items()}
        return nerd.output_normalizer.normalize(nerd.predict(current, history))

    stepped = _stream(nerd, step, frame_count, history)
    _assert_close(stepped, whole, tolerance, "an uninterrupted sequence")

    # Stagger episode resets across worlds (world 0 starts at frame 0).
    starts = torch.arange(world_count, device=whole.device) * frame_count // world_count
    stepped = _stream(nerd, step, frame_count, history, starts=starts)
    for start in starts.unique().tolist():
        worlds = starts == start
        fresh = nerd({name: value[worlds, start:] for name, value in frames.items()})
        _assert_close(stepped[worlds, start:], fresh, tolerance, f"the worlds reset at frame {start}")

    for name, (value, shape, dtype) in layout.items():
        assert history[name] is value, f"history {name!r} was replaced instead of updated in place"
        assert value.shape == shape and value.dtype == dtype, f"history {name!r} changed shape or dtype"


def _stream(
    module: BackboneBase | NeRDBase,
    step: Callable[[int], Tensor],
    frame_count: int,
    history: dict[str, Tensor],
    starts: Tensor | None = None,
) -> Tensor:
    """Run ``step(frame)`` for ``0 .. frame_count - 1``, poisoning and resetting history on frames in ``starts``."""
    outputs = []
    for frame in range(frame_count):
        if starts is not None:
            mask = starts == frame
            for value in history.values():
                if value.is_floating_point():
                    value[mask] = torch.nan
            module.reset_history(history, mask)
        outputs.append(step(frame))
    return torch.stack(outputs, dim=1)


def _assert_close(actual: Tensor, expected: Tensor, tolerance: float, what: str) -> None:
    assert actual.shape == expected.shape, (
        f"stepping {what} gives {tuple(actual.shape)}, forward {tuple(expected.shape)}"
    )
    scale = max(1.0, expected.abs().max().item())
    error = (actual.double() - expected.double()).abs().max().item() / scale
    # Written with <= so NaN fails the assertion.
    assert error <= tolerance, f"stepping {what} differs from forward by {error:.3g}, more than {tolerance:.3g}"


@torch.no_grad()
def check_contact_input_recompute(
    contact_input: ContactInputBase,
    system: SystemArticulation,
    model: newton.Model,
    *,
    articulation: Pattern = "*",
    noise: float = 0.1,
    tolerance: float = 1e-4,
) -> None:
    """Verify that a recomputable contact input's ``evaluate()`` matches its live Warp sensor on ``model``.

    Tests two perturbed poses for signal schema (keys, shapes, dtypes, device), agreement with ``sensor.update()``,
    windowed vs. single-frame consistency, and non-aliasing across consecutive calls.

    Args:
        contact_input: Recomputable contact input to validate.
        system: Robot system specification (not modified).
        model: Newton model containing the robot and sensed geometry.
        articulation: Articulation label pattern in ``model``.
        noise: Perturbation scale in ``system.increment`` units.
        tolerance: Max allowed absolute difference per element.
    """
    name = type(contact_input).__name__
    assert contact_input.recomputable, f"{name} is not recomputable"
    binding = system.bind(model, articulation)
    sensor = contact_input.sensor(model, binding)
    state = model.state()
    start = {signal: value.clone() for signal, value in binding.read_state(state).items()}
    device = start["joint_q"].device
    system = copy.deepcopy(system).to(device)

    generator = torch.Generator().manual_seed(0)
    frames, sensed = [], []
    for _ in range(2):
        increments = noise * torch.randn(binding.world_count, system.increment_count, generator=generator)
        frames.append(system.perturb(start, increments.to(device)))
        binding.write_state(state, frames[-1])
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        # Recomputable inputs depend only on joint state, so contacts=None.
        sensed.append({signal: value.clone() for signal, value in sensor.update(state, None).items()})
    evaluated = [contact_input.evaluate(system, frame) for frame in frames]
    window = contact_input.evaluate(
        system, {signal: torch.stack([frame[signal] for frame in frames], dim=1) for signal in start}
    )

    for signal, spec in contact_input.signals.items():
        for index in range(2):
            assert signal in evaluated[index], f"{name}.evaluate does not return its signal {signal!r}"
            value = evaluated[index][signal]
            expected = (binding.world_count, *spec.shape)
            assert value.shape == expected, f"{signal!r} is {tuple(value.shape)} from evaluate, declared {expected}"
            assert value.dtype == spec.torch_dtype, f"{signal!r} is {value.dtype} from evaluate, declared {spec.dtype}"
            assert value.device == device, f"{signal!r} is on {value.device} from evaluate, the frames on {device}"
            for what, other in (("the sensor", sensed[index][signal]), ("a window", window[signal][:, index])):
                error = (value.double() - other.double()).abs().max().item()
                # Written with <= so NaN fails the assertion.
                assert error <= tolerance, (
                    f"{signal!r} from evaluate differs from {what} by {error:.3g}, more than {tolerance:.3g}"
                )
