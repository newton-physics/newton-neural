# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-step open-loop rollout error evaluation (:func:`rollout_error`, :class:`RolloutStatistics`)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.duration import same_duration
from ..core.precision import Precision
from ..core.signal import Frames, SignalRole
from ..data.windows import TrajectoryWindows
from ..models.nerd import NeRDBase
from ..models.stepper import NeRDStepper

_QUANTILES = (0.5, 0.9, 0.99)
"""Error quantiles reported by :class:`RolloutStatistics`."""


@dataclass(frozen=True)
class RolloutStatistics:
    """Aggregated open-loop rollout errors against ground truth at a single horizon.

    Per-signal stats aggregate ``system.error`` across all rollouts and coordinates in each signal's SI unit
    (floating-base orientation is one angle in radians).
    """

    mean: dict[str, float]
    """Mean absolute error per ``STATE`` signal."""

    p50: dict[str, float]
    """Median (50th percentile) absolute error per ``STATE`` signal."""

    p90: dict[str, float]
    """90th percentile absolute error per ``STATE`` signal."""

    p99: dict[str, float]
    """99th percentile absolute error per ``STATE`` signal."""

    nrmse: float
    """Prefix-averaged normalized RMSE over frames ``1 .. H`` scaled by ``nerd.output_normalizer.std``."""


@torch.no_grad()
def rollout_error(
    nerd: NeRDBase,
    windows: TrajectoryWindows,
    *,
    horizons: Sequence[int] = (1, 16, 64),
    rollout_count: int | None = 4096,
    from_episode_start: bool = False,
    batch_size: int = 1024,
    precision: Precision | str | None = None,
    compile: bool | None = None,
) -> dict[int, RolloutStatistics]:
    """Roll ``nerd`` forward autoregressively on recorded controls and measure state drift at ``horizons``.

    By default, primes history with ``receptive_field - 1`` recorded frames starting anywhere in an episode; pass
    ``from_episode_start=True`` to test fresh resets with an empty history. Recomputable contact inputs are
    re-evaluated on predicted states; non-recomputable inputs reuse recorded contact signals.

    ``nrmse`` at horizon ``H`` is the RMS of ``system.increment(recorded, predicted) / nerd.output_normalizer.std``
    over frames ``1 .. H``, channels, and rollouts (``1.0`` equals one typical training step change).

    Args:
        nerd: Fitted integrating model.
        windows: Recorded windows of any length (re-sliced internally to ``history + max(horizons)``).
        horizons: Non-negative frame horizons to evaluate.
        rollout_count: Evenly spaced start points to evaluate, or ``None`` for all available starts.
        from_episode_start: Start every rollout at frame 0 of an episode with empty history.
        batch_size: Rollouts stepped in parallel.
        precision: Override :class:`Precision` (defaults to ``nerd.precision``).
        compile: Override ``torch.compile`` (defaults to ``True`` on CUDA).

    Returns:
        Mapping from each horizon to its :class:`RolloutStatistics`.
    """
    if not nerd.fitted:
        raise ValueError(
            "the normalization of the model was never fitted, so the model is not trained and its error would "
            "be scaled by statistics that hold no data. Pass the trained model"
        )
    if not nerd.prediction.integrates:
        raise ValueError(
            f"{type(nerd.prediction).__name__} does not give the next state, so the model cannot roll out alone"
        )
    if nerd.receptive_field is None and not from_episode_start:
        raise ValueError(
            "a model with unbounded history cannot be primed from the frames before a rollout. Roll it out from "
            "the starts of episodes, with from_episode_start=True"
        )
    if not horizons or min(horizons) < 0:
        raise ValueError(f"horizons must name at least one frame count of zero or more, got {tuple(horizons)}")
    if rollout_count is not None and rollout_count < 1:
        raise ValueError(f"rollout_count must be 1 or more, or None for every rollout, got {rollout_count}")
    if nerd.frame_dt is None or not same_duration(windows.frame_dt, nerd.frame_dt):
        raise ValueError(f"the model steps frames of {nerd.frame_dt} s, the windows hold {windows.frame_dt} s")
    history = 0 if from_episode_start else nerd.receptive_field - 1
    windows = windows.with_length(history + max(horizons))
    candidates = torch.arange(len(windows), device=windows.device)
    if from_episode_start:
        candidates = candidates[windows.episode_start]
    if len(candidates) == 0:
        beginning = "the start of an episode" if from_episode_start else f"{history} frames of history"
        raise ValueError(
            f"the recording holds no episode of {windows.length + 1} valid frames, which a rollout over "
            f"{max(horizons)} frames after {beginning} needs. Use shorter horizons, or record longer episodes "
            "with episode_frames of the recorder"
        )
    count = len(candidates) if rollout_count is None else min(rollout_count, len(candidates))
    chosen = candidates[torch.arange(count, device=windows.device) * len(candidates) // count]

    batch_size = min(batch_size, count)
    stepper = NeRDStepper(nerd, batch_size, precision=precision, compile=compile)
    errors: dict[int, list[Frames]] = {horizon: [] for horizon in horizons}
    squares: dict[int, list[Tensor]] = {horizon: [] for horizon in horizons}
    for first in range(0, count, batch_size):
        # Pad the final batch by repeating its last index so batch_size stays constant for torch.compile.
        indices = chosen[torch.arange(first, first + batch_size, device=windows.device).clamp_max(count - 1)]
        kept = min(batch_size, count - first)
        frames = windows.batch(indices).to(nerd.device).frames
        for horizon, error, square in _rollout(nerd, stepper, frames, horizons):
            errors[horizon].append({name: value[:kept] for name, value in error.items()})
            squares[horizon].append(square[:kept])
    return {horizon: _statistics(errors[horizon], squares[horizon]) for horizon in horizons}


def _rollout(
    nerd: NeRDBase, stepper: NeRDStepper, frames: Frames, horizons: Sequence[int]
) -> Iterator[tuple[int, Frames, Tensor]]:
    """Prime ``stepper`` on prefix frames, roll out ``max(horizons)`` steps, and yield stats at each horizon."""
    system, contact_input = nerd.system, nerd.contact_input
    states = [name for name, spec in system.signals.items() if spec.role is SignalRole.STATE]
    longest = max(horizons)
    # Transpose to [T, B, ...] contiguous so recorded and predicted frames share identical strides for torch.compile.
    recorded = {name: value.transpose(0, 1).contiguous() for name, value in frames.items()}
    start = len(next(iter(recorded.values()))) - longest - 1
    stepper.reset()
    for frame in range(start):
        stepper.step({name: value[frame] for name, value in recorded.items()})

    state = {name: recorded[name][start] for name in states}
    square = torch.zeros(stepper.world_count, device=stepper.device)
    for offset in range(longest + 1):
        reference = {name: recorded[name][start + offset] for name in states}
        if offset > 0:
            square += (system.increment(reference, state) / nerd.output_normalizer.std).square().mean(dim=-1)
        if offset in horizons:
            yield offset, system.error(reference, state), square / max(offset, 1)
        if offset < longest:
            current = {name: value[start + offset] for name, value in recorded.items()} | state
            if contact_input is not None and contact_input.recomputable:
                # Make contact outputs contiguous so non-contiguous views do not trigger torch.compile recompilation.
                contact = contact_input.evaluate(system, state)
                current |= {name: value.contiguous() for name, value in contact.items()}
            state = stepper.step(current)


def _statistics(errors: list[Frames], squares: list[Tensor]) -> RolloutStatistics:
    """Aggregate batched per-horizon errors and mean-squared normalized increments into :class:`RolloutStatistics`."""
    rows = {}
    for name in errors[0]:
        # Sort manually because torch.quantile fails above 2**24 elements (nearest-rank quantiles).
        ordered = torch.cat([error[name] for error in errors]).flatten().sort().values
        ranks = [round(quantile * (ordered.numel() - 1)) for quantile in _QUANTILES]
        rows[name] = (ordered.mean().item(), *ordered[ranks].tolist())
    mean, p50, p90, p99 = ({name: row[i] for name, row in rows.items()} for i in range(4))
    return RolloutStatistics(mean, p50, p90, p99, torch.cat(squares).mean().sqrt().item())
