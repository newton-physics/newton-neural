# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Live simulation stability and divergence audits (:func:`audit_rest`, :func:`audit_paired`)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core.duration import same_duration
from ..core.signal import Frames
from ..data.commands import CommandBase, CommandHeld, start_commands
from ..data.simulation import SimulationBase


@dataclass(frozen=True)
class AuditRest:
    """Per-signal drift and detrended vibration reported by :func:`audit_rest` (in each signal's SI unit)."""

    drift_max: dict[str, float]
    """Max start-to-end absolute state difference across worlds and coordinates."""

    drift_mean: dict[str, float]
    """Mean start-to-end absolute state difference across worlds and coordinates."""

    vibration_max: dict[str, float]
    """Max per-coordinate RMS residual after subtracting the least-squares linear trend over time."""

    vibration_mean: dict[str, float]
    """Mean per-coordinate RMS residual after subtracting the least-squares linear trend over time."""


@dataclass(frozen=True)
class AuditPaired:
    """Per-frame divergence between reference and candidate simulations reported by :func:`audit_paired`."""

    mean: dict[str, Tensor]
    """Mean absolute error ``[frame_count]`` per ``STATE`` signal (index ``k`` = after ``k + 1`` steps)."""

    max: dict[str, Tensor]
    """Max absolute error ``[frame_count]`` per ``STATE`` signal (index ``k`` = after ``k + 1`` steps)."""


def audit_rest(simulation: SimulationBase, *, frame_count: int) -> AuditRest:
    """Step ``simulation`` for ``frame_count`` frames with held controls and measure drift and vibration.

    Drift is ``system.error(state_0, state_T)``; vibration is the per-coordinate RMS residual after removing a
    least-squares linear trend over ``0 .. frame_count`` (separating steady creep from jitter). The trend is fitted
    to ``system.increment(state_0, state_t)``, so a rotation is detrended as a rotation and a quaternion that
    changes sign between frames adds no vibration. A base that turns more than half a turn from its first frame
    wraps and is reported as vibration.

    Args:
        simulation: Simulation initialized to the steady pose under test (mutated in place).
        frame_count: Number of frames to step (``>= 1``).
    """
    if frame_count < 1:
        raise ValueError(f"an audit steps at least one frame, got frame_count {frame_count}")
    system = simulation.system
    steps = [_state(simulation)]
    for _ in range(frame_count):
        simulation.step()
        steps.append(_state(simulation))
    frames = {name: torch.stack([step[name] for step in steps], dim=1) for name in steps[0]}
    start = {name: value[:, :1].expand_as(value) for name, value in frames.items()}

    # Fit the OLS linear trend to increments from the first frame, over centered time [-1, 1] so intercept and
    # slope decouple. An increment has no quaternion components, whose sign may differ from frame to frame.
    increments = system.increment(start, frames)  # [world_count, frame_count + 1, increment_count]
    time = torch.linspace(-1.0, 1.0, len(steps), device=simulation.device).view(-1, 1)
    mean = increments.mean(dim=1, keepdim=True)
    slope = ((increments - mean) * time).sum(dim=1, keepdim=True) / time.square().sum()
    trend = system.integrate(start, mean + time * slope)

    drift = system.error(steps[0], steps[-1])
    vibration = {name: error.square().mean(dim=1).sqrt() for name, error in system.error(trend, frames).items()}
    return AuditRest(
        drift_max={name: error.max().item() for name, error in drift.items()},
        drift_mean={name: error.mean().item() for name, error in drift.items()},
        vibration_max={name: error.max().item() for name, error in vibration.items()},
        vibration_mean={name: error.mean().item() for name, error in vibration.items()},
    )


def audit_paired(
    reference: SimulationBase, candidate: SimulationBase, *, commands: CommandBase | None = None, frame_count: int
) -> AuditPaired:
    """Restart both simulations from ``reference``'s state, step both under shared ``commands``, and track divergence.

    The state is that of the system. What else a scene holds, such as a free object, starts where a restart puts
    it in both simulations, so that neither begins with a scene the other does not have.

    Args:
        reference: Reference simulation (its system starts from its current state).
        candidate: Candidate simulation with identical signals, ``world_count``, ``frame_dt``, and device.
        commands: Command generator (defaults to :class:`CommandHeld` over ``reference.command_bounds()``).
        frame_count: Number of frames to step (``>= 1``).
    """
    if frame_count < 1:
        raise ValueError(f"an audit steps at least one frame, got frame_count {frame_count}")
    system, world_count = reference.system, reference.world_count
    if system.signals != candidate.system.signals:
        raise ValueError("the two simulations hold different signals, so their states cannot be compared")
    if world_count != candidate.world_count:
        raise ValueError(f"the reference steps {world_count} worlds, the candidate {candidate.world_count}")
    if not same_duration(candidate.frame_dt, reference.frame_dt):
        raise ValueError(f"the reference steps frames of {reference.frame_dt} s, the candidate {candidate.frame_dt} s")

    if commands is None:
        commands = CommandHeld()
    next_command = start_commands(commands, reference)
    commands.reset(torch.ones(world_count, dtype=torch.bool, device=reference.device))

    start = _state(reference)
    reference.restart(start)
    candidate.restart(start)
    errors = []
    for _ in range(frame_count):
        command = next_command(_state(reference))
        reference.step(command)
        candidate.step(command)
        errors.append(system.error(_state(reference), _state(candidate)))
    return AuditPaired(
        mean={name: torch.stack([error[name].mean() for error in errors]) for name in errors[0]},
        max={name: torch.stack([error[name].max() for error in errors]) for name in errors[0]},
    )


def _state(simulation: SimulationBase) -> Frames:
    """Return a cloned snapshot of ``simulation``'s current ``STATE`` signals (views are overwritten on step)."""
    return {name: value.clone() for name, value in simulation.binding.read_state(simulation.state).items()}
