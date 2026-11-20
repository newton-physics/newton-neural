# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Open-loop rollout metrics and live simulation audits.

``rollout_error`` evaluates multi-step autoregressive drift on recorded trajectories. ``audit_rest`` measures drift
and detrended vibration when controls are held constant, and ``audit_paired`` compares live divergence between a
reference simulation and a neural candidate under shared commands.
"""

from ._src.evaluation.audit import AuditPaired, AuditRest, audit_paired, audit_rest
from ._src.evaluation.rollout import RolloutStatistics, rollout_error

__all__ = ["AuditPaired", "AuditRest", "RolloutStatistics", "audit_paired", "audit_rest", "rollout_error"]
