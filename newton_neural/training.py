# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Training objectives and optimizers for PyTorch training loops.

``LossOneStep`` computes normalized one-step prediction error over trajectory windows with optional state-noise
injection and ``torch.compile`` support. ``create_optimizer`` builds either hybrid Muon + AdamW (``OptimizerMuon``)
or plain AdamW. Subclass ``LossBase`` to implement custom training objectives.
"""

from ._src.training.loss import ErrorKind, LossBase, LossOneStep
from ._src.training.optim import OptimizerMuon, create_optimizer

__all__ = ["ErrorKind", "LossBase", "LossOneStep", "OptimizerMuon", "create_optimizer"]
