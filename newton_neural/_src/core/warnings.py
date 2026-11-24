# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Library-specific warning category for filtering ``newton_neural`` warnings independently."""

from __future__ import annotations


class NewtonNeuralWarning(UserWarning):
    """Warning for actionable runtime issues, such as Newton model edits ignored by trained weights."""
