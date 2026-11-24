# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Floating-point tolerant duration comparison shared by models, solvers, and evaluation."""

from __future__ import annotations


def same_duration(duration: float, expected: float) -> bool:
    """Return whether ``duration`` matches ``expected`` within relative tolerance ``1e-6``.

    Args:
        duration: Candidate duration [s].
        expected: Reference duration [s].

    Returns:
        ``True`` if ``abs(duration - expected) <= 1e-6 * expected``.
    """
    return abs(duration - expected) <= 1e-6 * expected
