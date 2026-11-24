# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulation wrappers, trajectory recording, sharded datasets, and GPU-resident training windows.

Depends on ``core`` and ``systems`` only; models plug into recording solely via lists of :class:`SignalSource`.
"""
