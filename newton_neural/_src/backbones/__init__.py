# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Streaming temporal backbones independent of robots, signals, and Newton.

Every backbone computes the exact same causal mapping over full training windows (``forward``) and one frame at a
time with fixed-shape in-place history buffers (``step``).
"""
