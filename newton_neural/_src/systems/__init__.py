# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""System specifications and Newton array bindings.

Defines what a model simulates (state/control signal layout, feature extraction, state integration) and how those
signals map onto Newton's flat Warp arrays. Higher layers access robots through :class:`SystemBase` and
:class:`BindingBase` rather than indexing Newton arrays directly. Depends only on ``core``.
"""
