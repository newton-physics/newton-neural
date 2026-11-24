# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conformance checks for custom backbones, model families, and contact inputs.

``check_backbone_streaming`` and ``check_nerd_streaming`` verify that streaming single-frame execution matches
windowed training across mid-episode resets and NaN-poisoned history buffers. ``check_contact_input_recompute``
verifies that a recomputable contact input's ``evaluate()`` matches its Warp sensor output.

.. experimental::
"""

from ._src.testing.conformance import check_backbone_streaming, check_contact_input_recompute, check_nerd_streaming

__all__ = ["check_backbone_streaming", "check_contact_input_recompute", "check_nerd_streaming"]
