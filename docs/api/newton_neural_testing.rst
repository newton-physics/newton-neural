.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.testing
=====================

Conformance checks for custom backbones, model families, and contact inputs.

``check_backbone_streaming`` and ``check_nerd_streaming`` verify that streaming single-frame execution matches
windowed training across mid-episode resets and NaN-poisoned history buffers. ``check_contact_input_recompute``
verifies that a recomputable contact input's ``evaluate()`` matches its Warp sensor output.

.. py:module:: newton_neural.testing
.. currentmodule:: newton_neural.testing

.. rubric:: Functions

.. autosummary::
   :toctree: _generated

   check_backbone_streaming
   check_contact_input_recompute
   check_nerd_streaming
