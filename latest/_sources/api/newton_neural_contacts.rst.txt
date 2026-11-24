.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.contacts
======================

External contact sensing and encoding for learned dynamics models.

``ContactInputProbes`` senses spheres fixed on specified bodies against a flat ground plane without collision
detection. ``ContactInputSet`` encodes arbitrary rigid contacts reported by Newton's collision pipeline into a
fixed-size permutation-invariant code. Omit contact input for free-space robots or two-way MuJoCo Warp coupling.
Subclass ``ContactInputBase`` and ``ContactSensorBase`` to add custom contact representations.

.. experimental::

.. py:module:: newton_neural.contacts
.. currentmodule:: newton_neural.contacts

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ContactInputBase
   ContactInputProbes
   ContactInputSet
   ContactSensorBase
