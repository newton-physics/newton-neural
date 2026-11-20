.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.systems
=====================

System specifications and Newton array bindings.

``SystemArticulation`` describes an articulated robot (fixed or floating base), maps raw Newton signals to
network features, and integrates predicted increments; ``BindingArticulation`` binds those signals to Newton's Warp
arrays. ``robot_frame`` and ``robot_frame_floating`` compute the same robot coordinate frame inside Warp kernels.
Subclass ``SystemBase`` and ``BindingBase`` for non-articulated systems.

.. experimental::

.. py:module:: newton_neural.systems
.. currentmodule:: newton_neural.systems

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   BindingArticulation
   BindingBase
   SystemArticulation
   SystemBase

.. rubric:: Functions

.. autosummary::
   :toctree: _generated

   robot_frame
   robot_frame_floating
