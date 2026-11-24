.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton_neural.backbones
=======================

Streaming temporal backbones mapping embedded frame sequences to hidden states.

``BackboneFlash`` is the default backbone for :class:`~newton_neural.models.FlashNeRD`. ``BackboneTransformer``
provides a standard multi-head attention baseline, and ``BackboneMLP`` provides a low-cost MLP over a fixed frame
window (use ``window=1`` for single-frame predictions such as free motion). Subclass ``BackboneBase`` to define a
custom backbone and verify it with :func:`newton_neural.testing.check_backbone_streaming`.

.. experimental::

.. py:module:: newton_neural.backbones
.. currentmodule:: newton_neural.backbones

.. rubric:: Classes

.. autosummary::
   :toctree: _generated
   :nosignatures:

   BackboneBase
   BackboneFlash
   BackboneMLP
   BackboneTransformer
