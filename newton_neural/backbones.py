# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Streaming temporal backbones mapping embedded frame sequences to hidden states.

``BackboneFlash`` is the default backbone for :class:`~newton_neural.models.FlashNeRD`. ``BackboneTransformer``
provides a standard multi-head attention baseline, and ``BackboneMLP`` provides a low-cost MLP over a fixed frame
window (use ``window=1`` for single-frame predictions such as free motion). Subclass ``BackboneBase`` to define a
custom backbone and verify it with :func:`newton_neural.testing.check_backbone_streaming`.

.. experimental::
"""

from ._src.backbones.backbone import BackboneBase
from ._src.backbones.flash import BackboneFlash
from ._src.backbones.mlp import BackboneMLP
from ._src.backbones.transformer import BackboneTransformer

__all__ = ["BackboneBase", "BackboneFlash", "BackboneMLP", "BackboneTransformer"]
