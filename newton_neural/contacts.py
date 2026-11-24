# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""External contact sensing and encoding for learned dynamics models.

``ContactInputProbes`` senses spheres fixed on specified bodies against a flat ground plane without collision
detection. ``ContactInputSet`` encodes arbitrary rigid contacts reported by Newton's collision pipeline into a
fixed-size permutation-invariant code. Omit contact input for free-space robots or two-way MuJoCo Warp coupling.
Subclass ``ContactInputBase`` and ``ContactSensorBase`` to add custom contact representations.

.. experimental::
"""

from ._src.contacts.contact_input import ContactInputBase, ContactSensorBase
from ._src.contacts.contact_probes import ContactInputProbes
from ._src.contacts.contact_set import ContactInputSet

__all__ = ["ContactInputBase", "ContactInputProbes", "ContactInputSet", "ContactSensorBase"]
