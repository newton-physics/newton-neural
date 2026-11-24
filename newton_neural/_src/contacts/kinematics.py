# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp helpers encoding Newton's body-point velocity and rigid-contact gap conventions.

Used by both contact sensors and ``SolverNeRD``'s contact force estimator so both agree on surface separation.
"""

from __future__ import annotations

import newton
import warp as wp


def require_unreduced(contacts: newton.Contacts) -> None:
    """Raise ``ValueError`` if ``contacts`` came from a collision pipeline with ``body_pair_reduction`` enabled."""
    if contacts.rigid_contact_body_pair_reduced:
        raise ValueError(
            "the contacts come from a collision pipeline with body pair reduction, which keeps a few "
            "representatives of every contact patch and stores another separation for them than the gap between "
            "the two surfaces. Build the pipeline without body_pair_reduction"
        )


@wp.func
def point_velocity(
    body: int,
    point: wp.vec3,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
) -> wp.vec3:
    """Compute world velocity [m/s] of a point attached to ``body`` at world position ``point`` (zero if ``body < 0``).

    Args:
        body: Body index, or ``-1`` for static world geometry.
        point: World-frame query position [m].
        body_q: Body world transforms (``state.body_q``).
        body_qd: Body spatial velocities at center of mass (``state.body_qd``).
        body_com: Local body centers of mass [m] (``model.body_com``).
    """
    if body < 0:
        return wp.vec3()
    # Newton stores the linear velocity at the body CoM, so the point is taken relative to it.
    com = wp.transform_point(body_q[body], body_com[body])
    return newton.math.velocity_at_point(body_qd[body], point - com)


@wp.func
def contact_gap(normal: wp.vec3, support0: wp.vec3, support1: wp.vec3, margin0: float, margin1: float) -> float:
    """Signed separation [m] between margin-expanded shape surfaces along ``normal`` (negative = penetration).

    Args:
        normal: Unit contact normal pointing from shape 0 into shape 1.
        support0: World-frame support point on shape 0 [m].
        support1: World-frame support point on shape 1 [m].
        margin0: Contact margin of shape 0 [m].
        margin1: Contact margin of shape 1 [m].
    """
    return wp.dot(normal, support1 - support0) - margin0 - margin1
