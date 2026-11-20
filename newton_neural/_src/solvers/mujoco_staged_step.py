# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Staged MuJoCo Warp step hook replacing robot ``qacc_smooth`` before constraint resolution.

Isolates all access to ``SolverMuJoCo`` private attributes (``_mujoco_warp_step``, ``_use_speculative_contacts``)
in one place and verifies at construction that the staged sequence reproduces MuJoCo Warp's native step.
"""

from __future__ import annotations

from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

STAGES = (
    "step1",
    "fwd_actuation",
    "fwd_acceleration",
    "mul_m",
    "solve",
    "rne_postconstraint",
    "sensor_acc",
    "euler",
    "implicit",
)
"""Public ``mujoco_warp`` stage functions composing a staged step."""


@wp.kernel(enable_backward=False)
def _replace_free_acceleration_kernel(
    mjc_dof_to_newton_dof: wp.array2d[wp.int32],
    dof_mask: wp.array[wp.bool],
    free_acceleration: wp.array[wp.float32],
    qacc_smooth: wp.array2d[wp.float32],
):
    world, dof = wp.tid()
    newton_dof = mjc_dof_to_newton_dof[world, dof]
    if newton_dof >= 0 and dof_mask[newton_dof]:
        qacc_smooth[world, dof] = free_acceleration[newton_dof]


def replace_mujoco_warp_step(
    solver: SolverMuJoCo, free_acceleration: wp.array[wp.float32], dof_mask: wp.array[wp.bool]
) -> None:
    """Replace ``solver._mujoco_warp_step`` with public MuJoCo Warp stages injecting ``free_acceleration``.

    Before activating ``dof_mask``, verifies that one unmasked staged step from the model's default state matches
    MuJoCo Warp's native step within ``rtol=1e-4, atol=1e-5``.

    Args:
        solver: MuJoCo Warp solver instance (GPU, no sleeping, no speculative contacts, Euler/implicit/implicitfast).
        free_acceleration: Per-DOF free acceleration buffer ``[joint_dof_count]`` [rad/s^2 or m/s^2].
        dof_mask: Boolean mask ``[joint_dof_count]`` selecting revolute/prismatic DOFs to overwrite.
    """
    import mujoco
    import mujoco_warp

    if solver.use_mujoco_cpu:
        raise ValueError("free accelerations are replaced inside MuJoCo Warp. Set use_mujoco_cpu=False")
    if solver.enable_sleeping:
        raise ValueError("free accelerations cannot be replaced while bodies may sleep. Set enable_sleeping=False")
    if solver._use_speculative_contacts:
        raise ValueError(
            "free accelerations cannot be replaced in a step with speculative contacts. "
            "Set use_speculative_contacts=False"
        )
    integrate = {
        mujoco.mjtIntegrator.mjINT_EULER: "euler",
        mujoco.mjtIntegrator.mjINT_IMPLICIT: "implicit",
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "implicit",
    }.get(solver.mj_model.opt.integrator)
    if integrate is None:
        raise ValueError("free accelerations can be replaced under Euler, implicit or implicitfast integration only")
    try:
        stage = SimpleNamespace(**{name: getattr(mujoco_warp, name) for name in STAGES})
    except AttributeError as error:
        raise RuntimeError(
            f"MuJoCo Warp {mujoco_warp.__version__} has no function {error.name}, which a step with replaced "
            "free accelerations calls"
        ) from error

    def staged_step(dof_mask: wp.array[wp.bool] = dof_mask) -> None:
        m, d = solver.mjw_model, solver.mjw_data
        stage.step1(m, d)
        stage.fwd_actuation(m, d)
        stage.fwd_acceleration(m, d)
        wp.launch(
            _replace_free_acceleration_kernel,
            dim=d.qacc_smooth.shape,
            inputs=[solver.mjc_dof_to_newton_dof, dof_mask, free_acceleration],
            outputs=[d.qacc_smooth],
            device=solver.model.device,
        )
        # Recompute smooth generalized forces qfrc_smooth = M * qacc_smooth before constraint solve.
        stage.mul_m(m, d, d.qfrc_smooth, d.qacc_smooth)
        stage.solve(m, d)
        if m.opt.run_rne_postconstraint or (
            not (m.opt.disableflags & mujoco_warp.DisableBit.SENSOR) and m.sensor_rne_postconstraint
        ):
            stage.rne_postconstraint(m, d)
        stage.sensor_acc(m, d, skip_rne_postconstraint=True)
        getattr(stage, integrate)(m, d)

    expected = _step_from_default(solver)
    solver._mujoco_warp_step = lambda: staged_step(wp.zeros_like(dof_mask))
    found = _step_from_default(solver)
    if not np.allclose(found, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"the stages of MuJoCo Warp {mujoco_warp.__version__} do not reproduce its step. The joint state "
            f"after one step differs by up to {np.abs(found - expected).max():.3g}"
        )
    solver._mujoco_warp_step = staged_step


def _step_from_default(solver: SolverMuJoCo) -> np.ndarray:
    """Take one reference step from default state, reset ``solver``, and return concatenated ``[joint_q, joint_qd]``."""
    model = solver.model
    state, state_out = model.state(), model.state()
    # Supply empty contacts if the solver expects external Newton collision detection.
    contacts = None if solver.mjw_model.opt.run_collision_detection else newton.Contacts(0, 0, device=model.device)
    # Invoke SolverMuJoCo base methods directly since SolverNeRDMuJoCo is still inside __init__.
    SolverMuJoCo.step(solver, state, state_out, model.control(), contacts, solver.mj_model.opt.timestep)
    SolverMuJoCo.reset(solver, state)
    return np.concatenate([state_out.joint_q.numpy(), state_out.joint_qd.numpy()])
