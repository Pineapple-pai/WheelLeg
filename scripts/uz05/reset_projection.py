"""Position/velocity projection of UZ05's passive hinges at reset only.

No integration, actuator command, active-coordinate correction, or model edits.
The local solution is continued from the validated XML assembly branch.
"""
from __future__ import annotations

import mujoco
import numpy as np


CLOSURES = tuple(
    (f'{side}_close_{link}', f'{side}_site_{a}', f'{side}_site_{b}')
    for side in ('left', 'right')
    for link, a, b in (
        ('link3_link4', 'link3_to_link4', 'link4_to_link3'),
        ('link6_link2', 'link6_to_link2', 'link2_to_link6'),
    )
)
PASSIVE_JOINTS = tuple(
    f'{side}_J_{link}' for side in ('left', 'right')
    for link in ('link2_link5', 'link5_link3', 'link5_link1', 'link1_link6')
)


class ProjectionError(RuntimeError):
    pass


class ClosedChainReset:
    position_tolerance_m = 1e-8
    velocity_tolerance_m_s = 1e-8
    max_iterations = 20
    # Reset samples are within 0.05 rad of the XML reference. Keep the solver
    # local instead of accepting other periodic/folded assemblies.
    passive_branch_limit_rad = 0.5

    def __init__(self, model):
        self.model = model

        def named(kind, name):
            i = mujoco.mj_name2id(model, kind, name)
            if i < 0:
                raise ValueError(f'missing reset projection object: {name}')
            return i

        self.sites = [(named(mujoco.mjtObj.mjOBJ_SITE, a),
                       named(mujoco.mjtObj.mjOBJ_SITE, b)) for _, a, b in CLOSURES]
        for name, _, _ in CLOSURES:
            eid = named(mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if model.eq_type[eid] != mujoco.mjtEq.mjEQ_CONNECT:
                raise ValueError(f'{name} is not a connect constraint')
        joints = [named(mujoco.mjtObj.mjOBJ_JOINT, n) for n in PASSIVE_JOINTS]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError('reset projection expects passive scalar hinges')
        self.qpos = np.array([model.jnt_qposadr[j] for j in joints], dtype=int)
        self.dofs = np.array([model.jnt_dofadr[j] for j in joints], dtype=int)
        self.reference = model.qpos0[self.qpos].copy()
        self._ja = np.zeros((3, model.nv))
        self._jb = np.zeros((3, model.nv))
        self.chassis = named(mujoco.mjtObj.mjOBJ_BODY, 'chassis')
        reference_data = mujoco.MjData(model)
        mujoco.mj_forward(model, reference_data)
        if self.errors(reference_data).max() > self.position_tolerance_m:
            raise ValueError('XML reference assembly does not close')
        self.reference_orientation = np.sign(self._branch_determinants(reference_data))
        if np.any(self.reference_orientation == 0):
            raise ValueError('singular XML reference assembly')

    def residual(self, data):
        return np.array([data.site_xpos[a] - data.site_xpos[b] for a, b in self.sites])

    def errors(self, data):
        return np.linalg.norm(self.residual(data), axis=1)

    def jacobian(self, data):
        rows = []
        for a, b in self.sites:
            mujoco.mj_jacSite(self.model, data, self._ja, None, a)
            mujoco.mj_jacSite(self.model, data, self._jb, None, b)
            rows.append((self._ja - self._jb).copy())
        return np.vstack(rows)

    def _branch_determinants(self, data):
        # In each leg's XZ plane, four passive DOFs close four scalar
        # constraints. A branch change passes a singularity/sign change.
        rotation = data.xmat[self.chassis].reshape(3, 3)
        jac = self.jacobian(data).reshape(4, 3, self.model.nv)
        local = np.einsum('ab,ibj->iaj', rotation.T, jac)
        return np.array([np.linalg.det(local[2*s:2*s+2][:, [0, 2]][:,
                         :, self.dofs[4*s:4*s+4]].reshape(4, 4)) for s in range(2)])

    def project(self, data):
        """Keep all non-passive qpos/qvel exactly; raise on an invalid assembly."""
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ProjectionError('nonfinite reset position or velocity')
        data.qpos[self.qpos] = self.reference
        mujoco.mj_forward(self.model, data)
        iterations = 0
        for iterations in range(self.max_iterations + 1):
            error = self.residual(data).ravel()
            if np.max(self.errors(data)) <= self.position_tolerance_m:
                break
            if iterations == self.max_iterations:
                raise ProjectionError('position projection did not converge')
            jac = self.jacobian(data)[:, self.dofs]
            # SVD least squares avoids squaring the Jacobian condition number.
            step, _, rank, _ = np.linalg.lstsq(jac, -error, rcond=1e-10)
            if rank != len(self.dofs):
                raise ProjectionError('singular passive assembly Jacobian')
            step *= min(1., .15 / max(float(np.max(np.abs(step))), 1e-15))
            old = data.qpos[self.qpos].copy()
            old_norm = float(error @ error)
            for k in range(10):
                data.qpos[self.qpos] = old + step * (0.5 ** k)
                mujoco.mj_forward(self.model, data)
                r = self.residual(data).ravel()
                if float(r @ r) < old_norm:
                    break
            else:
                raise ProjectionError('position projection line search failed')
            if np.max(np.abs(data.qpos[self.qpos] - self.reference)) > self.passive_branch_limit_rad:
                raise ProjectionError('passive joints left the local XML assembly branch')

        if np.any(np.sign(self._branch_determinants(data)) != self.reference_orientation):
            raise ProjectionError('assembly branch orientation differs from XML')
        jac = self.jacobian(data)
        # Correct passive velocities only, preserving sampled base/hip/wheel speeds.
        correction, _, rank, _ = np.linalg.lstsq(
            jac[:, self.dofs], -(jac @ data.qvel), rcond=1e-10)
        if rank != len(self.dofs):
            raise ProjectionError('singular velocity projection Jacobian')
        data.qvel[self.dofs] += correction
        mujoco.mj_forward(self.model, data)
        speed_error = float(np.max(np.linalg.norm((jac @ data.qvel).reshape(-1, 3), axis=1)))
        if not np.isfinite(speed_error) or speed_error > self.velocity_tolerance_m_s:
            raise ProjectionError(f'velocity constraint error {speed_error:g} m/s')
        return {'iterations': iterations, 'max_error_m': float(self.errors(data).max()),
                'max_velocity_error_m_s': speed_error}
