"""Diagnostic-only speed loops and contact kinematics. No production imports."""
import numpy as np
import mujoco
from .m3508_diagnostics import DiagnosticWheelEsc


class SpeedLoopProbe(DiagnosticWheelEsc):
    def __init__(self,params,scheme='P',damping=(.02,.02),model='B_rated',ki=.5,integral_current_limit=4.):
        if scheme not in ('P','PI','P_FF'):raise ValueError(scheme)
        self.scheme=scheme;self.damping=np.asarray(damping,dtype=float)
        self.ki=ki;self.integral_current_limit=integral_current_limit
        super().__init__(params,model)

    def reset(self):
        super().reset()
        self.integral_current=np.zeros(2)
        self.p_current=np.zeros(2);self.feedforward_current=np.zeros(2)
        self.raw_current=np.zeros(2);self.antiwindup_blocked=np.zeros(2,dtype=bool)

    def current(self,target,actual,dt):
        if self.scheme=='P':
            result=super().current(target,actual,dt)
            self.p_current=result.copy();self.raw_current=result.copy()
            return result
        target=np.clip(target,-self.params.velocity_limit,self.params.velocity_limit)
        error=np.clip(target-actual,-self.params.speed_error_limit,self.params.speed_error_limit)
        self.p_current=self.params.speed_kp_a_per_rad_s*error
        if self.scheme=='PI':
            proposal=np.clip(self.integral_current+self.ki*error*dt,
                             -self.integral_current_limit,self.integral_current_limit)
            demand=self.p_current+proposal
            limit=self.params.current_limit_a
            allowed=((demand<=limit)&(demand>=-limit))|((demand>limit)&(error<0))|((demand<-limit)&(error>0))
            self.antiwindup_blocked=~allowed
            self.integral_current=np.where(allowed,proposal,self.integral_current)
        else:
            # Verified simulated viscous loss only; no claims of hardware fit.
            self.feedforward_current=self.damping*target/self.params.torque_per_amp_joint
        self.raw_current=self.p_current+self.integral_current+self.feedforward_current
        return np.clip(self.raw_current,-self.params.current_limit_a,self.params.current_limit_a)


def contact_diagnostics(s):
    """Solved end-of-cycle forces, signed tangential slip and exact kinematics.

    For each contacted wheel, v_base_x = slip_x + normal_x - spin_x
    - carrier_rotation_x - (wheel_center_vx - base_vx). Average using normal
    forces per wheel; do not fill absent contacts with invented slip data.
    """
    normal,tangent,mu,slip=[np.zeros(2) for _ in range(4)]
    parts=np.zeros((2,6));rotation=s.rotation_matrix();bx=rotation[:,0]
    base_vx=float(bx@s.base_lin_vel_world)
    for ci in range(s.data.ncon):
        ct=s.data.contact[ci]
        for side,geom in enumerate(s.wheel_geoms):
            if geom not in (ct.geom1,ct.geom2):continue
            other=ct.geom2 if geom==ct.geom1 else ct.geom1
            if s.model.geom_bodyid[other]!=0:continue
            force=np.zeros(6);mujoco.mj_contactForce(s.model,s.data,ci,force)
            fn=abs(force[0]);normal[side]+=fn;tangent[side]+=np.linalg.norm(force[1:3]);mu[side]=max(mu[side],ct.friction[0])
            jp=np.zeros((3,s.model.nv));jr=np.zeros_like(jp)
            body=int(s.model.geom_bodyid[geom]);mujoco.mj_jac(s.model,s.data,jp,jr,ct.pos,body)
            point_v=jp@s.data.qvel;angular=jr@s.data.qvel
            frame=np.asarray(ct.frame).reshape(3,3);nv=frame[0]*np.dot(frame[0],point_v)
            tv=point_v-nv;slip[side]=max(slip[side],float(np.linalg.norm(tv)))
            cp=np.zeros_like(jp);cr=np.zeros_like(jp)
            center=s.data.site_xpos[s.wheel_sites[side]]
            mujoco.mj_jac(s.model,s.data,cp,cr,center,body)
            center_v=cp@s.data.qvel;relative_spin=s.data.xaxis[s.wheel_joint_ids[side]]*s.data.qvel[s.wheel_dof_adr[side]]
            lever=ct.pos-center
            values=np.array([bx@tv,bx@nv,bx@np.cross(relative_spin,lever),
                             bx@np.cross(angular-relative_spin,lever),bx@center_v-base_vx,base_vx])
            parts[side]+=fn*values
    valid=normal>1e-6
    parts[valid]/=normal[valid,None]
    parts[~valid]=np.nan
    residual=parts[:,5]-(parts[:,0]+parts[:,1]-parts[:,2]-parts[:,3]-parts[:,4])
    result={}
    for i in range(2):
        result.update({f'normal_force_n_{i}':normal[i],f'contact_slip_m_s_{i}':slip[i],
                       f'friction_utilization_{i}':tangent[i]/max(normal[i]*mu[i],1e-12)})
        for k,name in enumerate(('signed_slip_body_vx','normal_body_vx','spin_body_vx','carrier_body_vx','center_minus_base_vx','base_vx_check')):
            result[f'{name}_{i}']=parts[i,k]
        result[f'kinematic_identity_error_{i}']=residual[i]
        result[f'wheel_passive_nm_{i}']=s.data.qfrc_passive[s.wheel_dof_adr[i]]
        result[f'wheel_viscous_predicted_nm_{i}']=-s.model.dof_damping[s.wheel_dof_adr[i]]*s.data.qvel[s.wheel_dof_adr[i]]
    return result
