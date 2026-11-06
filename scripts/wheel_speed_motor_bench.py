"""No-PPO fixed-wheel tests and extracted-rotor MuJoCo bench."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from copy import deepcopy
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import mujoco
from run_wheel_speed_steady_state import OUT,ROOT,PILOT,freeze,evaluate,verify_protected
from uz05.model import UZ05Model,TerrainSpec
from uz05.spec import env_params_for_stage,MODEL_XML
from uz05.wheel_speed_diagnostics import SpeedLoopProbe


def audit():
    s=UZ05Model(TerrainSpec());m=s.model;p=env_params_for_stage('joint_tracking_v16')
    rows=[]
    for j in range(m.njnt):
        first=int(m.jnt_dofadr[j]);last=int(m.jnt_dofadr[j+1]) if j+1<m.njnt else m.nv
        for d in range(first,last):
            rows.append({'joint':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j),'dof':d,
                'damping':m.dof_damping[d],'frictionloss':m.dof_frictionloss[d],
                'armature':m.dof_armature[d],'joint_stiffness':m.jnt_stiffness[j]})
    pd.DataFrame(rows).to_csv(OUT/'compiled_joint_losses.csv',index=False)
    source=ET.parse(MODEL_XML).getroot()
    model_info={'actual_training_xml':str(MODEL_XML),'xml_wheel_joints':{n:source.find(f".//joint[@name='{n}']").attrib for n in ['left_J_link6_wheel','right_J_link6_wheel']},
        'compiled_wheel_damping':m.dof_damping[s.wheel_dof_adr].tolist(),
        'wheel_frictionloss':m.dof_frictionloss[s.wheel_dof_adr].tolist(),
        'wheel_stiffness':m.jnt_stiffness[s.wheel_joint_ids].tolist(),
        'wheel_armature':m.dof_armature[s.wheel_dof_adr].tolist(),
        'wheel_contact_condim':m.geom_condim[s.wheel_geoms].tolist(),
        'fluid_viscosity':float(m.opt.viscosity),'fluid_density':float(m.opt.density),
        'integrator':int(m.opt.integrator),'timestep':float(m.opt.timestep),
        'actuator_biasprm':m.actuator_biasprm.tolist(),'actuator_gainprm':m.actuator_gainprm.tolist(),
        'actuator_dyntype':m.actuator_dyntype.tolist(),
        'training_damping_scale_range':p.domain_randomization.joint_damping_scale,
        'product_inputs':{'rated_rpm':571.,'no_load_rpm':587.,'no_load_current_a':p.wheel.no_load_current_a,
           'voltage_v':24.,'rated_output_nm':2.46,'rated_current_a':10.,'current_definition':'unverified; DC bus vs torque current not interchangeable'}}
    (OUT/'damping_audit.json').write_text(json.dumps(model_info,indent=2)+'\n')
    powers=[]
    for b in [.02,.01,.005]:
      for w in [0.,20.,36.36,571*2*np.pi/60,587*2*np.pi/60]:
        powers.append({'damping':b,'omega_rad_s':w,'torque_nm':b*w,'power_w':b*w*w,
                       'equivalent_torque_current_a_if_Kt0246':b*w/.246})
    pd.DataFrame(powers).to_csv(OUT/'damping_power.csv',index=False)
    return s


def isolated_plant(source):
    m=source.model;root=ET.Element('mujoco',model='extracted_wheels_no_contact_no_gravity')
    integrator={0:'Euler',1:'RK4',2:'implicit',3:'implicitfast'}[int(m.opt.integrator)]
    ET.SubElement(root,'option',timestep=str(m.opt.timestep),gravity='0 0 0',integrator=integrator)
    world=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
    fmt=lambda x:' '.join(format(float(v),'.17g') for v in np.ravel(x))
    details=[]
    for i in range(2):
        body_id=source.wheel_bodies[i];joint_id=source.wheel_joint_ids[i];dof=source.wheel_dof_adr[i];aid=source.wheel_actuator_ids[i]
        b=ET.SubElement(world,'body',name=f'wheel{i}',pos=f'0 {i*.2} 0')
        ET.SubElement(b,'inertial',pos=fmt(m.body_ipos[body_id]),quat=fmt(m.body_iquat[body_id]),mass=str(m.body_mass[body_id]),diaginertia=fmt(m.body_inertia[body_id]))
        ET.SubElement(b,'joint',name=f'wheel{i}',type='hinge',axis=fmt(m.jnt_axis[joint_id]),damping=str(m.dof_damping[dof]),armature=str(m.dof_armature[dof]),frictionloss=str(m.dof_frictionloss[dof]))
        ET.SubElement(act,'motor',name=f'motor{i}',joint=f'wheel{i}',gear=fmt(m.actuator_gear[aid,0]),ctrllimited='true',ctrlrange=fmt(m.actuator_ctrlrange[aid]),forcelimited='true',forcerange=fmt(m.actuator_forcerange[aid]))
        details.append({'side':i,'mass':float(m.body_mass[body_id]),'ipos':m.body_ipos[body_id].tolist(),
            'iquat':m.body_iquat[body_id].tolist(),'inertia':m.body_inertia[body_id].tolist(),'axis':m.jnt_axis[joint_id].tolist(),
            'armature':float(m.dof_armature[dof]),'gear':float(m.actuator_gear[aid,0])})
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    mass=np.zeros((model.nv,model.nv));mujoco.mj_fullM(model,mass,data.qM)
    (OUT/'isolated_plant.json').write_text(json.dumps({'wheels':details,'mass_matrix':mass.tolist(),
        'omitted':'Floating chassis, legs, closed chains, terrain/contact/friction, gravity; fixed wheel axes. No guarantee of whole-robot stability.',
        'retained':'Compiled wheel mass/inertia/COM/orientation/axis, joint armature, damping/frictionloss, motor gear and motor limits; original timestep/integrator.'},indent=2)+'\n')
    return model


def isolated(source):
    model=isolated_plant(source);p=env_params_for_stage('joint_tracking_v16');rows=[]
    folder=OUT/'isolated';folder.mkdir(exist_ok=True)
    experiments=[('B_rated',b,c) for b in [.02,.01,.005] for c in ['P','PI','P_FF']]+[('A_legacy',.02,'P')]
    cols=['time_s','target_rad_s','omega_0','omega_1','current_0','current_1','torque_0','torque_1','passive_0','passive_1','integral_a_0','integral_a_1','ff_a_0','ff_a_1','aw_block_0','aw_block_1']
    for motor_model,b,scheme in experiments:
      model.dof_damping[:]=b
      for target in [20.,-20.,36.36,-36.36]:
        path=folder/f'{motor_model}_{b}_{scheme}_{target}.npz'
        data=mujoco.MjData(model);esc=SpeedLoopProbe(deepcopy(p.wheel),scheme,[b,b],motor_model)
        samples=[]
        for k in range(10000):
            current=esc.current(np.full(2,target),data.qvel.copy(),.002)
            torque=esc.torque_from_current(current);data.ctrl[:]=esc.motor_command(torque)
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
            np.testing.assert_allclose(data.qfrc_actuator,torque,atol=1e-9)
            samples.append([(k+1)*.002,target,*data.qvel,*current,*data.qfrc_actuator,*data.qfrc_passive,
                            *esc.integral_current,*esc.feedforward_current,*esc.antiwindup_blocked])
        a=np.array(samples);final=a[a[:,0]>18]
        kt=.246*(.7 if motor_model=='A_legacy' else 1.)
        expected=kt*1.25/(kt*1.25+b)*target if scheme=='P' else target
        mean=float(final[:,2:4].mean());error=target-mean
        rows.append({'model':motor_model,'damping':b,'controller':scheme,'target_rad_s':target,
            'steady_omega_rad_s':mean,'steady_error_rad_s':error,'theory_omega_rad_s':expected,
            'theory_difference_rad_s':mean-expected,'steady_motor_torque_nm':float(final[:,6:8].mean()),
            'steady_passive_torque_nm':float(final[:,8:10].mean()),
            'steady_saturation_fraction':float((np.abs(final[:,4:6])>=9.9).mean()),
            'peak_current_a':float(np.abs(a[:,4:6]).max()),'peak_abs_omega_rad_s':float(np.abs(a[:,2:4]).max()),
            'overshoot_rad_s':float(max(0.,(np.sign(target)*a[:,2:4]).max()-abs(target))),
            'steady_omega_std_rad_s':float(final[:,2:4].std()),
            'antiwindup_block_fraction':float(a[:,14:16].mean()),'contact':'none; unloaded extracted wheels'})
        np.savez_compressed(path,samples=a,columns=np.array(cols))
        print('ISOLATED',motor_model,b,scheme,target,'error',round(error,7),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'isolated_metrics.csv',index=False)
    p_cases=df[(df.controller=='P')&(df.model=='B_rated')]
    result={'p_theory_max_error_rad_s':float(p_cases.theory_difference_rad_s.abs().max()),
            'p_theory_verified':bool(p_cases.theory_difference_rad_s.abs().max()<1e-5),
            'PI_FF_max_final_error_rad_s':float(df[df.controller!='P'].steady_error_rad_s.abs().max()),
            'cases':len(df),'no_ppo':True,'no_contact':True}
    (OUT/'isolated_verification.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    freeze();source=audit()
    if not (OUT/'fixed_robot_summary.json').exists():
        rows=[]
        for target in [20.,-20.,36.36,-36.36]:rows+=evaluate('base','stand',PILOT,fixed_target=target)
        (OUT/'fixed_robot_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    rows=json.loads((OUT/'fixed_robot_summary.json').read_text())
    print('FULL ROBOT fixed targets survived',sum(r['success'] for r in rows),'/',len(rows),flush=True)
    # Independent plant needed when no reproducible balanced fixed-target test.
    if not all(r['success'] for r in rows):
        if not (OUT/'isolated_verification.json').exists():isolated(source)
    else:isolated(source)  # also separates contact load from the analytical no-load case
    verify_protected()

if __name__=='__main__':main()
